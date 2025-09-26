import fcntl
import json
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from ray import serve
from ray.serve.config import HTTPOptions
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    Qwen2ForCausalLM,
)

from utils import is_chinese_text

temp_dir = "/data0/zejun7/vscode/zj_cloud/tmp"
ray.init(_temp_dir=temp_dir)  # 指定ray缓存文件夹，写绝对路径，否则运行很长时间后，很可能在一个奇怪的地方把磁盘撑爆
LOG_FILE = "logs/ray.log"  # 建议使用共享存储路径
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RewardInput(BaseModel):
    user_prompt: str        # 问题
    response: str           # 回答
    openai_api_base: list   # 生成批判模型的vllm api列表，支持多个节点

class RewardOutput(BaseModel):
    reward_score: float     # 奖励分数
    critique: str           # 批判


def call_model(messages, openai_api_base):
    """调用生成批判模型的vllm api"""
    openai_api_key = "EMPTY"
    openai_api_base_sample = random.choice(openai_api_base)
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base_sample,
    )
    chat_response = client.chat.completions.create(
        model="qwen2",
        messages=messages,
        max_tokens=1024,
    ).choices[0].message.content
    return chat_response


class RewardHead(nn.Module):
    """奖励头"""
    def __init__(self, cfg: PretrainedConfig, n_labels: int):
        super().__init__()
        self.reward_dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # cfg.hidden_size=3584
        self.reward_out_proj = nn.Linear(cfg.hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any):
        hidden_states = self.reward_dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        output = self.reward_out_proj(hidden_states)
        return output


class CloudforQwen2Model(Qwen2ForCausalLM):
    """cloud模型"""
    def __init__(self, config):
        super().__init__(config)
        self.reward_head = RewardHead(config, n_labels=1)

    @torch.inference_mode()
    def predict(
        self,
        messages: List[Dict[str, str]]=None,
        tokenizer: Optional[AutoTokenizer]=None,
        **kwargs,
    ):
        formatted_prompts = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(formatted_prompts, max_length=4096, truncation=True, add_special_tokens=False, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            output2 = self.forward(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = output2.hidden_states[-1]
        hidden_states_last_token = hidden_states[:, -1, :]
        with torch.no_grad():
            rewards = self.reward_head(hidden_states_last_token).flatten().tolist()
        return rewards


def build_message(user_prompt, response):
    COT_PROMPT_ENGLISH = "The following is a break down on the correctness and usefulness of the assistant's response to my question: "
    COT_PROMPT_CHINESE = "以下是assistant对问题回答的帮助性和安全性的分解："
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "<question>" + user_prompt + "</question>\n<response>" + response + "</response>"},
    ]
    message[-1]["content"] += COT_PROMPT_CHINESE if is_chinese_text(user_prompt+response) else COT_PROMPT_ENGLISH
    return message


# 这里需设置部署方式，num_replicas代表部署几个节点，num_gpus代表每个节点部署几个gpu的显存，1代表一个模型占一张卡，可设置为0.5，0.25等
# 当前配置代表：每个模型部署一张卡，共部署两个模型，ray会自动检测当前共有几张卡然后平均分配（用 CUDA_VISIBLE_DEVICES=6,7 控制）
@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class BatchRewardInferModel:
    def __init__(self, base_model_path):
        config = AutoConfig.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = CloudforQwen2Model.from_pretrained(base_model_path, config)

        state = load_file(os.path.join(base_model_path, "reward_head.safetensors"))
        self.model.reward_head.load_state_dict(state, strict=True)

        self.model = self.model.to(torch.bfloat16)
        self.model = self.model.to("cuda")
        self.model.eval()

        torch.cuda.empty_cache()

    @app.post("/api")
    def api(self, input_data: RewardInput):
        """推理"""
        with torch.inference_mode():
            message = build_message(input_data.user_prompt, input_data.response)
            critique_response = call_model(message, input_data.openai_api_base)
            message.append({"role": "assistant", "content": critique_response})
            res = self.model.predict(message, self.tokenizer)
            output = RewardOutput(reward_score=res[0], critique=critique_response)

            # 记录日志（放在最后保证主流程优先）
            self._log_request(input_data, output)
            return output

    def _log_request(self, input_data: RewardInput, output: RewardOutput):
        """带文件锁的日志记录方法"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data.dict(),
            "output": output.dict()
        }

        try:
            # 使用a+模式打开，自动创建文件
            with open(LOG_FILE, "a+") as f:
                # 获取排他锁（非阻塞模式）
                fcntl.flock(f, fcntl.LOCK_EX)
                # 写入单行JSON
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                # 文件关闭时自动释放锁
        except IOError as e:
            # 这里可以添加更细致的异常处理
            print(f"Logging failed: {str(e)}")

# -------------- 修改参数 --------------
base_model_path = 'model_checkpoint/cloud_0901'  # 模型地址
# -------------------------------------

serve.start(http_options=HTTPOptions(host="0.0.0.0", port=5018))
serve.run(
    BatchRewardInferModel.bind(base_model_path=base_model_path),
    route_prefix="/reward",
)
while True:
    time.sleep(1000)

'''
# 1. 先单独启动vllm
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5003 --served-model-name qwen2 --model model_checkpoint/cloud_0901 --tensor_parallel_size 2 --gpu-memory-utilization 0.9

# 2. 然后启动这个ray脚本
CUDA_VISIBLE_DEVICES=6,7 /data0/anaconda3/envs/zejun7-latest/bin/python /data0/zejun7/vscode/zj_cloud/cloud_ray_api_test.py


# 3. 部署完成后进行测试，将以下代码复制到jupyter中运行

import requests

user_prompt = "给我写一个谜语"
response_str = """这是一个谜语，请猜一猜：
不吃饭的东西，
天天吃东西；
不喝水的东西，
天天喝水。
——打一物
你能猜出是什么吗？

谜底是"水缸"。
水缸本身不吃饭，但人们每天往里面"放"（吃）东西；
水缸本身不喝水，但每天都"装"（喝）水。
这个谜语通过拟人的手法，把水缸描述成一个不吃饭却天天吃东西、不喝水却天天喝水的物体，形成了有趣的矛盾修辞。"""

def func_classifier(user_prompt, response_str):
    response = requests.post(
        "http://10.93.240.70:5018/reward/api",
        json={"user_prompt": user_prompt, "response": response_str, "openai_api_base": ["http://10.93.240.70:5003/v1"]},  # openai_api_base写入vllm部署的ip+port
    )
    return response.json()

res = func_classifier(user_prompt, response_str)
res
'''