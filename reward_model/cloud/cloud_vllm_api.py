import logging
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 关闭httpx的日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 或者更严格地关闭所有HTTP相关日志
logging.getLogger("httpx").disabled = True
logging.getLogger("openai").disabled = True
logging.getLogger("httpcore").disabled = True

app = FastAPI(
    title="Cloud API Server",
    description="基于vLLM的生成和奖励模型API服务",
    version="1.0.0"
)

PROCESS_NUM = 128  # 多进程调用接口，模型越小或机器越多可设置越大
# vLLM服务配置
REWARD_API_ENDPOINTS = [
    "http://0.0.0.0:5001/pooling",  # 奖励头服务
    # "http://0.0.0.0:5001/pooling",
]

CHAT_API_ENDPOINTS = [
    "http://0.0.0.0:5002/v1",  # 生成模型服务
    # "http://0.0.0.0:5002/v1",
]

# Pydantic模型
class RewardInput(BaseModel):
    user_prompt: str
    response: str

class RewardOutput(BaseModel):
    reward_score: float
    critique: str

class BatchRewardInput(BaseModel):
    data: List[RewardInput]

class BatchRewardOutput(BaseModel):
    results: List[RewardOutput]

# 调用生成模型获取批判
def get_cloud_critique(messages: List[Dict[str, str]]) -> str:
    """调用生成模型获取评论"""
    try:
        openai_api_key = "EMPTY"
        openai_api_base = random.choice(CHAT_API_ENDPOINTS)
        
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        
        chat_response = client.chat.completions.create(
            model="qwen2",
            messages=messages,
        )
        
        return chat_response.choices[0].message.content
    except Exception as e:
        logger.error(f"生成评论失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成评论失败: {str(e)}")

# 调用奖励头获取分数
def get_cloud_score(messages: List[Dict[str, str]]) -> float:
    """调用奖励头获取分数"""
    try:
        api_url = random.choice(REWARD_API_ENDPOINTS)
        sys_msg = messages[0]['content']
        qa_content = messages[1]['content']
        critique = messages[2]['content']
        prompt = {
            "model": "r1-reward",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                {"role": "user", "content": [{"type": "text", "text": qa_content}]},
                {"role": "assistant", "content": [{"type": "text", "text": critique}]},
            ],
        }
        headers = {
            "Content-Type": "application/json", 
            "Authorization": "Bearer EMPTY"
        }
        response = requests.post(api_url, headers=headers, json=prompt, timeout=30)
        score = response.json()["data"][0]["data"][0]
        return float(score)
    except Exception as e:
        logger.error(f"获取奖励分数失败: {e}")
        return 0.0  # 注意，如果分数全是0说明模型出现问题

@app.post("/api/reward")
def get_reward(input_data: RewardInput):
    """获取单个样本的奖励分数和评论"""
    try:
        # 构建评论提示
        COT_PROMPT_ENGLISH = "The following is a break down on the helpfulness and safety of the assistant's response to my question: "
        input_text = f"<question>{input_data.user_prompt}</question>\n<response>{input_data.response}</response>\n{COT_PROMPT_ENGLISH}"
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': input_text},
        ]
        # 获取评论
        critique = get_cloud_critique(messages)
        # 添加评论到messages列表
        messages.append({'role': 'assistant', 'content': critique})
        # 获取奖励分数
        score = get_cloud_score(messages)
        return RewardOutput(reward_score=score, critique=critique)
    except Exception as e:
        logger.error(f"获取奖励失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取奖励失败: {str(e)}")

@app.post("/api/reward/batch", response_model=BatchRewardOutput)
async def get_batch_reward(data: BatchRewardInput):
    """批量获取奖励分数和评论"""
    try:
        with ProcessPoolExecutor(max_workers=PROCESS_NUM) as executor:
            results = list(tqdm(executor.map(get_reward, data.data), total=len(data.data)))
        return BatchRewardOutput(results=results)
        
    except Exception as e:
        logger.error(f"批量获取奖励失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量获取奖励失败: {str(e)}")


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5008,
        log_level="info",
        access_log=True
    )