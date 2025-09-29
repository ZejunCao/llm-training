
<div align="center">

# 判别式奖励模型训练与推理

</div>

本仓库聚焦“基础判别式奖励模型”的全流程实现：数据准备 → 训练（Deepspeed）→ 调试（JSON 参数）→ 与 SFT 的关键差异 → 推理（Transformers 与 vLLM 的要点）。

### 数据准备与格式（适配多源偏好数据）

当前开源偏好数据格式不统一，虽然trl内部对多种格式做了归一化，但还是无法适配所有情况。推荐提前处理成以下格式，或在 `train.py` 中进行自定义数据处理。

单条数据格式示例：
```json
{
  "chosen": [
    {"role": "user", "content": "用一句话介绍Python"},
    {"role": "assistant", "content": "Python是一种简洁优雅的语言"}
  ],
  "rejected": [
    {"role": "user", "content": "用一句话介绍Python"},
    {"role": "assistant", "content": "Python是爬行动物，与编程无关"}
  ]
}
```

自定义处理代码参考：
```81:101:reward_model/model/base_reward/train.py
    def tokenize_function(examples):
        # 1. 拼接成字符串
        '''
        examples['chosen'] = [
            {"role": "user", "content": '用一句话介绍Python'},  # prompt
            {"role": "assistant", "content": 'Python是一种简洁优雅的语言'}  # chosen
        ]
        '''
        chosen_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples[text_column_name[0]]]
        rejected_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples[text_column_name[1]]]

        # 2. 批量 tokenizer
        chosen_batch = tokenizer(chosen_texts, add_special_tokens=False)
        rejected_batch = tokenizer(rejected_texts, add_special_tokens=False)

        return {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
        }
```

关于 `data_collator` 的说明：奖励模型的padding处理较为简单，可直接使用trl默认的 `RewardDataCollatorWithPadding`。

---

### 训练（Deepspeed）
示例命令（4 卡）：
```bash
sh train.sh
```
- 可按需打开 `--gradient_checkpointing True`（降低显存占用，略降吞吐）
- `--max_length` 为过滤样本长度上限（pair 中取较长者）
- LoRA 训练：
  ```bash
  --use_peft \
  --lora_r 32 \
  --lora_alpha 16
  ```
---

### 快速调试（JSON 参数文件）

为方便调试及逐行阅读源码，提供json文件加载参数方式，修改 `train_args.json` 后，在 vscode 进入`train.py`后直接点击 debug 即可。

### vllm 推理部署

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve save/base_reward_0926/checkpoint-50 --served-model-name base_reward --tensor-parallel-size 2 --port 5001 --override-pooler-config '{"pooling_type": "LAST", "normalize": false, "softmax": false}'
```
按需修改：
- gpu索引：CUDA_VISIBLE_DEVICES=0,1；以及对应的显卡数量：--tensor-parallel-size 2
- 模型路径：/data0/zejun7/open_source/llm-training/reward_model/model/base_reward/save/base_reward_0926
- 模型名称：r1-reward
- 端口：5001

调用示例：可见 `杂记.ipynb`。
```python
"""纯判别式奖励模型测试"""
import requests
def get_score_api(query: str, response: str):
    api_url = "http://0.0.0.0:5001/pooling"
    # Input like Chat API
    prompt = {
        "model": "base_reward",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ],
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    response = requests.post(api_url, headers=headers, json=prompt)
    return float(response.json()["data"][0]["data"][0])

user_prompt = "给我写一个谜语"
response_str = """我就不写"""
get_score_api(user_prompt, response_str)
```