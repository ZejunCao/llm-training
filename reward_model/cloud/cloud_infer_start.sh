# start_services.sh
#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
conda dactivate
conda activate zejun7-vllm

# 启动奖励头服务
echo "启动奖励头服务..."
CUDA_VISIBLE_DEVICES=0,1 vllm serve save/cloud_sft2_0827/checkpoint-501 \
    --served-model-name r1-reward \
    --tensor-parallel-size 2 \
    --port 5001 \
    --override-pooler-config '{"pooling_type": "LAST", "normalize": false, "softmax": false}' &

# 等待奖励头服务启动
sleep 10

# 启动生成模型服务
echo "启动生成模型服务..."
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 5002 \
    --served-model-name qwen2 \
    --model save/cloud_sft2_0827/checkpoint-502 \
    --tensor_parallel_size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 &

# 等待生成模型服务启动
sleep 15

# 启动FastAPI服务
echo "启动FastAPI服务..."
python cloud_vllm_api.py &

echo "所有服务已启动！"
echo "FastAPI服务地址: http://localhost:8000"
echo "API文档地址: http://localhost:8000/docs"

# 等待所有后台进程
wait