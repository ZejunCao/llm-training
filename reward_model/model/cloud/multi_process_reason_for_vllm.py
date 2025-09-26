import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm


'''
多进程调用vllm大模型推理
'''
def call_model(messages, api_base, model_name, max_tokens=3000):
    """单个模型调用，指定API基础URL"""
    openai_api_key = "EMPTY"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )
        response_text = chat_response.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"API请求失败: {e}")  # 这里打印错误信息
        return ''
    
# 将process_item函数移到外部，使其可以被序列化
def process_item(args):
    """可序列化的处理函数"""
    messages, api_base, model_name, max_tokens = args
    return call_model(messages, api_base, model_name, max_tokens)

def process_node_batch(batch_info):
    """处理分配给特定节点的一批数据"""
    batch_data, api_base, model_name, max_tokens, batch_indices, max_workers = batch_info
    
    # 准备参数列表，包含消息和API基础URL
    args_list = [(messages, api_base, model_name, max_tokens) for messages in batch_data]
    
    batch_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch_results = list(executor.map(process_item, args_list))
    
    # 返回结果和它们的原始索引
    return list(zip(batch_indices, batch_results))

def distributed_inference_parallel(data_list, api_bases, model_name, max_tokens=3000, batch_size=1000, workers_per_node=128, save_file_name='tmp.json'):
    """
    并行跨多个API端点分布推理
    
    参数:
        data_list: 要处理的数据完整列表
        api_bases: 不同节点的API基础URL列表
        batch_size: 每批的样本数
        workers_per_node: 每个节点的工作进程数
        
    返回:
        与输入数据相同顺序的结果列表
    """
    total_data = len(data_list)
    num_nodes = len(api_bases)
    results = [None] * total_data  # 预分配结果列表
    
    # 计算批次数量
    num_batches = (total_data + batch_size - 1) // batch_size
    
    print(f"并行处理: 跨{num_nodes}个节点以{num_batches}批处理{total_data}项数据")
    
    # 准备每个节点的批次
    node_batches = []
    
    for batch_idx in range(num_batches):
        # 确定用于此批次的节点（轮询方式）
        node_idx = batch_idx % num_nodes
        api_base = api_bases[node_idx]
        
        # 计算批次开始和结束索引
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_data)
        
        # 获取数据批次
        batch_data = data_list[start_idx:end_idx]
        batch_indices = list(range(start_idx, end_idx))
        
        # 将批次信息添加到节点批次列表
        node_batches.append((batch_data, api_base, model_name, max_tokens, batch_indices, workers_per_node))
    
    # 使用线程并行处理不同节点的批次
    with ThreadPoolExecutor(max_workers=num_nodes) as executor:
        # 提交所有批次并获取future对象
        futures = [executor.submit(process_node_batch, batch_info) for batch_info in node_batches]
        
        # 显示总体进度
        for future in tqdm(futures, desc="批次完成进度"):
            # 获取结果并更新结果列表
            batch_results = future.result()
            for idx, result in batch_results:
                results[idx] = result
                
            # 每次写完都保存，防止中断丢失数据
            with open(save_file_name, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
    return results

# -------注意修改以下参数-------
save_file_name = 'tmp/pure_critique.json'  # 保存文件名
batch_size = 200  # 每批的样本数
workers_per_node = 32   # 多线程调用大模型的线程数
model_name = "qwen2"  # 大模型名称
api_bases = [  # API节点
    "http://0.0.0.0:5001/v1",
    "http://0.0.0.0:5002/v1",
    # "http://0.0.0.0:5003/v1",
    # "http://0.0.0.0:5004/v1"
]
if __name__ == '__main__':
    # 自定义函数，将数据转换为模型输入格式
    # with open('sky_original_dataset.json', 'r') as f:
    #     sky_dataset = json.load(f)
    # print(f'sky_original_dataset 数量：{len(sky_dataset)}')
    # def get_datalist(data):
    #     messages = [{"role": "user", "content": data['input']}]
    #     return messages
    # data_list = [get_datalist(sky_dataset[i]) for i in range(len(sky_dataset))]
    with open('data_list.json', 'r') as f:
        data_list = json.load(f)
    print(f'data_list 数量：{len(data_list)}')
    # ----------------------------

    # call_model(data_list[0], api_bases[0]) 单条测试
    max_tokens = 3000
    # 运行分布式推理
    all_results = distributed_inference_parallel(
        data_list=data_list,
        api_bases=api_bases,
        model_name=model_name,
        max_tokens=max_tokens,
        batch_size=batch_size,
        workers_per_node=workers_per_node,
        save_file_name=save_file_name,
    )

'''
vllm 部署多个模型实例，可多机部署，有时模型无法8卡部署也需拆分成两个
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5001 --served-model-name qwen2 --model Qwen/Qwen2-72B-Instruct --tensor_parallel_size 4 --gpu-memory-utilization 0.9
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 5002 --served-model-name qwen2 --model Qwen/Qwen2-72B-Instruct --tensor_parallel_size 4 --gpu-memory-utilization 0.9
'''

''' 
------------------------ 调用示例 ------------------------ 
import json
from single_function.multi_process_reason_for_vllm import distributed_inference_parallel

save_file_name = 'tmp.json'  # 保存文件名
batch_size = 50  # 每批的样本数
workers_per_node = 32   # 多线程调用大模型的线程数
model_name = "qwen2"  # 大模型名称
api_bases = [  # API节点
    "http://0.0.0.0:5001/v1",
    "http://0.0.0.0:5002/v1",
    # "http://0.0.0.0:5003/v1",
    # "http://0.0.0.0:5004/v1"
]
with open('data_list.json', 'r') as f:
    data_list = json.load(f)[:100]
print(f'data_list 数量：{len(data_list)}')

all_results = distributed_inference_parallel(
    data_list=data_list,
    api_bases=api_bases,
    model_name=model_name,
    batch_size=batch_size,
    workers_per_node=workers_per_node,
    save_file_name=save_file_name,
)
------------------------------  ------------------------------- 
'''