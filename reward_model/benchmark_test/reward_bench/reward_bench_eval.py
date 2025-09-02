import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from datasets import load_dataset
from tqdm import tqdm

REWARD_BENCH_TO_CATEGORY_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

###########
# Build eval data
###########

def load_reward_bench():
    data = load_dataset("data/reward-bench")['filtered']
    eval_data = []
    eval_metadata = []
    for example in data:
        eval_data.append({
            "id": f"{example['id']}-chosen",
            "prompt": example["prompt"],
            "response": example["chosen"]
        })
        eval_data.append({
            "id": f"{example['id']}-rejected",
            "prompt": example["prompt"],
            "response": example["rejected"]
        })
        eval_metadata.append({
            "id": str(example["id"]),
            "subset": example["subset"]
        })
    return eval_data, eval_metadata


###########
# Post-process Scores
###########
def post_process_reward_bench(eval_metadata, rewards):
    per_category_scores = {category: [] for category in REWARD_BENCH_TO_CATEGORY_MAPPING.keys()}
    for example in eval_metadata:
        id_ = example["id"]
        chosen_reward = rewards[id_ + "-chosen"]
        rejected_reward = rewards[id_ + "-rejected"]
        for category, subsets in REWARD_BENCH_TO_CATEGORY_MAPPING.items():
            if example["subset"] in subsets:
                per_category_scores[category].append(int(chosen_reward > rejected_reward))
                break
    per_category_scores = {category: np.mean(scores) * 100 for category, scores in per_category_scores.items()}
    per_category_scores["Average"] = np.mean([score for score in per_category_scores.values()])

    # Print scores in a pretty way
    print("\nReward Bench Scores:")
    print("=" * 40)
    max_category_length = max(len(category) for category in per_category_scores.keys())
    for category, score in per_category_scores.items():
        print(f"{category:<{max_category_length}} : {score:.2f}%")
    print("=" * 40)

    return per_category_scores


def generate_rewards_vllm_api(eval_data, num_workers):
    rewards_results = {}
    rewards = {}
    def fetch_reward(example):
        """测试奖励评分"""
        url = "http://0.0.0.0:5008/api/reward"
        data = {
            "user_prompt": example["prompt"],
            "response": example["response"]
        }
        response = requests.post(url, json=data).json()
        return response["critique"], response["reward_score"], example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward
            rewards_results[id_] = {
                "critique": critique,
                "reward_score": reward,
            }
    with open("data/reward_results.json", "w") as f:
        json.dump(rewards_results, f, indent=4, ensure_ascii=False)
    return rewards


def generate_rewards_vllm_api_batch(eval_data):
    batch_size = 128*8  # 可根据服务端最大进程数调整
    rewards = {}  # 
    final_rewards = {}
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Generating rewards"):
        batch = eval_data[i:i+batch_size]
        data = {
            "data": [{"user_prompt": b["prompt"], "response": b["response"]} for b in batch]
        }
        response = requests.post(
            "http://0.0.0.0:5008/api/reward/batch",
            json=data,
        ).json()
        for j, r in enumerate(response["results"]):
            rewards[batch[j]["id"]] = r["reward_score"]
            final_rewards[batch[j]["id"]] = {
                "critique": r["critique"],
                "reward_score": r["reward_score"],
            }
    with open("data/reward_results.json", "w") as f:
        json.dump(final_rewards, f, indent=4, ensure_ascii=False)
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="reward-bench", choices=["reward-bench", "arena-hard"])

    # Vllm args
    parser.add_argument("--inference-method", type=str, default="vllm+api_batch", choices=["vllm+api_batch", "vllm+api"])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hosted", action="store_true")

    # HF args
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.benchmark == "reward-bench":
        eval_data, eval_metadata = load_reward_bench()
    
    # batch传输速度更快，实测能快5倍左右
    if args.inference_method == "vllm+api_batch":
        rewards = generate_rewards_vllm_api_batch(eval_data)
    elif args.inference_method == "vllm+api":
        rewards = generate_rewards_vllm_api(eval_data, num_workers=args.num_workers)

    post_process_reward_bench(eval_metadata, rewards)

'''
python reward_bench_eval.py --inference-method vllm+api_batch --num-workers 128
'''