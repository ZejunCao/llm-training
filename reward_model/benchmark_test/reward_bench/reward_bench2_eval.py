import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import requests
from datasets import Dataset, load_dataset
from tqdm import tqdm


def load_reward_bench():
    data = load_dataset("data/reward-bench-2", split="test")
    eval_data = []
    total_completions = []
    for example in data:
        for i in range(len(example["chosen"])):  
            eval_data.append({
                "id": f"{example['id']}-chosen-{i}-{example['subset']}",
                "prompt": example["prompt"],
                "response": example["chosen"][i]
            })
            total_completions.append(example["chosen"][i])
        for i in range(len(example["rejected"])):  
            eval_data.append({
                "id": f"{example['id']}-rejected-{i}-{example['subset']}",
                "prompt": example["prompt"],
                "response": example["rejected"][i]
            })
        total_completions.append(example['total_completions'])
    return eval_data


def process_single_model(dataset):
    """
    Process a single-model ties evaluation dataset and return
        (dataset_with_results_column, overall_score)
    Each row in the dataset contains a list of "scores", where the first "num_correct" correspond to
        correct answers, and the rest are incorrect. The "id" field is formatted as "sample_type:prompt_id",
        where sample_type is either "ref" for reference prompts with 1 correct answer or "tied" for tied samples
        with multiple correct answers.
    Overall score is essentially 60% accuracy, 40% margin. Accuracy is broken down equally
        across ref and tied accuracy, while margin is broken down into whether the margin between
        correct answers < margin between correct and incorrect answers for tied prompts only (correctness_preferred)
        and whether this margin also holds when the margin between correct and incorrect answers is the min of the
        margin for a tied prompt and its associated reference prompt (correctness_preferred_hard).
    
    处理单模型 ties 评测数据集并返回
        （带有结果列的数据集，整体得分）
    数据集中的每一行包含一个“scores”列表，其中前“num_correct”个分数对应正确答案，其余为错误答案。
    “id”字段格式为“sample_type:prompt_id”，其中 sample_type 可以是“ref”（参考样本，仅有 1 个正确答案）或“tied”（tie 样本，有多个正确答案）。
    整体得分本质上为 60% 准确率 + 40% 边际分数。准确率部分在 ref 和 tied 两类样本中平均分配；边际分数部分则根据以下两点计算：
        1. 对于 tied 样本，仅当正确答案之间的分差小于正确与错误答案之间的分差时（correctness_preferred），记为有效。
        2. correctness_preferred_hard 进一步要求该分差关系在 tied 样本和其对应的 ref 样本的最小分差中也成立。
    """
    grouped_samples: Dict[Tuple[str, int], List[Tuple[bool, float]]] = defaultdict(list)

    for sample in dataset:
        # Split samples into ref and tied
        sample_type, prompt_id_str = sample["id"].split(":")
        prompt_id = int(prompt_id_str)

        # Each score position i is “correct” if i < num_correct
        for i, raw_score in enumerate(sample["scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped_samples[(sample_type, prompt_id)].append((i < sample["num_correct"], score))

    # Calculate per-prompt stats
    ref_stats = {}
    tied_stats = {}

    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = _compute_prompt_stats(samples)
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:  # "tied"
            tied_stats[prompt_id] = stats

    # Calculate global metrics
    # Average accuracy (element 0 of each tuple) over ref and tied samples
    ref_accuracy = np.mean([s[0] for s in ref_stats.values()]) if ref_stats else 0.0
    tied_accuracy = np.mean([s[0] for s in tied_stats.values()]) if tied_stats else 0.0

    # Margins: compute whether margin within correct answers < margin between correct and incorrect answers
    all_prompts = set(ref_stats) & set(tied_stats)

    # correct margin is element 1 in stats tuple, correct-incorrect margin is element 2
    diff_corr_margin = np.array([tied_stats[pid][1] for pid in all_prompts])
    corr_incorrect_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
    corr_incorrect_ref = np.array([ref_stats[pid][2] for pid in all_prompts])

    correctness_preferred = np.mean(corr_incorrect_ties > diff_corr_margin)
    correctness_preferred_hard = np.mean(np.minimum(corr_incorrect_ref, corr_incorrect_ties) > diff_corr_margin)

    # Tie-breaking term, optional, not much effect in practice
    # Normalised gap, then tanh to keep it in (‑1, 1)
    margin_scores = np.tanh(np.minimum(corr_incorrect_ref, corr_incorrect_ties) / diff_corr_margin - 1)
    # if nan (divide by 0), set to 0
    margin_scores = np.nan_to_num(margin_scores, nan=0.0)
    correctness_margin_score = float(np.mean(margin_scores))

    # Compute the overall score
    overall_score = (
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )

    # Package results — there is less of a sense of per-prompt results for the Ties subset,
    # as overall_score is computed across the subset, so set "results" to None for clarity
    if "results" in dataset.column_names:
        dataset = dataset.remove_columns(["results"])
    results_dataset = dataset.add_column("results", [None] * len(dataset))

    return results_dataset, float(overall_score)

def _compute_prompt_stats(samples: List[Tuple[bool, float]]) -> Tuple[bool, float | None, float | None]:
    """
    Given a list of (is_correct, score) tuples for one prompt,
    return:
        accurate ................ True if every correct answer outscores the best wrong one
        different_correct_margin  Spread between best and worst correct answers (None if <2)
        correct_incorrect_margin  Gap between worst correct and best wrong (None if N/A)
    """
    correct_scores = [s for is_corr, s in samples if is_corr]
    incorrect_scores = [s for is_corr, s in samples if not is_corr]
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)

    # Calculate the margins with correct scores, and also the margin between correct and incorrect scores
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0

    return accurate, different_correct_margin, correct_incorrect_margin


def add_scores(example, rewards):
    scores = []
    num_correct = example['num_correct']
    total_completions = example['total_completions']
    for j in range(total_completions):
        if j < num_correct:
            _id = f"{example['id']}-chosen-{j}-{example['subset']}"
            scores.append(rewards[_id])
        else:
            _id = f"{example['id']}-rejected-{j-num_correct}-{example['subset']}"
            scores.append(rewards[_id])
    example['scores'] = scores
    max_val = np.max(scores)
    example['results'] = (1 / np.sum(np.array(scores) == max_val)) if scores[0] == max_val else 0
    return example

def post_process_reward_bench(out_dataset):
    # present_subsets = set(raw_dataset['subset'])
    present_subsets = ['Factuality', 'Precise IF', 'Math', 'Safety', 'Focus', 'Ties']
    results_grouped = {}
    for subset in present_subsets:
        # subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        subset_dataset = Dataset.from_list([out_dataset[i] for i in range(len(out_dataset)) if out_dataset[i]['subset'] == subset])
        if subset.lower() == "ties":
            ties_subset_with_results, overall_score = process_single_model(subset_dataset)
            subset_dataset = ties_subset_with_results

            # Update the results for the ties subset in the original dataset
            ties_indices = [i for i, s in enumerate(out_dataset["subset"]) if s == "Ties"]
            out_dataset_df = out_dataset.to_pandas()
            for i, ties_idx in enumerate(ties_indices):
                out_dataset_df.at[ties_idx, "results"] = ties_subset_with_results["results"][i]
            out_dataset = Dataset.from_pandas(out_dataset_df)

            print(f"{subset}: Overall score {overall_score}")
            results_grouped[subset] = overall_score
        else:
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

    for k, v in results_grouped.items():
        print(f"{k:<12}: {v*100:6.2f}")
    print(f'{"mean score":<12}: {np.mean(list(results_grouped.values()))*100:6.2f}')
    

###########
# Scoring
###########

def generate_rewards_hf(model, tokenizer, eval_data, batch_size):
    rewards = {}

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch = eval_data[i:i+batch_size]
        
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        ids = [item["id"] for item in batch]

        batch_rewards, _ = model.predict_reward(prompts, responses, tokenizer)
        
        for id_, reward in zip(ids, batch_rewards):
            rewards[id_] = reward

    return rewards

def generate_rewards_vllm(client, eval_data, num_workers):
    rewards = {}

    def fetch_reward(example):
        critique, reward = client.get_reward(
            example["prompt"],
            example["response"],
        )
        return critique, reward, example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward
    
    return rewards


def generate_rewards_vllm_ray(eval_data, num_workers):
    rewards = {}
    final_rewards = {}
    def fetch_reward(example):
        response = requests.post(
            "http://10.93.240.70:5018/reward/api",
            json={"user_prompt": example["prompt"], "response": example["response"]},
        ).json()
        return response["critique"], response["reward_score"], example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward
            final_rewards[id_] = {
                "critique": critique,
                "reward_score": reward,
            }
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"tmp/final_rewards_{cur_time}.json", "w") as f:
        json.dump(final_rewards, f, indent=4, ensure_ascii=False)
    return rewards


def generate_rewards_vllm_pooling(eval_data, num_workers):
    rewards = {}

    def fetch_reward(example):
        api_url = "http://10.93.240.70:7009/pooling"
        model_name = "r1-reward"

        # Input like Chat API
        prompt = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": example["prompt"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": example["response"]}]},
            ],
        }
        headers = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
        response = requests.post(api_url, headers=headers, json=prompt)
        score = response.json()["data"][0]["data"][0]
        score = float(score)

        return None, score, example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward
    
    return rewards


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


def generate_rewards_vllm_api_batch(eval_data, num_workers):
    batch_size = 128*8  # 可根据服务端最大进程数调整
    rewards = {}  # 
    final_rewards = {}
    for i in tqdm(range(0, len(eval_data), batch_size), desc="Generating rewards"):
        batch = eval_data[i:i+batch_size]
        data = {
            "data": [{"user_prompt": b["prompt"], "response": b["response"]} for b in batch]
        }
        response = requests.post(
            "http://10.93.240.70:5008/api/reward/batch",
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

    raw_dataset = load_dataset("data/reward-bench-2", split="test")

    if args.benchmark == "reward-bench":
        eval_data = load_reward_bench()
    
    if args.inference_method == "vllm+api_batch":
        rewards = generate_rewards_vllm_api_batch(eval_data, num_workers=args.num_workers)
    elif args.inference_method == "vllm+api":
        rewards = generate_rewards_vllm_api(eval_data, num_workers=args.num_workers)

    # 使用map方法批量添加新字段
    out_dataset = raw_dataset.map(add_scores, fn_kwargs={'rewards': rewards})
    post_process_reward_bench(out_dataset)

'''
python reward_bench2_eval.py --inference-method vllm+api_batch --num-workers 128
'''