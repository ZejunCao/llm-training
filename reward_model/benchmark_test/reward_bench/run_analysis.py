"""
Reward-Bench 分析与可视化脚本
--------------------------------
功能：
1. 加载 Reward-Bench 格式的 JSON 数据
2. 统计奖励模型的分数与判别表现
3. 输出多种可视化图表（保存在 figures/ 文件夹下）

作者：加菲大杂烩
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

# 设置绘图风格
sns.set_theme(style="whitegrid", font="DejaVu Sans", palette="Set2")

# ========== 数据加载器 ==========
def load_rewardbench_json(json_path: str) -> pd.DataFrame:
    """
    加载 Reward-Bench JSON 数据，整理成 DataFrame。

    输入格式示例：
    {
        "30-chosen": {"critique": "...", "reward_score": 17.25},
        "30-rejected": {"critique": "...", "reward_score": -23.5}
    }

    输出 DataFrame 列：
    - id: 样本 ID
    - chosen_score / rejected_score
    - chosen_critique / rejected_critique
    - score_diff = chosen - rejected
    - correct = 1 表示 RM 判断正确，否则 0
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    ids = sorted(set(k.split("-")[0] for k in data.keys()))

    for id_ in ids:
        chosen = data[f"{id_}-chosen"]
        rejected = data[f"{id_}-rejected"]

        chosen_score = chosen["reward_score"]
        rejected_score = rejected["reward_score"]

        records.append({
            "id": id_,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "chosen_critique": chosen["critique"],
            "rejected_critique": rejected["critique"],
            "score_diff": chosen_score - rejected_score,
            "correct": 1 if chosen_score > rejected_score else 0
        })

    return pd.DataFrame(records)


# ========== 可视化函数 ==========

def plot_score_distribution(df: pd.DataFrame, save_dir="figures"):
    """
    绘制奖励分数分布（chosen vs rejected）
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df["chosen_score"], label="Chosen", fill=True)
    sns.kdeplot(df["rejected_score"], label="Rejected", fill=True)
    plt.title("Reward Score Distribution", fontsize=14, weight="bold")
    plt.xlabel("Reward Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/score_distribution.png", dpi=300)
    plt.close()


def plot_score_diff_distribution(df: pd.DataFrame, save_dir="figures"):
    """
    绘制分数差（chosen - rejected）的直方图
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(df["score_diff"], bins=30, kde=True, color="skyblue")
    plt.axvline(0, color="red", linestyle="--", label="Zero Difference")
    plt.title("Score Difference Distribution", fontsize=14, weight="bold")
    plt.xlabel("Score Difference (Chosen - Rejected)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/score_diff_distribution.png", dpi=300)
    plt.close()


def plot_accuracy_vs_score_diff(df: pd.DataFrame, save_dir="figures"):
    """
    绘制 准确率 vs 分数差 的曲线
    思路：
    - 把样本按 score_diff 绝对值分桶
    - 计算每个桶内的正确率
    - 通常可以说明分差越大的样本准确率越高
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["abs_diff"] = df["score_diff"].abs()
    df["bucket"] = pd.qcut(df["abs_diff"], q=10, duplicates="drop")

    acc_by_bucket = df.groupby("bucket", observed=True)["correct"].mean().reset_index()

    # 将 Interval -> 数值中点，便于 lineplot
    acc_by_bucket["bucket_mid"] = acc_by_bucket["bucket"].apply(
        lambda iv: (iv.left + iv.right) / 2
    )
    acc_by_bucket = acc_by_bucket.sort_values("bucket_mid")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=acc_by_bucket, x="bucket_mid", y="correct", marker="o")
    # 友好的刻度标签
    plt.xticks(
        acc_by_bucket["bucket_mid"],
        [f"{iv.left:.2f}-{iv.right:.2f}" for iv in acc_by_bucket["bucket"]],
        rotation=45
    )
    plt.title("Accuracy vs. Score Difference", fontsize=14, weight="bold")
    plt.xlabel("Score Difference Bucket (abs)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_vs_score_diff.png", dpi=300)
    plt.close()


def plot_critique_length_distribution(df: pd.DataFrame, save_dir="figures"):
    """
    批判（critique）长度分布
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    df["chosen_len"] = df["chosen_critique"].str.len()
    df["rejected_len"] = df["rejected_critique"].str.len()

    plt.figure(figsize=(8, 5))
    sns.kdeplot(df["chosen_len"], label="Chosen Critique", fill=True)
    sns.kdeplot(df["rejected_len"], label="Rejected Critique", fill=True)
    plt.title("Critique Length Distribution", fontsize=14, weight="bold")
    plt.xlabel("Critique Length (characters)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/critique_length_distribution.png", dpi=300)
    plt.close()


def save_error_cases(df: pd.DataFrame, id2data: list, save_dir="figures", top_n=20):
    """
    保存最严重的错误样本，包含完整的 chosen / rejected 信息
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 过滤错误样本
    errors = df[df["correct"] == 0].copy()
    errors["abs_diff"] = errors["score_diff"].abs()
    errors = errors.sort_values(by="abs_diff", ascending=False).head(top_n)
    
    out_path = Path(save_dir) / "error_cases.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in errors.iterrows():
            f.write(f"=== Sample ID: {row['id']} ===\n")
            f.write(f"Chosen Score   : {row['chosen_score']:.2f}\n")
            f.write(f"Rejected Score : {row['rejected_score']:.2f}\n")
            f.write(f"Score Diff     : {row['score_diff']:.2f}\n\n")
            f.write(f"[Chosen Response]\n{id2data[row['id']]['chosen']}\n\n")
            f.write(f"[Chosen Critique]\n{row['chosen_critique']}\n\n")
            f.write(f"[Rejected Response]\n{id2data[row['id']]['rejected']}\n\n")
            f.write(f"[Rejected Critique]\n{row['rejected_critique']}\n")
            f.write("="*60 + "\n\n")
    
    print(f"Saved error cases to {out_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reward-Bench 分析工具")
    parser.add_argument("--benchmark", type=str, default="reward-bench", help="Reward-Bench 数据集名称", choices=["reward-bench", "reward-bench-2"])
    parser.add_argument("--data", type=str, default="data/reward_results.json", help="Reward-Bench JSON 文件路径")
    parser.add_argument("--out", type=str, default="figures", help="输出目录")
    args = parser.parse_args()

    df = load_rewardbench_json(args.data)
    datasets = load_dataset(f'data/{args.benchmark}')['filtered']
    id2data = {str(d['id']): {'chosen': d['chosen'], 'rejected': d['rejected']} for d in datasets}

    print("[INFO] 总样本数：", len(df))
    print("[INFO] 总准确率：", df["correct"].mean())

    plot_score_distribution(df, save_dir=args.out)
    plot_score_diff_distribution(df, save_dir=args.out)
    plot_accuracy_vs_score_diff(df, save_dir=args.out)
    plot_critique_length_distribution(df, save_dir=args.out)
    save_error_cases(df, id2data, save_dir=args.out, top_n=20)

    print(f"[INFO] 所有图表与分析结果已保存至 {args.out}/")
