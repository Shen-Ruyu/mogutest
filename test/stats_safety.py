import json
from collections import defaultdict

def compute_safety_rate_by_model(filename):
    stats = defaultdict(lambda: {"safe": 0, "harmful": 0, "total": 0})

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            model = data.get("source_model", "unknown")
            label = data.get("label", None)

            if label == 1:
                stats[model]["safe"] += 1
            elif label == 0:
                stats[model]["harmful"] += 1

            stats[model]["total"] += 1

    # 打印结果
    for model, counts in stats.items():
        safe_rate = (counts["safe"] / counts["total"] * 100) if counts["total"] > 0 else 0.0
        print(f"模型: {model}")
        print(f"  总样本数: {counts['total']}")
        print(f"  安全 (label=1): {counts['safe']}")
        print(f"  有害 (label=0): {counts['harmful']}")
        print(f"  安全回答率: {safe_rate:.2f}%")
        print("-" * 40)

    return stats


if __name__ == "__main__":
    filename = "align_data/router_eval.jsonl"  # 你的结果文件
    compute_safety_rate_by_model(filename)
