import json

# 输入文件路径
input_file = "align_data/safety_align_v6.jsonl"
# 输出文件路径
output_file = "MoGU/data/unsafety_reject.json"

# 读取数据
filtered = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳过空行
            item = json.loads(line)
            if item.get("label") == 1:
                filtered.append({
                    "question": item["question"],
                    "answer": item["answer"]
                })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(filtered)} 条数据到 {output_file}")
