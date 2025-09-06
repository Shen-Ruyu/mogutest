import json

# 输入文件和输出文件路径
input_file = "unsafety_reject.json"   # 你的原始文件
output_file = "unsafety_reject_fixed.json"  # 修改后的文件

# 读取原始 JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 遍历数据并修改键名
new_data = []
for item in data:
    new_item = {}
    for k, v in item.items():
        if k == "question":
            new_item["instruction"] = v
        elif k == "answer":
            new_item["output"] = v
        else:
            new_item[k] = v
    new_data.append(new_item)

# 保存修改后的 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"转换完成 ✅ 已保存到 {output_file}")
