import json

# 输入文件（之前提取好的 JSON）
input_file = "safety_affirm.json"
# 输出文件
output_file = "safety_reject.json"

# 读取 JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 替换 output
for record in data:
    record["output"] = "Sorry, I can't answer this question"

# 保存为新 JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"修改完成，新文件保存为 {output_file}")
