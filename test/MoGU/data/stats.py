import json
from collections import Counter

def stats_routes(json_path):
    # 读取 json 文件（假设是一个列表，每个元素是字典）
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统计 route 出现次数
    routes = [item.get("route", "") for item in data]
    counter = Counter(routes)

    total = sum(counter.values())
    print(f"总样本数: {total}\n")

    for route, count in counter.items():
        ratio = count / total * 100 if total > 0 else 0
        print(f"{route}: {count} ({ratio:.2f}%)")

if __name__ == "__main__":
    stats_routes("answers.json")  # ⚡ 修改为你的文件路径
