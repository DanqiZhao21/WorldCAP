import os
import csv
from glob import glob

# METHODS = ["default", "attn", "film03"]
METHODS = ["ctrl64_attn", "ctrl128_attn", "ctrl1024_attn"]

def find_latest_csv(folder):
    csv_files = glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        return None
    # 按文件名排序，取最后一个（时间戳一般在文件名里）
    csv_files.sort()
    return csv_files[-1]

def extract_average_row(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1 and row[1] == "average":
                # 去掉最后可能的空字符串
                row = [x for x in row if x != ""]
                return row
    return None

results = {}

for method in METHODS:
    csv_path = find_latest_csv(method)
    if csv_path is None:
        print(f"[Warning] No csv found in {method}")
        continue

    avg_row = extract_average_row(csv_path)
    if avg_row is None:
        print(f"[Warning] No average row found in {csv_path}")
        continue

    # 格式: id, average, True, metric1, metric2, ...
    metrics = avg_row[3:]
    results[method] = metrics

# 构建 Markdown 表格
if not results:
    print("No results found.")
    exit()

# 假设三个方法指标数量一致
metric_num = len(next(iter(results.values())))

header = ["Method"] + [f"Metric{i+1}" for i in range(metric_num)]

print("| " + " | ".join(header) + " |")
print("|" + " --- |" * len(header))

for method in METHODS:
    if method not in results:
        continue
    row = [method] + results[method]
    print("| " + " | ".join(row) + " |")