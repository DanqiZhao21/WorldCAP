# import pandas as pd

# # ===== 修改成你的两个 CSV =====
# csv1 = "/mnt/data/navsim_workspace/exp/eval/WoTE/default/2026.01.06.10.29.38.csv"
# csv2 = "/mnt/data/navsim_workspace/exp/eval/WoTE/default/2026.01.06.10.36.10.csv"

# # ===== 读取 CSV（有 header）=====
# df1 = pd.read_csv(csv1)
# df2 = pd.read_csv(csv2)

# # ===== 只取你关心的列 =====
# cols = [
#     "token",
#     "driving_direction_compliance",
#     "ego_progress",
#     "score",
# ]

# df1_sub = df1[cols].copy()
# df2_sub = df2[cols].copy()

# # ===== 重命名，区分两次 eval =====
# df1_sub.columns = [
#     "token",
#     "driving_direction_compliance_run1",
#     "ego_progress_run1",
#     "score_run1",
# ]

# df2_sub.columns = [
#     "token",
#     "driving_direction_compliance_run2",
#     "ego_progress_run2",
#     "score_run2",
# ]

# # ===== 按 token 对齐 =====
# merged = pd.merge(df1_sub, df2_sub, on="token", how="inner")

# # ===== 输出 =====
# output_csv = "compare_10.29.38_vs_10.36.10.csv"
# merged.to_csv(output_csv, index=False)

# print(f"Saved comparison CSV to: {output_csv}")


import pandas as pd

# ===== 输入 / 输出 =====
input_csv = "/home/zhaodanqi/clone/WoTE/tool/smalltool/compare_10.29.38_vs_10.36.10.csv"
output_csv = "compare_10.29.38_vs_10.36.10_beautified.csv"

# ===== 读取 =====
df = pd.read_csv(input_csv)

# ===== 重新排列列顺序（更好看）=====
ordered_cols = [
    "token",
    "driving_direction_compliance_run1",
    "driving_direction_compliance_run2",
    "ego_progress_run1",
    "ego_progress_run2",
    "score_run1",
    "score_run2",
]
df = df[ordered_cols]

# ===== 数值列统一保留 4 位小数 =====
numeric_cols = df.columns.drop("token")
df[numeric_cols] = df[numeric_cols].round(4)

# ===== 保存 =====
df.to_csv(output_csv, index=False)

print(f"Beautified CSV saved to: {output_csv}")
