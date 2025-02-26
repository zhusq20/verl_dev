import pandas as pd

# 读取原始 parquet 文件
input_file = "/workspace/HuggingFace-Download-Accelerator/hf_hub/datasets--PRIME-RL--Eurus-2-RL-Data/train.parquet"
output_file = "/workspace/HuggingFace-Download-Accelerator/hf_hub/datasets--PRIME-RL--Eurus-2-RL-Data/train_short.parquet"
n_rows = 128  # 需要提取的前几项

# 加载数据
df = pd.read_parquet(input_file)

# 提取前 n 行
df_subset = df.head(n_rows)

# 保存为新的 parquet 文件
df_subset.to_parquet(output_file, index=False)

print(f"已成功提取前 {n_rows} 行并保存到 {output_file}")