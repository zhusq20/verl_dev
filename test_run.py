import torch

# 假设原始tensor (batch_size=4, seq_len=3 为例子)
idx = torch.tensor([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11]
])

attention_mask = torch.tensor([
    [1, 1, 1],
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

position_ids = torch.tensor([
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2],
    [0, 1, 2]
])

# batch_size = 4, mask长度和batch_size一致
mask = torch.tensor([1, 0, 1, 0])  # 第0,2个样本复制n次；第1,3个不复制
n = 3

# 计算repeat_counts (shape=[batch_size])
repeat_counts = torch.where(mask.bool(), torch.tensor(n-1), torch.tensor(1))

# 在dim=0维度上repeat_interleave，保持batch内部顺序不变
expanded_idx = idx.repeat_interleave(repeat_counts, dim=0)
expanded_attention_mask = attention_mask.repeat_interleave(repeat_counts, dim=0)
expanded_position_ids = position_ids.repeat_interleave(repeat_counts, dim=0)

batch_size = expanded_idx.shape[0]

print("Expanded idx:\n", expanded_idx)
print("Expanded attention_mask:\n", expanded_attention_mask)
print("Expanded position_ids:\n", expanded_position_ids)
print("New batch_size:", batch_size)