import numpy as np

# 读取原始文件
data = np.load('/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_41.npy')

# 检查原始形状
print("Original shape:", data.shape)  # (256, 41, 11)

# 从第2维（41）均匀采样8个点
indices = np.linspace(0, data.shape[1] - 1, 8, dtype=int)
data_sampled = data[:, indices, :3]  # 取第3维的前三个

# 检查新形状
print("New shape:", data_sampled.shape)  # (256, 8, 3)

# 保存为新的文件
np.save('/home/zhaodanqi/clone/WoTE/ControllerExp/LAB3_LQRstyle_aggressive/LQRstyle_aggressive_8.npy', data_sampled)
