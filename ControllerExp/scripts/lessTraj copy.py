import numpy as np
from sklearn.cluster import KMeans

# 加载原始 anchor 轨迹
anchors = np.load('/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_Original_256.npy')  # shape: [256, T, 3]

# 聚类时需要将每条轨迹展平成一维
B, T, D = anchors.shape
anchors_flat = anchors.reshape(B, -1)  # shape: [256, T*3]

# KMeans 聚成 128 类
kmeans_128 = KMeans(n_clusters=128, random_state=42)
labels_128 = kmeans_128.fit_predict(anchors_flat)
centers_128 = kmeans_128.cluster_centers_.reshape(128, T, D)
np.save('/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_KMeans_128.npy', centers_128)

# KMeans 聚成 64 类
kmeans_64 = KMeans(n_clusters=64, random_state=42)
labels_64 = kmeans_64.fit_predict(anchors_flat)
centers_64 = kmeans_64.cluster_centers_.reshape(64, T, D)
np.save('/home/zhaodanqi/clone/WoTE/ControllerExp/Anchors_KMeans_64.npy', centers_64)

print('KMeans聚类完成，已保存128和64类的轨迹中心。')