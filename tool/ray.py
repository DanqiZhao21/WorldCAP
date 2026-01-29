import ray

ray.init()  # 如果还没启动 Ray

# 查看每个节点信息
nodes = ray.nodes()
for node in nodes:
    print("Node ID:", node['NodeID'])
    print("Alive:", node['Alive'])
    # Ray 节点的 CPU 总量
    print("Resources:", node['Resources'])
