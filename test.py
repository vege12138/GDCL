import torch

# 定义嵌入
Z = torch.randn(10, 4)

# 定义正样本对
pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])

# 获取正样本对的节点索引
indices1 = pairs[:, 0]
indices2 = pairs[:, 1]

# 获取正样本对的嵌入
embeddings_i = torch.index_select(Z, 0, indices1)
embeddings_j = torch.index_select(Z, 0, indices2)
embeddings = torch.cat((embeddings_i, embeddings_j), 1)
# 将正样本对列表内节点切换为对应节点的嵌入
pairs_embeddings = [(embeddings_i[i], embeddings_j[i]) for i in range(len(pairs))]

print(pairs_embeddings)

