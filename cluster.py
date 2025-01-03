from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
def community (z1, clusters):
    # z1 = z1.detach().numpy()
    #for i in range(1,11):
    # ts = manifold.TSNE(n_components=2, perplexity=3, early_exaggeration=10, n_iter=50000, learning_rate=500,  angle=0.5, init='random')
    ts = TSNE(n_components=2, perplexity=50, early_exaggeration=500, n_iter=100000, learning_rate=100, angle=0.5,
                       init='random')
    z = ts.fit_transform(z1)
    #C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=1000, tol=0.001, n_init=20, init='k-means++')
    # C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=1000, tol=0.001, n_init=20, init='k-means++')
    C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=100, tol=0.01, n_init=3)
    C_model.fit(z)
    commu_predict = C_model.labels_
    #torch.save(commu_predict, './pred.pt')
    # plt.figure(figsize=(10,10), dpi=80)
    # z = plt.scatter(z[:, 0], z[:, 1], c=commu_predict, marker='o', s=10)  # 不同类别不同颜色
    # plt.title("k-means")
    #plt.savefig('./cora{}.pdf'.format(i))
    # plt.show()
    # print(i)
    return commu_predict