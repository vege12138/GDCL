import numpy as np
import os
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
def construct_graph( features, topk , name):
    fname = './data/{}/knn_{}/tmp.txt'.format(name, name)
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(name, path):
    # dataset_folder = os.sep.join([DATA_FOLDER, "cora"])
    # adj_complete = np.loadtxt(open(os.sep.join([dataset_folder, "W.csv"]), "rb"), delimiter=",")
    # adj_sparse = sp.csr_matrix(adj_complete)

    # features = np.loadtxt(open(os.sep.join([dataset_folder, "fea.csv"]), "rb"), delimiter=",")
    features = np.loadtxt(path)
    data = sp.csr_matrix(features)
    # with np.load(data, allow_pickle=True) as loader:
    #     loader = dict(loader)
    #     if 'attr_matrix.data' in loader.keys():
    #         data = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
    #                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape']).toarray()
    for topk in range(9, 15):
        print(data)
        construct_graph(data, topk , name)
        f1 = open('./data/{}/knn_{}/tmp.txt'.format(dataName, name), 'r')
        f2 = open('./data/{}/knn_{}/c'.format(dataName, name) + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

if __name__ == '__main__':
    dataName = "amac"
    dataPath = "./data/{}/{}_fea.txt".format(dataName , dataName)
    generate_knn(dataName, dataPath)