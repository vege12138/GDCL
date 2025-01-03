import numpy as np
import warnings
import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score as ari_score
import torch.utils.data as data_utils
from typing import Union
from kmeans_gpu import kmeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def eva(y_true, y_pred, show_details=True, epoch=0):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    # ari = ari_score(y_true, y_pred)
    if show_details:
        pass
        # print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
        #       ', f1 {:.4f}'.format(f1))
        # print('epoch {}'.format(epoch),':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi),
        #       ', f1 {:.4f}'.format(f1))
    return acc, nmi, f1
def clustering(feature, true_labels, cluster_num, epoch):
    predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, f1 = eva(true_labels, predict_labels.numpy(), show_details=True, epoch = epoch)
    # return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels.numpy()
    # return round(100 * acc, 2), round(100 * nmi, 2), round(100 * f1, 2), predict_labels.numpy()
    return acc, nmi ,f1
def coms_list_to_matrix(communities_list, num_nodes=None):
    """Convert a communities list of len [C] to an [N, C] communities matrix.
    Parameters
    ----------
    communities_list : list
        List of lists of nodes belonging to respective community.
    num_nodes : int, optional
        Total number of nodes. This needs to be here in case
        some nodes are not in any communities, but the resulting
        matrix must have the correct shape [num_nodes, num_coms].
    Returns
    -------
    communities_matrix : np.array, shape [num_nodes, num_coms]
        Binary matrix of community assignments.
    """
    num_coms = len(communities_list)
    if num_nodes is None:
        num_nodes = max(max(cmty) for cmty in communities_list) + 1
    communities_matrix = np.zeros([num_nodes, num_coms], dtype=np.float32)
    for cmty_idx, nodes in enumerate(communities_list):
        communities_matrix[nodes, cmty_idx] = 1
    return communities_matrix


def coms_matrix_to_list(communities_matrix):
    """Convert an [N, C] communities matrix to a communities list of len [C].

    Parameters
    ----------
    communities_matrix : np.ndarray or sp.spmatrix, shape [num_nodes, num_coms]
        Binary matrix of community assignments.

    Returns
    -------
    communities_list : list
        List of lists of nodes belonging to respective community.

    """
    num_nodes, num_coms = communities_matrix.shape
    communities_list = [[] for _ in range(num_coms)]
    nodes, communities = communities_matrix.nonzero()
    for node, cmty in zip(nodes, communities):
        communities_list[cmty].append(node)
    return communities_list


def plot_sparse_clustered_adjacency(A, num_coms, z, o, ax=None, markersize=0.25):
    import seaborn as sns
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()

    colors = sns.color_palette('hls', num_coms)
    sns.set_style('white')

    crt = 0
    for idx in np.where(np.diff(z[o]))[0].tolist() + [z.shape[0]]:
        ax.axhline(y=idx, linewidth=0.5, color='black', linestyle='--')
        ax.axvline(x=idx, linewidth=0.5, color='black', linestyle='--')
        crt = idx + 1

    ax.spy(A[o][:, o], markersize=markersize)
    ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off', labeltop='off')


def adjacency_split_naive(A, p_val, neg_mul=1, max_num_val=None):
    edges = np.column_stack(sp.tril(A).nonzero())
    num_edges = edges.shape[0]
    num_val_edges = int(num_edges * p_val)
    if max_num_val is not None:
        num_val_edges = min(num_val_edges, max_num_val)

    shuffled = np.random.permutation(num_edges)
    which_val = shuffled[:num_val_edges]
    which_train = shuffled[num_val_edges:]
    train_ones = edges[which_train]
    val_ones = edges[which_val]
    A_train = sp.coo_matrix((np.ones_like(train_ones.T[0]), (train_ones.T[0], train_ones.T[1])),
                            shape=A.shape).tocsr()
    A_train = A_train.maximum(A_train.T)

    num_nodes = A.shape[0]
    num_val_nonedges = neg_mul * num_val_edges
    candidate_zeros = np.random.choice(np.arange(num_nodes, dtype=np.int32),
                                       size=(2 * num_val_nonedges, 2), replace=True)
    cne1, cne2 = candidate_zeros[:, 0], candidate_zeros[:, 1]
    to_keep = (1 - A[cne1, cne2]).astype(np.bool).A1
    val_zeros = candidate_zeros[to_keep][:num_val_nonedges]
    if to_keep.sum() < num_val_nonedges:
        raise ValueError("Couldn't produce enough non-edges")

    return A_train, val_ones, val_zeros

def l2_reg_loss(model, scale=1e-5):
    """Get L2 loss for model weights."""
    loss = 0.0
    for w in model.get_weights():
        loss += (w.pow(2.).sum()).cpu()
    return loss * scale
def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))

    return adjs


def laplacian(adj):
    rowsum = np.array(adj.sum(1))
    degree_mat = sp.diags(rowsum.flatten())
    lap = degree_mat - adj
    return torch.FloatTensor(lap.toarray())
def to_sparse_tensor(matrix: Union[sp.spmatrix, torch.Tensor],
                     ) -> Union[torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Args:
        matrix: Sparse matrix to convert.
        cuda: Whether to move the resulting tensor to GPU.

    Returns:
        sparse_tensor: Resulting sparse tensor (on CPU or on GPU).

    """
    if sp.issparse(matrix):
        coo = matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
        return sparse_tensor
    elif torch.is_tensor(matrix):
        row, col = matrix.nonzero().t()
        indices = torch.stack([row, col])
        values = matrix[row, col]
        shape = torch.Size(matrix.shape)
        sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    else:
        raise ValueError(f"matrix must be scipy.sparse or torch.Tensor (got {type(matrix)} instead).")
    # if cuda:
    #     sparse_tensor = sparse_tensor.cuda()
    # return sparse_tensor.coalesce()


# class EdgeSampler(data_utils.Dataset):
#     """Sample edges and non-edges uniformly from a graph.
#
#     Args:
#         A: adjacency matrix as a torch.Tensor.
#         A_fadj: another adjacency matrix as a torch.Tensor.
#         num_pos: number of edges per batch.
#         num_neg: number of non-edges per batch.
#     """
#     def __init__(self, A, A_fadj, num_pos=1000, num_neg=1000):
#         self.num_pos = num_pos
#         self.num_neg = num_neg
#         self.A = A
#         self.A_fadj = A_fadj
#
#         self.edges_adj = torch.nonzero(A).t()
#         self.edges_fadj = torch.nonzero(A_fadj).t()
#
#         self.num_nodes = A.shape[0]
#         self.num_edges_adj = self.edges_adj.shape[0]
#         self.num_edges_fadj = self.edges_fadj.shape[0]
#
#     def __getitem__(self, key):
#         torch.manual_seed(key)  # Set seed for torch operations
#         edges_idx1 = torch.randint(0, self.num_edges_adj, size=(self.num_pos,), dtype=torch.long)
#         next_edges_adj = self.edges_adj[edges_idx1, :]
#
#         edges_idx2 = torch.randint(0, self.num_edges_fadj, size=(self.num_pos,), dtype=torch.long)
#         next_edges_fadj = self.edges_fadj[edges_idx2, :]
#
#         # Select num_neg non-edges
#         generated1 = False
#         generated2 = False
#
#         while not generated1:
#             candidate_ne_adj = torch.randint(0, self.num_nodes, size=(2 * self.num_neg, 2), dtype=torch.long)
#             cne1_a, cne2_a = candidate_ne_adj[:, 0], candidate_ne_adj[:, 1]
#
#             to_keep_adj = (1 - self.A[cne1_a, cne2_a]).bool().view(-1) * (cne1_a != cne2_a).cuda()
#             candidate_ne_adj = candidate_ne_adj.to(to_keep_adj.device)
#             next_nonedges_adj = candidate_ne_adj[to_keep_adj][:self.num_neg]
#             generated1 = to_keep_adj.sum() >= self.num_neg
#
#         while not generated2:
#             candidate_ne_fadj = torch.randint(0, self.num_nodes, size=(2 * self.num_neg, 2), dtype=torch.long).cuda()
#             cne1_a, cne2_a = candidate_ne_fadj[:, 0], candidate_ne_fadj[:, 1]
#
#             to_keep_fadj = (1 - self.A_fadj[cne1_a, cne2_a]).bool().view(-1) * (cne1_a != cne2_a).cuda()
#             candidate_ne_fadj = candidate_ne_fadj.to(to_keep_fadj.device)
#             next_nonedges_fadj = candidate_ne_fadj[to_keep_fadj][:self.num_neg]
#             generated2 = to_keep_fadj.sum() >= self.num_neg
#
#         return next_edges_adj, next_edges_fadj, next_nonedges_adj, next_nonedges_fadj
#
#     def __len__(self):
#         return 2**32
#
#     def __len__(self):
#         return 2**32
#
def collate_fn(batch):
    edges1,edges2 ,  nonedges1 , nonedges2 = batch[0]
    return (edges1,edges2, nonedges1 , nonedges2)

def get_edge_sampler(A , A_fadj , num_pos=1000, num_neg=1000, num_workers=5):
    data_source = EdgeSampler(A ,A_fadj ,  num_pos, num_neg)
    data_source.__getitem__(4)
    return data_utils.DataLoader(data_source, num_workers=num_workers, collate_fn=collate_fn)
class EdgeSampler(data_utils.Dataset):
    """Sample edges and non-edges uniformly from a graph.

    Args:
        A: adjacency matrix.
        num_pos: number of edges per batch.
        num_neg: number of non-edges per batch.
    """
    def __init__(self, A, A_fadj , num_pos=1000, num_neg=1000):
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.A = A
        self.A_fadj = A_fadj

        self.edges_adj = np.transpose(A.nonzero())
        self.edges_fadj = np.transpose(A_fadj.nonzero())

        self.num_nodes = A.shape[0]
        self.num_edges_adj = self.edges_adj.shape[0]
        self.num_edges_fadj = self.edges_fadj.shape[0]


    def __getitem__(self, key):
        np.random.seed(key)
        edges_idx1 = np.random.randint(0, self.num_edges_adj, size=self.num_pos, dtype=np.int64)
        next_edges_adj = self.edges_adj[edges_idx1, :]

        edges_idx2 = np.random.randint(0, self.num_edges_fadj, size=self.num_pos, dtype=np.int64)
        next_edges_fadj = self.edges_fadj[edges_idx2, :]
        # next_edges = np.vstack((next_edges_adj , next_edges_fadj))
        # np.random.shuffle(next_edges_adj)
        # Select num_neg non-edges
        generated1 = False
        generated2 = False


        while not generated1:
            candidate_ne_adj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            # candidate_ne_fadj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)

            cne1_a, cne2_a = candidate_ne_adj[:, 0], candidate_ne_adj[:, 1]
            # cne1_f, cne2_f = candidate_ne_fadj[:, 0], candidate_ne_fadj[:, 1]

            to_keep_adj = (1 - self.A[cne1_a, cne2_a]).astype(bool).A1 * (cne1_a != cne2_a)
            # to_keep_fadj = (1 - self.A[cne1_f, cne2_f]).astype(np.bool).A1 * (cne1_f != cne2_f)

            next_nonedges_adj = candidate_ne_adj[to_keep_adj][:self.num_neg]
            # next_nonedges_fadj = candidate_ne_fadj[to_keep_fadj][:self.num_neg]
            # next_nonedges = np.vstack((next_nonedges_adj , next_nonedges_fadj))
            # np.random.shuffle(next_nonedges)
            generated1= to_keep_adj.sum() >= self.num_neg
            # generated_fadj = to_keep_fadj.sum() >= self.num_neg

        while not generated2:
            candidate_ne_fadj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)
            # candidate_ne_fadj = np.random.randint(0, self.num_nodes, size=(2*self.num_neg, 2), dtype=np.int64)

            cne1_a, cne2_a = candidate_ne_fadj[:, 0], candidate_ne_fadj[:, 1]
            # cne1_f, cne2_f = candidate_ne_fadj[:, 0], candidate_ne_fadj[:, 1]

            to_keep_fadj = (1 - self.A_fadj[cne1_a, cne2_a]).astype(bool).A1 * (cne1_a != cne2_a)
            # to_keep_fadj = (1 - self.A[cne1_f, cne2_f]).astype(np.bool).A1 * (cne1_f != cne2_f)

            next_nonedges_fadj = candidate_ne_fadj[to_keep_fadj][:self.num_neg]
            # next_nonedges_fadj = candidate_ne_fadj[to_keep_fadj][:self.num_neg]
            # next_nonedges = np.vstack((next_nonedges_adj , next_nonedges_fadj))
            # np.random.shuffle(next_nonedges)
            generated2= to_keep_fadj.sum() >= self.num_neg
            # generated_fadj = to_keep_fadj.sum() >= self.num_neg

        return torch.LongTensor(next_edges_adj),torch.LongTensor(next_edges_fadj) ,  torch.LongTensor(next_nonedges_adj) ,  torch.LongTensor(next_nonedges_fadj)

    def __len__(self):
        return 2**32
#data1 : adj    data2 : feature
# def load_cora(data1 , data2, N):
#     A = np.genfromtxt(data1, dtype=np.int32)
#     adj = sp.csr_matrix(A)
#     A = adj
#     adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#     adj.eliminate_zeros()
#     adj_label = A
#     adjs = preprocess_graph(adj, 3, norm='sym', renorm=True)
#     feature_edges = np.genfromtxt(data2, dtype=np.int32)
#
#     fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
#     fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(N, N),
#                          dtype=np.float32)
#     A_fea = sp.csr_matrix(fadj)
#     fadj = fadj - sp.dia_matrix((fadj.diagonal()[np.newaxis, :], [0]), shape=fadj.shape)
#     adj.eliminate_zeros()
#     fadjs = preprocess_graph(fadj, 3, norm='sym', renorm=True)
#     return adjs[0], fadjs[0], adj_label, A_fea

import networkx as nx
def load_cora(data1 , data2, N):
    adj = np.genfromtxt(data1, dtype=np.int32)
    adj = sp.coo_matrix(adj)
    # aedges = np.array(list(adj_edges), dtype=np.int32).reshape(adj_edges.shape)
    # adj = sp.coo_matrix((np.ones(aedges.shape[0]), (aedges[:, 0], aedges[:, 1])), shape=(N, N),
    #                      dtype=np.float32)
    adjL = adj
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_lable = A + sp.eye(A.shape[0])
    # sadj = adj
    # sadj = normalize_adj(adj)
    sadj = normalize_adj(adj_lable)
    # A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # A = A + sp.eye(A.shape[0])
    feature_edges = np.genfromtxt(data2, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    fadjL = fadj
    A_fea = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj_label = A_fea +sp.eye(A.shape[0])
    # fadj = normalize_adj(fadj)
    fadj = normalize_adj(fadj_label)
    return sadj, fadj, adj_lable, A, A_fea, fadj_label, adjL, fadjL

def load_cora_wo(data1 , N):
    adj = np.genfromtxt(data1, dtype=np.int32)
    adj = sp.coo_matrix(adj)
    adjL = adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    sadj = normalize_adj(adj)
    return sadj,  adjL

def load_acm(data1 , data2, N):
    adj_edges = np.genfromtxt(data1, dtype=np.int32)
    aedges = np.array(list(adj_edges), dtype=np.int32).reshape(adj_edges.shape)
    adj = sp.coo_matrix((np.ones(aedges.shape[0]), (aedges[:, 0], aedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    adjL = adj
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_lable = A + sp.eye(A.shape[0])
    # sadj = adj
    sadj = normalize_adj(adj)
    # A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # A = A + sp.eye(A.shape[0])
    feature_edges = np.genfromtxt(data2, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    fadjL = fadj
    A_fea = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj_label = A_fea +sp.eye(A.shape[0])
    fadj = normalize_adj(fadj)
    return sadj, fadj, adj_lable, A, A_fea, fadj_label, adjL, fadjL

def load_acm_wo(data1 , N):
    adj_edges = np.genfromtxt(data1, dtype=np.int32)
    aedges = np.array(list(adj_edges), dtype=np.int32).reshape(adj_edges.shape)
    adj = sp.coo_matrix((np.ones(aedges.shape[0]), (aedges[:, 0], aedges[:, 1])), shape=(N, N),dtype=np.float32)
    adjL = adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    sadj = normalize_adj(adj)
    return sadj, adjL

def load_other4(data1 , data2, N):
    adj_edges = np.genfromtxt(data1, dtype=np.int32)-1
    aedges = np.array(list(adj_edges), dtype=np.int32).reshape(adj_edges.shape)
    adj = sp.coo_matrix((np.ones(aedges.shape[0]), (aedges[:, 0], aedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    adjL = adj
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_lable = A + sp.eye(A.shape[0])
    # sadj = adj
    # sadj = normalize_adj(adj)
    sadj = normalize_adj(adj_lable)
    # A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # A = A + sp.eye(A.shape[0])
    feature_edges = np.genfromtxt(data2, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(N, N),
                         dtype=np.float32)
    fadjL = fadj
    A_fea = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    fadj_label = A_fea +sp.eye(A.shape[0])
    # fadj = normalize_adj(fadj)
    fadj = normalize_adj(fadj_label)

    return sadj, fadj, adj_lable, A, A_fea, fadj_label, adjL, fadjL

def load_other4_wo(data1 , N):
    adj_edges = np.genfromtxt(data1, dtype=np.int32)-1
    aedges = np.array(list(adj_edges), dtype=np.int32).reshape(adj_edges.shape)
    adj = sp.coo_matrix((np.ones(aedges.shape[0]), (aedges[:, 0], aedges[:, 1])), shape=(N, N), dtype=np.float32)
    adjL = adj
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    sadj = normalize_adj(adj)
    return sadj, adjL

def load_dataset(file_name ,fea_path):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        A = sp.csr_matrix((loader['adj_matrix.data'], loader['adj_matrix.indices'],
                           loader['adj_matrix.indptr']), shape=loader['adj_matrix.shape'])
        adj_label = A.todense()

        feature_edges = np.genfromtxt(fea_path, dtype=np.int32)
        fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
        fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(A.shape[0], A.shape[0]),
                             dtype=np.float32)
        A_fea = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
        # nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

        if 'attr_matrix.data' in loader.keys():
            X = sp.csr_matrix((loader['attr_matrix.data'], loader['attr_matrix.indices'],
                               loader['attr_matrix.indptr']), shape=loader['attr_matrix.shape'])
        else:
            X = None
        # generate_knn(X.toarray())
        Z = sp.csr_matrix((loader['labels.data'], loader['labels.indices'],
                           loader['labels.indptr']), shape=loader['labels.shape'])

        # Remove self-loops
        A = A.tolil()
        A.setdiag(0)
        A = A.tocsr()

        A_fea = A_fea.tolil()
        A_fea.setdiag(0)
        A_fea = A_fea.tocsr()

        # Convert label matrix to numpy
        if sp.issparse(Z):
            Z = Z.toarray().astype(np.float32)

        graph = {
            'A': A,
            'A_fea': A_fea,
            'X': X,
            'Z': Z,
            'adj_label' : adj_label
        }

        node_names = loader.get('node_names')
        if node_names is not None:
            node_names = node_names.tolist()
            graph['node_names'] = node_names

        attr_names = loader.get('attr_names')
        if attr_names is not None:
            attr_names = attr_names.tolist()
            graph['attr_names'] = attr_names

        class_names = loader.get('class_names')
        if class_names is not None:
            class_names = class_names.tolist()
            graph['class_names'] = class_names

        return graph

__all__ = [
    'symmetric_jaccard',
    'overlapping_nmi',
]


def symmetric_jaccard(coms_1, coms_2):
    """Quantify agreement between two community assignments based on symmetric Jaccard similarity.

    Computed as in the CESNA paper as
    0.5 * (1 / |C1| * sum_{c1 in C1} max_{c2 in C2} jac(c1, c2) +
           1 / |C2| * sum_{c2 in C2} max_{c1 in C1} jac(c1, c2))

    Parameters
    ----------
    coms_1 : list of len [num_coms] or array-like of shape [num_nodes, num_coms]
        First community assignment to compare.
    coms_2 : list of len [num_coms] or array-like of shape [num_nodes, num_coms]
        Second community assignment to compare.

    Returns
    -------
    symmetric_jaccard_similarity : float
        Symmetric average best Jaccard similarity between two community assignments.

    """
    # Convert community assignments to matrix format
    if isinstance(coms_1, list):
        F1 = coms_list_to_matrix(coms_1)
    elif len(coms_1.shape) == 2:
        F1 = coms_1
    else:
        raise ValueError("coms_1 must be either a list or a matrix.")
    if isinstance(coms_2, list):
        F2 = coms_list_to_matrix(coms_2)
    elif len(coms_2.shape) == 2:
        F2 = coms_2
    else:
        raise ValueError("coms_2 must be either a list or a matrix.")

    intersections = F1.T.dot(F2)
    sum_F1 = F1.sum(0)
    sum_F2 = F2.sum(0)
    unions = (np.ravel(sum_F2) + np.ravel(sum_F1)[:, None]) - intersections
    jacs = intersections / unions
    return 0.5 * (jacs.max(0).mean() + jacs.max(1).mean())

def compute_laplacian(A):
  D = torch.zeros((A.shape[0], A.shape[0]))
  for i in range(A.shape[0]):
      for j in range(A.shape[0]):
          if A[i][j] == 1:
              D[i][i] += 1
  D_inv_sqrt = torch.sqrt(1.0 / D)
  D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
  L = torch.eye(A.size(0)).to(A.device) - D_inv_sqrt @ A @ D_inv_sqrt
  return L

def overlapping_nmi(X, Y):
    """Compute NMI between two overlapping community covers.

    Parameters
    ----------
    X : array-like, shape [N, m]
        Matrix with samples stored as columns.
    Y : array-like, shape [N, n]
        Matrix with samples stored as columns.

    Returns
    -------
    nmi : float
        Float in [0, 1] quantifying the agreement between the two partitions.
        Higher is better.

    References
    ----------
    McDaid, Aaron F., Derek Greene, and Neil Hurley.
    "Normalized mutual information to evaluate overlapping
    community finding algorithms."
    arXiv preprint arXiv:1110.2515 (2011).

    """
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forgot to transpose the F matrix")
    X = X.T
    Y = Y.T
    def cmp(x, y):
        """Compare two binary vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        """Compute contribution of a single term to the entropy."""
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        """Compute conditional entropy between two vectors."""
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)
    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)