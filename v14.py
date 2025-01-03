from __future__ import division
import warnings
import torch
warnings.filterwarnings('ignore')

from torch import optim
from net_v7 import *




from dataprocess1 import load_other4 , get_edge_sampler , l2_reg_loss, load_cora, sparse_mx_to_torch_sparse_tensor, clustering, eva, load_acm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
def log(x):
    return torch.log(x + 1e-8)
def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj
def laplacian_filtering(A, X, t):
    A_tmp = A - torch.diag_embed(torch.diag(A))
    A_norm = normalize_adj(A_tmp, self_loop=True, symmetry=True)
    A_norm = torch.tensor(A_norm, dtype=torch.float32)
    I = torch.eye(A.shape[0],dtype=torch.float32)
    L = I - A_norm
    for i in range(t):
        X = (I - L) @ X
    return X.float()


def adaptive_alpha(adj_matrix: np.ndarray) -> float:
    # 设定alpha的基础值
    base_alpha = 0.6
    # 计算图的稀疏度
    sparsity = 1.0 - np.count_nonzero(adj_matrix) / adj_matrix.size
    # 根据稀疏度来调整alpha
    alpha = base_alpha * (1 + sparsity)
    # 确保alpha值在合理的范围内
    return min(max(alpha, 0.1), 0.3)
def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.1) -> np.ndarray:
    # alpha = adaptive_alpha(adj_matrix)
    adj_matrix = np.array(adj_matrix)
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix + np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)
def to_csr(tensor: torch.Tensor) -> sp.csr_matrix:
    indices = tensor.nonzero()
    nnz = indices.size(0)
    # 构建CSR矩阵
    row = indices[:, 0]
    col = indices[:, 1]
    data = tensor[indices[:, 0], indices[:, 1]]
    return sp.csr_matrix((data, (row, col)), shape=(tensor.size(0), tensor.size(1)))
def cal_K(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    degree_sum = np.sum(adj_matrix, axis=1)
    # 找到最大度数
    max_degree = np.max(degree_sum)
    return int(max_degree //2 )
def get_top_k_matrix(A: np.ndarray, k: int = 128) -> np.ndarray:
    # sparsity = 1.0 - np.count_nonzero(A) / A.size
    # target_density = 0.3  # 假设我们想要的目标密度
    # k = int(target_density * A.shape[0] / (1 - sparsity))
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return torch.tensor(A/norm, dtype=torch.float32)


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # Self-loops
    A_loop = sp.eye(N) + A

    # Symmetric transition matrix
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # PPR-based diffusion
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # Sparsify using threshold epsilon
    S_tilde = S.multiply(S >= eps)

    # Column-normalized transition matrix on graph S_tilde
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S

def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def run(exp_seed, a):
    alpha = (a+1)/10

    acc_mean = []
    nmi_mean = []
    f1_mean = []
    ari_mean = []
    random.seed(exp_seed)
    np.random.seed(exp_seed)
    torch.manual_seed(exp_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(exp_seed)

    NMIS = []
    F1 = []
    ARI = []
    ACC = []
    NCE = newNCE(0.6)  # wiki 0.8

    def loss_NCE(embedding_topo1, embedding_topo2, embedding_fea1, embedding_fea2):
        loss1 = NCE(embedding_topo1, embedding_topo2)
        loss2 = NCE(embedding_fea1, embedding_fea2)
        return loss1, loss2
    dataName = "wiki"  #可选：wiki cora citeseer amac；其余数据集需更换读取处理方式
    data1 = 'D:./data/{}/{}_adj.txt'.format(dataName, dataName)
    data2 = 'D:./data/{}/knn_{}/c9.txt'.format(dataName, dataName)
    label_file = 'D:./data/{}/{}_label.txt'.format(dataName, dataName)
    features_file = 'D:./data/{}/{}_fea.txt'.format(dataName, dataName)
    #labels = np.genfromtxt(label_file, dtype=np.int32)[:, 1]-1
    labels = np.genfromtxt(label_file, dtype=np.int32)
    N, K = labels.shape[0], len(np.unique(labels))  # cora、citeseer的label从1开始
    # N, K = labels.shape[0] , labels.max()+1 # cora、citeseer的label从1开始
    sadj, fadj ,adj_lable, A , A_fea, fadj_label = load_cora(data1 ,data2, N)
    #sadj, fadj, adj_lable, A, A_fea, fadj_label = load_other4(data1, data2, N)
    # sadj, fadj, adj_lable, A_fea= load_cora(data1 ,data2, N)
    # sadj = torch.FloatTensor(sadj.toarray())
    # fadj = torch.FloatTensor(fadj.toarray())
    # A = adj_lable

    # sadj, fadj, adj_lable, A, A_fea, fadj_label = load_other4(data1, data2, N)
    # sadj, fadj ,adj_lable, A , A_fea, fadj_label = load_acm(data1 ,data2, N)
    cuda = torch.cuda.is_available()
    if cuda:
        sadj = sadj.cuda()
        fadj = fadj.cuda()

    adj_lable = adj_lable.todense()
    diff_A = sparse_mx_to_torch_sparse_tensor(A).cuda()
    diff_A_fea = sparse_mx_to_torch_sparse_tensor(A_fea).cuda()
    # fadj_label = fadj_label.todense()
    features = np.genfromtxt(features_file)
    X_ori = torch.tensor(features, dtype=torch.float32).cuda()
    y_true = labels - 1
    # y_true = labels
    weight_decay = 1e-2  # strength of L2 regularization on GNN weights
    dropout = 0.5  # whether to use dropout
    batch_norm = True  # whether to use batch norm
    lr = 0.01  # learning rate
    max_epochs = 500  # number of epochs to train
    balance_loss = True  # whether to use balanced loss
    stochastic_loss = True  # whether to use stochastic or full-batch training
    batch_size = 20000  # batch size (only for stochastic training)
    # x_norm = torch.tensor(features, dtype=torch.float32).cuda()
    x_norm = laplacian_filtering(torch.tensor(A.todense(), dtype=torch.float32),  torch.tensor(features, dtype=torch.float32), 2).cuda()

    feat_dim = x_norm.shape[1]

    model = NOCD_DL(500, 500, 2000, 2000, 500, 500, feat_dim, K, K, 1, dropout=dropout, batch_norm=batch_norm,
                    name=dataName)
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    adj_norm = sadj
    fadj_norm = fadj
    topK1 = cal_K(A.todense())
    topK2 = cal_K(A_fea.todense())
    # sadj_diff = gdc(adj_norm, )
    sadj_diff = get_ppr_matrix(A.todense(), alpha=0.05) #citeseer
    sadj_diff = get_top_k_matrix(sadj_diff, 128)
    fadj_diff = get_ppr_matrix(A_fea.todense(), alpha=0.05)  # acm 0.8    wiki 0.2
    fadj_diff = get_top_k_matrix(fadj_diff, 128)   # CITERSEER 500 0.05

    A2 = to_csr(sadj_diff)
    A2_fea = to_csr(fadj_diff)
    sampler = get_edge_sampler(A, A_fea, A2, A2_fea, batch_size, batch_size, num_workers=5)
    decoder = BerpoDecoder(N, A.nnz, balance_loss=balance_loss)
    sadj_diff = sadj_diff.cuda()
    fadj_diff = fadj_diff.cuda()
    weight_tensor, norm = compute_loss_para(torch.tensor(A_fea.todense()))
    # with torch.no_grad():
    #     z =  F.relu(model(x_norm , adj_norm , fadj_norm, sadj_diff, fadj_diff)[1])
    with torch.no_grad():
        _, _, _, _, z = model.ae(X_ori)
    # kmeans = KMeans(n_clusters=K, random_state=0).fit(z.data.detach().cpu().numpy())
    kmeans = KMeans(n_clusters=K, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()
    print(f'___________________________________________{exp_seed}____________________________________________')
    for epoch, batch in enumerate(sampler):
        if epoch >= max_epochs:
            break

        if epoch % 1 == 0:
            # with torch.no_grad():
            #     model.eval()
            # Compute validation loss
            x_bar, z, q, emb, pred, fea_h4, topo_emb, fea_emb, newEmb1, newEmb2 = model(x_norm, adj_norm, fadj_norm,
                                                                                        sadj_diff, fadj_diff)
            tmp_q = q.data
            p = target_distribution(tmp_q)

            res = pred.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(y_true, res, show_details=True, epoch=epoch)
            NMIS.append(nmi)
            F1.append(f1)
            ACC.append(acc)
            ARI.append(ari)

        # model.train()
        # optimizer.zero_grad()
        x_bar, z, q, emb, pred, fea_h4, topo_emb, fea_emb, newEmb1, newEmb2 = model(x_norm, adj_norm, fadj_norm,
                                                                                    sadj_diff,
                                                                                    fadj_diff)  # recovered是编码器重构的拓扑结构 ， mu是encoder后的嵌入
        one1_idx, one2_idx, one3_idx, one4_idx, zero1_idx, zero2_idx, zero3_idx, zero4_idx = batch
        # all_targets_RGC, all_similarities_RGC = generate_targets_and_similarities_AGC(sparse_mx_to_torch_sparse_tensor(A).to_dense().cuda(), diff_A.to_dense(), sparse_mx_to_torch_sparse_tensor(A_fea).to_dense().cuda(), diff_A_fea.to_dense(), topo_emb, newEmb1, fea_emb,
        #                                                                               newEmb2)
        # # all_similarities_RGC = torch.sigmoid(all_similarities_RGC)
        # loss_contrastive_RGC = F.binary_cross_entropy(all_similarities_RGC, all_targets_RGC)
        # loss_contrastive_RGC = loss_contrastive_RGC.mean().cpu()
        loss_nce1, loss_nce2 = loss_NCE(topo_emb, newEmb1, fea_emb, newEmb2)
        loss_contrastive = loss_nce2 + loss_nce1
        # loss_decoder = decoder.loss_batch(F.relu(emb), one1_idx, zero1_idx)
        loss_decoder1 = decoder.loss_batch(F.relu(pred), one1_idx, zero1_idx).cpu()
        loss_decoder2 = decoder.loss_batch(F.relu(pred), one2_idx, zero2_idx).cpu()
        loss_decoder3 = decoder.loss_batch(F.relu(pred), one3_idx, zero3_idx).cpu()
        loss_decoder4 = decoder.loss_batch(F.relu(pred), one4_idx, zero4_idx).cpu()
        #
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean').cpu()
        re_loss = F.mse_loss(x_bar, X_ori).cpu()
        ce_loss = F.kl_div(log(pred), p, reduction='batchmean').cpu()
        # loss = 1 * loss_decoder + 100 * kl_loss + 100 * re_loss + 0.001 * ce_loss + loss_contrastive_RGC
        # loss = 1 * loss_decoder + 100 * kl_loss + 100 * re_loss + 0.001 * ce_loss
        # loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 +1 * loss_decoder4 +  1 * kl_loss + 1 * re_loss + 0.01 * ce_loss + loss_contrastive_RGC
        # loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 +1 * loss_decoder4 +  100 * kl_loss + 100 * re_loss + 0.01 * ce_loss + 1*loss_contrastive_RGC
        # loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 + 1 * loss_decoder4 + 0.00005 * loss_nce1 + 0.00005*loss_nce2+  1000 * kl_loss + 1000 * re_loss + 0.01 * ce_loss
        # loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 + 1 * loss_decoder4 +  1 * kl_loss + 1 * re_loss + 0.05 * ce_loss +  0.000001 * loss_contrastive
        # loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 + 1 * loss_decoder4 + 1000 * kl_loss + 1000 * re_loss + 0.01 * ce_loss + 0.1 * loss_contrastive
        loss = 1 * loss_decoder1 + 1 * loss_decoder2 + 1 * loss_decoder3 +1 * loss_decoder4  + 1000 * kl_loss + 1000 * re_loss + 0.001 * ce_loss + 0.1 * loss_contrastive
        # loss /= 100
        # print("loss_decoder1: {loss_decoder1}\n"
        #       f"loss_decoder2: {loss_decoder2}\n"
        #       f"loss_decoder3: {loss_decoder3}\n"
        #       f"loss_decoder4: {loss_decoder4}\n"
        #       f"kl_loss: {kl_loss}\n"
        #       f"re_loss: {re_loss}\n"
        #       f"ce_loss: {ce_loss}\n"
        #       f"loss_contrastive: {loss_contrastive}\n"
        #       f"Total Loss: {loss}")
        loss = loss/10
        loss += l2_reg_loss(model, scale=weight_decay)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    max_nmi = (np.array(NMIS)).max()
    max_f1score = (np.array(F1)).max()
    max_Acc = (np.array(ACC)).max()
    max_ari = (np.array(ARI)).max()

    results.append([
        (np.array(ARI)).mean(), max_Acc, (np.array(NMIS)).mean(), max_nmi,
        (np.array(ARI)).mean(), max_ari, (np.array(F1)).mean(), max_f1score
    ])
    print("max_acc : {:.3f}".format( max_Acc))
    print("---------------------------------------------------------------------------------")
    print(" max_nmi : {:.3f}".format( max_nmi))
    print("---------------------------------------------------------------------------------")
    print("max_ari : {:.3f}".format( max_ari))
    print("---------------------------------------------------------------------------------")
    print("max_f1score : {:.3f}".format( max_f1score))
    print("---------------------------------------------------------------------------------")
    acc_mean.append(max_Acc)
    nmi_mean.append(max_nmi)
    ari_mean.append(max_ari)
    f1_mean.append(max_f1score)

if __name__ == '__main__':

        # if(epoch == 200):
        #     ts = manifold.TSNE(n_components=2, perplexity=35, early_exaggeration=500,n_iter=2000, learning_rate=500, angle=0.5,init='random')
        #     hidemb = pred.detach().cpu().numpy()
        #     z = ts.fit_transform(hidemb)
        #     plt.figure(figsize=(4,4),dpi=120)
        #     plt.scatter(z[:,0], z[:,1],c=res, marker='o', s=5)
        #     plt.title("k-means")
        #     plt.show()

        # def run_all_experiments():
        #     data_names = ["cora", "citeseer", "wiki"]  # 示例数据集列表
        #     experiment_seeds = list(range(10))  # 10次实验的种子
        #     results = {'acc_mean': [], 'nmi_mean': [], 'f1_mean': [], 'ari_mean': []}
        #
        #     for data_name in data_names:
        #         print(f"Running experiments for dataset: {data_name}")
        #         for seed in experiment_seeds:
        #             run(data_name, seed)
        #         # 将结果存储在列表中，以便稍后写入Excel
        #         results['acc_mean'].append(acc_mean)
        #         results['nmi_mean'].append(nmi_mean)
        #         results['f1_mean'].append(f1_mean)
        #         results['ari_mean'].append(ari_mean)


        # experiment_seeds = [1,2,3,4,5,6,7,8,9,10]
        experiment_seeds = list(range(1))

        for a in range(1):
            results = []
            for seed in experiment_seeds:
                # 设置随机种子
                # run(seed)
                # random.seed()
                # np.random.seed()
                # torch.manual_seed(torch.initial_seed())

                run(seed, a)

        # acc_val = np.mean(acc_mean)
        # acc_val_max = np.max(acc_mean)
        # nmi_val = np.mean(nmi_mean)
        # nmi_val_max = np.max(nmi_mean)
        # ari_val = np.mean(ari_mean)
        # ari_val_max = np.max(ari_mean)
        # f1_val = np.mean(f1_mean)
        # f1_val_max = np.max(f1_mean)



            results = np.array(results)

            # 计算均值和标准差
            means = results.mean(axis=0)
            stds = results.std(axis=0)
            vars = results.var(axis=0)

            # 打印所有结果，格式化为制表符分隔
            print(f"_______________amac____________________")
            print("final_acc\tmax_acc\tfinal_nmi\tmax_nmi\tfinal_ari\tmax_ari\tfinal_f1\tmax_f1")
            for row in results:
                print("\t".join(f"{x:.3f}" for x in row))

            # 打印均值和标准差
            # print("Mean")
            print("\t".join(f"{x:.3f}" for x in means))
            # print("Std")
            print("\t".join(f"{x:.3f}" for x in stds))
            print("\t".join(f"{x:.4f}" for x in vars))
        # print("average_acc : {:.3f}, max_acc : {:.3f}".format( acc_val, acc_val_max))
        # print("---------------------------------------------------------------------------------")
        # print(" average_nmi : {:.3f}, max_nmi : {:.3f}".format( nmi_val, nmi_val_max))
        # print("---------------------------------------------------------------------------------")
        # print("average_ari : {:.3f}, max_ari : {:.3f}".format( ari_val, ari_val_max))
        # print("---------------------------------------------------------------------------------")
        # print("average_f1score : {:.3f}, max_f1 : {:.3f}".format( f1_val, f1_val_max))
        # print("---------------------------------------------------------------------------------")