from __future__ import division
import warnings

import torch
import random
from sklearn import manifold
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import networkx as nx
# from plot_tsne import plot_tsne
import scipy.io as sio
from evaluation import  eva
import networkx.algorithms.community as nx_comm
from cluster import community
from torch import optim
from net_work1 import *
from create_graph import  generate_knn
# dataName = "winconsin"
# dataPath = "./data/winconsin_fea.txt"
# generate_knn(dataName, dataPath)
from dataprocess import load_other4 , to_sparse_tensor , get_edge_sampler ,l2_reg_loss, load_cora, load_acm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(x):
    return torch.log(x + 1e-8)

dataName = "wiki"
data1 = './data/{}/{}_adj.txt'.format(dataName , dataName)
data2 = './data/{}/knn_{}/c9.txt'.format(dataName , dataName)
label_file = './data/{}/{}_label.txt'.format(dataName , dataName)
features_file = './data/{}/{}_fea.txt'.format(dataName , dataName)
# labels = np.genfromtxt(label_file, dtype=np.int32)[: , 1]
labels = np.genfromtxt(label_file, dtype=np.int32)
N, K = labels.shape[0] , labels.max() # cora、citeseer的label从1开始
# N, K = labels.shape[0] , labels.max()+1 # acm的label从1开始
sadj, fadj , adj_lable, A , A_fea , fadj_label, adjL, fadjL= load_cora(data1 ,data2, N)
# sadj, fadj ,adj_lable, A , A_fea , fadj_label, adjL, fadjL= load_acm(data1 ,data2, N)
# sadj, fadj ,adj_lable, A , A_fea, fadj_label, adjL, fadjL = load_other4(data1 ,data2, N)
adjL = np.array(adjL.todense())
# fadjL = np.array(fadjL.todense())
adj_lable = adj_lable.todense()
# fadj_label = torch.tensor(fadj_label).to(device)
# fadj_label = torch.tensor(fadjL.todense(),dtype=torch.float32)
fadj_label = torch.tensor(fadjL.todense(),dtype=torch.float32).to(device)
# fadj_label = A_fea.todense()
# L_adj = compute_laplacian(sadj.to_dense())
# L_fadj = compute_laplacian(fadj.to_dense())
features = np.genfromtxt(features_file)
# X_ori = torch.tensor(features ,dtype=torch.float32)
X_ori = torch.tensor(features ,dtype=torch.float32).to(device)
# y_true = labels
y_true = labels -1
weight_decay = 1e-2  # strength of L2 regularization on GNN weights
dropout = 0.5           # whether to use dropout
batch_norm = True       # whether to use batch norm
lr = 0.01   # learning rate
max_epochs = 500        # number of epochs to train
balance_loss = True     # whether to use balanced loss
stochastic_loss = True  # whether to use stochastic or full-batch training
batch_size = 20000    # batch size (only for stochastic training)
x_norm =  torch.tensor(features ,dtype=torch.float32)
x_norm = x_norm.to(device)
feat_dim =x_norm.shape[1]

model = NOCD_DL(500 ,500 , 2000 , 2000 , 500 , 500 ,feat_dim, 10, K , 1 ,  dropout=dropout , batch_norm=batch_norm ,name = dataName)
model = model.to(device)
optimizer = optim.Adam(model.parameters())
adj_norm = sadj.to(device)
fadj_norm  = fadj.to(device)
sampler = get_edge_sampler( A, A_fea , batch_size, batch_size, num_workers=5)
decoder = BerpoDecoder(N, A.nnz, balance_loss=balance_loss)

with torch.no_grad():
    _, _, _, _, z =  model.ae(x_norm)
kmeans = KMeans(n_clusters=K, n_init=20)
y_pred = kmeans.fit_predict(z.data.cpu().numpy())
y_pred_last = y_pred
model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)


def build_loss(recons_A):
    # diagonal elements
    epsilon = torch.tensor(10 ** -7)
    recons_A = recons_A - recons_A.diag().diag()
    pos_weight = (N * N - fadj_label.sum()) / A_fea.sum()
    loss_1 = pos_weight * fadj_label.mul((1 / torch.max(recons_A, epsilon)).log()) + \
             (1 - fadj_label).mul((1 / torch.max((1 - recons_A), epsilon)).log())
    loss_1 = loss_1.sum() / (N ** 2)

    loss = loss_1
    return loss
def compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm

# weight_tensor, norm = compute_loss_para(torch.tensor(A_fea.todense()))
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

if __name__ == '__main__':
    NMIS=[]
    F1 = []
    ARI = []
    Con = []
    ACC =[]
    Qs = []

    for epoch, batch in enumerate(sampler):
        if epoch >=500:
            break
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                # Compute validation loss
                # x_bar, z, q, emb, pred, fea_h4, fadj_recon = model(x_norm, adj_norm, fadj_norm)
                x_bar, z, q, emb, pred, fea_h4, fadj_recon = model(x_norm, adj_norm, fadj_norm)
                tmp_q = q.data
                p = target_distribution(tmp_q)
                val_loss = decoder.loss_full(F.relu(pred), A)
                # epsilon = torch.tensor(10 ** -7).to(device)
                # indicator = pred / pred.norm(dim=1).reshape((N, -1)).max(epsilon)
                # indicator = indicator.detach().cpu().numpy()
                # km = KMeans(n_clusters=K).fit(indicator)
                # prediction = km.predict(indicator)
                # acc, nmi, ari, f1 = cal_clustering_metric(self.labels.cpu().numpy(), prediction)
                # acc, nmi, ari, f1, con, m = eva(y_true, prediction, np.array(A.todense()))
                res2 = pred.data.cpu().numpy().argmax(1)
                # kmeans = KMeans(n_clusters=K).fit(pred.cpu())
                # res2 = kmeans.predict(pred.cpu())
                # res2 = community(pred, K)
                acc, nmi, ari, f1, con, m = eva(y_true, res2, adjL)
                NMIS.append(nmi)
                F1.append(f1)
                Qs.append(m)
                ACC.append(acc)
                ARI.append(ari)
                Con.append(con)
                # losses.append(val_loss)
                print(
                    f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, acc = {acc:.3f} , nmi = {nmi:.3f} , ari = {ari:.3f} , F1-score = {f1:.3f}, conductance = {con:.3f},  modularity = {m:.3f}')

        model.train()
        optimizer.zero_grad()
        x_bar ,  z , q , emb , pred, fea_h4, fadj_recon = model(x_norm, adj_norm, fadj_norm)  # recovered是编码器重构的拓扑结构 ， mu是encoder后的嵌入
        # q = model.get_Q(Z)

        one1_idx, one2_idx, zero1_idx, zero2_idx = batch

        loss_decoder = decoder.loss_batch(F.relu(pred), one1_idx, zero1_idx)
        loss_decoder2 = decoder.loss_batch(F.relu(pred), one2_idx, zero2_idx)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # re_loss = F.mse_loss(x_bar.detach().cpu(), X_ori)
        re_loss = F.mse_loss(x_bar, x_norm)
        # fadj_re_loss = build_loss(fadj_recon.detach().cpu())
        # fadj_re_loss = F.binary_cross_entropy(fadj_recon.detach().cpu().view(-1), fadj_label.view(-1))
        fadj_re_loss = F.binary_cross_entropy(fadj_recon, fadj_label)
        # fadj_re_loss = norm * F.binary_cross_entropy(fadj_recon.view(-1), fadj_label, weight = weight_tensor)

        ce_loss = F.kl_div(log(F.relu(pred)), p, reduction='batchmean')
        # loss = loss_decoder+  0.01 * kl_loss  + 0.01 * re_loss+ 0.1*fadj_re_loss + 0.01 * ce_loss
        # loss = loss_decoder+  100 * kl_loss  + 10 * re_loss+ 1*fadj_re_loss + 1 * ce_loss
        # loss = loss_decoder+r  100 * kl_loss  + 100 * re_loss+ 1*fadj_re_loss + 0.1 * ce_loss
        # loss = loss_decoder+  1000 * kl_loss  + 100 * re_loss+ 1*fadj_re_loss + 0.01 * ce_loss
        # loss = 1 * loss_decoder + loss_decoder2 + re_loss + fadj_re_loss
        # loss = 1 * loss_decoder +  1000 * kl_loss  + 1000 * re_loss + 0.001 * ce_loss  + loss_decoder2 + 0.0005 * fadj_re_loss
        loss = 1 * loss_decoder +  1000 * kl_loss + 1000*re_loss  + 0.01 * ce_loss  + loss_decoder2+ 1 * fadj_re_loss
        # loss /= 10
        loss += l2_reg_loss(model, scale=1e-2)
        loss.backward()
        optimizer.step()

    # ts = manifold.TSNE(n_components=2, random_state=0)
    # km = KMeans(n_clusters=K).fit(pred.detach().cpu().numpy())
    # res2 = km.predict(pred.detach().cpu().numpy())
    # hidemb = pred
    # z = ts.fit_transform(hidemb.detach().cpu().numpy())
    # # C_model = KMeans(n_clusters=clusters, verbose=0, max_iter=1000, tol=0.001, n_init=20, init='k-means++')
    # plt.figure(figsize=(8, 8))
    # plt.scatter(z[:, 0], z[:, 1], c=res2, cmap='tab10', s=20, alpha=0.7)  # 不同类别不同颜色
    # plt.title("k-means")
    # plt.show()
    final_nmi = (np.array(NMIS[-10:])).sum() /10
    max_nmi = (np.array(NMIS)).max()
    final_f1score = (np.array(F1[-10:])).sum() / 10
    max_f1score = (np.array(F1)).max()
    final_Acc = (np.array(ACC[-10:])).sum() / 10
    max_Acc = (np.array(ACC)).max()
    final_ari = (np.array(ARI[-10:])).sum() / 10
    max_ari = (np.array(ARI)).max()
    final_con = (np.array(Con[-10:])).sum() / 10
    min_con = (np.array(Con)).min()
    final_q = (np.array(Qs[-10:])).sum() / 10
    max_q = (np.array(Qs)).max()
    print("final_acc : {:.3f} . max_acc : {:.3f}".format(final_Acc, max_Acc))
    print("---------------------------------------------------------------------------------")
    print("final_nmi : {:.3f} . max_nmi : {:.3f}".format(final_nmi , max_nmi))
    print("---------------------------------------------------------------------------------")
    print("final_ariscore : {:.3f} . max_ariscore : {:.3f}".format(final_ari, max_ari))
    print("---------------------------------------------------------------------------------")
    print("final_f1score : {:.3f} . max_f1score : {:.3f}".format(final_f1score , max_f1score))
    print("---------------------------------------------------------------------------------")
    print("final_con : {:.3f} . max_con : {:.3f}".format(final_con, min_con))
    print("---------------------------------------------------------------------------------")
    print("final_q : {:.3f} . max_q : {:.3f}".format(final_q, max_q))
    print("---------------------------------------------------------------------------------")
