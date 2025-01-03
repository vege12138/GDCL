from __future__ import print_function, division
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from dataprocess import to_sparse_tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
# from utils import load_data, load_graph

# from evaluation import eva
from collections import Counter
from datetime import datetime
import time
cuda = torch.cuda.is_available()
import scipy.io as scio
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
# from get_net_par_num import num_net_parameter

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def sparse_or_dense_dropout(x, p=0.5, training=True):
    if isinstance(x, (torch.sparse.FloatTensor)):
        new_values = F.dropout(x.values(), p=p, training=training)
        return torch.sparse.FloatTensor(x.indices(), new_values, x.size())
    else:
        return F.dropout(x, p=p, training=training)
class GNNLayer(Module):
    def __init__(self, in_features, out_features,dropout):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        if self.dropout != 0:
            features = sparse_or_dense_dropout(features, p=self.dropout, training=self.training)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.leaky_relu(output, negative_slope=0.2)
        return output


class BernoulliDecoder(nn.Module):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        """Base class for Bernoulli decoder.

        Args:
            num_nodes: Number of nodes in a graph.
            num_edges: Number of edges in a graph.
            balance_loss: Whether to balance contribution from edges and non-edges.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_possible_edges = num_nodes**2 - num_nodes
        self.num_nonedges = self.num_possible_edges - self.num_edges
        self.balance_loss = balance_loss

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        raise NotImplementedError

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        raise NotImplementedError

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute loss for given edges and non-edges."""
        raise NotImplementedError

    def loss_full(self, emb, adj):
        """Compute loss for all edges and non-edges."""
        raise NotImplementedError


class BerpoDecoder(BernoulliDecoder):
    def __init__(self, num_nodes, num_edges, balance_loss=False):
        super().__init__(num_nodes, num_edges, balance_loss)
        edge_proba = num_edges / (num_nodes**2 - num_nodes)
        self.eps = -np.log(1 - edge_proba)

    def forward_batch(self, emb, idx):
        """Compute probabilities of given edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)
            idx: edge indices, shape (batch_size, 2)

        Returns:
            edge_probs: Bernoulli distribution for given edges, shape (batch_size)
        """
        e1, e2 = idx.t()
        logits = torch.sum(emb[e1] * emb[e2], dim=1)
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def forward_full(self, emb):
        """Compute probabilities for all edges.

        Args:
            emb: embedding matrix, shape (num_nodes, emb_dim)

        Returns:
            edge_probs: Bernoulli distribution for all edges, shape (num_nodes, num_nodes)
        """
        logits = emb @ emb.t()
        logits += self.eps
        probs = 1 - torch.exp(-logits)
        return td.Bernoulli(probs=probs)

    def loss_batch(self, emb, ones_idx, zeros_idx):
        """Compute BerPo loss for a batch of edges and non-edges."""
        # Loss for edges
        e1, e2 = ones_idx[:, 0], ones_idx[:, 1]
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Loss for non-edges
        ne1, ne2 = zeros_idx[:, 0], zeros_idx[:, 1]
        loss_nonedges = torch.mean(torch.sum(emb[ne1] * emb[ne2], dim=1))
        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges + neg_scale * loss_nonedges) / (1 + neg_scale)

    def loss_full(self, emb, adj):
        """Compute BerPo loss for all edges & non-edges in a graph."""
        e1, e2 = adj.nonzero()
        edge_dots = torch.sum(emb[e1] * emb[e2], dim=1)
        loss_edges = -torch.sum(torch.log(-torch.expm1(-self.eps - edge_dots)))

        # Correct for overcounting F_u * F_v for edges and nodes with themselves
        self_dots_sum = torch.sum(emb * emb)
        correction = self_dots_sum + torch.sum(edge_dots)
        sum_emb = torch.sum(emb, dim=0, keepdim=True).t()
        loss_nonedges = torch.sum(emb @ sum_emb) - correction

        if self.balance_loss:
            neg_scale = 1.0
        else:
            neg_scale = self.num_nonedges / self.num_edges
        return (loss_edges / self.num_edges + neg_scale * loss_nonedges / self.num_nonedges) / (1 + neg_scale)


from torch.nn import Linear
import torch.nn.functional as F
class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        # extracted feature by AE
        self.z_layer = Linear(n_enc_3, n_z)
        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.elu(self.enc_1(x))
        enc_z3 = F.elu(self.enc_2(enc_z2))
        enc_z4 = F.elu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)
        dec_z2 = F.elu(self.dec_1(z))
        dec_z3 = F.elu(self.dec_2(dec_z2))
        dec_z4 = F.elu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.wl(mlp_in)), dim=1)

        return weight_output

class MLP(nn.Module):
    def __init__(self , in_fea , out_fea):
        super(MLP, self).__init__()
        self.mlp= nn.Linear(in_fea , out_fea)

    def forward(self , x):
        out = F.softmax(F.leaky_relu(self.mlp(x)) ,dim=1)
        return out

class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.tanh(self.w3(mlp_in)), dim=1)

        return weight_output
class fea(nn.Module):
    def __init__(self , x_input , x_enc1 , x_enc2 , x_enc3 , z_input ,dropout):
        super(fea, self).__init__()
        self.agcn1_fea = GNNLayer(x_input, x_enc1, dropout)
        self.agcn2_fea = GNNLayer(x_enc1, x_enc2, dropout)
        self.agcn3_fea = GNNLayer(x_enc2, x_enc3, dropout)
        self.agcn4_fea = GNNLayer(x_enc3, z_input, dropout)
        # self.agcn1_fea = nn.GraphConv(x_input, x_enc1)
        # self.agcn2_fea = nn.GraphConv(x_enc1, x_enc2)
        # self.agcn3_fea = nn.GraphConv(x_enc2, x_enc3)
        # self.agcn4_fea = nn.GraphConv(x_enc3, z_input)
        hidden_dims = [x_enc1 , x_enc2 , x_enc3 , z_input]
        if True:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x,fadj):

        fea_h1 = self.agcn1_fea(x,fadj)
        fea_h1 = F.elu(fea_h1)
        fea_h1 = self.batch_norm[0](fea_h1)

        fea_h2 = self.agcn2_fea(fea_h1,fadj)
        fea_h2 = F.elu(fea_h2)
        fea_h2 = self.batch_norm[1](fea_h2)

        fea_h3 = self.agcn3_fea(fea_h2,fadj)
        fea_h3 = F.elu(fea_h3)
        fea_h3 = self.batch_norm[2](fea_h3)
        fea_h4 = self.agcn4_fea(fea_h3,fadj)
        fea_h4 = F.elu(fea_h4)
        fea_h4 = self.batch_norm[3](fea_h4)
        return fea_h1,fea_h2,fea_h3,fea_h4


class CGCL1(nn.Module):
    def __init__(self, x_input, x_enc1, x_enc2, x_enc3, z_input, dropout):
        super(fea, self).__init__()
        self.fea_gcn = fea(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, dropout)
        self.topo_gcn = topo(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, dropout)
        self.agcn1_fea = GNNLayer(x_input, x_enc1, dropout)
        self.agcn2_fea = GNNLayer(x_enc1, x_enc2, dropout)
        self.agcn3_fea = GNNLayer(x_enc2, x_enc3, dropout)
        self.agcn4_fea = GNNLayer(x_enc3, z_input, dropout)
        hidden_dims = [x_enc1, x_enc2, x_enc3, z_input]
        if True:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x, fadj):

        fea_h1 = self.agcn1_fea(x, fadj)
        fea_h1 = F.relu(fea_h1)
        fea_h1 = self.batch_norm[0](fea_h1)

        fea_h2 = self.agcn2_fea(fea_h1, fadj)
        fea_h2 = F.relu(fea_h2)
        fea_h2 = self.batch_norm[1](fea_h2)

        fea_h3 = self.agcn3_fea(fea_h2, fadj)
        fea_h3 = F.relu(fea_h3)
        fea_h3 = self.batch_norm[2](fea_h3)
        fea_h4 = self.agcn4_fea(fea_h3, fadj)
        fea_h4 = F.relu(fea_h4)
        fea_h4 = self.batch_norm[3](fea_h4)
        return fea_h1, fea_h2, fea_h3, fea_h4
class topo(nn.Module):
    def __init__(self , x_input , x_enc1 , x_enc2 , x_enc3 , z_input ,dropout):
        super(topo, self).__init__()
        self.agcn1_fea = GNNLayer(x_input, x_enc1, dropout)
        self.agcn2_fea = GNNLayer(x_enc1, x_enc2, dropout)
        self.agcn3_fea = GNNLayer(x_enc2, x_enc3, dropout)
        self.agcn4_fea = GNNLayer(x_enc3, z_input, dropout)
        hidden_dims = [x_enc1 , x_enc2 , x_enc3 , z_input]
        if True:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None

    def forward(self, x,fadj):
        fea_h1 = self.agcn1_fea(x,fadj)
        fea_h1 = F.elu(fea_h1)
        fea_h1 = self.batch_norm[0](fea_h1)

        fea_h2 = self.agcn2_fea(fea_h1,fadj)
        fea_h2 = F.elu(fea_h2)
        fea_h2 = self.batch_norm[1](fea_h2)

        fea_h3 = self.agcn3_fea(fea_h2,fadj)
        fea_h3 = F.elu(fea_h3)
        fea_h3 = self.batch_norm[2](fea_h3)
        fea_h4 = self.agcn4_fea(fea_h3,fadj)
        fea_h4 = F.elu(fea_h4)
        fea_h4 = self.batch_norm[3](fea_h4)
        return fea_h1,fea_h2,fea_h3,fea_h4
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
class NOCD_DL(nn.Module):
    @staticmethod
    def normalize_adj(adj: sp.csr_matrix):
        """Normalize adjacency matrix and convert it to a sparse tensor."""
        if sp.isspmatrix(adj):
            adj = adj.tolil()
            adj.setdiag(1)
            adj = adj.tocsr()
            deg = np.ravel(adj.sum(1))
            deg_sqrt_inv = 1 / np.sqrt(deg)
            adj_norm = adj.multiply(deg_sqrt_inv[:, None]).multiply(deg_sqrt_inv[None, :])
        elif torch.is_tensor(adj):
            deg = adj.sum(1)
            deg_sqrt_inv = 1 / torch.sqrt(deg)
            adj_norm = adj * deg_sqrt_inv[:, None] * deg_sqrt_inv[None, :]
            return adj_norm
        return to_sparse_tensor(adj_norm)

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.named_parameters() if 'bias' not in n]
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.named_parameters() if 'bias' in n]
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v,dropout,batch_norm , name):
        super(NOCD_DL, self).__init__()
        self.n_cluster = n_clusters
        self.dropout = dropout
        self.n_z = n_z
        # AE
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.ae.load_state_dict(torch.load('./data/pre_model/{}_{}.pkl'.format(name , n_z), map_location='cpu'))
        self.fea_gcn = fea(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, dropout)
        self.topo_gcn = topo(n_input, n_enc_1, n_enc_2, n_enc_3, n_z, dropout)
        # self.agcn_0 = GNNLayer(n_input, n_enc_1,dropout)
        # self.agcn_1 = GNNLayer(n_enc_1, n_enc_2,dropout)
        # self.agcn_2 = GNNLayer(n_enc_2, n_enc_3,dropout)
        # self.agcn_3 = GNNLayer(n_enc_3, n_z,dropout)
        self.agcn_z = GNNLayer(3000 + 1 * n_z, n_clusters,dropout)

        self.mlp = MLP_L(3000+n_z)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(3 * n_enc_1)
        self.mlp2 = MLP_2(3 * n_enc_2)
        self.mlp3 = MLP_3(3 * n_enc_3)
        # self.mlp_com = MLP(3000 + n_z)
        self.mlp_ae = MLP(3000 + n_z, n_clusters)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        hidden_dims = [ n_enc_1, n_enc_2, n_enc_3, n_z, n_clusters]
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(dim, affine=False, track_running_stats=False) for dim in hidden_dims
            ]
        else:
            self.batch_norm = None
        # degree
        self.v = v
        self.attention = Attention(n_z)
        self.MLP = nn.Sequential(
            nn.Linear(n_enc_3, n_z),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, adj):
        # AE Module
        x_bar, h1, h2, h3, z = self.ae(x)
        # fea_h1, fea_h2, fea_h3, fea_h4 = self.fea_gcn(x, fadj)
        topo_h1, topo_h2, topo_h3, topo_h4 = self.topo_gcn(x, adj)


        # AGCN-S
        # coms = torch.cat((fea_h1, fea_h2, fea_h3, fea_h4), 1)
        # u = self.mlp(coms)
        # u = F.normalize(u, p=2)
        # u1 = u[:, 0].reshape(-1, 1).repeat(1, 500)
        # u2 = u[:, 1].reshape(-1, 1).repeat(1, 500)
        # u3 = u[:, 2].reshape(-1, 1).repeat(1, 2000)
        # u4 = u[:, 3].reshape(-1, 1).repeat(1, self.n_z)
        # # u5 = u[:, 4].reshape(-1, 1).repeat(1, self.n_clusters)
        # z_out_fea = torch.cat((u1.mul(fea_h1), u2.mul(fea_h2), u3.mul(fea_h3), u4.mul(fea_h4)), 1)
        # z_out_fea = self.agcn_z(z_out_fea, fadj, active=False)
        # z_out_fea = F.elu(z_out_fea)
        # if self.batch_norm is not None:
        #     z_out_fea = self.batch_norm[3](z_out_fea)

        # AGCN-S
        coms = torch.cat((topo_h1, topo_h2, topo_h3, topo_h4), 1)
        u = self.mlp(coms)
        u = F.normalize(u, p=2)
        u1 = u[:, 0].reshape(-1, 1).repeat(1, 500)
        u2 = u[:, 1].reshape(-1, 1).repeat(1, 500)
        u3 = u[:, 2].reshape(-1, 1).repeat(1, 2000)
        u4 = u[:, 3].reshape(-1, 1).repeat(1, self.n_z)
        # u5 = u[:, 4].reshape(-1, 1).repeat(1, self.n_clusters)
        z_out_topo = torch.cat((u1.mul(topo_h1), u2.mul(topo_h2), u3.mul(topo_h3), u4.mul(topo_h4)), 1)
        z_out_topo = self.agcn_z(z_out_topo, adj, active=False)
        z_out_topo = F.elu(z_out_topo)
        if self.batch_norm is not None:
            z_out_topo = self.batch_norm[3](z_out_topo)

        coms_emb = torch.stack([z, z_out_topo], 1)

        emb, att = self.attention(coms_emb)

        if self.batch_norm is not None:
            emb = self.batch_norm[4](emb)

        pred = F.softmax(emb, dim=1)



        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1)- self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # fadj_recon = self.dot_product_decode(fea_h4)


        return x_bar ,  z , q , emb , pred , z_out_topo


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
