# import numpy as np
# import h5py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from torch.utils.data import DataLoader
# from torch.optim import Adam, SGD
# from torch.nn import Linear
# from torch.utils.data import Dataset
# from sklearn.cluster import KMeans
# from evaluation import eva
# import os
# os.environ["OMP_NUM_THREADS"] = "2"
#
# #torch.cuda.set_device(3)
#
#
# class AE(nn.Module):
#
#     def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
#                  n_input, n_z):
#         super(AE, self).__init__()
#         self.enc_1 = Linear(n_input, n_enc_1)
#         self.enc_2 = Linear(n_enc_1, n_enc_2)
#         self.enc_3 = Linear(n_enc_2, n_enc_3)
#         self.z_layer = Linear(n_enc_3, n_z)
#
#         self.dec_1 = Linear(n_z, n_dec_1)
#         self.dec_2 = Linear(n_dec_1, n_dec_2)
#         self.dec_3 = Linear(n_dec_2, n_dec_3)
#         self.x_bar_layer = Linear(n_dec_3, n_input)
#
#     def forward(self, x):
#         enc_h1 = F.relu(self.enc_1(x))
#         enc_h2 = F.relu(self.enc_2(enc_h1))
#         enc_h3 = F.relu(self.enc_3(enc_h2))
#         z = self.z_layer(enc_h3)
#
#         dec_h1 = F.relu(self.dec_1(z))
#         dec_h2 = F.relu(self.dec_2(dec_h1))
#         dec_h3 = F.relu(self.dec_3(dec_h2))
#         x_bar = self.x_bar_layer(dec_h3)
#
#         return x_bar, z
#
#
# class LoadDataset(Dataset):
#     def __init__(self, data):
#         self.x = data
#
#     def __len__(self):
#         return self.x.shape[0]
#
#     def __getitem__(self, idx):
#         return torch.from_numpy(np.array(self.x[idx])).float(), \
#                torch.from_numpy(np.array(idx))
#
#
# def adjust_learning_rate(optimizer, epoch):
#     lr = 0.001 * (0.1 ** (epoch // 20))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def pretrain_ae(model, dataset, y):
#     train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
#     print(model)
#     optimizer = Adam(model.parameters(), lr=1e-3)
#     for epoch in range(300):
#         # adjust_learning_rate(optimizer, epoch)
#         for batch_idx, (x, _) in enumerate(train_loader):
#             x = x.cuda()
#
#             x_bar, _ = model(x)
#             loss = F.mse_loss(x_bar, x)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         with torch.no_grad():
#             x = torch.Tensor(dataset.x).cuda().float()
#             x_bar, z = model(x)
#             loss = F.mse_loss(x_bar, x)
#             print('{} loss: {}'.format(epoch, loss))
#             kmeans = KMeans(n_clusters=K, n_init=20).fit(z.data.cpu().numpy())
#             eva(y, kmeans.labels_, epoch)
#
#         torch.save(model.state_dict(), 'D:/zqwu/FGCDv1/data/pre_model/{}_{}.pkl'.format(dataName,K))
#
# dataName = "citeseer"
# label_file = './data/{}/{}_label.txt'.format(dataName, dataName)
# feature_path = './data/{}/{}_fea.txt'.format(dataName, dataName)
# x = np. loadtxt(feature_path , dtype=float)
# # x = np.loadtxt('dblp.txt', dtype=float)
# # y = np.genfromtxt(label_file, dtype=np.int32)[: , 1]
# y = np.genfromtxt(label_file, dtype=np.int32)
# K = y.max()
# y_true = y-1
# feat_dim = x.shape[1]
# model = AE(
#         n_enc_1=500,
#         n_enc_2=500,
#         n_enc_3=2000,
#         n_dec_1=2000,
#         n_dec_2=500,
#         n_dec_3=500,
#         n_input=feat_dim,
#         n_z=K,).cuda()
#
#
# # y = np.loadtxt('dblp_label.txt', dtype=int)
#
# dataset = LoadDataset(x)
# pretrain_ae(model, dataset, y_true)


import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
# import os
# os.environ["OMP_NUM_THREADS"] = "2"


#torch.cuda.set_device(3)


# class AE(nn.Module):
#
#     def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
#                  n_input, n_z):
#         super(AE, self).__init__()
#         self.enc_1 = Linear(n_input, n_enc_1)
#         self.enc_2 = Linear(n_enc_1, n_enc_2)
#         self.enc_3 = Linear(n_enc_2, n_enc_3)
#         self.z_layer = Linear(n_enc_3, n_z)
#
#         self.dec_1 = Linear(n_z, n_dec_1)
#         self.dec_2 = Linear(n_dec_1, n_dec_2)
#         self.dec_3 = Linear(n_dec_2, n_dec_3)
#         self.x_bar_layer = Linear(n_dec_3, n_input)
#
#     def forward(self, x):
#         enc_h1 = F.relu(self.enc_1(x))
#         enc_h2 = F.relu(self.enc_2(enc_h1))
#         enc_h3 = F.relu(self.enc_3(enc_h2))
#         z = self.z_layer(enc_h3)
#
#         dec_h1 = F.relu(self.dec_1(z))
#         dec_h2 = F.relu(self.dec_2(dec_h1))
#         dec_h3 = F.relu(self.dec_3(dec_h2))
#         x_bar = self.x_bar_layer(dec_h3)
#
#         return x_bar, z

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_cluster):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)

        # extracted feature by AE
        self.z_layer1 = Linear(n_enc_3, n_z)
        self.z_layer2 = Linear(n_z, n_cluster)
        # decoder
        self.dec_0 = Linear(n_cluster, n_z)
        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z1 = self.z_layer1(enc_z4)
        z2 = self.z_layer2(z1)
        dec_z1 = F.relu(self.dec_0(z2))
        dec_z2 = F.relu(self.dec_1(dec_z1))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar,  z2
class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=K, n_init=20).fit(z.data.cpu().numpy())
            # eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), './data/pre_model/{}_K_{}.pkl'.format(dataName,K))

dataName = "amac"
label_file = './data/{}/{}_label.txt'.format(dataName, dataName)
feature_path = './data/{}/{}_fea.txt'.format(dataName, dataName)
x = np. loadtxt(feature_path , dtype=float)
# x = np.loadtxt('dblp.txt', dtype=float)
# y = np.genfromtxt(label_file, dtype=np.int32)[: , 1]
y = np.genfromtxt(label_file, dtype=np.int32)
#y_true = y-1
y_true = y
# y = np.genfromtxt(label_file, dtype=np.int32)
# K = y.max()+1
K = len(np.unique(y))
feat_dim = x.shape[1]
model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,

        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=feat_dim,
        n_z=K,
n_cluster = K).cuda()


# y = np.loadtxt('dblp_label.txt', dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y_true)
