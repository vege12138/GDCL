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
    def __init__(self, in_features, out_features, dropout):
        super(GNNLayer, self).__init__()
        self.dropout  =  dropout
        self.in_features = in_features
        self.out_features = out_features
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