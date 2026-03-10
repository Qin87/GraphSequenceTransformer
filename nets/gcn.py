"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/gcn_conv.py
"""
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_scatter import scatter_add
from torch_sparse import SparseTensor





class StandGCNXBN(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        self._cached_adj_t = None
        self.BN_model = args.BN_model
        nhid = args.hid_dim
        dropout = args.dropout
        nlayer = args.layer
        is_add_self_loops = args.First_self_loop
        norm = args.gcn_norm
        self.is_add_self_loops = is_add_self_loops  # Qin True is the original
        if nlayer == 1:
            self.conv1 = GCNConv(nfeat, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        else:
            self.conv1 = GCNConv(nfeat, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)

        self.mlp1 = torch.nn.Linear(nhid, nclass)
        self.conv2 = GCNConv(nhid, nclass, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops)
        self.convx = nn.ModuleList([GCNConv(nhid, nhid, cached=False, normalize=norm, add_self_loops=self.is_add_self_loops) for _ in range(nlayer-2)])
        self.dropout_p = dropout

        self.batch_norm1 = nn.BatchNorm1d(nhid)
        self.batch_norm2 = nn.BatchNorm1d(nclass)
        self.batch_norm3 = nn.BatchNorm1d(nhid)

        self.layer = nlayer

    def forward(self, x, adj, edge_weight=None):
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(adj, sparse_sizes=(num_nodes, num_nodes)).t()

        adj = self._cached_adj_t
        x = self.conv1(x, adj)

        if self.layer == 1:
            # x = F.dropout(x,p= self.dropout_p, training=self.training)
            # if self.BN_model:
            #     x = self.batch_norm2(x)
            return x
        x = F.relu(x)

        if self.layer>2:
            for iter_layer in self.convx:
                # x = F.dropout(x,p= self.dropout_p, training=self.training)
                x = iter_layer(x, adj, edge_weight)
                # if self.BN_model:
                #     x= self.batch_norm3(x)
                x = F.relu(x)

        # x = F.dropout(x, p= self.dropout_p, training=self.training)
        x = self.conv2(x, adj, edge_weight)
        # if self.BN_model:
        #     x = self.batch_norm2(x)
        # # x = F.dropout(x, p=self.dropout_p, training=self.training)      # this is the best dropout arrangement
        return x