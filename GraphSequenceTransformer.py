from abc import ABCMeta
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor,  fill_diag, mul
from torch_geometric.nn.inits import zeros

from nets.geometric_baselines import get_norm_adj
from util import spanning_tree_diameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


class GraphSequenceTransformer(nn.Module):
    def __init__(self, nfeat, nclass, args):
        super().__init__()
        self._cached_adj_t = None
        hid_dim = args.hid_dim        # your hidden dimension
        dropout = args.dropout
        nhead = args.heads
        num_layers = args.layer

        self.prep = PrepSequence(nfeat, hid_dim, args)
        self.input_lin = nn.Linear(hid_dim, hid_dim)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nhead,
            dim_feedforward=4 * hid_dim,
            dropout=dropout,
            batch_first=False,      # or True if you prefer
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(hid_dim, nclass)

    def forward(self, x, edge_index):
        x_seq_list = self.prep(x, edge_index)
        x_seq = torch.stack(x_seq_list, dim=0)

        x_seq = self.input_lin(x_seq)      # [T, N, hid_dim]
        x_enc = self.encoder(x_seq)       # [T, N, hid_dim]

        pooled = x_enc.mean(dim=0)        # [N, hid_dim] (example pooling)
        logits = self.classifier(pooled)  # [N, nclass]
        return logits


class PrepSequence(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        self._cached_adj_t = None
        norm = args.inci_norm

        self.lin = nn.Linear(in_dim, out_dim, bias=True)

        self.diam = None
        self.conv = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm = norm

    def build_layers(self, x, edges):
        self.diam = spanning_tree_diameter(x, edges)
        self.conv = nn.ModuleList(
            [MPTConvUndirected(self.in_dim, self.out_dim, norm=self.norm, cached=True).to(x.device)
             for _ in range(self.diam)]
        )

    def forward(self, x, edges):
        num_nodes = x.size(0)
        if self._cached_adj_t is None:
            self._cached_adj_t = SparseTensor.from_edge_index(edges, sparse_sizes=(num_nodes, num_nodes)).t()
        adj = self._cached_adj_t

        if self.diam is None:
            self.build_layers(x, edges)

        x = self.lin(x)
        x_sequence = [x]
        for convx in self.conv:
            x = convx(x, adj)  # if conv needs edges; otherwise convx(x)
            x_sequence.append(x)

        return x_sequence


# noinspection PyAbstractClass
class MPTConvUndirected(MessagePassing):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            add_self_loops: Optional[bool] = None,
            norm: str = 'gcn',
            bias: bool = True,
            aggr: str = 'add',  # or 'mean', 'max' as you need
            cached: bool = False,
    ):
        super().__init__(aggr=aggr)   # important for MessagePassing

        self.diam = None
        self.norm = norm

        self.add_self_loops = add_self_loops
        # self.in_dim = in_dim
        # self.out_dim = out_dim
        self.cached = cached

        self._cached_edge_index = None
        self._cached_adj_t = None

        # if bias:
        #     self.bias = Parameter(torch.empty(out_dim))
        # else:
        #     self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.norm:
            if isinstance(edge_index, SparseTensor):
                if self._cached_adj_t is None:
                    edge_index = get_norm_adj(edge_index,  self.norm)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = self._cached_adj_t
            else:
                raise NotImplementedError('Please convert edge_index to SparseTensor for efficiency.')

        out1 = self.propagate(edge_index, x=x, edge_weight=edge_weight)


        return out1

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
