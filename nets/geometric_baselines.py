import torch
from torch_geometric.utils import remove_self_loops, softmax
from torch_sparse import sum as sparsesum
from torch_sparse import mul
from torch_sparse import SparseTensor, fill_diag, mul


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=0)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    elif norm == "softmax":
        logsumexp = sparselogsumexp(adj, dim=1)
        return logsumexp
    elif norm is None or norm=="0":
        if adj.has_value():  # adj stores values already
            return adj
        else:
            return adj.set_value(torch.ones(adj.nnz(), device=adj.device()))
    else:
        raise ValueError(f"{norm} normalization is not supported")


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=0, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops == 1:
            adj_t = fill_diag(adj_t, fill_value)
        deg = adj_t.sum(dim=1)
        deg = deg.clamp(min=1e-12)
        deg_inv_sqrt = deg.pow_(-0.5)
        if torch.isnan(deg_inv_sqrt).any():
            raise RuntimeError("NaN detected in deg_inv_sqrt — stopping training to prevent corrupt gradients.")
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)
    eps = 1e-12

    row_sum = torch.clamp(row_sum, min=eps)  # preventing: if sum=0 or neg, got inf or nan.
    out = mul(adj, 1 / row_sum.view(-1, 1))

    # Debugging: sanity check
    values = out.storage.value()
    if torch.isnan(values).any():
        raise RuntimeError("NaN detected in out of row_norm — stopping training.")
    if torch.isinf(values).any():
        raise RuntimeError("Inf detected in out of row_norm — stopping training.")

    return out


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg = in_deg.clamp(min=1e-12)
    in_deg_inv_sqrt = in_deg.pow(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)
    if torch.isnan(in_deg_inv_sqrt).any():
        raise RuntimeError("NaN detected in in_deg_inv_sqrt — stopping training to prevent corrupt gradients.")

    out_deg = sparsesum(adj, dim=1)
    out_deg = out_deg.clamp(min=1e-12)
    out_deg_inv_sqrt = out_deg.pow(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)
    if torch.isnan(out_deg_inv_sqrt).any():
        raise RuntimeError("NaN detected in in_deg_inv_sqrt — stopping training to prevent corrupt gradients.")

    adj0 = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj1 = mul(adj0, in_deg_inv_sqrt.view(1, -1))

    return adj1


def sparselogsumexp(adj, dim=1):
    """Sparse row-wise logsumexp for PyG SparseTensor"""
    N = adj.sizes()[0]  # ← Use .sizes()[0] instead of adj.shape[0]

    row, col, value = adj.coo()  # ← Unpack 3 values
    if value is None:
        value = torch.ones(row.shape[0], device=adj.device())

    norm_value = softmax(value, row, num_nodes=N)

    # Return SparseTensor with normalized values
    return adj.set_value_(norm_value, layout='coo')