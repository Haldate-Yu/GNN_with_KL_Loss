# @Time     : 2022/11/3
# @Author   : Haldate
from typing import Union, Tuple, Optional, Any

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_scipy_sparse_matrix


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.topk_alpha = None
        self.noise_alpha = None
        self.edge_index = None
        self.num_nodes = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, topk: int,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        self.num_nodes = x.size(0)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # Wh1 & Wh2
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        self.edge_index = edge_index

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out, noise = self.propagate(edge_index, x=x, alpha=alpha, size=size, topk=topk)

        alpha = self._alpha

        assert alpha is not None
        self._alpha = None
        assert edge_index is not None
        self.edge_index = None

        self.num_nodes = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            noise = noise.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            noise = noise.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, noise, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, noise, edge_index.set_value(alpha, layout='coo')
        else:
            return out, noise

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            out = self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            for hook in self._message_forward_pre_hooks.values():
                res = hook(self, (msg_kwargs,))
                if res is not None:
                    msg_kwargs = res[0] if isinstance(res, tuple) else res
            # message step
            topk_out, noise_out = self.message(**msg_kwargs)
            for hook in self._message_forward_hooks.values():
                topk_res = hook(self, (msg_kwargs,), topk_out)
                noise_res = hook(self, (msg_kwargs,), noise_out)
                if topk_res is not None:
                    topk_out = topk_res
                if noise_res is not None:
                    noise_out = noise_res

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            for hook in self._aggregate_forward_pre_hooks.values():
                res = hook(self, (aggr_kwargs,))
                if res is not None:
                    aggr_kwargs = res[0] if isinstance(res, tuple) else res
            # aggregate step
            topk_out = self.aggregate(topk_out, **aggr_kwargs)
            noise_out = self.aggregate(noise_out, **aggr_kwargs)
            for hook in self._aggregate_forward_hooks.values():
                topk_res = hook(self, (msg_kwargs,), topk_out)
                noise_res = hook(self, (msg_kwargs,), noise_out)
                if topk_res is not None:
                    topk_out = topk_res
                if noise_res is not None:
                    noise_out = noise_res

            update_kwargs = self.inspector.distribute('update', coll_dict)
            # update step
            topk_out = self.update(topk_out, **update_kwargs)
            noise_out = self.update(noise_out, **update_kwargs)

        for hook in self._propagate_forward_hooks.values():
            topk_res = hook(self, (msg_kwargs,), topk_out)
            noise_res = hook(self, (msg_kwargs,), noise_out)
            if topk_res is not None:
                topk_out = topk_res
            if noise_res is not None:
                noise_out = noise_res

        return topk_out, noise_out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], topk: int):
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # final alpha
        self._alpha = alpha  # Save for later use.

        # split it into topk alpha & noise alpha, return x_j * topk alpha and noise alpha part
        perm = self.split_attn(alpha, topk)
        # perm = self.split_attn_bottom(alpha, topk)
        topk_alpha = alpha * perm
        noise_alpha = alpha * ~perm
        # save for after use
        self.noise_alpha = noise_alpha

        topk_alpha = F.dropout(topk_alpha, p=self.dropout, training=self.training)
        # noise_alpha = F.dropout(noise_alpha, p=self.dropout, training=self.training)

        return x_j * topk_alpha.unsqueeze(-1), x_j * noise_alpha.unsqueeze(-1)

    # with a topk value as a hyper-parameter, default = 5
    def split_attn(self, alpha, topk=5):
        # with no grad
        attn = alpha.detach()
        # for multi-head
        masks = []

        for i in range(attn.size(1)):
            # adj = to_scipy_sparse_matrix(edge_index, attn[:, i])
            # adjs.append(adj)

            # convert to tensor adj
            adj = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=attn[:, i],
                               sparse_sizes=(self.num_nodes, self.num_nodes)).to_dense()

            # select topk by cols
            values, idx = adj.topk(topk, 1)
            topk_adj = torch.zeros_like(adj).scatter_(1, idx, values)

            # get new edge_weights
            row, col = self.edge_index
            edge_weights = topk_adj[row, col]
            mask = edge_weights == attn[:, i]
            masks.append(mask)
        # transpose and stack in cols
        attn_mask = torch.stack(masks, dim=1)
        return attn_mask

    # with a bottomk values as a hyper-parameter, default=1
    def split_attn_bottom(self, alpha, bottomk=1):
        # with no grad
        attn = alpha.detach()
        # for multi head
        masks = []

        for i in range(attn.size(1)):
            # convert to tensor adj
            adj = SparseTensor(row=self.edge_index[0], col=self.edge_index[1], value=attn[:, i],
                               sparse_sizes=(self.num_nodes, self.num_nodes)).to_dense()
            zero_values = -9e15
            reverse_adj = torch.where(-adj == 0, zero_values, -adj)

            # select bottomk by cols
            values, idx = reverse_adj.topk(bottomk, 1)
            bottomk_adj = torch.zeros_like(reverse_adj).scatter_(1, idx, values)

            # get new edge_weights
            row, col = self.edge_index
            edge_weights = bottomk_adj[row, col]
            mask = edge_weights == attn[:, i]
            masks.append(mask)
        # transpose and stack in cols
        attn_mask = torch.stack(masks, dim=1)
        return ~attn_mask

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
