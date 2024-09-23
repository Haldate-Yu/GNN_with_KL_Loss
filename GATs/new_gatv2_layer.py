# @Time     : 2022/11/10
# @Author   : Haldate
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GATv2Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

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
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, topk: int,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
        self.num_nodes = x.size(0)
        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        self.edge_index = edge_index

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out, noise = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                                    size=None, topk=topk)

        alpha = self._alpha
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
            assert alpha is not None
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

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], topk: int):
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        # final alpha
        self._alpha = alpha  # Save for later use.
        # split it into topk alpha & noise alpha, return x_j * topk alpha and noise alpha part
        perm = self.split_attn(alpha, topk)
        topk_alpha = alpha * perm
        noise_alpha = alpha * ~perm
        # save for after use
        self.noise_alpha = noise_alpha

        topk_alpha = F.dropout(topk_alpha, p=self.dropout, training=self.training)
        # noise_alpha = F.dropout(noise_alpha, p=self.dropout, training=self.training)

        return x_j * topk_alpha.unsqueeze(-1), x_j * noise_alpha.unsqueeze(-1)

    # with a topk value as a hyper-parameter
    def split_attn(self, alpha, topk):
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

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
