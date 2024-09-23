from typing import Optional
from torch_geometric.typing import OptTensor

import math

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.typing import Adj, OptTensor, PairTensor, Size
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_sparse import SparseTensor
from torch_geometric.utils import (remove_self_loops, add_self_loops, softmax,
                                   is_undirected, negative_sampling,
                                   batched_negative_sampling, to_undirected,
                                   dropout_adj)

from torch_geometric.nn.inits import glorot, zeros


class SuperGATConv(MessagePassing):
    att_x: OptTensor
    att_y: OptTensor

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, attention_type: str = 'MX',
                 neg_sample_ratio: float = 0.5, edge_sample_ratio: float = 1.0,
                 is_undirected: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.is_undirected = is_undirected

        assert attention_type in ['MX', 'SD']
        assert 0.0 < neg_sample_ratio and 0.0 < edge_sample_ratio <= 1.0

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        if self.attention_type == 'MX':
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        else:  # self.attention_type == 'SD'
            self.register_parameter('att_l', None)
            self.register_parameter('att_r', None)

        self.att_x = self.att_y = None  # x/y for self-supervision

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
        self.lin.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, topk: int,
                neg_edge_index: OptTensor = None,
                batch: OptTensor = None):
        r"""
        Args:
            neg_edge_index (Tensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        N, H, C = x.size(0), self.heads, self.out_channels

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        self.edge_index = edge_index

        x = self.lin(x).view(-1, H, C)
        self.num_nodes = x.size(0)
        # propagate_type: (x: Tensor)
        out, noise = self.propagate(edge_index, x=x, size=None, topk=topk)

        if self.training:
            pos_edge_index = self.positive_sampling(edge_index)

            pos_att = self.get_attention(
                edge_index_i=pos_edge_index[1],
                x_i=x[pos_edge_index[1]],
                x_j=x[pos_edge_index[0]],
                num_nodes=x.size(0),
                return_logits=True,
            )

            if neg_edge_index is None:
                neg_edge_index = self.negative_sampling(edge_index, N, batch)

            neg_att = self.get_attention(
                edge_index_i=neg_edge_index[1],
                x_i=x[neg_edge_index[1]],
                x_j=x[neg_edge_index[0]],
                num_nodes=x.size(0),
                return_logits=True,
            )

            self.att_x = torch.cat([pos_att, neg_att], dim=0)
            self.att_y = self.att_x.new_zeros(self.att_x.size(0))
            self.att_y[:pos_edge_index.size(1)] = 1.

        self.edge_index = None

        self.num_nodes = None

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            noise = noise.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            noise = noise.mean(dim=1)

        if self.bias is not None:
            out += self.bias

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


    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], topk: int):
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i)
        alpha = F.leaky_relu(alpha, self.negative_slope)
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


    def negative_sampling(self, edge_index: Tensor, num_nodes: int,
                          batch: OptTensor = None) -> Tensor:
        num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio *
                              edge_index.size(1))

        if not self.is_undirected and not is_undirected(
                edge_index, num_nodes=num_nodes):
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if batch is None:
            neg_edge_index = negative_sampling(edge_index, num_nodes,
                                               num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index, batch, num_neg_samples=num_neg_samples)

        return neg_edge_index


    def positive_sampling(self, edge_index: Tensor) -> Tensor:
        pos_edge_index, _ = dropout_adj(edge_index,
                                        p=1. - self.edge_sample_ratio,
                                        training=self.training)
        return pos_edge_index


    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                      num_nodes: Optional[int],
                      return_logits: bool = False) -> Tensor:
        if self.attention_type == 'MX':
            logits = (x_i * x_j).sum(dim=-1)
            if return_logits:
                return logits

            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            alpha = alpha * logits.sigmoid()

        else:  # self.attention_type == 'SD'
            alpha = (x_i * x_j).sum(dim=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha


    def get_attention_loss(self) -> Tensor:
        r"""Compute the self-supervised graph attention loss."""
        if not self.training:
            return torch.tensor([0], device=self.lin.weight.device)

        return F.binary_cross_entropy_with_logits(
            self.att_x.mean(dim=-1),
            self.att_y,
        )


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


    def __repr__(self):
        return '{}({}, {}, heads={}, type={})'.format(self.__class__.__name__,
                                                      self.in_channels,
                                                      self.out_channels,
                                                      self.heads,
                                                      self.attention_type)
