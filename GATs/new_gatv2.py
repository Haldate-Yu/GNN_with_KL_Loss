# @Time     : 2022/11/10
# @Author   : Haldate
import argparse

import torch
from torch.nn import Linear
import torch.nn.functional as F
from load_data import load_data
from logger import logger
from train_eval import random_planetoid_splits, run

from new_gatv2_layer import GATv2Conv

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--splits', type=str, default="geom-gcn")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--hidden_mlp', type=int, default=16, help='hidden size for MLP in layer 1')
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--share_weights', type=bool, default=False)

parser.add_argument('--kl_loss', type=bool, default=False, help='use kl loss or not')
parser.add_argument('--kl_alpha1', type=float, default=0.01, help='percentage of kl loss for layer 1')
parser.add_argument('--kl_alpha2', type=float, default=0.01, help='percentage of kl loss for layer 2')
parser.add_argument('--attn_topk', type=int, default=2, help='topk value to select attn')
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GATv2Conv(dataset.num_features, args.hidden,
                               heads=args.heads, dropout=args.dropout, share_weights=args.share_weights)
        self.conv2 = GATv2Conv(args.hidden * args.heads, dataset.num_classes,
                               heads=args.output_heads, concat=False,
                               dropout=args.dropout, share_weights=args.share_weights)

        self.lin1 = Linear(args.heads * args.hidden, args.hidden_mlp)
        self.lin2 = Linear(args.hidden_mlp, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x, noise1 = self.conv1(x, edge_index, args.attn_topk)
        x = F.elu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)
        ### MLP for noise1 ###
        noise1 = self.lin1(noise1)
        noise1 = F.elu(noise1)
        noise1 = F.dropout(noise1, p=args.dropout, training=self.training)
        noise1 = self.lin2(noise1)

        x, noise2 = self.conv2(x, edge_index, args.attn_topk)

        return F.log_softmax(x, dim=1), F.log_softmax(noise1, dim=1), F.log_softmax(noise2, dim=1)


# if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
#     dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# elif args.dataset in ['Computers', 'Photo']:
#     dataset = get_amazon_dataset(args.dataset, args.normalize_features)

dataset = load_data(args)

permute_masks = random_planetoid_splits if args.random_splits else None
loss, acc, duration = run(args, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
                          args.early_stopping, permute_masks,
                          kl_loss=args.kl_loss, kl_alpha1=args.kl_alpha1, kl_alpha2=args.kl_alpha2)
logger(model_name='new_gatv2', loss=loss, acc=acc, duration=duration, args=args)
