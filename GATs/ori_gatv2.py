# @Time     : 2022/11/10
# @Author   : Haldate
import argparse

import torch
import torch.nn.functional as F

from datasets import get_planetoid_dataset, get_amazon_dataset
from train_eval import random_planetoid_splits, run

from torch_geometric.nn import GATv2Conv
from logger import logger
from load_data import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
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

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# if args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
#     dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
# elif args.dataset in ['Computers', 'Photo']:
#     dataset = get_amazon_dataset(args.dataset, args.normalize_features)

dataset = load_data(args)

permute_masks = random_planetoid_splits if args.random_splits else None
loss, acc, duration = run(args, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
                          args.early_stopping, permute_masks)
logger(model_name='ori_gatv2', loss=loss, acc=acc, duration=duration, args=args)
