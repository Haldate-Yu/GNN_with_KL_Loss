import os
import os.path as osp
import random
from typing import Callable, Optional

import torch
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.data import Data, InMemoryDataset

from torch_geometric.datasets import (Planetoid, CitationFull, CoraFull, WebKB, Coauthor, Amazon, Actor,
                                      LINKXDataset, FakeDataset, KarateClub, WikipediaNetwork)

from torch_geometric.utils import (to_undirected, add_self_loops,
                                   index_to_mask, softmax, k_hop_subgraph)

from ogb.nodeproppred import PygNodePropPredDataset
from torch_scatter import scatter


def load_data(args):
    ###################################################################################################
    #############################  Small-Scale Homophily datasets ################################
    ###################################################################################################
    if args.data in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', args.data)
        dataset = Planetoid(path, args.data, split=args.split, transform=T.NormalizeFeatures())
        data = dataset[0]

        data.num_classes = dataset.num_classes
        data.num_edges_orig = data.num_edges
        data.num_features = dataset.num_features

        if args.split == "geom-gcn":
            data.train_mask = data.train_mask[:, args.seed % 10]
            data.val_mask = data.val_mask[:, args.seed % 10]
            data.test_mask = data.test_mask[:, args.seed % 10]



    elif args.data in ["CS", "Physics"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = Coauthor(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        # transform = RandomNodeSplit(split= "test_rest",
        #                                     num_train_per_class = 20,
        #                                     num_val = 30* data.num_classes,)

        transform = RandomNodeSplit(split="train_rest",
                                    num_val=0.2,
                                    num_test=0.2)
        transform(data)
        # print_and_log(data)
        # raise Exception('pause!! ')

    elif args.data in ["Computers", "Photo"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = Amazon(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        # transform = RandomNodeSplit(split= "test_rest",
        #                                     num_train_per_class = 20,
        #                                     num_val = 30* data.num_classes,)
        transform = RandomNodeSplit(split="train_rest",
                                    num_val=0.2,
                                    num_test=0.2)
        transform(data)
        # print_and_log(data)

        # raise Exception('pause!! ')


    elif args.data in ["CoraFull"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = CitationFull(path, 'cora', transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        # transform = RandomNodeSplit(split= "test_rest",
        #                                     num_train_per_class = 20,
        #                                     num_val = 30* data.num_classes,)
        transform = RandomNodeSplit(split="train_rest",
                                    num_val=0.2,
                                    num_test=0.2)
        transform(data)
        # print_and_log(data)

        # raise Exception('pause!! ')

    elif args.data in ["DBLP"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = CitationFull(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        # transform = RandomNodeSplit(split= "test_rest",
        #                                     num_train_per_class = 20,
        #                                     num_val = 30* data.num_classes,)
        transform = RandomNodeSplit(split="train_rest",
                                    num_val=0.2,
                                    num_test=0.2)
        transform(data)
        # print_and_log(data)

        # raise Exception('pause!! ')


    ###################################################################################################
    #############################  Large-Scale Homophily datasets (OGBN) ################################
    ###################################################################################################

    elif args.data in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M']:

        print("Loading  Dataset: {}".format(args.data))

        dataset = PygNodePropPredDataset(name=args.data, root='../data', transform=T.ToUndirected())
        data = dataset[0]
        data.num_features = dataset.num_features
        split_idx = dataset.get_idx_split()
        # evaluator = Evaluator(args.data)

        edge_index = to_undirected(data.edge_index, data.num_nodes)
        # edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

        data.edge_index = edge_index
        for split in ['train', 'valid', 'test']:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f'{split}_mask'] = mask

        data.val_mask = data.valid_mask

        if args.data in ['ogbn-proteins']:
            data.y = data.y.to(torch.float)
            data.num_classes = dataset.num_tasks
            data.node_species = None
            row, col = data.edge_index
            data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
        else:
            data.num_classes = dataset.num_classes

        if args.data in ['ogbn-arxiv']:
            data.y = data.y.squeeze(1)

        print("Load Done !")
        # print_and_log(data)

    ###################################################################################################
    #############################  Small-Scale Heterophily datasets ################################
    ###################################################################################################
    elif args.data in ["Cornell", "Texas", "Wisconsin"]:

        print("Loading  Dataset: {}".format(args.data))

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = WebKB(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        # print_and_log(data)
        data.train_mask = data.train_mask[:, args.seed % 10]
        data.val_mask = data.val_mask[:, args.seed % 10]
        data.test_mask = data.test_mask[:, args.seed % 10]
        # print_and_log(data)
        # raise Exception('pause!! ')

    elif args.data in ["Actor"]:

        print("Loading  Dataset: {}".format(args.data))

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = Actor(path, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features

        # print_and_log(data)
        data.train_mask = data.train_mask[:, args.seed % 10]
        data.val_mask = data.val_mask[:, args.seed % 10]
        data.test_mask = data.test_mask[:, args.seed % 10]
        # print_and_log(data)
        # raise Exception('pause!! ')

    elif args.data in ["chameleon", "crocodile", "squirrel"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = WikipediaNetwork(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features

        data.train_mask = data.train_mask[:, args.seed % 10]
        data.val_mask = data.val_mask[:, args.seed % 10]
        data.test_mask = data.test_mask[:, args.seed % 10]
        # print_and_log(data)

        # Multi Spilt
        # raise Exception('pause!! ')

    ###################################################################################################
    #############################  Large-Scale Heterophily datasets ################################
    ###################################################################################################

    elif args.data in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)
        dataset = LINKXDataset(path, args.data, transform=T.NormalizeFeatures())
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features

        data.train_mask = data.train_mask[:, args.seed % 5]
        data.val_mask = data.val_mask[:, args.seed % 5]
        data.test_mask = data.test_mask[:, args.seed % 5]
        # print_and_log(data)

        print(data)

        # Multi Spilt
        # raise Exception('pause!! ')


    ###################################################################################################
    #############################  Long Range Datasets (Graph Level Datasets) #########################
    ###################################################################################################
    elif args.data in ["voc", ]:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data', args.data)

        dataset = VOCSuperpixels(path)
        data = dataset[0]
        # print_and_log(data)
        # print(dataset)

        print(data)
        print(dataset[1])
        raise Exception('pause!! ')

    ###################################################################################################
    #############################  Fake Datasets ################################
    ###################################################################################################

    elif args.data in ["fake_node", "fake_graph"]:

        dataset = FakeDataset(num_graph=1,
                              avg_num_nodes=2500,
                              avg_degree=10,
                              num_channels=64,
                              edge_dim=0,
                              num_classes=10,
                              task='auto',
                              is_undirected=True,
                              )  # transform=T.NormalizeFeatures()
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features

        data.train_mask = torch.ones_like(data.y)
        data.val_mask = torch.ones_like(data.y)
        data.test_mask = torch.ones_like(data.y)

        # print_and_log(data)
        # print(dataset)

        # print(data)
        # raise Exception('pause!! ')

    elif args.data in ["karate"]:
        dataset = KarateClub()
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.num_features = dataset.num_features
        print(data.train_mask)

        data.val_mask = data.train_mask
        data.test_mask = index_to_mask(torch.arange(data.y.size(0)))
        print(data.val_mask)
        print(data.test_mask)

    else:
        raise Exception('Not implement for this data: {} '.format(args.data))

    return data


class KhopDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """

    def __init__(
            self,
            args,
            org_data,
            num_channels: int = 10,
            num_classes: int = 10,
            transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__('.', transform, pre_filter, pre_transform)

        self.org_data = org_data
        self.args = args

        self.num_channels = num_channels
        self._num_classes = num_classes
        self.kwargs = kwargs

        data_list = [self.generate_data(i) for i in range(max(self.org_data.x.size(0), 1))]

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self, n_id) -> Data:

        data = Data()

        subset, edge_index, inv, edge_mask = k_hop_subgraph(node_idx=n_id,
                                                            num_hops=self.args.num_hop,
                                                            edge_index=self.org_data.edge_index,
                                                            relabel_nodes=True
                                                            )

        data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        data.x = torch.randn(subset.size(0), self.num_channels) + data.y

        data.edge_index = edge_index

        # data.y = self.org_data.y[n_id]

        data.subset = subset

        data.node_id = n_id

        # if self.num_channels > 0 and self.task == 'graph':
        #     data.x = torch.randn(num_nodes, self.num_channels) + data.y
        # elif self.num_channels > 0 and self.task == 'node':
        #     data.x = torch.randn(num_nodes,
        #                          self.num_channels) + data.y.unsqueeze(1)
        # else:
        #     data.num_nodes = num_nodes

        # if self.edge_dim > 1:
        #     data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        # elif self.edge_dim == 1:
        #     data.edge_weight = torch.rand(data.num_edges)

        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))

        return data
