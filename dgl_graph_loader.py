import dgl
import os
import torch
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


class ChameleonDataset:
    def __init__(self, raw_dir, reverse_edge=True, verbose=True):
        self.raw_dir = raw_dir
        self.g_list, self.graph_label = dgl.load_graphs(os.path.join(raw_dir, 'chameleon_dgl_graph.bin'))
        attributes_dict = torch.load(os.path.join(raw_dir, 'chameleon_attributes.dic'))
        if reverse_edge:
            self.g_list[0] = dgl.to_bidirected(self.g_list[0])
        self.num_classes = attributes_dict['num_class']
        for k, v in attributes_dict['attributes'].items():
            self.g_list[0].ndata[k] = v

    def __getitem__(self, item):
        return self.g_list[item]

    def __len__(self):
        return len(self.g_list)


class AirportDataset:
    def __init__(self, raw_dir, reverse_edge=True, verbose=True):
        self.raw_dir = raw_dir
        self.g_list, self.graph_label = dgl.load_graphs(os.path.join(raw_dir, 'airport_dgl_graph.bin'))
        if reverse_edge:
            self.g_list[0] = dgl.to_bidirected(self.g_list[0], copy_ndata=True)
        self.g_list[0].ndata['train_mask'] = self.g_list[0].ndata['train_mask'].bool()
        self.g_list[0].ndata['val_mask'] = self.g_list[0].ndata['val_mask'].bool()
        self.g_list[0].ndata['test_mask'] = self.g_list[0].ndata['test_mask'].bool()
        self.num_classes = len(torch.unique(self.g_list[0].ndata['label']))

    def __getitem__(self, item):
        return self.g_list[item]

    def __len__(self):
        return len(self.g_list)


class OgbnArxivDataset:
    def __init__(self, raw_dir, reverse_edge=True, verbose=True):
        self.raw_dir = raw_dir
        from ogb.nodeproppred import DglNodePropPredDataset
        data_loader = DglNodePropPredDataset(name='ogbn-arxiv', root=raw_dir)
        self.g_list = list(data_loader[0])
        year = 2018
        self.g_list[0].ndata['train_mask'] = (self.g_list[0].ndata['year'] < year).squeeze()
        self.g_list[0].ndata['val_mask'] = (self.g_list[0].ndata['year'] == year).squeeze()
        self.g_list[0].ndata['test_mask'] = (self.g_list[0].ndata['year'] > year).squeeze()
        self.num_classes = data_loader.num_classes
        self.g_list[0].ndata['label'] = data_loader.labels.squeeze()
        if reverse_edge:
            self.g_list[0] = dgl.to_bidirected(self.g_list[0], copy_ndata=True)

    def __getitem__(self, item):
        return self.g_list[item]

    def __len__(self):
        return len(self.g_list)


DATASET_MAP = {
    'cora': CoraGraphDataset,
    'citeseer': CiteseerGraphDataset,
    'pubmed': PubmedGraphDataset,
    'chameleon': ChameleonDataset,
    'airport': AirportDataset,
    'ogbn-arxiv': OgbnArxivDataset
}
