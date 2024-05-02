import dgl
import torch
from dgl.nn import GraphConv, SAGEConv, GATConv
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feat, hidden_dim, layers, num_class, dropout=0.5, norm_type='both', batch_norm=False):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList(
            [GraphConv(in_feat, hidden_dim, norm=norm_type)] +
            [GraphConv(hidden_dim, hidden_dim, norm=norm_type) for i in range(layers-2)] +
            [GraphConv(hidden_dim, num_class, norm=norm_type)]
        )
        if batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers-1)])
        else:
            self.bns = [lambda x: x for _ in range(layers - 1)]
        self.dropout = dropout
        assert len(self.convs) == len(self.bns) + 1

    def forward(self, g, feat, edge_weight=None, **kwargs):
        h = feat
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(g, h, edge_weight=edge_weight)
            h = self.bns[i](h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](g, h, edge_weight=edge_weight)
        return h


class MultiLayerJKNet(nn.Module):
    def __init__(self, in_feat, hidden_dim, layers, num_class, dropout=0.5, mode='cat'):
        super().__init__()
        self.convs = nn.ModuleList(
            [GraphConv(in_feat, hidden_dim)] + [GraphConv(hidden_dim, hidden_dim) for i in range(layers - 1)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers - 1)])
        self.dropout = dropout
        # self.mode=  mode
        self.lin = nn.Linear(layers * hidden_dim, num_class)
        # if mode == 'cat':
        #     self.lin = nn.Linear(layers * hidden_dim, num_class)
        # else:
        #     self.lin = nn.Linear(hidden_dim, num_class)

    def forward(self, g, feat, weight=None):
        h = feat
        hs = []
        for i, conv in enumerate(self.convs):
            h = conv(g, h, weight=weight)
            h = self.bns[i](h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs += [h]
        # h = self.convs[-1](g, h, weight=weight)
        h = torch.cat(hs, dim=-1)
        h = self.lin(h)
        return h
    
    
class GCNWithLinear(nn.Module):
    def __init__(self, in_feat, hidden_dim, layers, num_class, dropout=0.5, norm_type='both', batch_norm=True):
        super().__init__()
        self.input_proj = nn.Linear(in_feat, hidden_dim)
        self.convs = nn.ModuleList(
            [GraphConv(hidden_dim, hidden_dim, norm=norm_type) for _ in range(layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, num_class)
        if batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(layers+1)])
        else:
            self.bns = [lambda x: x for _ in range(layers+1)]
        self.dropout = dropout
        assert len(self.convs) == len(self.bns) - 1

    def forward(self, g, feat, edge_weight=None, **kwargs):
        h = feat
        # init proj
        h = self.input_proj(h)
        h = self.bns[0](h)
        h = F.relu(h, inplace=True)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # GNN conv
        for i, conv in enumerate(self.convs):
            h = conv(g, h, edge_weight=edge_weight)
            h = self.bns[i+1](h)
            h = F.relu(h, inplace=True)
            h = F.dropout(h, p=self.dropout, training=self.training)
        # output
        h = self.output_proj(h)
        return h


GNN_MODEL_CONSTURCTOR = {
    'GCN': GCN,
    'GCNLinear': GCNWithLinear
    # 'GraphSAGE': GraphSAGEModel,
    # 'GAT': GATModel,
    # 'JKNet': JKNetModel,
    # 'NoLinSAGE': NoLinearGraphSAGEModel
}


if __name__ == '__main__':
    model = GCNWithLinear(140, 256, 4, 40, dropout=0.5, norm_type='both', batch_norm=True)
    print(model)
