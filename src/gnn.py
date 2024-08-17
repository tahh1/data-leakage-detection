import dgl
from dgl.nn import GatedGraphConv
import os
import torch
import dgl
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import dgl.nn.pytorch as dglnn

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,n_convs):
        super(GNN, self).__init__()
        self.gatedconv = GatedGraphConv(in_dim, hidden_dim,n_convs,1)
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(0.05)

    def forward(self, g, h):
        h = self.dropout(self.gatedconv(g, h))
        with g.local_scope():
            g.ndata['features'] = h
            hg = dgl.readout_nodes(g, 'features',op='max')
            return self.classify(hg)