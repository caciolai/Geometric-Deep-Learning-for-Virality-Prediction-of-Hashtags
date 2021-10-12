from typing import *

import os
import json

import torch
from torch._C import get_device
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dropout_adj
import pytorch_lightning.metrics as pl_metrics

from src.data.configuration import get_configuration 

from .misc import SumPooling, MRSELoss
from .lightning import LitModule


class Net(nn.Module):
    def __init__(self, num_node_features, num_layers, 
                 hidden_dim, dropout, graph_prediction):
        
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.p_dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = dropout_adj

        self.graph_prediction = graph_prediction

        self.sum_pooling = SumPooling()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Convolutions
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            edge_index, _ = self.edge_dropout(edge_index, p=self.p_dropout)
        
        # Last convolution + sigmoid computes nodes 'activation state'
        x = self.convs[-1](x, edge_index)
        
        if self.graph_prediction:
            x = torch.sigmoid(x)
            
            # Sum pooling over the nodes computes the spread of the information
            x = self.sum_pooling(x, batch)
        return x

class GCNNet(Net):
    def __init__(self, num_node_features, num_layers, 
                 hidden_dim, dropout, graph_prediction):
        
        super().__init__(
            num_node_features, num_layers, 
            hidden_dim, dropout, graph_prediction
        )
        
        self.convs.append(
            GCNConv(num_node_features, hidden_dim, normalize=True)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_dim, hidden_dim, normalize=True)
            )
        
        # If graph prediction, then output a popularity for the whole graph, 
        # otherwise, output logits for binary classification
        out_dim = 1 if self.graph_prediction else 2
        self.convs.append(
            GCNConv(hidden_dim, out_dim, normalize=True)
        )

class GATNet(Net):
    def __init__(self, num_node_features, num_layers, 
                 hidden_dim, dropout, graph_prediction):
                
        super().__init__(
            num_node_features, num_layers, 
            hidden_dim, dropout, graph_prediction
        )

        self.convs.append(
            GATConv(num_node_features, hidden_dim)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim)
            )
        
        # If graph prediction, then output a popularity for the whole graph, 
        # otherwise, output logits for binary classification
        out_dim = 1 if self.graph_prediction else 2
        self.convs.append(
            GATConv(hidden_dim, out_dim)
        )


def build_model():
    CONFIG = get_configuration()
    device = get_device()
    
    if CONFIG["loss"] == "mrse":
        loss_fn = MRSELoss()
        val_metric = None
        graph_prediction = True
    elif CONFIG["loss"] == "cross-entropy":
        loss_fn = nn.CrossEntropyLoss()
        val_metric = pl_metrics.classification.F1()
        graph_prediction = False
    else:
        raise NotImplementedError

    
    if CONFIG["use_model"] == "GCN":
        model = GCNNet(num_node_features=6,
                    num_layers=CONFIG["num_layers"],
                    hidden_dim=CONFIG["hidden_dim"],
                    dropout=CONFIG["dropout"],
                    graph_prediction=graph_prediction)
    elif CONFIG["use_model"] == "GAT":
        model = GATNet(num_node_features=6,
                    num_layers=CONFIG["num_layers"],
                    hidden_dim=CONFIG["hidden_dim"],
                    dropout=CONFIG["dropout"],
                    graph_prediction=graph_prediction)
    else:
        raise NotImplementedError    

    model = model.to(device)
    
    lit_model = LitModule(model, loss_fn, val_metric)
