from typing import *

import torch
import torch.nn as nn


class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self._summing_matrix = None

    def forward(self, x, batch):
        """
        Args:
            data: a batch of graphs
        """
        batch_size = torch.max(batch) + 1
        num_nodes = x.shape[0] // batch_size

        sum_pool = torch.zeros((batch_size,), dtype=x.dtype, device=x.device)
        
        for i in range(batch_size):
            x_graph = x[i*num_nodes:(i+1)*num_nodes, ...]
            sum_pool[i] = torch.sum(x_graph)

        return sum_pool


class MRSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        """
        Returns the Mean Relative Squared Error, defined as
            L_MRSE = 1/M \sum_{m = 1}^{M} [( \hat{y}_m - y_m )/y_m]^2
        """
        relative_error = (y_pred - y_true) / y_true
        rse = torch.pow(relative_error, 2)
        mrse = torch.mean(rse, dim=-1)

        return mrse