from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

from ..data.configuration import get_device
from .misc import SumPooling


def predict(model, batch):
    device = get_device()
    assert not model.graph_prediction 
    with torch.no_grad():
        batch = batch.to(device)
        
        output = model(batch)
        output = output.to('cpu').detach()

        logits_softmaxed = F.softmax(output, dim=-1)
        y_pred = torch.argmax(logits_softmaxed, dim=-1)
        
        return y_pred

def predict_aggregate(model, batch):
    device = get_device()
    sum_pool = SumPooling()
    with torch.no_grad():
        batch = batch.to(device)
        
        output = model(batch)
        output = output.to('cpu').detach()
        if model.graph_prediction:
            # rounding the predicted popularity for the hashtag
            y_pred = torch.round(output)
        else:
            # softmaxing the activation of each node and then counting
            logits_softmaxed = F.softmax(output, dim=-1)
            y_pred = torch.argmax(logits_softmaxed, dim=-1)
            y_pred = sum_pool(y_pred, batch.batch)
        
        return y_pred

def compute_predictions(model, test_loader, aggregate=True):
    y_test_pred = []
    y_test_true = []

    for batch in tqdm(test_loader, total=len(test_loader)):
        y_batch_true = batch["y"]
        if aggregate:
            y_batch_pred = predict_aggregate(model, batch)
            y_test_pred.extend(y_batch_pred.tolist())
            y_test_true.extend(y_batch_true.tolist())
        else:
            y_batch_pred = predict(model, batch)
            y_test_pred.extend(y_batch_pred.tolist())
            y_test_true.extend(y_batch_true.tolist())
    
    return y_test_pred, y_test_true

def plot_predictions(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    # plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 5))

    num_samples = len(y_pred)

    x_axis = np.arange(0, num_samples)

    sorting_indices = np.argsort(y_true)
    sorted_y_true = np.array(np.sort(y_true))
    sorted_y_pred = np.array([y_pred[i] for i in sorting_indices])
    
    ax.plot(x_axis, sorted_y_true, '-', label="y true")
    ax.plot(x_axis, sorted_y_pred, '-', label="y pred")

    plt.legend(loc='upper left')
    plt.grid(axis='y')
    plt.show()

