from typing import *

import random
import math
import os
import json
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from .configuration import get_configuration
from .datasets import prepare_data, GraphDataset

BATCH_SIZE = 16

def prepare_datasets() -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """Prepares train, validation, test datasets.

    :return: Train, validation and test graph datasets
    :rtype: Tuple[GraphDataset, GraphDataset, GraphDataset]
    """
    graph, samples = prepare_data()

    random.shuffle(samples)
    num_samples = len(samples)

    TRAIN_RATIO = 0.7
    DEV_RATIO = 0.2
    TEST_RATIO = 0.1

    train_ub = int(math.floor(num_samples*TRAIN_RATIO))
    val_ub = train_ub + int(math.floor(num_samples*DEV_RATIO))

    train_set = samples[0: train_ub]
    val_set = samples[train_ub: val_ub]
    test_set = samples[val_ub:]

    CONFIG = get_configuration()

    train_dataset = GraphDataset(graph, train_set, CONFIG["train_target"])
    val_dataset = GraphDataset(graph, val_set, CONFIG["train_target"])

    if CONFIG["loss"] == "mrse":
        test_dataset = GraphDataset(graph, test_set, "graph")
    elif CONFIG["loss"] == "cross-entropy":
        test_dataset = GraphDataset(graph, test_set, "node")
    else:
        raise NotImplementedError


    return train_dataset, val_dataset, test_dataset


def graph_collate(batch: List[dict]) -> Batch: 
    """Collate function to make pytorch geometric structures work inside pytorch lightining training loop

    :param batch: A batch at training time
    :type batch: List[dict]
    :return: A pytorch geometric batch
    :rtype: torch_geometric.data.Batch
    """
    data_list = []
    for sample in batch:
        
        edge_index = sample['edge_index']
        x = sample['x']
        static_features = sample['static_features']

        x = torch.cat((x, static_features), dim=1)
        
        y = sample['y']

        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    
    return Batch.from_data_list(data_list)


def prepare_dataloaders():
    train_dataset, val_dataset, test_dataset = prepare_datasets()

    train_loader = DataLoader(train_dataset, collate_fn=graph_collate, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, collate_fn=graph_collate, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, collate_fn=graph_collate, batch_size=BATCH_SIZE)

    