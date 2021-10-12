from typing import *

import os
import json

import torch

from ..data.configuration import get_configuration, get_device
from .lightning import LitModule
from .models import GCNNet, GATNet

WEIGHTS_FOLDER = "../../data/weights"


def get_savedmodel_path() -> str:
    """Sets up path for saving a model

    :return: path to saved model instance
    :rtype: str
    """
    CONFIG = get_configuration()
    model_name = f"{CONFIG['use_model']}-{CONFIG['loss']}-{CONFIG['use_data']}"
    save_folder = os.path.join(WEIGHTS_FOLDER, 'SavedModel')
    save_folder_model = os.path.join(save_folder, model_name)

    if not os.path.exists(save_folder_model):
        os.makedirs(save_folder_model)

    savedmodel_path = os.path.join(save_folder_model, "weights.pt")
    
    return savedmodel_path


def save_model(model: LitModule):
    """Saves a model

    :param model: model
    :type model: LitModule
    """
    savedmodel_path = get_savedmodel_path()
    torch.save(model.state_dict(), savedmodel_path)


def load_model() -> LitModule:
    """Loads model

    :raises NotImplementedError: In case of invalid configuration
    :return: Loaded model from disk
    :rtype: LitModule
    """

    CONFIG = get_configuration()
    device = get_device()
    
    if CONFIG["loss"] == "mrse":
        graph_prediction = True
    elif CONFIG["loss"] == "cross-entropy":
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

    savedmodel_path = get_savedmodel_path()
    model.load_state_dict(torch.load(savedmodel_path))
    model.to(device)

    return model

