from typing import *

import os
import json
import torch

DATA_FOLDER = "../../data"
CONFIG_PATH = os.path.join(DATA_FOLDER, "config.json")
CONFIG = json.load(CONFIG_PATH)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_configuration() -> dict:
    return CONFIG

def get_device() -> torch.device:
    """Returns the type of device currently in use.

    :return: The type of device currently in use
    :rtype: torch.device
    """
    return DEVICE

