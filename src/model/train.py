import os
import json
import random
import torch
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from ..data.configuration import get_configuration
from .lightning import LitModule

def train(
    lit_model: LitModule,
    train_loader: DataLoader,
    val_loader: DataLoader
):
    """Trains the given model on the givend ata

    :param lit_model: The model to train
    :type lit_model: LitModule
    :param train_loader: The training data
    :type train_loader: DataLoader
    :param val_loader: The validation data
    :type val_loader: DataLoader
    """ 
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    CONFIG = get_configuration()

    early_stop_callback = EarlyStopping(
        min_delta=CONFIG["min_delta"],
        patience=CONFIG["patience"],
        verbose=True,
        mode='max'
    )

    model_name = f"{CONFIG['use_model']}-{CONFIG['loss']}-{CONFIG['use_data']}"
    logger = TensorBoardLogger('./tb_logs', name=model_name)

    trainer = Trainer(
        gpus=1, 
        max_epochs=CONFIG["num_epochs"], 
        num_sanity_val_steps=0, 
        logger=logger,
        early_stop_callback=early_stop_callback if CONFIG["early_stopping"] else None,
    )

    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=val_loader)

    