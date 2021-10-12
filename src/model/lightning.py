from typing import *

import os
import json

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from ..data.configuration import get_configuration


class LitModule(pl.LightningModule):
    
    def __init__(self, model, 
                 loss_fn=None, 
                 val_metric=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.val_metric = val_metric
    
    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        assert self.loss_fn is not None

        y_train_true = batch['y']

        output = self(batch)
        
        loss = self.loss_fn(output, y_train_true)
        
        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)

        return result

    def predict(self, output):
        if self.model.graph_prediction:
            y_pred = torch.round(output)
        else:
            logits_softmaxed = F.softmax(output, dim=-1)
            y_pred = torch.argmax(logits_softmaxed, dim=-1)
        
        return y_pred

    def validation_step(self, batch, batch_idx):
        assert self.loss_fn is not None

        y_val_true = batch['y']
        output = self(batch)

        val_loss = self.loss_fn(output, y_val_true)

        if self.val_metric is not None:
            y_val_pred = self.predict(output)
            val_metric = self.val_metric(y_val_pred, y_val_true)
            log_dict = {'val_loss': val_loss, 'val_metric': val_metric}
        else:
            val_metric = val_loss
            log_dict = {'val_loss': val_loss}

        result = pl.EvalResult(early_stop_on=val_metric)
        result.log_dict(
            log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return result

    def test_step(self, batch, batch_idx):
        assert self.loss_fn is not None
        
        y_test_true = batch['y']

        output = self(batch) 

        test_loss = self.loss_fn(output, y_test_true)

        if self.val_metric is not None:
            y_test_pred = self.predict(output)
            test_metric = self.val_metric(y_test_pred, y_test_true)
            log_dict = {'test_loss': test_loss, 'test_metric': test_metric}
        else:
            val_metric = test_loss
            log_dict = {'test_loss': test_loss}

        result = pl.EvalResult()
        result.log_dict(
            log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return result

    def configure_optimizers(self):
        CONFIG = get_configuration()
        return torch.optim.Adam(self.parameters(), lr=CONFIG["lr"])