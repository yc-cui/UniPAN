import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import os
import torchmetrics.functional as MF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from UniPAN.metric import D_lambda_torch, D_s_torch
from UniPAN.model.PanNet.network import PanNetModel
from UniPAN.util.misc import check_and_make
from UniPAN.loss import l1_loss
from UniPAN.base.base_model import BaseModel
from sorcery import dict_of
from thop import profile
from thop import clever_format


class PanNet(BaseModel):
    def __init__(self,
                 min_lr,
                 lr,
                 warmup_epochs,
                 epochs,
                 bands,
                 rgb_c,
                 sensor,
                 scaler_type,
                 viz_dir
                 ):
        super().__init__(
                         lr,
                         epochs,
                         bands,
                         rgb_c,
                         sensor,
                         scaler_type,
                         viz_dir
                         )

        self.model = PanNetModel(bands)
        self.loss = l1_loss

        self.visual_idx = [i for i in range(20)]

    def forward(self, transformed_ms_down, transformed_pan_down, 
                ori_ms_down, ori_ms_down_up, ori_pan_down):
        pred = self.model(transformed_ms_down, transformed_pan_down) + ori_ms_down_up
        out = dict_of(pred)
        return out
