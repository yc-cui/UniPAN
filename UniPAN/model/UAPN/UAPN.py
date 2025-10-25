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
from UniPAN.base.base_model import BaseModel
from UniPAN.metric import D_lambda_torch, D_s_torch
from UniPAN.model.UAPN.network import TotalLoss, UAPNModel, UncertaintyLoss
from UniPAN.util.misc import check_and_make
from UniPAN.loss import l1_loss
from sorcery import dict_of
from thop import profile
from thop import clever_format


class UAPN(BaseModel):
    def __init__(self,
                 min_lr,
                 lr,
                 warmup_epochs,
                 epochs,
                 bands,
                 rgb_c,
                 sensor,
                 scaler_type,
                 viz_dir):
        super().__init__(
                        lr,
                         epochs,
                         bands,
                         rgb_c,
                         sensor,
                         scaler_type,
                         viz_dir)

        self.model = UAPNModel(bands, 4)
        self.loss = TotalLoss()

        self.visual_idx = [i for i in range(20)]


    def forward(self, transformed_ms_down, transformed_pan_down, 
                ori_ms_down, ori_ms_down_up, ori_pan_down):
        AU, EU, hrms1, hrms2 = self.model(lms=transformed_ms_down, pan=transformed_pan_down)
        pred = hrms2 + ori_ms_down_up
        return dict_of(AU, EU, hrms1, pred)
    

    
    def training_step(self, batch, batch_idx):
        
        transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down = batch['transformed_ms_down'], batch['transformed_pan_down'], batch['ori_ms_down'], batch['ori_ms_down_up'], batch['ori_pan_down']
        out = self.forward(transformed_ms_down, transformed_pan_down, ori_ms_down, ori_ms_down_up, ori_pan_down)
        gt = batch['gt']
        AU, EU, hrms1, hrms2 = out["AU"], out["EU"], out["hrms1"], out["pred"]
        total_loss, log_dict = self.loss(AU, EU, hrms1, hrms2, gt)

        opt = self.optimizers()
        opt.zero_grad()

        self.manual_backward(total_loss)
        opt.step()

        log_dict["lr"] = opt.param_groups[0]["lr"]
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)

