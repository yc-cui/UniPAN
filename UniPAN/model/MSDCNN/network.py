#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


class MSDCNNNet(nn.Module):
    def __init__(self):
        super(MSDCNNNet, self).__init__()

        base_filter = 64
        num_channels = 5
        out_channels = 4
        self.head = ConvBlock(num_channels, 64, 9, 1, 4, activation='prelu', norm=None, bias = True)

        self.body = ConvBlock(64, 32, 5, 1, 2, activation='prelu', norm=None, bias = True)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='prelu', norm=None, bias = True)

        self.ms_head = ConvBlock(num_channels, 60, 7, 1, 3, activation='prelu', norm=None, bias = True)

        self.ms_body1_3 = ConvBlock(60, 20, 3, 1, 1, activation='prelu', norm=None, bias = True)
        self.ms_body1_5 = ConvBlock(60, 20, 5, 1, 2, activation='prelu', norm=None, bias = True)
        self.ms_body1_7 = ConvBlock(60, 20, 7, 1, 3, activation='prelu', norm=None, bias = True)
        self.ms_body1 = ConvBlock(60, 30, 3, 1, 1, activation='prelu', norm=None, bias = True)


        self.ms_body2_3 = ConvBlock(30, 10, 3, 1, 1, activation='prelu', norm=None, bias = True)
        self.ms_body2_5 = ConvBlock(30, 10, 5, 1, 2, activation='prelu', norm=None, bias = True)
        self.ms_body2_7 = ConvBlock(30, 10, 7, 1, 3, activation='prelu', norm=None, bias = True)
        self.ms_body2 = ConvBlock(30, out_channels, 5, 1, 2, activation='prelu', norm=None, bias = True)

        # self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='prelu', norm=None, bias = True)


    def forward(self, b_ms, x_pan):

        x_f_i = torch.cat([b_ms, x_pan], 1)
        x_f = self.head(x_f_i)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)

        ms_x_f = self.ms_head(x_f_i)
        ms_x_f = torch.cat([self.ms_body1_3(ms_x_f), self.ms_body1_5(ms_x_f), self.ms_body1_7(ms_x_f)], 1) + ms_x_f
        ms_x_f = self.ms_body1(ms_x_f)
        ms_x_f = torch.cat([self.ms_body2_3(ms_x_f), self.ms_body2_5(ms_x_f), self.ms_body2_7(ms_x_f)], 1) + ms_x_f
        ms_x_f = self.ms_body2(ms_x_f)
        
        return x_f + ms_x_f