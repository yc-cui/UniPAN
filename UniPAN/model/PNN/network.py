#!/usr/bin/env python
# coding=utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import *
import torch.nn.functional as F


class PNNNet(nn.Module):
    def __init__(self, in_channels=5):
        super(PNNNet, self).__init__()

        # Network variables

        # Network structure
        self.conv1 = nn.Conv2d(in_channels, 48, 9, 1, 4)
        self.conv2 = nn.Conv2d(48, 32, 5, 1, 2)
        self.conv3 = nn.Conv2d(32, 4, 5, 1, 2)

    def forward(self, ms, pan):
        inp = torch.cat([ms, pan], dim=1)
        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x