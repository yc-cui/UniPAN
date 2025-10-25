import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, kernel_num):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.Conv2 = nn.Conv2d(in_channels=in_channels, padding=1, kernel_size=kernel_size, out_channels=kernel_num)

    def forward(self, x):
        y = F.relu(self.Conv1(x), False)
        y = self.Conv2(y)
        return x + y


class PanNetModel(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNetModel, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
                                       nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel + 1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=kernel_num, out_channels=channel, padding=1, kernel_size=kernel_size)


class PanNetModel(nn.Module):

    def __init__(self, channel, kernel_size=(3, 3), kernel_num=32):
        super(PanNetModel, self).__init__()
        # Conv2d默认stride=1, bias=True
        self.ConvTrans = nn.Sequential(nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
                                       nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=(3, 3), stride=2, padding=1, output_padding=1))
        self.Conv1 = nn.Conv2d(in_channels=channel + 1, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num),
                                            ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num))
        self.Conv2 = nn.Conv2d(in_channels=kernel_num, out_channels=channel, padding=1, kernel_size=kernel_size)

    def forward(self, hms, hpan):
        up_ms = self.ConvTrans(hms)
        x = torch.cat([hpan, up_ms], dim=1)
        y = F.relu(self.Conv1(x))
        y = self.ResidualBlocks(y)
        y = self.Conv2(y)
        return y
