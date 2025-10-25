import torch
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
    
    
class FusionNetModel(nn.Module):
    def __init__(self, channel=8, num_res=4, kernel_num=32):
        super(FusionNetModel, self).__init__()

        kernel_size = 3
        self.Conv1 = nn.Conv2d(in_channels=channel, padding=1, kernel_size=kernel_size, out_channels=kernel_num)
        self.ResidualBlocks = nn.Sequential(*[ResidualBlock(in_channels=kernel_num, kernel_size=kernel_size, kernel_num=kernel_num) for _ in range(num_res)])
        self.Conv2 = nn.Conv2d(in_channels=kernel_num, out_channels=channel, padding=1, kernel_size=kernel_size)

    def forward(self, up_ms, pan):
        
        pan = pan.repeat(1, up_ms.shape[1], 1, 1)
        inp = pan - up_ms
        
        out = F.relu(self.Conv1(inp))
        out = self.ResidualBlocks(out)
        out = self.Conv2(out)

        return out