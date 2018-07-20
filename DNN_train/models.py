import torch
import torch.nn as nn
import torch.nn.functional as F
from region_loss import RegionLoss
from utils import *

class HalfChannels(nn.Module):
    def __init__(self):
        super(HalfChannels, self).__init__()
        self.width = int(320)
        self.height = int(176)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_dw( 3,  16, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 16,  32, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 32, 64, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(64, 128, 1),
            nn.Conv2d(128, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1,1.06357021727,1,2.65376815391],2)
        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
    def forward(self, x):
        x = self.model(x)
        return x

class FullChannels(nn.Module):
    def __init__(self):
        super(FullChannels, self).__init__()
        self.width = int(320)
        self.height = int(176)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=True),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_dw( 3,  32, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 32,  64, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 64, 128, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(128, 256, 1),
            nn.Conv2d(256, 10, 1, 1,bias=False),
        )
        self.loss = RegionLoss([1,1.06357021727,1,2.65376815391],2)
        self.anchors = self.loss.anchors
        self.num_anchors = self.loss.num_anchors
        self.anchor_step = self.loss.anchor_step
    def forward(self, x):
        x = self.model(x)
        return x