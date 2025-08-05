"""Tiniest model shared as-is by github.com/xvel/cifar10-tiniest."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffdim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=3, padding='same')
        self.dwconv2 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=7, padding='same')
        self.dwconv3 = nn.Conv2d(in_channels=dim//4, out_channels=dim//4, groups=dim//4, kernel_size=7, dilation=3, padding='same')
        self.ln = LayerNorm2d(dim)
        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=ffdim*2, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels=ffdim, out_channels=dim, kernel_size=1)
        self.rs = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, xx):
        xc = torch.chunk(xx, 4, dim=1)
        x1 = self.dwconv1(xc[1])
        x2 = self.dwconv2(xc[2])
        x3 = F.avg_pool2d(xc[3], 3, stride=1, padding=1)
        x3 = self.dwconv3(x3)
        x = torch.cat([xc[0], x1, x2, x3], dim=1)
        x = self.ln(x)
        x = self.fc1(x)
        x = torch.chunk(x, 2, dim=1)
        x = torch.tanh(torch.exp(x[0])) * x[0] * x[1]
        x = self.fc2(x)
        return x + xx * self.rs


class Tiniest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=3, padding=1)
        self.ln1 = LayerNorm2d(48)
        self.l1 = Block(48, 48)
        self.l2 = Block(48, 48)
        self.l3 = Block(48, 48)
        self.dsconv = nn.Conv2d(in_channels=48, out_channels=80, kernel_size=4, groups=16, stride=2, padding=1)
        self.ln2 = LayerNorm2d(80)
        self.l4 = Block(80, 80)
        self.l5 = Block(80, 80)
        self.l6 = Block(80, 80)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.dsconv(x)
        x = self.ln2(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.avgpool(x)
        x = self.fc(torch.flatten(x, 1))
        return x
