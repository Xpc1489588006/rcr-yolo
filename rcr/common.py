"""Lightweight building blocks shared by RCR-YOLO modules."""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard conv + BN + SiLU (matches Ultralytics convention)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depth-wise conv + BN + act."""

    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (k // 2) * d, groups=c1, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GSConv(nn.Module):
    """Ghost Shuffle Convolution (from SlimNeck / YOLO-SM lineage).

    Generates half of the output channels with a cheap depth-wise linear
    operation and shuffles them with the standard-conv channels, keeping
    cross-channel information flow at low cost.
    """

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        assert c2 % 2 == 0, f"GSConv requires even output channels, got {c2}"
        self.conv = Conv(c1, c2 // 2, k, s, act=act)
        self.dw = DWConv(c2 // 2, c2 // 2, k=5, s=1, act=act)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dw(x1)
        # channel shuffle: interleave the two halves
        b, c, h, w = x1.shape
        out = torch.stack((x1, x2), dim=2).reshape(b, 2 * c, h, w)
        return out
