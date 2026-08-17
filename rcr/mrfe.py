"""MRFE: Multiple Receptive Field adaptive feature Enhancement (neck plug-in).

Adapted from ORFENet's MRFAFEM (Liu et al., TGRS 2024) and lightweighted with
the GSConv / depth-wise design practice of YOLO-SM (Yue & Meng, TETCI 2024):

* three parallel depth-wise separable branches with dilations (1, 2, 3)
  provide small / medium / large receptive fields for tiny, normal and
  partially-occluded indoor objects;
* a channel-global gate produces *dynamic* per-branch fusion weights
  (input-adaptive, unlike fixed concat/add);
* a 1x1 fuse conv plus residual keeps the block channel-preserving so it can
  be inserted anywhere in the YOLO neck without rewiring the yaml.
"""

import torch
import torch.nn as nn

from .common import DWConv, GSConv


class _Branch(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.dw = DWConv(c, c, k=3, s=1, d=d)
        self.pw = GSConv(c, c)

    def forward(self, x):
        return self.pw(self.dw(x))


class MRFE(nn.Module):
    """Channel-preserving multi-receptive-field adaptive enhancement block.

    yaml args: [c1] or [c1, [1, 2, 3]]
    """

    def __init__(self, c1, dilations=(1, 2, 3)):
        super().__init__()
        c = c1
        self.branches = nn.ModuleList(_Branch(c, d) for d in dilations)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, len(dilations)),
            nn.Softmax(dim=-1),
        )
        self.fuse = nn.Conv2d(c, c, 1)

    def forward(self, x):
        w = self.gate(x)  # (B, n_branch)
        out = None
        for i, b in enumerate(self.branches):
            fi = b(x)
            wi = w[:, i].view(-1, 1, 1, 1)
            out = fi * wi if out is None else out + fi * wi
        return x + self.fuse(out)
