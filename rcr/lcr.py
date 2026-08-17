"""LCR: Lightweight Context Reasoning (easy-help-hard) for one-stage YOLO.

Lightweight adaptation of the graph reasoning idea from CRNet (Leng et al.,
TMM 2024) / IENet (Leng et al., Neurocomputing 2021).  The original works run
GRU message passing over proposal graphs built on heavy two-stage detectors.
LCR transfers the *mechanism* — easy regions/objects supply context to hard
ones under a global scene prior — into a cheap feature-map operator that can
be plugged into a real-time YOLO neck:

* a learned difficulty gate ``w`` plays the role of CRNet's easy/hard split
  (high-response "easy" regions vs. low-response "hard" regions);
* the easy context is the response-weighted aggregate of easy-region features
  (message from easy to hard);
* the scene node uses multi-granularity pooling (CRNet MGF spirit) as a global
  prior;
* message passing is a single 1x1-conv MLP instead of iterative GRU updates,
  and the enhancement is applied only where ``w`` is large (hard regions),
  keeping the compute bounded and latency-friendly.

``LCRBase`` is the degraded fallback (uniform gate, pure scene-context
augmentation) used in ablations and as the risk-mitigation fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SceneNode(nn.Module):
    """Multi-granularity global scene encoding (GAP at 1x1 / 2x2 / 4x4)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in ((1, 1), (2, 2), (4, 4))])
        self.proj = nn.Conv2d(c1 * 3, c2, 1)

    def forward(self, x):
        feats = [p(x) for p in self.pools]
        up = [F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False) for f in feats]
        return self.proj(torch.cat(up, dim=1))


class LCR(nn.Module):
    """Easy-help-hard lightweight context reasoning block (channel preserving).

    yaml args: [c1] (explicit channels; parse_model keeps c2 = c1).
    """

    def __init__(self, c1):
        super().__init__()
        c = c1
        mid = max(c // 4, 8)
        self.gate = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())  # difficulty map w
        self.scene = _SceneNode(c, c)
        self.msg = nn.Sequential(
            nn.Conv2d(3 * c, mid, 1),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1),
        )

    def forward(self, x):
        w = self.gate(x)  # (B,1,H,W): 1 => hard region needing help
        easy_w = 1.0 - w
        # easy-region context: response-weighted global aggregate (easy -> hard)
        denom = easy_w.sum(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        easy_ctx = (x * easy_w).sum(dim=(2, 3), keepdim=True) / denom
        s = self.scene(x)
        m = self.msg(torch.cat([x, s, easy_ctx.expand_as(x)], dim=1))
        return x + w * m  # enhance hard regions only


class LCRBase(nn.Module):
    """Fallback variant: uniform gate (w = 1), scene-context augmentation only.

    yaml args: [c1]
    """

    def __init__(self, c1):
        super().__init__()
        c = c1
        mid = max(c // 4, 8)
        self.scene = _SceneNode(c, c)
        self.msg = nn.Sequential(
            nn.Conv2d(2 * c, mid, 1),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1),
        )

    def forward(self, x):
        return x + self.msg(torch.cat([x, self.scene(x)], dim=1))
