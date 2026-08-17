"""ORB-In: indoor-adapted Object Reconstruction Branch (training-time only).

Adapted from ORFENet (Liu et al., TGRS 2024) ORB, redesigned for indoor
service-robot scenes:

* The reconstruction target is upgraded from a plain binary GT mask to a
  two-channel target [object mask, mask contour/edge map], which forces the
  high-resolution feature to keep both object-extent and boundary detail —
  the two cues most damaged by indoor blur, clutter and occlusion.
* The branch is attached to the stride-4 (P2) feature map and is used ONLY
  during training; ``forward`` returns its input unchanged, so the branch is
  discarded at inference and adds ZERO parameters / FLOPs / latency.

The reconstruction prediction is cached on ``self.recon`` during training;
the custom trainer (``rcr.trainer.RCRTrainer``) reads it and adds the
reconstruction loss to the detection loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ORBIn(nn.Module):
    """Identity-pass-through module with a training-only reconstruction head.

    yaml args: [c1, n_body, out_channels]  (channels are written explicitly in
    the yaml because Ultralytics parse_model does not inject them for custom
    modules; the module is channel-preserving / identity on the detect path).
    """

    def __init__(self, c1, n_body=2, out_channels=2):
        super().__init__()
        c = c1
        self.head = nn.Conv2d(c, c, 3, padding=1)
        self.body = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
            for _ in range(int(n_body))
        )
        self.end = nn.Conv2d(c, int(out_channels), 3, padding=1)
        self.recon = None  # cached prediction, consumed by RCRTrainer

    def forward(self, x):
        if self.training:
            y = torch.relu(self.head(x))
            for blk in self.body:
                y = F.interpolate(blk(y), scale_factor=2.0, mode="bilinear", align_corners=False)
            self.recon = self.end(y)  # (B, 2, H_in, W_in): [mask logits, edge logits]
        return x  # identity: detection path untouched, zero inference cost

    @staticmethod
    def build_target(batch_bboxes, batch_masks, h, w, device):
        """Build the two-channel reconstruction target.

        Args:
            batch_bboxes: list of (N_i, 4) tensors, normalized cxcywh per image.
            batch_masks: unused placeholder for future instance-mask extension.
            h, w: input image resolution.
        Returns:
            (B, 2, h, w) target tensor with channels [mask, edge].
        """
        b = len(batch_bboxes)
        target = torch.zeros(b, 2, h, w, device=device)
        for i, boxes in enumerate(batch_bboxes):
            if boxes.numel() == 0:
                continue
            cx, cy, bw, bh = boxes.unbind(-1)
            x1 = torch.clamp((cx - bw / 2) * w, 0, w - 1).long()
            x2 = torch.clamp((cx + bw / 2) * w, 0, w - 1).long()
            y1 = torch.clamp((cy - bh / 2) * h, 0, h - 1).long()
            y2 = torch.clamp((cy + bh / 2) * h, 0, h - 1).long()
            m = target[i, 0]
            for a, bb, cc, dd in zip(y1, y2, x1, x2):
                if bb > a and dd > cc:
                    m[a : bb + 1, cc : dd + 1] = 1.0
            # contour/edge channel: boundary of the mask (1-px dilation diff)
            mb = m > 0.5
            edge = torch.zeros_like(mb)
            edge[1:, :] |= mb[1:, :] != mb[:-1, :]
            edge[:, 1:] |= mb[:, 1:] != mb[:, :-1]
            target[i, 1] = edge.float()
        return target
