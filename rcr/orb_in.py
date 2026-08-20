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
* Design note: the head performs a single 2x bilinear upsample, so the
  reconstruction supervision lives at stride-2 (2x the P2 resolution) rather
  than full input resolution.  This keeps the extra training cost ~6x lower
  (memory and FLOPs) than a 4x upsample while preserving boundary detail far
  beyond what stride-4 features alone can encode.

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

        Fully vectorized (no per-image/per-box Python loops): drawing every
        box as a single scatter and deriving the edge channel from a shifted
        difference keeps this O(1) GPU launches per batch, which matters a
        lot at high batch sizes / dense indoor scenes.
        """
        b = len(batch_bboxes)
        target = torch.zeros(b, 2, h, w, device=device)
        boxes = [bb.to(device) for bb in batch_bboxes]
        counts = torch.tensor([bb.shape[0] for bb in boxes], device=device)
        if int(counts.sum()) == 0:
            return target
        all_boxes = torch.cat(boxes)
        batch_idx = torch.arange(b, device=device).repeat_interleave(counts)
        cx, cy, bw, bh = all_boxes.unbind(-1)
        x1 = torch.clamp((cx - bw / 2) * w, 0, w - 1).long()
        x2 = torch.clamp((cx + bw / 2) * w, 0, w - 1).long()
        y1 = torch.clamp((cy - bh / 2) * h, 0, h - 1).long()
        y2 = torch.clamp((cy + bh / 2) * h, 0, h - 1).long()
        # per-box pixel grids -> single scatter fills the mask channel
        heights = (y2 - y1 + 1).clamp_min(0)
        widths = (x2 - x1 + 1).clamp_min(0)
        areas = heights * widths
        total = int(areas.sum())
        if total > 0:
            bi = torch.repeat_interleave(batch_idx, areas)
            seq = torch.arange(total, device=device)
            start = torch.repeat_interleave(torch.cumsum(areas, 0) - areas, areas)
            local = seq - start  # per-pixel offset within its box
            w_each = torch.repeat_interleave(widths, areas)
            yy = torch.repeat_interleave(y1, areas) + local // w_each
            xx = torch.repeat_interleave(x1, areas) + local % w_each
            target[bi, 0, yy, xx] = 1.0
        # contour/edge channel: boundary of the mask (1-px dilation diff)
        mb = target[:, :1] > 0.5
        edge = torch.zeros_like(mb)
        edge[:, :, 1:, :] |= mb[:, :, 1:, :] != mb[:, :, :-1, :]
        edge[:, :, :, 1:] |= mb[:, :, :, 1:] != mb[:, :, :, :-1]
        target[:, 1:2] = edge.float()
        return target
