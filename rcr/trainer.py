"""RCRTrainer: Ultralytics DetectionTrainer + ORB-In reconstruction loss.

During training the ORBIn module caches its reconstruction prediction on
``module.recon``; this trainer builds the two-channel indoor target
(object mask + contour/edge map) from the batch ground truth and adds

    L_recon = L_mask(BCE, pos-weighted) + 0.5 * L_edge(BCE)

to the detection loss.  At validation / inference the branch is inactive, so
the exported model keeps exactly the baseline cost (zero overhead).

Compatibility: ultralytics >= 8.4 moved loss computation into
``model.loss(batch)`` (called by ``BaseModel.forward`` when the input is a
dict), so we patch ``DetectionModel.loss`` at the CLASS level (idempotent).
A class-level patch leaves no function attribute on the model instance, so
checkpoints stay picklable and load back with stock Ultralytics.

The reconstruction weight can be set with env var ``RCR_RECON_WEIGHT``
(default 1.0).
"""

import os

import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer

from .orb_in import ORBIn

_RECON_WEIGHT = float(os.environ.get("RCR_RECON_WEIGHT", "1.0"))


def recon_loss(orb, batch):
    """Build the indoor mask+edge target from GT boxes and score ``orb.recon``."""
    pred = orb.recon
    h, w = pred.shape[-2:]  # build the target directly at recon resolution
    boxes_per_img = [
        batch["bboxes"][batch["batch_idx"] == i] for i in range(pred.shape[0])
    ]
    target = ORBIn.build_target(boxes_per_img, None, h, w, pred.device)

    pos_mask = (target[:, :1] < 0.5).sum().float() / (target[:, :1].sum() + 1.0)
    pos_edge = (target[:, 1:2] < 0.5).sum().float() / (target[:, 1:2].sum() + 1.0)
    l_mask = F.binary_cross_entropy_with_logits(
        pred[:, :1], target[:, :1], pos_weight=pos_mask.clamp_min(1.0)
    )
    l_edge = F.binary_cross_entropy_with_logits(
        pred[:, 1:2], target[:, 1:2], pos_weight=pos_edge.clamp_min(1.0)
    )
    return (l_mask + 0.5 * l_edge) * _RECON_WEIGHT


def patch_detection_model_loss():
    """Add the ORB-In reconstruction term to DetectionModel.loss (idempotent).

    Only active for models that actually contain an ORBIn module and only in
    training mode; baselines/eval keep the stock loss untouched.
    """
    from ultralytics.nn.tasks import DetectionModel

    if hasattr(DetectionModel, "_rcr_orig_loss"):
        return

    orig_loss = DetectionModel.loss

    def loss_with_recon(self, batch, *args, **kwargs):
        loss, items = orig_loss(self, batch, *args, **kwargs)
        if self.training:
            orb = next((m for m in self.modules() if isinstance(m, ORBIn)), None)
            if orb is not None and orb.recon is not None:
                loss = loss + recon_loss(orb, batch)
        return loss, items

    DetectionModel._rcr_orig_loss = orig_loss
    DetectionModel.loss = loss_with_recon


class RCRTrainer(DetectionTrainer):
    """Detection trainer with training-time-only reconstruction supervision."""

    def setup_model(self):
        patch_detection_model_loss()
        super().setup_model()
