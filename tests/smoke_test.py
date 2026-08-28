"""RCR-YOLO smoke test (CPU-friendly, no dataset download required).

Covers:
  1. unit forward of ORBIn / LCR / LCRBase / MRFE / GSConv
  2. build all 9 ablation yamls + train/eval forward
  3. ORB-In zero-inference-overhead property (identity in eval, recon cached in train)
  4. RCRTrainer.criterion adds a reconstruction loss when ORB-In is present
  5. one-epoch training on a synthetic YOLO-format dataset

Run:  python tests/smoke_test.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from rcr.orb_in import ORBIn  # noqa: E402
from rcr.lcr import LCR, LCRBase  # noqa: E402
from rcr.mrfe import MRFE  # noqa: E402
from rcr.common import GSConv  # noqa: E402
from rcr.trainer import RCRTrainer  # noqa: E402
YAMLS = [
    "cfg/yolo11n-orb.yaml", "cfg/yolo11n-mrfe.yaml", "cfg/yolo11n-lcr.yaml",
    "cfg/yolo11n-lcrbase.yaml", "cfg/yolo11n-rcrfb.yaml", "cfg/yolo11n-rcr.yaml",
    "cfg/yolo11n-orbmrfe.yaml", "cfg/yolo11n-mrfelcrb.yaml", "cfg/yolo11n-orblcrb.yaml",
]


def step1_modules():
    x = torch.randn(1, 64, 40, 40)
    orb = ORBIn(64)
    orb.eval()
    assert torch.equal(orb(x), x), "ORBIn must be identity in eval"
    assert orb.recon is None
    orb.train()
    y = orb(x)
    assert torch.equal(y, x) and orb.recon is not None
    # n_body=2 -> 2x upsample each body block: 40 -> 160 (recovers input res from P2)
    assert orb.recon.shape == (1, 2, 160, 160), f"recon shape {orb.recon.shape}"
    tgt = ORBIn.build_target([torch.tensor([[0.5, 0.5, 0.4, 0.4]])], None, 40, 40, "cpu")
    assert tgt.shape == (1, 2, 40, 40) and tgt[:, 0].sum() > 0 and tgt[:, 1].sum() > 0
    for mod in (LCR(64), LCRBase(64), MRFE(64), GSConv(64, 64)):
        out = mod(x)
        assert out.shape[0] == 1 and out.shape[1] in (64,), f"{type(mod).__name__} out {out.shape}"
    print("[1/5] module forwards ................ OK")


def step2_yamls():
    from ultralytics.nn.tasks import DetectionModel
    for p in YAMLS:
        cfg = yaml.safe_load((ROOT / p).read_text())
        m = DetectionModel(str(ROOT / p), nc=cfg["nc"])
        m.eval()
        with torch.no_grad():
            r = m(torch.randn(1, 3, 320, 320))
        # new ultralytics returns (preds, attrs) in eval; older returns a tensor
        assert isinstance(r, torch.Tensor) or (isinstance(r, tuple) and isinstance(r[0], torch.Tensor))
        n = sum(t.numel() for t in m.parameters())
        print(f"      {Path(p).stem:<20} params={n/1e6:.2f}M")
    print("[2/5] yaml builds + eval forward ..... OK")


def step3_zero_overhead():
    from ultralytics.nn.tasks import DetectionModel
    base = DetectionModel("yolo11n.yaml", nc=80)
    rcr = DetectionModel(str(ROOT / "cfg/yolo11n-rcr.yaml"), nc=80)
    # ORB-In is identity in eval: detection output path identical, recon inactive
    orbin = next(m for m in rcr.modules() if isinstance(m, ORBIn))
    orbin.eval()
    assert orbin.recon is None
    base.eval(), rcr.eval()
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        o1, o2 = base(x), rcr(x)
    if isinstance(o1, tuple):
        o1, o2 = o1[0], o2[0]
    assert o1.shape == o2.shape, f"{o1.shape} vs {o2.shape}"
    print(f"      baseline={sum(t.numel() for t in base.parameters())/1e6:.2f}M "
          f"rcr={sum(t.numel() for t in rcr.parameters())/1e6:.2f}M "
          f"(ORB-In only active in training -> zero inference cost)")
    print("[3/5] ORB-In zero-overhead ........... OK")


def step4_criterion():
    from ultralytics import YOLO
    from rcr.trainer import patch_detection_model_loss
    model = YOLO(str(ROOT / "cfg/yolo11n-rcr.yaml")).model
    from ultralytics.cfg import get_cfg
    model.args = get_cfg()  # default hyperparameters (box/cls/dfl gains, etc.)
    model.criterion = model.init_criterion()
    patch_detection_model_loss()  # same patch RCRTrainer.setup_model applies
    orb = next(m for m in model.modules() if isinstance(m, ORBIn))
    model.train()
    batch = {
        "img": torch.randn(2, 3, 320, 320),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [3.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.25, 0.3]]),
    }
    loss, items = model(batch)
    loss = loss.sum()
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    assert orb.end.weight.grad is not None, "recon loss must flow gradients into ORB-In"
    print(f"      detection items={dict(items)} total loss={loss.item():.3f} (includes recon)")
    print("[4/5] RCRTrainer recon loss .......... OK")


def step5_synthetic_train():
    from ultralytics import YOLO
    tmp = Path(tempfile.mkdtemp(prefix="rcr_smoke_"))
    try:
        rng = np.random.default_rng(0)
        for split in ("train", "val"):
            (tmp / split / "images").mkdir(parents=True)
            (tmp / split / "labels").mkdir(parents=True)
        for split, n in (("train", 4), ("val", 2)):
            for i in range(n):
                im = rng.integers(0, 255, (320, 320, 3), np.uint8)
                import cv2
                cv2.imwrite(str(tmp / split / "images" / f"{i}.jpg"), im)
                cx, cy, bw, bh = rng.uniform(0.25, 0.75, 2).tolist() + [0.3, 0.3]
                (tmp / split / "labels" / f"{i}.txt").write_text(f"0 {cx:.4f} {cy:.4f} {bw} {bh}\n")
        data = {
            "path": str(tmp), "train": "train/images", "val": "val/images",
            "nc": 1, "names": ["obj"],
        }
        data_yaml = tmp / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(data))

        trainer = RCRTrainer(overrides=dict(
            model=str(ROOT / "cfg/yolo11n-rcr.yaml"), data=str(data_yaml),
            epochs=1, batch=2, imgsz=320, device="cpu", workers=0,
            project=str(tmp / "runs"), name="smoke", amp=False,
            mosaic=0.0, plots=False, verbose=False, exist_ok=True,
        ))
        trainer.train()
        best = Path(trainer.save_dir) / "weights" / "best.pt"
        if not best.exists():
            best = Path(trainer.save_dir) / "weights" / "last.pt"
        assert best.exists(), "no weights saved"
        # reload and verify zero-overhead inference
        m = YOLO(str(best))
        r = m.predict(str(tmp / "val" / "images" / "0.jpg"), imgsz=320, verbose=False)
        assert len(r) == 1
        print(f"      trained 1 epoch on 4 synthetic imgs, weights={best.name}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[5/5] synthetic one-epoch training ... OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    step1_modules()
    step2_yamls()
    step3_zero_overhead()
    step4_criterion()
    step5_synthetic_train()
    print("\nALL SMOKE TESTS PASSED")
