"""Deploy-path params / GFLOPs for the modern-YOLO comparison models.

Same recipe as tools/model_stats.py (unfused conv+BN, fp32, imgsz=640), with
one fix needed for end-to-end (NMS-free) heads: YOLOv10n and YOLO26n carry a
training-only one-to-many head (cv2/cv3). In eval mode the unfused Detect head
still runs it, so tools/model_stats.py would over-count them (v10n: 2.725M /
8.55 GFLOPs). Here it is removed with Ultralytics' own Detect.fuse()
(cv2 = cv3 = None), which is exactly what happens at deployment.

Latency is NOT measured here (needs the same RTX 4090 batch=1 setup as
tools/model_stats.csv).

Usage: python tools/model_stats_comparison.py
"""
import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

MODELS = ["yolov8n", "yolov10n", "yolo26n"]
CALIBRATION = "baseline-yolo11n"  # must reproduce tools/model_stats.csv (2.599M, 6.55)
OUT = ROOT / "tools" / "model_stats_comparison.csv"


def stats(name):
    ckpt = ROOT / "runs" / "rcr300" / f"{name}-s0" / "weights" / "best.pt"
    model = YOLO(str(ckpt)).model
    ckpt_params = sum(p.numel() for p in model.parameters())
    model = copy.deepcopy(model).eval()
    head = model.model[-1]
    e2e = bool(getattr(head, "end2end", False))
    if e2e:
        head.fuse()  # drop one2many head, as at deployment
    deploy = sum(p.numel() for p in model.parameters())
    return {
        "name": name,
        "end2end": e2e,
        "params_ckpt_M": round(ckpt_params / 1e6, 3),
        "params_deploy_M": round(deploy / 1e6, 3),
        "gflops": round(float(get_flops(model, 640)), 2),
        "latency_ms": "",
        "weights": ckpt.relative_to(ROOT).as_posix(),
    }


def main():
    cal = stats(CALIBRATION)
    assert (cal["params_deploy_M"], cal["gflops"]) == (2.599, 6.55), cal
    rows = [stats(m) for m in MODELS]
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"calibration ok ({CALIBRATION}); wrote {OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
