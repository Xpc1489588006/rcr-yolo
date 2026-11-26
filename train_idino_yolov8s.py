# -*- coding: utf-8 -*-
"""YOLOv8s under the I-DINO (Electronics 2024, 13, 4419) protocol, as an
INDEPENDENT second anchor on the same Indoor-COCO dataset.

Protocol (I-DINO paper):
  - single 8:2 split: 9041 train / 2261 test  (= our indoor_coco_fitted main split)
  - COCO-pretrained yolov8s.pt (I-DINO's experimental convention is pretrained
    weights + short-ish training; 80.9 mAP50 is unreachable from scratch)
  - 100 epochs, ultralytics default hyperparameters (I-DINO gives no YOLOv8
    hyperparameter details), imgsz 640, seed 0, no early stop

Target (I-DINO table 1, YOLOv8s):
  mAP@0.5:0.95 60.7  mAP@0.5 80.9  mAP@0.5:0.75 66.1

Usage:
    python train_idino_yolov8s.py --data datasets/indoor_coo_fitted
    python train_idino_yolov8s.py --summarize
"""
import argparse
from pathlib import Path

IDINO_TABLE1 = {"mAP50-95": 60.7, "mAP50": 80.9, "mAP75": 66.1}
COLS = {"mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)",
        "mAP50-95": "metrics/mAP50-95(B)"}


def train(data_root: Path, device: str, workers: int, project: str, epochs: int):
    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")  # COCO-pretrained; ultralytics rebuilds the head for nc=8
    model.train(
        data=str(data_root / "indoor_coco_fitted.yaml"),  # 8:2 main split, val = 2261
        epochs=epochs,
        imgsz=640,
        patience=0,          # fixed budget, no early stop
        seed=0,
        device=device, workers=workers,
        project=str(Path(project).resolve()), name="yolov8s_8_2",
        exist_ok=True, verbose=True,
    )


def summarize(project: Path):
    import csv

    p = project / "yolov8s_8_2" / "results.csv"
    if not p.exists():
        print(f"no results at {p}")
        return
    last = list(csv.DictReader(p.open()))[-1]
    print(f"\n{'metric':<12}{'ours':<10}{'I-DINO table 1':<16}diff")
    for k in ["mAP50", "mAP75", "mAP50-95"]:
        v = float(last[COLS[k]]) * 100
        print(f"{k:<12}{v:7.2f}    {IDINO_TABLE1[k]:7.2f}       {v - IDINO_TABLE1[k]:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--data", default="datasets/indoor_coco_fitted")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/idino")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    if args.summarize:
        summarize(Path(args.project))
        return
    train(Path(args.data), args.device, args.workers, args.project, args.epochs)


if __name__ == "__main__":
    main()
