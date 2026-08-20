"""Single-stage training entry for RCR-YOLO variants and baselines.

Examples:
    # full RCR-YOLO on COCO-indoor
    python train.py --model cfg/yolo11n-rcr.yaml --data datasets/coco_indoor/coco_indoor.yaml
    # baseline
    python train.py --model yolo11n.yaml --data ... --name baseline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402

from rcr.trainer import RCRTrainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cfg/yolo11n-rcr.yaml", help="model yaml or .pt weights")
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--project", default="runs/rcr")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    name = args.name or Path(args.model).stem
    trainer = RCRTrainer(
        overrides=dict(
            model=args.model,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr0,
            project=args.project,
            name=name,
            amp=True,
            mosaic=1.0,
            close_mosaic=15,
            resume=args.resume,
            exist_ok=True,
        )
    )
    trainer.train()
    # DetectionTrainer has no .val(); validate best.pt standalone and report.
    metrics = YOLO(str(trainer.best)).val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=f"{name}-final",
        exist_ok=True,
    )
    print(f"[{name}] mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
