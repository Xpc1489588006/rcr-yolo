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

import torch

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
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (multi-seed runs)")
    ap.add_argument("--project", default="runs/rcr")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="force fully-deterministic ops (very slow for large-spatial convs; "
        "seeds stay fixed without this, runs are reproducible in distribution)",
    )
    args = ap.parse_args()
    print("argv:", " ".join(sys.argv), flush=True)

    name = args.name or Path(args.model).stem
    if not args.deterministic:
        # fixed 640 input + explicit batch -> safe to autotune cuDNN algos (~15-20% faster)
        torch.backends.cudnn.benchmark = True
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
            # 2026-09-03 事故根因：resume 传布尔值时，ultralytics check_resume
            # 会改用 get_latest_run() 全局扫描最近修改的 last.pt，导致并发作业
            # 全部从同一个检查点续训、交叉写入同一目录。改传路径字符串，
            # 强制从指定检查点恢复。
            resume=args.model if args.resume else False,
            deterministic=args.deterministic,
            seed=args.seed,
            exist_ok=True,
        )
    )
    # 身份护栏（2026-09-03 事故：4 个作业交叉写入同一目录）。
    # resume 合并 checkpoint 的 train_args 后，若生效的 name/save_dir
    # 与本次运行身份不符，立即中止，防止权重被写到错误的目录。
    save_dir = str(getattr(trainer, "save_dir", "")).replace("\\", "/")
    if trainer.args.name != name or not save_dir.endswith(f"/{name}"):
        print(
            f"[{name}] IDENTITY MISMATCH after trainer init: "
            f"args.name={trainer.args.name} save_dir={save_dir}; aborting",
            flush=True,
        )
        sys.exit(3)
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
