"""PK two-stage prior-knowledge training (inherited from IOVRM, JKSU-CIS 2026).

Stage 1: train on the large indoor prior set (COCO-indoor) to obtain prior
         knowledge weights.
Stage 2: fine-tune on the target domain (e.g. SUN RGB-D / AVD) with a smaller
         lr, using the stage-1 weights as part of the initialization.

Example:
    python train_pk.py --model cfg/yolo11n-rcr.yaml \
        --stage1-data datasets/coco_indoor/coco_indoor.yaml \
        --stage2-data datasets/sunrgbd/sunrgbd.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules

register_rcr_modules()

from rcr.trainer import RCRTrainer  # noqa: E402


def run_stage(model, data, epochs, lr0, project, name, batch, imgsz, device):
    trainer = RCRTrainer(
        overrides=dict(
            model=model,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            lr0=lr0,
            patience=50,
            project=project,
            name=name,
            amp=True,
            exist_ok=True,
        )
    )
    trainer.train()
    best = Path(trainer.save_dir) / "weights" / "best.pt"
    return trainer, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cfg/yolo11n-rcr.yaml")
    ap.add_argument("--stage1-data", required=True, help="prior set, e.g. COCO-indoor")
    ap.add_argument("--stage2-data", required=True, help="target set, e.g. SUN RGB-D")
    ap.add_argument("--stage1-weights", default="", help="skip stage 1 if given")
    ap.add_argument("--epochs1", type=int, default=150)
    ap.add_argument("--epochs2", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/rcr-pk")
    args = ap.parse_args()

    if args.stage1_weights:
        w1 = Path(args.stage1_weights)
    else:
        _, w1 = run_stage(args.model, args.stage1_data, args.epochs1, 0.01,
                          args.project, "stage1-prior", args.batch, args.imgsz, args.device)
    print(f"[PK] stage-1 prior weights: {w1}")

    trainer2, _ = run_stage(str(w1), args.stage2_data, args.epochs2, 0.001,
                            args.project, "stage2-target", args.batch, args.imgsz, args.device)
    m = trainer2.metrics or {}
    map50 = float(m.get("metrics/mAP50(B)", float("nan")))
    map5095 = float(m.get("metrics/mAP50-95(B)", float("nan")))
    print(f"[PK] target mAP50={map50:.4f} mAP50-95={map5095:.4f}")


if __name__ == "__main__":
    main()
