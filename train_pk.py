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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules

register_rcr_modules()

from rcr.trainer import RCRTrainer  # noqa: E402


def run_stage(model, data, epochs, lr0, project, name, batch, imgsz, device, workers, seed=0):
    trainer = RCRTrainer(
        overrides=dict(
            model=model,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            lr0=lr0,
            workers=workers,
            seed=seed,
            patience=50,
            project=project,
            name=name,
            amp=True,
            deterministic=False,  # fully-deterministic mode is ~5x slower for large-spatial convs
            exist_ok=True,
        )
    )
    trainer.train()
    best = Path(trainer.save_dir) / "weights" / "best.pt"
    return trainer, best


def main():
    # fixed 640 input + explicit batch -> safe to autotune cuDNN algos (~15-20% faster)
    torch.backends.cudnn.benchmark = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cfg/yolo11n-rcr.yaml")
    ap.add_argument("--stage1-data", required=True, help="prior set, e.g. COCO-indoor")
    ap.add_argument("--stage2-data", required=True, help="target set, e.g. SUN RGB-D")
    ap.add_argument("--stage1-weights", default="", help="skip stage 1 if given")
    ap.add_argument("--init", choices=["pk", "coco", "scratch"], default="pk",
                    help="pk: stage-1 prior weights (default); coco: partial transfer "
                         "from generic COCO weights; scratch: no prior, full budget")
    ap.add_argument("--coco-weights", default="yolo11n.pt",
                    help="generic COCO-pretrained weights for --init coco")
    ap.add_argument("--epochs-scratch", type=int, default=300,
                    help="epochs for --init scratch (t300 protocol)")
    ap.add_argument("--lr-scratch", type=float, default=0.01,
                    help="lr0 for --init scratch")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; stage-1 prior stays fixed, seed varies the "
                         "stage-2 fine-tune (aug order, head re-init)")
    ap.add_argument("--epochs1", type=int, default=150)
    ap.add_argument("--epochs2", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers; 4 (not ultralytics' 8) matches the cluster "
                         "convention for shard-submitted jobs")
    ap.add_argument("--project", default="runs/rcr-pk")
    ap.add_argument("--name", default="",
                    help="run-name tag (stage dirs become <tag>-stage1-prior / <tag>-stage2-target); "
                         "defaults to model stem minus the 'yolo11n-' prefix. REQUIRED when "
                         "multiple PK jobs share one project dir, else they overwrite each other.")
    args = ap.parse_args()

    tag = args.name or Path(args.model).stem
    if tag.startswith("yolo11n-"):
        tag = tag[len("yolo11n-"):]  # yolo11n-rcr -> rcr; bare yolo11n stays intact

    if args.init == "scratch":
        # control line: no prior, full t300-protocol budget on the target set
        print(f"[PK] scratch line: {args.model} on {args.stage2_data} "
              f"(epochs={args.epochs_scratch}, lr0={args.lr_scratch}, seed={args.seed})")
        run_stage(args.model, args.stage2_data, args.epochs_scratch, args.lr_scratch,
                  args.project, f"{tag}-scratch-target", args.batch, args.imgsz,
                  args.device, args.workers, args.seed)
        return

    if args.init == "coco":
        # control line: same stage-2 protocol as pk, prior source swapped for
        # generic COCO weights. Non-RCR layers transfer, RCR modules stay random.
        from ultralytics import YOLO
        print(f"[PK] coco line: {args.model} + partial transfer from {args.coco_weights} "
              f"(seed={args.seed})")
        _m = YOLO(args.model)
        _m.load(args.coco_weights)
        _init = Path(args.project) / f"{tag}-coco-init.pt"
        _init.parent.mkdir(parents=True, exist_ok=True)
        _m.save(_init)
        print(f"[PK] partial-transfer init saved: {_init}")
        run_stage(str(_init), args.stage2_data, args.epochs2, 0.001,
                  args.project, f"{tag}-coco-target", args.batch, args.imgsz,
                  args.device, args.workers, args.seed)
        return

    if args.stage1_weights:
        w1 = Path(args.stage1_weights)
    else:
        _, w1 = run_stage(args.model, args.stage1_data, args.epochs1, 0.01,
                          args.project, f"{tag}-stage1-prior", args.batch, args.imgsz, args.device,
                          args.workers)
    print(f"[PK] stage-1 prior weights: {w1}")

    trainer2, _ = run_stage(str(w1), args.stage2_data, args.epochs2, 0.001,
                            args.project, f"{tag}-stage2-target", args.batch, args.imgsz, args.device,
                            args.workers, args.seed)
    m = trainer2.metrics or {}
    map50 = float(m.get("metrics/mAP50(B)", float("nan")))
    map5095 = float(m.get("metrics/mAP50-95(B)", float("nan")))
    print(f"[PK] target mAP50={map50:.4f} mAP50-95={map5095:.4f}")


if __name__ == "__main__":
    main()
