# -*- coding: utf-8 -*-
"""Train RCR-YOLO variants (e.g. ab4: yolo11n-rcrfb) under the exact Cui-2026
5-fold protocol, so results are directly comparable with train_cui_yolov8s.py.

Same protocol as the YOLOv8s run (Cui sec. 3.1):
  (i)   5-fold stratified KFold on the 9041 Indoor-COCO training images
  (ii)  from scratch (model yaml, no .pt)
  (iii) AdamW, lr 1e-4 cosine, wd 5e-4, batch 4, 640x640, 100 epochs
  (iv)  seed = fold index (distinct init per fold)
RCR extras: register_rcr_modules() before ultralytics import, RCRTrainer adds
the ORB-In reconstruction loss during training (zero inference overhead).
RCR_RECON_WEIGHT env var controls the recon weight (default 1.0).

Usage:
    python train_cui_rcr.py --fold 0 --data datasets/indoor_coco_fitted
    python train_cui_rcr.py --summarize          # 5-fold stats + yolov8s side-by-side
    python train_cui_rcr.py --fold 0 --model cfg/yolo11n-rcr.yaml --tag rcr
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from rcr.trainer import RCRTrainer  # noqa: E402

COLS = {"P": "metrics/precision(B)", "R": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)",
        "mAP50-95": "metrics/mAP50-95(B)"}


def train_fold(fold: int, data_root: Path, model: str, tag: str, device: str,
               workers: int, project: str):
    RCRTrainer(
        overrides=dict(
            model=model,             # yaml -> from scratch (Cui (ii))
            data=str(data_root / "folds" / f"cui_fold{fold}.yaml"),
            epochs=100,              # Cui (iii)
            batch=4,                 # Cui (iii)
            imgsz=640,               # Cui (iii)
            optimizer="AdamW",       # Cui (iii)
            lr0=1e-4,                # Cui (iii)
            weight_decay=5e-4,       # Cui (iii)
            cos_lr=True,             # cosine schedule (ultralytics default lrf)
            patience=0,              # fixed 100 epochs, no early stop
            seed=fold,               # distinct init per fold (Cui (iv))
            pretrained=False,        # Cui (ii): strictly from scratch
            device=device, workers=workers, amp=True,
            project=str(Path(project).resolve()),  # absolute: avoids re-rooting
            name=f"{tag}_fold{fold}",
            exist_ok=True,
        )
    ).train()


def fold_csv(fold: int, project: Path, tag: str):
    for base in (project, Path("runs/detect") / project):
        p = base / f"{tag}_fold{fold}" / "results.csv"
        if p.exists():
            return p
    return None


def summarize(project: Path, tag: str):
    import csv
    import statistics as st

    def read_run(p):
        last = list(csv.DictReader(p.open()))[-1]  # final epoch = fixed protocol
        return {k: float(last[c]) * 100 for k, c in COLS.items()}

    rows = {}
    for fold in range(5):
        p = fold_csv(fold, project, tag)
        if p is None:
            print(f"[warn] results.csv for {tag} fold {fold} not found")
            continue
        rows[fold] = read_run(p)
    if not rows:
        print("no finished folds; nothing to summarize")
        return

    # optional side-by-side with the yolov8s baseline from train_cui_yolov8s.py
    v8_rows = {}
    for fold in range(5):
        p = fold_csv(fold, Path("runs/cui"), "yolov8s")
        if p is not None:
            v8_rows[fold] = read_run(p)

    print(f"\n{'metric':<10}{tag + ' (n=%d)' % len(rows):<20}", end="")
    if v8_rows:
        print(f"{'yolov8s (n=%d)' % len(v8_rows):<20}")
    else:
        print()
    for k in ["P", "R", "mAP50", "mAP75", "mAP50-95"]:
        vals = [rows[f][k] for f in sorted(rows)]
        m, s = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
        line = f"{k:<10}{m:7.2f} ± {s:5.2f}    "
        if v8_rows:
            v = [v8_rows[f][k] for f in sorted(v8_rows)]
            vm, vs = st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)
            line += f"{vm:7.2f} ± {vs:5.2f}   diff {m - vm:+.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1, help="fold index 0-4 to train")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--data", default="datasets/indoor_coco_fitted")
    ap.add_argument("--model", default="cfg/yolo11n-rcrfb.yaml", help="ab4 variant")
    ap.add_argument("--tag", default="rcrfb", help="run-name prefix")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/cui_rcrfb")
    args = ap.parse_args()

    if args.summarize:
        summarize(Path(args.project), args.tag)
        return
    if args.fold not in range(5):
        ap.error("give --fold 0..4 or --summarize")
    train_fold(args.fold, Path(args.data), args.model, args.tag,
               args.device, args.workers, args.project)


if __name__ == "__main__":
    main()
