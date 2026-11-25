# -*- coding: utf-8 -*-
"""Train YOLOv8s under the exact Cui-2026 (MST 37 415406) protocol.

Protocol mapping (Cui sec. 3.1, quoted):
  (i)   5-fold stratified KFold on the 9041 Indoor-COCO training images
        (~7233 train / ~1808 val per fold)  -> data/folds/cui_fold{i}.yaml
  (ii)  "All detectors are trained from scratch without any external
        pre-training"                       -> model = yolov8s.yaml (no .pt)
  (iii) "all hyperparameters follow the ultralytics RT-DETR defaults
        (AdamW optimizer, initial learning rate 1e-4 with cosine schedule,
        weight decay 5e-4, batch size 4, input resolution 640x640, and
        100 epochs)"
  (iv)  "No random seed is fixed manually; each fold is trained once from
        a fresh random initialization"      -> seed = fold index (distinct
        per fold; fixed seeds would be MORE deterministic than the paper)

Target (Cui table 6, YOLOv8s 5-fold mean +/- std):
  Precision 71.59+/-0.75  Recall 57.83+/-0.11  mAP@0.5 64.36+/-0.49
  mAP@0.75 49.93+/-0.66   mAP@0.5:0.95 45.13+/-0.56

Usage:
    python train_cui_yolov8s.py --fold 0 --data datasets/indoor_coco_fitted
    python train_cui_yolov8s.py --summarize          # after all 5 folds
"""
import argparse
from pathlib import Path

# paper anchor for the summary table
CUI_TABLE6 = {
    "P": (71.59, 0.75), "R": (57.83, 0.11), "mAP50": (64.36, 0.49),
    "mAP75": (49.93, 0.66), "mAP50-95": (45.13, 0.56),
}
COLS = {"P": "metrics/precision(B)", "R": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)", "mAP75": "metrics/mAP75(B)",
        "mAP50-95": "metrics/mAP50-95(B)"}


def train_fold(fold: int, data_root: Path, device: str, workers: int, project: str):
    from ultralytics import YOLO

    model = YOLO("yolov8s.yaml")  # from scratch (Cui sec 3.1 (ii))
    model.train(
        data=str(data_root / "folds" / f"cui_fold{fold}.yaml"),
        epochs=100,          # Cui (iii)
        batch=4,             # Cui (iii)
        imgsz=640,           # Cui (iii)
        optimizer="AdamW",   # Cui (iii)
        lr0=1e-4,            # Cui (iii)
        weight_decay=5e-4,   # Cui (iii)
        cos_lr=True,         # cosine schedule, lrf stays at ultralytics default
        patience=0,          # fixed 100 epochs, no early stop
        seed=fold,           # distinct init per fold (Cui (iv): no fixed seed)
        pretrained=False,    # Cui (ii): strictly from scratch
        device=device, workers=workers,
        # absolute project: relative paths get re-rooted under runs/detect/ in
        # ultralytics 8.4.x, breaking the summarize lookup
        project=str(Path(project).resolve()), name=f"yolov8s_fold{fold}",
        exist_ok=True, verbose=True,
    )


def fold_csv(fold: int, project: Path):
    """results.csv of a fold, tolerating ultralytics' run-dir re-rooting."""
    for base in (project, Path("runs/detect") / project):
        p = base / f"yolov8s_fold{fold}" / "results.csv"
        if p.exists():
            return p
    return None


def summarize(project: Path):
    import csv
    import statistics as st

    rows = {}
    for fold in range(5):
        p = fold_csv(fold, project)
        if p is None:
            print(f"[warn] results.csv for fold {fold} not found (checked {project} and runs/detect/{project})")
            continue
        last = list(csv.DictReader(p.open()))[-1]  # final epoch = fixed protocol
        rows[fold] = {k: float(last[c]) * 100 for k, c in COLS.items()}

    if not rows:
        print("no finished folds; nothing to summarize")
        return
    if len(rows) < 5:
        print(f"only {len(rows)}/5 folds available; partial summary below")
    print(f"\n{'metric':<10}{'ours (mean±std, n=%d)' % len(rows):<24}{'Cui table 6':<20}diff(mean)")
    for k in ["P", "R", "mAP50", "mAP75", "mAP50-95"]:
        vals = [rows[f][k] for f in sorted(rows)]
        m, s = st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)
        cm, cs = CUI_TABLE6[k]
        print(f"{k:<10}{m:7.2f} ± {s:5.2f}{'':<9}{cm:7.2f} ± {cs:5.2f}   {m - cm:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1, help="fold index 0-4 to train")
    ap.add_argument("--summarize", action="store_true", help="aggregate 5-fold results")
    ap.add_argument("--data", default="datasets/indoor_coco_fitted")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/cui")
    args = ap.parse_args()

    if args.summarize:
        summarize(Path(args.project))
        return
    if args.fold not in range(5):
        ap.error("give --fold 0..4 or --summarize")
    train_fold(args.fold, Path(args.data), args.device, args.workers, args.project)


if __name__ == "__main__":
    main()
