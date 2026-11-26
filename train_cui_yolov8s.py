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


def train_fold(fold: int, data_root: Path, device: str, workers: int, project: str,
               preset: str, tag: str):
    from ultralytics import YOLO

    model = YOLO("yolov8s.yaml")  # from scratch (Cui sec 3.1 (ii))
    common = dict(
        data=str(data_root / "folds" / f"cui_fold{fold}.yaml"),
        epochs=100,          # Cui (iii)
        imgsz=640,           # Cui (iii)
        patience=0,          # fixed 100 epochs, no early stop
        seed=fold,           # distinct init per fold (Cui (iv): no fixed seed)
        pretrained=False,    # Cui (ii): strictly from scratch
        device=device, workers=workers,
        project=str(Path(project).resolve()), name=f"{tag}_fold{fold}",
        exist_ok=True, verbose=True,
    )
    if preset == "cui":
        # verbatim Cui (iii): "ultralytics RT-DETR defaults (AdamW, 1e-4 cosine,
        # wd 5e-4, batch 4, 640, 100 epochs)" -- measured: 5-fold mAP50 49.1,
        # 15 pts below table 6, so this reading is likely wrong for YOLOv8s
        model.train(optimizer="AdamW", lr0=1e-4, weight_decay=5e-4, cos_lr=True,
                    batch=4, **common)
    elif preset == "yolo":
        # alternative reading: YOLOv8s used its OWN ultralytics defaults
        # (auto optimizer -> SGD lr0 0.01 at this dataset size, batch 16,
        # linear lr decay) -- hypothesis under verification
        model.train(**common)
    else:
        raise ValueError(preset)


def fold_csv(fold: int, project: Path, tag: str):
    """results.csv of a fold, tolerating ultralytics' run-dir re-rooting."""
    for base in (project, Path("runs/detect") / project):
        p = base / f"{tag}_fold{fold}" / "results.csv"
        if p.exists():
            return p
    return None


def summarize(project: Path, tag: str):
    import csv
    import statistics as st

    rows = {}
    for fold in range(5):
        p = fold_csv(fold, project, tag)
        if p is None:
            print(f"[warn] results.csv for {tag} fold {fold} not found (checked {project} and runs/detect/{project})")
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
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/cui")
    ap.add_argument("--preset", choices=["cui", "yolo"], default="cui",
                    help="cui: verbatim RT-DETR-style hp (measured 15 pts low); "
                         "yolo: ultralytics YOLOv8 own defaults (hypothesis)")
    ap.add_argument("--tag", default=None, help="run-name prefix (default: yolov8s | yd_yolov8s)")
    args = ap.parse_args()
    tag = args.tag or ("yolov8s" if args.preset == "cui" else "yd_yolov8s")

    if args.summarize:
        summarize(Path(args.project), tag)
        return
    if args.fold not in range(5):
        ap.error("give --fold 0..4 or --summarize")
    train_fold(args.fold, Path(args.data), args.device, args.workers, args.project,
               args.preset, tag)


if __name__ == "__main__":
    main()
