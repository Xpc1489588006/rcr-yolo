# -*- coding: utf-8 -*-
"""Generate per-fold data yamls for the Cui-2026 5-fold protocol.

Usage (local, verify only):
    python tools/make_cui_folds.py --root datasets/indoor_coco_fitted

Usage (server, after dataset is in place):
    python tools/make_cui_folds.py --root /data/home/zhaozhanshan/rcr/rcr_yolo/datasets/indoor_coco_fitted

Reads folds.json (built by data/build_indoor_coco_fitted.py, stratified
KFold-style proxy on the 9041 training images) and writes for each fold:
    folds/fold{i}_train.txt   image list relative to <root>
    folds/fold{i}_val.txt
    folds/cui_fold{i}.yaml    ultralytics data yaml (path = --root)
"""
import argparse
import json
from pathlib import Path

NAMES = ["toilet", "tv", "refrigerator", "microwave", "sink", "chair", "couch", "bed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="indoor_coco_fitted dataset root")
    args = ap.parse_args()

    root = Path(args.root)
    folds = json.loads((root / "folds.json").read_text())
    out = root / "folds"
    out.mkdir(exist_ok=True)

    n_va = 0
    val_union, val_all = set(), []
    for k in sorted(folds):  # fold0..fold4
        i = int(k.replace("fold", ""))
        tr = sorted(folds[k]["train"])
        va = sorted(folds[k]["val"])
        (out / f"fold{i}_train.txt").write_text("\n".join(f"train/images/{f}" for f in tr) + "\n")
        (out / f"fold{i}_val.txt").write_text("\n".join(f"train/images/{f}" for f in va) + "\n")
        yml = (out / f"cui_fold{i}.yaml")
        yml.write_text(
            f"# Cui-2026 5-fold protocol, fold {i}: {len(tr)} train / {len(va)} val images\n"
            f"path: {root.resolve()}\n"
            f"train: folds/fold{i}_train.txt\n"
            f"val: folds/fold{i}_val.txt\n"
            f"nc: 8\n"
            f"names: {NAMES}\n"
        )
        print(f"fold{i}: train={len(tr)}  val={len(va)}")
        n_va += len(va)
        val_union.update(va)
        val_all.extend(va)
    # each fold's val set must be disjoint and their union must be all 9041
    print(f"total val: {n_va} (union {len(val_union)}), must equal 9041")
    assert n_va == 9041 and len(val_union) == 9041, "fold val sets must partition the 9041 images"


if __name__ == "__main__":
    main()
