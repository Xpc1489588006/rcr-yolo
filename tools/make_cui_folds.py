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
import os
import subprocess
from pathlib import Path

NAMES = ["toilet", "tv", "refrigerator", "microwave", "sink", "chair", "couch", "bed"]


def _link_dir(link: Path, target: Path):
    """Per-fold images/labels dir so each fold gets its own labels.cache.

    Concurrent folds sharing one labels dir race on the SAME cache file
    (train/labels.cache): a half-written cache raises EOFError on np.load,
    which ultralytics does NOT catch -> job dies. Junction on Windows,
    symlink on Linux.
    """
    if link.is_symlink() and not link.exists():
        link.unlink()  # replace a broken (dangling) symlink from earlier runs
    if link.exists():
        return
    # absolute target is CRITICAL: a relative symlink target resolves
    # against the LINK's parent dir on Linux (folds/foldX_data/), not the
    # CWD -> every image lookup fails with ENOENT
    tgt = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(tgt)],
                       check=True, capture_output=True)
    else:
        link.symlink_to(tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="indoor_coco_fitted dataset root")
    args = ap.parse_args()

    root = Path(args.root)
    folds = json.loads((root / "folds.json").read_text())
    out = root / "folds"
    out.mkdir(exist_ok=True)
    root_abs = root.resolve().as_posix()  # txt lines MUST be absolute: ultralytics
    # only resolves lines starting with './' or '/', relative lines are used
    # as-is against the CWD and labels are then not found (train/labels.cache bug)

    n_va = 0
    val_union, val_all = set(), []
    for k in sorted(folds):  # fold0..fold4
        i = int(k.replace("fold", ""))
        tr = sorted(folds[k]["train"])
        va = sorted(folds[k]["val"])
        # per-fold data dir (junction/symlink) -> cache becomes
        # folds/fold{i}_data/labels.cache, disjoint across folds
        _link_dir(out / f"fold{i}_data" / "images", root / "train" / "images")
        _link_dir(out / f"fold{i}_data" / "labels", root / "train" / "labels")
        fd_abs = (out / f"fold{i}_data").resolve().as_posix()
        (out / f"fold{i}_train.txt").write_text("\n".join(f"{fd_abs}/images/{f}" for f in tr) + "\n")
        (out / f"fold{i}_val.txt").write_text("\n".join(f"{fd_abs}/images/{f}" for f in va) + "\n")
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
