"""Best-effort converters for the cross-domain indoor test sets.

Sub-commands:
    sunrgbd  SUN RGB-D (official toolbox .mat annotations, requires scipy)
    avd      Active Vision Dataset (per-scene 2D annotation txt files)
    tut      TUT Indoor object set (generic txt parser, --fmt selectable)

Split hygiene (fixed 2026-08-31, per 总结分析.md §5):
* sunrgbd expects the OFFICIAL traintestsplit.txt (--split-file); without it
  the dataset is written under an ``_orderSplit`` suffix and must not be
  used for paper-level generalization claims.
* avd isolates train/val at the SCENE level (whole scenes never straddle the
  split) and reads every annotation directory of a scene, not just the first.
* The former hardcoded 37-class list contained duplicates (``dresser`` twice,
  ``tv``/``television``, ``nightstand``/``night stand``); the vocabulary is
  now built from the annotations themselves in first-appearance order.

All converters emit an Ultralytics YOLO dataset (images/ labels/ + data yaml).
These are intentionally defensive: if the on-disk layout differs, the script
prints what it expected instead of failing silently.
"""

import argparse
import shutil
from pathlib import Path

import cv2

SUNRGBD_CLASS10 = ["bed", "book", "bottle", "chair", "cup", "desk", "door", "dresser", "monitor", "nightstand"]


def build_vocab(records):
    """Unique class names in first-appearance order, taken from the data.

    Replaces the former hardcoded 37-class list, which contained duplicates
    (``dresser`` twice, ``tv``/``television``, ``nightstand``/``night stand``)
    and silently collapsed distinct categories onto one index.
    """
    names = []
    for rec in records:
        c = str(rec.classname).strip()
        if c and c not in names:
            names.append(c)
    return names


def _norm_path(s):
    return str(s).strip().replace("\\", "/").rsplit(".", 1)[0].lower()


def load_split_file(path):
    """Normalized path tails (extension stripped) from an official split file."""
    return {_norm_path(l) for l in Path(path).read_text().splitlines() if l.strip()}


def in_split(rgbname, split_set):
    """Match by path tail: basenames repeat across sensors (img-000001.jpg),
    so compare the last path components, progressively widening."""
    rel = _norm_path(rgbname)
    parts = rel.split("/")
    for depth in (4, 3, 2):
        if len(parts) >= depth:
            tail = "/".join(parts[-depth:])
            if any(s.endswith(tail) for s in split_set):
                return True
    return False


def write_set(out: Path, entries, names, split):
    """entries: list of (src_img_path, [(cls, cx, cy, w, h) normalized])."""
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "labels").mkdir(parents=True, exist_ok=True)
    for src, boxes in entries:
        dst = out / split / "images" / Path(src).name
        if not dst.exists():
            shutil.copy2(src, dst)
        with open(out / split / "labels" / (Path(src).stem + ".txt"), "w") as f:
            for b in boxes:
                f.write("%d %.6f %.6f %.6f %.6f\n" % b)
    yaml_txt = f"path: {out.resolve().as_posix()}\ntrain: train/images\nval: val/images\nnc: {len(names)}\nnames: {names}\n"
    (out / f"{out.name}.yaml").write_text(yaml_txt)
    print(f"{split}: {len(entries)} images -> {out}")


def cmd_sunrgbd(args):
    try:
        from scipy.io import loadmat
    except ImportError:
        raise SystemExit("scipy is required: pip install scipy")
    root = Path(args.root)
    mat = loadmat(str(root / "SUNRGBD2Dbox.mat"), squeeze_me=True, struct_as_record=False)
    boxes2d = mat["SUNRGBD2Dbox"]
    if hasattr(boxes2d, "flatten"):
        boxes2d = boxes2d.flatten()
    names = SUNRGBD_CLASS10 if args.classes10 else build_vocab(boxes2d)
    split_set = load_split_file(args.split_file) if args.split_file else None
    train, test = {}, {}
    for rec in boxes2d:
        img_rel = str(rec.rgbname).replace("\\", "/")
        cls = str(rec.classname).strip()
        if cls not in names:
            continue
        x1, y1, x2, y2 = [float(v) for v in rec.bbox[:4]]
        img_path = root / "images" / Path(img_rel).name
        if not img_path.exists():
            for cand in (root / img_rel, root / "SUNRGBD" / img_rel):
                if cand.exists():
                    img_path = cand
                    break
            else:
                continue
        h, w = (float(rec.bbox[4]), float(rec.bbox[5])) if rec.bbox.size > 4 else (None, None)
        if not h:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]
        cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            continue
        key = str(img_path)
        if split_set is not None:
            target = train if in_split(img_rel, split_set) else test
        else:
            target = train  # staging; order-based partition applied below
        target.setdefault(key, []).append((names.index(cls), cx, cy, bw, bh))
    if split_set is not None:
        print(f"official split: {len(train)} train / {len(test)} test images")
        out = Path(args.out)
        write_set(out, sorted(train.items()), names, "train")
        write_set(out, sorted(test.items()), names, "val")
        return
    # No official split file: fall back to order-based partition, clearly flagged
    print("WARNING: --split-file not given; falling back to ORDER-BASED 80/20 "
          "split. Output is suffixed '_orderSplit' and must NOT be used for "
          "paper-level generalization claims. Pass the official "
          "SUNRGBDtoolbox/traintestsplit.txt via --split-file.")
    items = sorted(train.items())
    n = len(items)
    out = Path(str(args.out).rstrip("/") + "_orderSplit")
    write_set(out, items[: int(n * 0.8)], names, "train")
    write_set(out, items[int(n * 0.8):], names, "val")


def cmd_avd(args):
    root = Path(args.root)
    names = args.names.split(",") if args.names else None
    by_scene = {}  # scene_id -> {abs_img_path: boxes}; ALL annotation dirs read
    for scene in sorted(root.rglob("*")):
        if not scene.is_dir():
            continue
        rgb = scene / "rgb"
        ann_dirs = [d for d in scene.iterdir() if d.is_dir() and "annot" in d.name.lower()]
        if not rgb.exists() or not ann_dirs:
            continue
        scene_imgs = {}
        for ann_dir in ann_dirs:
            for txt in ann_dir.glob("*.txt"):
                img = rgb / (txt.stem + ".jpg")
                if not img.exists():
                    img = rgb / (txt.stem + ".png")
                if not img.exists():
                    continue
                key = str(img.resolve())
                if key not in scene_imgs:
                    im = cv2.imread(str(img))
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                    scene_imgs[key] = {"img": img, "hw": (h, w), "boxes": []}
                info = scene_imgs[key]
                h, w = info["hw"]
                for line in txt.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cls = p[0]
                    if not cls.isdigit():
                        if names is None or cls not in names:
                            continue
                        cid = names.index(cls)
                    else:
                        cid = int(cls)
                    x1, y1, x2, y2 = map(float, p[1:5])
                    box = (cid, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)
                    if box not in info["boxes"]:  # dedupe across annotation dirs
                        info["boxes"].append(box)
        by_scene[scene.name] = {k: v for k, v in scene_imgs.items() if v["boxes"]}
    entries_by_scene = {s: [(Path(k), v["boxes"]) for k, v in d.items()]
                        for s, d in by_scene.items() if d}
    total = sum(len(v) for v in entries_by_scene.values())
    if total == 0:
        raise SystemExit(f"no AVD scenes parsed under {root}; expected <scene>/rgb/*.jpg + <scene>/*annot*/*.txt")
    names = names or ["class%d" % i for i in range(max(b[0] for e in entries_by_scene.values() for _, bs in e for b in bs) + 1)]
    # SCENE-level isolation: whole scenes go entirely to train or val, so
    # near-duplicate frames never straddle the split
    scenes = sorted(entries_by_scene)
    n_train_scenes = max(1, int(len(scenes) * 0.8)) if len(scenes) > 1 else len(scenes)
    train_scenes, val_scenes = scenes[:n_train_scenes], scenes[n_train_scenes:]
    print(f"scene-level split: train={train_scenes}")
    print(f"                  val  ={val_scenes or '(none: only one scene)'}")
    train = [(p, b) for s in train_scenes for p, b in entries_by_scene[s]]
    val = [(p, b) for s in val_scenes for p, b in entries_by_scene[s]]
    out = Path(args.out)
    write_set(out, train, names, "train")
    if val:
        write_set(out, val, names, "val")
    else:
        print("WARNING: single AVD scene; val set empty, cannot evaluate generalization")


def cmd_tut(args):
    root = Path(args.root)
    names = args.names.split(",")
    entries = []
    for txt in sorted(root.rglob("*.txt")):
        img = None
        for ext in (".jpg", ".png", ".bmp"):
            c = txt.with_suffix(ext)
            if c.exists():
                img = c
                break
        if img is None:
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = []
        for line in txt.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            a, b, c, d = map(float, p[1:5])
            if args.fmt == "xyxy":
                cx, cy, bw, bh = (a + c) / 2 / w, (b + d) / 2 / h, (c - a) / w, (d - b) / h
            else:  # normalized cxcywh already
                cx, cy, bw, bh = a, b, c, d
            if bw > 0 and bh > 0:
                boxes.append((cid, cx, cy, bw, bh))
        if boxes:
            entries.append((img, boxes))
    if not entries:
        raise SystemExit(f"no TUT pairs parsed under {root}")
    print("WARNING: TUT has no official train/test split; writing an order-based "
          "80/20 partition. Prefer using TUT as an external TEST set with its "
          "published annotations instead of training on it.")
    n = len(entries)
    write_set(Path(args.out), entries[: int(n * 0.8)], names, "train")
    write_set(Path(args.out), entries[int(n * 0.8):], names, "val")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sunrgbd")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="datasets/sunrgbd")
    p.add_argument("--classes10", action="store_true")
    p.add_argument("--split-file", default="",
                   help="official SUNRGBDtoolbox/traintestsplit.txt; without it "
                        "the output is an order-based split (suffixed _orderSplit)")
    p.set_defaults(fn=cmd_sunrgbd)
    p = sub.add_parser("avd")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="datasets/avd")
    p.add_argument("--names", default="", help="comma-separated class names")
    p.set_defaults(fn=cmd_avd)
    p = sub.add_parser("tut")
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="datasets/tut")
    p.add_argument("--names", required=True)
    p.add_argument("--fmt", choices=["xyxy", "cxcywh"], default="xyxy")
    p.set_defaults(fn=cmd_tut)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
