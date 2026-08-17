"""Best-effort converters for the cross-domain indoor test sets.

Sub-commands:
    sunrgbd  SUN RGB-D (official toolbox .mat annotations, requires scipy)
    avd      Active Vision Dataset (per-scene 2D annotation txt files)
    tut      TUT Indoor object set (generic txt parser, --fmt selectable)

All converters emit an Ultralytics YOLO dataset (images/ labels/ + data yaml).
These are intentionally defensive: if the on-disk layout differs, the script
prints what it expected instead of failing silently.
"""

import argparse
import shutil
from pathlib import Path

import cv2

SUNRGBD_CLASS10 = ["bed", "book", "bottle", "chair", "cup", "desk", "door", "dresser", "monitor", "nightstand"]
SUNRGBD_CLASS37 = SUNRGBD_CLASS10 + [
    "bathtub", "blinds", "bookshelf", "box", "cabinet", "cabinetshelves", "clothes",
    "counter", "dresser", "fridge", "keyboard", "lamp", "laptop", "mirror", "mouse",
    "night stand", "paper", "piano", "picture", "pillow", "plant", "plate", "rack",
    "shelf", "shoes", "sink", "sofa", "table", "television", "toilet", "towel", "tv",
    "wardrobe", "window",
]


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
    names = SUNRGBD_CLASS10 if args.classes10 else SUNRGBD_CLASS37
    entries = {}
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
        entries.setdefault(str(img_path), []).append((names.index(cls), cx, cy, bw, bh))
    items = [(Path(k), v) for k, v in entries.items()]
    n = len(items)
    write_set(Path(args.out), items[: int(n * 0.8)], names, "train")
    write_set(Path(args.out), items[int(n * 0.8):], names, "val")


def cmd_avd(args):
    root = Path(args.root)
    names = args.names.split(",") if args.names else None
    entries = []
    for scene in sorted(root.rglob("*")):
        if not scene.is_dir():
            continue
        rgb = scene / "rgb"
        ann_dirs = [d for d in scene.iterdir() if d.is_dir() and "annot" in d.name.lower()] if scene.exists() else []
        if not rgb.exists() or not ann_dirs:
            continue
        for txt in ann_dirs[0].glob("*.txt"):
            img = rgb / (txt.stem + ".jpg")
            if not img.exists():
                img = rgb / (txt.stem + ".png")
            if not img.exists():
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
                cls = p[0]
                if not cls.isdigit():
                    if names is None:
                        continue
                    if cls not in names:
                        continue
                    cid = names.index(cls)
                else:
                    cid = int(cls)
                x1, y1, x2, y2 = map(float, p[1:5])
                boxes.append((cid, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            if boxes:
                entries.append((img, boxes))
    if not entries:
        raise SystemExit(f"no AVD scenes parsed under {root}; expected <scene>/rgb/*.jpg + <scene>/*annot*/*.txt")
    names = names or ["class%d" % i for i in range(max(b[0] for _, e in entries for b in e) + 1)]
    n = len(entries)
    write_set(Path(args.out), entries[: int(n * 0.8)], names, "train")
    write_set(Path(args.out), entries[int(n * 0.8):], names, "val")


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
