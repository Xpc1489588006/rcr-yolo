"""Build the COCO-indoor subset (main training set, zero collection cost).

Filters COCO 2017 to ~45 indoor-relevant classes (furniture / appliances /
tableware / small household objects / person), remaps class ids and writes an
Ultralytics-ready YOLO dataset:

    out/
      images/train|val/
      labels/train|val/
      coco_indoor.yaml

Usage:
    python data/make_coco_indoor.py --coco /path/to/coco2017 --out ./datasets/coco_indoor

Modes:
    (default)   copy filtered images into out/
    --symlink   symlink filtered images into out/
    --no-links  write ONLY labels + yaml (yaml uses relative path `path: .`,
                so you can sync train2017/val2017 into out/train/images etc.
                on another machine; fully portable across machines)
"""

import argparse
import json
import shutil
from pathlib import Path

INDOOR_CLASSES = [
    "person", "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def convert_split(ann_json: Path, img_dir: Path, out: Path, id_map: dict, copy: bool):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    data = json.loads(ann_json.read_text())
    imgs = {im["id"]: im for im in data["images"]}
    boxes = {}
    n_box = 0
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            continue
        cid = a["category_id"]
        if cid not in id_map:
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        iw, ih = imgs[a["image_id"]]["width"], imgs[a["image_id"]]["height"]
        cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
        nw, nh = w / iw, h / ih
        boxes.setdefault(a["image_id"], []).append((id_map[cid], cx, cy, nw, nh))
        n_box += 1
    n_img = 0
    for img_id, img in imgs.items():
        if img_id not in boxes:
            continue  # keep only images containing indoor classes
        stem = Path(img["file_name"]).stem
        if copy:
            shutil.copy2(img_dir / img["file_name"], out / "images" / img["file_name"])
        else:
            (out / "images" / img["file_name"]).symlink_to((img_dir / img["file_name"]).resolve())
        with open(out / "labels" / f"{stem}.txt", "w") as f:
            for b in boxes[img_id]:
                f.write("%d %.6f %.6f %.6f %.6f\n" % b)
        n_img += 1
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO 2017 root (contains train2017/, val2017/, annotations/)")
    ap.add_argument("--out", default="datasets/coco_indoor")
    ap.add_argument("--copy", action="store_true", default=True, help="copy images (default)")
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--no-links", action="store_true",
                    help="write labels+yaml only; yaml is relative-path (portable)")
    args = ap.parse_args()
    coco = Path(args.coco)
    out = Path(args.out)

    train_ann = coco / "annotations" / "instances_train2017.json"
    cats = json.loads(train_ann.read_text())["categories"]
    id_map = {c["id"]: INDOOR_CLASSES.index(c["name"]) for c in cats if c["name"] in INDOOR_CLASSES}
    print(f"keeping {len(id_map)} indoor classes")

    stats = {}
    for split, ann, imgdir in (
        ("train", train_ann, coco / "train2017"),
        ("val", coco / "annotations" / "instances_val2017.json", coco / "val2017"),
    ):
        if args.no_links:
            # labels only; images are synced separately into out/<split>/images
            data = json.loads(ann.read_text())
            imgs = {im["id"]: im for im in data["images"]}
            boxes = {}
            n_box = 0
            for a in data["annotations"]:
                if a.get("iscrowd", 0) or a["category_id"] not in id_map:
                    continue
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                iw, ih = imgs[a["image_id"]]["width"], imgs[a["image_id"]]["height"]
                boxes.setdefault(a["image_id"], []).append(
                    (id_map[a["category_id"]], (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih))
                n_box += 1
            lbl_dir = out / split / "labels"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            n_img = 0
            for img_id, bl in boxes.items():
                stem = Path(imgs[img_id]["file_name"]).stem
                with open(lbl_dir / f"{stem}.txt", "w") as f:
                    for b in bl:
                        f.write("%d %.6f %.6f %.6f %.6f\n" % b)
                n_img += 1
        else:
            n_img, n_box = convert_split(ann, imgdir, out / split, id_map, not args.symlink)
        stats[split] = (n_img, n_box)
        print(f"{split}: {n_img} images, {n_box} boxes")

    path_line = "path: ." if args.no_links else f"path: {out.resolve().as_posix()}"
    yaml_txt = f"""# COCO-indoor subset for RCR-YOLO (auto-generated)
{path_line}
train: train/images
val: val/images
nc: {len(INDOOR_CLASSES)}
names: {INDOOR_CLASSES}
"""
    (out / "coco_indoor.yaml").write_text(yaml_txt)
    print("wrote", out / "coco_indoor.yaml")


if __name__ == "__main__":
    main()
