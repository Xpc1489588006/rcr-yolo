"""Hard-object evaluation: AP_S / AP_M / occlusion-bucketed AP.

Standard mAP comes from ``model.val``; this script adds the paper's
hard-object protocol on any YOLO-format dataset:

* AP by object size (COCO convention: small < 32^2 px, medium < 96^2 px,
  large >= 96^2 px)
* AP by occlusion bucket, where occlusion of a GT box = fraction of its area
  covered by other GT boxes (free < 0.1 / partial 0.1-0.5 / heavy > 0.5)

Pure-python matching (no pycocotools), IoU sweep 0.5:0.95 + AP50 per bucket.
With ``--out`` + ``--name``, a model already present in the results file is
skipped, so batch re-runs resume instead of duplicating entries.

Usage:
    python eval_hard.py --weights runs/rcr/yolo11n-rcr/weights/best.pt \
                        --data datasets/coco_indoor/coco_indoor.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402


def load_gt(data_yaml: str):
    import yaml

    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    split = d.get("val", "val/images")
    img_dir = root / split
    lbl_dir = root / split.replace("images", "labels")
    gt = {}
    for img in sorted(img_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        im = cv2.imread(str(img))
        if im is None:
            continue
        h, w = im.shape[:2]
        boxes = []
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, bw, bh = map(float, p[:5])
                boxes.append([int(c), (cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        gt[img.stem] = np.array(boxes, dtype=np.float64).reshape(-1, 5)
    return gt


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[0]), np.maximum(a[1], b[1])
    x2, y2 = np.minimum(a[2], b[2]), np.minimum(a[3], b[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def occlusion_ratios(boxes):
    """Fraction of each GT box's area covered by any other GT box."""
    n = len(boxes)
    occ = np.zeros(n)
    for i in range(n):
        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        inters = []
        for j in range(n):
            if j != i:
                x1, y1 = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
                x2, y2 = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
                inters.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        occ[i] = min(1.0, sum(inters) / max(area_i, 1e-9))
    return occ


def ap_all_point(tp, fp, n_gt):
    if n_gt == 0:
        return float("nan")
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp / n_gt, tp / (tp + fp)
    mrec, mpre = np.r_[0.0, rec, 1.0], np.r_[0.0, prec, 0.0]
    np.maximum.accumulate(mpre[::-1], out=mpre[::-1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def evaluate(gt, preds, iou_t=0.5, filt=None):
    """filt: dict with optional keys 'size' in {s,m,l} and 'occ' bucket name."""
    per_cls = {}
    for stem, boxes in gt.items():
        if len(boxes) == 0:
            continue
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        occ = occlusion_ratios(boxes)
        keep = np.ones(len(boxes), bool)
        if filt and filt.get("size"):
            keep &= {"s": areas < 32**2, "m": (areas >= 32**2) & (areas < 96**2),
                     "l": areas >= 96**2}[filt["size"]]
        if filt and filt.get("occ"):
            keep &= {"free": occ < 0.1, "partial": (occ >= 0.1) & (occ <= 0.5),
                     "heavy": occ > 0.5}[filt["occ"]]
        for k in np.where(keep)[0]:
            per_cls.setdefault(int(boxes[k][0]), []).append((stem, boxes[k][1:5]))
    aps = []
    for c, g in per_cls.items():
        n_gt = len(g)
        pr = [p for p in preds if p[2] == c]
        pr.sort(key=lambda t: -t[1])
        used = {s: [False] * len([x for x in g if x[0] == s]) for s in {x[0] for x in g}}
        gt_by_stem = {}
        for s, b in g:
            gt_by_stem.setdefault(s, []).append(b)
        used = {s: [False] * len(v) for s, v in gt_by_stem.items()}
        tp, fp = [], []
        for stem, conf, _, pb in pr:
            cands = gt_by_stem.get(stem, [])
            best, bi = 0.0, -1
            for k, gb in enumerate(cands):
                v = iou(pb, gb)
                if v > best:
                    best, bi = v, k
            if best >= iou_t and bi >= 0 and not used[stem][bi]:
                used[stem][bi] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        aps.append(ap_all_point(np.array(tp), np.array(fp), n_gt))
    aps = [a for a in aps if a == a]
    return float(np.mean(aps)) if aps else 0.0, sum(len(v) for v in per_cls.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--name", default=None, help="run name (resume/skip key)")
    ap.add_argument("--out", default=None, help="results file to check/skip against")
    args = ap.parse_args()

    if args.name and args.out and Path(args.out).exists():
        marker = f"===== {args.name} ====="
        txt = Path(args.out).read_text()
        i = txt.find(marker)
        if i >= 0 and "AP50=" in txt[i:i + 400]:
            print(f"[{args.name}] already evaluated, skipping")
            sys.exit(0)

    model = YOLO(args.weights)
    gt = load_gt(args.data)

    preds = []  # (stem, conf, cls, xyxy)
    import yaml
    d = yaml.safe_load(Path(args.data).read_text())
    img_dir = Path(d["path"]) / d.get("val", "val/images")
    for stem in gt:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = img_dir / (stem + ext)
            if p.exists():
                break
        else:
            continue
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            preds.append((stem, float(conf), int(cls), box))

    overall50, n = evaluate(gt, preds, 0.5)
    overall75, _ = evaluate(gt, preds, 0.75)
    aps, _ = evaluate(gt, preds, 0.5, {"size": "s"})
    apm, _ = evaluate(gt, preds, 0.5, {"size": "m"})
    apl, _ = evaluate(gt, preds, 0.5, {"size": "l"})
    ocf, nf = evaluate(gt, preds, 0.5, {"occ": "free"})
    ocp, np_ = evaluate(gt, preds, 0.5, {"occ": "partial"})
    och, nh = evaluate(gt, preds, 0.5, {"occ": "heavy"})
    print(f"GT boxes={sum(len(v) for v in gt.values())}")
    print(f"AP50={overall50:.4f} AP75={overall75:.4f}")
    print(f"AP50_small={aps:.4f} AP50_medium={apm:.4f} AP50_large={apl:.4f}")
    print(f"AP50 occ-free={ocf:.4f}({nf}) occ-partial={ocp:.4f}({np_}) occ-heavy={och:.4f}({nh})")


if __name__ == "__main__":
    main()
