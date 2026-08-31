"""Hard-object evaluation protocol v2 (fixed 2026-08-31, per 总结分析.md §3.1/3.2).

Fixes relative to v1:
* Real IoU sweep 0.50:0.05:0.95 for overall AP (v1 computed only AP50/AP75
  despite claiming a full sweep).
* Area-range buckets (small/medium/large) now follow COCO ignore semantics:
  a prediction whose best overlap is with a GT *outside* the bucket is
  ignored -- it is neither TP nor FP (v1 counted such predictions as FP,
  biasing the bucket APs downward).
* The former "occlusion" buckets are renamed crowded-overlap buckets: they
  measure 2D box-overlap density (a proxy for scene crowding), NOT physical
  occlusion. Coverage is now computed with a rasterized geometric union,
  eliminating v1's double counting when several boxes cover the same region.
* Optional pycocotools COCOeval cross-check prints standard
  AP / AP50 / AP75 / AP_S / AP_M / AP_L when pycocotools is installed.

Resume protocol (shared with slurm/eval_hard.sh): with ``--out`` + ``--name``,
a model whose block already contains the v2 token ``AP50-95=`` is skipped;
v1 blocks (no such token) are purged by the shell script and re-evaluated.

Usage:
    python eval_hard.py --weights runs/rcr/mrfelcrb/weights/best.pt \
                        --data datasets/coco_indoor/coco_indoor.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402

SIZE_BOUNDS = {"s": (0.0, 32**2), "m": (32**2, 96**2), "l": (96**2, np.inf)}
CROWD_BOUNDS = {"free": (0.0, 0.1), "partial": (0.1, 0.5), "heavy": (0.5, 1.01)}


def iou_vec(box, boxes):
    """IoU of one xyxy box against an (N,4) array, vectorized."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1, y1 = np.maximum(box[0], boxes[:, 0]), np.maximum(box[1], boxes[:, 1])
    x2, y2 = np.minimum(box[2], boxes[:, 2]), np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (box[2] - box[0]) * (box[3] - box[1])
    bb = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (aa + bb - inter + 1e-9)


def crowded_overlap(boxes, grid=64):
    """Fraction of each GT box's area covered by the UNION of other GT boxes.

    Rasterizes the clipped intersections onto a grid per box, so a region
    covered by several boxes is only counted once (v1 summed raw intersection
    areas and double-counted). Resolution is ample for the 0.1/0.5 bucket
    thresholds. This is a 2D box-overlap density proxy, not true occlusion.

    Accepts (N,4) xyxy or (N,5) [cls,x1,y1,x2,y2] arrays.
    """
    boxes = boxes[:, 1:5] if boxes.ndim == 2 and boxes.shape[1] == 5 else boxes
    n = len(boxes)
    occ = np.zeros(n)
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        mask = np.zeros((grid, grid), dtype=bool)
        for j in range(n):
            if j == i:
                continue
            gx1 = max(0, min(grid, int(np.floor((boxes[j][0] - x1) / bw * grid))))
            gx2 = max(0, min(grid, int(np.ceil((boxes[j][2] - x1) / bw * grid))))
            gy1 = max(0, min(grid, int(np.floor((boxes[j][1] - y1) / bh * grid))))
            gy2 = max(0, min(grid, int(np.ceil((boxes[j][3] - y1) / bh * grid))))
            if gx2 > gx1 and gy2 > gy1:
                mask[gy1:gy2, gx1:gx2] = True
        occ[i] = mask.sum() / (grid * grid)
    return occ


def load_gt(data_yaml):
    """GT as {stem: {"hw": (h, w), "boxes": (N,5) [cls, x1, y1, x2, y2]}}."""
    import cv2
    import yaml

    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    if not root.exists():  # yaml 'path' may be stale relative to this cwd
        root = Path(data_yaml).resolve().parent / d["path"]
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
        gt[img.stem] = {"hw": (h, w), "boxes": np.array(boxes, dtype=np.float64).reshape(-1, 5)}
    return gt


def box_areas(boxes):
    """Areas of (N,4) xyxy or (N,5) [cls,x1,y1,x2,y2] boxes."""
    if boxes.ndim == 2 and boxes.shape[1] == 5:
        return (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def gt_keep_flags(gt, size=None, crowd=None):
    """Per-stem boolean keep mask; ~keep GTs become *ignore* in matching."""
    flags = {}
    for stem, info in gt.items():
        boxes = info["boxes"]
        if len(boxes) == 0:
            flags[stem] = np.zeros(0, dtype=bool)
            continue
        keep = np.ones(len(boxes), dtype=bool)
        if size is not None:
            lo, hi = SIZE_BOUNDS[size]
            areas = box_areas(boxes)
            keep &= (areas >= lo) & (areas < hi)
        if crowd is not None:
            lo, hi = CROWD_BOUNDS[crowd]
            ov = crowded_overlap(boxes)
            keep &= (ov >= lo) & (ov < hi)
        flags[stem] = keep
    return flags


def ap_from_flags(tp_flags, fp_flags, n_gt):
    """101-point-interpolated AP from per-detection TP/FP flag arrays."""
    if n_gt == 0:
        return float("nan")
    if len(tp_flags) == 0:
        return 0.0  # every detection ignored, or none made
    tp, fp = np.cumsum(tp_flags), np.cumsum(fp_flags)
    rec, prec = tp / n_gt, tp / (tp + fp)
    mrec, mpre = np.r_[0.0, rec, 1.0], np.r_[0.0, prec, 0.0]
    np.maximum.accumulate(mpre[::-1], out=mpre[::-1])
    thr = np.linspace(0.0, 1.0, 101)
    p_at_r = np.array([mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0 for r in thr])
    return float(p_at_r.mean())


def match_one_threshold(cls_preds, cls_gt, flags, iou_t):
    """COCO-style greedy matching at one IoU threshold.

    cls_preds: [(stem, conf, xyxy)] sorted by conf desc.
    cls_gt:    {stem: (K,4) boxes of this class}; flags: {stem: keep mask}.
    Predictions whose best overlap is an *ignored* (out-of-bucket) GT are
    dropped from TP/FP entirely; only kept GTs can be matched.
    """
    tp, fp = [], []
    matched = {}
    for stem, _, pb in cls_preds:
        gb = cls_gt.get(stem)
        if gb is None or len(gb) == 0:
            fp.append(1)
            tp.append(0)
            continue
        ious = iou_vec(np.asarray(pb, dtype=np.float64), gb)
        kf = flags[stem]
        keep_order = np.where(kf)[0][np.argsort(-ious[kf])] if kf.any() else np.zeros(0, int)
        used = matched.setdefault(stem, set())
        hit = -1
        for k in keep_order:
            if ious[k] >= iou_t and k not in used:
                hit = k
                break
        if hit >= 0:
            used.add(hit)
            tp.append(1)
            fp.append(0)
            continue
        if (~kf).any() and ious[~kf].max() >= iou_t:
            continue  # prediction absorbed by an ignored (out-of-bucket) GT
        fp.append(1)
        tp.append(0)
    return np.array(tp), np.array(fp)


def evaluate(gt, preds, iou_thrs, size=None, crowd=None):
    """Mean per-class AP over ``iou_thrs`` restricted to a GT subset.

    Returns (AP, n_kept_gt). With size/crowd=None this is the overall AP
    (no GT is ignored).
    """
    flags = gt_keep_flags(gt, size=size, crowd=crowd)
    n_kept = int(sum(int(f.sum()) for f in flags.values()))
    # per-class GT with the keep flag carried along each box (flags are
    # computed over ALL classes per image; matching is per class, so the
    # flag array must be sliced to the class's own boxes)
    cls_preds, cls_gt = {}, {}
    for stem, info in gt.items():
        kf = flags[stem]
        for i, row in enumerate(info["boxes"]):
            cls_gt.setdefault(int(row[0]), {}).setdefault(stem, []).append((row[1:5], bool(kf[i])))
    for stem, conf, c, pb in preds:
        cls_preds.setdefault(int(c), []).append((stem, conf, pb))
    aps = []
    for c, per_img in cls_gt.items():
        # COCO rule: only kept (non-ignored) GTs count toward recall; a class
        # with no kept GT in this bucket does not enter the class mean
        n_gt_c = int(sum(int(np.asarray([k for _, k in v]).sum()) for v in per_img.values()))
        if n_gt_c == 0:
            continue
        dets = sorted(cls_preds.get(c, []), key=lambda t: -t[1])
        if not dets:
            aps.append(0.0)
            continue
        class_flags = {s: np.asarray([k for _, k in v], dtype=bool) for s, v in per_img.items()}
        box_arr = {s: np.asarray([b for b, _ in v], dtype=np.float64) for s, v in per_img.items()}
        ap_t = []
        for t in iou_thrs:
            tp, fp = match_one_threshold(dets, box_arr, class_flags, t)
            ap_t.append(ap_from_flags(tp, fp, n_gt_c))
        aps.append(float(np.mean(ap_t)))
    aps = [a for a in aps if a == a]
    return float(np.mean(aps)) if aps else 0.0, n_kept


def coco_crosscheck(gt, preds):
    """Standard pycocotools COCOeval numbers; returns stats array or None."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("COCOeval: pycocotools not installed; "
              "run `pip install pycocotools` for standard AP_S/AP_M/AP_L")
        return None
    images, annotations, cat_ids, ann_id = [], [], set(), 1
    for idx, (stem, info) in enumerate(sorted(gt.items()), start=1):
        h, w = info["hw"]
        images.append({"id": idx, "width": int(w), "height": int(h)})
        for row in info["boxes"]:
            c, x1, y1, x2, y2 = row
            cat_ids.add(int(c) + 1)
            annotations.append({"id": ann_id, "image_id": idx,
                                "category_id": int(c) + 1,
                                "bbox": [float(x1), float(y1),
                                         float(x2 - x1), float(y2 - y1)],
                                "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
            ann_id += 1
    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations,
                       "categories": [{"id": int(c)} for c in sorted(cat_ids)]}
    coco_gt.createIndex()
    stem2id = {s: i for i, (s, _) in enumerate(sorted(gt.items()), start=1)}
    results = []
    for stem, conf, c, box in preds:
        x1, y1, x2, y2 = [float(v) for v in box]
        results.append({"image_id": stem2id[stem], "category_id": int(c) + 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(conf)})
    if not results:
        print("COCOeval: no predictions above --conf, skipped")
        return None
    try:
        ev = COCOeval(coco_gt, coco_gt.loadRes(results), "bbox")
        ev.params.imgIds = sorted(stem2id.values())
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        return ev.stats
    except Exception as e:  # defensive: never kill the hard-object protocol
        print(f"COCOeval: failed ({e}); custom protocol numbers above stand")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--name", default=None, help="run name (resume/skip key)")
    ap.add_argument("--out", default=None, help="results file to check/skip against")
    ap.add_argument("--no-coco", action="store_true",
                    help="skip the pycocotools COCOeval cross-check")
    args = ap.parse_args()

    if args.name and args.out and Path(args.out).exists():
        marker = f"===== {args.name} ====="
        txt = Path(args.out).read_text()
        i = txt.find(marker)
        if i >= 0 and "AP50-95=" in txt[i:i + 600]:  # v2 protocol token
            print(f"[{args.name}] already evaluated (protocol v2), skipping")
            sys.exit(0)

    model = YOLO(args.weights)
    gt = load_gt(args.data)

    preds = []  # (stem, conf, cls, xyxy)
    import yaml
    d = yaml.safe_load(Path(args.data).read_text())
    root = Path(d["path"])
    if not root.exists():
        root = Path(args.data).resolve().parent / d["path"]
    img_dir = root / d.get("val", "val/images")
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

    sweep = np.round(np.arange(0.5, 0.951, 0.05), 2)
    overall, n_all = evaluate(gt, preds, sweep)
    ap50, _ = evaluate(gt, preds, [0.5])
    ap75, _ = evaluate(gt, preds, [0.75])
    aps, n_s = evaluate(gt, preds, [0.5], size="s")
    apm, n_m = evaluate(gt, preds, [0.5], size="m")
    apl, n_l = evaluate(gt, preds, [0.5], size="l")
    cf, n_f = evaluate(gt, preds, [0.5], crowd="free")
    cp, n_p = evaluate(gt, preds, [0.5], crowd="partial")
    ch, n_h = evaluate(gt, preds, [0.5], crowd="heavy")

    print(f"GT boxes={n_all} images={len(gt)} detections={len(preds)} protocol=v2")
    print(f"AP50={ap50:.4f} AP75={ap75:.4f} AP50-95={overall:.4f}")
    print(f"AP50-small={aps:.4f}({n_s}) AP50-medium={apm:.4f}({n_m}) AP50-large={apl:.4f}({n_l})")
    print(f"crowd-free={cf:.4f}({n_f}) crowd-partial={cp:.4f}({n_p}) crowd-heavy={ch:.4f}({n_h})")
    if not args.no_coco:
        stats = coco_crosscheck(gt, preds)
        if stats is not None:
            print(f"COCOeval AP={stats[0]:.4f} AP50={stats[1]:.4f} AP75={stats[2]:.4f} "
                  f"AP_S={stats[3]:.4f} AP_M={stats[4]:.4f} AP_L={stats[5]:.4f}")


if __name__ == "__main__":
    main()
