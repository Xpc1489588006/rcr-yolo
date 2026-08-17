"""Paper visualizations: Grad-CAM saliency and ORB-In reconstruction.

Sub-commands:

    # Grad-CAM heatmap on the last neck feature (default layer = -2, before Detect)
    python vis/visualize.py gradcam --weights runs/rcr/yolo11n-rcr/weights/best.pt \
        --img some.jpg --out vis_out/cam.jpg

    # ORB-In reconstruction (pred vs GT mask/edge, training-time branch)
    python vis/visualize.py recon --weights runs/rcr/yolo11n-rcr/weights/best.pt \
        --img some.jpg --label labels/some.txt --out vis_out/recon.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402

from rcr.orb_in import ORBIn  # noqa: E402


def _load_model(weights, device, imgsz):
    yolo = YOLO(weights)
    model = yolo.model.to(device).float()
    names = yolo.names
    return model, names, imgsz


def _preprocess(img_path, imgsz, device):
    im0 = cv2.imread(str(img_path))
    if im0 is None:
        raise FileNotFoundError(img_path)
    lb = LetterBox(imgsz, auto=False, stride=32)
    im = lb(image=im0)
    t = torch.from_numpy(im.transpose(2, 0, 1)).to(device).float().unsqueeze(0) / 255.0
    return im0, t


def find_orb(model):
    for m in model.modules():
        if isinstance(m, ORBIn):
            return m
    return None


# ---------------------------------------------------------------- Grad-CAM

def gradcam(args):
    model, names, imgsz = _load_model(args.weights, args.device, args.imgsz)
    model.eval()
    im0, x = _preprocess(args.img, imgsz, args.device)

    layer_idx = args.layer if args.layer >= 0 else len(model.model) + args.layer
    target_layer = model.model[layer_idx]
    print(f"[Grad-CAM] target layer #{layer_idx}: {type(target_layer).__name__}")

    fmap, grad = {}, {}

    def fwd_hook(m, inp, out):
        fmap["v"] = out

    def bwd_hook(m, ginp, gout):
        grad["v"] = gout[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x.requires_grad_(True)
    preds = model(x)  # eval mode: (B, 4+nc, anchors)
    if isinstance(preds, tuple):  # new ultralytics returns (preds, attrs)
        preds = preds[0]
    cls_scores = preds[:, 4:, :]
    if args.cls is not None:
        score = cls_scores[:, args.cls].sum()
    else:
        score = cls_scores.max(dim=1)[0].sum()
    model.zero_grad()
    score.backward()

    a, g = fmap["v"], grad["v"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = (cam.cpu().numpy() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    h, w_ = im0.shape[:2]
    heat = cv2.resize(heat, (w_, h))
    overlay = cv2.addWeighted(im0, 0.55, heat, 0.45, 0)
    top = f"Grad-CAM" + (f" cls={names.get(args.cls, args.cls)}" if args.cls is not None
                        else " (max-conf class)")
    cv2.putText(overlay, top, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.hstack([im0, heat, overlay]))
    print(f"[Grad-CAM] saved -> {args.out}")
    h1.remove()
    h2.remove()


# ------------------------------------------------------- ORB-In reconstruction

def recon(args):
    model, names, imgsz = _load_model(args.weights, args.device, args.imgsz)
    orb = find_orb(model)
    if orb is None:
        raise RuntimeError("weights do not contain an ORB-In module (use a *rcr* model)")
    model.train()  # activate the training-time reconstruction branch
    im0, x = _preprocess(args.img, imgsz, args.device)
    with torch.no_grad():
        model(x)
    pred = torch.sigmoid(orb.recon)[0]  # (2, H, W): [mask, edge]
    model.eval()

    h, w = x.shape[-2:]
    # GT target from YOLO-format label
    boxes = []
    if args.label and Path(args.label).exists():
        for line in Path(args.label).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                boxes.append(list(map(float, p[:5])))
    tgt = ORBIn.build_target(
        [torch.tensor([b[1:] for b in boxes], dtype=torch.float32)], None, h, w, x.device
    )[0]

    if pred.shape[-2:] != (h, w):
        pred = F.interpolate(pred.unsqueeze(0), size=(h, w), mode="bilinear",
                             align_corners=False)[0]

    def panel(t, label):
        v = (t.cpu().numpy() * 255).astype(np.uint8)
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        cv2.putText(v, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return v

    im_show = cv2.resize(im0, (w, h))
    grid = np.hstack([
        im_show,
        panel(pred[0], "pred mask"), panel(tgt[0], "gt mask"),
        panel(pred[1], "pred edge"), panel(tgt[1], "gt edge"),
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"[recon] saved -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gradcam", "recon"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", default="vis_out/out.jpg")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cls", type=int, default=None, help="gradcam: focus class id")
    ap.add_argument("--layer", type=int, default=-2, help="gradcam: layer index in model.model")
    ap.add_argument("--label", default="", help="recon: YOLO-format label txt")
    args = ap.parse_args()
    (gradcam if args.mode == "gradcam" else recon)(args)


if __name__ == "__main__":
    main()
