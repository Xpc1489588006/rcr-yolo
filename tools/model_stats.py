"""Unified model efficiency report (answers 总结分析.md §3.5/§3.6).

Every run is measured with the SAME recipe so the paper's efficiency table
is internally consistent:

* checkpoint params  -- every tensor stored in best.pt (training-time view)
* deploy params      -- excluding ORBIn reconstruction branches, i.e. what
                        the exported detector actually carries (ORB forward is
                        identity, its weights are dead weight unless pruned)
* GFLOPs             -- real inference-path MACs (ORB identity contributes 0)
* batch=1 latency    -- warmup + timed fp32 forward on the chosen device

Usage:
    python tools/model_stats.py --weights "runs/rcr/*/weights/best.pt" \
        --data datasets/coco_indoor/coco_indoor.yaml --device 0 \
        --csv tools/model_stats.csv

Notes:
* Latency depends on the device; report the GPU model in the paper.
* ``--data`` is only used to repair a stale relative ``path:`` inside the
  dataset yaml (same sed-style fix slurm scripts apply); metrics themselves
  do not need images.
"""

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcr.ultralytics_patch import register_rcr_modules  # noqa: E402

register_rcr_modules()

from ultralytics import YOLO  # noqa: E402

ORB_PATTERNS = ("ORB",)  # training-only reconstruction branch (rcr/orb_in.py)


def _is_orb(module):
    return any(p in type(module).__name__ for p in ORB_PATTERNS)


def count_params(model):
    """(checkpoint params, trainable params, deploy params, orb params).

    Deploy = own params of every module outside ORB branches; an ORB module
    and everything inside it is excluded entirely.
    """
    m = model.model
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    deploy = orb = 0
    for mod in m.modules():
        own = sum(p.numel() for p in mod.parameters(recurse=False))
        if own == 0:
            continue
        if _is_orb(mod) or any(_is_orb(a) for a in mod.modules()):
            orb += own
        else:
            deploy += own
    return total, trainable, deploy, orb


def count_flops(model, imgsz):
    """Inference-path GFLOPs; falls back to thop if ultralytics helper moves."""
    m = model.model
    m.eval()
    try:
        from ultralytics.utils.torch_utils import get_flops
        return round(float(get_flops(m, imgsz)), 2)
    except Exception:
        pass
    try:
        import thop
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=next(m.parameters()).device)
        macs, _ = thop.profile(m, inputs=[dummy], verbose=False)
        return round(float(macs) * 2 / 1e9, 2)
    except Exception as e:
        print(f"  warning: GFLOPs unavailable ({e})")
        return float("nan")


def measure_latency(model, imgsz, device, n=100, warmup=20):
    """batch=1 fp32 forward latency in ms/img (warmup excluded)."""
    m = model.model.to(device).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
    is_cuda = torch.device(device).type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            m(dummy)
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            m(dummy)
        if is_cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return dt / n * 1000.0


def fix_data_path(data_yaml):
    """Make the yaml's ``path:`` absolute and existing (best-effort)."""
    if not data_yaml:
        return
    import yaml
    p = Path(data_yaml)
    if not p.exists():
        print(f"  warning: data yaml {p} not found, skipping path fix")
        return
    d = yaml.safe_load(p.read_text())
    root = Path(d.get("path", "."))
    if root.exists():
        return
    fixed = (p.resolve().parent / d["path"]).resolve()
    if fixed.exists():
        d["path"] = fixed.as_posix()
        p.write_text(yaml.safe_dump(d, sort_keys=False))
        print(f"  fixed yaml path -> {fixed}")


def device_name(device):
    try:
        if torch.device(device).type == "cuda":
            return torch.cuda.get_device_name(device)
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help='glob pattern, e.g. "runs/rcr/*/weights/best.pt"')
    ap.add_argument("--data", default=None, help="dataset yaml (path repair only)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--n", type=int, default=100, help="timed forward iterations")
    ap.add_argument("--csv", default=None, help="optional CSV output")
    ap.add_argument("--no-latency", action="store_true",
                    help="skip latency measurement (e.g. CPU-only nodes)")
    args = ap.parse_args()

    fix_data_path(args.data)
    files = sorted(glob.glob(args.weights))
    if not files:
        raise SystemExit(f"no weights match {args.weights}")
    dev = device_name(args.device)
    print(f"device={dev} imgsz={args.imgsz} fp32 batch=1")
    header = ["name", "params_ckpt_M", "params_trainable_M", "params_deploy_M",
              "params_orb_M", "gflops", "latency_ms"]
    rows = []
    for f in files:
        name = Path(f).resolve().parents[1].name
        try:
            model = YOLO(f)
            tot, trn, dep, orb = count_params(model)
            gf = count_flops(model, args.imgsz)
            lat = measure_latency(model, args.imgsz, args.device, n=args.n) \
                if not args.no_latency else float("nan")
            rows.append([name, round(tot / 1e6, 3), round(trn / 1e6, 3),
                         round(dep / 1e6, 3), round(orb / 1e6, 3), gf, round(lat, 2)])
            print(f"{name:20s} ckpt={tot/1e6:.3f}M train={trn/1e6:.3f}M "
                  f"deploy={dep/1e6:.3f}M orb={orb/1e6:.3f}M "
                  f"GFLOPs={gf} latency={lat:.2f}ms")
        except Exception as e:
            print(f"{name:20s} FAILED: {e}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
        print(f"CSV written: {args.csv}")
    # markdown table ready for the paper
    print("\n| model | ckpt params (M) | deploy params (M) | GFLOPs | latency (ms) |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[3]} | {r[5]} | {r[6]} |")


if __name__ == "__main__":
    main()
