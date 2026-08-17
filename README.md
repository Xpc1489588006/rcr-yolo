# RCR-YOLO

Lightweight indoor difficult-object detector for unstructured service-robot scenes.

RCR-YOLO builds on YOLO11n and combines:

- **ORB-In** — training-time object reconstruction branch (mask + edge dual target,
  attached to P2). Discarded at inference: **zero extra params/FLOPs/latency**.
- **LCR** — lightweight easy-help-hard context reasoning (with `LCRBase` fallback variant).
- **MRFE-Neck** — multi-receptive-field adaptive enhancement with dynamic fusion (GSConv).
- **PK training** — two-stage prior-knowledge transfer (COCO-indoor prior → target domain).

## Layout

```
rcr/                 core modules + ultralytics registration + trainer
cfg/                 model yaml generator + 6 ablation variants
data/                COCO-indoor builder; SUN RGB-D / AVD / TUT converters
train.py             single-stage training entry
train_pk.py          PK two-stage prior-knowledge training
eval_hard.py         AP_S / AP_M / occlusion-bucketed AP (pure python)
vis/visualize.py     Grad-CAM + ORB-In reconstruction visualization
scripts/             local ablation queue (PowerShell)
slurm/               GPU-platform job scripts (baseline / full / ablation / PK / eval)
tests/smoke_test.py  CPU smoke test (module forward, yaml build, 1-epoch synthetic train)
```

## Setup

```bash
pip install -r requirements.txt        # CUDA wheel of torch installed separately
python tests/smoke_test.py             # verify the whole pipeline (CPU-friendly)
```

## Data

Build the COCO-indoor training set from COCO 2017 (45 indoor classes):

```bash
python data/make_coco_indoor.py --coco /path/to/coco2017 --out datasets/coco_indoor
# --no-links: labels+yaml only (portable relative-path yaml for cluster sync)
```

Cross-domain test sets (SUN RGB-D / AVD / TUT Indoor):

```bash
python data/convert_robot_datasets.py --help
```

## Train

```bash
# full model
python train.py --model cfg/yolo11n-rcr.yaml --data datasets/coco_indoor/coco_indoor.yaml

# PK two-stage transfer (e.g. to SUN RGB-D)
python train_pk.py --model cfg/yolo11n-rcr.yaml \
    --stage1-data datasets/coco_indoor/coco_indoor.yaml \
    --stage2-data datasets/sunrgbd/sunrgbd.yaml
```

On a SLURM cluster: `sbatch slurm/ablation.sh` (7-variant ablation array job),
see `slurm/*.sh` for baseline / full / PK / hard-eval jobs.

## Evaluate

```bash
python eval_hard.py --weights runs/rcr/rcr-full/weights/best.pt \
                    --data datasets/coco_indoor/coco_indoor.yaml
```

## Cited foundations

Adapted from ORFENet (TGRS 2024), CRNet (TMM 2024), IENet (Neurocomputing 2021),
IOVRM (JKSU-CIS 2026) and YOLO-SM (TETCI 2024) for indoor service-robot detection.
