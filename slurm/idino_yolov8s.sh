#!/bin/bash
#SBATCH --job-name=idino_v8s
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/idino_v8s_%j.out
#SBATCH --error=logs/idino_v8s_%j.err

# YOLOv8s under the I-DINO protocol (independent 2nd anchor on the same
# Indoor-COCO): 8:2 main split (9041/2261), COCO-pretrained yolov8s.pt,
# 100 epochs, ultralytics defaults, 640, seed 0.
# Target (I-DINO table 1): mAP 60.7 / mAP50 80.9 / mAP75 66.1.
# Requires yolov8s.pt in the repo root (committed via LFS).
#
# Submit:
#   sbatch -p p3 --export=ALL,DATA=$HOME/rcr/rcr_yolo_1122/datasets/indoor_coco_fitted \
#     slurm/idino_yolov8s.sh
# Summary: python train_idino_yolov8s.py --summarize

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/rcr/rcr_yolo_1122}"
mkdir -p logs runs/idino

DATA="${DATA:-datasets/indoor_coco_fitted}"
if [ ! -f yolov8s.pt ]; then
    echo "[idino_v8s] ERROR: yolov8s.pt not in repo root (git lfs pull?)"
    exit 1
fi

echo "[idino_v8s] data=$DATA host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u train_idino_yolov8s.py --data "$DATA" --device 0 --workers 4 --project runs/idino
rc=$?
echo "[idino_v8s] exit=$rc"
exit $rc
