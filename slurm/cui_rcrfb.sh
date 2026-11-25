#!/bin/bash
#SBATCH --job-name=cui_rcrfb
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/cui_rcrfb_%A_%a.out
#SBATCH --error=logs/cui_rcrfb_%A_%a.err

# ab4 (yolo11n-rcrfb) under the exact Cui-2026 5-fold protocol, comparable
# with the YOLOv8s run (cui_yolov8s.sh): same folds, AdamW 1e-4 cosine
# wd 5e-4, batch 4, 640, 100 epochs, from scratch, seed=fold.
# RCRTrainer adds the ORB-In reconstruction loss (training-only, RCR_RECON_WEIGHT
# env var, default 1.0).
#
# Submit (gres defaults to shard:16; queues behind the yolov8s array):
#   sbatch --array=0-4%5 -p p3 \
#     --export=ALL,DATA=$HOME/rcr/rcr_yolo_1122/datasets/indoor_coco_fitted \
#     slurm/cui_rcrfb.sh
#
# After all 5 folds finish:
#   python train_cui_rcr.py --summarize           # + side-by-side with yolov8s

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/rcr/rcr_yolo_1122}"
mkdir -p logs runs/cui_rcrfb

DATA="${DATA:-datasets/indoor_coco_fitted}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[cui_rcrfb] fold=$FOLD data=$DATA host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# stagger array starts (same convention as cui_yolov8s.sh)
sleep $((FOLD * 15))
python -u train_cui_rcr.py --fold "$FOLD" --data "$DATA" --model cfg/yolo11n-rcrfb.yaml \
    --tag rcrfb --device 0 --workers 4 --project runs/cui_rcrfb
rc=$?
echo "[cui_rcrfb] fold=$FOLD exit=$rc"
exit $rc
