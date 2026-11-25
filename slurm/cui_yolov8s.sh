#!/bin/bash
#SBATCH --job-name=cui_v8s
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%5
#SBATCH --output=logs/cui_v8s_%A_%a.out
#SBATCH --error=logs/cui_v8s_%A_%a.err

# YOLOv8s under the exact Cui-2026 (MST 37 415406) 5-fold protocol:
#   folds from folds.json (9041 train images, ~7233/1808 per fold),
#   from scratch, AdamW lr 1e-4 cosine, wd 5e-4, batch 4, 640x640, 100 epochs,
#   seed = fold index (paper fixes no seed; per-fold distinct init).
#
# Prereqs on the server (do once):
#   1) dataset:  either copy local datasets/indoor_coco_fitted (~1.4 GB) here,
#      or rebuild:   # server already has the coco_indoor images
#        mkdir -p datasets/mini_coco
#        ln -s /data/home/zhaozhanshan/rcr/rcr_yolo/datasets/coco_indoor/train/images \
#              datasets/mini_coco/train2017
#        cp <uploaded instances_train2017.json> datasets/mini_coco/annotations/
#        python data/build_indoor_coco_fitted.py --coco datasets/mini_coco \
#               --out datasets/indoor_coco_fitted     # same seed=0, t=103, deterministic
#   2) folds: python tools/make_cui_folds.py --root datasets/indoor_coco_fitted
#
# Submit (from ~/rcr/rcr_yolo_c0ffed2 on control03):
#   sbatch --array=0-4%5 -p p3 -w control03 \
#     --gres=gpu:NVIDIA_A40:1 --cpus-per-task=8 --mem=32G --time=24:00:00 \
#     --export=ALL,DATA=$HOME/rcr/rcr_yolo_c0ffed2/datasets/indoor_coco_fitted \
#     slurm/cui_yolov8s.sh
#
# After all 5 folds finish, aggregate and compare with Cui table 6:
#   python train_cui_yolov8s.py --summarize --project runs/cui

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/rcr/rcr_yolo_c0ffed2}"
mkdir -p logs runs/cui

DATA="${DATA:-datasets/indoor_coco_fitted}"
FOLD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[cui_v8s] fold=$FOLD data=$DATA host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -u train_cui_yolov8s.py --fold "$FOLD" --data "$DATA" --device 0 --workers 8 --project runs/cui
rc=$?
echo "[cui_v8s] fold=$FOLD exit=$rc"
exit $rc
