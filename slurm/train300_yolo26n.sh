#!/bin/bash
#SBATCH --job-name=yolo26n_t300
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/t300_yolo26n_%A_%a.out
#SBATCH --error=logs/t300_yolo26n_%A_%a.err

# YOLO26n comparison under the same protocol as the RCR-YOLO 300-epoch runs:
#   300 epochs, patience 50, close_mosaic 15, batch 64, imgsz 640,
#   lr0 0.01/lrf 0.01, nondeterministic cuDNN, seeds 0/1/2.
#
# For the first single-seed probe on a 4090, submit with 24 shards:
#   sbatch --array=0 -p p3 -w control06 \
#     --gres=shard:24 --cpus-per-task=8 --mem=64G --time=96:00:00 \
#     --export=ALL,DATA=/data/home/zhaozhanshan/rcr/rcr_yolo/datasets/coco_indoor/coco_indoor.yaml \
#     slurm/train300_yolo26n.sh

set -o pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT" || exit 1

SEED="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODEL:-yolo26n.yaml}"
NAME="yolo26n-s${SEED}"
PROJECT="${PROJECT_ROOT}/runs/rcr300"
DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"

echo "============================================"
echo "  YOLO26n comparison - 300 epoch protocol"
echo "============================================"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job:        ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-0})"
echo "Node:       ${SLURM_NODELIST}"
echo "Model:      ${MODEL}  seed=${SEED}"
echo "Data:       ${DATA}"
echo "Start:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs "$PROJECT/$NAME"

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

if [ ! -f "$DATA" ]; then
    echo "ERROR: data yaml not found: $DATA"
    exit 1
fi
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
grep -q "^path: $DATA_DIR$" "$DATA" || sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"

# Completion is defined by .done, matching train300.sh.
if [ -f "$PROJECT/$NAME/.done" ]; then
    echo "[$NAME] already complete (.done), skipping"
    exit 0
fi

COMMON="--data $DATA --name $NAME --project $PROJECT \
    --epochs 300 --batch 64 --imgsz 640 --patience 50 --lr0 0.01 \
    --workers 8 --device ${DEVICE:-0} --seed $SEED"

if [ -f "$PROJECT/$NAME/weights/last.pt" ]; then
    echo "[$NAME] resuming from last.pt"
    python -u train.py $COMMON --model "$PROJECT/$NAME/weights/last.pt" --resume
else
    python -u train.py $COMMON --model "$MODEL"
fi
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
    touch "$PROJECT/$NAME/.done"
    echo "[$NAME] completed successfully"
else
    echo "[$NAME] failed with exit code $STATUS"
fi
exit "$STATUS"
