#!/bin/bash
#SBATCH --job-name=rcr_pk
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pk_%j.out
#SBATCH --error=logs/pk_%j.err

# ============================================================
# PK 两阶段先验知识训练（IOVRM 策略）
#   阶段一：COCO-indoor 大先验集训练 -> 先验权重
#   阶段二：目标域（SUN RGB-D / AVD）小 lr 微调
# 硬件要求：1× RTX 4090 24GB
# 用法：先用 data/convert_robot_datasets.py 转换目标域数据，然后
#   sbatch slurm/train_pk.sh
#   STAGE2_DATA=datasets/avd/avd.yaml sbatch slurm/train_pk.sh   # 换目标域
#   STAGE1_WEIGHTS=runs/rcr-pk/stage1-prior/weights/best.pt sbatch slurm/train_pk.sh  # 复用阶段一
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - PK 两阶段先验知识训练"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID}"
echo "运行节点:    ${SLURM_NODELIST}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"
STAGE2_DATA="${STAGE2_DATA:-datasets/sunrgbd/sunrgbd.yaml}"
if [ ! -f "$DATA" ]; then
    echo "错误: 未找到先验集配置 $DATA"
    exit 1
fi
if [ ! -f "$STAGE2_DATA" ]; then
    echo "错误: 未找到目标域配置 $STAGE2_DATA，请先运行 data/convert_robot_datasets.py！"
    exit 1
fi
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"
STAGE2_DATA_DIR="$(cd "$(dirname "$STAGE2_DATA")" && pwd)"
sed -i "s|^path: .*|path: $STAGE2_DATA_DIR|" "$STAGE2_DATA"
echo "先验集: $DATA"
echo "目标域: $STAGE2_DATA"

EXTRA=()
if [ -n "$STAGE1_WEIGHTS" ]; then
    EXTRA+=(--stage1-weights "$STAGE1_WEIGHTS")
    echo "复用已有阶段一权重: $STAGE1_WEIGHTS"
fi

python train_pk.py --model cfg/yolo11n-rcr.yaml \
    --stage1-data "$DATA" --stage2-data "$STAGE2_DATA" \
    --epochs1 "${EPOCHS1:-150}" --epochs2 "${EPOCHS2:-100}" \
    --batch "${BATCH:-128}" --device "${DEVICE:-0}" \
    --project runs/rcr-pk "${EXTRA[@]}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "结束时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "运行时长:    $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
