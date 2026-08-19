#!/bin/bash
#SBATCH --job-name=rcr_ablation
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%2
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err

# ============================================================
# RCR-YOLO 消融矩阵（SLURM 数组作业，每个变体一个独立作业）
# %2 表示最多 2 个数组任务并行（可按平台配额调整）
# 硬件要求：每个任务 1× RTX 4090 24GB
# 提交：sbatch slurm/ablation.sh
# 变体清单（index -> name:model）：
#   0 baseline-yolo11n : yolo11n.yaml          原生基线
#   1 ab1-orb          : cfg/yolo11n-orb.yaml     +ORB-In
#   2 ab2-mrfe         : cfg/yolo11n-mrfe.yaml    +MRFE-Neck
#   3 ab3-lcr          : cfg/yolo11n-lcr.yaml     +LCR
#   4 ab3b-lcrbase     : cfg/yolo11n-lcrbase.yaml +LCR 降级回退版
#   5 ab4-rcr-fb       : cfg/yolo11n-rcrfb.yaml   全量去掉 ORB-In
#   6 rcr-full         : cfg/yolo11n-rcr.yaml     全量 RCR-YOLO
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

VARIANTS=(
    "baseline-yolo11n:yolo11n.yaml"
    "ab1-orb:cfg/yolo11n-orb.yaml"
    "ab2-mrfe:cfg/yolo11n-mrfe.yaml"
    "ab3-lcr:cfg/yolo11n-lcr.yaml"
    "ab3b-lcrbase:cfg/yolo11n-lcrbase.yaml"
    "ab4-rcr-fb:cfg/yolo11n-rcrfb.yaml"
    "rcr-full:cfg/yolo11n-rcr.yaml"
)
IFS=: read -r NAME MODEL <<< "${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - 消融实验 [$NAME]"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT})"
echo "运行节点:    ${SLURM_NODELIST}"
echo "模型配置:    ${MODEL}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"
if [ ! -f "$DATA" ]; then
    echo "错误: 未找到数据配置文件 $DATA，请先运行 data/make_coco_indoor.py！"
    exit 1
fi
sed -i 's|^path: .*|path: .|' "$DATA"

# 已有 best.pt 则跳过（重提交时可断点跳过已完成变体）
if [ -f "runs/rcr/$NAME/weights/best.pt" ]; then
    echo "[$NAME] 已存在 best.pt，跳过"
    exit 0
fi

python train.py --model "$MODEL" --data "$DATA" --name "$NAME" \
    --epochs "${EPOCHS:-150}" --batch "${BATCH:-32}" --device "${DEVICE:-0}" \
    --project runs/rcr

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "[$NAME] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[$NAME] 运行时长: $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
