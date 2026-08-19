#!/bin/bash
#SBATCH --job-name=rcr_full
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err

# ============================================================
# RCR-YOLO 完整模型训练（ORB-In + LCR + MRFE-Neck，COCO-indoor）
# 硬件要求：1× RTX 4090 24GB
# 预计时长：150 epochs ≈ 10-12 小时
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - 完整模型训练"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID}"
echo "运行节点:    ${SLURM_NODELIST}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "项目根目录:  ${PROJECT_ROOT}"
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
# 修正从 Windows 同步过来的 yaml：把 path 指向数据集根目录（绝对路径，避免 ultralytics 解析歧义）
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"
echo "数据配置文件检查通过: $DATA"

python train.py --model cfg/yolo11n-rcr.yaml --data "$DATA" --name rcr-full \
    --epochs "${EPOCHS:-150}" --batch "${BATCH:-32}" --device "${DEVICE:-0}" \
    --project runs/rcr

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "结束时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "运行时长:    $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
