#!/bin/bash
#SBATCH --job-name=rcr_stats
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/modelstats_%j.out
#SBATCH --error=logs/modelstats_%j.err

# ============================================================
# 统一参数量/效率统计（论文效率表数据源）：
# checkpoint/训练态/部署态参数量 + GFLOPs + batch=1 延迟
# 必须在 GPU 节点运行（登录节点无卡，延迟会被跳过）。
# 提交：sbatch -w control06 slurm/model_stats.sh
# （control03 故障期间务必把 -w 放在脚本名之前）
# 300 轮目录：sbatch -w control06 \
#   --export=ALL,RUNS_DIR=runs/rcr300,CSV=tools/model_stats_rcr300.csv slurm/model_stats.sh
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

mkdir -p logs tools

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

echo "============================================"
echo "  RCR-YOLO - 统一效率统计"
echo "============================================"
echo "GPU 型号:    $(srun nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"

python tools/model_stats.py --weights "${RUNS_DIR:-runs/rcr}/*/weights/best.pt" \
    --data "$DATA" --device "${DEVICE:-0}" \
    --csv "${CSV:-tools/model_stats.csv}"

echo "--------------------------------------------"
echo "完成，结果: tools/model_stats.csv"
echo "============================================"
