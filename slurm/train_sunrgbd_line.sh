#!/bin/bash
#SBATCH --job-name=rcr_line
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/line_%j.out
#SBATCH --error=logs/line_%j.err

# ============================================================
# 目标域三线对照矩阵（先验来源消融），适用 SUN RGB-D / HomeObjects-3K
#   LINE=pk      98K COCO-indoor 先验（stage-1 权重复用 runs/rcr300）
#   LINE=coco    通用 COCO 预训练 yolo11n.pt 部分迁移（同 stage-2 协议）
#   LINE=scratch 从零训练（t300 协议：300 epochs, lr0=0.01）
#   TAG ∈ {baseline, ab4, rcrfull}，SEED ∈ {0, 1, 2}（三种子）
#   STAGE2_DATA 目标域 yaml（默认 SUN RGB-D；HO 传
#     datasets/homeobjects-3K/HomeObjects-3K.yaml），输出目录名自动带数据集标签
# SUN RGB-D 已完成（seed 0，旧目录名，建议重命名/软链为 *_s0 保持一致）：
#   pk_23072 = pk-baseline-s0 (mAP50=0.3736)  pk_23073 = pk-ab4-s0 (mAP50=0.3688)
# HomeObjects-3K 已完成（ho300 系列，即 scratch 线 s0，协议一致可直接引用）：
#   baseline-s0 0.6367/0.4178  ab4-s0 0.6281/0.4176  rcr-full-s0 0.6537/0.4317
#   （还有 ab2-mrfe/ab3b-lcrbase/mrfelcrb-s0 可作扩展）
# 提交矩阵每条：
#   LINE=<pk|coco|scratch> TAG=<baseline|ab4|rcrfull> SEED=<0|1|2> \
#     [STAGE2_DATA=datasets/homeobjects-3K/HomeObjects-3K.yaml] \
#     sbatch slurm/train_sunrgbd_line.sh
# shard 模式示例（A40 节点）：
#   sbatch -p p3 -w control03 --gres=shard:NVIDIA_A40:12 --cpus-per-task=8 --mem=64G \
#     --export=ALL,LINE=scratch,TAG=rcrfull slurm/train_sunrgbd_line.sh
# 协议对齐说明：
#   pk/coco 线同协议（lr0=0.001, 100 epochs, batch 64, patience 50）——唯一变量是先验来源
#   scratch 线走 t300 协议（lr0=0.01, 300 epochs）——与主表可比（唯一变量是数据集）
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - SUN RGB-D 三线对照"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID}"
echo "运行节点:    ${SLURM_NODELIST}"
echo "LINE/TAG/SEED: ${LINE}/${TAG}/${SEED:-0}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

case "$TAG" in
  baseline) CFG="yolo11n.yaml";           S1="runs/rcr300/baseline-yolo11n-s0/weights/best.pt" ;;
  ab4)      CFG="cfg/yolo11n-rcrfb.yaml"; S1="runs/rcr300/ab4-rcr-fb-s0/weights/best.pt" ;;
  rcrfull)  CFG="cfg/yolo11n-rcr.yaml";   S1="runs/rcr300/rcr-full-s0/weights/best.pt" ;;
  *) echo "错误: TAG 必须是 baseline|ab4|rcrfull（当前: ${TAG}）"; exit 1 ;;
esac

STAGE2_DATA="${STAGE2_DATA:-datasets/sunrgbd/sunrgbd.yaml}"
if [ ! -f "$STAGE2_DATA" ]; then
    echo "错误: 未找到目标域配置 $STAGE2_DATA，请先运行 data/convert_robot_datasets.py！"
    exit 1
fi
STAGE2_DATA_DIR="$(cd "$(dirname "$STAGE2_DATA")" && pwd)"
sed -i "s|^path: .*|path: $STAGE2_DATA_DIR|" "$STAGE2_DATA"
echo "模型配置: $CFG"
echo "目标域:   $STAGE2_DATA"

# 数据集标签：从 STAGE2_DATA 目录名推导（sunrgbd / homeobjects-3K），可用 DS_TAG 覆盖
DS_TAG="${DS_TAG:-$(basename "$(dirname "$STAGE2_DATA")")}"
EXTRA=(--name "line${LINE}-${TAG}-${DS_TAG}-s${SEED:-0}" --seed "${SEED:-0}")
case "$LINE" in
  scratch)
    EXTRA+=(--init scratch --epochs-scratch "${EPOCHS_SCRATCH:-300}" --lr-scratch "${LR_SCRATCH:-0.01}")
    echo "线: 从零（t300 协议）"
    ;;
  coco)
    EXTRA+=(--init coco --coco-weights "${COCO_WEIGHTS:-yolo11n.pt}")
    echo "线: 通用 COCO 预训练迁移（${COCO_WEIGHTS:-yolo11n.pt}）"
    ;;
  pk)
    if [ ! -f "$S1" ]; then
        echo "错误: 未找到 stage-1 先验权重 $S1"
        exit 1
    fi
    EXTRA+=(--init pk --stage1-weights "$S1")
    echo "线: 98K 同域先验（复用 $S1）"
    ;;
  *) echo "错误: LINE 必须是 pk|coco|scratch（当前: ${LINE}）"; exit 1 ;;
esac

python train_pk.py --model "$CFG" \
    --stage1-data datasets/coco_indoor/coco_indoor.yaml \
    --stage2-data "$STAGE2_DATA" \
    --epochs2 "${EPOCHS2:-100}" \
    --batch "${BATCH:-64}" --device "${DEVICE:-0}" \
    --workers "${WORKERS:-4}" \
    --project "$PROJECT_ROOT/runs/rcr-pk" "${EXTRA[@]}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "结束时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "运行时长:    $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
