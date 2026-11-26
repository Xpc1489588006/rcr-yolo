#!/bin/bash
#SBATCH --job-name=rcr_ho300
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/ho300_%A_%a.out
#SBATCH --error=logs/ho300_%A_%a.err

# ============================================================
# HomeObjects-3K 跨数据集训练（2026-11-26）
# 目的：架构泛化验证。与 train300.sh 完全同协议
# （300ep / batch64 / patience50 / imgsz640 / lr0 0.01 / close_mosaic15，
#  COCO 预训练权重初始化），仅换 DATA 与 PROJECT。
# 数据：datasets/homeobjects-3K/HomeObjects-3K.yaml（12类，2285 train / 404 val）
# 6 模型 × 1 seed（array=0-5；补种子时 export SEED=1 后重新提交）：
#    0 baseline-yolo11n : yolo11n.yaml              原生基线
#    1 ab2-mrfe         : cfg/yolo11n-mrfe.yaml     +MRFE
#    2 ab3b-lcrbase     : cfg/yolo11n-lcrbase.yaml  +LCRBase
#    3 mrfelcrb         : cfg/yolo11n-mrfelcrb.yaml MRFE+LCRBase（与 ab4 差=ORB 贡献）
#    4 ab4-rcr-fb       : cfg/yolo11n-rcrfb.yaml    ORB+MRFE+LCRBase（论文最终模型 RCR-YOLO）
#    5 rcr-full         : cfg/yolo11n-rcr.yaml      ORB+MRFE+full LCR（对照变体）
# 训练目录：runs/rcr300ho/<variant>-s<seed>
# 断点保护与 train300.sh 相同：
#   .done 哨兵存在   -> 跳过（已完成）
#   仅有 last.pt     -> resume 续训（best.pt 不能作完成判据）
# 提交：sbatch -w control06 slurm/train300_ho.sh
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

VARIANTS=(
    "baseline-yolo11n:yolo11n.yaml"
    "ab2-mrfe:cfg/yolo11n-mrfe.yaml"
    "ab3b-lcrbase:cfg/yolo11n-lcrbase.yaml"
    "mrfelcrb:cfg/yolo11n-mrfelcrb.yaml"
    "ab4-rcr-fb:cfg/yolo11n-rcrfb.yaml"
    "rcr-full:cfg/yolo11n-rcr.yaml"
)
VIDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEED:-0}
IFS=: read -r VARIANT MODEL <<< "${VARIANTS[$VIDX]}"
NAME="${VARIANT}-s${SEED}"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - HomeObjects-3K 300轮 [$NAME]"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT})"
echo "运行节点:    ${SLURM_NODELIST}"
echo "模型配置:    ${MODEL}  seed=${SEED}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

DATA="${DATA:-datasets/homeobjects-3K/HomeObjects-3K.yaml}"
if [ ! -f "$DATA" ]; then
    echo "错误: 未找到数据配置文件 $DATA，请先解压 homeobjects-3K.zip 并写好 yaml！"
    exit 1
fi
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
# 与 train300.sh 相同的 path 兜底（yaml 已是绝对路径时 grep 命中、跳过改写）
grep -q "^path: $DATA_DIR$" "$DATA" || sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"

PROJECT="${PROJECT:-$PROJECT_ROOT/runs/rcr300ho}"
mkdir -p "$PROJECT/$NAME"

# 断点保护：.done 哨兵才算完成；被中断（有 last 无 .done）则续训
if [ -f "$PROJECT/$NAME/.done" ]; then
    echo "[$NAME] 已完成（.done 哨兵存在），跳过"
    exit 0
fi

COMMON="--data $DATA --name $NAME --project $PROJECT \
    --epochs ${EPOCHS:-300} --batch ${BATCH:-64} --device ${DEVICE:-0} --seed $SEED"

if [ -f "$PROJECT/$NAME/weights/last.pt" ]; then
    echo "[$NAME] 检测到 last.pt，断点续训"
    python train.py $COMMON --model "$PROJECT/$NAME/weights/last.pt" --resume \
        && touch "$PROJECT/$NAME/.done"
else
    python train.py $COMMON --model "$MODEL" \
        && touch "$PROJECT/$NAME/.done"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "[$NAME] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[$NAME] 运行时长: $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "[$NAME] final mAP 见上方 train.py 输出及 $PROJECT/$NAME/results.csv"
echo "============================================"
