#!/bin/bash
#SBATCH --job-name=rcr_t300
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --array=0-29%4
#SBATCH --output=logs/t300_%A_%a.out
#SBATCH --error=logs/t300_%A_%a.err

# ============================================================
# 300 轮统一协议全量重训（2026-08-31 决议）
# 10 模型 × 3 种子（seed 0/1/2）= 30 个独立训练任务，
# %4 并发占满 control06 的 4× RTX 4090。
# 训练目录：runs/rcr300/<模型>-s<种子>（150 轮旧结果保留在 runs/rcr 作存档）
#
# 协议要点（相对 150 轮版仅改 epochs 与 batch）：
#   epochs=300  patience=50（早停兜底，收敛即停）
#   close_mosaic=15（train.py 硬编码，距结束 15 轮关闭，语义不变）
#   batch=64（24G 4090 约束；全部任务同 batch，协议内部可比）
#
# 断点保护：
#   best.pt 已存在      -> 跳过（已完成）
#   仅有 last.pt        -> resume 续训（应对时限中断/掉卡）
# 提交：
#   sbatch -w control06 slurm/train300.sh
#   训练完成后接评测（<JOBID> 为本作业数组 ID）：
#   sbatch -w control06 --dependency=afterok:<JOBID>_% \
#       --export=ALL,RUNS_DIR=runs/rcr300 slurm/eval_hard.sh
# 变体清单（task_id = 变体序号×3 + 种子）：
#    0 baseline-yolo11n : yolo11n.yaml             原生基线
#    1 ab1-orb          : cfg/yolo11n-orb.yaml     +ORB-In
#    2 ab2-mrfe         : cfg/yolo11n-mrfe.yaml    +MRFE
#    3 ab3-lcr          : cfg/yolo11n-lcr.yaml     +LCR
#    4 ab3b-lcrbase     : cfg/yolo11n-lcrbase.yaml +LCRBase
#    5 ab5-orbmrfe      : cfg/yolo11n-orbmrfe.yaml ORB+MRFE
#    6 orblcrb          : cfg/yolo11n-orblcrb.yaml ORB+LCRBase
#    7 mrfelcrb         : cfg/yolo11n-mrfelcrb.yaml MRFE+LCRBase（最终模型）
#    8 ab4-rcr-fb       : cfg/yolo11n-rcrfb.yaml   三模块
#    9 rcr-full         : cfg/yolo11n-rcr.yaml     ORB+MRFE+LCR
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

VARIANTS=(
    "baseline-yolo11n:yolo11n.yaml"
    "ab1-orb:cfg/yolo11n-orb.yaml"
    "ab2-mrfe:cfg/yolo11n-mrfe.yaml"
    "ab3-lcr:cfg/yolo11n-lcr.yaml"
    "ab3b-lcrbase:cfg/yolo11n-lcrbase.yaml"
    "ab5-orbmrfe:cfg/yolo11n-orbmrfe.yaml"
    "orblcrb:cfg/yolo11n-orblcrb.yaml"
    "mrfelcrb:cfg/yolo11n-mrfelcrb.yaml"
    "ab4-rcr-fb:cfg/yolo11n-rcrfb.yaml"
    "rcr-full:cfg/yolo11n-rcr.yaml"
)
SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
VIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
IFS=: read -r VARIANT MODEL <<< "${VARIANTS[$VIDX]}"
NAME="${VARIANT}-s${SEED}"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - 300轮协议 [$NAME]"
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

DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"
if [ ! -f "$DATA" ]; then
    echo "错误: 未找到数据配置文件 $DATA，请先运行 data/make_coco_indoor.py！"
    exit 1
fi
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
# 30 任务并发时避免重复改写 data yaml（写前检查，减小竞争窗口）
grep -q "^path: $DATA_DIR$" "$DATA" || sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"

PROJECT="$PROJECT_ROOT/runs/rcr300"
mkdir -p "$PROJECT/$NAME"

# 断点保护：已完成跳过；被中断（有 last 无 best）则续训
if [ -f "$PROJECT/$NAME/weights/best.pt" ]; then
    echo "[$NAME] 已存在 best.pt，跳过"
    exit 0
fi

COMMON="--data $DATA --name $NAME --project $PROJECT \
    --epochs ${EPOCHS:-300} --batch ${BATCH:-64} --device ${DEVICE:-0} --seed $SEED"

if [ -f "$PROJECT/$NAME/weights/last.pt" ]; then
    echo "[$NAME] 检测到 last.pt，断点续训"
    python train.py $COMMON --model "$PROJECT/$NAME/weights/last.pt" --resume
else
    python train.py $COMMON --model "$MODEL"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "[$NAME] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[$NAME] 运行时长: $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
