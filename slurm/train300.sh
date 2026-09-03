#!/bin/bash
#SBATCH --job-name=rcr_t300
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00
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
#   .done 哨兵存在   -> 跳过（已完成）
#   仅有 last.pt     -> resume 续训（应对时限中断/掉卡/假 best.pt）
# 注意：ultralytics 第 1 轮验证后即写 best.pt，故 best.pt 不能作完成判据
# （18573 曾因此假跳过被杀作业的 1 轮权重）。
# 提交（推荐：单任务显式指定；本集群数组任务 ID 曾错乱串模型，见下）：
#   sbatch -w control06 --export=ALL,VARIANT_OVERRIDE=ab3-lcr,\
#       MODEL_OVERRIDE=cfg/yolo11n-lcr.yaml,SEED_OVERRIDE=0 slurm/train300.sh
# 全套 30 任务（仅集群数组机制验证可靠时使用）：
#   sbatch -w control06 --array=0-29%4 slurm/train300.sh
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
#
# 单任务提交（推荐）：通过环境变量显式指定，绕开数组任务号推导：
#   sbatch -w control06 --job-name=t300_ab3-lcr-s0 \
#       --output=logs/t300_ab3-lcr-s0.%j.out --error=logs/t300_ab3-lcr-s0.%j.err \
#       --export=ALL,VARIANT_OVERRIDE=ab3-lcr,MODEL_OVERRIDE=cfg/yolo11n-lcr.yaml,SEED_OVERRIDE=0 \
#       slurm/train300.sh
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
# 变体解析：优先环境变量显式指定（推荐）；否则按数组任务号推导。
# 2026-09-03 事故：本集群数组任务 ID 传递错乱（索引 9 的任务实际训练了
# 索引 21 的模型，且 scancel 后仍有分叉子任务继续写权重），数组方式慎用。
if [ -n "$VARIANT_OVERRIDE" ]; then
    VARIANT="$VARIANT_OVERRIDE"
    MODEL="$MODEL_OVERRIDE"
    SEED=${SEED_OVERRIDE:-0}
else
    SEED=$(( SLURM_ARRAY_TASK_ID % 3 ))
    VIDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
    IFS=: read -r VARIANT MODEL <<< "${VARIANTS[$VIDX]}"
fi
NAME="${VARIANT}-s${SEED}"

START_TIME=$(date +%s)

echo "============================================"
echo "  RCR-YOLO - 300轮协议 [$NAME]"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID:-N/A}/${SLURM_ARRAY_TASK_COUNT:-1})"
echo "运行节点:    ${SLURM_NODELIST}"
echo "模型配置:    ${MODEL}  seed=${SEED}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# 日志瘦身: 训练输出先落盘临时文件（保住 python 真实退出码，不影响 .done 哨兵语义），
# 结束后按 \r 只保留每段最后一行（即每个 epoch 的 100% 汇总行）再写入 .out。
# 避免每轮 1476 行进度条刷进日志（历史 300 轮日志曾达 68MB/文件，清洗后约 1.6MB）。
logstrip() {
    python -u -c '
import sys
with open(sys.argv[1], "rb") as f:
    for raw in f:
        text = raw.decode("utf-8", "replace")
        for seg in text.split("\n"):
            sys.stdout.write(seg.split("\r")[-1] + "\n")
' "$1"
}

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

# 断点保护：.done 哨兵才算完成；被中断（有 last 无 .done）则续训
if [ -f "$PROJECT/$NAME/.done" ]; then
    echo "[$NAME] 已完成（.done 哨兵存在），跳过"
    exit 0
fi

COMMON="--data $DATA --name $NAME --project $PROJECT \
    --epochs ${EPOCHS:-300} --batch ${BATCH:-64} --device ${DEVICE:-0} --seed $SEED"

TMPLOG=$(mktemp logs/.train_${NAME}.${SLURM_JOB_ID}.XXXX)
if [ -f "$PROJECT/$NAME/weights/last.pt" ]; then
    # 身份核验：last.pt 内部记录的 name 必须与本任务目录一致，防止交叉写入（2026-09-03 事故）
    CKPT_NAME=$(python -c "import torch; print(((torch.load('$PROJECT/$NAME/weights/last.pt', map_location='cpu', weights_only=False).get('train_args')) or {}).get('name',''))" 2>/dev/null)
    if [ "$CKPT_NAME" != "$NAME" ]; then
        echo "[$NAME] 中止：last.pt 内部身份为 '$CKPT_NAME'，与目录不符"
        rm -f "$TMPLOG"
        exit 1
    fi
    echo "[$NAME] 检测到 last.pt（身份核验通过：$CKPT_NAME），断点续训"
    echo "CMD: python -u train.py $COMMON --model $PROJECT/$NAME/weights/last.pt --resume"
    python -u train.py $COMMON --model "$PROJECT/$NAME/weights/last.pt" --resume > "$TMPLOG" 2>&1
else
    echo "CMD: python -u train.py $COMMON --model $MODEL"
    python -u train.py $COMMON --model "$MODEL" > "$TMPLOG" 2>&1
fi
RC=$?
logstrip "$TMPLOG"   # 清洗后的精简日志进 .out（SLURM stdout）
rm -f "$TMPLOG"
if [ $RC -eq 0 ]; then
    touch "$PROJECT/$NAME/.done"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "--------------------------------------------"
echo "[$NAME] 结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[$NAME] 运行时长: $((DURATION/3600))小时 $(((DURATION%3600)/60))分钟 $((DURATION%60))秒"
echo "============================================"
exit $RC
