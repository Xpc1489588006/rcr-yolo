#!/bin/bash
#SBATCH --job-name=rcr_evalhard
#SBATCH --partition=p3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=logs/evalhard_%j.out
#SBATCH --error=logs/evalhard_%j.err

set -o pipefail  # 让管道内 python 崩溃能被 if ! 捕获

# ============================================================
# 困难目标批量评测：对 runs/rcr/ 下所有已完成 run 计算
# AP50 / AP75 / AP_S / AP_M / AP_L / 遮挡分桶 AP（free/partial/heavy）
# 结果追加写入 runs/rcr/eval_hard_results.txt
# 自动断点续跑：已评测的模型跳过、崩溃残留的不完整块自动清理，
# 任一模型评测失败立即中止（重提交即可继续）。
# 提交：sbatch slurm/eval_hard.sh（训练作业全部完成后）
# ============================================================

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_ROOT"

echo "============================================"
echo "  RCR-YOLO - 困难目标批量评测"
echo "============================================"
echo "GPU 型号:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "作业 ID:     ${SLURM_JOB_ID}"
echo "开始时间:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

mkdir -p logs

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh 2>/dev/null && conda activate rcr 2>/dev/null || \
  source /data/home/zhaozhanshan/ENTER/bin/activate rcr 2>/dev/null || true

DATA="${DATA:-datasets/coco_indoor/coco_indoor.yaml}"
if [ ! -f "$DATA" ]; then
    echo "错误: 未找到数据配置文件 $DATA"
    exit 1
fi
DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
sed -i "s|^path: .*|path: $DATA_DIR|" "$DATA"

OUT=runs/rcr/eval_hard_results.txt

# 清理上次崩溃残留的不完整块（完整块第 3 行必含 AP50=）
if [ -f "$OUT" ]; then
    awk '
      /^===== .* =====$/ { if (ok) for (i = 0; i < n; i++) print b[i]; n = 0; ok = 0 }
      { b[n++] = $0; if (n == 3 && $0 ~ /AP50=/) ok = 1 }
      END { if (ok) for (i = 0; i < n; i++) print b[i] }
    ' "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
fi

for d in runs/rcr/*/weights/best.pt; do
    [ -f "$d" ] || { echo "没有可评测的权重，先提交训练作业"; exit 0; }
    NAME=$(basename "$(dirname "$(dirname "$d")")")
    if grep -A 2 "===== $NAME =====" "$OUT" 2>/dev/null | grep -q "AP50="; then
        echo "[$NAME] 已评测过，跳过"
        continue
    fi
    echo "===== EVAL-HARD: $NAME ====="
    if ! {
        echo "===== $NAME ====="
        python eval_hard.py --weights "$d" --data "$DATA" --device "${DEVICE:-0}"
    } | tee -a "$OUT"; then
        echo "错误: $NAME 评测失败（如 GPU 掉卡），中止后续评测。重新提交即可断点续跑。"
        exit 1
    fi
done

echo "--------------------------------------------"
echo "全部评测完成，汇总结果: $OUT"
echo "============================================"
