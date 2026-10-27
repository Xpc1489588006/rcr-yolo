# YOLOv8n 300 轮对比重训

## 协议

与 RCR-YOLO 300 轮实验一致：`epochs=300`、`patience=50`、`close_mosaic=15`、`batch=64`、`imgsz=640`、`lr0=0.01`、非 deterministic，使用 seed 0/1/2，节点固定 `control03` 的 A40。

训练脚本：`slurm/train300_yolov8n.sh`

模型入口使用 Ultralytics 内置 `yolov8n.yaml`，输出目录为：

```text
runs/rcr300/yolov8n-s0/
runs/rcr300/yolov8n-s1/
runs/rcr300/yolov8n-s2/
```

## 提交

在服务器统一目录执行：

```bash
cd ~/rcr/rcr_yolo_c0ffed2
sbatch --array=0-2 -p p3 -w control03 \
  --gres=gpu:NVIDIA_A40:1 \
  --cpus-per-task=8 --mem=64G --time=96:00:00 \
  --export=ALL,DATA=/data/home/zhaozhanshan/rcr/rcr_yolo/datasets/coco_indoor/coco_indoor.yaml \
  --job-name=t300_yolov8n-A40-c0ffed2 \
  --output=logs/t300_yolov8n-A40-c0ffed2.%A_%a.out \
  --error=logs/t300_yolov8n-A40-c0ffed2.%A_%a.err \
  slurm/train300_yolov8n.sh
```

训练全部完成后，使用已有困难目标 v2 评测脚本：

```bash
sbatch -p p3 -w control03 --gres=gpu:NVIDIA_A40:1 \
  --export=ALL,DATA=/data/home/zhaozhanshan/rcr/rcr_yolo/datasets/coco_indoor/coco_indoor.yaml,RUNS_DIR=runs/rcr300 \
  slurm/eval_hard.sh
```

完成判据仍为各目录中的 `.done` 哨兵；`best.pt` 仅作为权重文件，不单独代表训练完成。
