# YOLOv10n 300 轮对比重训

YOLOv10n 使用与 YOLOv8n 相同的 300 轮协议：`epochs=300`、`patience=50`、`close_mosaic=15`、`batch=64`、`imgsz=640`、`lr0=0.01`、非 deterministic，输出目录为 `runs/rcr300/yolov10n-s<seed>/`。

模型配置为 Ultralytics 内置 `yolov10n.yaml`。本地环境已验证该配置可解析，n 规模参数量为 2,775,520；数据集加载时由 Ultralytics 将类别数覆盖为 COCO-indoor 的 45 类。

首次单种子提交（A40 shard）：

```bash
cd ~/rcr/rcr_yolo_c0ffed2
sbatch --array=0 -p p3 -w control03 \
  --gres=shard:NVIDIA_A40:12 \
  --cpus-per-task=8 --mem=64G --time=96:00:00 \
  --export=ALL,DATA=/data/home/zhaozhanshan/rcr/rcr_yolo/datasets/coco_indoor/coco_indoor.yaml \
  --job-name=t300_yolov10n-s0-shard12 \
  --output=logs/t300_yolov10n-s0-shard12.%j.out \
  --error=logs/t300_yolov10n-s0-shard12.%j.err \
  slurm/train300_yolov10n.sh
```

训练完成后沿用 `slurm/eval_hard.sh`，设置 `RUNS_DIR=runs/rcr300` 进行困难目标 v2 评测。
