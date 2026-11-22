# RCR-YOLO

面向室内非结构化场景困难目标（小目标、遮挡、拥挤场景）的轻量级目标检测研究：基于 YOLO11n 的 ORB-In / MRFE / LCRBase 三模块改进（RCR-YOLO）。

## 目录结构

| 目录 | 说明 |
|---|---|
| `rcr/` | 核心模块代码（ORB-In、MRFE、LCRBase、trainer 补丁） |
| `cfg/` | 模型结构 YAML（基线与消融组合） |
| `data/` | 数据集构建脚本（SUN RGB-D → COCO-indoor） |
| `datasets/` | 数据集存放（git 忽略；`coco_indoor` 约 15GB，`sunrgbd_raw` 为原始压缩包） |
| `slurm/` | 集群训练 / 评估 / 统计作业脚本 |
| `scripts/` | 本地辅助脚本 |
| `tools/` | 论文图表与统计工具（图生成、指标核验、统计检验） |
| `tests/` | 冒烟测试 |
| `vis/` | 可视化脚本 |
| `paper/` | 论文工作区：`paper_english.tex`（英文最终版）、`paper_chinese.tex`（中文版）、`figures/`、`tables/` |
| `docs/` | `manuals/`（部署与运维手册）、`records/`（实验记录）、`reports/`（过程报告，git 忽略） |
| `results/` | 核验指标（verified_metrics）与统计检验结果 |
| `runs/` | 训练输出（`rcr300/` 为 300 轮正式结果；git 忽略，权重经 LFS） |
| `logs/` | 训练日志（`t300/` 为 300 轮瘦身日志） |
| `reference/` | 参考文献 PDF（git 忽略） |

## 快速入口

- 训练（300 轮全量协议）：`slurm/train300.sh`；本地入口 `train.py`
- 困难子集评估：`eval_hard.py`
- 数据准备：`data/make_coco_indoor.py`（SUN RGB-D → coco_indoor）
- 论文编译：英文版 `cd paper && latexmk -pdf paper_english.tex`（pdfLaTeX）；中文版 `latexmk -xelatex paper_chinese.tex`（XeLaTeX）
- 部署与运维：`docs/manuals/`
