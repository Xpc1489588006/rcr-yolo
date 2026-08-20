# RCR-YOLO · GPU 平台部署与运行手册

> 本文档记录 RCR-YOLO 在 SLURM GPU 平台上的完整部署流程、资源配置、以及本次上线过程中踩过的所有坑与修复方式，供后续复现与排查。
>
> 仓库：https://github.com/Xpc1489588006/rcr-yolo

---

## 1. 平台与项目概览

| 项目 | 值 |
|---|---|
| 登录节点 | `control06`（Ubuntu 22.04.5） |
| 计算节点示例 | `control03` |
| GPU | **NVIDIA A40（44GB 显存）**，分区 `p3` |
| 项目路径 | `/data/home/zhaozhanshan/rcr/rcr_yolo` |
| conda 环境 | `rcr` |
| Python | 3.12.13 |
| torch | 2.5.1+cu121 |
| ultralytics | 8.4.121 |

> ⚠️ 实际分配的 GPU 是 **A40（44G）**，不是最初假设的 4090（24G）。显存更充裕，batch 可开更大。

---

## 2. 环境搭建

### 2.1 Python 版本
推荐 **Python 3.10–3.13**（最佳 3.11/3.12）。本地曾用 3.14 验证通过，但 GPU 平台对 3.14 的 CUDA wheel 支持有限，不建议。

### 2.2 安装步骤
```bash
conda create -n rcr python=3.12 -y
conda activate rcr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/smoke_test.py        # 自检，5 步全过即环境 OK
```

### 2.3 opencv 必用 headless 版（关键坑）
无头服务器没有 `libGL.so.1`，而 ultralytics 会自动拉入 GUI 版 `opencv-python` 导致 `import cv2` 报错。必须替换：
```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install --force-reinstall opencv-python-headless
```
> 以后每次重装/升级 ultralytics 后，GUI 版可能被再次引入，需重复上面两条。

---

## 3. Git 同步流程

### 3.1 首次接入（已有手动上传的目录）
```bash
cd ~/rcr/rcr_yolo
git init -b main
git remote add origin https://github.com/Xpc1489588006/rcr-yolo.git
git fetch origin
git reset --hard origin/main     # 对齐远端；datasets/ 被 ignore，不受影响
```

### 3.2 日常更新（本地推送后）
```bash
cd ~/rcr/rcr_yolo
git pull origin main
```

### 3.3 换行符
- `.sh` 由 `.gitattributes` 强制 LF，克隆/拉取后在 Linux 可直接执行。
- 手动从 Windows 上传的文件会带 CRLF，导致 `git status` 全部显示 modified（纯噪声）。用 `git reset --hard origin/main` 对齐即可，**不要** `git add . && commit`。

### 3.4 本机推送遇到代理报错
本机 git 走代理 `127.0.0.1:7897`。若报 `Failed to connect to 127.0.0.1:7897` 或 `Connection reset`：
```bash
git -c http.proxy= -c https.proxy= push origin main   # 临时禁用代理（不改配置）
# 或代理可用时直接 git push origin main
```

---

## 4. 数据集

| 项 | 值 |
|---|---|
| 路径 | `datasets/coco_indoor/` |
| 规模 | ~15GB（已被 .gitignore，不入库，需单独同步或平台生成） |
| train | 94,451 图 / 604,943 框 |
| val | 3,970 图 / 25,982 框 |
| 类别 | 45 个室内类（nc=45） |
| 结构 | `datasets/coco_indoor/{train,val}/{images,labels}` |

### 4.1 平台已有 COCO2017 时重新生成（推荐，免传 15G）
```bash
python data/make_coco_indoor.py --coco /path/to/coco2017 --out datasets/coco_indoor --no-links
```

### 4.2 从 Windows 同步时清理旧缓存
```bash
find datasets -name "*.cache" -delete
```

### 4.3 yaml 路径修正（已自动）
Windows 生成的 yaml 是绝对路径；SLURM 脚本内置 `sed` 会把 `path` 改写为数据集目录的**绝对路径**，无需手动处理。

---

## 5. SLURM 资源配置

| 脚本 | GPU | CPU | 内存 | 时限 | 说明 |
|---|---|---|---|---|---|
| `train_baseline.sh` | gpu:1 | 8 | **64G** | 24h | 纯 YOLO11n 基线 |
| `train_full.sh` | gpu:1 | 8 | **64G** | 24h | 全量 RCR |
| `ablation.sh` | gpu:1 | 8 | **64G** | 24h | 数组 `0-6%2`，7 变体最多 2 并行 |
| `train_pk.sh` | gpu:1 | 8 | **64G** | 48h | PK 两阶段 |
| `eval_hard.sh` | gpu:1 | 8 | 16G | 6h | 困难目标评测 |

### 5.1 关键参数（默认值，可用环境变量覆盖）
- `BATCH=128`（A40 44G 显存，实测占 ~18.6G，充足）
- `workers=8`（train.py 默认；受 cpus-per-task=8 与内存双重约束，见 §7）
- `EPOCHS=150`、`DEVICE=0`

### 5.2 提交（必须在项目目录下，脚本用 `SLURM_SUBMIT_DIR` 定位）
```bash
cd ~/rcr/rcr_yolo
mkdir -p logs
sbatch slurm/train_baseline.sh
squeue -u $USER
```

---

## 6. 训练与监控

```bash
tail -f logs/baseline_*.out      # 训练日志
squeue -u $USER                  # 作业状态
seff <作业ID>                    # 结束后看 CPU/GPU/内存实际峰值
ls runs/rcr/<name>/weights/      # last.pt / best.pt
```

训练完成后评测：
```bash
sbatch slurm/eval_hard.sh
```

---

## 7. batch / worker / 内存 调优结论

### 7.1 batch 32 → 128 的实测提速
| | batch32 | batch128 |
|---|---|---|
| 吞吐 | 192 img/s | 256 img/s（×1.33） |
| 每 epoch | ~8.2 min | ~6.2 min |
| 150 epochs | ~20.5h（撞 24h 上限） | **~15.5h** |

提速约 **33%**（瓶颈转移到数据加载，故非 4×）。batch128 的核心价值是**避开 24h 超时**。

### 7.2 worker 上限
- **CPU 约束**：作业仅 8 核，worker ≤ 8。
- **内存约束**：batch128 下单 worker 约 4–5GB（prefetch_factor=4 + mosaic），8 worker ≈ 40G，已占 64G 大半。
- **结论**：保持 workers=8 即当前最优；想加 worker 必须同时提高 `--cpus-per-task` 和 `--mem`，收益有限不建议。

---

## 8. 踩坑与修复记录（排查索引）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | SLURM 激活环境失败 | 脚本写死旧环境 `xpc` | 改为 `conda activate rcr` |
| 2 | `git status` 全文件 modified | 手动上传带 CRLF | `git reset --hard origin/main` |
| 3 | `import cv2` 报 `libGL.so.1` 缺失 | ultralytics 拉入 GUI 版 opencv | 换 `opencv-python-headless` |
| 4 | `Dataset images not found ... rcr_yolo/val/images` | yaml `path: .` 被按运行目录解析 | `sed` 写入数据集目录**绝对路径** |
| 5 | batch128 时 CPU 内存 OOM | 8 worker × prefetch4 ≈ 30G 超 32G | `--mem` 32G→64G |
| 6 | 输出被套层 `runs/detect/runs/rcr/` | ultralytics 对相对 `--project` 前置 `runs/detect` | `--project` 改**绝对路径** `$PROJECT_ROOT/runs/rcr` |

---

## 9. 当前状态

- 基线（YOLO11n）已在 A40 上稳定训练：`GPU_mem ~18.6G`，batch=128，~6.7 min/epoch，150 epochs ≈ 15.5h。
- epoch1 mAP≈0 属正常（warmup），epoch5-10 起 mAP50 应开始爬升；COCO-indoor 上基线收敛参考区间 **mAP50 ≈ 0.55–0.65**。
- 基线跑完后可提交 `sbatch slurm/ablation.sh` 跑其余 6 个消融变体（会自动跳过已完成的 baseline）。
