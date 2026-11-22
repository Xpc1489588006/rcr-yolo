"""
生成定性对比图（Fig. 3）用于论文
根据选定的样例，对比 baseline, ab2-mrfe, ab4-rcr-fb 的检测效果
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 rcr 模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import json
VAL_IMAGES = ROOT / "datasets" / "coco_indoor" / "val" / "images"
VAL_LABELS = ROOT / "datasets" / "coco_indoor" / "val" / "labels"
OUTPUT_DIR = ROOT / "paper_ieee_draft" / "3" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载类别名称
with open(ROOT / "datasets" / "coco_indoor" / "coco_indoor.yaml", "r") as f:
    import yaml
    dataset_config = yaml.safe_load(f)
    CLASS_NAMES = dataset_config["names"]

# IEEE 风格绘图参数
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# 颜色方案
COLOR_GT = "#4CAF50"      # 绿色 - Ground Truth
COLOR_BASELINE = "#2196F3"  # 蓝色 - Baseline
COLOR_AB2 = "#FF9800"       # 橙色 - ab2-mrfe
COLOR_AB4 = "#F44336"       # 红色 - ab4-rcr-fb (Ours)

print("Loading models...")
baseline = YOLO(ROOT / "runs/rcr300/baseline-yolo11n-s0/weights/best.pt")
ab2 = YOLO(ROOT / "runs/rcr300/ab2-mrfe-s0/weights/best.pt")
ab4 = YOLO(ROOT / "runs/rcr300/ab4-rcr-fb-s0/weights/best.pt")
print("Models loaded successfully.")

def load_image_with_padding(image_path, target_size=(640, 480)):
    """加载图像并添加padding保持宽高比"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # 计算缩放比例
    scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # 缩放图像
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 创建带padding的画布
    canvas = Image.new("RGB", target_size, (255, 255, 255))
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    return np.array(canvas), scale, offset_x, offset_y

def plot_boxes_on_ax(ax, image, boxes, color, linewidth=1.5, show_labels=False,
                     conf_thresh=0.25):
    """在matplotlib axes上绘制检测框"""
    ax.imshow(image)
    ax.axis("off")

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())

        if conf < conf_thresh:
            continue

        # 绘制矩形
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1,
                        fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)

        # 可选：显示类别和置信度
        if show_labels and conf > 0.3:
            label = f"{CLASS_NAMES[cls][:8]} {conf:.2f}"
            ax.text(x1, y1-2, label,
                   bbox=dict(facecolor=color, alpha=0.6, edgecolor="none", pad=1),
                   fontsize=5.5, color="white", va="bottom")

def plot_gt_boxes_on_ax(ax, image, label_path, img_shape, color):
    """在matplotlib axes上绘制GT框"""
    ax.imshow(image)
    ax.axis("off")

    if not label_path.exists():
        return

    h, w = img_shape[:2]
    lines = label_path.read_text().splitlines()

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls, xc, yc, bw, bh = map(float, parts[:5])

        # YOLO格式转换为像素坐标
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        box_w = bw * w
        box_h = bh * h

        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), box_w, box_h,
                        fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)

def generate_comparison_for_image(image_id, output_name):
    """为单张图像生成四列对比图"""
    image_path = VAL_IMAGES / f"{image_id}.jpg"
    label_path = VAL_LABELS / f"{image_id}.txt"

    if not image_path.exists():
        print(f"Warning: Image {image_id} not found")
        return False

    # 加载原图
    img_np = np.array(Image.open(image_path).convert("RGB"))

    # 推理三个模型
    print(f"  Running inference on {image_id}...")
    results_baseline = baseline(str(image_path), verbose=False)[0]
    results_ab2 = ab2(str(image_path), verbose=False)[0]
    results_ab4 = ab4(str(image_path), verbose=False)[0]

    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # (a) Ground Truth
    plot_gt_boxes_on_ax(axes[0], img_np, label_path, img_np.shape, COLOR_GT)
    axes[0].set_title("(a) Ground Truth", pad=4)

    # (b) Baseline
    plot_boxes_on_ax(axes[1], img_np, results_baseline.boxes, COLOR_BASELINE)
    axes[1].set_title("(b) Baseline (YOLO11n)", pad=4)

    # (c) ab2-mrfe
    plot_boxes_on_ax(axes[2], img_np, results_ab2.boxes, COLOR_AB2)
    axes[2].set_title("(c) +MRFE", pad=4)

    # (d) ab4-rcr-fb (Ours)
    plot_boxes_on_ax(axes[3], img_np, results_ab4.boxes, COLOR_AB4)
    axes[3].set_title("(d) RCR-YOLO (Ours)", pad=4)

    plt.tight_layout()

    # 保存为多种格式
    for ext in ["pdf", "png", "svg"]:
        output_path = OUTPUT_DIR / f"{output_name}.{ext}"
        fig.savefig(output_path, dpi=300 if ext == "png" else None,
                   bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close(fig)
    return True

def generate_combined_figure(image_ids, scenario_names, output_name="Fig3_qualitative_comparison"):
    """生成包含多个场景的垂直拼接大图"""
    print(f"\nGenerating combined figure with {len(image_ids)} scenarios...")

    n_scenarios = len(image_ids)
    fig, axes = plt.subplots(n_scenarios, 4, figsize=(14, 3.5 * n_scenarios))

    # 确保axes是2D数组
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (image_id, scenario_name) in enumerate(zip(image_ids, scenario_names)):
        image_path = VAL_IMAGES / f"{image_id}.jpg"
        label_path = VAL_LABELS / f"{image_id}.txt"

        if not image_path.exists():
            print(f"  Warning: {image_id} not found, skipping")
            continue

        print(f"  Processing scenario {row_idx+1}: {scenario_name} ({image_id})")

        img_np = np.array(Image.open(image_path).convert("RGB"))

        # 推理
        results_baseline = baseline(str(image_path), verbose=False)[0]
        results_ab2 = ab2(str(image_path), verbose=False)[0]
        results_ab4 = ab4(str(image_path), verbose=False)[0]

        # 绘制四列
        plot_gt_boxes_on_ax(axes[row_idx, 0], img_np, label_path, img_np.shape, COLOR_GT)
        plot_boxes_on_ax(axes[row_idx, 1], img_np, results_baseline.boxes, COLOR_BASELINE)
        plot_boxes_on_ax(axes[row_idx, 2], img_np, results_ab2.boxes, COLOR_AB2)
        plot_boxes_on_ax(axes[row_idx, 3], img_np, results_ab4.boxes, COLOR_AB4)

        # 添加行标签（场景名称）
        axes[row_idx, 0].text(-0.15, 0.5, scenario_name,
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=8, fontweight="bold", rotation=90,
                             va="center", ha="right")

        # 第一行添加列标题
        if row_idx == 0:
            axes[0, 0].set_title("(a) Ground Truth", pad=6, fontweight="bold")
            axes[0, 1].set_title("(b) Baseline", pad=6, fontweight="bold")
            axes[0, 2].set_title("(c) +MRFE", pad=6, fontweight="bold")
            axes[0, 3].set_title("(d) Ours", pad=6, fontweight="bold")

    plt.tight_layout()

    # 保存
    for ext in ["pdf", "png", "svg", "tiff"]:
        output_path = OUTPUT_DIR / f"{output_name}.{ext}"
        dpi = 600 if ext in ["png", "tiff"] else None
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close(fig)
    print(f"\nCombined figure saved successfully!")

if __name__ == "__main__":
    # 根据场景覆盖要求精心选择的典型场景
    # 基于 selected_samples.json 的分析结果

    selected_json = ROOT / "tools" / "selected_samples.json"
    if selected_json.exists():
        print("Found selected_samples.json, loading...")
        with open(selected_json, "r") as f:
            candidates = json.load(f)

        # 展示top候选
        print("\nTop candidates analysis:")
        for i, c in enumerate(candidates[:8]):
            print(f"{i+1}. {c['image_id']}: improvement={c['improvement']:+d}, "
                  f"GT={c['gt_total']}, small={c['gt_small']}")

        # 精简为3个场景：保留表现最好且确认为室内的场景
        # 最终版：只保留改进显著且确认为室内的场景
        selected_cases = [
            (candidates[0]["image_id"], "Maximum improvement"),     # 000000009891: GT=8, baseline:4→ab4:7 (+3, 75%提升)
            ("000000003934", "Small objects dense"),                # GT=14, small=6, 小目标密集室内场景
            ("000000018380", "Heavy occlusion"),                    # GT=53, baseline:27→ab4:34 (+7, 26%提升)
        ]
    else:
        # 如果没有预选文件，使用默认选择
        selected_cases = [
            ("000000009891", "Maximum improvement"),
            ("000000007281", "Small objects dense"),
            ("000000002685", "Crowded + mixed scale"),
            ("000000018380", "Heavy occlusion"),
        ]

    print("\n" + "="*60)
    print("Selected scenarios for qualitative comparison:")
    print("="*60)
    for idx, (img_id, scenario) in enumerate(selected_cases, 1):
        print(f"{idx}. {scenario:25s} - Image {img_id}")

    # 生成单独的对比图
    print("\n" + "="*60)
    print("Generating individual comparison figures...")
    print("="*60)
    for idx, (image_id, scenario) in enumerate(selected_cases):
        print(f"\n[{idx+1}/{len(selected_cases)}] {scenario} ({image_id})")
        generate_comparison_for_image(image_id, f"qualitative_{idx+1}_{scenario.replace(' ', '_')}")

    # 生成组合大图
    print("\n" + "="*60)
    print("Generating combined figure...")
    print("="*60)
    image_ids = [case[0] for case in selected_cases]
    scenario_names = [case[1] for case in selected_cases]
    generate_combined_figure(image_ids, scenario_names)

    print("\n" + "="*60)
    print("✓ All qualitative figures generated successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60)
