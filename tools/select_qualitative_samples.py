"""
快速浏览验证集图像，选择定性对比样例
使用方向键浏览，按空格标记候选图像
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入 rcr 模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
import json
VAL_IMAGES = ROOT / "datasets" / "coco_indoor" / "val" / "images"
VAL_LABELS = ROOT / "datasets" / "coco_indoor" / "val" / "labels"
OUTPUT_JSON = ROOT / "tools" / "selected_samples.json"

# 加载三个模型
print("Loading models...")
baseline = YOLO(ROOT / "runs/rcr300/baseline-yolo11n-s0/weights/best.pt")
ab2 = YOLO(ROOT / "runs/rcr300/ab2-mrfe-s0/weights/best.pt")
ab4 = YOLO(ROOT / "runs/rcr300/ab4-rcr-fb-s0/weights/best.pt")

def count_gt_objects(label_path):
    """统计GT框数量和特征"""
    if not label_path.exists():
        return {"total": 0, "small": 0}

    lines = label_path.read_text().splitlines()
    total = len(lines)
    small = 0

    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            w, h = float(parts[3]), float(parts[4])
            area = w * h
            if area < (32*32)/(640*640):  # COCO small threshold
                small += 1

    return {"total": total, "small": small}

def quick_inference(image_path):
    """快速推理三个模型，返回检测框数量"""
    results_baseline = baseline(str(image_path), verbose=False)[0]
    results_ab2 = ab2(str(image_path), verbose=False)[0]
    results_ab4 = ab4(str(image_path), verbose=False)[0]

    return {
        "baseline": len(results_baseline.boxes),
        "ab2": len(results_ab2.boxes),
        "ab4": len(results_ab4.boxes)
    }

# 预筛选候选图像
print("Pre-screening candidates...")
image_files = sorted(VAL_IMAGES.glob("*.jpg"))[:100]  # 先看前100张

candidates = []
for img_path in image_files:
    label_path = VAL_LABELS / f"{img_path.stem}.txt"
    gt_stats = count_gt_objects(label_path)

    # 筛选条件：GT框数量在5-30之间（避免太空或太挤）
    if 5 <= gt_stats["total"] <= 30:
        det_stats = quick_inference(img_path)

        # 优先选择baseline检测框少但ab4多的（说明有改进）
        improvement = det_stats["ab4"] - det_stats["baseline"]

        candidates.append({
            "image_id": img_path.stem,
            "gt_total": gt_stats["total"],
            "gt_small": gt_stats["small"],
            "det_baseline": det_stats["baseline"],
            "det_ab4": det_stats["ab4"],
            "improvement": improvement,
            "path": str(img_path)
        })

# 按改进程度排序
candidates.sort(key=lambda x: x["improvement"], reverse=True)

print(f"\n找到 {len(candidates)} 个候选图像")
print("\n前10个最有改进的样例:")
print("-" * 80)
for i, c in enumerate(candidates[:10]):
    print(f"{i+1}. {c['image_id']}: GT={c['gt_total']} (small={c['gt_small']}), "
          f"baseline={c['det_baseline']}, ab4={c['det_ab4']}, "
          f"改进={c['improvement']:+d}")

# 保存候选列表
OUTPUT_JSON.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
print(f"\n候选列表已保存到: {OUTPUT_JSON}")

# 推荐的场景类型
print("\n推荐选择策略:")
print("1. 小目标密集: 选 gt_small > 5 且 improvement > 0 的")
print("2. 中等拥挤: 选 10 < gt_total < 20 的")
print("3. 严重拥挤: 选 gt_total > 20 的")
print("4. 混合尺度: 选 gt_small > 0 且 (gt_total - gt_small) > 5 的")
