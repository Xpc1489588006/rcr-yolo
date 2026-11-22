"""
MRFE注意力权重可视化工具
生成论文 Fig. 4 - 展示MRFE模块如何为不同场景动态分配感受野权重
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO

# 配置
VAL_IMAGES = ROOT / "datasets" / "coco_indoor" / "val" / "images"
MODEL_PATH = ROOT / "runs/rcr300/ab4-rcr-fb-s0/weights/best.pt"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置matplotlib为IEEE论文风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'figure.dpi': 300,
})


class MRFEAttentionExtractor:
    """提取MRFE模块的注意力权重"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.mrfe_modules = []
        self.attention_weights = {}
        self._find_mrfe_modules()
        self._register_hooks()

    def _find_mrfe_modules(self):
        """找到模型中所有MRFE模块"""
        for name, module in self.model.model.named_modules():
            if module.__class__.__name__ == 'MRFE':
                self.mrfe_modules.append((name, module))
        print(f"Found {len(self.mrfe_modules)} MRFE modules")

    def _register_hooks(self):
        """注册前向钩子以捕获MRFE gate输出"""
        for name, module in self.mrfe_modules:
            # 在MRFE的gate.Softmax层注册钩子
            if hasattr(module, 'gate') and len(module.gate) > 0:
                # gate是Sequential，最后一层是Softmax
                softmax_layer = module.gate[-1]
                softmax_layer.register_forward_hook(
                    self._make_hook(name)
                )

    def _make_hook(self, name):
        def hook(module, input, output):
            # output是softmax后的权重 (B, n_branches)
            self.attention_weights[name] = output.detach().cpu()
        return hook

    def extract_weights(self, image_path):
        """对单张图像提取MRFE注意力权重"""
        self.attention_weights.clear()

        # 推理
        with torch.no_grad():
            results = self.model(str(image_path), verbose=False)

        # 收集所有MRFE层的权重
        weights_list = []
        for name, _ in self.mrfe_modules:
            if name in self.attention_weights:
                w = self.attention_weights[name]  # (1, 3)
                weights_list.append(w[0].numpy())  # (3,)
            else:
                print(f"Warning: No weights captured for {name}")

        # 返回 shape: (n_layers, 3)
        if weights_list:
            return np.array(weights_list)
        else:
            print("Error: No attention weights were captured!")
            print(f"Available keys: {list(self.attention_weights.keys())}")
            return None


def select_representative_samples():
    """选择代表性样例：小目标、遮挡、拥挤等场景"""
    # 基于之前的selected_samples.json或手动选择
    selected_samples_json = ROOT / "tools" / "selected_samples.json"

    if selected_samples_json.exists():
        import json
        candidates = json.loads(selected_samples_json.read_text())

        # 选择策略：
        # 1. 小目标密集 (small > 5)
        # 2. 中等拥挤 (10 < total < 20)
        # 3. 严重拥挤 (total > 20)
        # 4. 混合尺度

        samples = {
            'small_dense': None,
            'medium_crowd': None,
            'heavy_crowd': None,
            'mixed_scale': None,
        }

        for c in candidates:
            if samples['small_dense'] is None and c['gt_small'] > 5:
                samples['small_dense'] = c['image_id']
            elif samples['medium_crowd'] is None and 10 < c['gt_total'] < 20:
                samples['medium_crowd'] = c['image_id']
            elif samples['heavy_crowd'] is None and c['gt_total'] > 20:
                samples['heavy_crowd'] = c['image_id']
            elif samples['mixed_scale'] is None and c['gt_small'] > 0 and (c['gt_total'] - c['gt_small']) > 5:
                samples['mixed_scale'] = c['image_id']

        return [v for v in samples.values() if v is not None]

    else:
        # 默认样例（从COCO-indoor验证集中手动选择）
        print("未找到selected_samples.json，使用默认样例")
        print("建议先运行 select_qualitative_samples.py")

        # 返回验证集前几张作为示例
        all_images = sorted(VAL_IMAGES.glob("*.jpg"))
        return [img.stem for img in all_images[:6]]


def plot_attention_heatmap(weights, image, title, ax):
    """
    绘制单个样例的MRFE注意力热力图

    Args:
        weights: (n_layers, 3) - 每层MRFE的三个分支权重
        image: 输入图像
        title: 子图标题
        ax: matplotlib axes
    """
    # 权重热力图
    im = ax.imshow(weights.T, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # 设置标签
    ax.set_xlabel('MRFE Layer Index')
    ax.set_ylabel('Branch (Dilation Rate)')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['d=1 (Small RF)', 'd=2 (Medium RF)', 'd=3 (Large RF)'])
    ax.set_title(title, fontsize=9, pad=8)

    # 在每个单元格中显示权重值
    for i in range(weights.shape[1]):  # 3 branches
        for j in range(weights.shape[0]):  # n_layers
            text = ax.text(j, i, f'{weights[j, i]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    return im


def create_figure4(extractor, sample_ids, output_path):
    """
    生成论文 Figure 4: MRFE Attention Visualization

    布局：
    - 每行一个样例
    - 列1: 原图缩略图
    - 列2: 注意力权重热力图
    - 列3: 权重分布柱状图
    """
    n_samples = len(sample_ids)

    fig = plt.figure(figsize=(10, 2.5 * n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, width_ratios=[1, 2, 1.5],
                  hspace=0.4, wspace=0.3)

    for idx, img_id in enumerate(sample_ids):
        img_path = VAL_IMAGES / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"图像不存在: {img_path}")
            continue

        # 提取注意力权重
        weights = extractor.extract_weights(img_path)
        if weights is None:
            print(f"未能提取权重: {img_id}")
            continue

        # 读取图像
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 列1: 原图
        ax_img = fig.add_subplot(gs[idx, 0])
        ax_img.imshow(img_rgb)
        ax_img.axis('off')
        ax_img.set_title(f'({chr(97+idx)}) Sample {idx+1}', fontsize=9)

        # 列2: 热力图
        ax_heat = fig.add_subplot(gs[idx, 1])
        im = plot_attention_heatmap(weights, img_rgb,
                                     f'Layer-wise Branch Weights', ax_heat)

        # 列3: 平均权重柱状图
        ax_bar = fig.add_subplot(gs[idx, 2])
        mean_weights = weights.mean(axis=0)  # (3,)
        bars = ax_bar.bar(['d=1', 'd=2', 'd=3'], mean_weights,
                          color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel('Average Weight')
        ax_bar.set_title('Mean Branch Weight', fontsize=9)
        ax_bar.grid(axis='y', alpha=0.3)

        # 在柱子上标注数值
        for bar, w in zip(bars, mean_weights):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{w:.3f}', ha='center', va='bottom', fontsize=7)

    # 全局colorbar
    fig.colorbar(im, ax=fig.get_axes()[1::3], fraction=0.046, pad=0.04,
                 label='Softmax Weight')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {output_path}")
    plt.close()


def create_simplified_figure4(extractor, sample_ids, output_path):
    """
    简化版Figure 4: 只显示平均权重对比
    适合论文空间有限时使用
    """
    n_samples = len(sample_ids)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for idx, img_id in enumerate(sample_ids[:6]):  # 最多6个样例
        img_path = VAL_IMAGES / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        weights = extractor.extract_weights(img_path)
        if weights is None:
            continue

        # 读取图像缩略图
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (160, 120))

        ax = axes[idx]

        # 创建子图中的子图布局
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        ax_img = divider.append_axes("top", size="40%", pad=0.1)

        # 显示缩略图
        ax_img.imshow(img_small)
        ax_img.axis('off')

        # 显示权重柱状图
        mean_weights = weights.mean(axis=0)
        bars = ax.bar(['d=1', 'd=2', 'd=3'], mean_weights,
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Weight')
        ax.set_title(f'({chr(97+idx)}) Sample {idx+1}', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        for bar, w in zip(bars, mean_weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{w:.2f}', ha='center', va='bottom', fontsize=7)

    # 隐藏多余子图
    for idx in range(len(sample_ids), 6):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"简化图表已保存: {output_path}")
    plt.close()


def generate_statistics_table(extractor, sample_ids):
    """生成权重统计表格，用于论文或补充材料"""
    stats = []

    for img_id in sample_ids:
        img_path = VAL_IMAGES / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        weights = extractor.extract_weights(img_path)
        if weights is None:
            continue

        mean_w = weights.mean(axis=0)
        std_w = weights.std(axis=0)

        stats.append({
            'image_id': img_id,
            'mean_d1': mean_w[0],
            'mean_d2': mean_w[1],
            'mean_d3': mean_w[2],
            'std_d1': std_w[0],
            'std_d2': std_w[1],
            'std_d3': std_w[2],
            'dominant_branch': ['d=1', 'd=2', 'd=3'][mean_w.argmax()]
        })

    # 保存为CSV
    import pandas as pd
    df = pd.DataFrame(stats)
    csv_path = OUTPUT_DIR / "mrfe_attention_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n统计表格已保存: {csv_path}")
    print(df.to_string(index=False))

    return df


def main():
    print("=" * 70)
    print("MRFE注意力权重可视化工具")
    print("=" * 70)

    # 检查模型路径
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        print("请检查模型路径或先完成训练")
        return

    print(f"\n加载模型: {MODEL_PATH.name}")
    extractor = MRFEAttentionExtractor(MODEL_PATH)

    print("\n选择代表性样例...")
    sample_ids = select_representative_samples()
    print(f"已选择 {len(sample_ids)} 个样例: {sample_ids}")

    # 生成完整版Figure 4
    print("\n生成Figure 4 (完整版)...")
    output_full = OUTPUT_DIR / "Fig4_mrfe_attention_full.pdf"
    create_figure4(extractor, sample_ids[:4], output_full)  # 限制4个样例

    # 生成简化版
    print("\n生成Figure 4 (简化版)...")
    output_simple = OUTPUT_DIR / "Fig4_mrfe_attention_simple.pdf"
    create_simplified_figure4(extractor, sample_ids, output_simple)

    # 生成统计表
    print("\n生成统计表格...")
    df_stats = generate_statistics_table(extractor, sample_ids)

    print("\n" + "=" * 70)
    print("完成! 输出文件:")
    print(f"  - {output_full}")
    print(f"  - {output_simple}")
    print(f"  - {OUTPUT_DIR}/mrfe_attention_statistics.csv")
    print("=" * 70)

    # 论文写作建议
    print("\n论文写作建议:")
    print("------------------------------------------------------------------")
    print("Fig. 4 Caption建议:")
    print("""
Fig. 4. MRFE adaptive attention weights across different indoor scenarios.
(a-d) Four representative validation samples. Left: input image; Middle:
layer-wise branch weights (d=1,2,3 correspond to small/medium/large receptive
fields); Right: mean weight distribution. The adaptive weighting demonstrates
MRFE's ability to emphasize appropriate receptive fields based on scene
characteristics—small RF for dense tiny objects, large RF for occluded regions.
    """)
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
