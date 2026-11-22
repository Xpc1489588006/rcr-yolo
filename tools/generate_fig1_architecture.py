"""
生成 RCR-YOLO 架构图 (Fig. 1)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')

# 颜色方案
color_input = '#E8F4F8'
color_backbone = '#B3D9E6'
color_orb = '#FFE5CC'
color_neck = '#C9E4CA'
color_mrfe = '#FFD4D4'
color_lcr = '#E6D4F5'
color_detect = '#FFE6F0'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False, alpha=1.0):
    """绘制带文字的方框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          linewidth=2.5, edgecolor='black', facecolor=color, alpha=alpha)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, style='->', width=2.5, color='black'):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=25,
                           linewidth=width, color=color)
    ax.add_patch(arrow)

# 标题
ax.text(9, 10.5, 'RCR-YOLO Architecture', ha='center', va='center',
        fontsize=16, weight='bold')

# 1. 输入图像
draw_box(ax, 0.5, 8.5, 1.8, 1.2, 'Input\n640×640\nRGB Image', color_input, fontsize=11, bold=True)

# 箭头：Input -> Backbone
draw_arrow(ax, 2.3, 9.1, 3.0, 9.1)

# 2. Backbone (YOLO11n)
draw_box(ax, 3.0, 7.0, 2.5, 3.5, 'YOLO11n\nBackbone\n\nCSPDarknet\nC3k2\nC2PSA',
         color_backbone, fontsize=10, bold=True)

# Backbone 特征输出标注
feature_positions = [
    (5.5, 9.8, 'P2', '160×160'),
    (5.5, 9.0, 'P3', '80×80'),
    (5.5, 8.2, 'P4', '40×40'),
    (5.5, 7.4, 'P5', '20×20')
]

for x, y, name, size in feature_positions:
    draw_box(ax, x, y-0.25, 1.2, 0.5, f'{name}\n{size}', color_backbone, fontsize=9)
    # 箭头从Backbone指向特征
    draw_arrow(ax, 5.5, y, x, y, width=2.0)

# 3. ORB-In 模块 (从 P2 分支出来)
draw_box(ax, 7.2, 9.3, 2.0, 1.2, 'ORB-In\n(Training Only)\nRegion + Edge\nReconstruction',
         color_orb, fontsize=9, bold=True)
# 箭头：P2 -> ORB-In
draw_arrow(ax, 6.7, 9.55, 7.2, 9.9, width=2.0, color='orange')
# 标注：推理时旁路
ax.text(8.2, 10.7, 'Bypassed at\nInference', ha='center', va='center',
        fontsize=8, style='italic', color='darkred')

# 4. Neck (PANet)
draw_box(ax, 7.5, 6.0, 2.5, 2.5, 'Neck\n(PANet)\n\nC3k2\nC2PSA\n\nBottom-up &\nTop-down\nFusion',
         color_neck, fontsize=9, bold=True)

# 箭头：Backbone特征 -> Neck
for i, y in enumerate([9.55, 8.75, 7.95, 7.15]):
    draw_arrow(ax, 6.7, y, 7.5, 7.5-i*0.2, width=1.8)

# 5. MRFE 模块 (三个尺度)
mrfe_y_positions = [7.5, 6.5, 5.5]
mrfe_labels = ['MRFE\n(P3/16)', 'MRFE\n(P4/32)', 'MRFE\n(P5/64)']

for i, (y, label) in enumerate(zip(mrfe_y_positions, mrfe_labels)):
    draw_box(ax, 10.5, y-0.3, 1.8, 0.8, label, color_mrfe, fontsize=8, bold=True)
    # 箭头：Neck -> MRFE
    draw_arrow(ax, 10.0, y, 10.5, y, width=1.8)

# MRFE 内部细节标注
ax.text(11.4, 8.3, 'Multi-Receptive Field\nDilation rates: {1,2,3}\nDynamic Weighting',
        ha='center', va='center', fontsize=7, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=1))

# 6. LCRBase 模块 (三个尺度)
lcr_y_positions = [7.5, 6.5, 5.5]
lcr_labels = ['LCRBase\n(P3/16)', 'LCRBase\n(P4/32)', 'LCRBase\n(P5/64)']

for i, (y, label) in enumerate(zip(lcr_y_positions, lcr_labels)):
    draw_box(ax, 12.8, y-0.3, 1.8, 0.8, label, color_lcr, fontsize=8, bold=True)
    # 箭头：MRFE -> LCRBase
    draw_arrow(ax, 12.3, y, 12.8, y, width=1.8)

# LCRBase 内部细节标注
ax.text(13.7, 8.3, 'Scene Context\nMulti-scale Pooling\n{1×1, 2×2, 4×4}',
        ha='center', va='center', fontsize=7, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=1))

# 7. Detection Heads (三个尺度)
detect_y_positions = [7.5, 6.5, 5.5]
detect_labels = ['Detect Head\nP3/16\n(Class+Box)', 'Detect Head\nP4/32\n(Class+Box)', 'Detect Head\nP5/64\n(Class+Box)']

for i, (y, label) in enumerate(zip(detect_y_positions, detect_labels)):
    draw_box(ax, 15.0, y-0.3, 2.0, 0.8, label, color_detect, fontsize=8, bold=True)
    # 箭头：LCRBase -> Detection Head
    draw_arrow(ax, 14.6, y, 15.0, y, width=1.8)

# 8. 输出
draw_box(ax, 17.2, 6.0, 0.6, 2.5, 'Output', color_input, fontsize=10, bold=True, alpha=0.8)
ax.text(17.5, 4.8, 'Bounding Boxes\n+ Classes', ha='center', va='top', fontsize=8)

# 箭头：所有Detection Head -> Output
for y in [7.5, 6.5, 5.5]:
    draw_arrow(ax, 17.0, y, 17.2, 7.2, width=1.5, color='darkgreen')

# 添加模块标注框
# Training/Inference 标注
ax.text(1.0, 0.5, 'Training: ORB-In active (auxiliary supervision)\nInference: ORB-In bypassed (zero cost)',
        ha='left', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', linewidth=2))

# 图例
legend_x = 11.5
legend_y = 1.5
ax.text(legend_x, legend_y+0.8, 'Legend:', ha='left', va='top', fontsize=10, weight='bold')
legend_items = [
    (color_backbone, 'Backbone/Features'),
    (color_orb, 'ORB-In (Training)'),
    (color_neck, 'Neck (PANet)'),
    (color_mrfe, 'MRFE'),
    (color_lcr, 'LCRBase'),
    (color_detect, 'Detection Head')
]
for i, (color, text) in enumerate(legend_items):
    y_pos = legend_y + 0.4 - i*0.35
    rect = Rectangle((legend_x, y_pos-0.1), 0.3, 0.2, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(legend_x+0.4, y_pos, text, ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('figures/Fig1_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig1_architecture.png', dpi=300, bbox_inches='tight')
print("Fig. 1 architecture diagram generated:")
print("   - figures/Fig1_architecture.pdf")
print("   - figures/Fig1_architecture.png")
plt.close()
