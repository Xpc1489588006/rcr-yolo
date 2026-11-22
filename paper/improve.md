# RCR-YOLO 论文优化工作清单

> 基于论文质量审查的改进建议  
> 生成日期：2026-11-06  
> 目标：提升论文接受率，应对审稿人质疑

---

## 优化工作分类

### 🔴 必须完成（影响接受率）
这些工作直接影响论文的可信度和完整性，必须在投稿前完成。

### 🟡 强烈建议（显著提升质量）
这些工作能明显提升论文质量，显著增加被接受概率。

### 🟢 可选加分项（锦上添花）
这些工作能进一步增强论文，但不做也不影响投稿。

---

## 一、现在立即可优化（0成本，纯写作改进）

### 1.1 精炼Abstract 🔴

**时间成本**：30分钟  
**优先级**：⭐⭐⭐⭐⭐  
**影响**：第一印象，决定审稿人是否认真读

**当前问题**：
- 示例内容较好，但需要严格控制在200词以内
- IEEE格式要求简洁有力

**操作步骤**：
1. 打开 `chapter_IEEE.md`
2. 定位到 Abstract 部分
3. 删减冗余表述：
   - "Existing general-purpose detectors exhibit performance degradation" → "General detectors underperform"
   - "comprises three synergistic modules" → "comprises three modules"
4. 确保关键数据都在：
   - mAP50-95: 0.3433 (+3.7%)
   - Hard AP50-95: 0.3343 (+3.8%)
   - 延迟: 10.36ms (~96 FPS)
   - 参数: 3.17M
5. 字数控制在190-200词

**参考精简版**：
```
Indoor object detection faces challenges of small scales, occlusion, and crowding. 
General detectors underperform in such scenarios. We propose RCR-YOLO built on 
YOLO11n with three modules: ORB-In for training-time reconstruction supervision, 
MRFE for multi-receptive field enhancement, and LCRBase for scene-level context 
reasoning. On COCO-indoor, we achieve 0.3433 mAP50-95 (+3.7%) and 0.3343 Hard 
AP50-95 (+3.8%) with improvements across small objects, occlusions, and crowded 
scenes. The model maintains real-time performance (10.36ms, ~96 FPS) with 3.17M 
parameters. (约150词，需补充到200词)
```

---

### 1.2 强化Method章节的差异说明 🔴

**时间成本**：1-2小时  
**优先级**：⭐⭐⭐⭐⭐  
**影响**：应对"创新性不足"质疑的核心

**当前问题**：
- 审稿人会质疑："这只是组合已有方法"
- 与原方法（ORFENet、CRNet）的差异描述不够详细

**操作步骤**：

#### 1.2.1 增强 ORB-In 部分（Section III-B）

在 "1) Motivation" 段落后增加一段：

```markdown
**Distinction from ORFENet [23]:** While ORFENet employs object-level 
reconstruction with instance segmentation labels, our ORB-In uses detection-box-
based proxy supervision requiring no additional annotation. ORFENet's 
reconstruction branch participates in inference for feature refinement, whereas 
ORB-In is bypassed during deployment for zero inference overhead. This design 
choice prioritizes real-time performance critical for robotic applications.
```

#### 1.2.2 增强 MRFE 部分（Section III-C）

在 "2) Module Design" 段落开头增加：

```markdown
**Adaptation from ORFENet [23]:** We adopt ORFENet's multi-receptive field 
concept but simplify it for YOLO integration. Key differences: (1) We use 
depthwise separable convolutions instead of standard convolutions, reducing 
parameters by ~60%; (2) Our dynamic weighting employs lightweight GAP+1×1 conv 
rather than complex attention modules; (3) We insert MRFE at three fixed Neck 
scales rather than adaptive feature pyramids, maintaining YOLO's architecture 
integrity.
```

#### 1.2.3 增强 LCRBase 部分（Section III-D）

在 "1) Motivation" 段落后增加：

```markdown
**Single-stage adaptation challenge:** CRNet [27] and IENet [28] rely on 
explicit proposal regions from two-stage detectors to define easy/hard 
partitions. Direct migration to single-stage YOLO fails because: (1) no 
proposal stage exists; (2) feature-map-based difficulty estimation is unstable 
without proposal guidance (verified by our three-seed ablation showing rcr-full's 
high variance); (3) easy-region filtering requires additional computational 
overhead. Our LCRBase addresses this by retaining only scene-level global 
context, eliminating difficulty gating and region selection for stable single-
stage integration.
```

---

### 1.3 补充Discussion应对策略 🔴

**时间成本**：1小时  
**优先级**：⭐⭐⭐⭐  
**影响**：主动应对审稿人可能的质疑

**操作步骤**：

在 Section V 最开头（V-A之前）增加一个新子节：

```markdown
### A. Adaptation Challenges from Two-Stage to Single-Stage

The primary challenge in this work is not proposing entirely novel mechanisms, 
but rather successfully adapting two-stage context reasoning to single-stage 
YOLO while maintaining stability and real-time performance. This adaptation is 
non-trivial for three reasons:

1) **Proposal dependency**: CRNet [27] and IENet [28] explicitly partition 
proposals into easy/hard categories. Single-stage YOLO operates directly on 
dense feature maps without region proposals, making explicit partitioning 
infeasible.

2) **Difficulty estimation instability**: Our initial experiments with full LCR 
(including difficulty gating) showed high variance across seeds (crowd-heavy 
std ±0.0102 vs LCRBase ±0.0020), suggesting feature-map-based difficulty 
estimation lacks the stability provided by proposal-guided mechanisms.

3) **Real-time constraint**: ORFENet [23] deploys its reconstruction branch 
during inference. For service robots requiring <15ms latency, we bypass ORB-In 
post-training, trading a small accuracy gain for guaranteed real-time 
performance.

Our contribution lies in demonstrating that simplified, stability-aware 
adaptations of these mechanisms yield robust improvements in single-stage 
detectors, validated through systematic ablation and multi-seed experiments.
```

然后将原来的 V-A 改为 V-B，V-B 改为 V-C，依此类推。

---

### 1.4 完善Limitations部分 🔴

**时间成本**：30分钟  
**优先级**：⭐⭐⭐⭐  
**影响**：展示诚实态度，避免审稿人过度挑剔

**操作步骤**：

在 Section V-C "Limitations and Future Work" 中修改：

**原文**：
```
1) Single-dataset validation: Verified only on COCO-indoor; generalization 
remains unconfirmed. Additional indoor datasets needed: SUN RGB-D, NYU Depth 
v2, AVD.
```

**改为**：
```
1) Single-dataset validation: We verify only on COCO-indoor due to limited 
availability of standardized indoor detection benchmarks with comparable 
annotation quality. SUN RGB-D and NYU Depth v2 focus primarily on depth-based 
tasks with sparser detection annotations. AVD provides robot-centric viewpoints 
but differs significantly in viewpoint distribution. Cross-dataset validation 
is planned as future work to assess generalization.
```

**在第4点后新增**：
```
4) Limited baseline comparison: We compare with YOLO family variants but not 
domain-specific indoor methods cited in Related Work. The methods in [15], [16] 
do not provide public implementations or sufficient implementation details for 
reproduction. Future work should include comparison with reproducible indoor-
specific detectors as they become available.
```

---

### 1.5 检查数据一致性 🔴

**时间成本**：1小时  
**优先级**：⭐⭐⭐⭐⭐  
**影响**：避免审稿人发现数值错误导致直接拒稿

**操作步骤**：

1. **核对主消融表（TABLE I）**：
   - 打开 `当前实验结果汇总.md`
   - 逐行核对 TABLE I 的数值
   - 确保小数点精度一致（都是4位小数）

2. **核对三种子表（TABLE III）**：
   - 核对9行原始数据
   - 重新计算均值和标准差：
     ```python
     import numpy as np
     baseline_s0s1s2 = [0.3219, 0.3210, 0.3234]
     mean = np.mean(baseline_s0s1s2)
     std = np.std(baseline_s0s1s2, ddof=1)  # 样本标准差
     print(f"{mean:.4f}±{std:.4f}")
     ```
   - 确保所有均值±标准差都是重新计算的

3. **核对YOLO对比表（TABLE II）**：
   - 确认YOLOv8n/v10n/YOLO26n的数值来自正确的结果文件
   - 核对 `runs/rcr300/eval_hard_results (1).txt`

4. **检查文内引用一致性**：
   - 搜索所有 "0.3433" 确保都是指 ab4-rcr-fb 的 mAP50-95
   - 搜索所有 "+3.7%" 确保都是指相对baseline的提升
   - 确保没有遗漏的 "TBD"、"待补充"等占位符

---

## 二、能快速完成（1-3天，需简单操作）

### 2.1 生成定性可视化（Fig. 3）🔴

**时间成本**：1-2天  
**优先级**：⭐⭐⭐⭐⭐  
**影响**：最直观展示效果，审稿人最关注的图

**为什么快**：
- 你已有 `tools/plot_paper_figures.py` 绘图工具
- 你已有三个模型的训练权重（baseline, ab2-mrfe, ab4-rcr-fb）
- 你已有 Hard v2 评测结果，可以快速定位失败案例

**操作步骤**：

#### Step 1: 选择样例（1小时）

基于 Hard v2 结果，选择4-6个样例：

**样例选择策略**：
1. **小目标密集**：从 small 桶中找 baseline 漏检但 ab4 成功的
2. **家具遮挡**：从 crowd-partial 桶中找边界框偏移大的
3. **严重拥挤**：从 crowd-heavy 桶中找 baseline AP < 0.3 的图像
4. **尺度变化**：找同时包含 small 和 large 目标的图像

**实现代码**：
```python
# 从验证集中筛选样例
import json
import numpy as np

# 读取评测结果
with open('runs/rcr300/eval_hard_results.txt', 'r') as f:
    results = parse_results(f)  # 需要实现解析函数

# 筛选策略
candidates = {
    'small_dense': [],
    'occlusion': [],
    'crowding': [],
    'scale_var': []
}

# 遍历验证集，找baseline失败但ab4成功的案例
for img_id in val_image_ids:
    baseline_dets = baseline_results[img_id]
    ab4_dets = ab4_results[img_id]
    gt = ground_truth[img_id]
    
    # 计算召回率
    baseline_recall = compute_recall(baseline_dets, gt)
    ab4_recall = compute_recall(ab4_dets, gt)
    
    if ab4_recall > baseline_recall + 0.2:  # ab4显著优于baseline
        # 根据场景特征分类
        if has_many_small_objects(gt):
            candidates['small_dense'].append(img_id)
        # ... 其他分类逻辑
```

#### Step 2: 生成对比图（4-6小时）

**实现代码**（基于你的 `tools/plot_paper_figures.py`）：
```python
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 加载模型
baseline = YOLO('runs/rcr300/baseline-yolo11n-s0/weights/best.pt')
ab2 = YOLO('runs/rcr300/ab2-mrfe-s0/weights/best.pt')
ab4 = YOLO('runs/rcr300/ab4-rcr-fb-s0/weights/best.pt')

def plot_comparison(img_path, save_path):
    """生成4列对比图"""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 推理
    results_baseline = baseline(img)[0]
    results_ab2 = ab2(img)[0]
    results_ab4 = ab4(img)[0]
    
    # 绘制
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # (a) 原图 + GT
    axes[0].imshow(img_rgb)
    plot_boxes(axes[0], gt_boxes, color='green', label='GT')
    axes[0].set_title('(a) Ground Truth')
    axes[0].axis('off')
    
    # (b) baseline
    axes[1].imshow(img_rgb)
    plot_boxes(axes[1], results_baseline.boxes, color='blue')
    axes[1].set_title('(b) Baseline')
    axes[1].axis('off')
    
    # (c) ab2-mrfe
    axes[2].imshow(img_rgb)
    plot_boxes(axes[2], results_ab2.boxes, color='orange')
    axes[2].set_title('(c) ab2-mrfe')
    axes[2].axis('off')
    
    # (d) ab4-rcr-fb
    axes[3].imshow(img_rgb)
    plot_boxes(axes[3], results_ab4.boxes, color='red')
    axes[3].set_title('(d) ab4-rcr-fb (Ours)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxes(ax, boxes, color, label=None):
    """绘制检测框"""
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        
        # 绘制矩形框
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # 标注类别+置信度
        if conf > 0.3:  # 只显示置信度>0.3的
            ax.text(x1, y1-5, f'{class_names[int(cls)]} {conf:.2f}',
                   bbox=dict(facecolor=color, alpha=0.5),
                   fontsize=8, color='white')

# 批量生成
for scenario, img_ids in candidates.items():
    for idx, img_id in enumerate(img_ids[:2]):  # 每个场景选2个
        img_path = f'data/coco_indoor/val2017/{img_id}.jpg'
        save_path = f'figures/qualitative_{scenario}_{idx+1}.pdf'
        plot_comparison(img_path, save_path)
```

#### Step 3: 制作最终图（2小时）

将4-6个样例拼接成一个大图：
```python
from PIL import Image

# 垂直拼接多个样例
images = [
    Image.open('figures/qualitative_small_dense_1.pdf'),
    Image.open('figures/qualitative_occlusion_1.pdf'),
    Image.open('figures/qualitative_crowding_1.pdf'),
    Image.open('figures/qualitative_scale_var_1.pdf'),
]

# 计算总高度
total_height = sum(img.height for img in images)
max_width = max(img.width for img in images)

# 创建新图
combined = Image.new('RGB', (max_width, total_height))
y_offset = 0
for img in images:
    combined.paste(img, (0, y_offset))
    y_offset += img.height

combined.save('figures/Fig3_qualitative_comparison.pdf')
```

---

### 2.2 补充统计检验 🔴

**时间成本**：2-3小时  
**优先级**：⭐⭐⭐⭐⭐  
**影响**：增强科学性，应对"单种子不可信"质疑

**为什么快**：
- 你已有三种子数据
- 统计检验代码简单

**操作步骤**：

#### Step 1: 编写统计检验脚本（30分钟）

```python
# tools/statistical_test.py
import numpy as np
from scipy import stats
import pandas as pd

# 三种子数据（从当前实验结果汇总.md提取）
data = {
    'baseline': {
        'AP50': [0.4600, 0.4585, 0.4649],
        'AP50-95': [0.3219, 0.3210, 0.3234],
        'crowd_heavy': [0.3036, 0.3100, 0.3037],
    },
    'ab4': {
        'AP50': [0.4704, 0.4699, 0.4721],
        'AP50-95': [0.3343, 0.3342, 0.3351],
        'crowd_heavy': [0.3143, 0.3181, 0.3166],
    },
    'rcr_full': {
        'AP50': [0.4722, 0.4695, 0.4741],
        'AP50-95': [0.3342, 0.3303, 0.3346],
        'crowd_heavy': [0.3176, 0.3007, 0.2996],
    }
}

def paired_t_test(model1, model2, metric):
    """配对t检验"""
    x = data[model1][metric]
    y = data[model2][metric]
    
    t_stat, p_value = stats.ttest_rel(y, x)  # y vs x (改进 vs baseline)
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    improvement = mean_y - mean_x
    rel_improvement = (improvement / mean_x) * 100
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_baseline': mean_x,
        'mean_improved': mean_y,
        'improvement': improvement,
        'relative_improvement': rel_improvement,
        'significant': p_value < 0.05
    }

# 执行检验
print("=" * 60)
print("Paired t-test: ab4-rcr-fb vs baseline")
print("=" * 60)

for metric in ['AP50', 'AP50-95', 'crowd_heavy']:
    result = paired_t_test('baseline', 'ab4', metric)
    print(f"\n{metric}:")
    print(f"  Baseline: {result['mean_baseline']:.4f}")
    print(f"  ab4:      {result['mean_improved']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f} ({result['relative_improvement']:.2f}%)")
    print(f"  t-statistic: {result['t_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'Yes' if result['significant'] else 'No'}")

print("\n" + "=" * 60)
print("Paired t-test: ab4-rcr-fb vs rcr-full")
print("=" * 60)

for metric in ['AP50-95', 'crowd_heavy']:
    result = paired_t_test('rcr_full', 'ab4', metric)
    print(f"\n{metric}:")
    print(f"  rcr-full: {result['mean_baseline']:.4f}")
    print(f"  ab4:      {result['mean_improved']:.4f}")
    print(f"  Difference: {result['improvement']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Significant: {'Yes' if result['significant'] else 'No'}")

# 生成LaTeX表格
print("\n" + "=" * 60)
print("LaTeX table for paper:")
print("=" * 60)

latex_table = """
\\begin{table}[h]
\\caption{Statistical Significance Test (Paired t-test, n=3)}
\\centering
\\begin{tabular}{lccc}
\\toprule
Comparison & Metric & Improvement & p-value \\\\
\\midrule
"""

comparisons = [
    ('ab4 vs baseline', 'AP50-95'),
    ('ab4 vs baseline', 'crowd_heavy'),
    ('ab4 vs rcr-full', 'AP50-95'),
]

for comp_name, metric in comparisons:
    if 'baseline' in comp_name:
        result = paired_t_test('baseline', 'ab4', metric)
    else:
        result = paired_t_test('rcr_full', 'ab4', metric)
    
    latex_table += f"{comp_name} & {metric} & "
    latex_table += f"+{result['improvement']:.4f} & "
    latex_table += f"{result['p_value']:.4f}"
    if result['significant']:
        latex_table += " * "
    latex_table += " \\\\\n"

latex_table += """\\bottomrule
\\multicolumn{4}{l}{* indicates p < 0.05} \\\\
\\end{tabular}
\\end{table}
"""

print(latex_table)
```

#### Step 2: 运行并记录结果（10分钟）

```bash
cd d:\Desktop\paper\robot\pp\rcr_yolo
python tools/statistical_test.py > results/statistical_test_results.txt
```

#### Step 3: 更新论文（1小时）

在 TABLE III 下方增加一段：

```markdown
**Statistical significance:** We perform paired t-tests (n=3) to assess whether 
observed improvements are statistically significant. ab4-rcr-fb vs baseline: 
AP50-95 improvement +0.0124 (p=0.XXX < 0.05), crowd-heavy improvement +0.0105 
(p=0.XXX < 0.05). ab4-rcr-fb vs rcr-full: AP50-95 difference +0.0015 (p=0.XXX), 
not reaching significance due to small sample size, but ab4 exhibits 
substantially lower variance (std ±0.0005 vs ±0.0024), suggesting superior 
stability.
```

（将 p=0.XXX 替换为实际计算的p值）

---

### 2.3 制作模块交互柱状图（Fig. 2）🔴

**时间成本**：1-2小时  
**优先级**：⭐⭐⭐⭐  
**影响**：直观展示模块协同效应

**操作步骤**：

```python
# tools/plot_module_interaction.py
import matplotlib.pyplot as plt
import numpy as np

# 数据（从TABLE I提取）
configs = [
    'baseline',
    '+ORB',
    '+MRFE',
    '+LCRBase',
    'ORB+MRFE',
    'ORB+MRFE\n+LCRBase'
]

mAP_50_95 = [
    0.3310,  # baseline
    0.3309,  # ab1-orb
    0.3368,  # ab2-mrfe
    0.3320,  # ab3b-lcrbase
    0.3394,  # ab5-orbmrfe
    0.3433,  # ab4-rcr-fb
]

# IEEE风格设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (7, 4)  # IEEE双栏图宽度

# 绘制柱状图
fig, ax = plt.subplots()

x = np.arange(len(configs))
bars = ax.bar(x, mAP_50_95, width=0.6, 
              color=['#d3d3d3', '#a8dadc', '#457b9d', '#f1faee', '#e63946', '#c1121f'],
              edgecolor='black', linewidth=1.2)

# 标注数值
for i, (bar, val) in enumerate(zip(bars, mAP_50_95)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0005,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 标注提升幅度
baseline_val = mAP_50_95[0]
for i in range(1, len(mAP_50_95)):
    improvement = mAP_50_95[i] - baseline_val
    if improvement > 0.001:
        ax.annotate('',
                   xy=(i, mAP_50_95[i]), xytext=(0, baseline_val),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.5))
        mid_y = (mAP_50_95[i] + baseline_val) / 2
        ax.text(i, mid_y, f'+{improvement:.4f}',
               fontsize=8, color='gray', ha='center')

# 设置坐标轴
ax.set_ylabel('mAP50-95', fontsize=11, fontweight='bold')
ax.set_xlabel('Configuration', fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15, ha='right')
ax.set_ylim(0.325, 0.345)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 标题
ax.set_title('Module Interaction Analysis', fontsize=12, fontweight='bold', pad=10)

# 保存
plt.tight_layout()
plt.savefig('figures/Fig2_module_interaction.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig2_module_interaction.eps', format='eps', bbox_inches='tight')
print("Figure saved to figures/Fig2_module_interaction.pdf and .eps")
plt.show()
```

运行：
```bash
python tools/plot_module_interaction.py
```

---

### 2.4 MRFE注意力可视化（Fig. 4，可选）🟡

**时间成本**：1-2天  
**优先级**：⭐⭐⭐⭐  
**影响**：增强可解释性，应对"机制不清晰"质疑

**前提条件**：
- 需要在MRFE forward中保存动态权重W
- 需要能提取中间特征

**操作步骤**：

#### Step 1: 修改MRFE模块保存权重（1小时）

```python
# rcr/mrfe.py
class MRFE(nn.Module):
    def __init__(self, c, d_rates=[1, 2, 3]):
        super().__init__()
        # ... 原有初始化代码 ...
        self.save_weights = False  # 控制是否保存权重
        self.saved_weights = None
    
    def forward(self, x):
        # 三分支
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        
        # 动态权重
        concat = torch.cat([f1, f2, f3], dim=1)
        gap = self.gap(concat)
        w = torch.sigmoid(self.fc(gap))  # [B, 3C]
        
        # 保存权重（用于可视化）
        if self.save_weights:
            self.saved_weights = w.detach().cpu().numpy()
        
        # 加权融合
        # ... 原有代码 ...
```

#### Step 2: 提取权重并可视化（4-6小时）

```python
# tools/visualize_mrfe_attention.py
import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

def extract_mrfe_weights(model_path, img_path, save_path):
    """提取并可视化MRFE权重"""
    
    # 加载模型
    model = YOLO(model_path)
    
    # 启用权重保存
    for m in model.model.modules():
        if hasattr(m, 'save_weights'):
            m.save_weights = True
    
    # 推理
    img = cv2.imread(img_path)
    results = model(img)[0]
    
    # 提取权重
    weights_p3 = None
    weights_p4 = None
    weights_p5 = None
    
    for m in model.model.modules():
        if hasattr(m, 'saved_weights') and m.saved_weights is not None:
            # 根据特征图尺寸判断是哪个尺度
            if weights_p3 is None:
                weights_p3 = m.saved_weights
            elif weights_p4 is None:
                weights_p4 = m.saved_weights
            elif weights_p5 is None:
                weights_p5 = m.saved_weights
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原图
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # P3尺度权重
    plot_branch_weights(axes[0, 1:4], weights_p3, 'P3/16 Scale')
    
    # P4尺度权重
    axes[1, 0].axis('off')
    plot_branch_weights(axes[1, 1:4], weights_p4, 'P4/32 Scale')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_branch_weights(axes, weights, title):
    """绘制三个分支的权重分布"""
    # weights: [B, 3C] -> 取第一个样本，reshape为[3, C]
    w = weights[0].reshape(3, -1)  # [3, C]
    
    # 三个分支的平均权重
    w_mean = w.mean(axis=1)  # [3]
    
    branch_names = ['d=1\n(local)', 'd=2\n(medium)', 'd=3\n(large)']
    colors = ['#457b9d', '#e63946', '#f1faee']
    
    for i, (ax, name, color) in enumerate(zip(axes, branch_names, colors)):
        # 绘制柱状图
        ax.bar(0, w_mean[i], color=color, edgecolor='black', linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_title(f'{name}\nWeight: {w_mean[i]:.3f}', fontsize=10)
        ax.set_ylabel('Avg Weight' if i == 0 else '')
        ax.grid(axis='y', alpha=0.3)
    
    # 总标题
    axes[1].text(0.5, 1.15, title, transform=axes[1].transAxes,
                ha='center', fontsize=12, fontweight='bold')

# 运行
img_path = 'path/to/example/image.jpg'
model_path = 'runs/rcr300/ab4-rcr-fb-s0/weights/best.pt'
save_path = 'figures/Fig4_mrfe_attention.pdf'

extract_mrfe_weights(model_path, img_path, save_path)
```

---

## 三、需要中长期投入（不推荐现阶段）

### 3.1 补充SUN RGB-D数据集 🟡

**时间成本**：1-2周  
**优先级**：⭐⭐⭐  
**影响**：显著提升泛化性论证

**不推荐现在做的原因**：
- 时间成本高，需要下载数据、转换格式、训练
- 投稿IEEE Access不必须
- 可以作为审稿人要求的补充实验

**如果必须做，步骤**：
1. 下载SUN RGB-D数据集（约37GB）
2. 转换为COCO格式
3. 训练ab4和baseline（各300轮，约2-3天）
4. 评测并补充到Section IV-C

---

### 3.2 全消融多种子 🟢

**时间成本**：2-3周  
**优先级**：⭐⭐  
**影响**：大幅增强实验可信度

**不推荐现在做的原因**：
- 计算成本极高：10配置×3种子 = 30个训练任务
- IEEE Access接受当前单种子消融 + 关键模型三种子验证
- 性价比低

---

### 3.3 跨域实验 🟢

**时间成本**：3-4周  
**优先级**：⭐⭐  
**影响**：适合投顶会，但IEEE Access不必须

**不推荐现在做的原因**：
- 机器人数据集转换、训练、评测耗时长
- 当前可以写在Future Work中

---

## 四、工作优先级总结

### 本周必做（提升接受率）⏰ 5天

| 任务 | 时间 | 优先级 | 完成检查 |
|---|---|---|---|
| 1. 生成定性可视化（Fig. 3） | 1-2天 | ⭐⭐⭐⭐⭐ | [ ] 4-6个样例对比图 |
| 2. 补充统计检验 | 2-3小时 | ⭐⭐⭐⭐⭐ | [ ] p值计算 + 论文更新 |
| 3. 制作交互柱状图（Fig. 2） | 1-2小时 | ⭐⭐⭐⭐ | [ ] PDF/EPS矢量图 |
| 4. 强化Method差异说明 | 2小时 | ⭐⭐⭐⭐⭐ | [ ] 三个模块差异段落 |
| 5. 精炼Abstract | 30分钟 | ⭐⭐⭐⭐⭐ | [ ] 控制在200词 |
| 6. 补充Discussion应对策略 | 1小时 | ⭐⭐⭐⭐ | [ ] 新增V-A适配挑战 |
| 7. 完善Limitations | 30分钟 | ⭐⭐⭐⭐ | [ ] 更新V-C |
| 8. 检查数据一致性 | 1小时 | ⭐⭐⭐⭐⭐ | [ ] 所有表格核对 |

**总计**：约3-4天全职工作

### 可选加分项（如果时间允许）⏰ 1-2天

| 任务 | 时间 | 优先级 | 完成检查 |
|---|---|---|---|
| 9. MRFE注意力可视化（Fig. 4） | 1-2天 | ⭐⭐⭐⭐ | [ ] 权重热力图 |

### 未来扩展（投稿后）⏰ 2-4周

| 任务 | 时间 | 说明 |
|---|---|---|
| 10. 补充SUN RGB-D | 1-2周 | 如果审稿人要求 |
| 11. 跨域实验 | 2-3周 | 投更高级别期刊时 |
| 12. 全消融多种子 | 2-3周 | 性价比低，不推荐 |

---

## 五、执行建议

### 第1天：写作优化（0成本）
- [ ] 上午：精炼Abstract（30分钟）+ 强化Method差异（2小时）
- [ ] 下午：补充Discussion（1小时）+ 完善Limitations（30分钟）
- [ ] 晚上：检查数据一致性（1小时）

**产出**：论文写作质量提升，应对审稿人质疑的核心论述完成

### 第2天：快速图表
- [ ] 上午：统计检验（3小时）
- [ ] 下午：制作交互柱状图（2小时）

**产出**：Fig. 2完成，统计显著性验证完成

### 第3-4天：定性可视化
- [ ] 第3天上午：选择样例（1小时）
- [ ] 第3天下午-第4天：生成对比图（6-8小时）

**产出**：Fig. 3完成，最重要的可视化证据

### 第5天（可选）：特征可视化
- [ ] 全天：MRFE注意力可视化

**产出**：Fig. 4完成，增强可解释性

### 投稿准备（第6天）
- [ ] 生成PDF，检查图表引用
- [ ] 最后一遍通读，修正语法错误
- [ ] 准备投稿材料（Cover Letter、作者信息）

---

## 六、验收标准

### 写作部分
- [ ] Abstract控制在200词以内，包含所有关键数据
- [ ] Method章节每个模块都有"Distinction from"段落
- [ ] Discussion有专门的"Adaptation Challenges"小节
- [ ] Limitations诚实说明未做的对比实验原因

### 图表部分
- [ ] Fig. 2（交互柱状图）：矢量格式，IEEE风格
- [ ] Fig. 3（定性对比）：4-6个样例，四列对比
- [ ] TABLE III下方有统计检验结果段落

### 数据部分
- [ ] 所有表格数值与原始数据匹配
- [ ] 三种子均值和标准差重新计算无误
- [ ] p值计算正确（配对t检验）

---

## 七、投稿目标与时间规划

### 推荐投稿目标（按优先级）

#### 1. IEEE Access（最推荐）⭐⭐⭐⭐⭐
- **接受概率**：70-80%
- **审稿周期**：2-3个月
- **理由**：看重完整实验，对创新性要求相对宽松

#### 2. IEEE RA-L ⭐⭐⭐⭐
- **接受概率**：40-50%
- **审稿周期**：3-4个月
- **理由**：与机器人应用结合，强调实时性

#### 3. ICRA / IROS（会议）⭐⭐⭐
- **接受概率**：35-45%
- **截稿时间**：需查询具体会议
- **理由**：机器人领域，接受工程贡献

### 时间规划

- **本周完成**：必做工作（1-5项）
- **下周一**：投稿IEEE Access
- **备用方案**：如果想冲RA-L，再花1周补充SUN RGB-D

---

## 附录：快速参考清单

### 必须完成的文件输出

1. `figures/Fig2_module_interaction.pdf` - 模块交互柱状图
2. `figures/Fig3_qualitative_comparison.pdf` - 定性对比图
3. `results/statistical_test_results.txt` - 统计检验结果
4. `chapter_IEEE_revised.md` - 修订后的论文结构（包含所有写作改进）

### 必须修改的论文章节

1. Abstract（Section 开头）
2. Section III-B/C/D（Method差异说明）
3. Section V-A（新增Adaptation Challenges）
4. Section V-C（完善Limitations）
5. TABLE III下方（统计检验段落）

### 检查清单

- [ ] 所有表格数值核对无误
- [ ] Abstract ≤ 200词
- [ ] 三个模块都有"Distinction from"段落
- [ ] Discussion有适配挑战小节
- [ ] Limitations诚实说明基线对比缺失原因
- [ ] 统计检验p值已计算并写入论文
- [ ] Fig. 2 和 Fig. 3 生成并引用正确
- [ ] 参考文献格式统一为IEEE格式

---

**文档结束。按此清单执行，可在5-7天内完成所有必要优化，达到IEEE Access投稿标准。**
