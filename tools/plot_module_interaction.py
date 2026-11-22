# tools/plot_module_interaction.py
"""
Generate module interaction bar chart (Fig. 2) for RCR-YOLO paper
Shows ablation study results with IEEE-style formatting
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Data from TABLE I (ablation study results)
configs = [
    'Baseline',
    '+ORB',
    '+MRFE',
    '+LCRBase',
    'ORB+MRFE',
    'ORB+MRFE\n+LCRBase'
]

mAP_50_95 = [
    0.3310,  # baseline (YOLO11n)
    0.3309,  # ab1-orb
    0.3368,  # ab2-mrfe
    0.3320,  # ab3b-lcrbase
    0.3394,  # ab5-orbmrfe
    0.3433,  # ab4-rcr-fb (full model)
]

# IEEE style settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (8, 5)  # IEEE double-column width

# Create figure
fig, ax = plt.subplots()

# Plot bars with distinct colors
x = np.arange(len(configs))
colors = ['#d3d3d3', '#a8dadc', '#457b9d', '#f1faee', '#e63946', '#c1121f']
bars = ax.bar(x, mAP_50_95, width=0.6,
              color=colors,
              edgecolor='black', linewidth=1.2)

# Annotate values on top of bars
for i, (bar, val) in enumerate(zip(bars, mAP_50_95)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.0008,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Annotate improvement over baseline
baseline_val = mAP_50_95[0]
for i in range(1, len(mAP_50_95)):
    improvement = mAP_50_95[i] - baseline_val
    if improvement > 0.001:
        # Draw arrow from baseline to current (no alpha for EPS compatibility)
        ax.annotate('',
                   xy=(i, mAP_50_95[i] - 0.0005),
                   xytext=(0, baseline_val + 0.0005),
                   arrowprops=dict(arrowstyle='->', lw=1.2, color='gray'))

        # Text showing improvement (no alpha for EPS compatibility)
        mid_y = (mAP_50_95[i] + baseline_val) / 2
        ax.text(i - 0.2, mid_y, f'+{improvement:.4f}',
               fontsize=8, color='darkgray', ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray'))

# Set axis labels
ax.set_ylabel('mAP50-95', fontsize=12, fontweight='bold')
ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')

# Set x-axis ticks
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=15, ha='right', fontsize=10)

# Set y-axis limits for better visualization
ax.set_ylim(0.325, 0.346)

# Add grid for readability
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Title
ax.set_title('Module Interaction Analysis', fontsize=13, fontweight='bold', pad=15)

# Ensure output directory exists
os.makedirs('figures', exist_ok=True)

# Save figures
plt.tight_layout()
plt.savefig('figures/Fig2_module_interaction.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/Fig2_module_interaction.eps', format='eps', bbox_inches='tight')
plt.savefig('figures/Fig2_module_interaction.png', dpi=300, bbox_inches='tight')

print("Figure saved successfully!")
print("  - figures/Fig2_module_interaction.pdf")
print("  - figures/Fig2_module_interaction.eps")
print("  - figures/Fig2_module_interaction.png")

# Display the figure
plt.show()

print("\nKey findings:")
print(f"  Baseline: {mAP_50_95[0]:.4f}")
print(f"  Best (Full model): {mAP_50_95[-1]:.4f}")
print(f"  Total improvement: +{(mAP_50_95[-1] - mAP_50_95[0]):.4f} (+{((mAP_50_95[-1] - mAP_50_95[0])/mAP_50_95[0]*100):.2f}%)")
print(f"  MRFE contribution: +{(mAP_50_95[2] - mAP_50_95[0]):.4f}")
print(f"  LCRBase contribution: +{(mAP_50_95[3] - mAP_50_95[0]):.4f}")
print(f"  Synergistic effect: +{(mAP_50_95[-1] - mAP_50_95[4]):.4f}")
