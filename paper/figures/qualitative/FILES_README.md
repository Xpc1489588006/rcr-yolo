# Figures目录文件说明

## ✅ 主论文图（投稿使用）

### Fig. 3 - 定性对比图（4场景组合）
- `Fig3_qualitative_comparison.pdf` (2.6M) - **IEEE投稿用矢量图**
- `Fig3_qualitative_comparison.png` (6.0M) - 高分辨率位图备用
- `Fig3_qualitative_comparison.svg` (4.0M) - 可编辑矢量版
- `Fig3_qualitative_comparison.tiff` (252M) - TIFF格式（体积大，通常不需要）

## ✅ 最新版独立场景图（补充材料/演示用）

### 场景1: Maximum improvement
- `qualitative_1_Maximum_improvement.{pdf,png,svg}`
- 图像ID: 000000009891
- 特点: baseline检测4个 → ab4检测7个（+3改进）

### 场景2: Small objects dense  
- `qualitative_2_Small_objects_dense.{pdf,png,svg}`
- 图像ID: 000000003934
- 特点: GT=14，其中6个小目标（43%）

### 场景3: Crowded + mixed scale
- `qualitative_3_Crowded_+_mixed_scale.{pdf,png,svg}`
- 图像ID: 000000002685
- 特点: GT=19（5小+14大），拥挤+尺度混合

### 场景4: Heavy occlusion
- `qualitative_4_Heavy_occlusion.{pdf,png,svg}`
- 图像ID: 000000018380
- 特点: 拥挤餐桌场景，重度遮挡

## 🗑️ 旧版本文件（可安全删除）

以下是生成过程中的迭代版本，已被最新版本替代：

### 第1版迭代文件
- `qualitative_1_High_improvement.*` - 已替换为Maximum_improvement
- `qualitative_2_Small_objects.*` - 非室内场景，已废弃
- `qualitative_3_Crowded_scene.*` - 已替换为Crowded_+_mixed_scale
- `qualitative_4_Typical_room.*` - 已替换为Heavy_occlusion

## 📋 推荐操作

### 投稿时使用
```latex
\includegraphics[width=\textwidth]{figures/Fig3_qualitative_comparison.pdf}
```

### 清理旧文件
如果需要节省空间，可以删除：
1. 所有旧版本文件（High_improvement, Small_objects, Crowded_scene, Typical_room）
2. TIFF格式文件（除非期刊明确要求）

### 保留文件
- Fig3_qualitative_comparison.pdf (必须)
- Fig3_qualitative_comparison.png (推荐保留)
- 所有最新版独立场景图的PDF格式（用于演示或补充材料）

---
生成日期: 2026-11-06
