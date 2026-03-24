# 中期答辩 PPT 大纲（含图片文件名标注）
**毕业设计题目：基于微藻的太空生命维持系统设计**
共 14 页 · 建议时长 12–15 分钟

---

## Slide 1｜封面
- 题目 / 姓名 / 学号 / 导师 / 日期
- 📄 图片：无

---

## Slide 2｜背景介绍
**现有太空生命维持系统的问题**

文字（3条）：
- 物理化学系统（PCLSS）：重量大、体积大、依赖地球补给
- O₂ 再生率低，废物（CO₂、尿液、食物残余）无法有效循环
- 长期深空任务中不可持续

📄 图片：NASA/ESA PCLSS 示意图（自找）

---

## Slide 3｜PBR 生命维持系统介绍
**光生物反应器（PBR）——新一代 BLSS 方案**

文字（3条）：
- 用微藻光合作用：吸收 CO₂ → 释放 O₂
- 同步产出食物（蛋白质、碳水化合物、脂质）
- 实现物质循环，理论上可无限期运行

📄 图片：PBR-BLSS 系统概念图（自找）

---

## Slide 4｜现有 PBR 的问题 → 研究问题的提出
**左半页：现有 PBR 的 4 个问题**

| 问题 | 说明 |
|------|------|
| ① O₂ 产出不稳定 | 系统波动，难以保障持续供氧 |
| ② 碳水化合物产出少 | 不能满足宇航员能量需求 |
| ③ 营养物质不全面 | 蛋白质、脂质比例不达标 |
| ④ 系统稳定性差 | 难以预测和控制 |

**右半页：本研究的问题**

> 蛋白核小球藻（*C. pyrenoidosa* GY-D12）能否解决上述 4 个问题？

📄 图片：无（纯文字/自绘对比图）

---

## Slide 5｜预实验：OD₆₈₀—细胞密度标准曲线
**为后续生长曲线测量建立定量依据**

文字（2条）：
- 线性方程：y = 300479 + 1.184×10⁷ · x，R² = **0.990**
- OD₆₈₀ 与细胞密度高度线性相关，可用吸光度快速换算细胞数量

📄 图片：`标曲.jpg`（`/Users/2488mmabd/Downloads/标曲.jpg`）

---

## Slide 6｜实验一·图1：生长曲线
**五个葡萄糖浓度梯度下的细胞密度变化**

文字（2条）：
- Group 5（5 g/L）与 Group 10 最终密度最高（~8–9 ×10⁷ cells/mL）
- Group 0（纯光自养）密度最低，说明混养显著促进生长

📄 图片：`生长曲线.jpg`（`/Users/2488mmabd/Downloads/生长曲线.jpg`）

---

## Slide 7｜实验一·图2：光合活性（四合一）
**四项光合参数综合判断最优培养浓度**

文字（2条）：
- Group 5 的 Fv/Fm 在高密度组中下降最慢，光合系统保持相对完整
- Group 10 虽生物量最高，但 Fv/Fm 和 rETRmax 晚期急剧下降，光合活性受损

**结论：选择 5 g/L 葡萄糖——兼顾高生长量与相对健康的光合活性**

📄 图片：`光合活性.jpg`（`/Users/2488mmabd/Downloads/光合活性.jpg`）

---

## Slide 8｜实验二的设计与安排
**蛋白核小球藻在不同碳源模式下的代谢速率与生物量积累研究**

文字（对比两组）：

| | 组别 A：自养密闭监测组 | 组别 B：混养曝气实验组 |
|--|----------------------|----------------------|
| 葡萄糖 | 0 g/L | 5 g/L |
| 装置 | 密闭实验箱 + CO₂/O₂/温度传感器 | 锥形瓶 + 微型曝气泵 |
| 核心目的 | 定量计算CO₂固定率与O₂产出率 | 定量分析有机质积累效率 |
| 检测指标 | 气体浓度（传感器自动上传）+ OD₆₈₀日采样 | 生物量干重 + OD₆₈₀日采样 |

光暗周期：12 h/12 h；两组均每日取样用标准曲线换算细胞密度

📄 图片：两组实验装置对比示意图（自绘：左密闭箱+传感器，右锥形瓶+曝气泵）

---

## Slide 9｜为什么要用模型
**实验二存在风险 → 模型作为保底策略**

文字（2条 + 示意）：
- 密封系统调试复杂，数据质量存在不确定性
- 策略：**提前建立数学模型**，在任何情况下均能给出定量答案

```
实验二成功 ──→ 实测数据 + 模型预测  互相验证
实验二失败 ──→ 模型预测              独立支撑
```

📄 图片：无（PPT 自绘流程框）

---

## Slide 10｜建模流程
**从文献到预测的完整路径**

📄 图片：流程图（PPT 自绘）

```
① 文献检索：现有 Chlorella 动力学模型
        ↓
② 框架选定：Logistic-Monod 解析模型
        ↓
③ 数据输入：5 组生长曲线 → 参数拟合
        ↓
④ 迭代优化：v1 → v6，R² 从 0.74 → 0.92
        ↓
⑤ 双重外部验证（LOOCV + 跨物种）
        ↓
⑥ 产物预测：O₂ / 蛋白质 / 碳水化合物 / 脂质
```

---

## Slide 11｜模型图1：生长曲线拟合结果
**v6 模型准确捕捉混养生长动态**

文字（2条）：
- 总体 R²=**0.92**，五组浓度均实现良好拟合
- 饱和型承载力函数：葡萄糖越高，细胞最大密度越大（趋于饱和）

📄 图片：`pub_fig1_growth.png`（`/Users/2488mmabd/Documents/microalgae_model/pub_fig1_growth.png`，展示左图(a)）

---

## Slide 12｜建模遇到的问题
**两个关键问题及解决方法**

📄 图片：`v6_overlay.png`（右侧，展示v6模型五组拟合叠加效果）
路径：`/Users/2488mmabd/Documents/microalgae_model/v6_overlay.png`

| # | 问题 | 原因 | 解决方法 | 结果 |
|---|------|------|---------|------|
| 1 | 五组浓度拟合精度低，R²=0.74 | 固定承载力无法适配不同浓度组 | 引入饱和型承载力函数 K(S) | R² 提升至 **0.92** |
| 2 | O₂ 计算值偏大 **100 倍** | 误用细胞几何表面积代替光合吸收截面积 | 改用 a*=0.02 m²/g DW | 结果回到合理范围 |

---

## Slide 13｜模型图2：模型可靠性验证
**三层验证证明模型不是过拟合**

文字（2条）：
- LOOCV 交叉验证 R²=0.677；跨物种验证（C. sorokiniana）R²=**0.965**
- 参数 μ_max=0.71 d⁻¹，处于文献报道范围（0.45–3.40 d⁻¹）

📄 图片：`pub_fig5_validation.png`（`/Users/2488mmabd/Documents/microalgae_model/pub_fig5_validation.png`）

---

## Slide 14｜模型图3：产物预测——回答四个问题
**对应 Slide 4 的 4 个 PBR 问题逐一作答**

| Slide 4 的问题 | 模型预测结果 |
|--------------|------------|
| ① O₂ 产出不稳定 | 5 g/L 组 15天累积净 O₂ **7.3 g/L**，存在最优混养浓度 |
| ② 碳水化合物少 | 5 g/L 组碳水达 0.47 g/L，高糖组显著提升 |
| ③ 营养不全面 | 蛋白质 40–50%DW + 碳水 + 脂质同步产出 |
| ④ 系统不稳定 | 模型 R²=0.92，预测速度 176,000次/秒，可实时控制 |

**结论：5 g/L 蛋白核小球藻混养能够定量解决现有 PBR 的四个核心问题**

📄 图片：`pub_fig4_summary.png`（右侧汇总表）+ `pub_fig3_oxygen.png`（(c) 累积O₂）
路径：`/Users/2488mmabd/Documents/microalgae_model/`

---

## 图片文件清单

| Slide | 图片说明 | 文件路径 |
|-------|---------|---------|
| 5 | OD₆₈₀标准曲线 | `/Users/2488mmabd/Documents/microalgae_model/标曲.jpg` |
| 6 | 生长曲线（实验数据） | `/Users/2488mmabd/Documents/microalgae_model/生长曲线.jpg` |
| 7 | 光合活性四合一 | `/Users/2488mmabd/Documents/microalgae_model/光合活性.jpg` |
| 11 | 生长模型拟合（取(a)左图） | `/Users/2488mmabd/Documents/microalgae_model/pub_fig1_growth.png` |
| 12 | v6模型五组拟合叠加 | `/Users/2488mmabd/Documents/microalgae_model/v6_overlay.png` |
| 13 | 三层模型验证（LOOCV+跨物种+μ_max） | `/Users/2488mmabd/Documents/microalgae_model/pub_fig5_validation.png` |
| 14 | Day15产物汇总表 | `/Users/2488mmabd/Documents/microalgae_model/pub_fig4_summary.png` |
| 14 | 累积O₂曲线（取(c)右图） | `/Users/2488mmabd/Documents/microalgae_model/pub_fig3_oxygen.png` |

## 备用图片（可替换使用）

| 图片 | 内容 | 路径 |
|------|------|------|
| `pub_fig2_composition.png` | 蛋白质/碳水/脂质动态（2×3面板） | `/Users/2488mmabd/Documents/microalgae_model/pub_fig2_composition.png` |
| `pub_fig6_carbon.png` | Day15碳分配分组对比 | `/Users/2488mmabd/Documents/microalgae_model/pub_fig6_carbon.png` |
| `blss_pbr_volume.png` | BLSS所需PBR体积估算 | `/Users/2488mmabd/Documents/microalgae_model/blss_pbr_volume.png` |
| `blss_nutrition.png` | 营养素供给分析 | `/Users/2488mmabd/Documents/microalgae_model/blss_nutrition.png` |
| `validation_loocv.png` | LOOCV验证详图 | `/Users/2488mmabd/Documents/microalgae_model/validation_loocv.png` |
| `validation_cross_species.png` | 跨物种验证详图 | `/Users/2488mmabd/Documents/microalgae_model/validation_cross_species.png` |
