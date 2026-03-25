# Chlorella pyrenoidosa GY-D12 — 混合营养生长模型

> 基于 v6 Logistic-Monod 解析模型，研究蛋白核小球藻在不同葡萄糖浓度下的生长动力学，并评估其在太空生物再生生命保障系统（BLSS）中的应用潜力。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/abida304523-afk/microalgae-blss-model.git
cd microalgae-blss-model

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备数据（将 Excel 文件放入 data/ 目录）
mkdir data
cp /path/to/生长曲线.xlsx data/
cp /path/to/光合活性的变化.xlsx data/

# 4. 运行拟合
python fit_v6_optimized.py

# 5. 运行产物预测
python product_prediction.py

# 6. 运行 BLSS 工程分析
python blss_analysis.py
```

## 项目结构

```
microalgae-blss-model/
├── config.py               # 全局配置（路径、常数）
├── requirements.txt        # 依赖声明
├── .gitignore
│
├── core/                   # 核心模块（公共函数）
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载
│   ├── model.py            # 生长动力学模型
│   └── utils.py            # 通用计算工具
│
├── fit_v6_optimized.py     # v6 模型拟合（主脚本）
├── product_prediction.py   # 产物预测
├── blss_analysis.py        # BLSS 工程分析
├── validation.py           # 模型验证（LOOCV + 跨物种）
├── publication_figures.py  # 论文图表生成
│
├── data/                   # 原始实验数据（不提交）
│   ├── 生长曲线.xlsx
│   └── 光合活性的变化.xlsx
│
└── output/                 # 生成的图表和结果（自动创建）
    ├── params_v6.csv
    ├── product_predictions.csv
    └── *.png
```

## 核心模型

**v6 Logistic-Monod 解析模型**：

$$X(t) = \frac{K(S) \cdot X_0}{X_0 + (K(S) - X_0) e^{-\mu(S) t}}$$

$$\mu(S) = \mu_0 + \frac{\mu_S \cdot S}{K_S + S}, \quad K(S) = K_0 + \frac{K_{max} \cdot S}{S_K + S}$$

| 参数 | 含义 | 单位 | 拟合值 |
|------|------|------|--------|
| μ₀ | 自养最大比生长速率 | d⁻¹ | 0.120 |
| μ_S | 葡萄糖促进生长速率 | d⁻¹ | 0.499 |
| K_S | Monod 半饱和常数 | g/L | 0.519 |
| K₀ | 自养承载力 | ×10⁶ cells/mL | 22.5 |
| K_max | 葡萄糖增加的最大承载力 | ×10⁶ cells/mL | 106.8 |

**总体 R² = 0.92**（5 个浓度组均值）

## 主要结论

- **最优操作点**：5 g/L 葡萄糖，平均 O₂ 产率最高（~483 mg/L/day）
- **BLSS 体积需求**：满足 1 名宇航员 O₂ 需求约需 1740 L PBR
- **模型计算速度**：~17.6 万次预测/秒，支持实时控制

## 引用

如使用本项目，请引用：

```
[作者]. C. pyrenoidosa GY-D12 Mixotrophic Growth Model. GitHub, 2025.
https://github.com/abida304523-afk/microalgae-blss-model
```
