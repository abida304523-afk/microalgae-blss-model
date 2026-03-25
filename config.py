"""
config.py — 项目全局配置
========================
集中管理所有文件路径和物理/生物常数，
消除各脚本中的硬编码绝对路径。

使用方法:
    from config import PATHS, CONSTANTS
"""

from pathlib import Path

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

# 项目根目录（本文件所在目录）
BASE_DIR = Path(__file__).parent.resolve()

# 原始实验数据（Excel 文件）
# 默认放在项目根目录下的 data/ 子目录
DATA_DIR = BASE_DIR / 'data'
GROWTH_DATA_FILE   = DATA_DIR / '生长曲线.xlsx'
PHOTO_DATA_FILE    = DATA_DIR / '光合活性的变化.xlsx'

# 输出目录（图表、CSV 结果）
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 参数文件
PARAMS_V6_FILE = OUTPUT_DIR / 'params_v6.csv'

# ─────────────────────────────────────────────
# 生物/物理常数
# ─────────────────────────────────────────────

CONSTANTS = {
    # 细胞干重 (g/cell)，Chlorella 典型值 25 pg
    # 参考: Illman et al. 2000, Enzyme Microb. Technol.
    'CELL_MASS': 25e-12,

    # 干重中碳含量
    'CARBON_FRACTION': 0.50,

    # O₂ 摩尔质量 (g/mol)
    'O2_MW': 32.0,

    # 光合有效吸收截面积 (m²/g DW)
    # 参考: Kliphuis et al. 2012, Biotechnol. Bioeng.
    'ABS_CROSS_SECTION': 0.02,
}

# ─────────────────────────────────────────────
# BLSS 工程常数
# ─────────────────────────────────────────────

BLSS = {
    # 宇航员 O₂ 需求 (g/day)，NASA-STD-3001 轻体力活动
    'CREW_O2_DEMAND': 840.0,

    # 光暗周期比例 (12:12 L/D)
    'LD_RATIO': 12.0 / 24.0,

    # 连续收获稀释率 (d⁻¹)
    'DILUTION_RATE': 0.1,

    # NASA 营养标准 (g/person/day)
    'NASA_PROTEIN_MIN': 56.0,
    'NASA_PROTEIN_MAX': 91.0,
    'NASA_CARB_MIN': 200.0,
    'NASA_CARB_MAX': 400.0,
    'NASA_LIPID_MIN': 60.0,
    'NASA_LIPID_MAX': 80.0,

    # 轨道参数 (ISS-like)
    'ORBITAL_PERIOD': 90.0,   # 分钟
    'ORBITAL_SUNLIT': 45.0,   # 分钟
}

# ─────────────────────────────────────────────
# 实验分组
# ─────────────────────────────────────────────

GLC_LIST = [0, 1, 2, 5, 10]   # 葡萄糖浓度梯度 (g/L)

COLORS = {
    0:  '#2ca02c',
    1:  '#1f77b4',
    2:  '#ff7f0e',
    5:  '#d62728',
    10: '#9467bd',
}
