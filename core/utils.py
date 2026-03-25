"""
core/utils.py — 通用计算工具
==============================
提供干重转换、生物质组成预测等通用函数，
消除各脚本中重复的 cells_to_dw() / biomass_composition() 定义。
"""

import numpy as np
from config import CONSTANTS


def cells_to_dw(X_cells: np.ndarray) -> np.ndarray:
    """
    细胞浓度转换为干重。

    Parameters
    ----------
    X_cells : 细胞浓度 [cells/mL]

    Returns
    -------
    np.ndarray : 干重 [g/L]

    Notes
    -----
    转换系数: 25 pg/cell (Illman et al. 2000)
    单位换算: cells/mL × g/cell × 1000 mL/L = g/L
    """
    return X_cells * CONSTANTS['CELL_MASS'] * 1e3


def biomass_composition(S: float, t: np.ndarray) -> tuple:
    """
    基于文献经验关系预测生物质组成比例。

    Parameters
    ----------
    S : 葡萄糖浓度 [g/L]
    t : 时间数组 [天]

    Returns
    -------
    (f_prot, f_carb, f_lipid) : tuple of np.ndarray
        蛋白质、碳水化合物、脂质的质量分数（占干重）

    Notes
    -----
    经验关系来源:
      - 蛋白质: Liang et al. 2009 (C. vulgaris 混养)
      - 碳水化合物: Ho et al. 2012; Dragone et al. 2011
      - 脂质: Heredia-Arroyo et al. 2011

    警告: 以上数据基于 C. vulgaris，跨物种应用于 C. pyrenoidosa
    时存在不确定性，建议补充直接实验测量。
    """
    tau = np.clip(t / 15.0, 0, 1.0)  # 归一化时间 [0, 1]

    # 蛋白质: 自养 55% → 混养 43%, 晚期下降 10%
    f_prot  = (0.55 - 0.12 * S / (2.0 + S)) * (1.0 - 0.10 * tau)

    # 碳水化合物: 自养 15% → 混养 25%, 晚期积累 +15%
    f_carb  = (0.15 + 0.10 * S / (3.0 + S)) * (1.0 + 0.15 * tau)

    # 脂质: 自养 8% → 混养 20%, 晚期加速积累
    f_lipid = (0.08 + 0.12 * S / (2.5 + S)) * (1.0 + 0.40 * tau ** 2)

    # 归一化: 确保三项之和不超过 95%（留 5% 给灰分等）
    f_total = f_prot + f_carb + f_lipid
    mask = f_total > 0.95
    if np.any(mask):
        scale = np.where(mask, 0.95 / f_total, 1.0)
        f_prot  = f_prot  * scale
        f_carb  = f_carb  * scale
        f_lipid = f_lipid * scale

    return f_prot, f_carb, f_lipid


def r_squared(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算决定系数 R²。

    Parameters
    ----------
    y_obs  : 实测值数组
    y_pred : 预测值数组

    Returns
    -------
    float : R² 值
    """
    y_obs  = np.asarray(y_obs,  dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_obs) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float('nan')
    ss_res = np.sum((y_obs[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_obs[mask] - np.mean(y_obs[mask])) ** 2)
    if ss_tot < 1e-30:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def rmse(y_obs: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 RMSE。

    Parameters
    ----------
    y_obs  : 实测值数组
    y_pred : 预测值数组

    Returns
    -------
    float : RMSE 值（与 y_obs 同量纲）
    """
    y_obs  = np.asarray(y_obs,  dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_obs) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((y_obs[mask] - y_pred[mask]) ** 2)))
