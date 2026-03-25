"""
core/model.py — 生长动力学模型
================================
v6 Logistic-Monod 解析模型核心函数。
消除各脚本中重复的 logistic() / predict() 定义。

模型方程:
    X(t) = K·X₀ / (X₀ + (K - X₀)·exp(-μ·t))
    μ(S) = μ₀ + μ_S · S / (K_S + S)
    K(S) = K₀ + K_max · S / (S_K + S)
"""

import numpy as np


def logistic(t: np.ndarray, X0: float, mu: float, K: float) -> np.ndarray:
    """
    Logistic 生长方程解析解。

    Parameters
    ----------
    t   : 时间数组 [天]
    X0  : 初始细胞浓度 [cells/mL]
    mu  : 最大比生长速率 [d⁻¹]
    K   : 承载力 [cells/mL]

    Returns
    -------
    np.ndarray : 细胞浓度随时间的变化 [cells/mL]
    """
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def predict(t: np.ndarray, S: float, X0: float, p: np.ndarray) -> np.ndarray:
    """
    给定葡萄糖浓度 S，预测细胞生长曲线。

    Parameters
    ----------
    t  : 时间数组 [天]
    S  : 葡萄糖浓度 [g/L]
    X0 : 初始细胞浓度 [cells/mL]
    p  : 参数数组 [μ₀, μ_S, K_S, K₀, K_max, S_K, ...]

    Returns
    -------
    np.ndarray : 预测细胞浓度 [cells/mL]
    """
    mu0, mu_S, K_S, K0, K_max, S_K = p[:6]

    if S > 1e-10:
        mu = mu0 + mu_S * S / (K_S + S)
        K  = K0  + K_max * S / (S_K + S)
    else:
        mu = mu0
        K  = K0

    return logistic(t, X0, mu, K)


def predict_with_x0adj(
    t: np.ndarray,
    S: float,
    X0_raw: float,
    p: np.ndarray,
    glc_idx: int
) -> np.ndarray:
    """
    使用 X₀ 调整因子后预测生长曲线。

    Parameters
    ----------
    t       : 时间数组 [天]
    S       : 葡萄糖浓度 [g/L]
    X0_raw  : 实测初始细胞浓度 [cells/mL]
    p       : 参数数组（含 f₀..f₁₀ 调整因子）
    glc_idx : 葡萄糖浓度组索引 (0~4 对应 0,1,2,5,10 g/L)

    Returns
    -------
    np.ndarray : 预测细胞浓度 [cells/mL]
    """
    f  = p[6 + glc_idx]
    X0 = X0_raw * f
    return predict(t, S, X0, p)


def carrying_capacity(S: float, p: np.ndarray) -> float:
    """
    计算给定葡萄糖浓度下的承载力 K(S)。

    Parameters
    ----------
    S : 葡萄糖浓度 [g/L]
    p : 参数数组

    Returns
    -------
    float : 承载力 [cells/mL]
    """
    K0, K_max, S_K = p[3], p[4], p[5]
    if S > 1e-10:
        return K0 + K_max * S / (S_K + S)
    return K0


def growth_rate(S: float, p: np.ndarray) -> float:
    """
    计算给定葡萄糖浓度下的最大比生长速率 μ(S)。

    Parameters
    ----------
    S : 葡萄糖浓度 [g/L]
    p : 参数数组

    Returns
    -------
    float : 最大比生长速率 [d⁻¹]
    """
    mu0, mu_S, K_S = p[0], p[1], p[2]
    if S > 1e-10:
        return mu0 + mu_S * S / (K_S + S)
    return mu0
