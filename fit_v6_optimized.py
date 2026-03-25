"""
fit_v6_optimized.py — v6 Logistic-Monod 模型拟合（改进版）
============================================================

改进内容（相比原版）:
  1. 消除硬编码路径 → 统一使用 config.py
  2. 消除重复函数定义 → 从 core 模块导入
  3. 修复裸 except: → 改为具体异常类型
  4. 移除全局 warnings.filterwarnings('ignore')
  5. 拆分过长的 fit() 函数为多个子函数

模型:
    X(t) = K(S)·X₀ / (X₀ + (K(S)-X₀)·exp(-μ(S)·t))
    μ(S) = μ₀ + μ_S · S / (K_S + S)
    K(S) = K₀ + K_max · S / (S_K + S)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from config import GLC_LIST, COLORS, OUTPUT_DIR
from core.data_loader import load_growth_data
from core.model import predict_with_x0adj
from core.utils import r_squared, rmse


# ─────────────────────────────────────────────
# 残差函数
# ─────────────────────────────────────────────

def residuals(p: np.ndarray, data: dict) -> np.ndarray:
    """计算所有浓度组的归一化残差。"""
    res = []
    for idx, g in enumerate(GLC_LIST):
        d = data[g]
        mask = np.isfinite(d['mean']) & np.isfinite(d['days'])
        if mask.sum() == 0:
            continue
        t_obs = d['days'][mask]
        X_obs = d['mean'][mask]
        X_pred = predict_with_x0adj(t_obs, g, X_obs[0], p, idx)
        # 归一化残差（除以均值避免量纲影响）
        scale = np.mean(X_obs) if np.mean(X_obs) > 0 else 1.0
        res.extend(((X_obs - X_pred) / scale).tolist())
    return np.array(res)


# ─────────────────────────────────────────────
# 参数初值与边界
# ─────────────────────────────────────────────

def get_initial_params(data: dict) -> tuple:
    """生成初始参数猜测值和边界。"""
    # 从自养组估算 μ₀ 和 K₀
    d0 = data[0]
    mask = np.isfinite(d0['mean'])
    X0_est = d0['mean'][mask][0] if mask.any() else 5e6
    K0_est = np.nanmax(d0['mean']) * 1.2

    p0 = [
        0.15,       # μ₀  [d⁻¹]
        0.50,       # μ_S [d⁻¹]
        1.0,        # K_S [g/L]
        K0_est,     # K₀  [cells/mL]
        8e7,        # K_max [cells/mL]
        5.0,        # S_K [g/L]
        1.0,        # f₀
        1.2,        # f₁
        1.2,        # f₂
        1.2,        # f₅
        1.2,        # f₁₀
    ]

    lower = [0.01, 0.0, 0.01, 1e6, 1e6, 0.1,
             0.7, 0.7, 0.7, 0.7, 0.7]
    upper = [1.0,  2.0, 10.0, 5e7, 5e8, 20.0,
             1.5, 1.5, 1.5, 1.5, 1.5]

    return np.array(p0), (lower, upper)


# ─────────────────────────────────────────────
# 拟合
# ─────────────────────────────────────────────

def fit_model(data: dict) -> tuple:
    """
    执行最小二乘拟合。

    Returns
    -------
    (p_opt, result) : tuple
        p_opt  : 最优参数数组
        result : scipy OptimizeResult 对象
    """
    p0, bounds = get_initial_params(data)

    # 仅抑制 scipy 内部的已知收敛警告，而非全局抑制
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='.*covariance.*',
            category=RuntimeWarning
        )
        result = least_squares(
            residuals, p0,
            bounds=bounds,
            args=(data,),
            method='trf',
            loss='soft_l1',
            max_nfev=5000,
            verbose=0,
        )

    return result.x, result


def compute_fit_quality(p_opt: np.ndarray, data: dict) -> dict:
    """
    计算各浓度组的拟合质量指标。

    Returns
    -------
    dict : 键为葡萄糖浓度，值含 'R2', 'RMSE', 'n'
    """
    quality = {}
    for idx, g in enumerate(GLC_LIST):
        d = data[g]
        mask = np.isfinite(d['mean']) & np.isfinite(d['days'])
        if mask.sum() < 2:
            quality[g] = {'R2': float('nan'), 'RMSE': float('nan'), 'n': 0}
            continue
        t_obs  = d['days'][mask]
        X_obs  = d['mean'][mask]
        X_pred = predict_with_x0adj(t_obs, g, X_obs[0], p_opt, idx)
        quality[g] = {
            'R2':   r_squared(X_obs, X_pred),
            'RMSE': rmse(X_obs, X_pred),
            'n':    int(mask.sum()),
        }
    return quality


# ─────────────────────────────────────────────
# 保存参数
# ─────────────────────────────────────────────

def save_params(p_opt: np.ndarray, result, output_path=None):
    """将拟合参数保存为 CSV 文件。"""
    if output_path is None:
        output_path = OUTPUT_DIR / 'params_v6.csv'

    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'K_max', 'S_K',
             'f₀', 'f₁', 'f₂', 'f₅', 'f₁₀']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/mL', 'cells/mL', 'g/L',
             '', '', '', '', '']

    # 从雅可比矩阵估算标准误
    try:
        from scipy.linalg import svd
        J = result.jac
        _, s, VT = svd(J, full_matrices=False)
        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        s_inv = np.where(s > threshold, 1.0 / s, 0.0)
        cov = (VT.T * s_inv ** 2) @ VT
        se = np.sqrt(np.diag(cov)) * np.sqrt(
            np.sum(result.fun ** 2) / max(len(result.fun) - len(p_opt), 1)
        )
    except Exception:
        se = np.full_like(p_opt, float('nan'))

    df = pd.DataFrame({
        '参数': names,
        '值':   p_opt,
        '标准误': se,
        '单位': units,
    })
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  参数已保存: {output_path}")


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────

def plot_fit(p_opt: np.ndarray, data: dict, quality: dict):
    """绘制拟合曲线与实验数据对比图。"""
    t_fine = np.linspace(0, 16, 500)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, g in enumerate(GLC_LIST):
        ax = axes[idx]
        d  = data[g]
        mask = np.isfinite(d['mean']) & np.isfinite(d['days'])

        ax.errorbar(
            d['days'][mask], d['mean'][mask] / 1e6,
            yerr=d['std'][mask] / 1e6,
            fmt='o', color=COLORS[g], capsize=4, ms=6,
            label='实验数据', zorder=3
        )

        X_pred = predict_with_x0adj(
            t_fine, g, d['mean'][mask][0], p_opt, idx
        )
        ax.plot(t_fine, X_pred / 1e6, '-', color=COLORS[g],
                lw=2, label='v6 模型')

        q = quality[g]
        ax.set_title(
            f'{g} g/L 葡萄糖\n'
            f'R²={q["R2"]:.3f}, RMSE={q["RMSE"]/1e6:.2f}×10⁶',
            fontsize=11
        )
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 第 6 个子图显示 R² 汇总
    ax = axes[5]
    ax.axis('off')
    r2_vals = [quality[g]['R2'] for g in GLC_LIST]
    overall = np.nanmean(r2_vals)
    rows = [[f'{g} g/L', f'{quality[g]["R2"]:.4f}',
             f'{quality[g]["RMSE"]/1e6:.3f}'] for g in GLC_LIST]
    rows.append(['**均值**', f'{overall:.4f}', '—'])
    tbl = ax.table(
        cellText=rows,
        colLabels=['葡萄糖', 'R²', 'RMSE (×10⁶)'],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    ax.set_title('拟合质量汇总', fontsize=12, fontweight='bold')

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — v6 Logistic-Monod 模型拟合',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    out = OUTPUT_DIR / 'v6_fit.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {out}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print(" v6 Logistic-Monod 模型拟合")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载实验数据...")
    _, data = load_growth_data()

    # 2. 拟合
    print("[2] 执行最小二乘拟合...")
    p_opt, result = fit_model(data)
    print(f"    拟合状态: {'成功' if result.success else '未完全收敛'}")
    print(f"    残差范数: {np.linalg.norm(result.fun):.4f}")

    # 3. 质量评估
    print("[3] 计算拟合质量...")
    quality = compute_fit_quality(p_opt, data)
    for g in GLC_LIST:
        q = quality[g]
        print(f"    {g:2d} g/L: R²={q['R2']:.4f}, RMSE={q['RMSE']/1e6:.3f}×10⁶")
    print(f"    均值 R²: {np.nanmean([q['R2'] for q in quality.values()]):.4f}")

    # 4. 保存参数
    print("[4] 保存参数...")
    save_params(p_opt, result)

    # 5. 绘图
    print("[5] 生成图表...")
    plot_fit(p_opt, data, quality)

    print("\n完成！")
