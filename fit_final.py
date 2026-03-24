"""
蛋白核小球藻 (C. pyrenoidosa GY-D12) — Logistic-Haldane 解析模型
================================================================

使用 Logistic 方程的解析解, 避免 ODE 数值积分, 大幅加速拟合。

X(t) = K × X₀ / (X₀ + (K - X₀) × exp(-μ × t))

μ(S) = μ₀ + μ_S × S / (K_S + S + S²/K_I)
K(S) = K₀ + k_S × S
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = '/Users/2488mmabd/Documents/microalgae_model'


# =============================================================================
# 1. 数据
# =============================================================================

def load_data():
    df = pd.read_excel('/Users/2488mmabd/Downloads/生长曲线.xlsx',
                       sheet_name='Sheet2', header=None)
    days = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values
    glucose_levels = [0, 1, 2, 5, 10]
    data = {}
    for i, glc in enumerate(glucose_levels):
        mc, sc = 26 + 2*i, 27 + 2*i
        m = pd.to_numeric(df.iloc[3:, mc], errors='coerce').values
        s = pd.to_numeric(df.iloc[3:, sc], errors='coerce').values
        data[glc] = {'days': days, 'mean': m, 'std': s}
    return data, glucose_levels


def load_photo():
    return pd.read_excel('/Users/2488mmabd/Downloads/光合活性的变化.xlsx')


# =============================================================================
# 2. 解析模型
# =============================================================================

def logistic_analytical(t, X0, mu, K):
    """Logistic 方程解析解"""
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def mu_func(S, p):
    """μ(S) = μ₀ + μ_S × S / (K_S + S + S²/K_I)"""
    if S < 1e-10:
        return p[0]
    return p[0] + p[1] * S / (p[2] + S + S**2 / p[3])


def K_func(S, p):
    """K(S) = K₀ + k_S × S"""
    return p[4] + p[5] * S


def predict(t, S, X0, p):
    """预测给定葡萄糖浓度下的生长曲线"""
    mu = mu_func(S, p)
    K = K_func(S, p)
    return logistic_analytical(t, X0, mu, K)


# =============================================================================
# 3. 拟合
# =============================================================================

def residuals(p, data, glc_list):
    """归一化残差"""
    r = []
    for glc in glc_list:
        d = data[glc]
        X_pred = predict(d['days'], glc, d['mean'][0], p)
        scale = np.max(d['mean'])
        r.extend((X_pred - d['mean']) / scale)
    return np.array(r)


def cost(p, data, glc_list):
    return np.sum(residuals(p, data, glc_list)**2)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — Logistic-Haldane 拟合")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: X0={d['mean'][0]/1e6:.1f}M → "
              f"Xmax={np.max(d['mean'])/1e6:.1f}M cells/mL")

    # p = [mu_0, mu_S, K_S, K_I, K_0, k_S]
    bounds = [
        (0.01, 0.5),     # mu_0
        (0.05, 1.5),     # mu_S
        (0.5, 15.0),     # K_S
        (10.0, 200.0),   # K_I
        (8e6, 2e7),      # K_0
        (2e6, 1.5e7),    # k_S
    ]

    print("\n[全局搜索]...")
    res = differential_evolution(cost, bounds, args=(data, glc_list),
                                 maxiter=1000, tol=1e-10, seed=42,
                                 popsize=20, polish=True)
    p_opt = res.x
    print(f"  完成: cost = {res.fun:.6f}")

    # 局部精细化
    print("[局部优化]...")
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]
    res2 = least_squares(residuals, p_opt, args=(data, glc_list),
                         bounds=(lb, ub), method='trf')
    p_opt = res2.x

    # 标准误
    J = res2.jac
    n, k = len(res2.fun), len(p_opt)
    s2 = np.sum(res2.fun**2) / (n - k)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        stderr = np.sqrt(np.abs(np.diag(cov)))
    except:
        stderr = np.full(k, np.nan)

    names = ['μ₀', 'μ_S', 'K_S', 'K_I', 'K₀', 'k_S']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'g/L', 'cells/mL', 'cells/(mL·g/L)']

    print("\n" + "=" * 60)
    print(" 拟合参数")
    print("=" * 60)
    for nm, v, e, u in zip(names, p_opt, stderr, units):
        if abs(v) > 1e4:
            print(f"  {nm:6s} = {v:.4e} ± {e:.4e}  [{u}]")
        else:
            print(f"  {nm:6s} = {v:.6f} ± {e:.6f}  [{u}]")

    S_opt = np.sqrt(p_opt[2] * p_opt[3])
    mu_opt = mu_func(S_opt, p_opt)
    print(f"\n  最优葡萄糖浓度 S_opt = {S_opt:.2f} g/L")
    print(f"  对应最大生长速率 μ = {mu_opt:.4f} d⁻¹")

    print("\n  各组 R²:")
    r2_all = {}
    for g in glc_list:
        d = data[g]
        Xp = predict(d['days'], g, d['mean'][0], p_opt)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2 = 1 - ss_res / ss_tot
        r2_all[g] = r2
        print(f"    {g:2d} g/L:  R² = {r2:.4f}")

    return p_opt, names, units, stderr, data, glc_list, r2_all


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_all(p, names, data, glc_list, r2_all):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # --- 图 1: 拟合结果 (2×3) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, g in enumerate(glc_list):
        ax = axes[i // 3, i % 3]
        d = data[g]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, markersize=7,
                     label='实验数据', zorder=5)
        Xm = predict(t_fine, g, d['mean'][0], p)
        ax.plot(t_fine, Xm/1e6, '-', color=colors[i], lw=2.5, label='模型')
        ax.set_title(f'葡萄糖 {g} g/L', fontsize=13, fontweight='bold')
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R² = {r2_all[g]:.3f}', transform=ax.transAxes,
                fontsize=12, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Haldane 曲线
    ax6 = axes[1, 2]
    S_r = np.linspace(0, 15, 300)
    mu_r = [mu_func(s, p) for s in S_r]
    ax6.plot(S_r, mu_r, 'k-', lw=2.5)
    S_opt = np.sqrt(p[2] * p[3])
    mu_opt = mu_func(S_opt, p)
    ax6.plot(S_opt, mu_opt, 'ro', ms=10, zorder=5)
    ax6.axvline(S_opt, color='r', ls='--', alpha=0.5)
    ax6.annotate(f'S_opt = {S_opt:.1f} g/L\nμ = {mu_opt:.3f} d⁻¹',
                 xy=(S_opt, mu_opt), xytext=(S_opt+2, mu_opt-0.03),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    for i, g in enumerate(glc_list):
        ax6.plot(g, mu_func(g, p), 's', color=colors[i], ms=8, zorder=5)
    ax6.set_xlabel('葡萄糖浓度 [g/L]')
    ax6.set_ylabel('比生长速率 μ [d⁻¹]')
    ax6.set_title('Haldane 动力学', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — Logistic-Haldane 模型拟合\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fit_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  fit_final.png")

    # --- 图 2: 所有组叠加 ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, g in enumerate(glc_list):
        d = data[g]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=3, ms=7)
        Xm = predict(t_fine, g, d['mean'][0], p)
        ax.plot(t_fine, Xm/1e6, '-', color=colors[i], lw=2,
                label=f'{g} g/L (R²={r2_all[g]:.3f})')
    ax.set_xlabel('时间 [天]', fontsize=13)
    ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]', fontsize=13)
    ax.set_title('C. pyrenoidosa GY-D12 混合营养生长曲线\n'
                 '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='葡萄糖浓度', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/growth_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  growth_overlay.png")

    # --- 图 3: 承载力 + 生长速率 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    K_r = [K_func(s, p)/1e6 for s in S_r]
    ax1.plot(S_r, K_r, 'k-', lw=2.5, label='模型 K(S)')
    for i, g in enumerate(glc_list):
        ax1.plot(g, np.max(data[g]['mean'])/1e6, 'o', color=colors[i],
                 ms=10, label=f'{g} g/L')
    ax1.set_xlabel('葡萄糖 [g/L]')
    ax1.set_ylabel('K / Xmax [×10⁶ cells/mL]')
    ax1.set_title('(a) 承载力 K(S)', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(S_r, mu_r, 'k-', lw=2.5)
    ax2.fill_between(S_r, 0, mu_r, alpha=0.1, color='blue')
    ax2.plot(S_opt, mu_opt, 'ro', ms=10, zorder=5)
    for i, g in enumerate(glc_list):
        ax2.plot(g, mu_func(g, p), 's', color=colors[i], ms=8)
    ax2.set_xlabel('葡萄糖 [g/L]')
    ax2.set_ylabel('μ [d⁻¹]')
    ax2.set_title('(b) 比生长速率 μ(S)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/analysis_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  analysis_final.png")

    # --- 图 4: 光合活性 ---
    df = load_photo()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for j, (met, yl, tl) in enumerate([
        ('alpha', 'α', '(a) 光合效率 α'),
        ('ETR', 'ETR [μmol e⁻/m²/s]', '(b) 电子传递速率 ETR'),
        ('IK', 'Ik [μmol/m²/s]', '(c) 饱和光强 Ik')
    ]):
        ax = axes[j]
        for i, g in enumerate(glc_list):
            sub = df[df['group'] == g].groupby('Day')[met].agg(['mean','std']).reset_index()
            ax.errorbar(sub['Day'], sub['mean'], yerr=sub['std'],
                         fmt='o-', color=colors[i], capsize=3, lw=1.8, ms=5,
                         label=f'{g} g/L')
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel(yl)
        ax.set_title(tl, fontweight='bold')
        ax.legend(title='葡萄糖', fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('C. pyrenoidosa GY-D12 光合活性', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/photosynthesis_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  photosynthesis_final.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2_all = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2_all)

    # 保存参数
    pd.DataFrame({
        '参数': names, '拟合值': p, '标准误': stderr, '单位': units
    }).to_csv(f'{OUT_DIR}/params_final.csv', index=False, encoding='utf-8-sig')
    print("  params_final.csv")

    print("\n✓ 完成")
