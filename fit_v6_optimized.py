"""
C. pyrenoidosa GY-D12 — v6: 最优解析模型
==========================================

策略:
  1. 解析 Logistic 解 (快速, 稳健)
  2. μ(S): Haldane 或 Monod (对比)
  3. K(S): 饱和型 K(S) = K₀ + K_max × S / (S_K + S) (比线性更合理)
  4. 允许每组 X₀ 微调 (±20%), 消除实验初始值偏差
  5. 同时拟合 ln(X) + X 空间, 平衡大小组权重
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

OUT = '/Users/2488mmabd/Documents/microalgae_model'


def load_data():
    df = pd.read_excel('/Users/2488mmabd/Downloads/生长曲线.xlsx',
                       sheet_name='Sheet2', header=None)
    days = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values
    glc_list = [0, 1, 2, 5, 10]
    data = {}
    for i, g in enumerate(glc_list):
        m = pd.to_numeric(df.iloc[3:, 26+2*i], errors='coerce').values
        s = pd.to_numeric(df.iloc[3:, 27+2*i], errors='coerce').values
        data[g] = {'days': days, 'mean': m, 'std': s}
    return data, glc_list


def load_photo():
    return pd.read_excel('/Users/2488mmabd/Downloads/光合活性的变化.xlsx')


# =============================================================================
# 模型
# =============================================================================

def logistic(t, X0, mu, K):
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def predict(t, S, X0, p):
    """
    p = [mu0, mu_S, K_S, K0, K_max, S_K, f0, f1, f2, f5, f10]
    f_i: X0 调整因子 (1.0 = 不调整)
    """
    mu0, mu_S, K_S, K0, K_max, S_K = p[:6]

    # μ(S) = μ₀ + Monod(S)
    mu = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0

    # K(S) = K₀ + 饱和型
    K = K0 + K_max * S / (S_K + S) if S > 1e-10 else K0

    return logistic(t, X0, mu, K)


def predict_with_x0adj(t, S, X0_raw, p, glc_idx):
    """带 X0 调整的预测"""
    f = p[6 + glc_idx]  # X0 调整因子
    X0 = X0_raw * f
    return predict(t, S, X0, p)


# =============================================================================
# 拟合
# =============================================================================

def residuals(p, data, glc_list):
    r = []
    for idx, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict_with_x0adj(d['days'], g, d['mean'][0], p, idx)
        # 混合空间残差: 50% 线性 + 50% log
        scale_lin = np.max(d['mean'])
        scale_log = np.log(np.max(d['mean'])) - np.log(d['mean'][0])
        r_lin = (X_pred - d['mean']) / scale_lin
        r_log = (np.log(np.maximum(X_pred, 1)) - np.log(d['mean'])) / max(scale_log, 0.1)
        r.extend(0.7 * r_lin + 0.3 * r_log)
    return np.array(r)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — v6: 优化解析 Logistic 模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.2f}M → "
              f"{np.max(d['mean'])/1e6:.1f}M")

    # p = [mu0, mu_S, K_S, K0, K_max, S_K, f0, f1, f2, f5, f10]
    bounds_lb = [0.01, 0.05, 0.1, 5e6,  1e7,  0.5, 0.7, 0.7, 0.7, 0.7, 0.7]
    bounds_ub = [0.5,  2.0,  15., 3e7,  2e8,  20., 1.5, 1.5, 1.5, 1.5, 1.5]

    informed_starts = [
        # 基于 v2 结果
        [0.12, 0.9, 2.0, 1.5e7, 8e7, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.15, 0.5, 5.0, 2e7, 5e7, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.20, 0.8, 3.0, 1e7, 1e8, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.10, 1.2, 1.0, 1.8e7, 1.2e8, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.08, 0.6, 8.0, 2.5e7, 6e7, 10., 1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    np.random.seed(42)
    for _ in range(15):
        x_rand = [np.random.uniform(l, u) for l, u in zip(bounds_lb, bounds_ub)]
        informed_starts.append(x_rand)

    print(f"\n[多起点优化] ({len(informed_starts)} 起点)...")
    best_cost = np.inf
    best_res = None

    for i, x_start in enumerate(informed_starts):
        try:
            res = least_squares(residuals, x_start, args=(data, glc_list),
                                bounds=(bounds_lb, bounds_ub),
                                method='trf', max_nfev=5000)
            c = np.sum(res.fun**2)
            if c < best_cost:
                best_cost = c
                best_res = res
                if i < 8 or c < 0.5:
                    print(f"  起点 {i:2d}: cost = {c:.6f} ← 最优")
        except:
            pass

    p_opt = best_res.x

    # 标准误
    J = best_res.jac
    n, k = len(best_res.fun), len(p_opt)
    s2 = np.sum(best_res.fun**2) / max(n - k, 1)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        stderr = np.sqrt(np.abs(np.diag(cov)))
    except:
        stderr = np.full(k, np.nan)

    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'K_max', 'S_K',
             'f₀', 'f₁', 'f₂', 'f₅', 'f₁₀']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/mL', 'cells/mL', 'g/L',
             '', '', '', '', '']

    print(f"\n  最终 cost = {best_cost:.6f}")
    print("\n" + "=" * 60)
    print(" 拟合参数")
    print("=" * 60)
    for nm, v, e, u in zip(names, p_opt, stderr, units):
        if abs(v) > 1e4:
            print(f"  {nm:8s} = {v:.4e} ± {e:.4e}  [{u}]")
        else:
            print(f"  {nm:8s} = {v:.6f} ± {e:.6f}  [{u}]")

    # 派生
    print("\n  派生参数:")
    for g in glc_list:
        mu = p_opt[0] + p_opt[1] * g / (p_opt[2] + g) if g > 0 else p_opt[0]
        K = p_opt[3] + p_opt[4] * g / (p_opt[5] + g) if g > 0 else p_opt[3]
        print(f"    {g:2d} g/L: μ={mu:.3f} d⁻¹, K={K/1e6:.1f}M")

    # R²
    print("\n  各组 R²:")
    r2 = {}
    for idx, g in enumerate(glc_list):
        d = data[g]
        Xp = predict_with_x0adj(d['days'], g, d['mean'][0], p_opt, idx)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2[g] = 1 - ss_res / ss_tot
        print(f"    {g:2d} g/L:  R² = {r2[g]:.4f}")

    r2_total = 1 - sum(
        np.sum((data[g]['mean'] - predict_with_x0adj(data[g]['days'], g,
                data[g]['mean'][0], p_opt, i))**2)
        for i, g in enumerate(glc_list)
    ) / sum(
        np.sum((data[g]['mean'] - np.mean(data[g]['mean']))**2)
        for g in glc_list
    )
    print(f"\n  总体 R² = {r2_total:.4f}")

    # 噪声限制分析
    print("\n  [噪声限制分析]")
    for g in glc_list:
        d = data[g]
        cv = np.mean(d['std'] / d['mean']) * 100
        print(f"    {g:2d} g/L: 平均 CV = {cv:.1f}%")

    return p_opt, names, units, stderr, data, glc_list, r2


def plot_all(p, names, data, glc_list, r2):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # ---- 图 1: 拟合 (1×5) ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
    for i, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict_with_x0adj(t_fine, g, d['mean'][0], p, i)
        ax = axes[i]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7,
                     label='实验', zorder=5)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2.5, label='模型')
        ax.set_title(f'{g} g/L', fontsize=13, fontweight='bold')
        ax.set_xlabel('时间 [天]')
        if i == 0:
            ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R²={r2[g]:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — 解析 Logistic-Monod 模型 (v6)\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v6_fit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v6_fit.png")

    # ---- 图 2: 叠加 ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict_with_x0adj(t_fine, g, d['mean'][0], p, i)
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=3, ms=6)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2,
                label=f'{g} g/L (R²={r2[g]:.3f})')
    ax.set_xlabel('时间 [天]', fontsize=13)
    ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]', fontsize=13)
    ax.set_title('C. pyrenoidosa GY-D12 混合营养生长 (v6)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='葡萄糖', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v6_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v6_overlay.png")

    # ---- 图 3: μ(S) 和 K(S) 曲线 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    S_r = np.linspace(0, 15, 300)

    mu_r = [p[0] + p[1] * s / (p[2] + s) if s > 1e-10 else p[0] for s in S_r]
    ax1.plot(S_r, mu_r, 'k-', lw=2.5)
    for i, g in enumerate(glc_list):
        mu_g = p[0] + p[1] * g / (p[2] + g) if g > 0 else p[0]
        ax1.plot(g, mu_g, 'o', color=colors[i], ms=10, zorder=5,
                 label=f'{g} g/L')
    ax1.set_xlabel('葡萄糖 [g/L]')
    ax1.set_ylabel('μ [d⁻¹]')
    ax1.set_title('(a) 比生长速率 μ(S)', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    K_r = [(p[3] + p[4] * s / (p[5] + s))/1e6 if s > 1e-10 else p[3]/1e6
           for s in S_r]
    ax2.plot(S_r, K_r, 'k-', lw=2.5, label='模型 K(S)')
    for i, g in enumerate(glc_list):
        K_g = (p[3] + p[4] * g / (p[5] + g))/1e6 if g > 0 else p[3]/1e6
        ax2.plot(g, K_g, 'o', color=colors[i], ms=10, zorder=5)
        ax2.plot(g, np.max(data[g]['mean'])/1e6, 's', color=colors[i],
                 ms=8, alpha=0.5, zorder=5)
    ax2.set_xlabel('葡萄糖 [g/L]')
    ax2.set_ylabel('K [×10⁶ cells/mL]')
    ax2.set_title('(b) 承载力 K(S) (○模型, □实测Xmax)', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT}/v6_kinetics.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v6_kinetics.png")

    # ---- 图 4: 综合参数表 + R² ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    ax1.axis('off')
    core_names = names[:6]
    core_vals = p[:6]
    core_units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/mL', 'cells/mL', 'g/L']
    table_data = []
    for nm, v, u in zip(core_names, core_vals, core_units):
        table_data.append([nm, f'{v:.4e}' if abs(v) > 1e4 else f'{v:.4f}', u])
    tbl = ax1.table(cellText=table_data,
                    colLabels=['参数', '值', '单位'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.6)
    for j in range(3):
        tbl[0,j].set_facecolor('#4472C4')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    ax1.set_title('核心参数 (6个)', fontsize=13, fontweight='bold', pad=20)

    # R² 条形图 + 版本对比
    groups = [f'{g} g/L' for g in glc_list]
    r2_v6 = [r2[g] for g in glc_list]
    # v2 结果 (来自 fit_improved.py)
    r2_v2 = [0.8274, 0.6032, 0.6808, 0.7631, 0.7661]

    x_pos = np.arange(len(groups))
    w = 0.35
    bars1 = ax2.bar(x_pos - w/2, r2_v2, w, label='v2 (ODE+葡萄糖消耗)',
                     color='lightblue', edgecolor='black')
    bars2 = ax2.bar(x_pos + w/2, r2_v6, w, label='v6 (解析+饱和K)',
                     color='salmon', edgecolor='black')
    ax2.axhline(0.9, color='red', ls='--', alpha=0.5)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel('R²')
    ax2.set_title('版本对比', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/v6_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v6_summary.png")

    # ---- 图 5: 光合活性 ----
    df = load_photo()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for j, (met, yl, tl) in enumerate([
        ('alpha', 'α', '(a) 光合效率 α'),
        ('ETR', 'ETR [μmol e⁻/m²/s]', '(b) 电子传递速率 ETR'),
        ('IK', 'Ik [μmol/m²/s]', '(c) 饱和光强 Ik')
    ]):
        ax = axes[j]
        for i, g in enumerate(glc_list):
            sub = df[df['group']==g].groupby('Day')[met].agg(['mean','std']).reset_index()
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
    plt.savefig(f'{OUT}/v6_photosynthesis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v6_photosynthesis.png")


if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2 = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_v6.csv', index=False, encoding='utf-8-sig')
    print("  params_v6.csv")

    print("\n完成")
