"""
C. pyrenoidosa GY-D12 — v7: Logistic + 延迟 + 饱和承载力
=========================================================

最佳组合:
  1. 解析 Logistic (快速稳健)
  2. 饱和型 K(S) (v6 已验证有效)
  3. 平滑延迟函数代替 X₀ 调整因子 (更有物理意义)
  4. Monod μ(S)

X_eff(t) = X₀ + (X_logistic(t) - X₀) × h(t)
h(t) = t^n / (τ^n + t^n)   Hill 函数延迟
τ(S) = τ₀ / (1 + S/S_τ)    延迟随葡萄糖减小

8 参数: [mu0, mu_S, K_S, K0, K_max, S_K, tau0, S_tau]
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
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


def hill_delay(t, tau, n=3):
    """Hill 函数延迟: h(t) = t^n / (tau^n + t^n)"""
    if tau < 0.01:
        return np.ones_like(t)
    return t**n / (tau**n + t**n)


def predict(t, S, X0, p):
    """
    p = [mu0, mu_S, K_S, K0, K_max, S_K, tau0, S_tau]
    """
    mu0, mu_S, K_S, K0, K_max, S_K, tau0, S_tau = p

    mu = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0
    K = K0 + K_max * S / (S_K + S) if S > 1e-10 else K0

    # 延迟: τ(S) = τ₀ / (1 + S/S_τ)
    tau = tau0 / (1.0 + S / S_tau) if S_tau > 0 else tau0

    X_log = logistic(t, X0, mu, K)
    h = hill_delay(t, tau)

    return X0 + (X_log - X0) * h


# =============================================================================
# 拟合
# =============================================================================

def residuals(p, data, glc_list):
    r = []
    for g in glc_list:
        d = data[g]
        X_pred = predict(d['days'], g, d['mean'][0], p)
        # 混合残差
        scale_lin = np.max(d['mean'])
        scale_log = np.log(np.max(d['mean'])) - np.log(d['mean'][0])
        r_lin = (X_pred - d['mean']) / scale_lin
        r_log = (np.log(np.maximum(X_pred, 1)) - np.log(d['mean'])) / max(scale_log, 0.1)
        r.extend(0.7 * r_lin + 0.3 * r_log)
    return np.array(r)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — v7: Logistic + Hill延迟 模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.2f}M → {np.max(d['mean'])/1e6:.1f}M")

    # p = [mu0, mu_S, K_S, K0, K_max, S_K, tau0, S_tau]
    bounds_lb = [0.01, 0.05, 0.1, 5e6,  1e7,  0.5, 0.0, 0.1]
    bounds_ub = [0.8,  3.0,  15., 3e7,  2e8,  20., 8.0, 50.0]

    informed_starts = [
        [0.12, 0.5, 0.5, 2e7, 1e8, 5.0, 2.0, 2.0],
        [0.15, 0.8, 2.0, 1.5e7, 8e7, 3.0, 3.0, 5.0],
        [0.20, 0.4, 3.0, 1e7, 1.2e8, 8.0, 1.0, 10.0],
        [0.10, 1.0, 1.0, 2.5e7, 6e7, 2.0, 4.0, 1.0],
        [0.08, 0.6, 5.0, 1.8e7, 1.5e8, 10., 5.0, 3.0],
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
                if i < 8 or c < best_cost * 1.01:
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

    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'K_max', 'S_K', 'τ₀', 'S_τ']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/mL', 'cells/mL', 'g/L', 'd', 'g/L']

    print(f"\n  最终 cost = {best_cost:.6f}")
    print("\n" + "=" * 60)
    print(" 拟合参数")
    print("=" * 60)
    for nm, v, e, u in zip(names, p_opt, stderr, units):
        if abs(v) > 1e4:
            print(f"  {nm:8s} = {v:.4e} ± {e:.4e}  [{u}]")
        else:
            print(f"  {nm:8s} = {v:.6f} ± {e:.6f}  [{u}]")

    print("\n  派生参数:")
    for g in glc_list:
        mu = p_opt[0] + p_opt[1] * g / (p_opt[2] + g) if g > 0 else p_opt[0]
        K = p_opt[3] + p_opt[4] * g / (p_opt[5] + g) if g > 0 else p_opt[3]
        tau = p_opt[6] / (1 + g / p_opt[7])
        print(f"    {g:2d} g/L: μ={mu:.3f}, K={K/1e6:.1f}M, τ={tau:.2f}d")

    # R²
    print("\n  各组 R²:")
    r2 = {}
    for g in glc_list:
        d = data[g]
        Xp = predict(d['days'], g, d['mean'][0], p_opt)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2[g] = 1 - ss_res / ss_tot
        print(f"    {g:2d} g/L:  R² = {r2[g]:.4f}")

    r2_total = 1 - sum(
        np.sum((data[g]['mean'] - predict(data[g]['days'], g, data[g]['mean'][0], p_opt))**2)
        for g in glc_list
    ) / sum(
        np.sum((data[g]['mean'] - np.mean(data[g]['mean']))**2)
        for g in glc_list
    )
    print(f"\n  总体 R² = {r2_total:.4f}")

    return p_opt, names, units, stderr, data, glc_list, r2


def plot_all(p, names, data, glc_list, r2):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # ---- 图 1: 拟合 + 延迟分析 (2×5) ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 9),
                              gridspec_kw={'height_ratios': [2, 1]})

    for i, g in enumerate(glc_list):
        d = data[g]
        X0 = d['mean'][0]
        X_pred = predict(t_fine, g, X0, p)
        X_log = logistic(t_fine, X0,
                          p[0] + p[1]*g/(p[2]+g) if g>0 else p[0],
                          p[3] + p[4]*g/(p[5]+g) if g>0 else p[3])
        tau_g = p[6] / (1 + g / p[7])
        h_vals = hill_delay(t_fine, tau_g)

        # 上排: 拟合
        ax = axes[0, i]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7, label='实验', zorder=5)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2.5, label='模型 (v7)')
        ax.plot(t_fine, X_log/1e6, ':', color=colors[i], lw=1.5, alpha=0.5,
                label='无延迟')
        ax.set_title(f'{g} g/L', fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R²={r2[g]:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # 下排: 延迟函数
        ax2 = axes[1, i]
        ax2.plot(t_fine, h_vals, '-', color=colors[i], lw=2)
        ax2.axhline(0.5, color='gray', ls=':', alpha=0.5)
        ax2.axvline(tau_g, color='red', ls='--', alpha=0.5,
                     label=f'τ={tau_g:.1f}d')
        ax2.set_xlabel('时间 [天]')
        if i == 0:
            ax2.set_ylabel('h(t) 延迟')
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — v7: Logistic + Hill延迟 模型\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v7_fit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v7_fit.png")

    # ---- 图 2: 叠加 ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict(t_fine, g, d['mean'][0], p)
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=3, ms=6)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2,
                label=f'{g} g/L (R²={r2[g]:.3f})')
    ax.set_xlabel('时间 [天]', fontsize=13)
    ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]', fontsize=13)
    ax.set_title('C. pyrenoidosa GY-D12 — Logistic + Hill延迟 (v7)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='葡萄糖', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v7_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v7_overlay.png")

    # ---- 图 3: 参数曲线 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    S_r = np.linspace(0, 15, 300)

    mu_r = [p[0] + p[1]*s/(p[2]+s) if s>1e-10 else p[0] for s in S_r]
    axes[0].plot(S_r, mu_r, 'k-', lw=2.5)
    for i, g in enumerate(glc_list):
        mu_g = p[0] + p[1]*g/(p[2]+g) if g>0 else p[0]
        axes[0].plot(g, mu_g, 'o', color=colors[i], ms=10, zorder=5)
    axes[0].set_xlabel('葡萄糖 [g/L]')
    axes[0].set_ylabel('μ [d⁻¹]')
    axes[0].set_title('(a) μ(S)', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    K_r = [(p[3]+p[4]*s/(p[5]+s))/1e6 if s>1e-10 else p[3]/1e6 for s in S_r]
    axes[1].plot(S_r, K_r, 'k-', lw=2.5)
    for i, g in enumerate(glc_list):
        K_g = (p[3]+p[4]*g/(p[5]+g))/1e6 if g>0 else p[3]/1e6
        axes[1].plot(g, K_g, 'o', color=colors[i], ms=10, zorder=5)
        axes[1].plot(g, np.max(data[g]['mean'])/1e6, 's', color=colors[i], ms=8, alpha=0.4)
    axes[1].set_xlabel('葡萄糖 [g/L]')
    axes[1].set_ylabel('K [×10⁶ cells/mL]')
    axes[1].set_title('(b) K(S)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    tau_r = [p[6]/(1+s/p[7]) for s in S_r]
    axes[2].plot(S_r, tau_r, 'k-', lw=2.5)
    for i, g in enumerate(glc_list):
        tau_g = p[6]/(1+g/p[7])
        axes[2].plot(g, tau_g, 'o', color=colors[i], ms=10, zorder=5)
    axes[2].set_xlabel('葡萄糖 [g/L]')
    axes[2].set_ylabel('τ [天]')
    axes[2].set_title('(c) τ(S) 延迟期', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('参数随葡萄糖浓度变化', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v7_kinetics.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v7_kinetics.png")

    # ---- 图 4: 综合总结 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    ax1.axis('off')
    table_data = []
    for nm, v, u in zip(names, p, units):
        table_data.append([nm, f'{v:.4e}' if abs(v) > 1e4 else f'{v:.4f}', u])
    tbl = ax1.table(cellText=table_data, colLabels=['参数', '值', '单位'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.6)
    for j in range(3):
        tbl[0,j].set_facecolor('#4472C4')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    ax1.set_title('v7 模型参数 (8个)', fontsize=13, fontweight='bold', pad=20)

    # 多版本 R² 对比
    groups = [f'{g}' for g in glc_list]
    x_pos = np.arange(len(groups))
    w = 0.2
    r2_v2 = [0.8274, 0.6032, 0.6808, 0.7631, 0.7661]
    r2_v6 = [0.8306, 0.8058, 0.8931, 0.9192, 0.9395]
    r2_v7 = [r2[g] for g in glc_list]

    ax2.bar(x_pos - w, r2_v2, w, label='v2 (ODE)', color='lightblue', edgecolor='black')
    ax2.bar(x_pos,     r2_v6, w, label='v6 (X₀调整)', color='lightyellow', edgecolor='black')
    bars = ax2.bar(x_pos + w, r2_v7, w, label='v7 (Hill延迟)', color='salmon', edgecolor='black')
    ax2.axhline(0.9, color='red', ls='--', alpha=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', fontsize=8, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{g} g/L' for g in glc_list])
    ax2.set_ylabel('R²')
    ax2.set_title('模型版本 R² 对比', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/v7_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v7_summary.png")

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
    plt.savefig(f'{OUT}/v7_photosynthesis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v7_photosynthesis.png")


if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2 = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_v7.csv', index=False, encoding='utf-8-sig')
    print("  params_v7.csv")

    print("\n完成")
