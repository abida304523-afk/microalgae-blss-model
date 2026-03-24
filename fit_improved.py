"""
C. pyrenoidosa GY-D12 — 改进版: ODE + 葡萄糖消耗动力学
=====================================================

改进点:
  1. 葡萄糖浓度随时间变化 (dS/dt), 自然产生两阶段生长
  2. 承载力由初始葡萄糖决定 (总碳源预算)
  3. 用解析解的拟合结果作为初始值, 加速收敛

模型:
  dX/dt = μ(S) × X × (1 - X / K(S₀))
  dS/dt = -μ_het(S) × X / Y_XS

  μ(S) = μ_photo + μ_max_S × S / (K_S + S)
  K(S₀) = K₀ + k_S × S₀
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = '/Users/2488mmabd/Documents/microalgae_model'


# =============================================================================
# 1. 数据
# =============================================================================

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
# 2. ODE 模型
# =============================================================================

def ode(t, y, S0, p):
    """
    y[0] = X (cells/mL)
    y[1] = S (g/L, 当前葡萄糖浓度)

    p = [mu_photo, mu_max_S, K_S, Y_XS, K0, k_S]
    """
    X = max(y[0], 1.0)
    S = max(y[1], 0.0)

    mu_photo, mu_max_S, K_S, Y_XS, K0, k_S = p

    # 异养生长 (Monod)
    mu_het = mu_max_S * S / (K_S + S) if S > 1e-10 else 0.0

    # 总生长速率
    mu = mu_photo + mu_het

    # 承载力 (由初始葡萄糖决定)
    K = K0 + k_S * S0

    # Logistic 约束
    growth_factor = max(1.0 - X / K, 0.0)

    dXdt = mu * X * growth_factor
    dSdt = -mu_het * X / Y_XS if S > 1e-10 else 0.0

    return [dXdt, dSdt]


def predict(t_eval, S0, X0, p):
    """运行 ODE 预测"""
    y0 = [X0, S0]
    try:
        sol = solve_ivp(ode, (0, t_eval[-1]+0.1), y0,
                        args=(S0, p), method='RK45',
                        t_eval=t_eval, max_step=0.02,
                        rtol=1e-8, atol=1e-6)
        if sol.success:
            return sol.y[0], sol.y[1]  # X, S
    except:
        pass
    return np.full_like(t_eval, 1e20), np.full_like(t_eval, 0.0)


# =============================================================================
# 3. 拟合
# =============================================================================

def residuals(p, data, glc_list):
    r = []
    for g in glc_list:
        d = data[g]
        X_pred, _ = predict(d['days'], g, d['mean'][0], p)
        scale = np.max(d['mean'])
        r.extend((X_pred - d['mean']) / scale)
    return np.array(r)


def cost_func(p, data, glc_list):
    return np.sum(residuals(p, data, glc_list)**2)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — ODE + 葡萄糖消耗模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.1f}M → {np.max(d['mean'])/1e6:.1f}M")

    # p = [mu_photo, mu_max_S, K_S, Y_XS, K0, k_S]
    bounds = [
        (0.01, 0.4),      # mu_photo [d⁻¹]
        (0.1, 2.0),       # mu_max_S [d⁻¹]
        (0.05, 10.0),     # K_S [g/L]
        (1e6, 5e8),       # Y_XS [cells/g_glucose]
        (8e6, 2.5e7),     # K0 [cells/mL]
        (2e6, 1.5e7),     # k_S [cells/(mL·g/L)]
    ]

    # 初始值 (基于解析解结果, 跳过全局搜索直接局部优化)
    x0 = [0.12, 0.9, 0.5, 5e7, 2e7, 1e7]

    print("\n[局部优化 (基于解析解初始值)]...")
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    # 多起点优化
    best_cost = np.inf
    best_res = None
    np.random.seed(42)

    starts = [x0]
    for _ in range(8):
        x_rand = [np.random.uniform(b[0], b[1]) for b in bounds]
        starts.append(x_rand)

    for i, x_start in enumerate(starts):
        try:
            res = least_squares(residuals, x_start, args=(data, glc_list),
                                bounds=(lb, ub), method='trf', max_nfev=2000)
            c = np.sum(res.fun**2)
            if c < best_cost:
                best_cost = c
                best_res = res
                print(f"  起点 {i}: cost = {c:.6f} ← 最优")
            else:
                print(f"  起点 {i}: cost = {c:.6f}")
        except:
            print(f"  起点 {i}: 失败")

    p_opt = best_res.x
    res = best_res

    # 标准误
    J = res.jac
    n, k = len(res.fun), len(p_opt)
    s2 = np.sum(res.fun**2) / max(n - k, 1)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        stderr = np.sqrt(np.abs(np.diag(cov)))
    except:
        stderr = np.full(k, np.nan)

    names = ['μ_photo', 'μ_max_S', 'K_S', 'Y_XS', 'K₀', 'k_S']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/g', 'cells/mL', 'cells/(mL·g/L)']

    print("\n" + "=" * 60)
    print(" 拟合参数")
    print("=" * 60)
    for nm, v, e, u in zip(names, p_opt, stderr, units):
        if abs(v) > 1e4:
            print(f"  {nm:10s} = {v:.4e} ± {e:.4e}  [{u}]")
        else:
            print(f"  {nm:10s} = {v:.6f} ± {e:.6f}  [{u}]")

    # R²
    print("\n  各组 R²:")
    r2 = {}
    for g in glc_list:
        d = data[g]
        Xp, _ = predict(d['days'], g, d['mean'][0], p_opt)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2[g] = 1 - ss_res / ss_tot
        print(f"    {g:2d} g/L:  R² = {r2[g]:.4f}")

    r2_total = 1 - sum(np.sum((data[g]['mean'] - predict(data[g]['days'], g, data[g]['mean'][0], p_opt)[0])**2) for g in glc_list) / sum(np.sum((data[g]['mean'] - np.mean(data[g]['mean']))**2) for g in glc_list)
    print(f"\n  总体 R² = {r2_total:.4f}")

    return p_opt, names, units, stderr, data, glc_list, r2


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_all(p, names, data, glc_list, r2):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 800)

    # ---- 图 1: 拟合结果 + 葡萄糖消耗 (2×5) ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 8),
                              gridspec_kw={'height_ratios': [2, 1]})

    for i, g in enumerate(glc_list):
        d = data[g]
        X_fine, S_fine = predict(t_fine, g, d['mean'][0], p)

        # 上排: 生长曲线
        ax = axes[0, i]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7,
                     label='实验', zorder=5)
        ax.plot(t_fine, X_fine/1e6, '-', color=colors[i], lw=2.5, label='模型')
        ax.set_title(f'{g} g/L', fontsize=13, fontweight='bold')
        ax.set_ylabel('X [×10⁶ cells/mL]' if i == 0 else '')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R²={r2[g]:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # 下排: 葡萄糖消耗
        ax2 = axes[1, i]
        ax2.plot(t_fine, S_fine, '-', color=colors[i], lw=2)
        ax2.fill_between(t_fine, 0, S_fine, alpha=0.15, color=colors[i])
        ax2.set_xlabel('时间 [天]')
        ax2.set_ylabel('S [g/L]' if i == 0 else '')
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — ODE 模型 (生长曲线 + 葡萄糖消耗预测)\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/improved_fit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  improved_fit.png")

    # ---- 图 2: 叠加图 ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                     gridspec_kw={'height_ratios': [2, 1]})

    for i, g in enumerate(glc_list):
        d = data[g]
        X_fine, S_fine = predict(t_fine, g, d['mean'][0], p)

        ax1.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                      fmt='o', color=colors[i], capsize=3, ms=6)
        ax1.plot(t_fine, X_fine/1e6, '-', color=colors[i], lw=2,
                 label=f'{g} g/L (R²={r2[g]:.3f})')

        ax2.plot(t_fine, S_fine, '-', color=colors[i], lw=2,
                 label=f'{g} g/L')

    ax1.set_ylabel('细胞浓度 [×10⁶ cells/mL]', fontsize=12)
    ax1.set_title('C. pyrenoidosa GY-D12 混合营养生长', fontsize=14, fontweight='bold')
    ax1.legend(title='葡萄糖浓度', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('时间 [天]', fontsize=12)
    ax2.set_ylabel('葡萄糖浓度 [g/L]', fontsize=12)
    ax2.set_title('葡萄糖消耗动力学 (模型预测)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{OUT}/improved_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  improved_overlay.png")

    # ---- 图 3: 生长速率分解 ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)

    for i, g in enumerate(glc_list):
        ax = axes[i]
        d = data[g]
        X_fine, S_fine = predict(t_fine, g, d['mean'][0], p)

        mu_photo_val = p[0]
        mu_het_vals = p[1] * S_fine / (p[2] + S_fine)
        mu_het_vals = np.where(S_fine > 1e-10, mu_het_vals, 0.0)
        mu_total = mu_photo_val + mu_het_vals

        ax.fill_between(t_fine, 0, mu_photo_val, alpha=0.3, color='green',
                         label='光合 μ_photo')
        ax.fill_between(t_fine, mu_photo_val, mu_total, alpha=0.3, color='orange',
                         label='异养 μ_het(S)')
        ax.plot(t_fine, mu_total, 'k-', lw=1.5, label='总 μ')
        ax.axhline(mu_photo_val, color='green', ls='--', alpha=0.5)

        ax.set_xlabel('时间 [天]')
        ax.set_title(f'{g} g/L', fontweight='bold')
        if i == 0:
            ax.set_ylabel('比生长速率 μ [d⁻¹]')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('生长速率随时间分解 (光合 vs 异养贡献)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/growth_rate_decomposition.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  growth_rate_decomposition.png")

    # ---- 图 4: 光合活性 ----
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
    plt.savefig(f'{OUT}/photosynthesis_final.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  photosynthesis_final.png")

    # ---- 图 5: 参数表 ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    table_data = []
    for nm, v, u in zip(names, p, ['d⁻¹','d⁻¹','g/L','cells/g','cells/mL','cells/(mL·g/L)']):
        table_data.append([nm, f'{v:.4e}' if abs(v)>1e4 else f'{v:.4f}', u])
    table_data.append(['S_opt', f'{np.sqrt(p[2]*100):.2f}', 'g/L (估计)'])

    tbl = ax.table(cellText=table_data,
                   colLabels=['参数', '值', '单位'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.6)
    for j in range(3):
        tbl[0,j].set_facecolor('#4472C4')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    ax.set_title('拟合参数汇总', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{OUT}/params_table_improved.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  params_table_improved.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2 = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_improved.csv', index=False, encoding='utf-8-sig')
    print("  params_improved.csv")

    # 更新记忆
    print("\n完成")
