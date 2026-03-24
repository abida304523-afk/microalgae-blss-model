"""
C. pyrenoidosa GY-D12 — v3: ODE + 葡萄糖消耗 + 延迟相 + 光合适应
================================================================

改进点 (相比 v2):
  1. 异养延迟相: 葡萄糖利用能力随时间激活 (sigmoid)
  2. 光合抑制因子: 高糖浓度抑制光合效率 (竞争碳固定)
  3. 放宽 K_S 和 Y_XS 边界
  4. 承载力由初始葡萄糖决定 (非线性饱和)

模型:
  dX/dt = μ(S,t) × X × (1 - X / K(S₀))
  dS/dt = -μ_het(S,t) × X / Y_XS

  μ(S,t) = μ_photo × f_photo(S₀) + μ_het(S,t)
  μ_het(S,t) = μ_max_S × S / (K_S + S) × h(t, λ)
  h(t, λ) = 1 / (1 + exp(-k_lag × (t - λ)))    # 延迟激活
  f_photo(S₀) = 1 / (1 + S₀/K_photo)             # 高糖光合抑制
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
# 2. ODE 模型 (v3)
# =============================================================================

def sigmoid(t, lam, k=4.0):
    """延迟激活函数 h(t, λ) — 在 t=λ 处 50% 激活"""
    x = k * (t - lam)
    x = np.clip(x, -30, 30)  # 防溢出
    return 1.0 / (1.0 + np.exp(-x))


def ode(t, y, S0, p):
    """
    y[0] = X (cells/mL)
    y[1] = S (g/L, 当前葡萄糖浓度)

    p = [mu_photo, mu_max_S, K_S, Y_XS, K0, k_S, lam, K_photo]
    """
    X = max(y[0], 1.0)
    S = max(y[1], 0.0)

    mu_photo, mu_max_S, K_S, Y_XS, K0, k_S, lam, K_photo = p

    # 异养生长 (Monod + 延迟激活)
    if S > 1e-10 and S0 > 0.01:
        mu_het = mu_max_S * S / (K_S + S) * sigmoid(t, lam)
    else:
        mu_het = 0.0

    # 光合生长 (高糖抑制)
    f_photo = 1.0 / (1.0 + S0 / K_photo) if K_photo > 0 else 1.0
    mu_pho = mu_photo * f_photo

    # 总生长速率
    mu = mu_pho + mu_het

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
                        t_eval=t_eval, max_step=0.1,
                        rtol=1e-6, atol=1e-4)
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
    print(" C. pyrenoidosa GY-D12 — v3: 延迟相 + 光合抑制模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.1f}M → {np.max(d['mean'])/1e6:.1f}M")

    # p = [mu_photo, mu_max_S, K_S, Y_XS, K0, k_S, lam, K_photo]
    bounds = [
        (0.05, 0.5),       # mu_photo [d⁻¹]
        (0.1, 3.0),        # mu_max_S [d⁻¹]
        (0.01, 20.0),      # K_S [g/L] — 放宽下界
        (1e5, 1e10),       # Y_XS [cells/g] — 大幅放宽
        (5e6, 2.5e7),      # K0 [cells/mL]
        (1e6, 2e7),        # k_S [cells/(mL·g/L)]
        (0.0, 5.0),        # lam [d] — 延迟时间
        (2.0, 200.0),      # K_photo [g/L] — 光合抑制半抑制常数
    ]

    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    # 基于 v2 结果的知情初始值 + 新参数合理猜测
    informed_starts = [
        # 基于 v2 最优解, λ=1d (短延迟), K_photo=50 (弱抑制)
        [0.12, 0.9, 0.5, 5e7, 2e7, 1e7, 1.0, 50.0],
        # 更高光合, 更低异养
        [0.20, 0.5, 2.0, 1e8, 1.5e7, 8e6, 0.5, 30.0],
        # 更低光合, 更高异养, 长延迟
        [0.08, 1.5, 1.0, 2e7, 1.8e7, 1.2e7, 2.0, 100.0],
        # 中等参数
        [0.15, 1.0, 3.0, 5e8, 1.2e7, 5e6, 1.5, 20.0],
        # 快速异养启动
        [0.10, 2.0, 0.5, 1e8, 2e7, 8e6, 0.2, 80.0],
    ]

    # 加随机起点
    np.random.seed(42)
    for _ in range(5):
        x_rand = [np.random.uniform(b[0], b[1]) for b in bounds]
        informed_starts.append(x_rand)

    print("\n[多起点局部优化] (10 起点)...")
    best_cost = np.inf
    best_res = None

    for i, x_start in enumerate(informed_starts):
        try:
            res = least_squares(residuals, x_start, args=(data, glc_list),
                                bounds=(lb, ub), method='trf', max_nfev=1500)
            c = np.sum(res.fun**2)
            if c < best_cost:
                best_cost = c
                best_res = res
                print(f"  起点 {i:2d}: cost = {c:.6f} ← 最优")
            else:
                print(f"  起点 {i:2d}: cost = {c:.6f}")
        except:
            print(f"  起点 {i:2d}: 失败")

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

    names = ['μ_photo', 'μ_max_S', 'K_S', 'Y_XS', 'K₀', 'k_S', 'λ', 'K_photo']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/g', 'cells/mL',
             'cells/(mL·g/L)', 'd', 'g/L']

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

    r2_total = 1 - sum(
        np.sum((data[g]['mean'] - predict(data[g]['days'], g, data[g]['mean'][0], p_opt)[0])**2)
        for g in glc_list
    ) / sum(
        np.sum((data[g]['mean'] - np.mean(data[g]['mean']))**2)
        for g in glc_list
    )
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
        'C. pyrenoidosa GY-D12 — v3 模型 (延迟相 + 光合抑制)\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v3_fit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v3_fit.png")

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
    ax1.set_title('C. pyrenoidosa GY-D12 混合营养生长 (v3)', fontsize=14, fontweight='bold')
    ax1.legend(title='葡萄糖浓度', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('时间 [天]', fontsize=12)
    ax2.set_ylabel('葡萄糖浓度 [g/L]', fontsize=12)
    ax2.set_title('葡萄糖消耗动力学 (模型预测)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f'{OUT}/v3_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v3_overlay.png")

    # ---- 图 3: 生长速率分解 + 延迟激活 ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 8),
                              gridspec_kw={'height_ratios': [1.5, 1]})

    for i, g in enumerate(glc_list):
        d = data[g]
        X_fine, S_fine = predict(t_fine, g, d['mean'][0], p)

        # 各分量
        mu_photo_val = p[0]
        f_photo = 1.0 / (1.0 + g / p[7]) if p[7] > 0 else 1.0
        mu_pho_eff = mu_photo_val * f_photo

        mu_het_raw = np.where(S_fine > 1e-10,
                               p[1] * S_fine / (p[2] + S_fine), 0.0)
        h_vals = np.array([sigmoid(t, p[6]) for t in t_fine]) if g > 0.01 else np.zeros_like(t_fine)
        mu_het_eff = mu_het_raw * h_vals
        mu_total = mu_pho_eff + mu_het_eff

        # 上排: 速率分解
        ax = axes[0, i]
        ax.fill_between(t_fine, 0, mu_pho_eff, alpha=0.3, color='green',
                         label=f'光合 ({mu_pho_eff:.3f})')
        ax.fill_between(t_fine, mu_pho_eff, mu_total, alpha=0.3, color='orange',
                         label='异养')
        ax.plot(t_fine, mu_total, 'k-', lw=1.5, label='总 μ')
        ax.axhline(mu_pho_eff, color='green', ls='--', alpha=0.5)
        ax.set_title(f'{g} g/L', fontweight='bold')
        if i == 0:
            ax.set_ylabel('μ [d⁻¹]')
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # 下排: 延迟激活函数
        ax2 = axes[1, i]
        if g > 0.01:
            ax2.plot(t_fine, h_vals, '-', color=colors[i], lw=2)
            ax2.axvline(p[6], color='red', ls='--', alpha=0.5, label=f'λ={p[6]:.1f}d')
            ax2.axhline(0.5, color='gray', ls=':', alpha=0.5)
            ax2.legend(fontsize=8)
        else:
            ax2.text(0.5, 0.5, '无葡萄糖\n(纯光合)',
                     transform=ax2.transAxes, ha='center', va='center', fontsize=10)
        ax2.set_xlabel('时间 [天]')
        if i == 0:
            ax2.set_ylabel('h(t,λ) 激活度')
        ax2.set_ylim(-0.05, 1.1)
        ax2.grid(True, alpha=0.3)

    fig.suptitle('生长速率分解 + 异养延迟激活 h(t,λ)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v3_rate_decomposition.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v3_rate_decomposition.png")

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
    plt.savefig(f'{OUT}/v3_photosynthesis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v3_photosynthesis.png")

    # ---- 图 5: 参数表 + 模型对比 ----
    fig, (ax_tbl, ax_cmp) = plt.subplots(1, 2, figsize=(16, 5),
                                          gridspec_kw={'width_ratios': [1, 1.2]})

    # 参数表
    ax_tbl.axis('off')
    table_data = []
    for nm, v, u in zip(names, p, units):
        table_data.append([nm, f'{v:.4e}' if abs(v)>1e4 else f'{v:.4f}', u])

    tbl = ax_tbl.table(cellText=table_data,
                       colLabels=['参数', '值', '单位'],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    for j in range(3):
        tbl[0,j].set_facecolor('#4472C4')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    ax_tbl.set_title('v3 拟合参数', fontsize=13, fontweight='bold', pad=20)

    # R² 对比条形图
    groups = [f'{g} g/L' for g in glc_list]
    r2_vals = [r2[g] for g in glc_list]
    bar_colors = [colors[i] for i in range(len(glc_list))]
    bars = ax_cmp.bar(groups, r2_vals, color=bar_colors, alpha=0.8, edgecolor='black')
    ax_cmp.axhline(0.9, color='red', ls='--', alpha=0.7, label='R²=0.90')
    ax_cmp.axhline(0.95, color='darkred', ls='--', alpha=0.5, label='R²=0.95')
    for bar, val in zip(bars, r2_vals):
        ax_cmp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax_cmp.set_ylabel('R²')
    ax_cmp.set_title('各组拟合 R²', fontsize=13, fontweight='bold')
    ax_cmp.set_ylim(0, 1.1)
    ax_cmp.legend(fontsize=9)
    ax_cmp.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/v3_params_r2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v3_params_r2.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2 = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_v3.csv', index=False, encoding='utf-8-sig')
    print("  params_v3.csv")

    print("\n完成")
