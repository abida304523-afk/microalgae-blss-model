"""
C. pyrenoidosa GY-D12 — v5: 双 Logistic 模型 (两阶段生长)
============================================================

核心思路:
  混合营养生长 = 光合阶段 + 异养阶段 的叠加
  X(t) = X₀ + ΔX_photo(t) + ΔX_het(t, S)

  ΔX_photo(t) = A_p / (1 + exp(-r_p × (t - t_p)))      光合贡献
  ΔX_het(t,S) = A_h(S) / (1 + exp(-r_h × (t - t_h)))   异养贡献

参数化:
  A_p        = K₀ - X₀        光合最大增量 (固定, 所有组共享)
  r_p                          光合 logistic 速率 [d⁻¹]
  t_p                          光合 logistic 中点 [d]
  A_h(S)     = a_h × S         异养最大增量 (与葡萄糖成正比)
  r_h                          异养 logistic 速率 [d⁻¹]
  t_h                          异养 logistic 中点 [d]

总共 6 参数: [r_p, t_p, K0, a_h, r_h, t_h]
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
# 2. 双 Logistic 模型
# =============================================================================

def logistic_component(t, A, r, t_mid):
    """单个 Logistic 增长分量"""
    x = -r * (t - t_mid)
    x = np.clip(x, -30, 30)
    return A / (1.0 + np.exp(x))


def predict(t, S, X0, p):
    """
    双 Logistic 预测
    p = [r_p, t_p, K0, a_h, r_h, t_h]
    """
    r_p, t_p, K0, a_h, r_h, t_h = p

    # 光合增量 (所有组共享)
    A_p = max(K0 - X0, 1e4)
    delta_photo = logistic_component(t, A_p, r_p, t_p)

    # 异养增量 (与葡萄糖成正比)
    A_h = a_h * S
    delta_het = logistic_component(t, A_h, r_h, t_h) if S > 0.01 else 0.0

    return X0 + delta_photo + delta_het


# =============================================================================
# 3. 拟合
# =============================================================================

def residuals(p, data, glc_list):
    r = []
    for g in glc_list:
        d = data[g]
        X_pred = predict(d['days'], g, d['mean'][0], p)
        scale = np.max(d['mean'])
        r.extend((X_pred - d['mean']) / scale)
    return np.array(r)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — v5: 双 Logistic 模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.1f}M → "
              f"{np.max(d['mean'])/1e6:.1f}M  "
              f"(增长 {np.max(d['mean'])/d['mean'][0]:.1f}×)")

    # p = [r_p, t_p, K0, a_h, r_h, t_h]
    bounds_lb = [0.05,  2.0, 8e6,  1e6,  0.1, 0.0]
    bounds_ub = [2.0,  15.0, 3e7,  2e7,  3.0, 10.0]

    informed_starts = [
        [0.3, 8.0, 1.5e7, 8e6, 0.8, 3.0],
        [0.5, 6.0, 1.2e7, 1e7, 1.0, 2.0],
        [0.2, 10.0, 2e7, 5e6, 0.5, 4.0],
        [0.4, 7.0, 1.8e7, 6e6, 1.5, 1.5],
        [0.15, 9.0, 1.0e7, 1.2e7, 0.6, 5.0],
    ]

    np.random.seed(42)
    for _ in range(10):
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
                print(f"  起点 {i:2d}: cost = {c:.6f} ← 最优")
            else:
                print(f"  起点 {i:2d}: cost = {c:.6f}")
        except:
            print(f"  起点 {i:2d}: 失败")

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

    names = ['r_photo', 't_photo', 'K₀', 'a_het', 'r_het', 't_het']
    units = ['d⁻¹', 'd', 'cells/mL', 'cells/(mL·g/L)', 'd⁻¹', 'd']

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


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_all(p, names, data, glc_list, r2):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # ---- 图 1: 拟合 + 分量分解 ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 9),
                              gridspec_kw={'height_ratios': [2, 1]})

    for i, g in enumerate(glc_list):
        d = data[g]
        X0 = d['mean'][0]
        X_pred = predict(t_fine, g, X0, p)

        # 分量
        A_p = max(p[2] - X0, 1e4)
        delta_photo = logistic_component(t_fine, A_p, p[0], p[1])
        A_h = p[3] * g
        delta_het = logistic_component(t_fine, A_h, p[4], p[5]) if g > 0.01 else np.zeros_like(t_fine)

        # 上排: 总拟合
        ax = axes[0, i]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7,
                     label='实验', zorder=5)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2.5, label='模型')
        ax.set_title(f'{g} g/L', fontsize=13, fontweight='bold')
        if i == 0:
            ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R²={r2[g]:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # 下排: 分量分解
        ax2 = axes[1, i]
        ax2.fill_between(t_fine, 0, delta_photo/1e6, alpha=0.3, color='green',
                          label='光合')
        ax2.fill_between(t_fine, delta_photo/1e6, (delta_photo+delta_het)/1e6,
                          alpha=0.3, color='orange', label='异养')
        ax2.plot(t_fine, (delta_photo+delta_het)/1e6, 'k-', lw=1.5)
        ax2.set_xlabel('时间 [天]')
        if i == 0:
            ax2.set_ylabel('ΔX [×10⁶]')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — 双 Logistic 模型\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v5_fit.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v5_fit.png")

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
    ax.set_title('C. pyrenoidosa GY-D12 — 双 Logistic 模型',
                 fontsize=14, fontweight='bold')
    ax.legend(title='葡萄糖', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v5_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v5_overlay.png")

    # ---- 图 3: 参数表 + R² ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    ax1.axis('off')
    table_data = []
    for nm, v, u in zip(names, p, units):
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
    ax1.set_title('v5 双 Logistic 参数', fontsize=13, fontweight='bold', pad=20)

    groups = [f'{g} g/L' for g in glc_list]
    r2_vals = [r2[g] for g in glc_list]
    bar_colors = [colors[i] for i in range(len(glc_list))]
    bars = ax2.bar(groups, r2_vals, color=bar_colors, alpha=0.8, edgecolor='black')
    ax2.axhline(0.9, color='red', ls='--', alpha=0.5, label='R²=0.90')
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R²')
    ax2.set_title('各组 R²', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/v5_params_r2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v5_params_r2.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2 = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_v5.csv', index=False, encoding='utf-8-sig')
    print("  params_v5.csv")

    print("\n完成")
