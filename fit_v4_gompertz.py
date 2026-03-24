"""
C. pyrenoidosa GY-D12 — v4: 修正 Gompertz + Monod 混合模型
===========================================================

策略改变:
  - Gompertz 方程比 Logistic 更适合微生物生长 (不对称 S 曲线)
  - 先单组独立拟合 (诊断), 再全局联合拟合
  - 用 ln(X) 空间拟合, 避免大值主导残差

Gompertz 模型:
  ln(X(t)/X₀) = A × exp(-exp((μ_max×e/A)×(λ-t) + 1))

  A = ln(K/X₀)     最大相对增长量
  μ_max             最大比生长速率 [d⁻¹]
  λ                 延迟期 [d]

参数化:
  μ_max(S) = μ₀ + μ_S × S / (K_S + S)     Monod
  K(S)     = K₀ + k_S × S                  线性
  λ(S)     = λ₀ × exp(-α_λ × S)            随糖减小 (适应更快)
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
# 2. Gompertz 模型
# =============================================================================

def gompertz(t, X0, mu_max, K, lam):
    """修正 Gompertz 方程"""
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    A = np.log(K / X0)
    # ln(X/X0) = A × exp(-exp((mu_max×e/A)×(lam-t) + 1))
    exponent = (mu_max * np.e / A) * (lam - t) + 1.0
    exponent = np.clip(exponent, -30, 30)
    return X0 * np.exp(A * np.exp(-np.exp(exponent)))


def predict_combined(t, S, X0, p):
    """联合模型预测
    p = [mu0, mu_S, K_S, K0, k_S, lam0, alpha_lam]
    """
    mu0, mu_S, K_S, K0, k_S, lam0, alpha_lam = p

    # μ_max(S) = μ₀ + Monod(S)
    mu_max = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0

    # K(S) = K₀ + k_S × S
    K = K0 + k_S * S

    # λ(S) = λ₀ × exp(-α_λ × S) — 高糖减少延迟
    lam = lam0 * np.exp(-alpha_lam * S)

    return gompertz(t, X0, mu_max, K, lam)


# =============================================================================
# 3. 单组独立拟合 (诊断)
# =============================================================================

def fit_single_group(data, g):
    """独立拟合单组: 3 参数 (mu_max, K, lam)"""
    d = data[g]
    X0 = d['mean'][0]

    def resid(p):
        mu, K, lam = p
        Xp = gompertz(d['days'], X0, mu, K, lam)
        return (np.log(Xp) - np.log(d['mean'])) / np.log(np.max(d['mean']))

    # 初始值
    x0 = [0.3, np.max(d['mean']) * 1.2, 0.5]
    bounds = ([0.01, X0*1.1, -2.0], [3.0, 2e8, 10.0])

    try:
        res = least_squares(resid, x0, bounds=bounds, method='trf')
        return res.x, np.sum(res.fun**2)
    except:
        return x0, np.inf


# =============================================================================
# 4. 联合拟合
# =============================================================================

def residuals_combined(p, data, glc_list):
    """用 log 空间残差, 避免大值主导"""
    r = []
    for g in glc_list:
        d = data[g]
        X_pred = predict_combined(d['days'], g, d['mean'][0], p)
        # log 空间归一化残差
        log_pred = np.log(np.maximum(X_pred, 1.0))
        log_data = np.log(np.maximum(d['mean'], 1.0))
        scale = np.max(log_data) - np.min(log_data)
        r.extend((log_pred - log_data) / max(scale, 0.1))
    return np.array(r)


def fit():
    data, glc_list = load_data()

    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — v4: Gompertz + Monod 模型")
    print("=" * 60)
    print("\n[数据]")
    for g in glc_list:
        d = data[g]
        print(f"  {g:2d} g/L: {d['mean'][0]/1e6:.1f}M → "
              f"{np.max(d['mean'])/1e6:.1f}M  "
              f"(增长 {np.max(d['mean'])/d['mean'][0]:.1f}×)")

    # 单组诊断
    print("\n[单组独立拟合 (诊断)]")
    single_results = {}
    for g in glc_list:
        params, cost = fit_single_group(data, g)
        mu, K, lam = params
        d = data[g]
        Xp = gompertz(d['days'], d['mean'][0], mu, K, lam)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2 = 1 - ss_res / ss_tot
        single_results[g] = {'mu': mu, 'K': K, 'lam': lam, 'r2': r2}
        print(f"  {g:2d} g/L: μ={mu:.3f} d⁻¹, K={K/1e6:.1f}M, λ={lam:.2f}d, R²={r2:.4f}")

    # 联合拟合
    # p = [mu0, mu_S, K_S, K0, k_S, lam0, alpha_lam]
    bounds_lb = [0.01, 0.01, 0.1, 5e6,  1e6, -1.0, -1.0]
    bounds_ub = [1.0,  3.0,  20.0, 3e7, 2e7, 8.0,  5.0]

    # 用单组结果构建初始猜测
    mu0_guess = single_results[0]['mu']
    K0_guess = single_results[0]['K']
    lam0_guess = single_results[0]['lam']

    # mu_S 从高糖组推算
    mu10 = single_results[10]['mu']
    mu_S_guess = max(mu10 - mu0_guess, 0.1)
    K_S_guess = 2.0  # 假设
    k_S_guess = (single_results[10]['K'] - K0_guess) / 10.0
    alpha_lam_guess = 0.5

    informed_starts = [
        [mu0_guess, mu_S_guess, K_S_guess, K0_guess, k_S_guess, lam0_guess, alpha_lam_guess],
        [0.15, 0.5, 2.0, 1.5e7, 8e6, 1.0, 0.3],
        [0.20, 0.8, 5.0, 2.0e7, 5e6, 2.0, 0.1],
        [0.10, 1.0, 1.0, 1.0e7, 1e7, 0.5, 1.0],
        [0.25, 0.3, 3.0, 1.8e7, 6e6, 3.0, 0.5],
    ]

    # 随机起点
    np.random.seed(42)
    for _ in range(10):
        x_rand = [np.random.uniform(l, u) for l, u in zip(bounds_lb, bounds_ub)]
        informed_starts.append(x_rand)

    print(f"\n[联合拟合] (log 空间, {len(informed_starts)} 起点)...")
    best_cost = np.inf
    best_res = None

    for i, x_start in enumerate(informed_starts):
        try:
            res = least_squares(residuals_combined, x_start, args=(data, glc_list),
                                bounds=(bounds_lb, bounds_ub),
                                method='trf', max_nfev=2000)
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

    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'k_S', 'λ₀', 'α_λ']
    units = ['d⁻¹', 'd⁻¹', 'g/L', 'cells/mL', 'cells/(mL·g/L)', 'd', 'L/g']

    print("\n" + "=" * 60)
    print(" 拟合参数")
    print("=" * 60)
    for nm, v, e, u in zip(names, p_opt, stderr, units):
        if abs(v) > 1e4:
            print(f"  {nm:10s} = {v:.4e} ± {e:.4e}  [{u}]")
        else:
            print(f"  {nm:10s} = {v:.6f} ± {e:.6f}  [{u}]")

    # 派生参数
    for g in glc_list:
        mu = p_opt[0] + p_opt[1] * g / (p_opt[2] + g) if g > 0 else p_opt[0]
        K = p_opt[3] + p_opt[4] * g
        lam = p_opt[5] * np.exp(-p_opt[6] * g)
        print(f"  → {g:2d} g/L: μ={mu:.3f}, K={K/1e6:.1f}M, λ={lam:.2f}d")

    # R² (原始空间)
    print("\n  各组 R²:")
    r2 = {}
    for g in glc_list:
        d = data[g]
        Xp = predict_combined(d['days'], g, d['mean'][0], p_opt)
        ss_res = np.sum((d['mean'] - Xp)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2[g] = 1 - ss_res / ss_tot
        print(f"    {g:2d} g/L:  R² = {r2[g]:.4f}")

    r2_total = 1 - sum(
        np.sum((data[g]['mean'] - predict_combined(data[g]['days'], g, data[g]['mean'][0], p_opt))**2)
        for g in glc_list
    ) / sum(
        np.sum((data[g]['mean'] - np.mean(data[g]['mean']))**2)
        for g in glc_list
    )
    print(f"\n  总体 R² = {r2_total:.4f}")

    # 对比单组拟合
    print("\n  [对比: 单组独立 vs 联合]")
    print(f"  {'组':>6s}  {'单组 R²':>8s}  {'联合 R²':>8s}")
    for g in glc_list:
        print(f"  {g:2d} g/L  {single_results[g]['r2']:8.4f}  {r2[g]:8.4f}")

    return p_opt, names, units, stderr, data, glc_list, r2, single_results


# =============================================================================
# 5. 可视化
# =============================================================================

def plot_all(p, names, data, glc_list, r2, single_results):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # ---- 图 1: 拟合结果 (联合 vs 单组) ----
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    for i, g in enumerate(glc_list):
        d = data[g]
        X_combined = predict_combined(t_fine, g, d['mean'][0], p)
        sr = single_results[g]
        X_single = gompertz(t_fine, d['mean'][0], sr['mu'], sr['K'], sr['lam'])

        # 上排: 联合拟合
        ax = axes[0, i]
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7,
                     label='实验', zorder=5)
        ax.plot(t_fine, X_combined/1e6, '-', color=colors[i], lw=2.5,
                label='联合模型')
        ax.set_title(f'{g} g/L (联合)', fontsize=12, fontweight='bold')
        if i == 0:
            ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R²={r2[g]:.3f}', transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        # 下排: 单组独立拟合
        ax2 = axes[1, i]
        ax2.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                      fmt='o', color=colors[i], capsize=4, ms=7, zorder=5)
        ax2.plot(t_fine, X_single/1e6, '--', color=colors[i], lw=2.5,
                 label='独立拟合')
        ax2.set_title(f'{g} g/L (独立)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间 [天]')
        if i == 0:
            ax2.set_ylabel('X [×10⁶ cells/mL]')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, f'R²={sr["r2"]:.3f}', transform=ax2.transAxes,
                 fontsize=11, va='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — Gompertz 模型\n'
        '上排: 联合拟合 (7参数) | 下排: 独立拟合 (3参数/组)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v4_fit_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v4_fit_comparison.png")

    # ---- 图 2: 叠加图 ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict_combined(t_fine, g, d['mean'][0], p)
        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=3, ms=6)
        ax.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2,
                label=f'{g} g/L (R²={r2[g]:.3f})')
    ax.set_xlabel('时间 [天]', fontsize=13)
    ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]', fontsize=13)
    ax.set_title('C. pyrenoidosa GY-D12 混合营养生长 — Gompertz 模型\n'
                 '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='葡萄糖浓度', fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}/v4_overlay.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v4_overlay.png")

    # ---- 图 3: 参数随葡萄糖变化 ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    S_range = np.linspace(0, 12, 200)

    # μ(S)
    mu_vals = [p[0] + p[1] * s / (p[2] + s) if s > 1e-10 else p[0] for s in S_range]
    axes[0].plot(S_range, mu_vals, 'k-', lw=2.5)
    for i, g in enumerate(glc_list):
        mu_g = p[0] + p[1] * g / (p[2] + g) if g > 0 else p[0]
        axes[0].plot(g, mu_g, 'o', color=colors[i], ms=10, zorder=5)
        axes[0].plot(g, single_results[g]['mu'], 's', color=colors[i],
                     ms=8, zorder=5, alpha=0.5)
    axes[0].set_xlabel('葡萄糖 [g/L]')
    axes[0].set_ylabel('μ_max [d⁻¹]')
    axes[0].set_title('(a) 生长速率 μ(S)', fontweight='bold')
    axes[0].legend(['联合模型', '', '', '', '', '',
                    '单组拟合'], fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # K(S)
    K_vals = [(p[3] + p[4] * s)/1e6 for s in S_range]
    axes[1].plot(S_range, K_vals, 'k-', lw=2.5, label='联合模型')
    for i, g in enumerate(glc_list):
        axes[1].plot(g, (p[3] + p[4] * g)/1e6, 'o', color=colors[i], ms=10, zorder=5)
        axes[1].plot(g, single_results[g]['K']/1e6, 's', color=colors[i],
                     ms=8, zorder=5, alpha=0.5)
        axes[1].plot(g, np.max(data[g]['mean'])/1e6, '^', color=colors[i],
                     ms=8, zorder=5, alpha=0.3)
    axes[1].set_xlabel('葡萄糖 [g/L]')
    axes[1].set_ylabel('K [×10⁶ cells/mL]')
    axes[1].set_title('(b) 承载力 K(S)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # λ(S)
    lam_vals = [p[5] * np.exp(-p[6] * s) for s in S_range]
    axes[2].plot(S_range, lam_vals, 'k-', lw=2.5, label='联合模型')
    for i, g in enumerate(glc_list):
        lam_g = p[5] * np.exp(-p[6] * g)
        axes[2].plot(g, lam_g, 'o', color=colors[i], ms=10, zorder=5)
        axes[2].plot(g, single_results[g]['lam'], 's', color=colors[i],
                     ms=8, zorder=5, alpha=0.5)
    axes[2].set_xlabel('葡萄糖 [g/L]')
    axes[2].set_ylabel('λ [天]')
    axes[2].set_title('(c) 延迟期 λ(S)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('参数随葡萄糖浓度变化 (○联合模型, □单组拟合, △实测Xmax)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/v4_params_vs_glucose.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v4_params_vs_glucose.png")

    # ---- 图 4: R² 对比 + 参数表 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # 参数表
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
    ax1.set_title('v4 Gompertz 拟合参数', fontsize=13, fontweight='bold', pad=20)

    # R² 对比 (单组 vs 联合)
    groups = [f'{g}' for g in glc_list]
    x_pos = np.arange(len(groups))
    w = 0.35
    r2_single = [single_results[g]['r2'] for g in glc_list]
    r2_combined = [r2[g] for g in glc_list]

    bars1 = ax2.bar(x_pos - w/2, r2_single, w, label='独立拟合 (15参数)',
                     color='skyblue', edgecolor='black')
    bars2 = ax2.bar(x_pos + w/2, r2_combined, w, label='联合拟合 (7参数)',
                     color='salmon', edgecolor='black')

    ax2.axhline(0.9, color='red', ls='--', alpha=0.5)
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{g} g/L' for g in glc_list])
    ax2.set_ylabel('R²')
    ax2.set_title('拟合质量对比', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{OUT}/v4_params_r2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v4_params_r2.png")

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
    plt.savefig(f'{OUT}/v4_photosynthesis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  v4_photosynthesis.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    p, names, units, stderr, data, glc_list, r2, single_results = fit()

    print("\n[生成图表]")
    plot_all(p, names, data, glc_list, r2, single_results)

    pd.DataFrame({'参数': names, '值': p, '标准误': stderr, '单位': units}
    ).to_csv(f'{OUT}/params_v4.csv', index=False, encoding='utf-8-sig')
    print("  params_v4.csv")

    print("\n完成")
