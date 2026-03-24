"""
蛋白核小球藻 (C. pyrenoidosa GY-D12) — 简化 Logistic-Haldane 模型
=================================================================

简化策略:
  - 用 Logistic 模型替代 Monod 氮限制 (无氮源数据)
  - 葡萄糖效应用 Haldane 动力学
  - 总共 6 个参数, 分两步拟合

模型:
  dX/dt = μ(S) × X × (1 - X / K(S))

  μ(S) = μ₀ + μ_S × S / (K_S + S + S²/K_I)    # 比生长速率
  K(S) = K₀ + k_S × S                           # 承载力

  μ₀:  纯光合基础生长速率 [d⁻¹]
  μ_S: 最大异养贡献 [d⁻¹]
  K_S: 葡萄糖半饱和常数 [g/L]
  K_I: 葡萄糖抑制常数 [g/L]
  K₀:  无葡萄糖时的承载力 [cells/mL]
  k_S: 葡萄糖对承载力的贡献 [cells/(mL·g/L)]
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


# =============================================================================
# 1. 读取数据
# =============================================================================

def load_growth_data():
    df = pd.read_excel(
        '/Users/2488mmabd/Downloads/生长曲线.xlsx',
        sheet_name='Sheet2', header=None
    )
    days = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values
    glucose_levels = [0, 1, 2, 5, 10]
    data = {}
    mean_cols = [26, 28, 30, 32, 34]
    std_cols  = [27, 29, 31, 33, 35]

    for i, glc in enumerate(glucose_levels):
        mean_vals = pd.to_numeric(df.iloc[3:, mean_cols[i]], errors='coerce').values
        std_vals  = pd.to_numeric(df.iloc[3:, std_cols[i]], errors='coerce').values
        data[glc] = {'days': days, 'mean': mean_vals, 'std': std_vals}
    return data, glucose_levels


def load_photosynthesis_data():
    return pd.read_excel('/Users/2488mmabd/Downloads/光合活性的变化.xlsx')


# =============================================================================
# 2. 简化模型
# =============================================================================

def mu_total(S, p):
    """总比生长速率 μ(S) = μ₀ + Haldane(S)"""
    if S > 1e-8:
        mu_het = p['mu_S'] * S / (p['K_S'] + S + S**2 / p['K_I'])
    else:
        mu_het = 0.0
    return p['mu_0'] + mu_het


def carrying_capacity(S, p):
    """承载力 K(S) = K₀ + k_S × S"""
    return p['K_0'] + p['k_S'] * S


def ode_logistic(t, y, S0, p):
    """Logistic-Haldane ODE: dX/dt = μ(S) × X × (1 - X/K(S))"""
    X = max(y[0], 1.0)
    mu = mu_total(S0, p)
    K = carrying_capacity(S0, p)
    dXdt = mu * X * (1.0 - X / K)
    return [dXdt]


def run_model(params_vec, S0, X0, t_eval):
    """运行单组 Logistic-Haldane 模拟"""
    p = {
        'mu_0': params_vec[0],
        'mu_S': params_vec[1],
        'K_S':  params_vec[2],
        'K_I':  params_vec[3],
        'K_0':  params_vec[4],
        'k_S':  params_vec[5],
    }
    y0 = [X0]
    t_span = (0, t_eval[-1] + 0.1)

    try:
        sol = solve_ivp(
            ode_logistic, t_span, y0,
            args=(S0, p),
            method='RK45',
            t_eval=t_eval,
            max_step=0.05,
            rtol=1e-8, atol=1e-6
        )
        if sol.success:
            return sol.y[0]
    except Exception:
        pass
    return np.full_like(t_eval, 1e20, dtype=float)


# =============================================================================
# 3. 拟合
# =============================================================================

def objective(params_vec, data, glucose_levels):
    """加权最小二乘目标函数"""
    total_resid = []

    for glc in glucose_levels:
        d = data[glc]
        X0 = d['mean'][0]
        X_model = run_model(params_vec, S0=glc, X0=X0, t_eval=d['days'])

        # 归一化残差 (除以各组最大值, 使各组权重相当)
        scale = np.max(d['mean'])
        resid = (X_model - d['mean']) / scale
        total_resid.extend(resid)

    return np.array(total_resid)


def objective_scalar(params_vec, data, glucose_levels):
    """标量目标函数 (用于 differential_evolution)"""
    r = objective(params_vec, data, glucose_levels)
    return np.sum(r**2)


def fit_model():
    """两阶段拟合"""
    print("=" * 60)
    print(" Logistic-Haldane 简化模型拟合")
    print("=" * 60)

    data, glucose_levels = load_growth_data()

    print("\n[实验数据概览]")
    for glc in glucose_levels:
        d = data[glc]
        print(f"  Glucose {glc:2d} g/L: "
              f"X0 = {d['mean'][0]/1e6:.1f}M, "
              f"Xmax = {np.max(d['mean'])/1e6:.1f}M cells/mL")

    # 参数边界: [mu_0, mu_S, K_S, K_I, K_0, k_S]
    bounds = [
        (0.01, 0.8),     # mu_0: 光合基础生长速率 [d⁻¹]
        (0.05, 2.0),     # mu_S: 最大异养贡献 [d⁻¹]
        (0.1, 15.0),     # K_S: 葡萄糖半饱和常数 [g/L]
        (5.0, 200.0),    # K_I: 葡萄糖抑制常数 [g/L]
        (5e6, 3e7),      # K_0: 无糖承载力 [cells/mL]
        (1e6, 2e7),      # k_S: 糖对承载力的贡献 [cells/(mL·g/L)]
    ]

    # 阶段 1: 全局搜索 (differential evolution)
    print("\n[阶段 1] 全局搜索 (differential evolution)...")
    result_de = differential_evolution(
        objective_scalar, bounds,
        args=(data, glucose_levels),
        maxiter=500,
        tol=1e-8,
        seed=42,
        polish=True,
        popsize=15,
    )
    print(f"  全局搜索完成: cost = {result_de.fun:.6f}")

    # 阶段 2: 局部精细化 (least_squares)
    print("[阶段 2] 局部精细化 (least_squares)...")
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    result_ls = least_squares(
        objective, result_de.x,
        args=(data, glucose_levels),
        bounds=(lb, ub),
        method='trf',
        max_nfev=5000
    )

    params_opt = result_ls.x
    param_names = ['mu_0', 'mu_S', 'K_S', 'K_I', 'K_0', 'k_S']
    param_units = ['d⁻¹', 'd⁻¹', 'g/L', 'g/L', 'cells/mL', 'cells/(mL·g/L)']

    # 估计参数标准误
    J = result_ls.jac
    residuals = result_ls.fun
    n = len(residuals)
    p_count = len(params_opt)
    s2 = np.sum(residuals**2) / (n - p_count)

    try:
        cov = np.linalg.inv(J.T @ J) * s2
        param_stderr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        param_stderr = np.full(p_count, np.nan)

    print("\n" + "=" * 60)
    print(" 拟合结果")
    print("=" * 60)
    print(f"\n  {'参数':12s}  {'拟合值':>14s}  {'标准误':>14s}  {'单位'}")
    print("  " + "-" * 56)
    for name, val, err, unit in zip(param_names, params_opt, param_stderr, param_units):
        if abs(val) > 1e4:
            print(f"  {name:12s}  {val:14.4e}  {err:14.4e}  {unit}")
        else:
            print(f"  {name:12s}  {val:14.6f}  {err:14.6f}  {unit}")

    # 最优葡萄糖浓度
    S_opt = np.sqrt(params_opt[2] * params_opt[3])
    mu_at_opt = mu_total(S_opt, dict(zip(param_names, params_opt)))
    print(f"\n  最优葡萄糖浓度 = {S_opt:.2f} g/L")
    print(f"  最优比生长速率 = {mu_at_opt:.4f} d⁻¹")

    # 各组 R²
    print("\n  各组拟合 R²:")
    for glc in glucose_levels:
        d = data[glc]
        X_pred = run_model(params_opt, S0=glc, X0=d['mean'][0], t_eval=d['days'])
        ss_res = np.sum((d['mean'] - X_pred)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2 = 1 - ss_res / ss_tot
        print(f"    Glucose {glc:2d} g/L:  R² = {r2:.4f}")

    return params_opt, param_names, param_stderr, data, glucose_levels


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_fit(params_opt, param_names, data, glucose_levels):
    """拟合结果 vs 实验数据"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    for i, glc in enumerate(glucose_levels):
        ax = axes[i // 3, i % 3]
        d = data[glc]

        # 实验数据
        ax.errorbar(d['days'], d['mean'] / 1e6, yerr=d['std'] / 1e6,
                     fmt='o', color=colors[i], capsize=4, markersize=7,
                     label='实验数据', zorder=5)

        # 模型
        t_fine = np.linspace(0, 16, 500)
        X_model = run_model(params_opt, S0=glc, X0=d['mean'][0], t_eval=t_fine)
        ax.plot(t_fine, X_model / 1e6, '-', color=colors[i],
                linewidth=2.5, label='模型拟合')

        # R²
        X_pred = run_model(params_opt, S0=glc, X0=d['mean'][0], t_eval=d['days'])
        ss_res = np.sum((d['mean'] - X_pred)**2)
        ss_tot = np.sum((d['mean'] - np.mean(d['mean']))**2)
        r2 = 1 - ss_res / ss_tot

        ax.set_title(f'葡萄糖 {glc} g/L', fontsize=13, fontweight='bold')
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel('细胞浓度 [×10⁶ cells/mL]')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # 第 6 个子图: Haldane 曲线
    ax6 = axes[1, 2]
    S_range = np.linspace(0, 15, 300)
    p_dict = dict(zip(param_names, params_opt))
    mu_vals = [mu_total(s, p_dict) for s in S_range]
    ax6.plot(S_range, mu_vals, 'k-', linewidth=2.5)

    S_opt = np.sqrt(p_dict['K_S'] * p_dict['K_I'])
    mu_opt_val = mu_total(S_opt, p_dict)
    ax6.axvline(S_opt, color='r', linestyle='--', alpha=0.7)
    ax6.plot(S_opt, mu_opt_val, 'ro', markersize=10, zorder=5)
    ax6.annotate(f'S_opt = {S_opt:.1f} g/L\nμ_max = {mu_opt_val:.3f} d⁻¹',
                 xy=(S_opt, mu_opt_val),
                 xytext=(S_opt + 2, mu_opt_val - 0.05),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    # 标记实验浓度
    for glc in glucose_levels:
        mu_exp = mu_total(glc, p_dict)
        ax6.plot(glc, mu_exp, 's', color='blue', markersize=8, zorder=5)

    ax6.set_xlabel('葡萄糖浓度 [g/L]')
    ax6.set_ylabel('比生长速率 μ [d⁻¹]')
    ax6.set_title('Haldane 动力学曲线', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        '蛋白核小球藻 (C. pyrenoidosa GY-D12) Logistic-Haldane 模型拟合\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/fit_v2_results.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: fit_v2_results.png")


def plot_carrying_capacity(params_opt, param_names, data, glucose_levels):
    """承载力随葡萄糖浓度的变化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    p_dict = dict(zip(param_names, params_opt))

    # (a) 承载力
    S_range = np.linspace(0, 12, 200)
    K_vals = [carrying_capacity(s, p_dict) / 1e6 for s in S_range]
    ax1.plot(S_range, K_vals, 'k-', linewidth=2.5, label='模型: K(S)')

    for i, glc in enumerate(glucose_levels):
        Xmax = np.max(data[glc]['mean']) / 1e6
        ax1.plot(glc, Xmax, 'o', color=colors[i], markersize=10,
                 label=f'{glc} g/L (实测 Xmax)')

    ax1.set_xlabel('葡萄糖浓度 [g/L]', fontsize=12)
    ax1.set_ylabel('承载力 / 最大细胞浓度 [×10⁶ cells/mL]', fontsize=12)
    ax1.set_title('(a) 承载力 K(S) vs 实测最大浓度', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) 比生长速率
    mu_vals = [mu_total(s, p_dict) for s in S_range]
    ax2.plot(S_range, mu_vals, 'k-', linewidth=2.5)

    for i, glc in enumerate(glucose_levels):
        mu_val = mu_total(glc, p_dict)
        ax2.plot(glc, mu_val, 'o', color=colors[i], markersize=10,
                 label=f'{glc} g/L')

    ax2.set_xlabel('葡萄糖浓度 [g/L]', fontsize=12)
    ax2.set_ylabel('比生长速率 μ [d⁻¹]', fontsize=12)
    ax2.set_title('(b) 比生长速率 μ(S)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/fit_v2_analysis.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: fit_v2_analysis.png")


def plot_photosynthesis_v2():
    """光合活性图 (改进版)"""
    df = load_photosynthesis_data()
    glucose_levels = [0, 1, 2, 5, 10]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = ['alpha', 'ETR', 'IK']
    ylabels = ['α (初始斜率)', 'ETRmax [μmol e⁻/m²/s]', 'Ik [μmol/m²/s]']
    titles = ['(a) 光合效率 α', '(b) 最大电子传递速率 ETR',
              '(c) 饱和光强 Ik']

    for j, (metric, ylabel, title) in enumerate(zip(metrics, ylabels, titles)):
        ax = axes[j]
        for i, glc in enumerate(glucose_levels):
            sub = df[df['group'] == glc]
            grouped = sub.groupby('Day')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['Day'], grouped['mean'], yerr=grouped['std'],
                         fmt='o-', color=colors[i], capsize=3,
                         linewidth=1.8, markersize=5, label=f'{glc} g/L')
        ax.set_xlabel('时间 [天]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(title='葡萄糖', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        '蛋白核小球藻 (C. pyrenoidosa GY-D12) 光合活性\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/photosynthesis_v2.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: photosynthesis_v2.png")


def save_parameters(params_opt, param_names, param_stderr, param_units):
    """保存参数到 CSV"""
    df = pd.DataFrame({
        '参数': param_names,
        '拟合值': params_opt,
        '标准误': param_stderr,
        '单位': param_units
    })
    df.to_csv('/Users/2488mmabd/Documents/microalgae_model/fitted_parameters.csv',
              index=False, encoding='utf-8-sig')
    print("  参数已保存: fitted_parameters.csv")


# =============================================================================
# 5. 主程序
# =============================================================================

if __name__ == '__main__':
    params_opt, param_names, param_stderr, data, glucose_levels = fit_model()

    param_units = ['d⁻¹', 'd⁻¹', 'g/L', 'g/L', 'cells/mL', 'cells/(mL·g/L)']

    print("\n[生成图表]")
    plot_fit(params_opt, param_names, data, glucose_levels)
    plot_carrying_capacity(params_opt, param_names, data, glucose_levels)
    plot_photosynthesis_v2()
    save_parameters(params_opt, param_names, param_stderr, param_units)

    print("\n" + "=" * 60)
    print(" 所有文件保存在:")
    print(" /Users/2488mmabd/Documents/microalgae_model/")
    print("=" * 60)
