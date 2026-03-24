"""
蛋白核小球藻 (Chlorella pyrenoidosa) 混合营养生长动力学模型
============================================================

基于 Haldane-Monod 框架，参考:
- Yu et al. (2022) Algal Research — C. sorokiniana 异养/混养动力学模型
- Khajouei et al. (2016) J. Applied Phycology — C. vulgaris 混养建模
- Adesanya et al. (2014) Bioresource Technology — 混养存储分子模型

模型结构:
  μ_mix = (μ_hetero + μ_photo) × f(N) - m

  μ_hetero: Haldane 动力学 (葡萄糖底物抑制)
  μ_photo:  Haldane 动力学 (光抑制)
  f(N):     Monod 动力学 (氮源限制)

ODE 系统:
  dX/dt = μ_mix × X                        (生物量)
  dS/dt = -μ_mix × X / Y_XS                (葡萄糖消耗)
  dN/dt = -μ_mix × X / Y_XN                (氮源消耗)
  dL/dt = (α × μ_mix + β) × X × f_lipid(N) (脂质积累)

作者: 妙米东和玛木提江
日期: 2026-03-23
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 模型参数 (文献典型值, Chlorella spp. 混合营养)
# =============================================================================

params = {
    # --- 异养生长 (Haldane, 葡萄糖) ---
    'mu_max_S': 0.85,       # 最大异养比生长速率 [d⁻¹]
    'K_S': 2.0,             # 葡萄糖半饱和常数 [g/L]
    'K_I_S': 40.0,          # 葡萄糖抑制常数 [g/L]

    # --- 光合生长 (Haldane, 光照) ---
    'mu_max_I': 1.0,        # 最大光合比生长速率 [d⁻¹]
    'K_sI': 100.0,          # 光半饱和常数 [μmol/m²/s]
    'K_I_I': 800.0,         # 光抑制常数 [μmol/m²/s]

    # --- 氮源限制 (Monod) ---
    'K_N': 0.05,            # 氮源半饱和常数 [g/L]

    # --- 维持消耗 ---
    'm': 0.02,              # 维持系数 [d⁻¹]

    # --- 得率系数 ---
    'Y_XS': 0.45,           # 生物量/葡萄糖得率 [g_X/g_S]
    'Y_XN': 25.0,           # 生物量/氮源得率 [g_X/g_N]

    # --- 脂质产生 (Luedeking-Piret) ---
    'alpha': 0.15,          # 生长关联脂质系数 [g_L/g_X]
    'beta': 0.008,          # 非生长关联脂质系数 [g_L/(g_X·d)]
    'K_N_lipid': 0.02,      # 氮缺乏促进脂质积累的半饱和常数 [g/L]

    # --- 培养条件 ---
    'I': 150.0,             # 光照强度 [μmol/m²/s]
}


# =============================================================================
# 2. 模型方程
# =============================================================================

def growth_rate_hetero(S, p):
    """异养生长速率 — Haldane 动力学 (葡萄糖底物抑制)"""
    return p['mu_max_S'] * S / (K_S_eff(S, p))


def K_S_eff(S, p):
    """Haldane 有效半饱和常数"""
    return p['K_S'] + S + S**2 / p['K_I_S']


def growth_rate_photo(I, p):
    """光合生长速率 — Haldane 动力学 (光抑制)"""
    return p['mu_max_I'] * I / (p['K_sI'] + I + I**2 / p['K_I_I'])


def nitrogen_limitation(N, p):
    """氮源限制因子 — Monod"""
    return N / (p['K_N'] + N)


def lipid_enhancement(N, p):
    """氮缺乏脂质增强因子 (氮越少, 脂质合成越多)"""
    return p['K_N_lipid'] / (p['K_N_lipid'] + N)


def specific_growth_rate(S, N, I, p):
    """总混合营养比生长速率"""
    mu_het = growth_rate_hetero(S, p)
    mu_pho = growth_rate_photo(I, p)
    f_N = nitrogen_limitation(N, p)
    mu = (mu_het + mu_pho) * f_N - p['m']
    return max(mu, 0.0)  # 生长速率不为负


def ode_system(t, y, p):
    """
    ODE 系统

    状态变量:
      y[0] = X  生物量浓度 [g/L]
      y[1] = S  葡萄糖浓度 [g/L]
      y[2] = N  氮源浓度 [g/L]
      y[3] = L  脂质浓度 [g/L]
    """
    X, S, N, L = y

    # 防止负浓度
    S = max(S, 0.0)
    N = max(N, 0.0)
    X = max(X, 1e-10)

    I = p['I']
    mu = specific_growth_rate(S, N, I, p)

    # 微分方程
    dXdt = mu * X
    dSdt = -mu * X / p['Y_XS'] if S > 1e-8 else 0.0
    dNdt = -mu * X / p['Y_XN'] if N > 1e-8 else 0.0

    # 脂质: 生长关联 + 非生长关联 + 氮缺乏增强
    f_lip = lipid_enhancement(N, p)
    dLdt = (p['alpha'] * mu + p['beta']) * X * (0.3 + 0.7 * f_lip)

    return [dXdt, dSdt, dNdt, dLdt]


# =============================================================================
# 3. 模拟运行: 不同葡萄糖浓度
# =============================================================================

def run_simulation(S0, N0=0.5, X0=0.1, L0=0.0, t_end=10, p=params):
    """
    运行单次模拟

    Parameters:
      S0: 初始葡萄糖浓度 [g/L]
      N0: 初始氮源浓度 [g/L]
      X0: 初始生物量 [g/L]
      L0: 初始脂质 [g/L]
      t_end: 培养天数 [d]
      p: 参数字典
    """
    y0 = [X0, S0, N0, L0]
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)

    sol = solve_ivp(
        ode_system, t_span, y0,
        args=(p,),
        method='RK45',
        t_eval=t_eval,
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10
    )
    return sol


def run_glucose_gradient():
    """运行不同葡萄糖浓度梯度的模拟"""
    glucose_levels = [1, 5, 10, 20, 30, 50]  # g/L
    results = {}

    for S0 in glucose_levels:
        sol = run_simulation(S0=S0, t_end=12)
        results[S0] = sol
        X_max = np.max(sol.y[0])
        print(f"  葡萄糖 {S0:3d} g/L → 最大生物量: {X_max:.3f} g/L")

    return results, glucose_levels


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_results(results, glucose_levels):
    """绘制不同葡萄糖浓度下的模拟结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(glucose_levels)))

    labels_y = ['生物量 X [g/L]', '葡萄糖 S [g/L]',
                '氮源 N [g/L]', '脂质 L [g/L]']
    titles = ['(a) 生物量生长曲线', '(b) 葡萄糖消耗',
              '(c) 氮源消耗', '(d) 脂质积累']

    for idx, ax in enumerate(axes.flat):
        for i, S0 in enumerate(glucose_levels):
            sol = results[S0]
            ax.plot(sol.t, sol.y[idx], color=colors[i],
                    linewidth=2, label=f'{S0} g/L')
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel(labels_y[idx])
        ax.set_title(titles[idx])
        ax.legend(title='葡萄糖', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        '小球藻混合营养生长模型 — Haldane-Monod 框架\n'
        f'(光照 = {params["I"]} μmol/m²/s, N₀ = 0.5 g/L)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/growth_curves.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  图表已保存: growth_curves.png")


def plot_growth_rate_surface():
    """绘制生长速率随葡萄糖和光照的响应曲面"""
    S_range = np.linspace(0.01, 60, 200)
    I_range = np.linspace(1, 1000, 200)
    S_grid, I_grid = np.meshgrid(S_range, I_range)

    mu_grid = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            mu_het = growth_rate_hetero(S_grid[i, j], params)
            mu_pho = growth_rate_photo(I_grid[i, j], params)
            mu_grid[i, j] = mu_het + mu_pho  # 假设氮充足

    fig, ax = plt.subplots(figsize=(10, 7))
    cs = ax.contourf(S_grid, I_grid, mu_grid, levels=30, cmap='YlGnBu')
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label('比生长速率 μ [d⁻¹]')
    ax.set_xlabel('葡萄糖浓度 [g/L]')
    ax.set_ylabel('光照强度 [μmol/m²/s]')
    ax.set_title('比生长速率响应曲面 (氮充足条件)')
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/growth_rate_surface.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: growth_rate_surface.png")


def plot_haldane_curve():
    """单独绘制 Haldane 曲线展示底物抑制效应"""
    S = np.linspace(0.01, 80, 500)
    mu_het = [growth_rate_hetero(s, params) for s in S]

    I = np.linspace(1, 1500, 500)
    mu_pho = [growth_rate_photo(i, params) for i in I]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 葡萄糖 Haldane
    ax1.plot(S, mu_het, 'b-', linewidth=2)
    S_opt = np.sqrt(params['K_S'] * params['K_I_S'])
    mu_opt = growth_rate_hetero(S_opt, params)
    ax1.axvline(S_opt, color='r', linestyle='--', alpha=0.7,
                label=f'最优浓度 = {S_opt:.1f} g/L')
    ax1.plot(S_opt, mu_opt, 'ro', markersize=8)
    ax1.set_xlabel('葡萄糖浓度 [g/L]')
    ax1.set_ylabel('异养比生长速率 μ_het [d⁻¹]')
    ax1.set_title('Haldane 动力学 — 葡萄糖底物抑制')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 光照 Haldane
    ax2.plot(I, mu_pho, 'g-', linewidth=2)
    I_opt = np.sqrt(params['K_sI'] * params['K_I_I'])
    mu_I_opt = growth_rate_photo(I_opt, params)
    ax2.axvline(I_opt, color='r', linestyle='--', alpha=0.7,
                label=f'最优光照 = {I_opt:.0f} μmol/m²/s')
    ax2.plot(I_opt, mu_I_opt, 'ro', markersize=8)
    ax2.set_xlabel('光照强度 [μmol/m²/s]')
    ax2.set_ylabel('光合比生长速率 μ_photo [d⁻¹]')
    ax2.set_title('Haldane 动力学 — 光抑制')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/haldane_curves.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: haldane_curves.png")


# =============================================================================
# 5. 主程序
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" 小球藻混合营养 Haldane-Monod 动力学模型")
    print("=" * 60)

    # 打印关键参数
    print("\n[模型参数]")
    print(f"  μ_max (异养) = {params['mu_max_S']} d⁻¹")
    print(f"  μ_max (光合) = {params['mu_max_I']} d⁻¹")
    print(f"  K_S (葡萄糖) = {params['K_S']} g/L")
    print(f"  K_I (葡萄糖抑制) = {params['K_I_S']} g/L")
    S_opt = np.sqrt(params['K_S'] * params['K_I_S'])
    print(f"  最优葡萄糖浓度 = {S_opt:.1f} g/L")
    print(f"  光照 = {params['I']} μmol/m²/s")

    # 运行葡萄糖梯度模拟
    print("\n[模拟: 不同葡萄糖浓度]")
    results, glucose_levels = run_glucose_gradient()

    # 生成图表
    print("\n[生成图表]")
    plot_results(results, glucose_levels)
    plot_haldane_curve()
    plot_growth_rate_surface()

    print("\n" + "=" * 60)
    print(" 完成! 所有图表保存在:")
    print(" /Users/2488mmabd/Documents/microalgae_model/")
    print("=" * 60)
