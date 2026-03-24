"""
蛋白核小球藻 (Chlorella pyrenoidosa GY-D12) 参数拟合
====================================================

实验条件:
  - 藻株: Chlorella pyrenoidosa GY-D12 (上海光语生物科技)
  - 培养基: BG11 (pH 7.0)
  - 温度: 28°C
  - 光照: 320 μmol/(s·m²), 12h:12h 光暗周期
  - 葡萄糖梯度: 0, 1, 2, 5, 10 g/L

模型: Haldane-Monod 混合营养动力学
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from lmfit import Parameters, minimize, report_fit
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 1. 读取实验数据
# =============================================================================

def load_growth_data():
    """从 Excel 读取生长曲线数据 (均值 + 标准差)"""
    df = pd.read_excel(
        '/Users/2488mmabd/Downloads/生长曲线.xlsx',
        sheet_name='Sheet2',
        header=None
    )

    # 时间点 (天) — row 0,1,2 是标题/描述, 数据从 row 3 开始
    days = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values

    # 5 组均值和标准差 (Group 0, 1, 2, 5, 10)
    glucose_levels = [0, 1, 2, 5, 10]  # g/L
    data = {}

    mean_cols = [26, 28, 30, 32, 34]
    std_cols  = [27, 29, 31, 33, 35]

    for i, glc in enumerate(glucose_levels):
        mean_vals = pd.to_numeric(df.iloc[3:, mean_cols[i]], errors='coerce').values
        std_vals  = pd.to_numeric(df.iloc[3:, std_cols[i]], errors='coerce').values
        data[glc] = {
            'days': days,
            'mean': mean_vals,
            'std': std_vals
        }

    return data, glucose_levels


def load_photosynthesis_data():
    """读取光合活性数据"""
    df = pd.read_excel('/Users/2488mmabd/Downloads/光合活性的变化.xlsx')
    return df


# =============================================================================
# 2. 模型方程 (针对实验条件调整)
# =============================================================================

def ode_system(t, y, p):
    """
    ODE 系统 — 细胞计数版本

    状态变量:
      y[0] = X  细胞浓度 [cells/mL]
      y[1] = S  葡萄糖浓度 [g/L]
      y[2] = N  氮源浓度 [g/L]
    """
    X, S, N = y
    S = max(S, 0.0)
    N = max(N, 0.0)
    X = max(X, 1.0)

    # 异养生长 (Haldane — 葡萄糖底物抑制)
    if S > 1e-8:
        mu_het = p['mu_max_S'] * S / (p['K_S'] + S + S**2 / p['K_I_S'])
    else:
        mu_het = 0.0

    # 光合生长 (Haldane — 考虑 12:12 光暗周期)
    # 有效光照 = 320 × 光期占比
    I_eff = p['I'] * p['photoperiod']
    mu_pho = p['mu_max_I'] * I_eff / (p['K_sI'] + I_eff + I_eff**2 / p['K_I_I'])

    # 氮源限制 (Monod)
    f_N = N / (p['K_N'] + N)

    # 总生长速率
    mu = (mu_het + mu_pho) * f_N - p['m']
    mu = max(mu, 0.0)

    # 微分方程
    dXdt = mu * X
    dSdt = -mu * X / p['Y_XS'] * p['cell_to_g'] if S > 1e-8 else 0.0
    dNdt = -mu * X / p['Y_XN'] * p['cell_to_g'] if N > 1e-8 else 0.0

    return [dXdt, dSdt, dNdt]


def run_model(params_lmfit, S0, X0, N0, t_eval):
    """运行单组模拟"""
    p = {
        'mu_max_S':    params_lmfit['mu_max_S'].value,
        'K_S':         params_lmfit['K_S'].value,
        'K_I_S':       params_lmfit['K_I_S'].value,
        'mu_max_I':    params_lmfit['mu_max_I'].value,
        'K_sI':        params_lmfit['K_sI'].value,
        'K_I_I':       params_lmfit['K_I_I'].value,
        'K_N':         params_lmfit['K_N'].value,
        'm':           params_lmfit['m'].value,
        'Y_XS':        params_lmfit['Y_XS'].value,
        'Y_XN':        params_lmfit['Y_XN'].value,
        'I':           320.0,           # μmol/m²/s
        'photoperiod': 0.5,             # 12h:12h
        'cell_to_g':   params_lmfit['cell_to_g'].value,  # cells/mL → g/L 转换
    }

    y0 = [X0, S0, N0]
    t_span = (0, t_eval[-1] + 0.1)

    try:
        sol = solve_ivp(
            ode_system, t_span, y0,
            args=(p,),
            method='RK45',
            t_eval=t_eval,
            max_step=0.05,
            rtol=1e-8,
            atol=1e-6
        )
        if sol.success:
            return sol.y[0]  # 返回细胞浓度
        else:
            return np.full_like(t_eval, np.nan)
    except Exception:
        return np.full_like(t_eval, np.nan)


# =============================================================================
# 3. 参数拟合
# =============================================================================

def residuals(params_lmfit, data, glucose_levels):
    """计算所有组的残差"""
    resid = []

    # BG11 培养基中 NaNO₃ = 1.5 g/L → 氮含量约 0.247 g/L
    N0 = 0.247

    for glc in glucose_levels:
        d = data[glc]
        X0 = d['mean'][0]  # 初始细胞浓度
        t_eval = d['days']
        X_data = d['mean']
        X_std = d['std']

        X_model = run_model(params_lmfit, S0=glc, X0=X0, N0=N0, t_eval=t_eval)

        # 加权残差 (用标准差加权, 避免除以 0)
        weights = 1.0 / (X_std + 1e6)  # 加一个小量避免除零
        r = (X_model - X_data) * weights
        resid.extend(r)

    return np.array(resid)


def setup_parameters():
    """设置待拟合参数 (带边界约束)"""
    params = Parameters()

    # 异养生长参数
    params.add('mu_max_S', value=0.5, min=0.01, max=3.0)      # d⁻¹
    params.add('K_S',      value=2.0, min=0.01, max=20.0)      # g/L
    params.add('K_I_S',    value=30.0, min=5.0, max=200.0)     # g/L

    # 光合生长参数
    params.add('mu_max_I', value=0.6, min=0.01, max=3.0)       # d⁻¹
    params.add('K_sI',     value=100.0, min=10.0, max=500.0)   # μmol/m²/s
    params.add('K_I_I',    value=800.0, min=200.0, max=3000.0) # μmol/m²/s

    # 氮源限制
    params.add('K_N',      value=0.05, min=0.001, max=0.5)     # g/L

    # 维持系数
    params.add('m',        value=0.02, min=0.0, max=0.2)       # d⁻¹

    # 得率系数 (以细胞计, cells/g)
    params.add('Y_XS',     value=5e7, min=1e6, max=1e9)        # cells/g_glucose
    params.add('Y_XN',     value=5e8, min=1e7, max=1e10)       # cells/g_N

    # 细胞-生物量转换系数
    params.add('cell_to_g', value=2e-8, min=1e-10, max=1e-6)   # g/cell

    return params


def fit_model():
    """执行参数拟合"""
    print("=" * 60)
    print(" 加载实验数据...")
    data, glucose_levels = load_growth_data()

    for glc in glucose_levels:
        d = data[glc]
        print(f"  Glucose {glc:2d} g/L: X0={d['mean'][0]:.0f}, "
              f"Xmax={np.max(d['mean']):.0f} cells/mL")

    print("\n 设置参数...")
    params = setup_parameters()

    print(" 开始拟合 (least_squares)...\n")
    result = minimize(
        residuals, params,
        args=(data, glucose_levels),
        method='least_squares',
        max_nfev=5000
    )

    print("=" * 60)
    print(" 拟合结果")
    print("=" * 60)
    report_fit(result)

    return result, data, glucose_levels


# =============================================================================
# 4. 可视化
# =============================================================================

def plot_fit(result, data, glucose_levels):
    """绘制拟合结果 vs 实验数据"""
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    N0 = 0.247

    for i, glc in enumerate(glucose_levels):
        ax = axes[i]
        d = data[glc]

        # 实验数据 (误差棒)
        ax.errorbar(d['days'], d['mean'], yerr=d['std'],
                     fmt='o', color=colors[i], capsize=4,
                     markersize=6, label='实验数据')

        # 模型预测 (平滑曲线)
        t_fine = np.linspace(0, 15, 300)
        X_model = run_model(result.params, S0=glc, X0=d['mean'][0],
                           N0=N0, t_eval=t_fine)
        ax.plot(t_fine, X_model, '-', color=colors[i],
                linewidth=2, label='模型拟合')

        ax.set_xlabel('时间 [天]', fontsize=12)
        ax.set_title(f'葡萄糖 {glc} g/L', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 计算 R²
        X_pred = run_model(result.params, S0=glc, X0=d['mean'][0],
                          N0=N0, t_eval=d['days'])
        ss_res = np.nansum((d['mean'] - X_pred)**2)
        ss_tot = np.nansum((d['mean'] - np.nanmean(d['mean']))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[0].set_ylabel('细胞浓度 [cells/mL]', fontsize=12)

    fig.suptitle(
        '蛋白核小球藻 (C. pyrenoidosa GY-D12) 混合营养生长 — Haldane-Monod 拟合\n'
        '(BG11, 28°C, 320 μmol/m²/s, 12:12 L/D)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/fit_results.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  图表已保存: fit_results.png")


def plot_photosynthesis():
    """绘制光合活性随时间变化"""
    df = load_photosynthesis_data()
    glucose_levels = [0, 1, 2, 5, 10]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['alpha', 'ETR', 'IK']
    ylabels = ['α (光合效率)', 'ETR [μmol e⁻/m²/s]', 'Ik [μmol/m²/s]']
    titles = ['(a) 光合效率 α', '(b) 电子传递速率 ETR', '(c) 饱和光强 Ik']

    for j, (metric, ylabel, title) in enumerate(zip(metrics, ylabels, titles)):
        ax = axes[j]
        for i, glc in enumerate(glucose_levels):
            sub = df[df['group'] == glc]
            # 按天求均值和标准差
            grouped = sub.groupby('Day')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(grouped['Day'], grouped['mean'], yerr=grouped['std'],
                         fmt='o-', color=colors[i], capsize=3,
                         linewidth=1.5, label=f'{glc} g/L')
        ax.set_xlabel('时间 [天]', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(title='葡萄糖', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        '蛋白核小球藻 光合活性随时间变化\n(不同葡萄糖浓度)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/photosynthesis.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: photosynthesis.png")


def plot_parameter_summary(result):
    """参数汇总表"""
    names = []
    values = []
    errors = []
    units = []

    unit_map = {
        'mu_max_S': 'd⁻¹', 'K_S': 'g/L', 'K_I_S': 'g/L',
        'mu_max_I': 'd⁻¹', 'K_sI': 'μmol/m²/s', 'K_I_I': 'μmol/m²/s',
        'K_N': 'g/L', 'm': 'd⁻¹',
        'Y_XS': 'cells/g', 'Y_XN': 'cells/g', 'cell_to_g': 'g/cell'
    }

    for name, param in result.params.items():
        names.append(name)
        values.append(param.value)
        errors.append(param.stderr if param.stderr else 0)
        units.append(unit_map.get(name, ''))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    table_data = []
    for n, v, e, u in zip(names, values, errors, units):
        if abs(v) > 1e4 or abs(v) < 0.001:
            val_str = f'{v:.3e}'
            err_str = f'± {e:.2e}' if e else 'N/A'
        else:
            val_str = f'{v:.4f}'
            err_str = f'± {e:.4f}' if e else 'N/A'
        table_data.append([n, val_str, err_str, u])

    table = ax.table(
        cellText=table_data,
        colLabels=['参数', '拟合值', '标准误', '单位'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # 表头加粗
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('拟合参数汇总', fontsize=14, fontweight='bold', pad=20)
    plt.savefig('/Users/2488mmabd/Documents/microalgae_model/parameters_table.png',
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  图表已保存: parameters_table.png")


# =============================================================================
# 5. 主程序
# =============================================================================

if __name__ == '__main__':
    # 拟合
    result, data, glucose_levels = fit_model()

    # 绘图
    print("\n[生成图表]")
    plot_fit(result, data, glucose_levels)
    plot_photosynthesis()
    plot_parameter_summary(result)

    # 保存参数
    print("\n[拟合参数]")
    for name, param in result.params.items():
        stderr = f'± {param.stderr:.4e}' if param.stderr else '(fixed)'
        print(f"  {name:12s} = {param.value:.6e}  {stderr}")

    print("\n" + "=" * 60)
    print(" 完成!")
    print("=" * 60)
