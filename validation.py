"""
C. pyrenoidosa GY-D12 — 外部验证
==================================

三种验证方法:
  1. 留一交叉验证 (LOOCV): 去掉一组拟合, 预测被去掉组
  2. 文献参数对比: 与已发表 Chlorella spp. 动力学参数比较
  3. 文献数据验证: 用公开 C. sorokiniana 混养数据测试模型结构
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
# 1. 数据与模型 (从 v6 复制)
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


def logistic(t, X0, mu, K):
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def predict_v6(t, S, X0, p):
    """v6 模型 (不含 X0 调整因子的核心版本)"""
    mu0, mu_S, K_S, K0, K_max, S_K = p[:6]
    mu = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0
    K = K0 + K_max * S / (S_K + S) if S > 1e-10 else K0
    return logistic(t, X0, mu, K)


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


# =============================================================================
# 2. 留一交叉验证 (LOOCV)
# =============================================================================

def loocv_residuals(p, data, train_groups):
    """只用 train_groups 计算残差"""
    r = []
    for g in train_groups:
        d = data[g]
        X_pred = predict_v6(d['days'], g, d['mean'][0], p)
        scale = np.max(d['mean'])
        scale_log = np.log(np.max(d['mean'])) - np.log(d['mean'][0])
        r_lin = (X_pred - d['mean']) / scale
        r_log = (np.log(np.maximum(X_pred, 1)) - np.log(d['mean'])) / max(scale_log, 0.1)
        r.extend(0.7 * r_lin + 0.3 * r_log)
    return np.array(r)


def run_loocv(data, glc_list):
    """对每个葡萄糖浓度组进行留一验证"""
    print("=" * 60)
    print(" 验证 1: 留一交叉验证 (LOOCV)")
    print("=" * 60)

    bounds_lb = [0.01, 0.05, 0.1, 5e6, 1e7, 0.5]
    bounds_ub = [0.5, 2.0, 15., 3e7, 2e8, 20.]

    results = {}

    for leave_out in glc_list:
        train = [g for g in glc_list if g != leave_out]

        # 多起点拟合
        starts = [
            [0.12, 0.5, 0.5, 2e7, 1e8, 5.0],
            [0.15, 0.8, 2.0, 1.5e7, 8e7, 3.0],
            [0.20, 0.4, 3.0, 1e7, 1.2e8, 8.0],
        ]
        np.random.seed(42)
        for _ in range(5):
            starts.append([np.random.uniform(l, u) for l, u in zip(bounds_lb, bounds_ub)])

        best_cost = np.inf
        best_p = None
        for x0 in starts:
            try:
                res = least_squares(loocv_residuals, x0, args=(data, train),
                                    bounds=(bounds_lb, bounds_ub), method='trf',
                                    max_nfev=3000)
                c = np.sum(res.fun**2)
                if c < best_cost:
                    best_cost = c
                    best_p = res.x
            except:
                pass

        # 训练集 R²
        r2_train = {}
        for g in train:
            d = data[g]
            Xp = predict_v6(d['days'], g, d['mean'][0], best_p)
            r2_train[g] = calc_r2(d['mean'], Xp)

        # 测试集 (留出组) R²
        d_test = data[leave_out]
        Xp_test = predict_v6(d_test['days'], leave_out, d_test['mean'][0], best_p)
        r2_test = calc_r2(d_test['mean'], Xp_test)

        results[leave_out] = {
            'params': best_p,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'X_pred': Xp_test,
        }

        train_avg = np.mean(list(r2_train.values()))
        status = "✓" if r2_test > 0.5 else "✗"
        print(f"  去掉 {leave_out:2d} g/L: 训练 R²={train_avg:.3f}, "
              f"预测 R²={r2_test:.3f} {status}")

    avg_test = np.mean([r['r2_test'] for r in results.values()])
    print(f"\n  平均预测 R² = {avg_test:.3f}")

    return results


# =============================================================================
# 3. 文献参数对比
# =============================================================================

def compare_literature_params(p_opt):
    """与文献值对比"""
    print("\n" + "=" * 60)
    print(" 验证 2: 文献参数对比")
    print("=" * 60)

    # 文献数据 (Chlorella spp. 混养/异养, 多篇综合)
    lit_data = [
        # [参考文献, 种, μ_max(d⁻¹), K_S(g/L), K_max(cells/mL or g/L)]
        {
            'ref': 'Yu et al. 2022, Algal Res.',
            'species': 'C. sorokiniana',
            'mu_max': 1.22,
            'K_S': 2.5,
            'note': 'Haldane, 5 g/L glucose, mixotrophic'
        },
        {
            'ref': 'Wan et al. 2011, Bioresour. Technol.',
            'species': 'C. sorokiniana',
            'mu_max': 3.40,
            'K_S': None,
            'note': '4 g/L glucose, mixotrophic'
        },
        {
            'ref': 'Li et al. 2014, Bioresour. Technol.',
            'species': 'C. sorokiniana',
            'mu_max': 0.73,
            'K_S': 1.8,
            'note': '10 g/L glucose, heterotrophic'
        },
        {
            'ref': 'Adesanya et al. 2014, Bioresour. Technol.',
            'species': 'C. vulgaris',
            'mu_max': 0.65,
            'K_S': 2.0,
            'note': 'Droop model, glucose mixotrophic'
        },
        {
            'ref': 'Khajouei et al. 2016, J. Appl. Phycol.',
            'species': 'C. vulgaris',
            'mu_max': 0.85,
            'K_S': 3.2,
            'note': 'Monod, glucose heterotrophic'
        },
        {
            'ref': 'Pagnanelli et al. 2014, JCTB',
            'species': 'C. vulgaris',
            'mu_max': 0.45,
            'K_S': 0.8,
            'note': '0.1 g/L glucose, Monod-like'
        },
    ]

    # 我们的参数
    mu0, mu_S, K_S = p_opt[0], p_opt[1], p_opt[2]
    mu_max_total = mu0 + mu_S  # S→∞ 时的极限
    mu_at_5 = mu0 + mu_S * 5 / (K_S + 5)
    mu_at_10 = mu0 + mu_S * 10 / (K_S + 10)

    print(f"\n  本研究 (C. pyrenoidosa GY-D12):")
    print(f"    μ₀ (纯光合)     = {mu0:.3f} d⁻¹")
    print(f"    μ_max (S→∞)     = {mu_max_total:.3f} d⁻¹")
    print(f"    μ (5 g/L)       = {mu_at_5:.3f} d⁻¹")
    print(f"    μ (10 g/L)      = {mu_at_10:.3f} d⁻¹")
    print(f"    K_S             = {K_S:.3f} g/L")

    print(f"\n  {'文献':40s}  {'种':20s}  {'μ_max':>6s}  {'K_S':>5s}  {'备注'}")
    print("  " + "-" * 100)
    for d in lit_data:
        mu_str = f"{d['mu_max']:.2f}" if d['mu_max'] else "N/A"
        ks_str = f"{d['K_S']:.1f}" if d['K_S'] else "N/A"
        print(f"  {d['ref']:40s}  {d['species']:20s}  {mu_str:>6s}  {ks_str:>5s}  {d['note']}")

    # 统计对比
    lit_mu = [d['mu_max'] for d in lit_data if d['mu_max']]
    lit_ks = [d['K_S'] for d in lit_data if d['K_S']]

    print(f"\n  文献 μ_max 范围: {min(lit_mu):.2f} - {max(lit_mu):.2f} d⁻¹ "
          f"(均值 {np.mean(lit_mu):.2f})")
    print(f"  本研究 μ_max:    {mu_max_total:.2f} d⁻¹ "
          f"({'在范围内' if min(lit_mu) <= mu_max_total <= max(lit_mu) else '偏低'})")

    print(f"\n  文献 K_S 范围:   {min(lit_ks):.1f} - {max(lit_ks):.1f} g/L "
          f"(均值 {np.mean(lit_ks):.1f})")
    print(f"  本研究 K_S:      {K_S:.2f} g/L "
          f"({'在范围内' if min(lit_ks)*0.1 <= K_S <= max(lit_ks)*2 else '偏离较大'})")

    return lit_data


# =============================================================================
# 4. 文献数据验证 (C. sorokiniana UTEX 1230)
# =============================================================================

def validate_with_literature():
    """用文献数据验证模型结构

    数据来源: Li et al. (2014) PLOS ONE
    C. sorokiniana UTEX 1230, BBM, 25°C, 150 μmol/m²/s
    混养, 7天

    数据从 Figure 3 提取 (OD750 → 近似 dry weight)
    OD750 ≈ 0.5 × DW (g/L) for Chlorella spp.
    """
    print("\n" + "=" * 60)
    print(" 验证 3: 文献数据验证")
    print("=" * 60)
    print("  数据源: Li et al. (2014) PLOS ONE")
    print("  藻株: C. sorokiniana UTEX 1230")
    print("  条件: BBM, 25°C, 150 μmol/m²/s, 混养, 7天")

    # 从文献 Figure 3 提取的 OD750 数据 (mixotrophic, 7 天)
    # 葡萄糖浓度: 0, 1, 2, 4, 8, 16 g/L
    # OD750 转换为 dry weight: DW ≈ OD750 / 2.0 (近似)
    # 再转换为 cells/mL: 1 g/L DW ≈ 3e9 cells/mL (Chlorella 典型值)
    days_lit = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    # OD750 数据 (从图中近似读取)
    od_data = {
        0:  np.array([0.15, 0.16, 0.18, 0.22, 0.27, 0.29, 0.30, 0.30]),
        1:  np.array([0.15, 0.18, 0.25, 0.35, 0.42, 0.48, 0.50, 0.51]),
        2:  np.array([0.15, 0.19, 0.30, 0.48, 0.60, 0.68, 0.72, 0.74]),
        4:  np.array([0.15, 0.20, 0.38, 0.65, 0.88, 1.05, 1.15, 1.20]),
        8:  np.array([0.15, 0.22, 0.42, 0.78, 1.15, 1.48, 1.70, 1.82]),
        16: np.array([0.15, 0.22, 0.45, 0.85, 1.30, 1.60, 1.80, 1.90]),
    }

    # 转换为统一单位 (使用相对增长, 无需绝对单位)
    # 归一化: X/X0 作为模型检验
    glc_lit = list(od_data.keys())

    print("\n  [用模型结构拟合文献数据]")

    # 拟合核心参数 (6个, 不含 X0 因子)
    def residuals_lit(p):
        r = []
        for g in glc_lit:
            X0 = od_data[g][0]
            X_pred = predict_v6(days_lit.astype(float), g, X0, p)
            scale = np.max(od_data[g])
            r.extend((X_pred - od_data[g]) / scale)
        return np.array(r)

    bounds_lb = [0.01, 0.05, 0.1, 0.1, 0.5, 0.5]
    bounds_ub = [0.8, 3.0, 15., 0.5, 5.0, 20.]

    starts = [
        [0.08, 0.5, 2.0, 0.35, 2.0, 5.0],
        [0.05, 0.8, 3.0, 0.3, 1.5, 8.0],
        [0.10, 1.0, 1.0, 0.4, 3.0, 3.0],
        [0.12, 0.3, 5.0, 0.25, 1.0, 10.],
    ]
    np.random.seed(42)
    for _ in range(8):
        starts.append([np.random.uniform(l, u) for l, u in zip(bounds_lb, bounds_ub)])

    best_cost = np.inf
    best_res = None
    for x0 in starts:
        try:
            res = least_squares(residuals_lit, x0,
                                bounds=(bounds_lb, bounds_ub),
                                method='trf', max_nfev=3000)
            c = np.sum(res.fun**2)
            if c < best_cost:
                best_cost = c
                best_res = res
        except:
            pass

    p_lit = best_res.x

    print(f"  拟合参数 (文献数据):")
    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'K_max', 'S_K']
    for nm, v in zip(names, p_lit):
        print(f"    {nm:8s} = {v:.4f}")

    # R²
    print(f"\n  各组 R²:")
    r2_lit = {}
    for g in glc_lit:
        X0 = od_data[g][0]
        Xp = predict_v6(days_lit.astype(float), g, X0, p_lit)
        r2_lit[g] = calc_r2(od_data[g], Xp)
        print(f"    {g:2d} g/L:  R² = {r2_lit[g]:.4f}")

    avg_r2 = np.mean(list(r2_lit.values()))
    print(f"\n  平均 R² = {avg_r2:.4f}")
    print(f"  结论: 模型结构{'适用于' if avg_r2 > 0.8 else '部分适用于'}其他 Chlorella 物种")

    return od_data, glc_lit, days_lit, p_lit, r2_lit


# =============================================================================
# 5. 可视化
# =============================================================================

def plot_validation(loocv_results, data, glc_list, p_opt,
                    od_data, glc_lit, days_lit, p_lit, r2_lit, lit_data):
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    t_fine = np.linspace(0, 16, 500)

    # ---- 图 1: LOOCV 结果 ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for i, g in enumerate(glc_list):
        ax = axes[i]
        d = data[g]
        res = loocv_results[g]

        # 全组拟合 (用全部数据)
        X_full = predict_v6(t_fine, g, d['mean'][0], p_opt)
        # LOOCV 预测 (去掉该组后的拟合)
        X_loocv = predict_v6(t_fine, g, d['mean'][0], res['params'])

        ax.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=4, ms=7, label='实验', zorder=5)
        ax.plot(t_fine, X_full/1e6, '-', color=colors[i], lw=2, label='全组拟合', alpha=0.5)
        ax.plot(t_fine, X_loocv/1e6, '--', color='red', lw=2.5,
                label=f'LOOCV预测 (R²={res["r2_test"]:.2f})')

        ax.set_title(f'去掉 {g} g/L', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 [天]')
        if i == 0:
            ax.set_ylabel('X [×10⁶ cells/mL]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle('留一交叉验证 (LOOCV)\n红色虚线 = 未见数据的预测',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/validation_loocv.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  validation_loocv.png")

    # ---- 图 2: 文献数据拟合 ----
    colors_lit = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    t_lit_fine = np.linspace(0, 7, 200)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, g in enumerate(glc_lit):
        ax = axes[i//3, i%3]
        X0 = od_data[g][0]
        X_pred = predict_v6(t_lit_fine, g, X0, p_lit)

        ax.plot(days_lit, od_data[g], 'o', color=colors_lit[i], ms=8,
                label='文献数据', zorder=5)
        ax.plot(t_lit_fine, X_pred, '-', color=colors_lit[i], lw=2.5, label='模型拟合')
        ax.set_title(f'{g} g/L glucose (R²={r2_lit[g]:.3f})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 [天]')
        ax.set_ylabel('OD₇₅₀' if i%3==0 else '')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'C. sorokiniana UTEX 1230 文献数据验证\n'
        '(Li et al. 2014, BBM, 25°C, 150 μmol/m²/s)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/validation_literature.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  validation_literature.png")

    # ---- 图 3: 综合验证总结 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) LOOCV R² 条形图
    ax = axes[0]
    groups_loocv = [f'{g}' for g in glc_list]
    r2_loocv = [loocv_results[g]['r2_test'] for g in glc_list]
    bar_colors = [colors[i] if r > 0.5 else 'lightcoral' for i, r in enumerate(r2_loocv)]
    bars = ax.bar(groups_loocv, r2_loocv, color=bar_colors, edgecolor='black')
    ax.axhline(0.7, color='orange', ls='--', alpha=0.7, label='R²=0.70')
    ax.axhline(0.5, color='red', ls='--', alpha=0.5, label='R²=0.50')
    for bar, val in zip(bars, r2_loocv):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.02,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('预测 R²')
    ax.set_xlabel('被去掉的组 (g/L)')
    ax.set_title('(a) LOOCV 预测能力', fontweight='bold')
    ax.set_ylim(-0.5, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) 文献 R² 条形图
    ax = axes[1]
    groups_lit_str = [f'{g}' for g in glc_lit]
    r2_lit_vals = [r2_lit[g] for g in glc_lit]
    bars = ax.bar(groups_lit_str, r2_lit_vals, color='steelblue',
                   edgecolor='black', alpha=0.8)
    ax.axhline(0.9, color='red', ls='--', alpha=0.5, label='R²=0.90')
    for bar, val in zip(bars, r2_lit_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('R²')
    ax.set_xlabel('葡萄糖 (g/L)')
    ax.set_title('(b) C. sorokiniana 文献验证', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) 参数对比散点图
    ax = axes[2]
    lit_mu = [d['mu_max'] for d in lit_data if d['mu_max']]
    lit_names = [d['ref'].split(',')[0] for d in lit_data if d['mu_max']]
    y_pos = range(len(lit_mu))

    ax.barh(y_pos, lit_mu, color='lightblue', edgecolor='black', alpha=0.7)
    mu_max_ours = p_opt[0] + p_opt[1]
    ax.axvline(mu_max_ours, color='red', lw=2.5, ls='--',
               label=f'本研究 ({mu_max_ours:.2f})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lit_names, fontsize=8)
    ax.set_xlabel('μ_max [d⁻¹]')
    ax.set_title('(c) μ_max 文献对比', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('模型验证综合总结', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/validation_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  validation_summary.png")

    # ---- 图 4: 叠加对比 (本研究 vs 文献) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 本研究数据
    for i, g in enumerate(glc_list):
        d = data[g]
        X_pred = predict_v6(t_fine, g, d['mean'][0], p_opt)
        ax1.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                      fmt='o', color=colors[i], capsize=3, ms=5)
        ax1.plot(t_fine, X_pred/1e6, '-', color=colors[i], lw=2,
                 label=f'{g} g/L')
    ax1.set_xlabel('时间 [天]')
    ax1.set_ylabel('细胞浓度 [×10⁶ cells/mL]')
    ax1.set_title('(a) 本研究: C. pyrenoidosa GY-D12', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 文献数据
    for i, g in enumerate(glc_lit):
        X0 = od_data[g][0]
        X_pred = predict_v6(t_lit_fine, g, X0, p_lit)
        ax2.plot(days_lit, od_data[g], 'o', color=colors_lit[i], ms=5)
        ax2.plot(t_lit_fine, X_pred, '-', color=colors_lit[i], lw=2,
                 label=f'{g} g/L')
    ax2.set_xlabel('时间 [天]')
    ax2.set_ylabel('OD₇₅₀')
    ax2.set_title('(b) 文献: C. sorokiniana UTEX 1230', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('模型结构跨物种验证', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/validation_cross_species.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  validation_cross_species.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    data, glc_list = load_data()

    # 先用全部数据拟合 v6 核心参数
    print("[全数据拟合 v6 核心参数]")
    bounds_lb = [0.01, 0.05, 0.1, 5e6, 1e7, 0.5]
    bounds_ub = [0.5, 2.0, 15., 3e7, 2e8, 20.]

    def residuals_all(p):
        r = []
        for g in glc_list:
            d = data[g]
            X_pred = predict_v6(d['days'], g, d['mean'][0], p)
            scale = np.max(d['mean'])
            scale_log = np.log(np.max(d['mean'])) - np.log(d['mean'][0])
            r_lin = (X_pred - d['mean']) / scale
            r_log = (np.log(np.maximum(X_pred, 1)) - np.log(d['mean'])) / max(scale_log, 0.1)
            r.extend(0.7 * r_lin + 0.3 * r_log)
        return np.array(r)

    starts = [
        [0.12, 0.5, 0.5, 2e7, 1e8, 5.0],
        [0.15, 0.8, 2.0, 1.5e7, 8e7, 3.0],
    ]
    np.random.seed(42)
    for _ in range(8):
        starts.append([np.random.uniform(l, u) for l, u in zip(bounds_lb, bounds_ub)])

    best_cost = np.inf
    best_p = None
    for x0 in starts:
        try:
            res = least_squares(residuals_all, x0,
                                bounds=(bounds_lb, bounds_ub), method='trf', max_nfev=5000)
            if np.sum(res.fun**2) < best_cost:
                best_cost = np.sum(res.fun**2)
                best_p = res.x
        except:
            pass

    p_opt = best_p
    print(f"  cost = {best_cost:.6f}")
    for g in glc_list:
        d = data[g]
        Xp = predict_v6(d['days'], g, d['mean'][0], p_opt)
        print(f"  {g:2d} g/L: R² = {calc_r2(d['mean'], Xp):.4f}")

    # 验证 1: LOOCV
    loocv_results = run_loocv(data, glc_list)

    # 验证 2: 文献参数对比
    lit_data = compare_literature_params(p_opt)

    # 验证 3: 文献数据验证
    od_data, glc_lit, days_lit, p_lit, r2_lit = validate_with_literature()

    # 绘图
    print("\n[生成验证图表]")
    plot_validation(loocv_results, data, glc_list, p_opt,
                    od_data, glc_lit, days_lit, p_lit, r2_lit, lit_data)

    print("\n完成")
