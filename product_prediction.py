"""
C. pyrenoidosa GY-D12 — 产物预测模块
======================================

基于 v6 Logistic-Monod 生长模型输出，利用文献化学计量关系
预测四类产物:
  1. O₂ 产生 (基于 ETR 光合数据)
  2. 蛋白质含量
  3. 碳水化合物含量
  4. 脂质含量

数据源:
  - 生长曲线.xlsx (细胞浓度)
  - 光合活性的变化.xlsx (α, ETR, IK)
  - params_v6.csv (拟合参数)

所有产物预测均基于文献值，无直接实验测量。

文献参考:
  - Illman et al. 2000, Enzyme Microb. Technol. — 细胞干重 22-28 pg/cell
  - Safi et al. 2014, Renew. Sustain. Energy Rev. — Chlorella 综述
  - Liang et al. 2009, Bioresour. Technol. — C. vulgaris 混养组成
  - Heredia-Arroyo et al. 2011, Carbohydr. Polym. — 葡萄糖对组成影响
  - Ho et al. 2012, Bioresour. Technol. — 碳水化合物 12-30% DW
  - Dragone et al. 2011, Appl. Energy — 碳水随葡萄糖增加
  - Suggett et al. 2011, J. Phycol. — ETR 与 O₂ 转换
  - Kliphuis et al. 2012, Biotechnol. Bioeng. — 光合效率与 O₂
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = '/Users/2488mmabd/Documents/microalgae_model'

# =============================================================================
# 文献常数
# =============================================================================

CELL_MASS = 25e-12          # g/cell (25 pg, Chlorella 典型值)
CARBON_FRACTION = 0.50      # 干重中碳含量 50%
O2_MW = 32.0                # g/mol
# 光合有效吸收截面积 (Chlorella, Kliphuis 2012)
# a* ≈ 0.02 m²/g DW (远小于几何表面积)
ABS_CROSS_SECTION = 0.02    # m²/g DW


# =============================================================================
# 1. 数据加载 (与 v6 一致)
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


def load_params():
    df = pd.read_csv(f'{OUT}/params_v6.csv')
    return df['值'].values


# =============================================================================
# 2. v6 模型 (复制核心函数)
# =============================================================================

def logistic(t, X0, mu, K):
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def predict(t, S, X0, p):
    mu0, mu_S, K_S, K0, K_max, S_K = p[:6]
    mu = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0
    K = K0 + K_max * S / (S_K + S) if S > 1e-10 else K0
    return logistic(t, X0, mu, K)


def predict_with_x0adj(t, S, X0_raw, p, glc_idx):
    f = p[6 + glc_idx]
    X0 = X0_raw * f
    return predict(t, S, X0, p)


# =============================================================================
# 3. 干重转换
# =============================================================================

def cells_to_dw(X_cells):
    """细胞浓度 [cells/mL] → 干重 [g/L]"""
    return X_cells * CELL_MASS * 1e3  # ×1000 mL/L


# =============================================================================
# 4. 生物质组成预测 (文献经验关系)
# =============================================================================

def biomass_composition(S, t):
    """
    返回蛋白质、碳水、脂质的质量分数 (占干重)
    S: 葡萄糖浓度 [g/L]
    t: 时间 [天]

    趋势:
      - 高葡萄糖 → 蛋白质↓, 碳水↑, 脂质↑
      - 晚期培养 → 蛋白质↓, 碳水↑, 脂质↑↑ (N 限制)
    """
    tau = np.clip(t / 15.0, 0, 1.0)  # 归一化时间

    # 蛋白质: 自养 55% → 混养/异养 43%, 晚期下降 10%
    f_prot = (0.55 - 0.12 * S / (2.0 + S)) * (1.0 - 0.10 * tau)

    # 碳水化合物: 自养 15% → 混养 25%, 晚期积累 +15%
    f_carb = (0.15 + 0.10 * S / (3.0 + S)) * (1.0 + 0.15 * tau)

    # 脂质: 自养 8% → 混养 20%, 晚期加速积累
    f_lipid = (0.08 + 0.12 * S / (2.5 + S)) * (1.0 + 0.40 * tau**2)

    # 归一化: 确保三项不超过 95% (留 5% 给灰分等)
    f_total = f_prot + f_carb + f_lipid
    if np.any(f_total > 0.95):
        scale = np.where(f_total > 0.95, 0.95 / f_total, 1.0)
        f_prot = f_prot * scale
        f_carb = f_carb * scale
        f_lipid = f_lipid * scale

    return f_prot, f_carb, f_lipid


# =============================================================================
# 5. O₂ 产生预测
# =============================================================================

def compute_o2(photo_df, data, glc_list, p_opt, t_fine):
    """
    基于 ETR 数据计算 O₂ 产生

    ETR [μmol e⁻/m²/s] → O₂ [mg/L/day]

    使用光合有效吸收截面积 a* [m²/g DW] 而非几何表面积:
      total_absorbing_area [m²/L] = DW [g/L] × a* [m²/g]
      O₂_rate [μmol/L/s] = ETR × total_area / 4
      → mg/L/day

    文献: a* ≈ 0.02 m²/g DW (Kliphuis et al. 2012)
    """
    o2_results = {}

    for idx, g in enumerate(glc_list):
        # ETR 均值 per day
        sub = photo_df[photo_df['group'] == g].groupby('Day')['ETR'].mean()
        etr_days = sub.index.values.astype(float)
        etr_vals = sub.values

        # 插值 ETR 到 t_fine
        if len(etr_days) >= 2:
            etr_interp = interp1d(etr_days, etr_vals, kind='linear',
                                  bounds_error=False,
                                  fill_value=(etr_vals[0], etr_vals[-1]))
            etr_fine = etr_interp(t_fine)
        else:
            etr_fine = np.full_like(t_fine, etr_vals[0] if len(etr_vals) > 0 else 50.)

        # 模型预测 X(t) → DW(t)
        d = data[g]
        X_fine = predict_with_x0adj(t_fine, g, d['mean'][0], p_opt, idx)
        DW_fine = cells_to_dw(X_fine)  # g/L

        # 光合有效吸收面积 [m²/L]
        total_abs_area = DW_fine * ABS_CROSS_SECTION

        # O₂ 产生速率
        # ETR [μmol e⁻/m²/s] × area [m²/L] / 4 [e⁻/O₂] = μmol O₂/L/s
        o2_rate_umol = etr_fine * total_abs_area / 4.0

        # → mg O₂/L/day
        o2_gross = o2_rate_umol * O2_MW * 1e-3 * 86400

        # 呼吸损失: 自养 15%, 高糖增至 35%
        resp_frac = 0.15 + 0.20 * g / (5.0 + g)
        o2_net = o2_gross * (1.0 - resp_frac)

        # 累积 O₂ [mg/L]
        dt = np.gradient(t_fine)
        o2_cum = np.cumsum(o2_net * dt)

        o2_results[g] = {
            'etr_days': etr_days, 'etr_vals': etr_vals,
            'etr_fine': etr_fine,
            'o2_gross': o2_gross, 'o2_net': o2_net, 'o2_cum': o2_cum,
        }

    return o2_results


# =============================================================================
# 6. 综合计算
# =============================================================================

def compute_all(data, glc_list, p_opt, photo_df):
    t_fine = np.linspace(0, 16, 500)

    results = {}
    for idx, g in enumerate(glc_list):
        d = data[g]
        X_fine = predict_with_x0adj(t_fine, g, d['mean'][0], p_opt, idx)
        DW = cells_to_dw(X_fine)

        f_prot, f_carb, f_lipid = biomass_composition(g, t_fine)

        prot = DW * f_prot
        carb = DW * f_carb
        lipid = DW * f_lipid

        results[g] = {
            't': t_fine, 'X': X_fine, 'DW': DW,
            'f_prot': f_prot, 'f_carb': f_carb, 'f_lipid': f_lipid,
            'prot': prot, 'carb': carb, 'lipid': lipid,
        }

    # O₂
    o2_results = compute_o2(photo_df, data, glc_list, p_opt, t_fine)
    for g in glc_list:
        results[g].update(o2_results[g])

    return results, t_fine


# =============================================================================
# 7. 绘图
# =============================================================================

def plot_biomass(results, data, glc_list, t_fine):
    """图 1: 细胞浓度与干重"""
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, g in enumerate(glc_list):
        d = data[g]
        r = results[g]

        ax1.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt='o', color=colors[i], capsize=3, ms=6)
        ax1.plot(t_fine, r['X']/1e6, '-', color=colors[i], lw=2,
                 label=f'{g} g/L')

        DW_data = cells_to_dw(d['mean'])
        ax2.plot(d['days'], DW_data, 'o', color=colors[i], ms=6)
        ax2.plot(t_fine, r['DW'], '-', color=colors[i], lw=2,
                 label=f'{g} g/L')

    ax1.set_xlabel('时间 [天]')
    ax1.set_ylabel('X [×10⁶ cells/mL]')
    ax1.set_title('(a) 细胞浓度', fontweight='bold')
    ax1.legend(title='葡萄糖', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('时间 [天]')
    ax2.set_ylabel('干重 [g/L]')
    ax2.set_title('(b) 生物量干重 (25 pg/cell)', fontweight='bold')
    ax2.legend(title='葡萄糖', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('C. pyrenoidosa GY-D12 — 生物量', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/product_biomass.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  product_biomass.png")


def plot_composition(results, glc_list, t_fine):
    """图 2: 生物质组成"""
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    titles_top = ['(a) 蛋白质', '(b) 碳水化合物', '(c) 脂质']
    titles_bot = ['(d) 蛋白质比例', '(e) 碳水化合物比例', '(f) 脂质比例']
    keys_abs = ['prot', 'carb', 'lipid']
    keys_frac = ['f_prot', 'f_carb', 'f_lipid']
    ylabels_top = ['蛋白质 [g/L]', '碳水化合物 [g/L]', '脂质 [g/L]']
    ylabels_bot = ['蛋白质 [%DW]', '碳水化合物 [%DW]', '脂质 [%DW]']

    for j in range(3):
        for i, g in enumerate(glc_list):
            r = results[g]
            axes[0, j].plot(t_fine, r[keys_abs[j]], '-', color=colors[i], lw=2,
                            label=f'{g} g/L')
            axes[1, j].plot(t_fine, r[keys_frac[j]]*100, '-', color=colors[i], lw=2,
                            label=f'{g} g/L')

        axes[0, j].set_title(titles_top[j], fontweight='bold')
        axes[0, j].set_xlabel('时间 [天]')
        axes[0, j].set_ylabel(ylabels_top[j])
        axes[0, j].legend(fontsize=7)
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].set_title(titles_bot[j], fontweight='bold')
        axes[1, j].set_xlabel('时间 [天]')
        axes[1, j].set_ylabel(ylabels_bot[j])
        axes[1, j].legend(fontsize=7)
        axes[1, j].grid(True, alpha=0.3)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — 生物质组成预测 (文献化学计量法)\n'
        '(Liang 2009; Heredia-Arroyo 2011; Ho 2012)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/product_composition.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  product_composition.png")


def plot_o2(results, glc_list, t_fine):
    """图 3: O₂ 产生"""
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for i, g in enumerate(glc_list):
        r = results[g]

        # (a) ETR 实测
        axes[0].plot(r['etr_days'], r['etr_vals'], 'o-', color=colors[i],
                     ms=6, lw=1.5, label=f'{g} g/L')

        # (b) 净 O₂ 速率
        axes[1].plot(t_fine, r['o2_net'], '-', color=colors[i], lw=2,
                     label=f'{g} g/L')

        # (c) 累积 O₂
        axes[2].plot(t_fine, r['o2_cum'], '-', color=colors[i], lw=2,
                     label=f'{g} g/L')

    axes[0].set_xlabel('时间 [天]')
    axes[0].set_ylabel('ETR [μmol e⁻/m²/s]')
    axes[0].set_title('(a) 电子传递速率 (实测)', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('时间 [天]')
    axes[1].set_ylabel('净 O₂ 速率 [mg/L/day]')
    axes[1].set_title('(b) 净 O₂ 产生速率', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('时间 [天]')
    axes[2].set_ylabel('累积 O₂ [mg/L]')
    axes[2].set_title('(c) 累积 O₂ 产生', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        'C. pyrenoidosa GY-D12 — O₂ 产生预测\n'
        '(ETR → O₂, Suggett 2011; Kliphuis 2012)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/product_o2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  product_o2.png")


def plot_summary(results, glc_list, t_fine):
    """图 4: 综合总结"""
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    # (a) Day 15 组成堆叠柱图
    groups = [f'{g} g/L' for g in glc_list]
    idx_15 = np.argmin(np.abs(t_fine - 15.0))

    prot_vals = [results[g]['f_prot'][idx_15]*100 for g in glc_list]
    carb_vals = [results[g]['f_carb'][idx_15]*100 for g in glc_list]
    lipid_vals = [results[g]['f_lipid'][idx_15]*100 for g in glc_list]
    other_vals = [100 - p - c - l for p, c, l in zip(prot_vals, carb_vals, lipid_vals)]

    x_pos = np.arange(len(groups))
    w = 0.6

    ax1.bar(x_pos, prot_vals, w, label='蛋白质', color='#4472C4')
    ax1.bar(x_pos, carb_vals, w, bottom=prot_vals, label='碳水化合物', color='#ED7D31')
    bottom2 = [p+c for p, c in zip(prot_vals, carb_vals)]
    ax1.bar(x_pos, lipid_vals, w, bottom=bottom2, label='脂质', color='#A5A5A5')
    bottom3 = [b+l for b, l in zip(bottom2, lipid_vals)]
    ax1.bar(x_pos, other_vals, w, bottom=bottom3, label='其他', color='#FFC000', alpha=0.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel('组成比例 [%DW]')
    ax1.set_title('(a) Day 15 生物质组成', fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')

    # (b) 产物汇总表
    ax2.axis('off')
    table_data = []
    headers = ['葡萄糖\n[g/L]', '干重\n[g/L]', '蛋白质\n[g/L]', '碳水\n[g/L]',
               '脂质\n[g/L]', '累积O₂\n[mg/L]']

    for g in glc_list:
        r = results[g]
        table_data.append([
            f'{g}',
            f'{r["DW"][idx_15]:.2f}',
            f'{r["prot"][idx_15]:.3f}',
            f'{r["carb"][idx_15]:.3f}',
            f'{r["lipid"][idx_15]:.3f}',
            f'{r["o2_cum"][idx_15]:.0f}',
        ])

    tbl = ax2.table(cellText=table_data, colLabels=headers,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.8)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # 高亮最大值行
    for row_idx in range(1, len(table_data)+1):
        if table_data[row_idx-1][0] == '10':
            for j in range(len(headers)):
                tbl[row_idx, j].set_facecolor('#E2EFDA')

    ax2.set_title('(b) Day 15 产物预测汇总', fontsize=13, fontweight='bold', pad=20)

    fig.suptitle('C. pyrenoidosa GY-D12 — 产物预测总结',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/product_summary.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  product_summary.png")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print(" C. pyrenoidosa GY-D12 — 产物预测模块")
    print("=" * 60)

    # 加载
    data, glc_list = load_data()
    photo_df = load_photo()
    p_opt = load_params()

    print(f"\n[v6 参数] (from params_v6.csv)")
    names = ['μ₀', 'μ_S', 'K_S', 'K₀', 'K_max', 'S_K']
    for nm, v in zip(names, p_opt[:6]):
        print(f"  {nm:6s} = {v:.4e}" if abs(v) > 1e4 else f"  {nm:6s} = {v:.4f}")

    # 计算
    print("\n[计算产物预测]")
    results, t_fine = compute_all(data, glc_list, p_opt, photo_df)

    # 打印 Day 15 结果
    idx_15 = np.argmin(np.abs(t_fine - 15.0))
    print(f"\n{'='*60}")
    print(f" Day 15 产物预测")
    print(f"{'='*60}")
    print(f"  {'葡萄糖':>6s}  {'干重':>8s}  {'蛋白质':>8s}  {'碳水':>8s}  "
          f"{'脂质':>8s}  {'累积O₂':>10s}")
    print(f"  {'[g/L]':>6s}  {'[g/L]':>8s}  {'[g/L]':>8s}  {'[g/L]':>8s}  "
          f"{'[g/L]':>8s}  {'[mg/L]':>10s}")
    print("  " + "-" * 60)
    for g in glc_list:
        r = results[g]
        print(f"  {g:6d}  {r['DW'][idx_15]:8.3f}  {r['prot'][idx_15]:8.4f}  "
              f"{r['carb'][idx_15]:8.4f}  {r['lipid'][idx_15]:8.4f}  "
              f"{r['o2_cum'][idx_15]:10.1f}")

    print(f"\n  组成比例 (%DW):")
    for g in glc_list:
        r = results[g]
        fp = r['f_prot'][idx_15]*100
        fc = r['f_carb'][idx_15]*100
        fl = r['f_lipid'][idx_15]*100
        print(f"    {g:2d} g/L: 蛋白={fp:.1f}%, 碳水={fc:.1f}%, 脂质={fl:.1f}%, "
              f"其他={100-fp-fc-fl:.1f}%")

    # 绘图
    print("\n[生成图表]")
    plot_biomass(results, data, glc_list, t_fine)
    plot_composition(results, glc_list, t_fine)
    plot_o2(results, glc_list, t_fine)
    plot_summary(results, glc_list, t_fine)

    # 导出 CSV
    rows = []
    for g in glc_list:
        r = results[g]
        for i, t in enumerate(t_fine):
            rows.append({
                'glucose_gL': g, 'day': t,
                'X_cells_mL': r['X'][i], 'DW_gL': r['DW'][i],
                'protein_gL': r['prot'][i], 'carb_gL': r['carb'][i],
                'lipid_gL': r['lipid'][i],
                'f_protein': r['f_prot'][i], 'f_carb': r['f_carb'][i],
                'f_lipid': r['f_lipid'][i],
                'O2_net_mg_L_day': r['o2_net'][i],
                'O2_cumulative_mg_L': r['o2_cum'][i],
            })
    pd.DataFrame(rows).to_csv(f'{OUT}/product_predictions.csv',
                               index=False, encoding='utf-8-sig')
    print("  product_predictions.csv")

    print("\n完成")
