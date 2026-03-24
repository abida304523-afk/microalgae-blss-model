"""
Publication-Quality Figures for Thesis
======================================
Generates formal academic figures from the v6 Logistic-Monod model
and product prediction results.

All figures use:
  - English labels
  - Arial font family
  - 300 DPI
  - Panel labels (a), (b), (c) in bold
  - Consistent color scheme
  - Proper axis formatting
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

# ── Style setup ──────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'axes.unicode_minus': False,
})

OUT = '/Users/2488mmabd/Documents/microalgae_model'

# Color palette — distinguishable, colorblind-friendly
COLORS = {
    0:  '#2ca02c',   # green  — autotrophic control
    1:  '#1f77b4',   # blue
    2:  '#ff7f0e',   # orange
    5:  '#d62728',   # red
    10: '#9467bd',   # purple
}
MARKERS = {0: 'o', 1: 's', 2: '^', 5: 'D', 10: 'v'}

# ── Constants (same as product_prediction.py) ────────────────
CELL_MASS = 25e-12          # g/cell
ABS_CROSS_SECTION = 0.02    # m²/g DW
O2_MW = 32.0

# ── Data & Model (reuse from product_prediction.py) ─────────

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


def logistic(t, X0, mu, K):
    if K <= X0:
        return np.full_like(t, K, dtype=float)
    return K * X0 / (X0 + (K - X0) * np.exp(-mu * t))


def predict(t, S, X0, p):
    mu0, mu_S, K_S, K0, K_max, S_K = p[:6]
    mu = mu0 + mu_S * S / (K_S + S) if S > 1e-10 else mu0
    K  = K0 + K_max * S / (S_K + S) if S > 1e-10 else K0
    return logistic(t, X0, mu, K)


def predict_with_x0adj(t, S, X0_raw, p, glc_idx):
    f = p[6 + glc_idx]
    return predict(t, S, X0_raw * f, p)


def cells_to_dw(X):
    return X * CELL_MASS * 1e3


def biomass_composition(S, t):
    tau = np.clip(t / 15.0, 0, 1.0)
    f_prot  = (0.55 - 0.12 * S / (2.0 + S)) * (1.0 - 0.10 * tau)
    f_carb  = (0.15 + 0.10 * S / (3.0 + S)) * (1.0 + 0.15 * tau)
    f_lipid = (0.08 + 0.12 * S / (2.5 + S)) * (1.0 + 0.40 * tau**2)
    f_total = f_prot + f_carb + f_lipid
    if np.any(f_total > 0.95):
        scale = np.where(f_total > 0.95, 0.95 / f_total, 1.0)
        f_prot  *= scale
        f_carb  *= scale
        f_lipid *= scale
    return f_prot, f_carb, f_lipid


def compute_all(data, glc_list, p_opt, photo_df):
    t_fine = np.linspace(0, 16, 500)
    results = {}
    for idx, g in enumerate(glc_list):
        d = data[g]
        X_fine = predict_with_x0adj(t_fine, g, d['mean'][0], p_opt, idx)
        DW = cells_to_dw(X_fine)
        f_prot, f_carb, f_lipid = biomass_composition(g, t_fine)
        results[g] = {
            't': t_fine, 'X': X_fine, 'DW': DW,
            'f_prot': f_prot, 'f_carb': f_carb, 'f_lipid': f_lipid,
            'prot': DW * f_prot, 'carb': DW * f_carb, 'lipid': DW * f_lipid,
        }

    # O₂
    for idx, g in enumerate(glc_list):
        sub = photo_df[photo_df['group'] == g].groupby('Day')['ETR'].mean()
        etr_days = sub.index.values.astype(float)
        etr_vals = sub.values
        if len(etr_days) >= 2:
            etr_interp = interp1d(etr_days, etr_vals, kind='linear',
                                  bounds_error=False,
                                  fill_value=(etr_vals[0], etr_vals[-1]))
            etr_fine = etr_interp(t_fine)
        else:
            etr_fine = np.full_like(t_fine, etr_vals[0] if len(etr_vals) > 0 else 50.)

        d = data[g]
        X_fine = predict_with_x0adj(t_fine, g, d['mean'][0], p_opt, idx)
        DW_fine = cells_to_dw(X_fine)
        total_abs_area = DW_fine * ABS_CROSS_SECTION
        o2_rate_umol = etr_fine * total_abs_area / 4.0
        o2_gross = o2_rate_umol * O2_MW * 1e-3 * 86400
        resp_frac = 0.15 + 0.20 * g / (5.0 + g)
        o2_net = o2_gross * (1.0 - resp_frac)
        dt = np.gradient(t_fine)
        o2_cum = np.cumsum(o2_net * dt)

        results[g].update({
            'etr_days': etr_days, 'etr_vals': etr_vals, 'etr_fine': etr_fine,
            'o2_gross': o2_gross, 'o2_net': o2_net, 'o2_cum': o2_cum,
        })

    return results, t_fine


def _add_panel_label(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def _style_ax(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ═════════════════════════════════════════════════════════════
# Figure 1: Growth & Biomass
# ═════════════════════════════════════════════════════════════

def fig1_growth_biomass(results, data, glc_list, t_fine):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for g in glc_list:
        d = data[g]; r = results[g]
        ax1.errorbar(d['days'], d['mean']/1e6, yerr=d['std']/1e6,
                     fmt=MARKERS[g], color=COLORS[g], capsize=3, ms=5,
                     markeredgecolor='k', markeredgewidth=0.3, zorder=3)
        ax1.plot(t_fine, r['X']/1e6, '-', color=COLORS[g], lw=1.8,
                 label=f'{g} g/L glucose')

    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Cell concentration ($\\times 10^6$ cells/mL)')
    ax1.legend(frameon=True, edgecolor='0.8', fancybox=False)
    _add_panel_label(ax1, '(a)')
    _style_ax(ax1)

    for g in glc_list:
        d = data[g]; r = results[g]
        DW_data = cells_to_dw(d['mean'])
        DW_err  = cells_to_dw(d['std'])
        ax2.errorbar(d['days'], DW_data, yerr=DW_err,
                     fmt=MARKERS[g], color=COLORS[g], capsize=3, ms=5,
                     markeredgecolor='k', markeredgewidth=0.3, zorder=3)
        ax2.plot(t_fine, r['DW'], '-', color=COLORS[g], lw=1.8,
                 label=f'{g} g/L glucose')

    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Dry weight (g/L)')
    ax2.legend(frameon=True, edgecolor='0.8', fancybox=False)
    _add_panel_label(ax2, '(b)')
    _style_ax(ax2)

    plt.tight_layout(w_pad=3)
    path = f'{OUT}/pub_fig1_growth.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Figure 2: Biomass Composition (2×3 panel)
# ═════════════════════════════════════════════════════════════

def fig2_composition(results, glc_list, t_fine):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    comp_keys  = ['prot', 'carb', 'lipid']
    frac_keys  = ['f_prot', 'f_carb', 'f_lipid']
    names      = ['Protein', 'Carbohydrate', 'Lipid']
    units_abs  = ['Protein (g/L)', 'Carbohydrate (g/L)', 'Lipid (g/L)']
    units_frac = ['Protein (% DW)', 'Carbohydrate (% DW)', 'Lipid (% DW)']
    panels_top = ['(a)', '(b)', '(c)']
    panels_bot = ['(d)', '(e)', '(f)']

    for j in range(3):
        for g in glc_list:
            r = results[g]
            axes[0, j].plot(t_fine, r[comp_keys[j]], '-', color=COLORS[g],
                            lw=1.8, label=f'{g} g/L')
            axes[1, j].plot(t_fine, r[frac_keys[j]]*100, '-', color=COLORS[g],
                            lw=1.8, label=f'{g} g/L')

        axes[0, j].set_xlabel('Time (days)')
        axes[0, j].set_ylabel(units_abs[j])
        axes[0, j].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=7)
        _add_panel_label(axes[0, j], panels_top[j])
        _style_ax(axes[0, j])

        axes[1, j].set_xlabel('Time (days)')
        axes[1, j].set_ylabel(units_frac[j])
        axes[1, j].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=7)
        _add_panel_label(axes[1, j], panels_bot[j])
        _style_ax(axes[1, j])

    plt.tight_layout(h_pad=3, w_pad=2.5)
    path = f'{OUT}/pub_fig2_composition.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Figure 3: O₂ Production
# ═════════════════════════════════════════════════════════════

def fig3_oxygen(results, glc_list, t_fine):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for g in glc_list:
        r = results[g]
        axes[0].plot(r['etr_days'], r['etr_vals'], MARKERS[g]+'-',
                     color=COLORS[g], ms=6, lw=1.3, label=f'{g} g/L',
                     markeredgecolor='k', markeredgewidth=0.3)
        axes[1].plot(t_fine, r['o2_net'], '-', color=COLORS[g], lw=1.8,
                     label=f'{g} g/L')
        axes[2].plot(t_fine, r['o2_cum']/1000, '-', color=COLORS[g], lw=1.8,
                     label=f'{g} g/L')

    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('ETR ($\\mu$mol e$^-$/m$^2$/s)')
    axes[0].set_title('Electron Transport Rate', fontsize=11, fontweight='bold')
    axes[0].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=8)
    _add_panel_label(axes[0], '(a)')
    _style_ax(axes[0])

    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Net O$_2$ rate (mg/L/day)')
    axes[1].set_title('Net O$_2$ Production Rate', fontsize=11, fontweight='bold')
    axes[1].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=8)
    _add_panel_label(axes[1], '(b)')
    _style_ax(axes[1])

    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Cumulative O$_2$ (g/L)')
    axes[2].set_title('Cumulative Net O$_2$', fontsize=11, fontweight='bold')
    axes[2].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=8)
    _add_panel_label(axes[2], '(c)')
    _style_ax(axes[2])

    plt.tight_layout(w_pad=3)
    path = f'{OUT}/pub_fig3_oxygen.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Figure 4: Day-15 Summary (stacked bar + table)
# ═════════════════════════════════════════════════════════════

def fig4_summary(results, glc_list, t_fine):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    idx_15 = np.argmin(np.abs(t_fine - 15.0))

    groups = [f'{g}' for g in glc_list]
    prot_v  = [results[g]['f_prot'][idx_15]*100 for g in glc_list]
    carb_v  = [results[g]['f_carb'][idx_15]*100 for g in glc_list]
    lipid_v = [results[g]['f_lipid'][idx_15]*100 for g in glc_list]
    other_v = [100 - p - c - l for p, c, l in zip(prot_v, carb_v, lipid_v)]

    x = np.arange(len(groups))
    w = 0.55

    ax1.bar(x, prot_v, w, label='Protein', color='#4472C4')
    ax1.bar(x, carb_v, w, bottom=prot_v, label='Carbohydrate', color='#ED7D31')
    b2 = [p+c for p, c in zip(prot_v, carb_v)]
    ax1.bar(x, lipid_v, w, bottom=b2, label='Lipid', color='#A5A5A5')
    b3 = [a+b for a, b in zip(b2, lipid_v)]
    ax1.bar(x, other_v, w, bottom=b3, label='Other', color='#FFC000', alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_xlabel('Glucose concentration (g/L)')
    ax1.set_ylabel('Composition (% DW)')
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=9,
               loc='upper right')
    _add_panel_label(ax1, '(a)')
    ax1.grid(True, alpha=0.2, axis='y')

    # Table
    ax2.axis('off')
    headers = ['Glucose\n(g/L)', 'DW\n(g/L)', 'Protein\n(g/L)',
               'Carb.\n(g/L)', 'Lipid\n(g/L)', 'Cum. O$_2$\n(g/L)']
    tdata = []
    for g in glc_list:
        r = results[g]
        tdata.append([
            f'{g}',
            f'{r["DW"][idx_15]:.2f}',
            f'{r["prot"][idx_15]:.3f}',
            f'{r["carb"][idx_15]:.3f}',
            f'{r["lipid"][idx_15]:.3f}',
            f'{r["o2_cum"][idx_15]/1000:.2f}',
        ])

    tbl = ax2.table(cellText=tdata, colLabels=headers,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 1.8)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=9)
    # Highlight best row (10 g/L)
    for ri in range(1, len(tdata)+1):
        if tdata[ri-1][0] == '10':
            for j in range(len(headers)):
                tbl[ri, j].set_facecolor('#E2EFDA')

    _add_panel_label(ax2, '(b)', x=-0.05, y=1.02)
    ax2.set_title('Day 15 Product Prediction Summary', fontsize=11,
                  fontweight='bold', pad=15)

    plt.tight_layout(w_pad=2)
    path = f'{OUT}/pub_fig4_summary.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Figure 5: Model Validation Summary
# ═════════════════════════════════════════════════════════════

def fig5_validation(results, glc_list, t_fine, data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # (a) LOOCV R² bar chart
    loocv_r2 = {0: 0.48, 1: 0.73, 2: 0.84, 5: 0.83, 10: 0.56}  # from validation.py
    bars = axes[0].bar([f'{g}' for g in glc_list],
                       [loocv_r2[g] for g in glc_list],
                       color=[COLORS[g] for g in glc_list], width=0.5,
                       edgecolor='k', linewidth=0.5)
    axes[0].axhline(y=0.677, color='red', ls='--', lw=1.2, label='Mean = 0.677')
    axes[0].axhline(y=0.50, color='gray', ls=':', lw=0.8, alpha=0.5)
    axes[0].set_xlabel('Removed glucose group (g/L)')
    axes[0].set_ylabel('Prediction R$^2$')
    axes[0].set_title('LOOCV Prediction', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=9)
    _add_panel_label(axes[0], '(a)')
    _style_ax(axes[0])

    # (b) Cross-species validation R²
    cs_r2 = {0: 0.97, 1: 0.98, 2: 0.97, 4: 0.93, 8: 0.96, 16: 0.98}
    cs_glc = list(cs_r2.keys())
    axes[1].bar([f'{g}' for g in cs_glc],
                [cs_r2[g] for g in cs_glc],
                color='#4472C4', width=0.5, edgecolor='k', linewidth=0.5)
    axes[1].axhline(y=0.9645, color='red', ls='--', lw=1.2, label='Mean = 0.965')
    axes[1].set_xlabel('Glucose concentration (g/L)')
    axes[1].set_ylabel('R$^2$')
    axes[1].set_title('C. sorokiniana Cross-species', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0.85, 1.0)
    axes[1].legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=9)
    _add_panel_label(axes[1], '(b)')
    _style_ax(axes[1])

    # (c) μ_max literature comparison
    lit_data = {
        'Yu et al. 2022':       1.22,
        'Wan et al. 2011':      3.40,
        'Li et al. 2013':       0.73,
        'Adesanya et al. 2014': 0.65,
        'Pagnanelli et al. 2014': 0.45,
        'This study':           0.71,
    }
    names = list(lit_data.keys())
    vals  = list(lit_data.values())
    y_pos = np.arange(len(names))

    bar_colors = ['#B4C7E7'] * (len(names)-1) + ['#FF6B6B']
    axes[2].barh(y_pos, vals, color=bar_colors, edgecolor='k', linewidth=0.5,
                 height=0.5)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(names, fontsize=9)
    axes[2].set_xlabel('$\\mu_{max}$ (d$^{-1}$)')
    axes[2].set_title('Growth Rate Comparison', fontsize=11, fontweight='bold')
    axes[2].axvline(x=0.71, color='red', ls='--', lw=1.0, alpha=0.7)
    _add_panel_label(axes[2], '(c)')
    _style_ax(axes[2])

    plt.tight_layout(w_pad=3)
    path = f'{OUT}/pub_fig5_validation.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Figure 6: Carbon Allocation Analysis
# ═════════════════════════════════════════════════════════════

def fig6_carbon_allocation(results, glc_list, t_fine):
    idx_15 = np.argmin(np.abs(t_fine - 15.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Composition fractions at Day 15 — grouped bar
    x = np.arange(len(glc_list))
    w = 0.25
    prot_f  = [results[g]['f_prot'][idx_15]*100 for g in glc_list]
    carb_f  = [results[g]['f_carb'][idx_15]*100 for g in glc_list]
    lipid_f = [results[g]['f_lipid'][idx_15]*100 for g in glc_list]

    ax1.bar(x - w, prot_f, w, label='Protein', color='#4472C4',
            edgecolor='k', linewidth=0.3)
    ax1.bar(x, carb_f, w, label='Carbohydrate', color='#ED7D31',
            edgecolor='k', linewidth=0.3)
    ax1.bar(x + w, lipid_f, w, label='Lipid', color='#A5A5A5',
            edgecolor='k', linewidth=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{g}' for g in glc_list])
    ax1.set_xlabel('Glucose concentration (g/L)')
    ax1.set_ylabel('Mass fraction (% DW)')
    ax1.legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=9)
    _add_panel_label(ax1, '(a)')
    _style_ax(ax1)

    # (b) Absolute yield at Day 15
    prot_a  = [results[g]['prot'][idx_15] for g in glc_list]
    carb_a  = [results[g]['carb'][idx_15] for g in glc_list]
    lipid_a = [results[g]['lipid'][idx_15] for g in glc_list]

    ax2.bar(x - w, prot_a, w, label='Protein', color='#4472C4',
            edgecolor='k', linewidth=0.3)
    ax2.bar(x, carb_a, w, label='Carbohydrate', color='#ED7D31',
            edgecolor='k', linewidth=0.3)
    ax2.bar(x + w, lipid_a, w, label='Lipid', color='#A5A5A5',
            edgecolor='k', linewidth=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{g}' for g in glc_list])
    ax2.set_xlabel('Glucose concentration (g/L)')
    ax2.set_ylabel('Product yield (g/L)')
    ax2.legend(frameon=True, edgecolor='0.8', fancybox=False, fontsize=9)
    _add_panel_label(ax2, '(b)')
    _style_ax(ax2)

    plt.tight_layout(w_pad=3)
    path = f'{OUT}/pub_fig6_carbon.png'
    plt.savefig(path)
    plt.close()
    print(f'  {path}')


# ═════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 60)
    print(' Publication-Quality Figures Generator')
    print('=' * 60)

    data, glc_list = load_data()
    photo_df = load_photo()
    p_opt = load_params()

    print('\n[Computing predictions]')
    results, t_fine = compute_all(data, glc_list, p_opt, photo_df)

    print('\n[Generating figures]')
    fig1_growth_biomass(results, data, glc_list, t_fine)
    fig2_composition(results, glc_list, t_fine)
    fig3_oxygen(results, glc_list, t_fine)
    fig4_summary(results, glc_list, t_fine)
    fig5_validation(results, glc_list, t_fine, data)
    fig6_carbon_allocation(results, glc_list, t_fine)

    print('\nAll 6 publication figures generated.')
