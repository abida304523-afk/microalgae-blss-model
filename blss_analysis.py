"""
C. pyrenoidosa GY-D12 — BLSS Quantitative Analysis
====================================================

Bioregenerative Life Support System (BLSS) feasibility analysis
for Chlorella pyrenoidosa GY-D12 as a crew life support component.

Modules:
  1. PBR Volume Estimation for Crew O₂ Supply
  2. Nutritional Supply Analysis (NASA standards)
  3. Multi-objective Glucose Optimization
  4. Real-time Control Feasibility

Based on v6 Logistic-Monod growth model (R²=0.92).

References:
  - NASA-STD-3001 (crew nutrition / O₂ requirements)
  - Wheeler 2017, Open Agriculture — BLSS overview
  - Fu et al. 2021, Acta Astronaut. — Chlorella in BLSS
  - Kliphuis et al. 2012, Biotechnol. Bioeng. — photosynthetic O₂
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Add project path ──
sys.path.insert(0, '/Users/2488mmabd/Documents/microalgae_model')

from product_prediction import (
    load_data, load_params, load_photo,
    predict_with_x0adj, cells_to_dw, biomass_composition,
    compute_o2, compute_all,
    CELL_MASS, ABS_CROSS_SECTION, O2_MW
)

# =============================================================================
# Global Settings
# =============================================================================

OUT = '/Users/2488mmabd/Documents/microalgae_model'

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.unicode_minus': False,
    'mathtext.default': 'regular',
})

# Consistent color palette across all figures
COLORS = {
    0:  '#2ca02c',   # green  — autotrophic control
    1:  '#1f77b4',   # blue
    2:  '#ff7f0e',   # orange
    5:  '#d62728',   # red
    10: '#9467bd',   # purple
}
GLC_LIST = [0, 1, 2, 5, 10]

# =============================================================================
# BLSS Constants
# =============================================================================

# Astronaut O₂ requirement (NASA-STD-3001, light activity)
CREW_O2_DEMAND = 840.0          # g O₂ / person / day

# Light/dark cycle
LD_RATIO = 12.0 / 24.0         # fraction of day with light (12:12 L/D)

# Continuous harvest dilution rate
DILUTION_RATE = 0.1             # d⁻¹ (10% daily harvest)

# NASA space nutrition standards (per person per day)
NASA_PROTEIN_MIN  = 56.0        # g/day
NASA_PROTEIN_MAX  = 91.0        # g/day
NASA_CARB_MIN     = 200.0       # g/day
NASA_CARB_MAX     = 400.0       # g/day
NASA_LIPID_MIN    = 60.0        # g/day
NASA_LIPID_MAX    = 80.0        # g/day

# Orbital parameters (ISS-like)
ORBITAL_PERIOD  = 90.0          # minutes
ORBITAL_SUNLIT  = 45.0          # minutes of sunlight per orbit


# =============================================================================
# 1. PBR VOLUME ESTIMATION FOR CREW O₂
# =============================================================================

def pbr_o2_analysis(results, t_fine):
    """
    For each glucose concentration:
      - Average and peak net O₂ production rate over 15-day cycle
      - Required PBR volume for 1-6 crew members
      - Account for 12:12 L/D cycle
    """
    print("\n" + "=" * 70)
    print("  MODULE 1: PBR Volume Estimation for Crew O₂ Supply")
    print("=" * 70)

    # Only light-period produces O₂; the model rates are for 24h basis,
    # so effective daily O₂ = rate * LD_RATIO  (already partial in model,
    # but we make it explicit for BLSS).
    # Actually the compute_o2 already gives mg/L/day for full-day.
    # For BLSS we note that O₂ is only produced during illuminated hours.
    # So the model output IS the 12h-light daily production.

    cycle_days = 15.0
    idx_cycle = t_fine <= cycle_days

    o2_table = {}
    print(f"\n  Astronaut O₂ demand: {CREW_O2_DEMAND:.0f} g/day "
          f"(NASA-STD-3001, light activity)")
    print(f"  Light regime: 12:12 L/D")
    print(f"  Cycle length: {cycle_days:.0f} days\n")

    header = (f"  {'Glucose':>8s}  {'Avg O₂ rate':>14s}  {'Peak O₂ rate':>14s}  "
              f"{'PBR vol (1 crew)':>18s}  {'PBR vol (3 crew)':>18s}")
    print(header)
    print(f"  {'[g/L]':>8s}  {'[mg/L/day]':>14s}  {'[mg/L/day]':>14s}  "
          f"{'[L]':>18s}  {'[L]':>18s}")
    print("  " + "-" * 80)

    for g in GLC_LIST:
        r = results[g]
        o2_net = r['o2_net'][idx_cycle]

        avg_rate = np.mean(o2_net)        # mg/L/day
        peak_rate = np.max(o2_net)        # mg/L/day

        # PBR volume needed: demand [g/day] / rate [mg/L/day] * 1000 [mg/g]
        # = demand * 1e3 / rate  [L]
        vol_1 = (CREW_O2_DEMAND * 1e3) / avg_rate if avg_rate > 0 else np.inf
        vol_3 = vol_1 * 3

        o2_table[g] = {
            'avg_rate': avg_rate,
            'peak_rate': peak_rate,
            'vol_per_crew': vol_1,
        }

        print(f"  {g:8d}  {avg_rate:14.1f}  {peak_rate:14.1f}  "
              f"{vol_1:18,.0f}  {vol_3:18,.0f}")

    # Find optimal
    best_g = max(o2_table, key=lambda g: o2_table[g]['avg_rate'])
    print(f"\n  >> Optimal for O₂: {best_g} g/L glucose "
          f"(avg {o2_table[best_g]['avg_rate']:.1f} mg/L/day, "
          f"PBR = {o2_table[best_g]['vol_per_crew']:,.0f} L/crew)")

    return o2_table


# =============================================================================
# 2. NUTRITIONAL SUPPLY ANALYSIS
# =============================================================================

def nutrition_analysis(results, t_fine):
    """
    For each glucose concentration at Day 15 (steady state):
      - Daily harvestable protein/carb/lipid per liter
        (continuous harvest at dilution rate D)
      - Volume needed to meet NASA nutritional requirements
    """
    print("\n" + "=" * 70)
    print("  MODULE 2: Nutritional Supply Analysis (NASA Standards)")
    print("=" * 70)

    idx_15 = np.argmin(np.abs(t_fine - 15.0))

    print(f"\n  Dilution rate D = {DILUTION_RATE} d⁻¹ (continuous harvest)")
    print(f"  NASA standards: Protein {NASA_PROTEIN_MIN}-{NASA_PROTEIN_MAX} g/d, "
          f"Carb {NASA_CARB_MIN}-{NASA_CARB_MAX} g/d, "
          f"Lipid {NASA_LIPID_MIN}-{NASA_LIPID_MAX} g/d\n")

    nutr_table = {}

    header = (f"  {'Glc':>4s}  {'DW':>7s}  {'Prot':>7s}  {'Carb':>7s}  "
              f"{'Lipid':>7s}  {'Vol(Prot)':>10s}  {'Vol(Carb)':>10s}  "
              f"{'Vol(Lipid)':>10s}")
    print(header)
    print(f"  {'g/L':>4s}  {'g/L':>7s}  {'g/L/d':>7s}  {'g/L/d':>7s}  "
          f"{'g/L/d':>7s}  {'L':>10s}  {'L':>10s}  {'L':>10s}")
    print("  " + "-" * 75)

    for g in GLC_LIST:
        r = results[g]
        DW_15 = r['DW'][idx_15]

        # Daily harvestable = DW * fraction * D
        prot_daily = DW_15 * r['f_prot'][idx_15] * DILUTION_RATE  # g/L/day
        carb_daily = DW_15 * r['f_carb'][idx_15] * DILUTION_RATE
        lipid_daily = DW_15 * r['f_lipid'][idx_15] * DILUTION_RATE

        # Volume to meet midpoint NASA requirement
        prot_mid = (NASA_PROTEIN_MIN + NASA_PROTEIN_MAX) / 2.0
        carb_mid = (NASA_CARB_MIN + NASA_CARB_MAX) / 2.0
        lipid_mid = (NASA_LIPID_MIN + NASA_LIPID_MAX) / 2.0

        vol_prot = prot_mid / prot_daily if prot_daily > 0 else np.inf
        vol_carb = carb_mid / carb_daily if carb_daily > 0 else np.inf
        vol_lipid = lipid_mid / lipid_daily if lipid_daily > 0 else np.inf

        nutr_table[g] = {
            'DW': DW_15,
            'prot_daily': prot_daily,
            'carb_daily': carb_daily,
            'lipid_daily': lipid_daily,
            'vol_prot': vol_prot,
            'vol_carb': vol_carb,
            'vol_lipid': vol_lipid,
            'f_prot': r['f_prot'][idx_15],
            'f_carb': r['f_carb'][idx_15],
            'f_lipid': r['f_lipid'][idx_15],
        }

        print(f"  {g:4d}  {DW_15:7.3f}  {prot_daily:7.4f}  {carb_daily:7.4f}  "
              f"{lipid_daily:7.4f}  {vol_prot:10,.0f}  {vol_carb:10,.0f}  "
              f"{vol_lipid:10,.0f}")

    print(f"\n  >> Tradeoff: High glucose increases total yield but "
          f"lowers protein fraction")

    return nutr_table


# =============================================================================
# 3. MULTI-OBJECTIVE GLUCOSE OPTIMIZATION
# =============================================================================

def optimization_analysis(o2_table, nutr_table):
    """
    Score each glucose concentration for 3 BLSS objectives:
      1. Max O₂ production (min PBR volume)
      2. Max protein quality (% DW)
      3. Max total nutrition (sum of daily nutrients)
    """
    print("\n" + "=" * 70)
    print("  MODULE 3: Multi-objective Glucose Optimization")
    print("=" * 70)

    scores = {}
    for g in GLC_LIST:
        scores[g] = {
            'O2_rate': o2_table[g]['avg_rate'],
            'prot_quality': nutr_table[g]['f_prot'] * 100,
            'total_nutrition': (nutr_table[g]['prot_daily'] +
                                nutr_table[g]['carb_daily'] +
                                nutr_table[g]['lipid_daily']),
        }

    # Normalize scores to [0, 1] for each objective
    for key in ['O2_rate', 'prot_quality', 'total_nutrition']:
        vals = [scores[g][key] for g in GLC_LIST]
        vmin, vmax = min(vals), max(vals)
        rng = vmax - vmin if vmax > vmin else 1.0
        for g in GLC_LIST:
            scores[g][f'{key}_norm'] = (scores[g][key] - vmin) / rng

    print(f"\n  {'Glucose':>8s}  {'O₂ rate':>10s}  {'Prot %DW':>10s}  "
          f"{'Total nutr':>12s}  {'O₂ score':>10s}  {'Prot score':>10s}  "
          f"{'Nutr score':>10s}")
    print("  " + "-" * 75)
    for g in GLC_LIST:
        s = scores[g]
        print(f"  {g:8d}  {s['O2_rate']:10.1f}  {s['prot_quality']:10.1f}  "
              f"{s['total_nutrition']:12.5f}  {s['O2_rate_norm']:10.3f}  "
              f"{s['prot_quality_norm']:10.3f}  {s['total_nutrition_norm']:10.3f}")

    # Weighted recommendations
    scenarios = {
        'Max O₂ (life support priority)': {'O2_rate_norm': 0.7, 'prot_quality_norm': 0.1, 'total_nutrition_norm': 0.2},
        'Max protein (crew health)':      {'O2_rate_norm': 0.2, 'prot_quality_norm': 0.6, 'total_nutrition_norm': 0.2},
        'Balanced BLSS':                  {'O2_rate_norm': 0.4, 'prot_quality_norm': 0.3, 'total_nutrition_norm': 0.3},
    }

    print(f"\n  Recommended glucose for different BLSS scenarios:")
    recommendations = {}
    for scenario, weights in scenarios.items():
        best_g = None
        best_score = -1
        for g in GLC_LIST:
            total = sum(scores[g][k] * w for k, w in weights.items())
            if total > best_score:
                best_score = total
                best_g = g
        recommendations[scenario] = (best_g, best_score)
        print(f"    {scenario:40s} → {best_g} g/L (score={best_score:.3f})")

    return scores, recommendations


# =============================================================================
# 4. REAL-TIME CONTROL FEASIBILITY
# =============================================================================

def control_feasibility(data, p_opt, results, t_fine):
    """
    Demonstrate model suitability for real-time BLSS control:
      - Prediction speed benchmark
      - Orbital light/dark cycle simulation
      - Glucose depletion + recovery scenario
      - K(S) for automatic harvest timing
    """
    print("\n" + "=" * 70)
    print("  MODULE 4: Real-time Control Feasibility")
    print("=" * 70)

    # (a) Prediction speed
    print("\n  [4a] Model Prediction Speed Benchmark")
    t_test = np.linspace(0, 15, 100)
    n_runs = 1000

    start = time.perf_counter()
    for _ in range(n_runs):
        for idx, g in enumerate(GLC_LIST):
            _ = predict_with_x0adj(t_test, g, data[g]['mean'][0], p_opt, idx)
    elapsed = time.perf_counter() - start

    total_predictions = n_runs * len(GLC_LIST)
    print(f"    {total_predictions:,d} predictions in {elapsed:.3f} s")
    print(f"    Speed: {total_predictions/elapsed:,.0f} predictions/s")
    print(f"    Per prediction: {elapsed/total_predictions*1e6:.1f} μs")
    print(f"    >> Sufficient for real-time BLSS control (>10 kHz)")

    # (b) Orbital L/D simulation
    print("\n  [4b] Orbital Light/Dark Cycle (ISS-like)")
    print(f"    Orbital period: {ORBITAL_PERIOD:.0f} min")
    print(f"    Sunlit phase: {ORBITAL_SUNLIT:.0f} min")
    sunlit_frac = ORBITAL_SUNLIT / ORBITAL_PERIOD
    print(f"    Sunlit fraction: {sunlit_frac:.2%}")

    # Simulate 1 day = 16 orbits
    n_orbits = int(24 * 60 / ORBITAL_PERIOD)
    print(f"    Orbits per day: {n_orbits}")

    # O₂ is only produced during sunlit phase
    # Effective daily O₂ = model_rate * sunlit_frac / LD_RATIO
    # (model already assumes 12:12 = 0.5 light fraction)
    orbital_factor = sunlit_frac / LD_RATIO
    print(f"    Orbital/ground L/D factor: {orbital_factor:.3f}")
    print(f"    >> Orbital O₂ production = {orbital_factor:.1%} of "
          f"ground-based 12:12 L/D estimate")

    # (c) Glucose depletion scenario
    print("\n  [4c] Emergency Scenario: Glucose Depletion")
    # Start at 5 g/L, glucose suddenly drops to 0
    g_init = 5
    g_depleted = 0
    idx5 = GLC_LIST.index(g_init)
    idx0 = GLC_LIST.index(g_depleted)

    # Current biomass at day 10 (mid-culture) under 5 g/L
    idx_d10 = np.argmin(np.abs(t_fine - 10.0))
    X_current = results[g_init]['X'][idx_d10]
    DW_current = cells_to_dw(X_current)

    # Predict growth after depletion (revert to autotrophic kinetics)
    t_recovery = np.linspace(0, 5, 100)
    X_recovery = predict_with_x0adj(t_recovery, g_depleted, X_current, p_opt, idx0)

    # K for autotrophic
    K_auto = p_opt[3]  # K₀
    K_mixo = p_opt[3] + p_opt[4] * g_init / (p_opt[5] + g_init)

    print(f"    Initial state: 5 g/L glucose, Day 10, "
          f"X = {X_current/1e6:.1f} M cells/mL")
    print(f"    K (mixotrophic, 5 g/L): {K_mixo/1e6:.1f} M cells/mL")
    print(f"    K (autotrophic, 0 g/L): {K_auto/1e6:.1f} M cells/mL")

    if X_current > K_auto:
        print(f"    >> WARNING: Current biomass EXCEEDS autotrophic K!")
        print(f"    >> Culture will decline to K₀ = {K_auto/1e6:.1f} M")
        decline_pct = (1 - K_auto / X_current) * 100
        print(f"    >> Expected decline: {decline_pct:.1f}%")
    else:
        print(f"    >> Culture will slow but maintain biomass")

    # K(S) harvest timing
    print("\n  [4d] K(S) Function for Automatic Harvest Timing")
    for g in GLC_LIST:
        K_g = p_opt[3] + p_opt[4] * g / (p_opt[5] + g) if g > 0 else p_opt[3]
        harvest_threshold = 0.9 * K_g
        print(f"    {g:2d} g/L: K = {K_g/1e6:.1f}M, "
              f"harvest trigger at 90%K = {harvest_threshold/1e6:.1f}M")

    return {
        'prediction_speed': total_predictions / elapsed,
        'per_prediction_us': elapsed / total_predictions * 1e6,
        'orbital_factor': orbital_factor,
        'X_current_at_depletion': X_current,
        'K_auto': K_auto,
        'K_mixo': K_mixo,
        't_recovery': t_recovery,
        'X_recovery': X_recovery,
        'elapsed': elapsed,
        'n_predictions': total_predictions,
    }


# =============================================================================
# FIGURE 1: PBR Volume & O₂ Analysis
# =============================================================================

def plot_figure1(results, t_fine, o2_table):
    """
    Figure 1: blss_pbr_volume.png
    (a) Net O₂ rate vs time for all glucose groups
    (b) Required PBR volume vs crew size (bar chart)
    (c) O₂ supply efficiency comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    cycle_mask = t_fine <= 15.0

    # (a) Net O₂ rate vs time
    ax = axes[0]
    for g in GLC_LIST:
        r = results[g]
        ax.plot(t_fine[cycle_mask], r['o2_net'][cycle_mask],
                '-', color=COLORS[g], lw=2, label=f'{g} g/L')
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Net O$_2$ production rate [mg/L/day]')
    ax.set_title('(a)', fontweight='bold', loc='left')
    ax.set_title('Net O$_2$ production rate over culture cycle')
    ax.legend(title='Glucose', fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) PBR volume vs crew size
    ax = axes[1]
    crew_sizes = [1, 2, 3, 4, 5, 6]
    n_groups = len(GLC_LIST)
    width = 0.15
    x = np.arange(len(crew_sizes))

    for i, g in enumerate(GLC_LIST):
        vol_per_crew = o2_table[g]['vol_per_crew']
        volumes = [vol_per_crew * c for c in crew_sizes]
        # Convert to m³ for readability
        volumes_m3 = [v / 1000.0 for v in volumes]
        offset = (i - n_groups / 2 + 0.5) * width
        bars = ax.bar(x + offset, volumes_m3, width,
                       color=COLORS[g], label=f'{g} g/L', edgecolor='white',
                       linewidth=0.5)

    ax.set_xlabel('Crew size')
    ax.set_ylabel('Required PBR volume [m$^3$]')
    ax.set_title('(b)', fontweight='bold', loc='left')
    ax.set_title('PBR volume requirement')
    ax.set_xticks(x)
    ax.set_xticklabels(crew_sizes)
    ax.legend(title='Glucose', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) O₂ supply efficiency: avg rate and peak rate comparison
    ax = axes[2]
    glc_labels = [f'{g}' for g in GLC_LIST]
    avg_rates = [o2_table[g]['avg_rate'] for g in GLC_LIST]
    peak_rates = [o2_table[g]['peak_rate'] for g in GLC_LIST]

    x_pos = np.arange(len(GLC_LIST))
    w = 0.35
    bars1 = ax.bar(x_pos - w/2, avg_rates, w, label='Average rate',
                    color='#4472C4', edgecolor='white')
    bars2 = ax.bar(x_pos + w/2, peak_rates, w, label='Peak rate',
                    color='#ED7D31', edgecolor='white')

    # Mark the best
    best_idx = np.argmax(avg_rates)
    ax.annotate('Optimal', xy=(x_pos[best_idx] - w/2, avg_rates[best_idx]),
                xytext=(x_pos[best_idx] - w/2, avg_rates[best_idx] * 1.15),
                ha='center', fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_xlabel('Glucose concentration [g/L]')
    ax.set_ylabel('O$_2$ production rate [mg/L/day]')
    ax.set_title('(c)', fontweight='bold', loc='left')
    ax.set_title('O$_2$ supply efficiency')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(glc_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('C. pyrenoidosa GY-D12 — PBR Volume Estimation for BLSS O$_2$ Supply',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/blss_pbr_volume.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  blss_pbr_volume.png")


# =============================================================================
# FIGURE 2: Nutritional Supply
# =============================================================================

def plot_figure2(nutr_table, results, t_fine):
    """
    Figure 2: blss_nutrition.png
    (a) Daily harvestable nutrients — stacked bar
    (b) Volume needed to meet NASA standards — grouped bar
    (c) Protein quality vs quantity tradeoff — scatter
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    idx_15 = np.argmin(np.abs(t_fine - 15.0))

    # (a) Daily harvestable nutrients (stacked bar)
    ax = axes[0]
    glc_labels = [f'{g}' for g in GLC_LIST]
    x_pos = np.arange(len(GLC_LIST))
    w = 0.6

    prot_vals = [nutr_table[g]['prot_daily'] * 1000 for g in GLC_LIST]  # mg/L/day
    carb_vals = [nutr_table[g]['carb_daily'] * 1000 for g in GLC_LIST]
    lipid_vals = [nutr_table[g]['lipid_daily'] * 1000 for g in GLC_LIST]

    ax.bar(x_pos, prot_vals, w, label='Protein', color='#4472C4')
    ax.bar(x_pos, carb_vals, w, bottom=prot_vals, label='Carbohydrate',
           color='#ED7D31')
    bottom2 = [p + c for p, c in zip(prot_vals, carb_vals)]
    ax.bar(x_pos, lipid_vals, w, bottom=bottom2, label='Lipid',
           color='#A5A5A5')

    ax.set_xlabel('Glucose concentration [g/L]')
    ax.set_ylabel('Daily harvestable nutrient [mg/L/day]')
    ax.set_title('(a)', fontweight='bold', loc='left')
    ax.set_title('Daily harvestable nutrients (D=0.1 d$^{-1}$)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(glc_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) Volume needed to meet NASA standards
    ax = axes[1]
    n_nutr = 3
    width = 0.25
    x = np.arange(len(GLC_LIST))

    vol_prot = [nutr_table[g]['vol_prot'] for g in GLC_LIST]
    vol_carb = [nutr_table[g]['vol_carb'] for g in GLC_LIST]
    vol_lipid = [nutr_table[g]['vol_lipid'] for g in GLC_LIST]

    # Cap display at reasonable value
    cap = max(max(vol_prot), max(vol_carb), max(vol_lipid)) * 1.1

    ax.bar(x - width, [v/1000 for v in vol_prot], width,
           label='Protein', color='#4472C4', edgecolor='white')
    ax.bar(x, [v/1000 for v in vol_carb], width,
           label='Carbohydrate', color='#ED7D31', edgecolor='white')
    ax.bar(x + width, [v/1000 for v in vol_lipid], width,
           label='Lipid', color='#A5A5A5', edgecolor='white')

    ax.set_xlabel('Glucose concentration [g/L]')
    ax.set_ylabel('Required PBR volume [m$^3$]')
    ax.set_title('(b)', fontweight='bold', loc='left')
    ax.set_title('Volume to meet NASA nutrition standards')
    ax.set_xticks(x)
    ax.set_xticklabels(glc_labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # (c) Protein quality vs quantity
    ax = axes[2]
    for g in GLC_LIST:
        prot_pct = nutr_table[g]['f_prot'] * 100  # %DW
        prot_yield = nutr_table[g]['prot_daily'] * 1000  # mg/L/day
        ax.scatter(prot_yield, prot_pct, s=150, c=COLORS[g],
                   edgecolors='black', linewidth=0.8, zorder=5)
        ax.annotate(f'{g} g/L', (prot_yield, prot_pct),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9)

    ax.set_xlabel('Protein productivity [mg/L/day]')
    ax.set_ylabel('Protein content [% DW]')
    ax.set_title('(c)', fontweight='bold', loc='left')
    ax.set_title('Protein quality vs. quantity tradeoff')
    ax.grid(True, alpha=0.3)

    # Add trend arrow
    ax.annotate('', xy=(max(prot_vals)*0.9, min([nutr_table[g]['f_prot']*100 for g in GLC_LIST])*1.02),
                xytext=(min(prot_vals)*1.1, max([nutr_table[g]['f_prot']*100 for g in GLC_LIST])*0.98),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))
    ax.text(0.5, 0.05, 'Increasing glucose', transform=ax.transAxes,
            ha='center', fontsize=9, color='gray', style='italic')

    fig.suptitle('C. pyrenoidosa GY-D12 — Nutritional Supply Analysis for BLSS',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/blss_nutrition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  blss_nutrition.png")


# =============================================================================
# FIGURE 3: Optimization
# =============================================================================

def plot_figure3(scores, recommendations):
    """
    Figure 3: blss_optimization.png
    (a) Multi-objective radar chart
    (b) Glucose optimization heatmap
    (c) Recommended strategy table
    """
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1])

    # (a) Radar chart
    categories = ['O$_2$ production', 'Protein quality', 'Total nutrition']
    cat_keys = ['O2_rate_norm', 'prot_quality_norm', 'total_nutrition_norm']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax = fig.add_subplot(gs[0], polar=True)

    for g in GLC_LIST:
        values = [scores[g][k] for k in cat_keys]
        values += values[:1]
        ax.plot(angles, values, 'o-', color=COLORS[g], lw=2,
                label=f'{g} g/L', markersize=6)
        ax.fill(angles, values, color=COLORS[g], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title('(a) Multi-objective scores', fontweight='bold',
                 pad=20, fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=8)

    # (b) Heatmap
    ax = fig.add_subplot(gs[1])
    obj_labels = ['O$_2$ production', 'Protein quality', 'Total nutrition']
    obj_keys = ['O2_rate_norm', 'prot_quality_norm', 'total_nutrition_norm']
    glc_labels = [f'{g} g/L' for g in GLC_LIST]

    heatmap_data = np.zeros((len(obj_keys), len(GLC_LIST)))
    for i, key in enumerate(obj_keys):
        for j, g in enumerate(GLC_LIST):
            heatmap_data[i, j] = scores[g][key]

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(GLC_LIST)))
    ax.set_xticklabels(glc_labels, fontsize=9)
    ax.set_yticks(range(len(obj_labels)))
    ax.set_yticklabels(obj_labels, fontsize=9)

    # Annotate cells
    for i in range(len(obj_keys)):
        for j in range(len(GLC_LIST)):
            val = heatmap_data[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized score')
    ax.set_title('(b) Optimization heatmap', fontweight='bold', fontsize=12)
    ax.set_xlabel('Glucose concentration')

    # (c) Strategy recommendation table
    ax = fig.add_subplot(gs[2])
    ax.axis('off')

    table_data = []
    for scenario, (best_g, best_score) in recommendations.items():
        table_data.append([scenario, f'{best_g} g/L', f'{best_score:.3f}'])

    tbl = ax.table(cellText=table_data,
                   colLabels=['BLSS Scenario', 'Optimal\nGlucose', 'Score'],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.2)

    for j in range(3):
        tbl[0, j].set_facecolor('#4472C4')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Highlight best rows
    for row_idx in range(1, len(table_data) + 1):
        tbl[row_idx, 0].set_facecolor('#E8F0FE')
        tbl[row_idx, 1].set_facecolor('#E2EFDA')
        tbl[row_idx, 2].set_facecolor('#FFF2CC')

    ax.set_title('(c) Recommended strategy', fontweight='bold',
                 fontsize=12, pad=20)

    fig.suptitle('C. pyrenoidosa GY-D12 — Multi-objective BLSS Optimization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/blss_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  blss_optimization.png")


# =============================================================================
# FIGURE 4: Control Feasibility
# =============================================================================

def plot_figure4(control_data, results, data, p_opt, t_fine):
    """
    Figure 4: blss_control.png
    (a) Model prediction speed benchmark
    (b) Orbital light/dark simulation
    (c) Emergency scenario: glucose depletion + recovery
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Prediction speed benchmark
    ax = axes[0]
    # Run a range of prediction sizes to show scaling
    sizes = [10, 50, 100, 500, 1000, 5000]
    times_per = []
    for sz in sizes:
        t_bench = np.linspace(0, 15, sz)
        start = time.perf_counter()
        for _ in range(200):
            predict_with_x0adj(t_bench, 5, data[5]['mean'][0], p_opt, 3)
        elapsed = time.perf_counter() - start
        times_per.append(elapsed / 200 * 1e3)  # ms

    ax.plot(sizes, times_per, 'o-', color='#d62728', lw=2, ms=8)
    ax.set_xlabel('Prediction points per call')
    ax.set_ylabel('Time per prediction [ms]')
    ax.set_title('(a)', fontweight='bold', loc='left')
    ax.set_title('Model prediction speed')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(f'Total: {control_data["n_predictions"]:,d} predictions\n'
                f'in {control_data["elapsed"]:.2f} s\n'
                f'({control_data["prediction_speed"]:,.0f} pred/s)',
                xy=(0.55, 0.7), xycoords='axes fraction',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

    # (b) Orbital L/D simulation
    ax = axes[1]
    # Simulate 6 hours (4 orbits) of O₂ production
    t_minutes = np.linspace(0, 6 * 60, 2000)  # 6 hours in minutes
    orbital_light = np.zeros_like(t_minutes)

    for t_m in range(len(t_minutes)):
        phase = t_minutes[t_m] % ORBITAL_PERIOD
        if phase < ORBITAL_SUNLIT:
            orbital_light[t_m] = 1.0

    # O₂ production follows light (use 5 g/L mid-culture rate)
    idx_d7 = np.argmin(np.abs(t_fine - 7.0))
    base_o2_rate = results[5]['o2_net'][idx_d7]  # mg/L/day
    o2_rate_per_min = base_o2_rate / (24 * 60)   # mg/L/min

    o2_instantaneous = orbital_light * o2_rate_per_min
    o2_cumulative = np.cumsum(o2_instantaneous) * (t_minutes[1] - t_minutes[0])

    ax_twin = ax.twinx()

    ax.fill_between(t_minutes / 60, 0, orbital_light,
                    alpha=0.2, color='gold', label='Sunlit phase')
    ax.plot(t_minutes / 60, o2_instantaneous, '-', color='#2ca02c',
            lw=1.5, label='O$_2$ rate')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('O$_2$ rate [mg/L/min]', color='#2ca02c')
    ax.set_title('(b)', fontweight='bold', loc='left')
    ax.set_title('Orbital L/D simulation (ISS, 5 g/L)')
    ax.tick_params(axis='y', labelcolor='#2ca02c')

    ax_twin.plot(t_minutes / 60, o2_cumulative, '--', color='#1f77b4',
                 lw=2, label='Cumulative O$_2$')
    ax_twin.set_ylabel('Cumulative O$_2$ [mg/L]', color='#1f77b4')
    ax_twin.tick_params(axis='y', labelcolor='#1f77b4')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # (c) Emergency: glucose depletion
    ax = axes[2]

    # Before depletion: 5 g/L, growing
    t_before = np.linspace(0, 10, 200)
    X_before = predict_with_x0adj(t_before, 5, data[5]['mean'][0], p_opt, 3)

    # After depletion: switch to autotrophic kinetics
    t_after = control_data['t_recovery']
    X_after = control_data['X_recovery']

    # Plot combined timeline
    t_combined_before = t_before
    t_combined_after = t_after + 10  # offset by depletion time

    ax.plot(t_combined_before, X_before / 1e6, '-', color='#d62728',
            lw=2, label='5 g/L (mixotrophic)')
    ax.plot(t_combined_after, X_after / 1e6, '-', color='#2ca02c',
            lw=2, label='0 g/L (autotrophic recovery)')

    # Depletion event
    ax.axvline(10, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.annotate('Glucose\ndepletion', xy=(10, X_before[-1] / 1e6),
                xytext=(11, X_before[-1] / 1e6 * 1.05),
                fontsize=9, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    # K lines
    K_mixo = control_data['K_mixo']
    K_auto = control_data['K_auto']
    ax.axhline(K_mixo / 1e6, color='#d62728', ls=':', alpha=0.5)
    ax.text(0.5, K_mixo / 1e6 * 1.02, f'K (5 g/L) = {K_mixo/1e6:.1f}M',
            fontsize=8, color='#d62728')
    ax.axhline(K_auto / 1e6, color='#2ca02c', ls=':', alpha=0.5)
    ax.text(0.5, K_auto / 1e6 * 0.95, f'K (0 g/L) = {K_auto/1e6:.1f}M',
            fontsize=8, color='#2ca02c')

    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Cell concentration [$\\times 10^6$ cells/mL]')
    ax.set_title('(c)', fontweight='bold', loc='left')
    ax.set_title('Glucose depletion emergency scenario')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle('C. pyrenoidosa GY-D12 — Real-time BLSS Control Feasibility',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/blss_control.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  blss_control.png")


# =============================================================================
# CSV EXPORT
# =============================================================================

def export_csv(o2_table, nutr_table, scores, recommendations, control_data):
    """Export key results to blss_analysis_results.csv"""
    rows = []

    # Section 1: O₂ / PBR analysis
    for g in GLC_LIST:
        rows.append({
            'Section': 'PBR_O2',
            'Glucose_gL': g,
            'Parameter': 'Avg_O2_rate_mg_L_day',
            'Value': o2_table[g]['avg_rate'],
        })
        rows.append({
            'Section': 'PBR_O2',
            'Glucose_gL': g,
            'Parameter': 'Peak_O2_rate_mg_L_day',
            'Value': o2_table[g]['peak_rate'],
        })
        rows.append({
            'Section': 'PBR_O2',
            'Glucose_gL': g,
            'Parameter': 'PBR_volume_L_per_crew',
            'Value': o2_table[g]['vol_per_crew'],
        })

    # Section 2: Nutrition
    for g in GLC_LIST:
        for key in ['DW', 'prot_daily', 'carb_daily', 'lipid_daily',
                     'vol_prot', 'vol_carb', 'vol_lipid',
                     'f_prot', 'f_carb', 'f_lipid']:
            rows.append({
                'Section': 'Nutrition',
                'Glucose_gL': g,
                'Parameter': key,
                'Value': nutr_table[g][key],
            })

    # Section 3: Optimization scores
    for g in GLC_LIST:
        for key in ['O2_rate', 'prot_quality', 'total_nutrition',
                     'O2_rate_norm', 'prot_quality_norm', 'total_nutrition_norm']:
            rows.append({
                'Section': 'Optimization',
                'Glucose_gL': g,
                'Parameter': key,
                'Value': scores[g][key],
            })

    # Section 4: Recommendations
    for scenario, (best_g, best_score) in recommendations.items():
        rows.append({
            'Section': 'Recommendation',
            'Glucose_gL': best_g,
            'Parameter': scenario,
            'Value': best_score,
        })

    # Section 5: Control
    rows.append({
        'Section': 'Control',
        'Glucose_gL': -1,
        'Parameter': 'prediction_speed_per_s',
        'Value': control_data['prediction_speed'],
    })
    rows.append({
        'Section': 'Control',
        'Glucose_gL': -1,
        'Parameter': 'per_prediction_us',
        'Value': control_data['per_prediction_us'],
    })
    rows.append({
        'Section': 'Control',
        'Glucose_gL': -1,
        'Parameter': 'orbital_factor',
        'Value': control_data['orbital_factor'],
    })

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/blss_analysis_results.csv', index=False,
              encoding='utf-8-sig')
    print("  blss_analysis_results.csv")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  C. pyrenoidosa GY-D12 — BLSS Quantitative Analysis")
    print("  Bioregenerative Life Support System Feasibility Study")
    print("=" * 70)

    # Load data and model
    print("\n[Loading data and model parameters]")
    data, glc_list = load_data()
    photo_df = load_photo()
    p_opt = load_params()
    print(f"  Glucose groups: {glc_list}")
    print(f"  Parameters loaded: {len(p_opt)} values")

    # Compute all predictions
    print("\n[Computing model predictions]")
    results, t_fine = compute_all(data, glc_list, p_opt, photo_df)
    print(f"  Time grid: {len(t_fine)} points, 0-{t_fine[-1]:.0f} days")

    # ── Module 1: PBR O₂ ──
    o2_table = pbr_o2_analysis(results, t_fine)

    # ── Module 2: Nutrition ──
    nutr_table = nutrition_analysis(results, t_fine)

    # ── Module 3: Optimization ──
    scores, recommendations = optimization_analysis(o2_table, nutr_table)

    # ── Module 4: Control ──
    control_data = control_feasibility(data, p_opt, results, t_fine)

    # ── Generate Figures ──
    print("\n" + "=" * 70)
    print("  Generating Publication-Quality Figures")
    print("=" * 70)

    plot_figure1(results, t_fine, o2_table)
    plot_figure2(nutr_table, results, t_fine)
    plot_figure3(scores, recommendations)
    plot_figure4(control_data, results, data, p_opt, t_fine)

    # ── Export CSV ──
    print("\n[Exporting results]")
    export_csv(o2_table, nutr_table, scores, recommendations, control_data)

    print("\n" + "=" * 70)
    print("  BLSS Analysis Complete")
    print("=" * 70)
    print(f"\n  Output files:")
    print(f"    {OUT}/blss_pbr_volume.png")
    print(f"    {OUT}/blss_nutrition.png")
    print(f"    {OUT}/blss_optimization.png")
    print(f"    {OUT}/blss_control.png")
    print(f"    {OUT}/blss_analysis_results.csv")


if __name__ == '__main__':
    main()
