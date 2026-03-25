"""
product_prediction.py — 产物预测模块（改进版）
===============================================

改进内容:
  1. 消除硬编码路径 → 使用 config.py
  2. 消除重复函数 → 从 core 模块导入
  3. 移除全局 warnings 抑制
  4. 添加完整的类型注解和文档字符串

基于 v6 Logistic-Monod 模型预测:
  1. O₂ 产生 (基于 ETR 光合数据)
  2. 蛋白质 / 碳水化合物 / 脂质含量
"""

import warnings
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import GLC_LIST, COLORS, CONSTANTS, OUTPUT_DIR
from core.data_loader import load_growth_data, load_photo_data, load_params_v6
from core.model import predict_with_x0adj
from core.utils import cells_to_dw, biomass_composition


# ─────────────────────────────────────────────
# O₂ 计算
# ─────────────────────────────────────────────

def compute_o2(photo_df: pd.DataFrame, data: dict,
               p_opt: np.ndarray, t_fine: np.ndarray) -> dict:
    """
    基于 ETR 数据计算各组的 O₂ 产生。

    Parameters
    ----------
    photo_df : 光合活性数据框
    data     : 生长曲线数据字典
    p_opt    : v6 模型参数
    t_fine   : 时间插值数组

    Returns
    -------
    dict : 键为葡萄糖浓度，值含 O₂ 速率和累积量
    """
    ABS = CONSTANTS['ABS_CROSS_SECTION']
    O2_MW = CONSTANTS['O2_MW']
    o2_results = {}

    for idx, g in enumerate(GLC_LIST):
        sub = photo_df[photo_df['group'] == g].groupby('Day')['ETR'].mean()
        etr_days = sub.index.values.astype(float)
        etr_vals = sub.values

        if len(etr_days) >= 2:
            etr_interp = interp1d(
                etr_days, etr_vals, kind='linear',
                bounds_error=False,
                fill_value=(etr_vals[0], etr_vals[-1])
            )
            etr_fine = etr_interp(t_fine)
        else:
            etr_fine = np.full_like(
                t_fine, etr_vals[0] if len(etr_vals) > 0 else 50.0
            )

        d = data[g]
        mask = np.isfinite(d['mean'])
        X0 = d['mean'][mask][0]
        X_fine = predict_with_x0adj(t_fine, g, X0, p_opt, idx)
        DW_fine = cells_to_dw(X_fine)

        total_abs_area = DW_fine * ABS
        o2_rate_umol   = etr_fine * total_abs_area / 4.0
        o2_gross       = o2_rate_umol * O2_MW * 1e-3 * 86400  # mg/L/day

        # 呼吸损失（经验假设，建议未来实验验证）
        resp_frac = 0.15 + 0.20 * g / (5.0 + g)
        o2_net    = o2_gross * (1.0 - resp_frac)

        dt     = np.gradient(t_fine)
        o2_cum = np.cumsum(o2_net * dt)

        o2_results[g] = {
            'etr_days': etr_days, 'etr_vals': etr_vals,
            'etr_fine': etr_fine,
            'o2_gross': o2_gross, 'o2_net': o2_net, 'o2_cum': o2_cum,
        }

    return o2_results


# ─────────────────────────────────────────────
# 综合计算
# ─────────────────────────────────────────────

def compute_all(data: dict, p_opt: np.ndarray,
                photo_df: pd.DataFrame) -> tuple:
    """
    计算所有浓度组的产物预测结果。

    Returns
    -------
    (results, t_fine) : tuple
    """
    t_fine = np.linspace(0, 16, 500)
    results = {}

    for idx, g in enumerate(GLC_LIST):
        d = data[g]
        mask = np.isfinite(d['mean'])
        X0 = d['mean'][mask][0]
        X_fine = predict_with_x0adj(t_fine, g, X0, p_opt, idx)
        DW = cells_to_dw(X_fine)

        f_prot, f_carb, f_lipid = biomass_composition(g, t_fine)

        results[g] = {
            't': t_fine, 'X': X_fine, 'DW': DW,
            'f_prot': f_prot, 'f_carb': f_carb, 'f_lipid': f_lipid,
            'prot':  DW * f_prot,
            'carb':  DW * f_carb,
            'lipid': DW * f_lipid,
        }

    o2_results = compute_o2(photo_df, data, p_opt, t_fine)
    for g in GLC_LIST:
        results[g].update(o2_results[g])

    return results, t_fine


# ─────────────────────────────────────────────
# 导出 CSV
# ─────────────────────────────────────────────

def export_csv(results: dict, t_fine: np.ndarray):
    """将预测结果导出为 CSV 文件。"""
    rows = []
    for g in GLC_LIST:
        r = results[g]
        for i, t in enumerate(t_fine):
            rows.append({
                'glucose_gL': g, 'day': t,
                'X_cells_mL': r['X'][i], 'DW_gL': r['DW'][i],
                'protein_gL': r['prot'][i], 'carb_gL': r['carb'][i],
                'lipid_gL':   r['lipid'][i],
                'f_protein':  r['f_prot'][i],
                'f_carb':     r['f_carb'][i],
                'f_lipid':    r['f_lipid'][i],
                'O2_net_mg_L_day':    r['o2_net'][i],
                'O2_cumulative_mg_L': r['o2_cum'][i],
            })
    out = OUTPUT_DIR / 'product_predictions.csv'
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print(f"  CSV 已保存: {out}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print(" 产物预测模块")
    print("=" * 60)

    _, data    = load_growth_data()
    photo_df   = load_photo_data()
    p_opt      = load_params_v6()

    results, t_fine = compute_all(data, p_opt, photo_df)

    idx_15 = np.argmin(np.abs(t_fine - 15.0))
    print(f"\n Day 15 产物预测:")
    print(f"  {'葡萄糖':>6s}  {'干重':>8s}  {'蛋白质':>8s}  {'碳水':>8s}  "
          f"{'脂质':>8s}  {'累积O₂':>10s}")
    for g in GLC_LIST:
        r = results[g]
        print(f"  {g:6d}  {r['DW'][idx_15]:8.3f}  {r['prot'][idx_15]:8.4f}  "
              f"{r['carb'][idx_15]:8.4f}  {r['lipid'][idx_15]:8.4f}  "
              f"{r['o2_cum'][idx_15]:10.1f}")

    export_csv(results, t_fine)
    print("\n完成！")
