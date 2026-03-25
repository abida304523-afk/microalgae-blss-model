"""
core/data_loader.py — 实验数据加载模块
=======================================
集中管理所有实验数据的读取逻辑，
消除各脚本中重复的 load_data() / load_photo() 定义。
"""

import numpy as np
import pandas as pd
from pathlib import Path


GLC_LIST = [0, 1, 2, 5, 10]


def load_growth_data(filepath=None) -> tuple:
    """
    从 Excel 文件加载生长曲线数据。

    Returns
    -------
    (days, data) : tuple
        days: np.ndarray，时间点数组（天）
        data: dict，键为葡萄糖浓度，值含 'days','mean','std'
    """
    if filepath is None:
        from config import GROWTH_DATA_FILE
        filepath = GROWTH_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"生长曲线数据文件未找到: {filepath}\n"
            f"请将数据文件放置于 data/ 目录，或在 config.py 中修改路径。"
        )

    df = pd.read_excel(filepath, sheet_name='Sheet2', header=None)
    days = pd.to_numeric(df.iloc[3:, 0], errors='coerce').values

    data = {}
    for i, g in enumerate(GLC_LIST):
        mean = pd.to_numeric(df.iloc[3:, 26 + 2 * i], errors='coerce').values
        std  = pd.to_numeric(df.iloc[3:, 27 + 2 * i], errors='coerce').values
        data[g] = {'days': days, 'mean': mean, 'std': std}

    return days, data


def load_photo_data(filepath=None) -> pd.DataFrame:
    """
    从 Excel 文件加载光合活性数据。

    Returns
    -------
    pd.DataFrame
        包含 'group', 'Day', 'ETR' 等列的数据框。
    """
    if filepath is None:
        from config import PHOTO_DATA_FILE
        filepath = PHOTO_DATA_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"光合活性数据文件未找到: {filepath}\n"
            f"请将数据文件放置于 data/ 目录，或在 config.py 中修改路径。"
        )

    return pd.read_excel(filepath)


def load_params_v6(filepath=None) -> np.ndarray:
    """
    从 CSV 文件加载 v6 模型拟合参数。

    Returns
    -------
    np.ndarray
        参数数组 [μ₀, μ_S, K_S, K₀, K_max, S_K, f₀, f₁, f₂, f₅, f₁₀]
    """
    if filepath is None:
        from config import PARAMS_V6_FILE
        filepath = PARAMS_V6_FILE

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"参数文件未找到: {filepath}\n"
            f"请先运行 fit_v6_optimized.py 生成参数文件。"
        )

    df = pd.read_csv(filepath)
    return df['值'].values
