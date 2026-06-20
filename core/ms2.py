# core/ms2.py
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MS2Config:
    mass_tolerance: float = 2.0
    intensity_digits: int = 2


def _find_mass_intensity_cols(df: pd.DataFrame):
    mass_col = None
    int_col = None
    for col in df.columns:
        s = str(col).lower()
        if mass_col is None and ("mass" in s or "m/z" in s or "mz" in s):
            mass_col = col
        if int_col is None and ("intensity" in s or s == "int" or "abundance" in s):
            int_col = col
    if mass_col is None or int_col is None:
        if len(df.columns) < 2:
            raise ValueError("L2 至少需要两列（Mass 与 Intensity）")
        mass_col, int_col = df.columns[0], df.columns[1]
    return mass_col, int_col


def normalize_intensity_0_100(df: pd.DataFrame) -> pd.DataFrame:
    """
    将强度归一化到 0~100 范围（以最大强度为 100%）。
    避免使用 min-max 归一化导致最小峰强度被强行归零并被丢弃的问题。
    """
    x = df["Intensity"].to_numpy(dtype=np.float64)
    mx = float(np.max(x))
    if mx == 0:
        df["Intensity"] = 0.0
        return df
    df["Intensity"] = 100.0 * x / mx
    return df


def remove_isotope_peaks_keep_strongest(df: pd.DataFrame, mass_tolerance: float) -> pd.DataFrame:
    """
    同位素/近邻峰合并：±tolerance 内只保留强度最高峰。
    性能修复：由 O(n^2) -> 近似 O(n log n)
    """
    if df.empty:
        return df

    df2 = df.reset_index(drop=True).copy()
    mz = df2["Mass"].to_numpy(dtype=np.float64)
    it = df2["Intensity"].to_numpy(dtype=np.float64)
    n = int(mz.size)
    if n <= 1:
        return df2

    order_int = np.argsort(-it)
    order_mass = np.argsort(mz)
    mz_sorted = mz[order_mass]

    inv_mass = np.empty(n, dtype=np.int64)
    inv_mass[order_mass] = np.arange(n, dtype=np.int64)

    active = np.ones(n, dtype=bool)
    keep_orig = np.zeros(n, dtype=bool)

    tol = float(mass_tolerance)
    for idx in order_int:
        idx = int(idx)
        pos = int(inv_mass[idx])
        if not active[pos]:
            continue

        keep_orig[idx] = True
        left = int(np.searchsorted(mz_sorted, mz[idx] - tol, side="left"))
        right = int(np.searchsorted(mz_sorted, mz[idx] + tol, side="right"))
        active[left:right] = False

    out = df2.iloc[np.flatnonzero(keep_orig)].copy()
    return out.sort_values("Mass").reset_index(drop=True)


def to_peaks_string(df: pd.DataFrame, intensity_digits: int) -> str:
    if df.empty:
        return ""
    df2 = df.sort_values("Mass").reset_index(drop=True)
    mz = df2["Mass"].to_numpy(dtype=np.float64)
    it = df2["Intensity"].to_numpy(dtype=np.float64)
    fmt = f"{{:.4f}}:{{:.{intensity_digits}f}}"
    return ",".join([fmt.format(float(m), float(i)) for m, i in zip(mz, it, strict=False)])


def process_l2_excel_to_peaks(
        input_xlsx: str,
        output_xlsx: str | None = None,
        cfg: MS2Config = MS2Config(),
) -> str:
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"找不到 L2 输入: {input_xlsx}")

    raw = pd.read_excel(input_xlsx, sheet_name=0, engine="openpyxl")
    
    # 自动探测无表头的情况：如果前两列的列名可以直接转换为数值，说明首行其实是数据行而不是表头
    is_headerless = False
    if len(raw.columns) >= 2:
        try:
            float(str(raw.columns[0]).strip())
            float(str(raw.columns[1]).strip())
            is_headerless = True
        except ValueError:
            pass
            
    if is_headerless:
        raw = pd.read_excel(input_xlsx, sheet_name=0, header=None, engine="openpyxl")

    mcol, icol = _find_mass_intensity_cols(raw)

    df = raw.rename(columns={mcol: "Mass", icol: "Intensity"})[["Mass", "Intensity"]].copy()
    df["Mass"] = pd.to_numeric(df["Mass"], errors="coerce")
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df = df.dropna().copy()

    df = normalize_intensity_0_100(df)
    df = df[df["Intensity"] > 0].copy()
    # 恢复二级质谱的同位素合并，以保持与 qlc-0103.ipynb 原始定义 100% 一致，防特征分布偏移
    df = remove_isotope_peaks_keep_strongest(df, cfg.mass_tolerance)

    peaks = to_peaks_string(df, cfg.intensity_digits)

    # 可选落盘（Web 默认不写）
    if output_xlsx:
        out = pd.DataFrame({"peaks": [peaks]})
        out.to_excel(output_xlsx, sheet_name="Formatted Output", index=False)

    return peaks
