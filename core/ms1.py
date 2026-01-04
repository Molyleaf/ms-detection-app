# core/ms1.py
import os
from dataclasses import dataclass
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MS1Config:
    mass_tolerance: float = 2.0
    min_intensity: float = 1.0


def _find_mass_intensity_cols(df: pd.DataFrame):
    cols = [str(c) for c in df.columns]
    mass_candidates = ["Mass", "mass", "m/z", "M/Z", "mz", "MASS"]
    int_candidates = ["Intensity", "intensity", "Int", "int", "INTENSITY", "Abundance"]

    mass_col = None
    int_col = None

    for c in df.columns:
        s = str(c)
        if any(k in s for k in mass_candidates):
            mass_col = c
            break
    for c in df.columns:
        s = str(c)
        if any(k in s for k in int_candidates):
            int_col = c
            break

    if mass_col is None or int_col is None:
        if len(df.columns) < 2:
            raise ValueError("Excel 至少需要两列（Mass 与 Intensity）")
        mass_col = df.columns[0]
        int_col = df.columns[1]

    return mass_col, int_col


def normalize_intensity_0_100(df: pd.DataFrame) -> pd.DataFrame:
    x = df["Intensity"].to_numpy(dtype=np.float64)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx == mn:
        df["Intensity"] = 0.0
        return df
    df["Intensity"] = 100.0 * (x - mn) / (mx - mn)
    return df


def remove_isotope_peaks_keep_strongest(df: pd.DataFrame, mass_tolerance: float) -> pd.DataFrame:
    """
    逻辑：在 ±mass_tolerance 范围内，只保留强度最大的峰（同位素/近邻峰合并）。
    为了稳定：先按强度降序遍历，删除邻域内更弱者。
    """
    if df.empty:
        return df

    df2 = df.sort_values("Intensity", ascending=False).reset_index(drop=True)
    mz = df2["Mass"].to_numpy(dtype=np.float64)
    it = df2["Intensity"].to_numpy(dtype=np.float64)

    keep = np.ones(len(df2), dtype=bool)
    for i in range(len(df2)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(df2)):
            if not keep[j]:
                continue
            if abs(mz[j] - mz[i]) <= mass_tolerance:
                # i 一定不弱于 j（因为按强度降序），所以删 j
                keep[j] = False

    out = df2.loc[keep].copy()
    out = out.sort_values("Mass").reset_index(drop=True)
    return out


def process_l1_excel(
        input_xlsx: str,
        output_xlsx: str,
        cfg: MS1Config = MS1Config(),
) -> pd.DataFrame:
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"找不到输入文件: {input_xlsx}")

    raw = pd.read_excel(input_xlsx, sheet_name=0)
    mcol, icol = _find_mass_intensity_cols(raw)

    df = raw.rename(columns={mcol: "Mass", icol: "Intensity"})[["Mass", "Intensity"]].copy()
    df["Mass"] = pd.to_numeric(df["Mass"], errors="coerce")
    df["Intensity"] = pd.to_numeric(df["Intensity"], errors="coerce")
    df = df.dropna().copy()

    # 1) 归一化 0~100
    df = normalize_intensity_0_100(df)

    # 2) 删除 0 强度
    df = df[df["Intensity"] > 0].copy()

    # 3) 2Da 同位素清理
    df = remove_isotope_peaks_keep_strongest(df, cfg.mass_tolerance)

    # 4) 低强度去除（>=1）
    df = df[df["Intensity"] >= cfg.min_intensity].copy()
    df = df.sort_values("Mass").reset_index(drop=True)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Filtered Data", index=False)

    return df


# -----------------------------
# 风险匹配（Risk0/Risk1/Risk2/Risk3/Low Risk）
# -----------------------------
@dataclass(frozen=True)
class RiskConfig:
    threshold: float = 0.005
    ion_mode: str = "positive"  # "positive" or "negative"


@lru_cache(maxsize=8)
def load_risk_db(joblib_path: str = "data_processed/risk_db.joblib"):
    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"找不到风险库 joblib: {joblib_path}")
    db = joblib.load(joblib_path)

    # 预编译：把需要频繁查询的结构变成“排序数组”，后续用二分找最近邻
    for mode in ("positive", "negative"):
        mode_db = db.get(mode, {})
        r1 = mode_db.get("risk1_precise", []) or []
        r1_arr = np.asarray(r1, dtype=np.float64)
        r1_arr.sort()
        mode_db["risk1_precise_sorted"] = r1_arr

        # rounded->precise 的候选也转成排序数组（候选一般不大，但做了更稳）
        r2p = mode_db.get("risk1_rounded_to_precise", {}) or {}
        r2p_arr = {}
        for k, arr in r2p.items():
            a = np.asarray(arr, dtype=np.float64)
            a.sort()
            r2p_arr[float(k)] = a
        mode_db["risk1_rounded_to_precise_sorted"] = r2p_arr

    return db


def _nearest_in_sorted(arr: np.ndarray, x: float) -> tuple[float | None, float]:
    """
    在已排序数组 arr 中找到离 x 最近的值，返回 (closest_value, abs_diff)。
    arr 为空时返回 (None, +inf)
    """
    if arr is None or arr.size == 0:
        return None, float("inf")

    j = int(np.searchsorted(arr, x))
    best_v = None
    best_d = float("inf")

    if 0 <= j < arr.size:
        d = abs(float(arr[j]) - x)
        best_v, best_d = float(arr[j]), float(d)

    if 0 <= j - 1 < arr.size:
        d = abs(float(arr[j - 1]) - x)
        if d < best_d:
            best_v, best_d = float(arr[j - 1]), float(d)

    return best_v, best_d


def match_one_mz(mz_value: float, risk_mode_db: dict, threshold: float):
    """
    notebook 逻辑复刻：
      - Risk0：与 risk1_precise 任意值差 <= threshold（输出列仍显示 Risk1）
      - Risk1：两位小数命中 risk1_rounded，但差 > threshold
      - Risk2/3：两位小数命中对应集合
      - 否则 Low Risk
    """
    mzv = float(mz_value)
    mz_rounded = round(mzv, 2)

    risk1_rounded = risk_mode_db["risk1_rounded"]

    # Risk0：改为“二分最近邻”，不再全表 diffs 扫描
    r1_sorted = risk_mode_db.get("risk1_precise_sorted", None)
    closest, min_diff = _nearest_in_sorted(r1_sorted, mzv)
    if min_diff <= threshold and closest is not None:
        return {
            "Actual Risk": "Risk0",
            "Output Risk": "Risk1",
            "Matched m/z": mz_rounded,
            "Matched to m/z": float(closest),
            "Match Type": f"精确匹配(阈值={threshold}Da)",
            "Difference (Da)": float(min_diff),
        }

    # Risk1：两位小数命中，但排除 Risk0 情况（diff > threshold）
    if mz_rounded in risk1_rounded:
        r2p_sorted = risk_mode_db.get("risk1_rounded_to_precise_sorted", {})
        candidates = r2p_sorted.get(float(mz_rounded), None)
        closest2, diff2 = _nearest_in_sorted(candidates, mzv)
        if closest2 is not None and diff2 > threshold:
            return {
                "Actual Risk": "Risk1",
                "Output Risk": "Risk1",
                "Matched m/z": mz_rounded,
                "Matched to m/z": float(closest2),
                "Match Type": "近似匹配(两位小数相同)",
                "Difference (Da)": float(diff2),
            }

    if mz_rounded in risk_mode_db["risk2"]:
        return {
            "Actual Risk": "Risk2",
            "Output Risk": "Risk2",
            "Matched m/z": mz_rounded,
            "Matched to m/z": mz_rounded,
            "Match Type": "两位小数匹配",
            "Difference (Da)": 0.0,
        }

    if mz_rounded in risk_mode_db["risk3"]:
        return {
            "Actual Risk": "Risk3",
            "Output Risk": "Risk3",
            "Matched m/z": mz_rounded,
            "Matched to m/z": mz_rounded,
            "Match Type": "两位小数匹配",
            "Difference (Da)": 0.0,
        }

    return {
        "Actual Risk": "Low Risk",
        "Output Risk": "Low Risk",
        "Matched m/z": mz_rounded,
        "Matched to m/z": None,
        "Match Type": "无匹配",
        "Difference (Da)": None,
    }


def risk_match_l1(
        processed_l1_xlsx: str,
        output_xlsx: str,
        cfg: RiskConfig = RiskConfig(),
        risk_db_joblib: str = "data_processed/risk_db.joblib",
) -> pd.DataFrame:
    risk_db = load_risk_db(risk_db_joblib)
    mode_db = risk_db[cfg.ion_mode]

    df = pd.read_excel(processed_l1_xlsx, sheet_name=0)
    if "Mass" in df.columns:
        mz_series = df["Mass"]
    elif "Original m/z" in df.columns:
        mz_series = df["Original m/z"]
    else:
        mz_series = df.iloc[:, 0]

    # 用“列式收集”比 append dict list 更快
    idx_list = []
    omz_list = []
    actual_list = []
    output_list = []
    matched_mz_list = []
    matched_to_list = []
    match_type_list = []
    diff_list = []

    for idx, mz in enumerate(mz_series, start=1):
        try:
            mzv = float(mz)
        except (TypeError, ValueError):
            continue

        r = match_one_mz(mzv, mode_db, cfg.threshold)

        idx_list.append(idx)
        omz_list.append(float(mzv))
        actual_list.append(r["Actual Risk"])
        output_list.append(r["Output Risk"])
        matched_mz_list.append(r["Matched m/z"])
        matched_to_list.append(r["Matched to m/z"])
        match_type_list.append(r["Match Type"])
        diff_list.append(r["Difference (Da)"])

    out = pd.DataFrame(
        {
            "Index": idx_list,
            "Original m/z": omz_list,
            "Actual Risk": actual_list,
            "Output Risk": output_list,
            "Matched m/z": matched_mz_list,
            "Matched to m/z": matched_to_list,
            "Match Type": match_type_list,
            "Difference (Da)": diff_list,
        }
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name=f"All Results ({cfg.ion_mode})", index=False)

    return out