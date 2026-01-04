# core/ms1.py
import os
from dataclasses import dataclass

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


def load_risk_db(joblib_path: str = "data_processed/risk_db.joblib"):
    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"找不到风险库 joblib: {joblib_path}")
    return joblib.load(joblib_path)


def match_one_mz(mz_value: float, risk_mode_db: dict, threshold: float):
    """
    notebook 逻辑复刻：
      - Risk0：与 risk1_precise 任意值差 <= threshold（输出列仍显示 Risk1）
      - Risk1：两位小数命中 risk1_rounded，但差 > threshold
      - Risk2/3：两位小数命中对应集合
      - 否则 Low Risk
    """
    mz_rounded = round(float(mz_value), 2)

    risk1_precise = risk_mode_db["risk1_precise"]
    risk1_rounded = risk_mode_db["risk1_rounded"]
    r2p = risk_mode_db["risk1_rounded_to_precise"]

    # Risk0: 精确阈值命中 risk1_precise
    if risk1_precise:
        diffs = [abs(float(mz_value) - float(v)) for v in risk1_precise]
        min_diff = min(diffs)
        if min_diff <= threshold:
            closest = float(risk1_precise[int(np.argmin(diffs))])
            return {
                "Actual Risk": "Risk0",
                "Output Risk": "Risk1",
                "Matched m/z": mz_rounded,
                "Matched to m/z": closest,
                "Match Type": f"精确匹配(阈值={threshold}Da)",
                "Difference (Da)": float(min_diff),
            }

    # Risk1: 两位小数命中，但排除 Risk0 情况
    if mz_rounded in risk1_rounded:
        candidates = r2p.get(mz_rounded, [])
        if candidates:
            diffs = [abs(float(mz_value) - float(v)) for v in candidates]
            closest = float(candidates[int(np.argmin(diffs))])
            diff = float(min(diffs))
            if diff > threshold:
                return {
                    "Actual Risk": "Risk1",
                    "Output Risk": "Risk1",
                    "Matched m/z": mz_rounded,
                    "Matched to m/z": closest,
                    "Match Type": "近似匹配(两位小数相同)",
                    "Difference (Da)": diff,
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
        # 兜底：第一列当 m/z
        mz_series = df.iloc[:, 0]

    results = []
    for idx, mz in enumerate(mz_series.tolist(), start=1):
        try:
            mzv = float(mz)
        except (TypeError, ValueError):
            continue

        r = match_one_mz(mzv, mode_db, cfg.threshold)
        r_row = {
            "Index": idx,
            "Original m/z": float(mzv),
            **r,
        }
        results.append(r_row)

    out = pd.DataFrame(results)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        out.to_excel(w, sheet_name=f"All Results ({cfg.ion_mode})", index=False)

    return out


def format_risk_output(
        risk_results_xlsx: str,
        output_xlsx: str,
        sheet_name: str,
) -> pd.DataFrame:
    df = pd.read_excel(risk_results_xlsx, sheet_name=sheet_name)
    if "Actual Risk" not in df.columns:
        raise ValueError("缺少列: Actual Risk")

    def _fmt(actual: str) -> str:
        if actual == "Low Risk":
            return "Negative, Low Risk"
        if actual == "Risk3":
            return "Negative, Risk3"
        if actual in ("Risk0", "Risk1"):
            return "Risk1高风险，需要进行二级质谱筛查"
        if actual == "Risk2":
            return "Risk2高风险，需要进行二级质谱筛查"
        return str(actual)

    df["Formatted Output"] = df["Actual Risk"].astype(str).map(_fmt)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="All Matching Results", index=False)

    return df