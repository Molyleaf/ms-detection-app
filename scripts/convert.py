# scripts/convert_assets.py
import os
from collections import defaultdict
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    base_dir: str
    data_dir: str
    out_dir: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_ku_line(line: str):
    """
    ku.txt: 每行形如  <smiles>\t<mz:intensity,mz:intensity,...>
    返回 (smiles, mz_array, intensity_array_norm_0_100)
    """
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return None

    smiles = parts[0].strip()
    peaks_raw = parts[1].strip().replace(";", ",")
    items = [p for p in peaks_raw.split(",") if ":" in p]

    mzs = []
    ints = []
    for it in items:
        a, b = it.split(":", 1)
        try:
            mzs.append(float(a.strip()))
            ints.append(float(b.strip()))
        except ValueError:
            continue

    if not mzs:
        return None

    mz_arr = np.asarray(mzs, dtype=np.float32)
    int_arr = np.asarray(ints, dtype=np.float32)

    # 归一化到 0~100（保持与 notebook 的习惯一致）
    mx = float(np.max(int_arr)) if len(int_arr) else 0.0
    if mx > 0:
        int_arr = (int_arr / mx) * 100.0

    # 按 m/z 升序（方便后续快速匹配）
    order = np.argsort(mz_arr)
    return smiles, mz_arr[order], int_arr[order]


def build_all_assets(
        data_dir: str = "data",
        out_dir: str = "data_processed",
        risk_xlsx: str = "risk_matching-1.xlsx",
        ku_txt: str = "ku.txt",
) -> None:
    """
    ✅ 一个入口完成全部转换：
      1) 风险库 joblib
      2) 谱库 joblib
      3) 统计量 joblib
    """

    paths = Paths(
        base_dir="../data",
        data_dir="../data",
        out_dir="../data_processed"
    )
    _ensure_dir(paths.out_dir)

    # -------------------------
    # 1) 风险库
    # -------------------------
    risk_path = os.path.join(paths.data_dir, risk_xlsx)
    if not os.path.exists(risk_path):
        raise FileNotFoundError(f"风险库文件不存在: {risk_path}")

    xls = pd.ExcelFile(risk_path)

    # 只认 notebook 中的离子列
    pos_cols = ["[M+H]+", "[M+Na]+", "[M+K]+"]
    neg_cols = ["[M-H]-"]

    # 风险表单映射：风险0 在 notebook 实际是 “risk1 精确阈值命中” 的一种输出逻辑
    sheet_map = {
        "风险0": "risk1",
        "风险1": "risk1",
        "风险2": "risk2",
        "风险3": "risk3",
    }

    def _empty_mode_db():
        return {
            "risk1_precise": [],
            "risk1_rounded": set(),
            "risk1_rounded_to_precise": defaultdict(list),
            "risk2": set(),
            "risk3": set(),
        }

    risk_db = {"positive": _empty_mode_db(), "negative": _empty_mode_db()}

    for sheet_name in xls.sheet_names:
        mapped = sheet_map.get(sheet_name)
        if mapped is None:
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)

        for mode, cols in (("positive", pos_cols), ("negative", neg_cols)):
            for col in cols:
                if col not in df.columns:
                    continue

                values = df[col].dropna().tolist()
                for v in values:
                    try:
                        mz = float(v)
                    except (TypeError, ValueError):
                        continue

                    if mapped == "risk1":
                        r2 = round(mz, 2)
                        risk_db[mode]["risk1_precise"].append(mz)
                        risk_db[mode]["risk1_rounded"].add(r2)
                        risk_db[mode]["risk1_rounded_to_precise"][r2].append(mz)
                    elif mapped in ("risk2", "risk3"):
                        risk_db[mode][mapped].add(round(mz, 2))

    # defaultdict -> dict（便于序列化与调试）
    for mode in ("positive", "negative"):
        risk_db[mode]["risk1_rounded_to_precise"] = dict(risk_db[mode]["risk1_rounded_to_precise"])

    risk_out = os.path.join(paths.out_dir, "risk_db.joblib")
    joblib.dump(risk_db, risk_out)

    # -------------------------
    # 2) 谱库
    # -------------------------
    ku_path = os.path.join(paths.data_dir, ku_txt)
    if not os.path.exists(ku_path):
        raise FileNotFoundError(f"谱库文件不存在: {ku_path}")

    library = []
    with open(ku_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parsed = _parse_ku_line(line)
            if parsed is None:
                continue
            smiles, mz_arr, int_arr = parsed
            library.append({"smiles": smiles, "mz": mz_arr, "intensities": int_arr})

    spec_out = os.path.join(paths.out_dir, "spectrum_db.joblib")
    joblib.dump(library, spec_out)

    # -------------------------
    # 3) 统计量（用于 MS2 特征）
    # -------------------------
    all_mz = []
    max_intensity_mz = []
    for entry in library:
        mz = entry["mz"]
        it = entry["intensities"]
        if len(mz) == 0:
            continue
        all_mz.append(mz.astype(np.float64))
        if len(it) > 0:
            max_intensity_mz.append(float(mz[int(np.argmax(it))]))

    if not all_mz:
        raise RuntimeError("谱库为空，无法计算 stats.joblib")

    all_mz_concat = np.concatenate(all_mz, axis=0)
    mz_mean = float(np.mean(all_mz_concat))
    mz_std = float(np.std(all_mz_concat)) or 1.0

    if max_intensity_mz:
        mx_mean = float(np.mean(max_intensity_mz))
        mx_std = float(np.std(max_intensity_mz)) or 1.0
    else:
        mx_mean, mx_std = 0.0, 1.0

    stats = {
        "mz_mean": mz_mean,
        "mz_std": mz_std,
        "max_intensity_mz_mean": mx_mean,
        "max_intensity_mz_std": mx_std,
    }

    stats_out = os.path.join(paths.out_dir, "stats.joblib")
    joblib.dump(stats, stats_out)

    print("✅ 转换完成：")
    print(f"  - {risk_out}")
    print(f"  - {spec_out}")
    print(f"  - {stats_out}")
    print(f"  风险库 risk1_precise(positive) 条目数: {len(risk_db['positive']['risk1_precise'])}")
    print(f"  谱库条目数: {len(library)}")


if __name__ == "__main__":
    build_all_assets()