# core/similarity.py
import os
import joblib
import numpy as np


def _match_count_similarity(test_mz: np.ndarray, lib_mz: np.ndarray, tol: float) -> int:
    """
    计算匹配峰数量（每个 test 峰只要在 lib 中找到一个 tol 内的峰就算匹配）
    简单但稳定，足够复刻 notebook 的“匹配峰数量”思路。
    """
    if len(test_mz) == 0 or len(lib_mz) == 0:
        return 0
    lib_mz_sorted = np.sort(lib_mz)
    cnt = 0
    for mz in test_mz:
        # 二分找最近邻
        j = int(np.searchsorted(lib_mz_sorted, mz))
        candidates = []
        if 0 <= j < len(lib_mz_sorted):
            candidates.append(lib_mz_sorted[j])
        if 0 <= j - 1 < len(lib_mz_sorted):
            candidates.append(lib_mz_sorted[j - 1])
        if any(abs(mz - c) <= tol for c in candidates):
            cnt += 1
    return cnt


def topk_library_matches(
        peaks: str,
        spectrum_db_joblib: str = "data_processed/spectrum_db.joblib",
        tol: float = 0.2,
        top_k: int = 10,
):
    if not os.path.exists(spectrum_db_joblib):
        raise FileNotFoundError(f"找不到谱库 joblib: {spectrum_db_joblib}")

    lib = joblib.load(spectrum_db_joblib)

    # 解析 test peaks（只取 mz）
    mzs = []
    for it in peaks.replace(";", ",").split(","):
        it = it.strip()
        if ":" in it:
            a = it.split(":", 1)[0].strip()
            try:
                mzs.append(float(a))
            except ValueError:
                continue
    test_mz = np.asarray(sorted(mzs), dtype=np.float32)

    results = []
    for entry in lib:
        m = _match_count_similarity(test_mz, entry["mz"], tol=tol)
        score = (m / len(test_mz)) if len(test_mz) else 0.0
        results.append(
            {
                "smiles": entry.get("smiles", "N/A"),
                "similarity": float(score),
                "matches": int(m),
                "test_peaks": int(len(test_mz)),
                "lib_peaks": int(len(entry["mz"])),
            }
        )
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]