# core/similarity.py
import os
import heapq
from functools import lru_cache

import joblib
import numpy as np


@lru_cache(maxsize=4)
def _load_spectrum_db(spectrum_db_joblib: str):
    if not os.path.exists(spectrum_db_joblib):
        raise FileNotFoundError(f"找不到谱库 joblib: {spectrum_db_joblib}")

    # 尽量用 mmap_mode 降低内存峰值（不支持就回退）
    try:
        lib = joblib.load(spectrum_db_joblib, mmap_mode="r")
    except Exception:
        lib = joblib.load(spectrum_db_joblib)

    # 防御性处理：确保每条 entry["mz"] 是 numpy array 且为升序
    for e in lib:
        mz = e.get("mz", None)
        if mz is None:
            e["mz"] = np.asarray([], dtype=np.float32)
            continue
        arr = np.asarray(mz, dtype=np.float32)
        if arr.size >= 2 and np.any(arr[1:] < arr[:-1]):
            arr = np.sort(arr)
        e["mz"] = arr
    return lib


def _match_count_similarity(test_mz: np.ndarray, lib_mz_sorted: np.ndarray, tol: float) -> int:
    """
    计算匹配峰数量（每个 test 峰只要在 lib 中找到一个 tol 内的峰就算匹配）
    注意：lib_mz_sorted 必须是升序数组（这样 searchsorted 才正确）
    """
    if test_mz.size == 0 or lib_mz_sorted.size == 0:
        return 0

    cnt = 0
    for mz in test_mz:
        j = int(np.searchsorted(lib_mz_sorted, mz))
        if 0 <= j < lib_mz_sorted.size and abs(float(mz) - float(lib_mz_sorted[j])) <= tol:
            cnt += 1
            continue
        if 0 <= j - 1 < lib_mz_sorted.size and abs(float(mz) - float(lib_mz_sorted[j - 1])) <= tol:
            cnt += 1
            continue
    return cnt


def topk_library_matches(
        peaks: str,
        spectrum_db_joblib: str = "data_processed/spectrum_db.joblib",
        tol: float = 0.2,
        top_k: int = 10,
):
    """
    性能修复：
      - 不再 results 全量 append + sort（库大时慢且易 OOM）
      - 用 heap 只保留 top_k
    """
    # 结构匹配：至少返回 5 个候选（即使相似度很低也照样列出）
    top_k = max(int(top_k), 5)

    lib = _load_spectrum_db(spectrum_db_joblib)


    # 解析 test peaks（只取 mz）
    mzs = []
    for it in peaks.replace(";", ",").split(","):
        it = it.strip()
        if not it:
            continue
        if ":" in it:
            a = it.split(":", 1)[0].strip()
            try:
                mzs.append(float(a))
            except ValueError:
                continue
        else:
            # 兼容偶发没有强度的输入
            try:
                mzs.append(float(it))
            except ValueError:
                continue

    test_mz = np.asarray(sorted(mzs), dtype=np.float32)
    denom = int(test_mz.size)

    # (score, tie_breaker, payload)
    heap: list[tuple[float, int, dict]] = []

    for i, entry in enumerate(lib):
        lib_mz = entry["mz"]
        m = _match_count_similarity(test_mz, lib_mz, tol=tol)
        score = (m / denom) if denom else 0.0

        payload = {
            "smiles": entry.get("smiles", "N/A"),
            "similarity": float(score),
            "matches": int(m),
            "test_peaks": int(denom),
            "lib_peaks": int(lib_mz.size),
        }

        item = (float(score), int(i), payload)
        if len(heap) < int(top_k):
            heapq.heappush(heap, item)
        else:
            if item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)

    heap.sort(key=lambda x: x[0], reverse=True)
    return [x[2] for x in heap]
