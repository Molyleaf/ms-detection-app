# core/features.py
import numpy as np


NONAFI_CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
    135.0441, 147.0077, 151.0866, 166.0975, 169.076,
    197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091,
    327.1274, 341.1608, 354.2, 377.1, 396.203
]

PEAK_GROUPS = {
    "low_mass": [58.0651, 72.0808, 84.0808, 99.0917, 113.1073],
    "middle_mass": [135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709],
    "high_mass": [250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812],
    "very_high_mass": [312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203],
}
KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]


def parse_peaks(peaks: str):
    """
    peaks: "mz:int,mz:int,..."
    返回 list[(mz, intensity)]
    """
    if not peaks:
        return []
    items = peaks.replace(";", ",").split(",")
    out = []
    for it in items:
        it = it.strip()
        if not it:
            continue
        if ":" in it:
            a, b = it.split(":", 1)
            try:
                out.append((float(a.strip()), float(b.strip())))
            except ValueError:
                continue
        else:
            try:
                out.append((float(it), 1.0))
            except ValueError:
                continue
    return out


def build_graph_inputs(
        peaks: str,
        stats: dict,
        max_nodes: int = 10,
        node_dim: int = 10,
):
    """
    输出：
      nodes: (1, 10, 10) float32
      adj:   (1, 10, 10) float32
    """
    peak_data = parse_peaks(peaks)
    # 选强度最高的前 max_nodes
    peak_data.sort(key=lambda x: x[1], reverse=True)
    peak_data = peak_data[:max_nodes]

    mz_values = [p[0] for p in peak_data]
    max_intensity_mz = mz_values[0] if mz_values else 0.0

    mz_mean = float(stats.get("mz_mean", 0.0))
    mz_std = float(stats.get("mz_std", 1.0)) or 1.0
    mx_mean = float(stats.get("max_intensity_mz_mean", 0.0))
    mx_std = float(stats.get("max_intensity_mz_std", 1.0)) or 1.0

    rounded_characteristic = {round(p, 1) for p in NONAFI_CHARACTERISTIC_PEAKS}
    rounded_key = {round(p, 1) for p in KEY_PEAKS}
    group_map = {}
    for name, arr in PEAK_GROUPS.items():
        for p in arr:
            group_map[round(p, 1)] = name

    def _mass_region_feature(rmz: float) -> float:
        g = group_map.get(rmz)
        if g == "low_mass":
            return 0.25
        if g == "middle_mass":
            return 0.5
        if g == "high_mass":
            return 0.75
        if g == "very_high_mass":
            return 1.0
        return 0.0

    nodes = np.zeros((max_nodes, node_dim), dtype=np.float32)

    for j in range(max_nodes):
        if j < len(peak_data):
            mz = float(peak_data[j][0])
        elif peak_data:
            mz = float(peak_data[-1][0])
        else:
            mz = 0.0

        total_peaks = max(len(peak_data), 1)
        mz_norm = (mz - mz_mean) / mz_std
        pos_ratio = j / total_peaks
        is_first = 1.0 if j == 0 else 0.0
        is_last = 1.0 if j == len(peak_data) - 1 else 0.0

        rmz = round(mz, 1)
        is_char = 1.0 if rmz in rounded_characteristic else 0.0
        min_diff = min((abs(mz - p) for p in NONAFI_CHARACTERISTIC_PEAKS), default=100.0)
        char_diff = (min_diff / 100.0)

        mass_region = _mass_region_feature(rmz)
        is_key = 1.0 if rmz in rounded_key else 0.0

        if max_intensity_mz > 0:
            max_mz_norm = (max_intensity_mz - mx_mean) / mx_std
            rel_to_max = mz / max_intensity_mz
        else:
            max_mz_norm = 0.0
            rel_to_max = 1.0

        feats = [
            mz_norm,
            pos_ratio,
            is_first,
            is_last,
            is_char,
            char_diff,
            mass_region,
            is_key,
            max_mz_norm,
            rel_to_max,
        ]
        nodes[j, :node_dim] = np.asarray(feats[:node_dim], dtype=np.float32)

    # 邻接矩阵：exp(-diff^2/(2*sigma^2))
    adj = np.eye(max_nodes, dtype=np.float32)
    sigma = 50.0
    for i in range(min(len(mz_values), max_nodes)):
        for j in range(min(len(mz_values), max_nodes)):
            if i == j:
                continue
            diff = abs(float(mz_values[i]) - float(mz_values[j]))
            adj[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2)).astype(np.float32)

    return nodes[None, ...], adj[None, ...]


def characteristic_rule_trigger(peaks: str) -> bool:
    """
    notebook 规则：
      - 两位小数匹配 >=3  或
      - 三位小数匹配 >=2
    """
    mzs = [mz for mz, _ in parse_peaks(peaks)]
    if not mzs:
        return False

    two = 0
    three = 0
    for mz in mzs:
        for cp in NONAFI_CHARACTERISTIC_PEAKS:
            if round(mz, 2) == round(cp, 2):
                two += 1
                break
            if round(mz, 3) == round(cp, 3):
                three += 1
                break
    return (two >= 3) or (three >= 2)