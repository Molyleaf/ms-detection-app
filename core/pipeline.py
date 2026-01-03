# core/pipeline.py
import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# 核心常量 (严格对齐 训练.ipynb)
CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073, 135.0441, 147.0077, 151.0866,
    166.0975, 169.076, 197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091, 327.1274, 341.1608,
    354.2, 377.1, 396.203
]
KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
PEAK_GROUPS = {
    'low_mass': ([58.0651, 72.0808, 84.0808, 99.0917, 113.1073], 0.25),
    'middle_mass': ([135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709], 0.5),
    'high_mass': ([250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812], 0.75),
    'very_high_mass': ([312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203], 1.0)
}

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        # 键名必须与 convert.py 生成的完全一致
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {'mz_mean': 0, 'mz_std': 1, 'max_intensity_mz_mean': 0, 'max_intensity_mz_std': 1}

    def _extract_single(self, ms_str):
        peaks = []
        for p in str(ms_str).replace(';', ',').split(','):
            try:
                parts = p.split(':')
                m, i = float(parts[0]), float(parts[1]) if len(parts) > 1 else 1.0
                peaks.append((m, i))
            except: continue

        if not peaks:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

        # 1. 排序与截断 (按强度降序)
        peaks.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peaks[0][0]
        top_peaks = peaks[:self.max_nodes]

        # 2. 填充 (使用最后一个峰填充，使总数固定为 max_nodes)
        while len(top_peaks) < self.max_nodes:
            top_peaks.append(top_peaks[-1])

        node_features = np.zeros((self.max_nodes, self.node_dim))
        mz_values = [p[0] for p in top_peaks]

        for j in range(self.max_nodes):
            mz = top_peaks[j][0]
            rmz = round(mz, 1) # 匹配精度：1位小数

            # 特征向量 F0-F9 严格对齐
            node_features[j, 0] = (mz - self.stats['mz_mean']) / (self.stats['mz_std'] + 1e-6)
            node_features[j, 1] = j / self.max_nodes # position_ratio
            node_features[j, 2] = 1.0 if j == 0 else 0.0 # is_first_peak
            node_features[j, 3] = 1.0 if j == self.max_nodes - 1 else 0.0 # is_last_peak
            node_features[j, 4] = 1.0 if any(round(cp, 1) == rmz for cp in CHARACTERISTIC_PEAKS) else 0.0
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0

            region_val = 0.0
            for _, (gp, val) in PEAK_GROUPS.items():
                if any(round(p, 1) == rmz for p in gp): region_val = val; break
            node_features[j, 6] = region_val

            node_features[j, 7] = 1.0 if any(round(kp, 1) == rmz for kp in KEY_PEAKS) else 0.0
            node_features[j, 8] = (max_intensity_mz - self.stats['max_intensity_mz_mean']) / (self.stats['max_intensity_mz_std'] + 1e-6)
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 3. 邻接矩阵 (高斯核: $e^{-\frac{(mz_i - mz_j)^2}{2 \cdot 50^2}}$)
        adj = np.eye(self.max_nodes)
        for r in range(self.max_nodes):
            for c in range(self.max_nodes):
                if r != c:
                    diff = abs(mz_values[r] - mz_values[c])
                    adj[r, c] = np.exp(-(diff**2) / (2 * 50.0**2))
        return node_features, adj

    def fit(self, X, y=None): return self
    def transform(self, X):
        results = [self._extract_single(s) for s in X]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])