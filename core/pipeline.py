import os
import logging
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# --- 常量定义 (对齐 Notebook) ---
CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073, 135.0441, 147.0077, 151.0866,
    166.0975, 169.076, 197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091, 327.1274, 341.1608,
    354.2, 377.1, 396.203
]
KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
MASS_REGIONS = {
    'low_mass': (CHARACTERISTIC_PEAKS[:7], 0.25),
    'middle_mass': (CHARACTERISTIC_PEAKS[7:12], 0.5),
    'high_mass': (CHARACTERISTIC_PEAKS[12:18], 0.75),
    'very_high_mass': (CHARACTERISTIC_PEAKS[18:], 1.0)
}

class MSIsotopeCleaner(BaseEstimator, TransformerMixin):
    """一级质谱同位素清理：对齐 Notebook 的 while 循环分组逻辑"""
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance

    def fit(self, X, y=None): return self

    def transform(self, X):
        if X.empty: return X
        df = X.sort_values('Mass').reset_index(drop=True)
        masses = df['Mass'].values
        intensities = df['Intensity'].values
        keep = np.ones(len(masses), dtype=bool)

        i = 0
        while i < len(masses):
            j = i + 1
            # 找到 tolerance 范围内的所有峰
            while j < len(masses) and masses[j] - masses[i] <= self.tolerance:
                j += 1
            if j - i > 1:
                # 仅保留该组内强度最大的索引
                max_idx = i + np.argmax(intensities[i:j])
                for k in range(i, j):
                    if k != max_idx: keep[k] = False
            i = j
        logger.info(f"IsotopeCleaner: 移除了 {len(df) - keep.sum()} 个同位素峰")
        return df[keep].copy()

class MSIntensityScaler(BaseEstimator, TransformerMixin):
    """归一化到 0-100"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X.empty: return X
        X = X.copy()
        max_v, min_v = X['Intensity'].max(), X['Intensity'].min()
        if max_v > min_v:
            X['Intensity'] = 100 * (X['Intensity'] - min_v) / (max_v - min_v)
        else:
            X['Intensity'] = 100.0
        return X

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """二级质谱图特征提取：对齐 SimplifiedAttentionClassifier"""
    def __init__(self, max_nodes=10, stats_path=None):
        self.max_nodes = max_nodes
        # 从 joblib 读取训练集全局统计变量，若无则从环境变量读取默认值
        if stats_path and os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {
                'mz_mean': float(os.getenv('MZ_MEAN', 0)),
                'mz_std': float(os.getenv('MZ_STD', 1)),
                'max_mz_mean': float(os.getenv('MAX_MZ_MEAN', 0)),
                'max_mz_std': float(os.getenv('MAX_MZ_STD', 1))
            }

    def fit(self, X, y=None): return self

    def _extract_one(self, ms_str):
        # 解析 mass:intensity 格式
        peaks = []
        for p in str(ms_str).replace(';', ',').split(','):
            try:
                parts = p.split(':')
                m = float(parts[0])
                i = float(parts[1]) if len(parts)>1 else 1.0
                peaks.append((m, i))
            except: continue

        # 按强度降序取 Top N
        peaks.sort(key=lambda x: x[1], reverse=True)
        top_peaks = peaks[:self.max_nodes]
        if not top_peaks: return np.zeros((self.max_nodes, 10)), np.eye(self.max_nodes)

        max_mz = top_peaks[0][0] # 降序后第一个是最大强度峰
        mz_vals = [p[0] for p in top_peaks]

        # 1. 构造 10 维节点特征
        node_features = np.zeros((self.max_nodes, 10))
        for i in range(self.max_nodes):
            if i < len(top_peaks):
                m, _ = top_peaks[i]
                node_features[i, 0] = (m - self.stats['mz_mean']) / self.stats['mz_std'] # mz_norm
                node_features[i, 1] = i / max(len(top_peaks), 1) # position_ratio
                node_features[i, 2] = 1.0 if i == 0 else 0.0 # is_first
                node_features[i, 3] = 1.0 if i == len(top_peaks)-1 else 0.0 # is_last

                # 那非匹配 (一位小数)
                rm = round(m, 1)
                node_features[i, 4] = 1.0 if any(round(cp, 1) == rm for cp in CHARACTERISTIC_PEAKS) else 0.0
                node_features[i, 5] = min([abs(m - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0

                # 区域特征
                for _, (p_list, val) in MASS_REGIONS.items():
                    if any(round(p, 1) == rm for p in p_list):
                        node_features[i, 6] = val; break

                node_features[i, 7] = 1.0 if any(round(kp, 1) == rm for kp in KEY_PEAKS) else 0.0
                node_features[i, 8] = (max_mz - self.stats['max_mz_mean']) / self.stats['max_mz_std']
                node_features[i, 9] = m / (max_mz + 1e-6)

        # 2. 构造邻接矩阵 (高斯核)
        adj = np.eye(self.max_nodes)
        for r in range(len(mz_vals)):
            for c in range(len(mz_vals)):
                if r != c:
                    diff = abs(mz_vals[r] - mz_vals[c])
                    adj[r, c] = np.exp(-(diff**2) / (2 * (50.0**2)))

        return node_features, adj

    def transform(self, X):
        """X 为包含 MS2 字符串的列表，返回 [Nodes_Batch, Adj_Batch]"""
        results = [self._extract_one(s) for s in X]
        nodes = np.array([r[0] for r in results])
        adjs = np.array([r[1] for r in results])
        return [nodes, adjs]

def get_ms1_pipeline():
    return Pipeline([
        ('cleaner', MSIsotopeCleaner(tolerance=2.0)),
        ('scaler', MSIntensityScaler()),
        ('filter', LowIntensityFilter(threshold=1.0)) # 之前定义的简单过滤
    ])

class LowIntensityFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.0): self.threshold = threshold
    def fit(self, X, y=None): return self
    def transform(self, X): return X[X['Intensity'] >= self.threshold].copy()