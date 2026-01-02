# core/pipeline.py
import os
import logging
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# --- 常量定义 (严格对齐 qlc.ipynb) ---
CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
    135.0441, 147.0077, 151.0866, 166.0975, 169.076,
    197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091,
    327.1274, 341.1608, 354.2, 377.1, 396.203
]

KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]

PEAK_GROUPS = {
    'low_mass': ([58.0651, 72.0808, 84.0808, 99.0917, 113.1073], 0.25),
    'middle_mass': ([135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709], 0.5),
    'high_mass': ([250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812], 0.75),
    'very_high_mass': ([312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203], 1.0)
}

class MSIntensityScaler(BaseEstimator, TransformerMixin):
    """将 Intensity 归一化到 0-100 (对齐 qlc.ipynb 步骤1)"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X is None or X.empty: return X
        X = X.copy()
        max_v, min_v = X['Intensity'].max(), X['Intensity'].min()
        if max_v > min_v:
            X['Intensity'] = 100 * (X['Intensity'] - min_v) / (max_v - min_v)
        else:
            X['Intensity'] = 100.0
        # 移除零强度行 (对齐 qlc.ipynb 步骤2)
        return X[X['Intensity'] > 0].copy()

class MSIsotopeCleaner(BaseEstimator, TransformerMixin):
    """对齐 qlc.ipynb 步骤3：按强度排序，严格保留 2Da 范围内的最强峰"""
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X is None or X.empty: return X
        # 1. 先按强度降序排序，确保最强峰优先处理
        df = X.sort_values('Intensity', ascending=False).reset_index(drop=True)
        masses = df['Mass'].values
        keep = np.ones(len(df), dtype=bool)

        for i in range(len(df)):
            if not keep[i]: continue
            for j in range(i + 1, len(df)):
                if not keep[j]: continue
                # 如果质量差在 tolerance 范围内，标记较弱的峰（j）为删除
                if abs(masses[i] - masses[j]) <= self.tolerance:
                    keep[j] = False

        # 返回按 Mass 排序的结果
        return df[keep].sort_values('Mass').copy()

class LowIntensityFilter(BaseEstimator, TransformerMixin):
    """对齐 qlc.ipynb 步骤5：过滤强度小于阈值的峰"""
    def __init__(self, threshold=1.0):
        self.threshold = threshold
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X is None or X.empty: return X
        return X[X['Intensity'] >= self.threshold].copy()

def get_ms1_pipeline():
    """一级质谱流水线：顺序对齐 qlc.ipynb"""
    return Pipeline([
        ('scaler', MSIntensityScaler()),  # 归一化并删零
        ('cleaner', MSIsotopeCleaner(tolerance=2.0)), # 同位素清理
        ('filter', LowIntensityFilter(threshold=1.0))  # 强度过滤
    ])

# MS2GraphExtractor 保持 10 维特征提取逻辑，与笔记本一致
class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {'mz_mean': 0, 'mz_std': 1, 'max_mz_mean': 0, 'max_mz_std': 1}

    # 修复点：添加 fit 方法以支持 sklearn 接口
    def fit(self, X, y=None):
        return self

    def _extract_single(self, ms_str):
        # 解析逻辑支持 qlc.ipynb 中的逗号/分号分隔格式
        peaks = []
        for p in str(ms_str).replace(';', ',').split(','):
            try:
                parts = p.split(':')
                m, i = float(parts[0]), float(parts[1]) if len(parts) > 1 else 1.0
                peaks.append((m, i))
            except: continue
        if not peaks:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

        peaks.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peaks[0][0]
        top_peaks = peaks[:self.max_nodes]
        # 填充不足的节点
        while len(top_peaks) < self.max_nodes:
            top_peaks.append(top_peaks[-1] if top_peaks else (0, 0))

        node_features = np.zeros((self.max_nodes, self.node_dim))
        mz_values = [p[0] for p in top_peaks]

        for j in range(self.max_nodes):
            mz, _ = top_peaks[j]
            rmz = round(mz, 1)
            # 10 维特征计算 (F0-F9)
            node_features[j, 0] = (mz - self.stats['mz_mean']) / (self.stats['mz_std'] + 1e-6)
            node_features[j, 1] = j / self.max_nodes
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            node_features[j, 3] = 1.0 if j == (len(peaks[:self.max_nodes]) - 1) else 0.0
            node_features[j, 4] = 1.0 if any(round(cp, 1) == rmz for cp in CHARACTERISTIC_PEAKS) else 0.0
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0
            # 质量区域特征 (F6)
            region_val = 0.0
            for _, (gp, val) in PEAK_GROUPS.items():
                if any(round(p, 1) == rmz for p in gp): region_val = val; break
            node_features[j, 6] = region_val
            node_features[j, 7] = 1.0 if any(round(kp, 1) == rmz for kp in KEY_PEAKS) else 0.0
            node_features[j, 8] = (max_intensity_mz - self.stats['max_mz_mean']) / (self.stats['max_mz_std'] + 1e-6)
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 邻接矩阵 (高斯核)
        adj = np.eye(self.max_nodes)
        for r in range(self.max_nodes):
            for c in range(self.max_nodes):
                if r != c:
                    diff = abs(mz_values[r] - mz_values[c])
                    adj[r, c] = np.exp(-(diff**2) / (2 * 50.0**2))
        return node_features, adj

    def transform(self, X):
        results = [self._extract_single(s) for s in X]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])