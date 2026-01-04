# core/pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# 核心常量：严格对齐 qlc-0103.ipynb
CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073, 135.0441, 147.0077, 151.0866,
    166.0975, 169.076, 197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091, 327.1274, 341.1608,
    354.2, 377.1, 396.203
]

KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]

PEAK_GROUPS = {
    'low_mass': [58.1, 72.1, 84.1, 99.1, 113.1],
    'middle_mass': [135.0, 147.0, 151.1, 166.1, 169.1, 197.1],
    'high_mass': [250.1, 256.1, 262.1, 283.1, 297.1, 299.1, 302.1],
    'very_high_mass': [312.2, 315.1, 327.1, 341.2, 354.2, 377.1, 396.2]
}

class MS1Cleaner(BaseEstimator, TransformerMixin):
    """一级质谱清理器：执行归一化、同位素清理(2Da)、低强度过滤"""
    def __init__(self, mass_tolerance=2.0, min_intensity=1.0):
        self.mass_tolerance = mass_tolerance
        self.min_intensity = min_intensity

    def fit(self, X, y=None): return self

    def transform(self, df):
        if df is None or df.empty: return pd.DataFrame(columns=['Mass', 'Intensity'])

        curr_df = df.copy()
        m_col = next((c for c in curr_df.columns if str(c).lower() in ['mass', 'm/z', 'mz']), curr_df.columns[0])
        i_col = next((c for c in curr_df.columns if str(c).lower() in ['intensity', 'int', 'abundance']), curr_df.columns[1])

        res = pd.DataFrame({
            'Mass': pd.to_numeric(curr_df[m_col], errors='coerce'),
            'Intensity': pd.to_numeric(curr_df[i_col], errors='coerce')
        }).dropna()

        # 1. 归一化强度 (0-100)
        max_i, min_i = res['Intensity'].max(), res['Intensity'].min()
        res['Intensity'] = 100 * (res['Intensity'] - min_i) / (max_i - min_i + 1e-9)

        # 2. 删除零强度并按质量排序
        res = res[res['Intensity'] > 0].sort_values('Mass').reset_index(drop=True)

        # 3. 同位素峰清理 (Sliding Window 2Da)
        masses, intensities = res['Mass'].values, res['Intensity'].values
        keep = np.ones(len(masses), dtype=bool)
        i = 0
        while i < len(masses):
            j = i + 1
            while j < len(masses) and masses[j] - masses[i] <= self.mass_tolerance:
                j += 1
            if j - i > 1:
                max_idx = i + np.argmax(intensities[i:j])
                for k in range(i, j):
                    if k != max_idx: keep[k] = False
                i = j
            else: i += 1

        res = res[keep]
        # 4. 过滤归一化后强度 < 1 的峰
        return res[res['Intensity'] >= self.min_intensity].reset_index(drop=True)

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """二级质谱图特征提取器：10 维特征"""
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {'mz_mean': 300, 'mz_std': 150, 'max_intensity_mz_mean': 350, 'max_intensity_mz_std': 100}

    def _extract_single(self, ms_str):
        peak_data = []
        try:
            for p in str(ms_str).replace(';', ',').split(','):
                if ':' in p:
                    parts = p.split(':')
                    peak_data.append((float(parts[0]), float(parts[1])))
        except: pass

        if not peak_data:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

        # 排序与截断
        peak_data.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peak_data[0][0]
        actual_count = len(peak_data)

        node_features = np.zeros((self.max_nodes, self.node_dim))
        top_mzs = []

        for j in range(self.max_nodes):
            if j < actual_count:
                mz, intensity = peak_data[j]
            elif actual_count > 0:
                mz, intensity = peak_data[-1]
            else: mz, intensity = 0.0, 0.0

            top_mzs.append(mz)
            rmz = round(mz, 1)

            # F0-F9 特征工程
            node_features[j, 0] = (mz - self.stats.get('mz_mean', 0)) / (self.stats.get('mz_std', 1) + 1e-6)
            node_features[j, 1] = j / max(actual_count, 1)
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            node_features[j, 3] = 1.0 if j == actual_count - 1 else 0.0
            node_features[j, 4] = 1.0 if any(abs(mz - cp) < 0.1 for cp in CHARACTERISTIC_PEAKS) else 0.0
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0 if CHARACTERISTIC_PEAKS else 1.0

            mass_reg = 0.0
            for name, pks in PEAK_GROUPS.items():
                if rmz in pks:
                    mass_reg = {'low_mass': 0.25, 'middle_mass': 0.5, 'high_mass': 0.75, 'very_high_mass': 1.0}[name]
                    break
            node_features[j, 6] = mass_reg
            node_features[j, 7] = 1.0 if any(abs(mz - kp) < 0.1 for kp in KEY_PEAKS) else 0.0
            node_features[j, 8] = (max_intensity_mz - self.stats.get('max_intensity_mz_mean', 0)) / (self.stats.get('max_intensity_mz_std', 1) + 1e-6)
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 邻接矩阵 (高斯内核)
        adj = np.eye(self.max_nodes)
        for r in range(min(actual_count, self.max_nodes)):
            for c in range(min(actual_count, self.max_nodes)):
                if r != c:
                    adj[r, c] = np.exp(-(top_mzs[r] - top_mzs[c])**2 / (2 * 50.0**2))
        return node_features, adj

    def fit(self, X, y=None): return self
    def transform(self, X):
        res = [self._extract_single(s) for s in X]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])

def get_ms1_pipeline():
    return Pipeline([('cleaner', MS1Cleaner())])