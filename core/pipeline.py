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
ROUNDED_CHAR_PEAKS = [round(p, 1) for p in CHARACTERISTIC_PEAKS]

KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
ROUNDED_KEY_PEAKS = [round(p, 1) for p in KEY_PEAKS]

PEAK_GROUPS = {
    'low_mass': [round(p, 1) for p in [58.1, 72.1, 84.1, 99.1, 113.1]],
    'middle_mass': [round(p, 1) for p in [135.0, 147.0, 151.1, 166.1, 169.1, 197.1]],
    'high_mass': [round(p, 1) for p in [250.1, 256.1, 262.1, 283.1, 297.1, 299.1, 302.1]],
    'very_high_mass': [round(p, 1) for p in [312.2, 315.1, 327.1, 341.2, 354.2, 377.1, 396.2]]
}


class MS1Cleaner(BaseEstimator, TransformerMixin):
    """
    一级质谱清理器：执行归一化、同位素清理(2Da)、低强度过滤

    同位素清理：严格对齐 Notebook 的“按质量排序 + 窗口聚类(2Da) + 组内取最大强度 + 跳跃”模式
    """
    def __init__(self, mass_tolerance=2.0, min_intensity=1.0):
        self.mass_tolerance = float(mass_tolerance)
        self.min_intensity = float(min_intensity)

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=['Mass', 'Intensity'])

        curr_df = df.copy()
        m_col = next((c for c in curr_df.columns if str(c).lower() in ['mass', 'm/z', 'mz']), curr_df.columns[0])
        i_col = next((c for c in curr_df.columns if str(c).lower() in ['intensity', 'int', 'abundance']), curr_df.columns[1])

        res = pd.DataFrame({
            'Mass': pd.to_numeric(curr_df[m_col], errors='coerce'),
            'Intensity': pd.to_numeric(curr_df[i_col], errors='coerce')
        }).dropna()

        if res.empty:
            return pd.DataFrame(columns=['Mass', 'Intensity'])

        # 1) 强度归一化到 0-100（Notebook 的做法）
        max_i = float(res['Intensity'].max())
        min_i = float(res['Intensity'].min())
        if max_i == min_i:
            # Notebook 在 max==min 情况下会给 0；但为了流程稳定，这里给常数也可。
            # 若你需要严格与 Notebook 的 0 对齐，可改为 0.0。
            res['Intensity'] = 0.0
        else:
            res['Intensity'] = 100.0 * (res['Intensity'] - min_i) / (max_i - min_i + 1e-9)

        # 2) 删除归一化后强度为 0 的峰，并按 Mass 升序（同位素窗口聚类准备）
        res = res[res['Intensity'] > 0].sort_values('Mass').reset_index(drop=True)
        if res.empty:
            return pd.DataFrame(columns=['Mass', 'Intensity'])

        # 3) 同位素清理：按 Mass 升序 + 2Da 窗口聚类 + 组内取最大强度 + 跳跃
        masses = res['Mass'].to_numpy(dtype=float)
        intensities = res['Intensity'].to_numpy(dtype=float)

        keep = np.ones(len(masses), dtype=bool)

        i = 0
        while i < len(masses):
            j = i + 1
            while j < len(masses) and (masses[j] - masses[i]) <= self.mass_tolerance:
                j += 1

            if j - i > 1:
                max_idx_in_group = int(np.argmax(intensities[i:j])) + i
                for k in range(i, j):
                    if k != max_idx_in_group:
                        keep[k] = False

            # 关键：跳到下一组
            i = j

        res = res[keep].reset_index(drop=True)

        # 4) 过滤低强度（Notebook 默认阈值 1.0）
        res = res[res['Intensity'] >= self.min_intensity].reset_index(drop=True)

        # 保持按 Mass 升序（对齐 Notebook 输出习惯）
        return res.sort_values('Mass').reset_index(drop=True)


class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """二级质谱图特征提取器：保持与模型输入对齐"""
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
        except:
            pass

        if not peak_data:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

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
            else:
                mz, intensity = 0.0, 0.0

            top_mzs.append(mz)
            rmz = round(mz, 1)

            node_features[j, 0] = (mz - self.stats.get('mz_mean', 0)) / (self.stats.get('mz_std', 1) + 1e-6)
            node_features[j, 1] = j / max(actual_count, 1)
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            node_features[j, 3] = 1.0 if j == actual_count - 1 else 0.0
            node_features[j, 4] = 1.0 if rmz in ROUNDED_CHAR_PEAKS else 0.0
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0

            mass_reg = 0.0
            for name, pks in PEAK_GROUPS.items():
                if rmz in pks:
                    mass_reg = {'low_mass': 0.25, 'middle_mass': 0.5, 'high_mass': 0.75, 'very_high_mass': 1.0}[name]
                    break
            node_features[j, 6] = mass_reg
            node_features[j, 7] = 1.0 if rmz in ROUNDED_KEY_PEAKS else 0.0
            node_features[j, 8] = (max_intensity_mz - self.stats.get('max_intensity_mz_mean', 0)) / (self.stats.get('max_intensity_mz_std', 1) + 1e-6)
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        adj = np.eye(self.max_nodes)
        for r in range(min(actual_count, self.max_nodes)):
            for c in range(min(actual_count, self.max_nodes)):
                if r != c:
                    adj[r, c] = np.exp(-(top_mzs[r] - top_mzs[c]) ** 2 / (2 * 50.0 ** 2))
        return node_features, adj

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = [self._extract_single(s) for s in X]
        return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def get_ms1_pipeline():
    return Pipeline([('cleaner', MS1Cleaner())])