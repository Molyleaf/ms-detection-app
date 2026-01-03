# core/pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# 核心常量 (严格对齐 qlc-0103.ipynb 中的常量定义和特征工程逻辑)
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
    """
    一级质谱清理器：严格执行 qlc-0103.ipynb 的逻辑
    1. 归一化强度到 0-100 (100 * (val - min) / (max - min))
    2. 删除强度为 0 的行
    3. 同位素清理：在 2.0 Da 范围内只保留最强峰 (Sliding Window 逻辑)
    4. 删除归一化强度 < 1 的峰
    """
    def __init__(self, mass_tolerance=2.0, min_intensity=1.0):
        self.mass_tolerance = mass_tolerance
        self.min_intensity = min_intensity

    def fit(self, X, y=None): return self

    def transform(self, df):
        if df is None or df.empty:
            return pd.DataFrame(columns=['Mass', 'Intensity'])

        # 1. 自动识别列名并预处理
        curr_df = df.copy()
        m_col = next((c for c in curr_df.columns if str(c).lower() in ['mass', 'm/z', 'mz']), curr_df.columns[0])
        i_col = next((c for c in curr_df.columns if str(c).lower() in ['intensity', 'int', 'abundance']), curr_df.columns[1])

        res = pd.DataFrame({
            'Mass': pd.to_numeric(curr_df[m_col], errors='coerce'),
            'Intensity': pd.to_numeric(curr_df[i_col], errors='coerce')
        }).dropna()

        if res.empty: return res

        # 2. 归一化强度 (0-100)
        max_i = res['Intensity'].max()
        min_i = res['Intensity'].min()
        if max_i > min_i:
            res['Intensity'] = 100 * (res['Intensity'] - min_i) / (max_i - min_i)
        else:
            res['Intensity'] = 0.0

        # 3. 删除强度为 0 的行
        res = res[res['Intensity'] > 0].sort_values('Mass').reset_index(drop=True)

        # 4. 同位素峰清理 (2Da 范围内保留最强) - 对齐 Notebook 的 Sliding Window 逻辑
        masses = res['Mass'].values
        intensities = res['Intensity'].values
        keep = np.ones(len(masses), dtype=bool)

        i = 0
        while i < len(masses):
            j = i + 1
            # 找到当前峰之后 2Da 范围内的所有峰
            while j < len(masses) and masses[j] - masses[i] <= self.mass_tolerance:
                j += 1

            if j - i > 1:
                # 在窗口 [i, j-1] 中找到最大强度索引
                max_idx = i + np.argmax(intensities[i:j])
                for k in range(i, j):
                    if k != max_idx:
                        keep[k] = False
                i = j # 跳过处理过的窗口
            else:
                i += 1

        res = res[keep]

        # 5. 过滤低强度峰
        res = res[res['Intensity'] >= self.min_intensity]
        return res.sort_values('Mass').reset_index(drop=True)

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """
    二级质谱图特征提取器：严格对齐 qlc-0103.ipynb 的 10 维特征提取逻辑
    """
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        # 加载标准化统计量
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {'mz_mean': 0, 'mz_std': 1, 'max_intensity_mz_mean': 0, 'max_intensity_mz_std': 1}

    def _extract_single(self, ms_str):
        # 解析质谱字符串 (m1:i1,m2:i2...)
        peak_data = []
        try:
            for p in str(ms_str).replace(';', ',').split(','):
                if ':' in p:
                    parts = p.split(':')
                    m = float(parts[0].strip())
                    i = float(parts[1].strip())
                    peak_data.append((m, i))
        except: pass

        if not peak_data:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

        # 1. 排序与截断 (按强度降序)
        peak_data.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peak_data[0][0]
        actual_peak_count = len(peak_data)

        # 2. 计算节点特征 (10维)
        node_features = np.zeros((self.max_nodes, self.node_dim))
        top_mz_values = []

        for j in range(self.max_nodes):
            # 填充逻辑：不足 max_nodes 时用最后一个峰填充
            if j < actual_peak_count:
                mz, intensity = peak_data[j]
            elif actual_peak_count > 0:
                mz, intensity = peak_data[-1]
            else:
                mz, intensity = 0.0, 0.0

            top_mz_values.append(mz)
            rmz = round(mz, 1)

            # F0: 标准化 m/z
            node_features[j, 0] = (mz - self.stats.get('mz_mean', 0)) / (self.stats.get('mz_std', 1) + 1e-6)
            # F1: 位置比例 (0 到 1)
            node_features[j, 1] = j / max(actual_peak_count, 1)
            # F2: 是否强度最大的第一个峰
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            # F3: 是否（截断前）的最后一个峰
            node_features[j, 3] = 1.0 if j == actual_peak_count - 1 else 0.0
            # F4: 那非特征峰匹配 (1位小数)
            char_rounded = [round(p, 1) for p in CHARACTERISTIC_PEAKS]
            node_features[j, 4] = 1.0 if rmz in char_rounded else 0.0
            # F5: 与最近特征峰的距离标准化
            min_diff = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) if CHARACTERISTIC_PEAKS else 100.0
            node_features[j, 5] = min_diff / 100.0
            # F6: 质量区域特征 (0.25, 0.5, 0.75, 1.0)
            mass_region = 0.0
            for group_name, group_peaks in PEAK_GROUPS.items():
                if rmz in group_peaks: # PEAK_GROUPS 已经是 round(p, 1) 后的值
                    mass_region = {'low_mass': 0.25, 'middle_mass': 0.5, 'high_mass': 0.75, 'very_high_mass': 1.0}[group_name]
                    break
            node_features[j, 6] = mass_region
            # F7: 是否关键特征峰 (1位小数)
            key_rounded = [round(p, 1) for p in KEY_PEAKS]
            node_features[j, 7] = 1.0 if rmz in key_rounded else 0.0
            # F8: 母离子 (最大强度 m/z) 标准化
            node_features[j, 8] = (max_intensity_mz - self.stats.get('max_intensity_mz_mean', 0)) / (self.stats.get('max_intensity_mz_std', 1) + 1e-6)
            # F9: 当前 m/z 与母离子的比值
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 3. 邻接矩阵 (高斯相似度)
        adj = np.eye(self.max_nodes)
        for r in range(self.max_nodes):
            for c in range(self.max_nodes):
                if r != c:
                    mz_diff = abs(top_mz_values[r] - top_mz_values[c])
                    adj[r, c] = np.exp(-(mz_diff**2) / (2 * 50.0**2))

        return node_features, adj

    def fit(self, X, y=None): return self
    def transform(self, X):
        results = [self._extract_single(s) for s in X]
        # 返回三维张量 [batch, nodes, features/nodes]
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])

def get_ms1_pipeline():
    """
    返回符合 qlc-0103.ipynb 逻辑的预处理 Pipeline
    """
    return Pipeline([
        ('cleaner', MS1Cleaner(mass_tolerance=2.0, min_intensity=1.0))
    ])