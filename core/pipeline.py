# core/pipeline.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# 核心常量 (严格对齐 训练.ipynb 和 qlc.ipynb)
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

class MS1Cleaner(BaseEstimator, TransformerMixin):
    """
    一级质谱清理器：执行 qlc.ipynb 的逻辑
    1. 归一化强度到 0-100
    2. 删除强度为 0 的行
    3. 2Da 范围内只保留最强峰（同位素清理）
    4. 删除归一化强度 < 1 的峰
    """
    def __init__(self, mass_tolerance=2.0, min_intensity=1.0):
        self.mass_tolerance = mass_tolerance
        self.min_intensity = min_intensity

    def fit(self, X, y=None): return self

    def transform(self, df):
        if df is None or df.empty: return pd.DataFrame(columns=['Mass', 'Intensity'])

        # 1. 自动识别列名并转换
        curr_df = df.copy()
        m_col = next((c for c in curr_df.columns if str(c).lower() in ['mass', 'm/z', 'mz']), curr_df.columns[0])
        i_col = next((c for c in curr_df.columns if str(c).lower() in ['intensity', 'int', 'abundance']), curr_df.columns[1])

        res = pd.DataFrame({
            'Mass': pd.to_numeric(curr_df[m_col], errors='coerce'),
            'Intensity': pd.to_numeric(curr_df[i_col], errors='coerce')
        }).dropna()

        if res.empty: return res

        # 2. 归一化强度
        max_i = res['Intensity'].max()
        min_i = res['Intensity'].min()
        if max_i > min_i:
            res['Intensity'] = 100 * (res['Intensity'] - min_i) / (max_i - min_i)
        else:
            res['Intensity'] = 0.0

        # 3. 删除强度为 0 的行
        res = res[res['Intensity'] > 0]

        # 4. 同位素峰清理 (2Da 范围内保留最强)
        # 性能优化：按强度降序处理
        res = res.sort_values('Intensity', ascending=False)
        masses = res['Mass'].values
        intensities = res['Intensity'].values
        keep = np.ones(len(masses), dtype=bool)

        for i in range(len(masses)):
            if not keep[i]: continue
            # 找到 2Da 范围内的其他峰并标记删除
            mask = (np.abs(masses - masses[i]) <= self.mass_tolerance)
            mask[i] = False # 保留自身
            keep[mask & keep] = False

        res = res[keep]

        # 5. 过滤低强度峰并按质量排序
        res = res[res['Intensity'] >= self.min_intensity]
        return res.sort_values('Mass').reset_index(drop=True)

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """
    二级质谱图特征提取器：严格对齐 训练.ipynb 的 10 维特征
    """
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
        else:
            self.stats = {'mz_mean': 0, 'mz_std': 1, 'max_intensity_mz_mean': 0, 'max_intensity_mz_std': 1}

    def _extract_single(self, ms_str):
        # 解析字符串
        peaks = []
        for p in str(ms_str).replace(';', ',').split(','):
            try:
                parts = p.split(':')
                m = float(parts[0].strip())
                i = float(parts[1].strip()) if len(parts) > 1 else 1.0
                peaks.append((m, i))
            except: continue

        if not peaks:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)

        # 1. 排序与截断 (按强度降序)
        peaks.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peaks[0][0]
        actual_peak_count = len(peaks)
        top_peaks = peaks[:self.max_nodes]

        # 2. 填充 (使用最后一个峰填充)
        while len(top_peaks) < self.max_nodes:
            top_peaks.append(top_peaks[-1])

        node_features = np.zeros((self.max_nodes, self.node_dim))
        mz_values = [p[0] for p in top_peaks]

        for j in range(self.max_nodes):
            mz = top_peaks[j][0]
            rmz = round(mz, 1)

            # 特征 F0-F9 严格对齐 训练.ipynb
            # F0: 标准化 m/z
            node_features[j, 0] = (mz - self.stats.get('mz_mean', 0)) / (self.stats.get('mz_std', 1) + 1e-6)
            # F1: 位置比例 (当前索引 / 总峰数)
            node_features[j, 1] = j / max(actual_peak_count, 1)
            # F2: 是否强度最大峰
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            # F3: 是否强度最小峰 (在截断后的集合中)
            node_features[j, 3] = 1.0 if j == min(actual_peak_count, self.max_nodes) - 1 else 0.0
            # F4: 是否那非特征峰
            node_features[j, 4] = 1.0 if any(round(cp, 1) == rmz for cp in CHARACTERISTIC_PEAKS) else 0.0
            # F5: 与最近特征峰的差异
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0
            # F6: 质量区域特征
            region_val = 0.0
            for _, (gp, val) in PEAK_GROUPS.items():
                if any(round(p, 1) == rmz for p in gp): region_val = val; break
            node_features[j, 6] = region_val
            # F7: 是否关键特征峰
            node_features[j, 7] = 1.0 if any(round(kp, 1) == rmz for kp in KEY_PEAKS) else 0.0
            # F8: 最大强度 m/z 标准化
            node_features[j, 8] = (max_intensity_mz - self.stats.get('max_intensity_mz_mean', 0)) / (self.stats.get('max_intensity_mz_std', 1) + 1e-6)
            # F9: 当前 m/z 与最大强度 m/z 的比值
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 3. 邻接矩阵 (高斯核)
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

def get_ms1_pipeline():
    """
    修复 app.py 的导入错误，返回 MS1 预处理 Pipeline
    """
    return Pipeline([
        ('cleaner', MS1Cleaner(mass_tolerance=2.0, min_intensity=1.0))
    ])