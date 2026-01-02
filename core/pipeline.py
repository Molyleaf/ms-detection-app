# core/pipeline.py
import os
import logging
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# --- 常量定义 (严格对齐 训练.ipynb) ---
# 那非类化合物特征峰
CHARACTERISTIC_PEAKS = [
    58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
    135.0441, 147.0077, 151.0866, 166.0975, 169.076,
    197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
    297.1346, 299.1139, 302.0812, 312.1581, 315.091,
    327.1274, 341.1608, 354.2, 377.1, 396.203
]

# 关键特征峰
KEY_PEAKS = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]

# 质量区域特征映射
PEAK_GROUPS = {
    'low_mass': ([58.0651, 72.0808, 84.0808, 99.0917, 113.1073], 0.25),
    'middle_mass': ([135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709], 0.5),
    'high_mass': ([250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812], 0.75),
    'very_high_mass': ([312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203], 1.0)
}

class MSIsotopeCleaner(BaseEstimator, TransformerMixin):
    """一级质谱同位素清理：在 2Da 范围内只保留最强的峰"""
    def __init__(self, tolerance=2.0):
        self.tolerance = tolerance

    def fit(self, X, y=None): return self

    def transform(self, X):
        if X is None or X.empty: return X
        # 确保按质量排序
        df = X.sort_values('Mass').reset_index(drop=True)
        masses = df['Mass'].values
        intensities = df['Intensity'].values
        keep = np.ones(len(masses), dtype=bool)

        i = 0
        while i < len(masses):
            j = i + 1
            # 找到 tolerance 范围内的所有峰组
            while j < len(masses) and masses[j] - masses[i] <= self.tolerance:
                j += 1
            if j - i > 1:
                # 仅保留该组内强度最大的峰
                max_idx_in_group = i + np.argmax(intensities[i:j])
                for k in range(i, j):
                    if k != max_idx_in_group:
                        keep[k] = False
            i = j
        return df[keep].copy()

class MSIntensityScaler(BaseEstimator, TransformerMixin):
    """将 Intensity 归一化到 0-100"""
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X is None or X.empty: return X
        X = X.copy()
        max_v = X['Intensity'].max()
        min_v = X['Intensity'].min()
        if max_v > min_v:
            X['Intensity'] = 100 * (X['Intensity'] - min_v) / (max_v - min_v)
        else:
            X['Intensity'] = 100.0
        return X

class MS2GraphExtractor(BaseEstimator, TransformerMixin):
    """
    二级质谱图特征提取器：
    1. 提取 10 维节点特征向量
    2. 基于 m/z 差异构建高斯核邻接矩阵
    """
    def __init__(self, max_nodes=10, node_dim=10, stats_path='data_processed/stats.joblib'):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        # 加载 convert.py 生成的全局统计量
        if os.path.exists(stats_path):
            self.stats = joblib.load(stats_path)
            logger.info(f"成功加载统计量: {self.stats}")
        else:
            logger.warning(f"统计文件 {stats_path} 不存在，使用默认值。")
            self.stats = {'mz_mean': 0, 'mz_std': 1, 'max_mz_mean': 0, 'max_mz_std': 1}

    def fit(self, X, y=None): return self

    def _extract_single(self, ms_str):
        """解析单条 MS2 字符串并生成特征矩阵和邻接矩阵"""
        # 1. 解析字符串 (支持逗号或分号分隔)
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

        # 2. 按强度降序排列并截取前 N 个
        peaks.sort(key=lambda x: x[1], reverse=True)
        max_intensity_mz = peaks[0][0] # 强度最大的峰的 m/z
        top_peaks = peaks[:self.max_nodes]

        # 填充逻辑：如果峰不够，用最后一个峰填充
        if len(top_peaks) < self.max_nodes:
            last_peak = top_peaks[-1]
            while len(top_peaks) < self.max_nodes:
                top_peaks.append(last_peak)

        actual_len = len(peaks[:self.max_nodes])
        mz_values = [p[0] for p in top_peaks]

        # 3. 构建 10 维节点特征向量
        node_features = np.zeros((self.max_nodes, self.node_dim))
        rounded_char = [round(p, 1) for p in CHARACTERISTIC_PEAKS]
        rounded_key = [round(p, 1) for p in KEY_PEAKS]

        for j in range(self.max_nodes):
            mz, _ = top_peaks[j]
            rmz = round(mz, 1)

            # F0: 标准化 m/z
            node_features[j, 0] = (mz - self.stats['mz_mean']) / self.stats['mz_std']
            # F1: 位置比例
            node_features[j, 1] = j / max(actual_len, 1)
            # F2: 是否强度最大的峰
            node_features[j, 2] = 1.0 if j == 0 else 0.0
            # F3: 是否强度最小的峰
            node_features[j, 3] = 1.0 if j == (actual_len - 1) else 0.0
            # F4: 是否那非特征峰 (小数点后1位匹配)
            node_features[j, 4] = 1.0 if rmz in rounded_char else 0.0
            # F5: 与最近特征峰的 m/z 差异
            node_features[j, 5] = min([abs(mz - cp) for cp in CHARACTERISTIC_PEAKS]) / 100.0
            # F6: 质量区域特征 (小数点后1位匹配)
            region_val = 0.0
            for _, (group_peaks, val) in PEAK_GROUPS.items():
                if any(round(gp, 1) == rmz for gp in group_peaks):
                    region_val = val
                    break
            node_features[j, 6] = region_val
            # F7: 是否关键峰
            node_features[j, 7] = 1.0 if rmz in rounded_key else 0.0
            # F8: 最大强度 m/z 标准化
            node_features[j, 8] = (max_intensity_mz - self.stats['max_mz_mean']) / self.stats['max_mz_std']
            # F9: 当前 m/z 与最大强度 m/z 的比值
            node_features[j, 9] = mz / (max_intensity_mz + 1e-6)

        # 4. 构建邻接矩阵 (高斯核，sigma=50.0)
        adj = np.eye(self.max_nodes)
        for r in range(self.max_nodes):
            for c in range(self.max_nodes):
                if r != c:
                    diff = abs(mz_values[r] - mz_values[c])
                    adj[r, c] = np.exp(-(diff**2) / (2 * (50.0**2)))

        return node_features, adj

    def transform(self, X):
        """输入 MS2 字符串列表，返回两个张量：节点特征批次和邻接矩阵批次"""
        results = [self._extract_single(s) for s in X]
        nodes = np.array([r[0] for r in results])
        adjs = np.array([r[1] for r in results])
        return [nodes, adjs]

def get_ms1_pipeline():
    """一级质谱预处理流水线"""
    return Pipeline([
        ('cleaner', MSIsotopeCleaner(tolerance=2.0)),
        ('scaler', MSIntensityScaler()),
        ('filter', LowIntensityFilter(threshold=1.0))
    ])

class LowIntensityFilter(BaseEstimator, TransformerMixin):
    """过滤归一化后强度过低的峰"""
    def __init__(self, threshold=1.0):
        self.threshold = threshold
    def fit(self, X, y=None): return self
    def transform(self, X):
        if X is None or X.empty: return X
        return X[X['Intensity'] >= self.threshold].copy()