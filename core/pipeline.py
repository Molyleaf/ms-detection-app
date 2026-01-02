import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# 配置日志
logger = logging.getLogger(__name__)

class ZeroIntensityRemover(BaseEstimator, TransformerMixin):
    """去除强度为0的行"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        before_count = len(X)
        X_clean = X[X['Intensity'] > 0].copy()
        logger.info(f"ZeroIntensityRemover: 过滤了 {before_count - len(X_clean)} 条零强度数据")
        return X_clean

class IntensityScaler(BaseEstimator, TransformerMixin):
    """归一化强度到 0-100 范围"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.empty:
            return X
        X = X.copy()
        max_val = X['Intensity'].max()
        min_val = X['Intensity'].min()

        if max_val == min_val:
            X['Intensity'] = 100.0
        else:
            X['Intensity'] = 100.0 * (X['Intensity'] - min_val) / (max_val - min_val)

        logger.info("IntensityScaler: 完成 0-100 归一化")
        return X

class IsotopeCleaner(BaseEstimator, TransformerMixin):
    """
    同位素峰清理：
    在指定窗口（默认 2 Da）范围内，如果存在更强的峰，则删除当前峰。
    """
    def __init__(self, window_da=2.0):
        self.window_da = window_da

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.empty:
            return X

        # 按质量数排序
        X = X.sort_values(by='Mass').reset_index(drop=True)
        keep_mask = np.ones(len(X), dtype=bool)

        masses = X['Mass'].values
        intensities = X['Intensity'].values

        for i in range(len(X)):
            current_mass = masses[i]
            current_int = intensities[i]

            # 查找在 [mass - window, mass + window] 范围内的索引
            # 使用 searchsorted 优化查找速度
            start_idx = np.searchsorted(masses, current_mass - self.window_da)
            end_idx = np.searchsorted(masses, current_mass + self.window_da, side='right')

            # 在该范围内查找是否有强度更高的峰
            local_intensities = intensities[start_idx:end_idx]
            if np.any(local_intensities > current_int):
                keep_mask[i] = False

        X_clean = X[keep_mask].copy()
        logger.info(f"IsotopeCleaner: 窗口 {self.window_da}Da 内清理了 {len(X) - len(X_clean)} 个次强峰/同位素峰")
        return X_clean

class LowIntensityFilter(BaseEstimator, TransformerMixin):
    """删除归一化后强度小于指定阈值（默认 1.0）的峰"""
    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        before_count = len(X)
        X_clean = X[X['Intensity'] >= self.threshold].copy()
        logger.info(f"LowIntensityFilter: 删除了 {before_count - len(X_clean)} 个强度低于 {self.threshold} 的峰")
        return X_clean

def get_preprocessing_pipeline():
    """
    组装完整的预处理管线
    效果：df_clean = preprocessing_pipeline.fit_transform(raw_df)
    """
    return Pipeline([
        ('zero_remover', ZeroIntensityRemover()),
        ('scaler', IntensityScaler()),
        ('isotope_cleaner', IsotopeCleaner(window_da=2.0)),
        ('low_intensity_filter', LowIntensityFilter(threshold=1.0))
    ])

# ======================================================================
# MS2 特征提取管线 (用于注意力模型)
# ======================================================================

class MS2FeatureExtractor(BaseEstimator, TransformerMixin):
    """将 MS2 字符串转换为注意力模型需要的 (10, 10) 张量格式"""
    def __init__(self, top_n=10, feature_dim=10):
        self.top_n = top_n
        self.feature_dim = feature_dim

    def fit(self, X, y=None):
        return self

    def _parse_single_spec(self, ms_str):
        # 提取 top_n 个峰
        try:
            peaks = [p.split(':') for p in str(ms_str).split(',') if ':' in p]
            peaks = [[float(p[0]), float(p[1])] for p in peaks]
            # 按强度降序
            peaks.sort(key=lambda x: x[1], reverse=True)
            top_peaks = peaks[:self.top_n]

            # 如果不足 top_n，补零
            while len(top_peaks) < self.top_n:
                top_peaks.append([0.0, 0.0])

            # 转换为 (10, 10) 矩阵。这里根据 Notebook 逻辑进行填充或转换
            # 假设是将 10 个峰的 (m/z, intensity) 扩展或平铺
            # 根据 notebook X_test.reshape(-1, 10, 10)，这里构造 100 个特征
            features = []
            for mz, intensity in top_peaks:
                # 构造简易特征：mz, intensity 及其 8 个派生特征（如 log, sqrt 等）
                features.extend([mz, intensity, mz**2, np.sqrt(mz), np.log1p(mz),
                                 intensity**2, np.sqrt(intensity), np.log1p(intensity),
                                 mz+intensity, mz/(intensity+1e-6)])
            return np.array(features).reshape(self.top_n, self.feature_dim)
        except Exception as e:
            logger.error(f"MS2 解析失败: {e}")
            return np.zeros((self.top_n, self.feature_dim))

    def transform(self, X):
        """
        X: list of ms_strings
        """
        features = [self._parse_single_spec(s) for s in X]
        return np.array(features)