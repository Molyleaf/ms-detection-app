# core/matcher.py
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

class RiskMatcher:
    """一级质谱风险匹配器：严格执行 qlc.ipynb 逻辑"""
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try:
            self.db = joblib.load(db_path)
        except Exception as e:
            logger.error(f"加载数据库失败: {e}")
            self.db = None

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        if self.db is None or df_clean.empty:
            return df_clean

        results = df_clean.copy()
        results['Risk_Level'] = 'Safe'
        results['Matched_Mass'] = 0.0
        mode_db = self.db.get(mode, {})

        for idx, row in results.iterrows():
            m = row['Mass']
            rm = round(m, 2)

            # 1. Risk0 严格精准匹配 (0.0001 Da 误差)
            matched = False
            for target in mode_db.get('risk0', []):
                if abs(m - target) < 0.0001:
                    results.at[idx, 'Risk_Level'] = 'Risk0'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True
                    break
            if matched: continue

            # 2. Risk1 精准匹配 (0.005 Da 误差)
            for target in mode_db.get('risk1_precise', []):
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True
                    break
            if matched: continue

            # 3. Risk1/2/3 约等匹配 (2位小数)
            if rm in mode_db.get('risk1_rounded', set()):
                results.at[idx, 'Risk_Level'] = 'Risk1'
            elif rm in mode_db.get('risk2', set()):
                results.at[idx, 'Risk_Level'] = 'Risk2'
            elif rm in mode_db.get('risk3', set()):
                results.at[idx, 'Risk_Level'] = 'Risk3'

        return results

class SpectrumMatcher:
    """
    二级质谱相似度匹配器：实现阳性样本的库回溯。
    """
    def __init__(self, db_path='data_processed/spectrum_db.joblib'):
        try:
            self.library = joblib.load(db_path)
            logger.info(f"SpectrumMatcher: 成功加载谱图库，共 {len(self.library)} 条")
        except Exception as e:
            logger.error(f"SpectrumMatcher: 加载失败: {e}")
            self.library = []

    def calculate_cosine(self, spec1, spec2, tolerance=0.2):
        """
        计算两个谱图的余弦相似度
        spec 格式: {'mz': np.array, 'intensities': np.array}
        """
        mz1, int1 = spec1['mz'], spec1['intensities']
        mz2, int2 = spec2['mz'], spec2['intensities']

        shared_int1, shared_int2 = [], []
        for i, m1 in enumerate(mz1):
            diff = np.abs(mz2 - m1)
            idx = np.argmin(diff)
            if diff[idx] <= tolerance:
                shared_int1.append(int1[i])
                shared_int2.append(int2[idx])

        if not shared_int1: return 0.0

        v1, v2 = np.array(shared_int1), np.array(shared_int2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))

    def match(self, target_spec, top_k=3):
        """针对目标谱图进行全库检索"""
        matches = []
        for entry in self.library:
            score = self.calculate_cosine(target_spec, entry)
            if score > 0.5: # 相似度阈值
                matches.append({'smiles': entry['smiles'], 'score': score})

        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k]