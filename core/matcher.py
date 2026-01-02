import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

class RiskMatcher:
    """
    一级质谱风险匹配器：实现对预处理后峰列表的快速库识别。
    对齐 qlc.ipynb 逻辑：精准匹配 (Risk0/1) 与 两位小数约等匹配 (Risk1/2/3)。
    """
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try:
            self.db = joblib.load(db_path)
            logger.info(f"RiskMatcher: 成功加载风险数据库 {db_path}")
        except Exception as e:
            logger.error(f"RiskMatcher: 加载数据库失败: {e}")
            self.db = None

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        """
        :param df_clean: 经过 MSIsotopeCleaner 处理后的 DataFrame (含 Mass, Intensity)
        :param mode: 'positive' 或 'negative' 离子模式
        :param tolerance: 精准匹配质量误差 (默认 0.005 Da)
        """
        if self.db is None or df_clean.empty:
            return df_clean

        results = df_clean.copy()
        results['Risk_Level'] = 'Safe'
        results['Matched_Mass'] = 0.0

        mode_db = self.db.get(mode, {})

        # 提取数据库内容
        risk0_masses = mode_db.get('risk0', [])
        risk1_precise = mode_db.get('risk1_precise', [])
        risk1_rounded = mode_db.get('risk1_rounded', set())
        risk2_rounded = mode_db.get('risk2', set())
        risk3_rounded = mode_db.get('risk3', set())

        for idx, row in results.iterrows():
            m = row['Mass']
            rm = round(m, 2)

            # 1. 优先级最高：Risk0 精准匹配 (用于触发二级质谱直接判定)
            for target in risk0_masses:
                if abs(m - target) < 0.0001: # 严格误差
                    results.at[idx, 'Risk_Level'] = 'Risk0'
                    results.at[idx, 'Matched_Mass'] = target
                    break
            if results.at[idx, 'Risk_Level'] != 'Safe': continue

            # 2. Risk1 精准匹配
            for target in risk1_precise:
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = target
                    break
            if results.at[idx, 'Risk_Level'] != 'Safe': continue

            # 3. 约等匹配 (两位小数)
            if rm in risk1_rounded:
                results.at[idx, 'Risk_Level'] = 'Risk1'
            elif rm in risk2_rounded:
                results.at[idx, 'Risk_Level'] = 'Risk2'
            elif rm in risk3_rounded:
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