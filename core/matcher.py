import numpy as np
import joblib
import logging
import pandas as pd

# 配置日志
logger = logging.getLogger(__name__)

class RiskMatcher:
    """
    一级质谱风险匹配器
    用于快速识别质谱中的已知风险物质峰
    """
    def __init__(self, db_path):
        try:
            self.db = joblib.load(db_path)
            logger.info(f"RiskMatcher: 成功加载风险数据库 {db_path}")
        except Exception as e:
            logger.error(f"RiskMatcher: 加载数据库失败: {e}")
            self.db = None

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        """
        将预处理后的峰与风险库比对
        :param df_clean: 经过 Pipeline 处理后的 DataFrame
        :param mode: 'positive' 或 'negative' 离子模式
        :param tolerance: 精确匹配的质量误差容限 (Da)
        :return: 标记了风险等级的 DataFrame
        """
        if self.db is None:
            return df_clean

        results = []
        mode_db = self.db.get(mode, {})

        # 提取库数据
        risk1_precise = mode_db.get('risk1_precise', [])
        risk1_rounded = mode_db.get('risk1_rounded', set())
        risk2_rounded = mode_db.get('risk2', set())
        risk3_rounded = mode_db.get('risk3', set())

        for _, row in df_clean.iterrows():
            mz = row['Mass']
            risk_level = "None"

            # 1. 风险1 精确匹配 (高优先级)
            # 使用 numpy 向量化计算误差
            if any(np.abs(np.array(risk1_precise) - mz) <= tolerance):
                risk_level = "Risk_1"
            else:
                # 2. 快速粗略匹配 (使用集合 O(1) 复杂度)
                mz_r = round(mz, 2)
                if mz_r in risk1_rounded:
                    risk_level = "Risk_1"
                elif mz_r in risk2_rounded:
                    risk_level = "Risk_2"
                elif mz_r in risk3_rounded:
                    risk_level = "Risk_3"

            if risk_level != "None":
                results.append({
                    'Mass': mz,
                    'Intensity': row['Intensity'],
                    'Risk': risk_level
                })

        match_df = pd.DataFrame(results)
        logger.info(f"RiskMatcher: 匹配完成，发现 {len(match_df)} 个危险峰")
        return match_df

class SpectrumMatcher:
    """
    二级质谱相似度匹配器
    用于计算实验谱图与数据库谱图的余弦相似度
    """
    def __init__(self, lib_path):
        try:
            self.library = joblib.load(lib_path)
            logger.info(f"SpectrumMatcher: 成功加载谱图库 {lib_path}，共 {len(self.library)} 条记录")
        except Exception as e:
            logger.error(f"SpectrumMatcher: 加载谱图库失败: {e}")
            self.library = []

    def calculate_cosine(self, spec1, spec2, tolerance=0.2):
        """
        计算两个谱图之间的余弦相似度
        spec 格式: {'mz': np.array, 'intensities': np.array}
        """
        # 简单的质谱对齐逻辑
        mz1, int1 = spec1['mz'], spec1['intensities']
        mz2, int2 = spec2['mz'], spec2['intensities']

        score = 0
        shared_int1 = []
        shared_int2 = []

        # 以 spec1 为基准匹配 spec2
        for i, m1 in enumerate(mz1):
            diff = np.abs(mz2 - m1)
            idx = np.argmin(diff)
            if diff[idx] <= tolerance:
                shared_int1.append(int1[i])
                shared_int2.append(int2[idx])

        if not shared_int1:
            return 0.0

        # 计算余弦值
        v1 = np.array(shared_int1)
        v2 = np.array(shared_int2)
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(cosine)

    def search_library(self, query_spec, top_k=3):
        """
        在库中搜索最相似的化合物
        """
        results = []
        for entry in self.library:
            score = self.calculate_cosine(query_spec, entry)
            if score > 0.5: # 相似度阈值
                results.append({
                    'smiles': entry['smiles'],
                    'score': score
                })

        # 按得分排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]