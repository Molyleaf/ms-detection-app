# core/matcher.py
import numpy as np
import joblib

class RiskMatcher:
    """一级质谱风险匹配器：严格执行 0.005 Da 阈值并映射 Output Risk"""
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try:
            self.db = joblib.load(db_path)
            # 统一将键名转为小写以增加鲁棒性
            if self.db:
                self.db = {k.lower(): v for k, v in self.db.items()}
        except:
            self.db = None

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        if self.db is None or df_clean.empty:
            return df_clean

        results = df_clean.copy()
        results['Risk_Level'] = 'Safe'
        results['Matched_Mass'] = 0.0

        # 兼容性处理：确保 mode 为小写
        mode_db = self.db.get(mode.lower(), {})
        if not mode_db:
            return results

        for idx, row in results.iterrows():
            # 优先获取 Mass 列，若无则使用 m/z (对齐 Notebook 列名逻辑)
            m = row.get('Mass', row.get('m/z', 0))
            if m == 0: continue

            rm = round(float(m), 2)
            matched = False

            # 1. 精确匹配逻辑 (Risk0 & Risk1 Precise) - 严格 0.005 Da
            # 检查 Risk0
            for target in mode_db.get('risk0', []):
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk0'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True
                    break
            if matched: continue

            # 检查 Risk1 Precise
            for target in mode_db.get('risk1_precise', []):
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True
                    break
            if matched: continue

            # 2. 模糊/约等匹配 (使用 round(m, 2))
            # 修正：为模糊匹配也填充 Matched_Mass 以便前端观察
            if rm in mode_db.get('risk1_rounded', set()):
                results.at[idx, 'Risk_Level'] = 'Risk1'
                results.at[idx, 'Matched_Mass'] = rm
            elif rm in mode_db.get('risk2', set()):
                results.at[idx, 'Risk_Level'] = 'Risk2'
                results.at[idx, 'Matched_Mass'] = rm
            elif rm in mode_db.get('risk3', set()):
                results.at[idx, 'Risk_Level'] = 'Risk3'
                results.at[idx, 'Matched_Mass'] = rm

        return results

class SpectrumMatcher:
    """二级质谱回溯匹配器"""
    def __init__(self, db_path='data_processed/spectrum_db.joblib'):
        try: self.library = joblib.load(db_path)
        except: self.library = []

    def calculate_cosine(self, spec1, spec2, tol=0.2):
        mz1, int1 = spec1['mz'], spec1['intensities']
        mz2, int2 = spec2['mz'], spec2['intensities']
        s1, s2 = [], []
        for i, m in enumerate(mz1):
            diff = np.abs(mz2 - m)
            idx = np.argmin(diff)
            if diff[idx] <= tol:
                s1.append(int1[i])
                s2.append(int2[idx])
        if not s1: return 0.0
        v1, v2 = np.array(s1), np.array(s2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))

    def match(self, target_spec, top_k=3):
        matches = []
        for entry in self.library:
            score = self.calculate_cosine(target_spec, entry)
            if score > 0.5:
                matches.append({'smiles': entry['smiles'], 'score': score})
        return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_k]