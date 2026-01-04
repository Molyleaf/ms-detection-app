# core/matcher.py
import numpy as np
import joblib
import logging

class RiskMatcher:
    """一级质谱风险匹配器：严格执行 qlc.ipynb 0.005 Da 阈值"""
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try: self.db = joblib.load(db_path)
        except: self.db = None

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        if self.db is None or df_clean.empty: return df_clean

        results = df_clean.copy()
        results['Risk_Level'] = 'Safe'
        results['Matched_Mass'] = 0.0
        mode_db = self.db.get(mode, {})

        for idx, row in results.iterrows():
            m = row['Mass']
            rm = round(m, 2)

            # 1. Risk0 & Risk1 精确匹配 (0.005 Da 容差)
            matched = False
            for target in mode_db.get('risk0', []):
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk0'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True; break
            if matched: continue

            for target in mode_db.get('risk1_precise', []):
                if abs(m - target) < tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = target
                    matched = True; break
            if matched: continue

            # 2. 约等匹配
            if rm in mode_db.get('risk1_rounded', set()): results.at[idx, 'Risk_Level'] = 'Risk1'
            elif rm in mode_db.get('risk2', set()): results.at[idx, 'Risk_Level'] = 'Risk2'
            elif rm in mode_db.get('risk3', set()): results.at[idx, 'Risk_Level'] = 'Risk3'
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
                s1.append(int1[i]); s2.append(int2[idx])
        if not s1: return 0.0
        v1, v2 = np.array(s1), np.array(s2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))

    def match(self, target_spec, top_k=3):
        matches = []
        for entry in self.library:
            score = self.calculate_cosine(target_spec, entry)
            if score > 0.5: matches.append({'smiles': entry['smiles'], 'score': score})
        return sorted(matches, key=lambda x: x['score'], reverse=True)[:top_k]