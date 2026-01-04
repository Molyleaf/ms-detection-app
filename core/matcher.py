# core/matcher.py
import numpy as np
import joblib

class RiskMatcher:
    """一级质谱风险匹配器：对齐 Notebook 展示逻辑"""
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try:
            self.db = joblib.load(db_path)
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

        mode_db = self.db.get(mode.lower(), self.db)
        r1_precise = mode_db.get('risk1_precise', [])
        r1_rounded = mode_db.get('risk1_rounded', set())
        r2 = mode_db.get('risk2', set())
        r3 = mode_db.get('risk3', set())

        for idx, row in results.iterrows():
            m = float(row.get('Mass', row.get('m/z', 0)))
            if m == 0: continue
            rm = round(m, 2)

            # --- 逻辑 1: 精确匹配 (Notebook 的 Risk0 逻辑，现在统一标记为 Risk1) ---
            matched_precise = False
            if r1_precise:
                diffs = [abs(m - target) for target in r1_precise]
                min_idx = np.argmin(diffs)
                if diffs[min_idx] <= tolerance:
                    results.at[idx, 'Risk_Level'] = 'Risk1'  # 修改：标签对齐为 Risk1
                    results.at[idx, 'Matched_Mass'] = r1_precise[min_idx]
                    matched_precise = True

            if matched_precise: continue

            # --- 逻辑 2: 近似匹配 (Risk1) ---
            if rm in r1_rounded:
                results.at[idx, 'Risk_Level'] = 'Risk1'
                candidates = [v for v in r1_precise if round(v, 2) == rm]
                results.at[idx, 'Matched_Mass'] = min(candidates, key=lambda x: abs(x - m)) if candidates else rm
                continue

            # --- 逻辑 3: Risk2 ---
            if rm in r2:
                results.at[idx, 'Risk_Level'] = 'Risk2'
                results.at[idx, 'Matched_Mass'] = rm
                continue

            # --- 逻辑 4: Risk3 ---
            if rm in r3:
                results.at[idx, 'Risk_Level'] = 'Risk3'
                results.at[idx, 'Matched_Mass'] = rm

        return results

class SpectrumMatcher:
    """二级质谱相似度匹配器"""
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