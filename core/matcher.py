# core/matcher.py
import numpy as np
import joblib


class RiskMatcher:
    """
    一级质谱风险匹配器：严格对齐 Notebook 展示逻辑

    Notebook 关键点：
    - Risk0：与 risk1_precise 最近值误差 <= tolerance（0.005Da）
      但“输出展示”仍记为 Risk1（Output Risk = Risk1）
    - Risk1：两位小数命中 risk1_rounded，但与最近 precise 的误差 > tolerance
    - Risk2/Risk3：两位小数集合命中
    """
    def __init__(self, db_path='data_processed/risk_db.joblib'):
        try:
            self.db = joblib.load(db_path)
            if self.db:
                self.db = {k.lower(): v for k, v in self.db.items()}
        except:
            self.db = None

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            v = float(x)
            if np.isnan(v):
                return default
            return v
        except:
            return default

    def match_ms1_peaks(self, df_clean, mode='positive', tolerance=0.005):
        if self.db is None or df_clean is None or df_clean.empty:
            return df_clean

        results = df_clean.copy()

        # 输出列：同时保留“实际风险”和“展示风险”，对齐 Notebook
        results['Actual_Risk'] = 'Safe'
        results['Output_Risk'] = 'Safe'
        results['Matched_Mass'] = 0.0

        mode_db = self.db.get(str(mode).lower(), self.db)

        r1_precise = list(mode_db.get('risk1_precise', []))
        r1_rounded = mode_db.get('risk1_rounded', set())

        # 用于避免 round 碰撞漏匹配：rounded -> 多个 precise 值
        # convert.py 会生成 risk1_rounded_to_precise（若旧库没有，这里也能回退）
        r1_round_map = mode_db.get('risk1_rounded_to_precise', None)
        if not isinstance(r1_round_map, dict):
            r1_round_map = None

        r2 = mode_db.get('risk2', set())
        r3 = mode_db.get('risk3', set())

        for idx, row in results.iterrows():
            m = self._safe_float(row.get('Mass', row.get('m/z', 0.0)), default=0.0)
            if m == 0.0:
                continue

            rm = round(m, 2)

            # 1) Risk0（实际）：精确阈值命中 risk1_precise
            if r1_precise:
                diffs = np.abs(np.array(r1_precise, dtype=float) - m)
                min_pos = int(np.argmin(diffs))
                min_diff = float(diffs[min_pos])

                if min_diff <= tolerance:
                    matched = float(r1_precise[min_pos])
                    results.at[idx, 'Actual_Risk'] = 'Risk0'
                    results.at[idx, 'Output_Risk'] = 'Risk1'  # 关键：展示对齐 Notebook
                    results.at[idx, 'Matched_Mass'] = matched
                    continue

            # 2) Risk1（实际）：两位小数命中 risk1_rounded，但误差 > tolerance
            if rm in r1_rounded:
                candidates = None

                if r1_round_map is not None:
                    # 重要：map 的 key 在 joblib 里可能是 float，也可能是 str
                    candidates = r1_round_map.get(rm)
                    if candidates is None:
                        candidates = r1_round_map.get(str(rm))
                else:
                    # 回退：从 r1_precise 过滤所有 round 后相同的候选（能覆盖 round 碰撞）
                    candidates = [v for v in r1_precise if round(float(v), 2) == rm]

                matched_mass = rm
                diff = float('inf')
                if candidates:
                    matched_mass = float(min(candidates, key=lambda x: abs(float(x) - m)))
                    diff = abs(m - matched_mass)

                # Notebook 的定义：Risk1 需要确保不是 Risk0（即 diff > tolerance）
                if diff > tolerance:
                    results.at[idx, 'Actual_Risk'] = 'Risk1'
                    results.at[idx, 'Output_Risk'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = matched_mass
                    continue
                else:
                    # 极少数情况下：两位小数命中但其实也在 tolerance 内
                    results.at[idx, 'Actual_Risk'] = 'Risk0'
                    results.at[idx, 'Output_Risk'] = 'Risk1'
                    results.at[idx, 'Matched_Mass'] = matched_mass
                    continue

            # 3) Risk2：两位小数命中
            if rm in r2:
                results.at[idx, 'Actual_Risk'] = 'Risk2'
                results.at[idx, 'Output_Risk'] = 'Risk2'
                results.at[idx, 'Matched_Mass'] = rm
                continue

            # 4) Risk3：两位小数命中
            if rm in r3:
                results.at[idx, 'Actual_Risk'] = 'Risk3'
                results.at[idx, 'Output_Risk'] = 'Risk3'
                results.at[idx, 'Matched_Mass'] = rm
                continue

            # 5) Safe
            results.at[idx, 'Actual_Risk'] = 'Safe'
            results.at[idx, 'Output_Risk'] = 'Safe'
            results.at[idx, 'Matched_Mass'] = 0.0

        return results


class SpectrumMatcher:
    """二级质谱相似度匹配器"""
    def __init__(self, db_path='data_processed/spectrum_db.joblib'):
        try:
            self.library = joblib.load(db_path)
        except:
            self.library = []

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
        if not s1:
            return 0.0
        v1, v2 = np.array(s1), np.array(s2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))

    def match(self, target_spec,