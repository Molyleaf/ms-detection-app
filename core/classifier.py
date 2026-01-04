# core/classifier.py
import numpy as np
import onnxruntime as ort
from core.pipeline import MS2GraphExtractor

class MS2Classifier:
    def __init__(self, model_path='models/model.onnx', stats_path='data_processed/stats.joblib'):
        self.extractor = MS2GraphExtractor(stats_path=stats_path)
        try:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path)
        except: self.session = None

    def check_characteristic_peaks_rule(self, ms_str):
        """核心碎片组合硬匹配逻辑"""
        try:
            peaks = [float(p.split(':')[0]) for p in str(ms_str).replace(';', ',').split(',') if ':' in p]
            # 常见那非类核心碎片
            hard_peaks = [166.1, 283.1, 312.2, 354.2]
            count = sum(1 for hp in hard_peaks if any(abs(p - hp) < 0.1 for p in peaks))
            return count >= 3 # 满足3个及以上直接判定
        except: return False

    def check_risk0_bypass(self, risk_level, ms2_data, matched_mass):
        """Risk0 旁路：对齐 Notebook 0.005 Da 判定"""
        if risk_level == 'Risk0' and matched_mass > 0:
            try:
                peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
                int_arr = np.array([float(p[1]) for p in peaks])
                mz_arr = np.array([float(p[0]) for p in peaks])
                max_mz = mz_arr[np.argmax(int_arr)]
                # 若二级最大峰与一级匹配质量误差 < 0.005，视为确认
                if abs(max_mz - matched_mass) < 0.005:
                    return True
            except: pass
        return False

    def predict(self, ms2_list):
        if not self.session: return [0.0], [0]
        probs, preds = [], []
        for ms_str in ms2_list:
            # 1. 硬规则优先
            if self.check_characteristic_peaks_rule(ms_str):
                probs.append(1.0); preds.append(1); continue

            # 2. 模型推理
            nodes, adjs = self.extractor.transform([ms_str])
            out = self.session.run(None, {
                "node_input": nodes.astype(np.float32),
                "adj_input": adjs.astype(np.float32)
            })
            p = float(out[0].flatten()[0])
            probs.append(p); preds.append(1 if p > 0.5 else 0)
        return probs, preds

    def get_risk_label(self, prob):
        if prob > 0.8: return "极高风险 (Positive)", "danger"
        if prob > 0.5: return "高风险 (Suspect)", "warning"
        return "低风险 (Negative)", "success"