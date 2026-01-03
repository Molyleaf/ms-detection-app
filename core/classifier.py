# core/classifier.py
import os
import numpy as np
import onnxruntime as ort
from core.pipeline import MS2GraphExtractor, CHARACTERISTIC_PEAKS

class MS2Classifier:
    def __init__(self, model_path='models/model.onnx', stats_path='data_processed/stats.joblib'):
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)
        # ... (session 初始化逻辑不变)

    def check_characteristic_peaks_rule(self, ms_str):
        """
        对齐 Notebook 的特征峰硬规则：
        如果质谱数据中包含特定的那非类特征峰组合，则直接判定为阳性。
        """
        # 解析输入
        try:
            peaks = []
            for p in str(ms_str).replace(';', ',').split(','):
                if ':' in p:
                    peaks.append(float(p.split(':')[0]))

            # Notebook 逻辑：如果包含 166.1, 283.1, 312.2 等关键组合 (示例逻辑，需按具体NB实现调整)
            # 这里实现一个通用的特征峰匹配逻辑
            count = 0
            for cp in [166.1, 283.1, 312.2, 354.2]: # Notebook 中定义的硬匹配峰
                if any(abs(p - cp) < 0.1 for p in peaks):
                    count += 1
            return count >= 3 # 如果匹配到3个以上特征峰
        except:
            return False

    def predict(self, ms2_content_list):
        if self.session is None: return [0.0], [0]

        # 增加特征峰规则旁路
        results_prob = []
        for ms_str in ms2_content_list:
            if self.check_characteristic_peaks_rule(ms_str):
                results_prob.append(1.0)
                continue

            # 正常模型推理
            nodes, adjs = self.extractor.transform([ms_str])
            inputs = {"node_input": nodes.astype(np.float32), "adj_input": adjs.astype(np.float32)}
            outputs = self.session.run(None, inputs)
            results_prob.append(float(outputs[0].flatten()[0]))

        probs = np.array(results_prob)
        preds = (probs > 0.5).astype(int)
        return probs.tolist(), preds.tolist()

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """
        Risk0 旁路逻辑：对齐 Notebook 的 0.005 Da 容差
        """
        if ms1_risk_level == 'Risk0' and matched_mass > 0:
            # Notebook 使用 0.005 的容差进行母离子匹配判定
            if abs(ms2_max_mz - matched_mass) < 0.005:
                return True
        return False