# core/classifier.py
import os
import numpy as np
import onnxruntime as ort
from core.pipeline import MS2GraphExtractor

class MS2Classifier:
    def __init__(self, model_path='models/model.onnx', stats_path='data_processed/stats.joblib'):
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)

        if os.path.exists(model_path):
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options)
            # 获取模型实际定义的输入名称
            self.input_names = [i.name for i in self.session.get_inputs()]
        else:
            self.session = None

    def predict(self, ms2_content_list):
        if self.session is None: return [0.0], [0]

        nodes, adjs = self.extractor.transform(ms2_content_list)

        # 映射输入 (严格对应 convert_to_onnx.py)
        inputs = {
            "node_input": nodes.astype(np.float32),
            "adj_input": adjs.astype(np.float32)
        }

        try:
            outputs = self.session.run(None, inputs)
            probs = outputs[0].flatten()
            preds = (probs > 0.5).astype(int)
            return probs.tolist(), preds.tolist()
        except Exception as e:
            print(f"ONNX推理失败: {e}")
            return [0.0], [0]

    def get_risk_label(self, prob):
        """对齐 qlc.ipynb 的判定阈值"""
        if prob > 0.9: return "极高风险 (阳性)", "danger"
        elif prob > 0.5: return "高风险 (阳性)", "warning"
        elif prob > 0.2: return "疑似 (阴性)", "info"
        else: return "低风险 (阴性)", "success"

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """Risk0 旁路逻辑：母离子精确匹配则直接判定"""
        if ms1_risk_level == 'Risk0' and matched_mass > 0:
            if abs(ms2_max_mz - matched_mass) < 0.0001:
                return True
        return False