# core/classifier.py
import os
import logging
import numpy as np
import onnxruntime as ort
from core.pipeline import MS2GraphExtractor

logger = logging.getLogger(__name__)

class MS2Classifier:
    def __init__(self, model_path='models/model.onnx', stats_path='data_processed/stats.joblib'):
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)

        if os.path.exists(model_path):
            # 优化推理设置
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options)
            self.input_names = [i.name for i in self.session.get_inputs()]
            logger.info(f"MS2Classifier: 加载 ONNX 模型 {model_path}, 输入节点: {self.input_names}")
        else:
            self.session = None
            logger.error(f"MS2Classifier: 模型文件 {model_path} 不存在")

    def predict(self, ms2_content_list):
        if self.session is None:
            return [0.0], [0]

        # 1. 提取特征 (返回节点特征和邻接矩阵)
        nodes, adjs = self.extractor.transform(ms2_content_list)

        # 2. 准备 ONNX 输入 (确保为 float32)
        # 假设 ONNX 模型输入顺序为: nodes_input, adj_input
        inputs = {
            self.input_names[0]: nodes.astype(np.float32),
            self.input_names[1]: adjs.astype(np.float32)
        }

        try:
            outputs = self.session.run(None, inputs)
            probs = outputs[0].flatten()
            preds = (probs > 0.5).astype(int)
            return probs.tolist(), preds.tolist()
        except Exception as e:
            logger.error(f"ONNX 推理失败: {e}")
            return [0.0] * len(ms2_content_list), [0] * len(ms2_content_list)

    def get_risk_label(self, prob):
        """对齐 qlc.ipynb 输出逻辑"""
        if prob > 0.9: return "极高风险 (阳性)", "danger"
        elif prob > 0.5: return "高风险 (阳性)", "warning"
        elif prob > 0.2: return "疑似 (阴性)", "info"
        else: return "低风险 (阴性)", "success"

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """对齐 qlc.ipynb 的 Risk0 旁路逻辑"""
        if ms1_risk_level == 'Risk0':
            # 如果母离子精确匹配数据库中的风险值，直接判定
            if abs(ms2_max_mz - matched_mass) < 0.0001:
                return True
        return False