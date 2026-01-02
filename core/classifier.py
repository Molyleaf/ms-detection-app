import os
import logging
import onnxruntime as ort
import numpy as np
from core.pipeline import MS2GraphExtractor

logger = logging.getLogger(__name__)

class MS2Classifier:
    """
    使用 onnxruntime 运行推理
    特征提取逻辑与之前保持一致，确保 10 维特征输入
    """
    def __init__(self, model_path='models/model.onnx', stats_path='data_processed/stats.joblib'):
        self.model_path = model_path
        self.stats_path = stats_path

        # 初始化 10 维特征提取器
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)

        if os.path.exists(self.model_path):
            # 启动推理会话
            self.session = ort.InferenceSession(self.model_path)
            # 获取输入名称
            self.input_names = [i.name for i in self.session.get_inputs()]
            logger.info(f"MS2Classifier: 成功加载 ONNX 模型 {model_path}")
        else:
            self.session = None
            logger.error(f"MS2Classifier: 未找到模型文件 {model_path}")

    def predict(self, ms2_content_list):
        if self.session is None:
            return [0.0] * len(ms2_content_list), [0] * len(ms2_content_list)

        # 1. 特征提取，返回 [Batch_Nodes, Batch_Adjs]
        nodes, adjs = self.extractor.transform(ms2_content_list)

        # 转换为 float32 格式
        nodes = nodes.astype(np.float32)
        adjs = adjs.astype(np.float32)

        # 2. ONNX 推理
        try:
            # 构造输入字典，顺序需匹配
            # 这里的输入名称应与转换时指定的 node_input, adj_input 一致
            inputs = {
                self.input_names[0]: nodes,
                self.input_names[1]: adjs
            }
            outputs = self.session.run(None, inputs)
            probs = outputs[0].flatten()
            preds = (probs > 0.5).astype(int)
            return probs.tolist(), preds.tolist()
        except Exception as e:
            logger.error(f"ONNX 推理失败: {e}")
            return [0.0], [0]

    def get_risk_label(self, prob):
        if prob > 0.9: return "极高风险 (阳性)", "danger"
        elif prob > 0.5: return "高风险 (阳性)", "warning"
        elif prob > 0.2: return "疑似 (阴性)", "info"
        else: return "低风险 (阴性)", "success"

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """Risk0 快速通道逻辑"""
        if ms1_risk_level == 'Risk0':
            if abs(ms2_max_mz - matched_mass) < 0.0001:
                return True
        return False