import os
import logging
import numpy as np
import onnxruntime as ort
from core.pipeline import MS2GraphExtractor

logger = logging.getLogger(__name__)

class MS2Classifier:
    """
    使用 ONNX Runtime 进行推理。
    对齐 训练.ipynb 的注意力机制判别逻辑。
    """
    def __init__(self, model_path='models/251229.onnx', stats_path='data_processed/stats.joblib'):
        self.model_path = model_path
        self.stats_path = stats_path

        # 初始化 10 维特征提取器
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)

        # 加载 ONNX 模型会话
        if os.path.exists(self.model_path):
            # 默认使用 CPU 执行提供程序，若有 GPU 环境可配置为 ['CUDAExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.input_names = [i.name for i in self.session.get_inputs()]
            logger.info(f"MS2Classifier: 成功加载 ONNX 模型 {model_path}，输入节点: {self.input_names}")
        else:
            self.session = None
            logger.error(f"MS2Classifier: 未找到模型文件 {model_path}")

    def predict(self, ms2_content_list):
        """
        输入 MS2 字符串列表，返回概率和标签。
        """
        if self.session is None:
            return [0.0] * len(ms2_content_list), [0] * len(ms2_content_list)

        # 1. 特征提取：返回 [Batch_Nodes, Batch_Adjs]
        nodes_batch, adjs_batch = self.extractor.transform(ms2_content_list)

        # 2. 准备 ONNX 输入字典
        # 假设模型转换时保留了 Keras 的输入名称: 'node_input' 和 'adj_input'
        # 注意：数据类型必须严格匹配 float32
        onnx_inputs = {
            self.input_names[0]: nodes_batch.astype(np.float32),
            self.input_names[1]: adjs_batch.astype(np.float32)
        }

        try:
            # 3. 执行推理
            outputs = self.session.run(None, onnx_inputs)
            probs = outputs[0].flatten() # 假设输出是单维度概率
            preds = (probs > 0.5).astype(int)
            return probs.tolist(), preds.tolist()
        except Exception as e:
            logger.error(f"ONNX 推理失败: {e}")
            return [0.0] * len(ms2_content_list), [0] * len(ms2_content_list)

    def get_risk_label(self, prob):
        if prob > 0.9: return "极高风险 (阳性)", "danger"
        elif prob > 0.5: return "高风险 (阳性)", "warning"
        elif prob > 0.2: return "疑似 (阴性)", "info"
        else: return "低风险 (阴性)", "success"

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """Risk0 精确质量直判逻辑"""
        if ms1_risk_level == 'Risk0' and abs(ms2_max_mz - matched_mass) < 0.0001:
            logger.info("触发 Risk0 快速通道。")
            return True
        return False