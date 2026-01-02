import os
import logging
import tensorflow as tf
from core.pipeline import MS2GraphExtractor

logger = logging.getLogger(__name__)

class MS2Classifier:
    """
    对齐 训练.ipynb 的注意力机制判别逻辑。
    负责将 MS2 文本转换为图特征并进行模型预测。
    """
    def __init__(self, model_path='models/251229.h5', stats_path='data_processed/stats.joblib'):
        self.model_path = model_path
        self.stats_path = stats_path

        # 初始化 10 维特征提取器
        self.extractor = MS2GraphExtractor(max_nodes=10, node_dim=10, stats_path=stats_path)

        # 加载模型 (compile=False 避免 Docker 内环境依赖问题)
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info(f"MS2Classifier: 成功加载模型 {model_path}")
        else:
            self.model = None
            logger.error(f"MS2Classifier: 未找到模型文件 {model_path}")

    def predict(self, ms2_content_list):
        """
        输入 MS2 字符串列表，返回概率和标签。
        :param ms2_content_list: list of strings (e.g., ["58.06:100,72.08:50..."])
        """
        if self.model is None:
            return [0.0] * len(ms2_content_list), [0] * len(ms2_content_list)

        # 1. 调用 Pipeline 进行 10 维特征提取
        # 返回格式为 [Batch_Nodes, Batch_Adjs]
        inputs = self.extractor.transform(ms2_content_list)

        # 2. 模型推理
        try:
            probs = self.model.predict(inputs, verbose=0).flatten()
            preds = (probs > 0.5).astype(int)
            return probs.tolist(), preds.tolist()
        except Exception as e:
            logger.error(f"模型推理失败: {e}")
            return [0.0], [0]

    def get_risk_label(self, prob):
        """
        根据概率值转换风险等级描述
        """
        if prob > 0.9:
            return "极高风险 (阳性)", "danger"
        elif prob > 0.5:
            return "高风险 (阳性)", "warning"
        elif prob > 0.2:
            return "疑似 (阴性)", "info"
        else:
            return "低风险 (阴性)", "success"

    def check_risk0_bypass(self, ms1_risk_level, ms2_max_mz, matched_mass):
        """
        对齐 qlc.ipynb 逻辑：如果是 Risk0 且母离子质量高度吻合，直接绕过模型判定为阳性。
        """
        if ms1_risk_level == 'Risk0':
            # 如果二级谱图的最大峰（母离子）与库中精确质量误差极小
            if abs(ms2_max_mz - matched_mass) < 0.0001:
                logger.info("触发 Risk0 快速通道，判定为阳性。")
                return True
        return False