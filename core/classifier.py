import os
import logging
import tensorflow as tf
from core.pipeline import MS2FeatureExtractor

# 配置日志
logger = logging.getLogger(__name__)

class MS2Classifier:
    """
    二级质谱危险性分类器
    封装了 TensorFlow 注意力机制模型，用于判定样本是否为阳性（危险）
    """
    def __init__(self, model_path, top_n=10, feature_dim=10):
        """
        :param model_path: .h5 模型文件路径
        :param top_n: 提取的最高峰数量 (对应模型输入维度)
        :param feature_dim: 每个峰对应的特征维度 (对应模型输入维度)
        """
        self.model_path = model_path
        self.top_n = top_n
        self.feature_dim = feature_dim
        self.model = self._load_model()

        # 初始化特征提取管线
        self.extractor = MS2FeatureExtractor(top_n=top_n, feature_dim=feature_dim)

    def _load_model(self):
        """加载 Keras 模型"""
        if not os.path.exists(self.model_path):
            logger.error(f"MS2Classifier: 模型文件不存在: {self.model_path}")
            return None

        try:
            # 加载模型，若包含自定义层可在 custom_objects 中指定
            model = tf.keras.models.load_model(self.model_path, compile=False)
            logger.info(f"MS2Classifier: 成功加载模型 {self.model_path}")
            # 打印模型简要结构
            model.summary(print_fn=lambda x: logger.debug(x))
            return model
        except Exception as e:
            logger.error(f"MS2Classifier: 模型加载失败: {e}")
            return None

    def predict(self, ms2_data_list):
        """
        执行危险性预测
        :param ms2_data_list: MS2 质谱字符串列表 (例如: ["121.1:100,150.2:50", ...])
        :return: (probabilities, predictions)
        """
        if self.model is None:
            logger.warning("MS2Classifier: 模型未就绪，无法预测")
            return None, None

        try:
            # 1. 特征提取 (转换为 10x10 张量)
            # 输出形状: (batch_size, 10, 10)
            features = self.extractor.transform(ms2_data_list)

            # 2. 模型推理
            # y_prob 形状通常为 (batch_size, 1) 或 (batch_size, 2)
            y_prob = self.model.predict(features, verbose=0)

            # 3. 处理输出结果
            # 假设二分类输出，取最后一列作为阳性概率
            if y_prob.shape[1] > 1:
                prob = y_prob[:, 1]
            else:
                prob = y_prob.flatten()

            # 判定标签 (阈值 0.5)
            predictions = (prob >= 0.5).astype(int)

            logger.info(f"MS2Classifier: 完成 {len(ms2_data_list)} 个样本的预测")
            return prob, predictions

        except Exception as e:
            logger.error(f"MS2Classifier: 推理过程中出错: {e}")
            return None, None

    def get_risk_label(self, prob):
        """根据概率值返回描述性的风险标签"""
        if prob >= 0.9:
            return "极高危 (Positive)", "danger"
        elif prob >= 0.5:
            return "高危 (Positive)", "warning"
        elif prob >= 0.2:
            return "可疑 (Borderline)", "info"
        else:
            return "低危/阴性 (Negative)", "success"