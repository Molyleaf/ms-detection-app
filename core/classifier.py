import os
import logging
import tensorflow as tf
from core.pipeline import MS2GraphExtractor

logger = logging.getLogger(__name__)

class MS2Classifier:
    """对齐 Notebook 的 L2 判别逻辑"""
    def __init__(self):
        # 从环境变量读取模型和统计路径
        self.model_path = os.getenv('MODEL_PATH', 'models/251229.h5')
        self.stats_path = os.getenv('STATS_PATH', 'data_processed/stats.joblib')

        self.extractor = MS2GraphExtractor(max_nodes=10, stats_path=self.stats_path)
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f"模型不存在: {self.model_path}")
            return None
        return tf.keras.models.load_model(self.model_path, compile=False)

    def check_risk0(self, risk_db_results, current_ms2_max_mz):
        """
        Risk0 逻辑：如果母离子质量与风险库 Risk0 记录精确匹配 (0.0001)，
        直接判定为 Positive。
        """
        for _, row in risk_db_results.iterrows():
            if row.get('Risk') == 'Risk0':
                if abs(row['Mass'] - current_ms2_max_mz) < 0.0001:
                    logger.info("匹配到 Risk0 精确质量，触发即时阳性判定")
                    return True
        return False

    def predict(self, ms2_str, risk_context_df=None):
        """
        :param ms2_str: 二级质谱字符串
        :param risk_context_df: 一级质谱识别出的风险列表 (用于 Risk0 校验)
        """
        # 1. 提取基础峰信息
        raw_peaks = [p.split(':') for p in ms2_str.split(',') if ':' in p]
        if not raw_peaks: return 0.0, "Invalid Data", "secondary"

        mzs = [float(p[0]) for p in raw_peaks]
        max_mz = max(mzs)

        # 2. Risk0 快速通道
        if risk_context_df is not None:
            if self.check_risk0(risk_context_df, max_mz):
                return 1.0, "极高危 (Risk0 精确匹配)", "danger"

        # 3. 模型深度预测
        if self.model is None:
            return 0.0, "模型未加载", "secondary"

        X_graph = self.extractor.transform([ms2_str]) # 返回 [nodes, adj]
        prob = float(self.model.predict(X_graph, verbose=0).flatten()[0])

        if prob >= 0.5:
            return prob, "那非类阳性 (Positive)", "danger"
        else:
            return prob, "阴性 (Negative)", "success"