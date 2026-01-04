# core/onnx_infer.py
import os
import joblib
import numpy as np
import onnxruntime as ort

from core.features import build_graph_inputs, characteristic_rule_trigger


class ONNXClassifier:
    def __init__(
            self,
            model_path: str = "models/model.onnx",
            stats_joblib: str = "data_processed/stats.joblib",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 ONNX 模型: {model_path}")
        if not os.path.exists(stats_joblib):
            raise FileNotFoundError(f"找不到 stats.joblib: {stats_joblib}")

        self.stats = joblib.load(stats_joblib)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def predict_from_peaks(self, peaks: str):
        # 规则触发：直接阳性
        if characteristic_rule_trigger(peaks):
            return {"label": "Positive", "probability": 1.0, "via": "rule"}

        nodes, adj = build_graph_inputs(peaks, self.stats, max_nodes=10, node_dim=10)
        feed = {
            self.input_names[0]: nodes.astype(np.float32),
            self.input_names[1]: adj.astype(np.float32),
        }
        out = self.sess.run(self.output_names, feed)
        prob = float(np.asarray(out[0]).reshape(-1)[0])

        # notebook 输出逻辑：若有任意阳性则 Positive，否则输出 1-avg_prob；
        # 这里我们按单样本：prob>0.5 => Positive，否则 Negative 且概率=1-prob
        if prob > 0.5:
            return {"label": "Positive", "probability": prob, "via": "onnx"}
        return {"label": "Negative", "probability": 1.0 - prob, "via": "onnx"}