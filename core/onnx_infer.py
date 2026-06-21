# core/onnx_infer.py
import os

import numpy as np
import onnxruntime as ort

from core.features import build_graph_inputs, characteristic_rule_trigger


class ONNXClassifier:
    def __init__(
            self,
            model_path: str = "models/model.onnx",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 ONNX 模型: {model_path}")

        # 性能/稳定性：限制 ORT 线程数，避免 gunicorn 多 worker 下线程过度订阅
        intra = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", "1"))
        inter = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(intra, 1)
        so.inter_op_num_threads = max(inter, 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    # @ai-intent Predict MS2 sample class and probability using ONNX model.
    # @ai-invariant Output probability MUST match raw sigmoid output P_GNN.
    # @ai-invariant Output probability MUST be within [0.0, 1.0].
    # @ai-boundary Read-only peaks string. No local file write.
    # @ai-context
    #   ContextData:
    #     Domain: core/onnx_infer.py
    #     Trigger: predict_from_peaks
    #     Return: Label (Positive/Negative), Probability (P_GNN)
    def predict_from_peaks(
            self,
            peaks: str,
            mz_mean: float | None = None,
            mz_std: float | None = None,
            max_intensity_mz_mean: float | None = None,
            max_intensity_mz_std: float | None = None,
    ):
        nodes, adj = build_graph_inputs(
            peaks,
            max_nodes=10,
            node_dim=10,
            mz_mean=mz_mean,
            mz_std=mz_std,
            max_intensity_mz_mean=max_intensity_mz_mean,
            max_intensity_mz_std=max_intensity_mz_std,
        )
        feed = {
            self.input_names[0]: nodes.astype(np.float32),
            self.input_names[1]: adj.astype(np.float32),
        }
        out = self.sess.run(self.output_names, feed)
        prob = float(np.asarray(out[0]).reshape(-1)[0])

        # 引入温度缩放 (Temperature Scaling) 校准置信度
        t_env = os.getenv("Temperature", "1.0")
        try:
            T = float(t_env)
        except ValueError:
            T = 1.0

        if T > 0 and T != 1.0:
            # 限制概率在 [1e-7, 1 - 1e-7] 防止数学溢出
            p_clipped = np.clip(prob, 1e-7, 1.0 - 1e-7)
            # 计算 Logit (反 Sigmoid 变换)
            z = np.log(p_clipped / (1.0 - p_clipped))
            # 重新计算概率
            prob = float(1.0 / (1.0 + np.exp(-z / T)))

        # 单样本：prob>0.5 => Positive，否则 Negative 且直接返回原始概率 prob
        if prob > 0.5:
            return {"label": "Positive", "probability": prob, "via": "onnx"}
        return {"label": "Negative", "probability": prob, "via": "onnx"}
