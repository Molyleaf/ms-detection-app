import onnxruntime as ort
import numpy as np
from core.features import build_graph_inputs
import joblib

def main():
    sess = ort.InferenceSession("models/model.onnx")
    stats = joblib.load("data_processed/stats.joblib")
    
    # Let's take a sample peak string
    peaks = "58.0651:100.0,72.0808:50.0"
    nodes, adj = build_graph_inputs(peaks, stats)
    
    feed = {
        sess.get_inputs()[0].name: nodes.astype(np.float32),
        sess.get_inputs()[1].name: adj.astype(np.float32),
    }
    
    # Run with both the final output and the logit output
    outputs = ["dense_728", "model_80/dense_728/BiasAdd:0"]
    res = sess.run(outputs, feed)
    print("Probability:", res[0])
    print("Logit:", res[1])

if __name__ == "__main__":
    main()
