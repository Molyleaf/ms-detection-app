import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
from core.onnx_infer import ONNXClassifier
from core.features import build_graph_inputs

def main():
    h5_path = "models/251229.h5"
    onnx_path = "models/model.onnx"
    stats_path = "data_processed/stats.joblib"
    
    # Load models
    h5_model = tf.keras.models.load_model(h5_path, compile=False)
    onnx_classifier = ONNXClassifier(model_path=onnx_path, stats_joblib=stats_path)
    
    stats = joblib.load(stats_path)
    df = pd.read_excel("data/化合物-7-1.xlsx")
    
    count = 0
    print("Idx | H5 output | ONNX output | Diff")
    print("-" * 45)
    for idx, row in df.iterrows():
        ms_str = str(row.get("MS", ""))
        if not ms_str or ms_str == "nan":
            continue
            
        nodes, adj = build_graph_inputs(ms_str, stats)
        
        # H5 prediction
        # Feed inputs to H5 model. In keras it takes: [nodes, adj]
        # nodes shape in build_graph_inputs is (1, 10, 10)
        # adj shape in build_graph_inputs is (1, 10, 10)
        h5_out = h5_model.predict([nodes, adj], verbose=0)
        h5_prob = float(h5_out[0][0])
        
        # ONNX prediction
        feed = {
            onnx_classifier.input_names[0]: nodes.astype(np.float32),
            onnx_classifier.input_names[1]: adj.astype(np.float32),
        }
        onnx_out = onnx_classifier.sess.run(onnx_classifier.output_names, feed)
        onnx_prob = float(np.asarray(onnx_out[0]).reshape(-1)[0])
        
        diff = abs(h5_prob - onnx_prob)
        print(f"{idx:3d} | {h5_prob:9.7f} | {onnx_prob:9.7f} | {diff:9.7f}")
        
        count += 1
        if count >= 10:
            break

if __name__ == "__main__":
    main()
