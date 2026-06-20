import pandas as pd
import numpy as np
import os
from core.onnx_infer import ONNXClassifier
from core.features import build_graph_inputs

def main():
    train_xlsx_path = "data/化合物-7-1.xlsx"
    classifier = ONNXClassifier(model_path="models/model.onnx", stats_joblib="data_processed/stats.joblib")
    df = pd.read_excel(train_xlsx_path)
    
    print("Index | Label | Raw Prob | Predicted Label | Predicted Prob")
    print("-" * 65)
    count = 0
    for idx, row in df.iterrows():
        label = row.get("label")
        if label != 0:
            continue
            
        ms_str = str(row.get("MS", ""))
        if pd.isna(ms_str) or ms_str == "nan" or ms_str.strip() == "":
            continue
        
        pred = classifier.predict_from_peaks(ms_str)
        nodes, adj = build_graph_inputs(ms_str, classifier.stats, max_nodes=10, node_dim=10)
        feed = {
            classifier.input_names[0]: nodes.astype(np.float32),
            classifier.input_names[1]: adj.astype(np.float32),
        }
        out = classifier.sess.run(classifier.output_names, feed)
        raw_prob = float(np.asarray(out[0]).reshape(-1)[0])
        
        print(f"{idx:5d} | {label:5.1f} | {raw_prob:8.6f} | {pred['label']:15s} | {pred['probability']:8.6f}")
        count += 1
        if count >= 30:
            break

if __name__ == "__main__":
    main()
