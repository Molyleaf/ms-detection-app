import joblib
import pandas as pd
import numpy as np
from core.onnx_infer import ONNXClassifier
from core.ad_checker import load_or_train_ad_checker
from scratch.test_correct_features import build_graph_inputs_qlc

def main():
    stats_path = "data_processed/stats.joblib"
    classifier = ONNXClassifier(model_path="models/model.onnx", stats_joblib=stats_path)
    
    ad_checker = load_or_train_ad_checker(model_path="data_processed/ad_checker_model.pkl")
    
    df = pd.read_excel("data/化合物-7-1.xlsx")
    
    print("Edge samples (0.07 <= leverage <= 0.165):")
    print("Index | Leverage | Raw GNN Prob | Correct QLC GNN Prob")
    print("-" * 60)
    
    count = 0
    for idx, row in df.iterrows():
        ms_str = str(row.get("MS", ""))
        if not ms_str or ms_str == "nan":
            continue
            
        # Calculate leverage using ONNX ad_checker session
        feature_vector = ad_checker.extract_feature_vector(ms_str).astype(np.float32)
        feed = {ad_checker.input_name: feature_vector.reshape(1, -1)}
        out = ad_checker.sess.run([ad_checker.output_name], feed)
        leverage = float(np.asarray(out[0]).reshape(-1)[0])
        
        if 0.07 <= leverage <= 0.165:
            # GNN prob with current features
            from core.features import build_graph_inputs
            nodes_curr, adj_curr = build_graph_inputs(ms_str, classifier.stats)
            feed_curr = {
                classifier.input_names[0]: nodes_curr.astype(np.float32),
                classifier.input_names[1]: adj_curr.astype(np.float32),
            }
            out_curr = classifier.sess.run(classifier.output_names, feed_curr)
            prob_curr = float(np.asarray(out_curr[0]).reshape(-1)[0])
            
            # GNN prob with correct qlc features
            nodes_qlc, adj_qlc = build_graph_inputs_qlc(ms_str, classifier.stats)
            feed_qlc = {
                classifier.input_names[0]: nodes_qlc.astype(np.float32),
                classifier.input_names[1]: adj_qlc.astype(np.float32),
            }
            out_qlc = classifier.sess.run(classifier.output_names, feed_qlc)
            prob_qlc = float(np.asarray(out_qlc[0]).reshape(-1)[0])
            
            print(f"{idx:5d} | {leverage:8.6f} | {prob_curr:12.10f} | {prob_qlc:12.10f}")
            count += 1
            if count >= 30:
                break

if __name__ == "__main__":
    main()
