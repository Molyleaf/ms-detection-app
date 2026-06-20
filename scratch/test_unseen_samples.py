import joblib
import pandas as pd
import numpy as np
from core.onnx_infer import ONNXClassifier
from core.features import build_graph_inputs
from sklearn.model_selection import train_test_split

def main():
    stats_path = "data_processed/stats.joblib"
    classifier = ONNXClassifier(model_path="models/model.onnx", stats_joblib=stats_path)
    df = pd.read_excel("data/化合物-7-1.xlsx")
    
    # Preprocess labels
    labels_cleaned = []
    for val in df['label']:
        if pd.isna(val):
            labels_cleaned.append(0)
        else:
            str_val = str(val).strip().lower()
            if str_val in ['0', '0.0', '非那非', '否', 'negative', 'n', 'false', 'f', '非']:
                labels_cleaned.append(0)
            elif str_val in ['1', '1.0', '那非', '是', 'positive', 'y', 'true', 't', '是']:
                labels_cleaned.append(1)
            else:
                try:
                    num_val = float(str_val)
                    labels_cleaned.append(int(num_val) if num_val in [0, 1] else 0)
                except:
                    labels_cleaned.append(0)
    df["label_cleaned"] = labels_cleaned
    
    # Split as in train_ad.py
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=df["label_cleaned"],
        random_state=42
    )
    
    train_val_labels = df["label_cleaned"].iloc[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25, # 0.2 / 0.8
        stratify=train_val_labels,
        random_state=42
    )
    
    # We will test on validation set (val_idx) and test set (test_idx)
    test_df = df.iloc[test_idx]
    
    print("Testing on unseen test set samples:")
    print("Index | True Label | Raw Prob | Pred Label | Pred Prob")
    print("-" * 65)
    
    probs = []
    count = 0
    for idx, row in test_df.iterrows():
        ms_str = str(row.get("MS", ""))
        if not ms_str or ms_str == "nan":
            continue
            
        nodes, adj = build_graph_inputs(ms_str, classifier.stats)
        feed = {
            classifier.input_names[0]: nodes.astype(np.float32),
            classifier.input_names[1]: adj.astype(np.float32),
        }
        out = classifier.sess.run(classifier.output_names, feed)
        raw_prob = float(np.asarray(out[0]).reshape(-1)[0])
        
        pred = classifier.predict_from_peaks(ms_str)
        probs.append(pred["probability"])
        
        if count < 20:
            print(f"{idx:5d} | {row.get('label_cleaned'):10d} | {raw_prob:8.6f} | {pred['label']:10s} | {pred['probability']:8.6f}")
        count += 1
        
    probs = np.array(probs)
    print("\nTest set prediction probability stats:")
    print("Min:", np.min(probs))
    print("Max:", np.max(probs))
    print("Mean:", np.mean(probs))
    print("Samples in [0.70, 0.95]:", np.sum((probs >= 0.70) & (probs <= 0.95)))
    print("Total test samples:", len(probs))

if __name__ == "__main__":
    main()
