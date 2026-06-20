import joblib
import pandas as pd
import numpy as np
from core.onnx_infer import ONNXClassifier
from sklearn.model_selection import train_test_split

# Replicate qlc-0103.ipynb feature extraction
def build_graph_inputs_qlc(peaks_str, stats, max_nodes=10, node_dim=10):
    # Parse peaks
    if not peaks_str:
        return np.zeros((1, max_nodes, node_dim)), np.eye(max_nodes)[None, ...]
    
    items = peaks_str.replace(";", ",").split(",")
    peak_data = []
    max_intensity = -1
    max_intensity_mz = 0
    for it in items:
        it = it.strip()
        if not it:
            continue
        if ":" in it:
            a, b = it.split(":", 1)
            try:
                mz = float(a.strip())
                intensity = float(b.strip())
                if intensity > max_intensity:
                    max_intensity = intensity
                    max_intensity_mz = mz
                peak_data.append((mz, intensity))
            except ValueError:
                continue
        else:
            try:
                mz = float(it)
                intensity = 1.0
                if intensity > max_intensity:
                    max_intensity = intensity
                    max_intensity_mz = mz
                peak_data.append((mz, intensity))
            except ValueError:
                continue
                
    peak_data.sort(key=lambda x: x[1], reverse=True)
    peak_data = peak_data[:max_nodes]
    
    mz_values = [p[0] for p in peak_data]
    
    mz_mean = stats["mz_mean"]
    mz_std = stats["mz_std"]
    mx_mean = stats["max_intensity_mz_mean"]
    mx_std = stats["max_intensity_mz_std"]
    
    nonafi_peaks = [
        58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
        135.0441, 147.0077, 151.0866, 166.0975, 169.076,
        197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
        297.1346, 299.1139, 302.0812, 312.1581, 315.091,
        327.1274, 341.1608, 354.2, 377.1, 396.203
    ]
    peak_groups = {
        'low_mass': [58.0651, 72.0808, 84.0808, 99.0917, 113.1073],
        'middle_mass': [135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709],
        'high_mass': [250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812],
        'very_high_mass': [312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203]
    }
    key_peaks = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
    
    rounded_characteristic = {round(p, 1) for p in nonafi_peaks}
    rounded_key = {round(p, 1) for p in key_peaks}
    group_map = {}
    for name, arr in peak_groups.items():
        for p in arr:
            group_map[round(p, 1)] = name
            
    def _mass_region_feature(rmz):
        g = group_map.get(rmz)
        if g == "low_mass": return 0.25
        if g == "middle_mass": return 0.5
        if g == "high_mass": return 0.75
        if g == "very_high_mass": return 1.0
        return 0.0

    nodes = np.zeros((max_nodes, node_dim), dtype=np.float32)
    for j in range(max_nodes):
        if j < len(peak_data):
            mz = float(peak_data[j][0])
        elif len(peak_data) > 0:
            mz = float(peak_data[-1][0])
        else:
            mz = 0.0
            
        total_peaks = max(len(peak_data), 1)
        mz_norm = (mz - mz_mean) / mz_std
        pos_ratio = j / total_peaks
        is_first = 1.0 if j == 0 else 0.0
        is_last = 1.0 if j == len(peak_data) - 1 else 0.0
        
        rmz = round(mz, 1)
        is_char = 1.0 if rmz in rounded_characteristic else 0.0
        min_diff = min((abs(mz - p) for p in nonafi_peaks), default=100.0)
        char_diff = min_diff / 100.0
        
        mass_region = _mass_region_feature(rmz)
        is_key = 1.0 if rmz in rounded_key else 0.0
        
        if max_intensity_mz > 0:
            max_mz_norm = (max_intensity_mz - mx_mean) / mx_std
            rel_to_max = mz / max_intensity_mz
        else:
            max_mz_norm = 0.0
            rel_to_max = 1.0
            
        feats = [
            mz_norm,
            pos_ratio,
            is_first,
            is_last,
            is_char,
            char_diff,
            mass_region,
            is_key,
            max_mz_norm,
            rel_to_max
        ]
        nodes[j, :] = np.array(feats, dtype=np.float32)
        
    adj = np.eye(max_nodes, dtype=np.float32)
    sigma = 50.0
    for i in range(min(len(mz_values), max_nodes)):
        for j in range(min(len(mz_values), max_nodes)):
            if i == j: continue
            diff = abs(float(mz_values[i]) - float(mz_values[j]))
            adj[i, j] = np.exp(-diff**2 / (2 * sigma**2))
            
    return nodes[None, ...], adj[None, ...]

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
    
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=df["label_cleaned"],
        random_state=42
    )
    
    test_df = df.iloc[test_idx]
    
    print("Testing on unseen test set samples with CORRECT qlc feature extraction:")
    print("Index | True Label | Raw Prob | Pred Label | Pred Prob")
    print("-" * 65)
    
    probs = []
    count = 0
    for idx, row in test_df.iterrows():
        ms_str = str(row.get("MS", ""))
        if not ms_str or ms_str == "nan":
            continue
            
        nodes, adj = build_graph_inputs_qlc(ms_str, classifier.stats)
        feed = {
            classifier.input_names[0]: nodes.astype(np.float32),
            classifier.input_names[1]: adj.astype(np.float32),
        }
        out = classifier.sess.run(classifier.output_names, feed)
        raw_prob = float(np.asarray(out[0]).reshape(-1)[0])
        
        pred_label = "Positive" if raw_prob > 0.5 else "Negative"
        pred_prob = raw_prob if raw_prob > 0.5 else 1.0 - raw_prob
        probs.append(pred_prob)
        
        if count < 20:
            print(f"{idx:5d} | {row.get('label_cleaned'):10d} | {raw_prob:8.6f} | {pred_label:10s} | {pred_prob:8.6f}")
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
