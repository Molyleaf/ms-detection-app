import joblib
import numpy as np
import pandas as pd
from core.features import build_graph_inputs
from scripts.train_ad import ADDataPreprocessor

def main():
    # Load stats
    stats = joblib.load("data_processed/stats.joblib")
    
    # Initialize train_ad preprocessor
    train_prep = ADDataPreprocessor()
    train_prep.mz_mean = stats["mz_mean"]
    train_prep.mz_std = stats["mz_std"]
    train_prep.max_intensity_mz_mean = stats["max_intensity_mz_mean"]
    train_prep.max_intensity_mz_std = stats["max_intensity_mz_std"]
    
    # Test spectrum string
    # Let's take a line from training data
    df = pd.read_excel("data/化合物-7-1.xlsx")
    test_str = ""
    for idx, row in df.iterrows():
        test_str = str(row.get("MS", ""))
        if test_str and test_str != "nan":
            break
            
    print("Test spectrum string:", test_str)
    
    # Features from core/features.py
    nodes_core, adj_core = build_graph_inputs(test_str, stats)
    
    # Features from scripts/train_ad.py
    nodes_train, adj_train = train_prep._parse_ms_string(test_str)
    
    # Compare
    nodes_core = nodes_core[0]  # Remove batch dim
    adj_core = adj_core[0]      # Remove batch dim
    
    print("\nNodes core shape:", nodes_core.shape)
    print("Nodes train shape:", nodes_train.shape)
    
    nodes_equal = np.allclose(nodes_core, nodes_train, atol=1e-5)
    adj_equal = np.allclose(adj_core, adj_train, atol=1e-5)
    
    print("Nodes equal?", nodes_equal)
    print("Adj equal?", adj_equal)
    
    if not nodes_equal:
        diff_idx = np.where(~np.isclose(nodes_core, nodes_train, atol=1e-5))
        print("Diff index in nodes:", diff_idx)
        print("Core values:\n", nodes_core[diff_idx[0][:5], diff_idx[1][:5]])
        print("Train values:\n", nodes_train[diff_idx[0][:5], diff_idx[1][:5]])
        
if __name__ == "__main__":
    main()
