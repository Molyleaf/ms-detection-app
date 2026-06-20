import pickle
import numpy as np

def main():
    with open("data_processed/ad_checker_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    X_train_pca = model_data["X_train_pca"]
    inv_cov = model_data["inv_cov"]
    h_star = model_data["h_star"]
    
    # Calculate training leverages
    train_leverages = np.sum((X_train_pca @ inv_cov) * X_train_pca, axis=1)
    print("Train leverages shape:", train_leverages.shape)
    print("Min:", np.min(train_leverages))
    print("Max:", np.max(train_leverages))
    print("Mean:", np.mean(train_leverages))
    print("Median:", np.median(train_leverages))
    print("90th percentile:", np.percentile(train_leverages, 90))
    print("95th percentile:", np.percentile(train_leverages, 95))
    print("99th percentile:", np.percentile(train_leverages, 99))
    
    # Let's test a query leverage of 0.0769
    query_lev = 0.0769
    percentile = np.sum(train_leverages <= query_lev) / len(train_leverages)
    print(f"\nQuery leverage {query_lev} is at percentile: {percentile:.4f}")
    
if __name__ == "__main__":
    main()
