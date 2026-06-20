import pickle
import numpy as np

def main():
    with open("data_processed/ad_checker_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    pca = model_data["pca"]
    X_train_pca = model_data["X_train_pca"]
    inv_cov = model_data["inv_cov"]
    h_star = model_data["h_star"]
    
    explained_variance = sum(pca.explained_variance_ratio_)
    print("Explained variance ratio sum:", explained_variance)
    print("h_star:", h_star)
    
    # Calculate training leverages
    train_leverages = np.sum((X_train_pca @ inv_cov) * X_train_pca, axis=1)
    
    # Formula: confidence = explained_variance * (0.70 / explained_variance) ** (leverage / (1.5 * h_star))
    confidences = explained_variance * (0.70 / explained_variance) ** (train_leverages / (1.5 * h_star))
    
    print("\nTraining set confidence stats with new formula:")
    print("Min:", np.min(confidences))
    print("Max:", np.max(confidences))
    print("Mean:", np.mean(confidences))
    print("Median:", np.median(confidences))
    print("90th percentile:", np.percentile(confidences, 90))
    print("95th percentile:", np.percentile(confidences, 95))
    print("99th percentile:", np.percentile(confidences, 99))
    print("Samples in [0.70, 0.95]:", np.sum((confidences >= 0.70) & (confidences <= 0.95)))
    print("Total training samples:", len(confidences))

if __name__ == "__main__":
    main()
