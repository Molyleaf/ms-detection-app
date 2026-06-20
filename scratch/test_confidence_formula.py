import pickle
import numpy as np

def main():
    with open("data_processed/ad_checker_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        
    pca = model_data["pca"]
    X_train_pca = model_data["X_train_pca"]
    inv_cov = model_data["inv_cov"]
    
    # Calculate explained variance dynamically
    explained_variance = sum(pca.explained_variance_ratio_)
    print("Explained variance ratio sum:", explained_variance)
    
    # Training leverages
    train_leverages = np.sum((X_train_pca @ inv_cov) * X_train_pca, axis=1)
    
    # Calculate confidence probabilities using the formula: explained_variance - leverage
    confidences = explained_variance - train_leverages
    
    print("\nTraining set confidence stats:")
    print("Min:", np.min(confidences))
    print("Max:", np.max(confidences))
    print("Mean:", np.mean(confidences))
    print("Median:", np.median(confidences))
    print("Samples in [0.70, 0.95]:", np.sum((confidences >= 0.70) & (confidences <= 0.95)))
    print("Total training samples:", len(confidences))

if __name__ == "__main__":
    main()
