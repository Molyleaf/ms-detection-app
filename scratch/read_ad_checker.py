import pickle

with open("data_processed/ad_checker_model.pkl", "rb") as f:
    model_data = pickle.load(f)

for k, v in model_data.items():
    print(k, type(v))
    if isinstance(v, (int, float)):
        print("  Value:", v)
