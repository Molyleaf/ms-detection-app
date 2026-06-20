# scripts/convert_ad_to_onnx.py
import os
import pickle
import numpy as np
import onnx
from onnx import helper, TensorProto

def convert_to_onnx(
    pkl_path: str = "data_processed/ad_checker_model.pkl",
    onnx_path: str = "data_processed/ad_checker_model.onnx"
):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到 Pickle 模型文件: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        model_data = pickle.load(f)
    
    pca = model_data["pca"]
    inv_cov = model_data["inv_cov"]
    h_star = model_data["h_star"]
    mz_mean = model_data["mz_mean"]
    mz_std = model_data["mz_std"]
    max_intensity_mz_mean = model_data["max_intensity_mz_mean"]
    max_intensity_mz_std = model_data["max_intensity_mz_std"]

    # Extract model parameters
    mean = pca.mean_.astype(np.float32)            # (200,)
    components_T = pca.components_.T.astype(np.float32) # (200, n_components)
    inv_cov = inv_cov.astype(np.float32)           # (n_components, n_components)
    
    n_features = mean.shape[0]
    n_components = components_T.shape[1]
    
    print(f"Features: {n_features}, Components: {n_components}")
    
    # Create initializer tensors
    init_mean = helper.make_tensor(
        name="mean",
        data_type=TensorProto.FLOAT,
        dims=[1, n_features],
        vals=mean.flatten().tolist()
    )
    
    init_components_T = helper.make_tensor(
        name="components_T",
        data_type=TensorProto.FLOAT,
        dims=[n_features, n_components],
        vals=components_T.flatten().tolist()
    )
    
    # inv_cov matrix
    init_inv_cov = helper.make_tensor(
        name="inv_cov",
        data_type=TensorProto.FLOAT,
        dims=[n_components, n_components],
        vals=inv_cov.flatten().tolist()
    )
    
    # Inputs and Outputs
    # Input X: (1, n_features)
    input_tensor = helper.make_tensor_value_info(
        name="X",
        elem_type=TensorProto.FLOAT,
        shape=[1, n_features]
    )
    
    # Output leverage: (1, 1)
    output_tensor = helper.make_tensor_value_info(
        name="leverage",
        elem_type=TensorProto.FLOAT,
        shape=[1, 1]
    )
    
    # Nodes in computation graph
    # 1. X_centered = Sub(X, mean)
    node_sub = helper.make_node(
        op_type="Sub",
        inputs=["X", "mean"],
        outputs=["X_centered"],
        name="sub_mean"
    )
    
    # 2. X_pca = MatMul(X_centered, components_T) -> shape (1, n_components)
    node_matmul_pca = helper.make_node(
        op_type="MatMul",
        inputs=["X_centered", "components_T"],
        outputs=["X_pca"],
        name="matmul_pca"
    )
    
    # 3. temp = MatMul(X_pca, inv_cov) -> shape (1, n_components)
    node_matmul_inv = helper.make_node(
        op_type="MatMul",
        inputs=["X_pca", "inv_cov"],
        outputs=["temp"],
        name="matmul_inv"
    )
    
    # 4. X_pca_T = Transpose(X_pca, perm=[1, 0]) -> shape (n_components, 1)
    node_transpose = helper.make_node(
        op_type="Transpose",
        inputs=["X_pca"],
        outputs=["X_pca_T"],
        perm=[1, 0],
        name="transpose_pca"
    )
    
    # 5. leverage = MatMul(temp, X_pca_T) -> shape (1, 1)
    node_matmul_lev = helper.make_node(
        op_type="MatMul",
        inputs=["temp", "X_pca_T"],
        outputs=["leverage"],
        name="matmul_lev"
    )
    
    # Build graph
    graph = helper.make_graph(
        nodes=[node_sub, node_matmul_pca, node_matmul_inv, node_transpose, node_matmul_lev],
        name="AD_leverage_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[init_mean, init_components_T, init_inv_cov]
    )
    
    # Make model
    model = helper.make_model(graph, producer_name="ad-converter")
    
    # Add metadata
    metadata = {
        "h_star": str(h_star),
        "mz_mean": str(mz_mean),
        "mz_std": str(mz_std),
        "max_intensity_mz_mean": str(max_intensity_mz_mean),
        "max_intensity_mz_std": str(max_intensity_mz_std),
    }
    
    for k, v in metadata.items():
        meta = model.metadata_props.add()
        meta.key = k
        meta.value = v
        
    # Save ONNX model
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    onnx.save(model, onnx_path)
    print(f"ONNX 模型转换成功: {onnx_path}")

if __name__ == "__main__":
    convert_to_onnx()
