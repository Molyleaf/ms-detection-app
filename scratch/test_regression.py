# scratch/test_regression.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 将当前目录加入系统路径
sys.path.append(os.getcwd())

from core.onnx_infer import ONNXClassifier
from core.features import build_graph_inputs

def test_regression():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=== 开始运行 GNN 直出概率定义回归测试 ===")
    
    classifier = ONNXClassifier(model_path="models/model.onnx", stats_joblib="data_processed/stats.joblib")
    
    # 读取原始数据集以获取样本进行比对
    df = pd.read_excel("data/化合物-7-1.xlsx")
    
    # 过滤出合规的 MS2 peaks 样本
    valid_samples = []
    for idx, row in df.iterrows():
        ms_str = str(row.get("MS", ""))
        if not ms_str or ms_str == "nan":
            continue
        valid_samples.append((idx, ms_str))
        
    print(f"共加载到 {len(valid_samples)} 个包含 MS2 的样本")
    
    # 抽样 50 个样本进行比对
    np.random.seed(42)
    sample_indices = np.random.choice(len(valid_samples), min(50, len(valid_samples)), replace=False)
    test_subset = [valid_samples[i] for i in sample_indices]
    
    passed_count = 0
    for original_idx, ms_str in test_subset:
        # 1. 运行 features 提取与 ONNX 会话，直接获取未处理的 raw_prob
        nodes, adj = build_graph_inputs(ms_str, classifier.stats, max_nodes=10, node_dim=10)
        feed = {
            classifier.input_names[0]: nodes.astype(np.float32),
            classifier.input_names[1]: adj.astype(np.float32),
        }
        out = classifier.sess.run(classifier.output_names, feed)
        raw_prob = float(np.asarray(out[0]).reshape(-1)[0])
        
        # 2. 运行 classifier.predict_from_peaks 预测，获取其置信度 probability
        pred = classifier.predict_from_peaks(ms_str)
        pred_label = pred["label"]
        pred_prob = pred["probability"]
        
        # 3. 校验公式
        if raw_prob > 0.5:
            expected_label = "Positive"
            expected_prob = raw_prob
        else:
            expected_label = "Negative"
            expected_prob = 1.0 - raw_prob
            
        assert pred_label == expected_label, f"样本 {original_idx} 类别不匹配！"
        assert abs(pred_prob - expected_prob) < 1e-7, f"样本 {original_idx} 概率公式不匹配！"
        passed_count += 1
        
    print(f"[SUCCESS] 已通过：抽样的 {passed_count} 个样本的概率值与公式定义 100% 对齐。")
    print("=== 测试成功 ===")

if __name__ == "__main__":
    try:
        test_regression()
    except Exception as e:
        print(f"[FAIL] 测试失败: {e}")
        sys.exit(1)
