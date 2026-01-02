import os
import pandas as pd
import numpy as np
import joblib
import json
import logging
import tensorflow as tf
from safetensors.tensorflow import save_file

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_model_to_safetensors(h5_path, out_dir):
    """
    将 Keras .h5 模型转换为 safetensors 权重 + JSON 架构
    """
    if not os.path.exists(h5_path):
        logger.error(f"模型文件 {h5_path} 不存在，跳过转换")
        return

    logger.info(f"正在转换模型: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)

    # 1. 保存权重为 safetensors
    weights = {v.name: v for v in model.weights}
    save_file(weights, os.path.join(out_dir, "model_weights.safetensors"))

    # 2. 保存架构定义 (用于重建模型)
    model_json = model.to_json()
    with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        f.write(model_json)

    logger.info("模型转换完成：model_weights.safetensors & model_config.json")

def convert_risk_database():
    """
    转换风险数据库：合并 1, 2, 3 级风险
    """
    logger.info("开始转换风险数据库...")
    processed_db = {
        'positive': {'risk0': [], 'risk1_precise': [], 'risk1_rounded': set(), 'risk2': set(), 'risk3': set()},
        'negative': {'risk0': [], 'risk1_precise': [], 'risk1_rounded': set(), 'risk2': set(), 'risk3': set()}
    }

    # 对应的 CSV 文件名 (根据上传的文件名)
    files = {
        1: 'data/risk_matching-new-new-new.xlsx - 风险1.csv',
        2: 'data/risk_matching-new-new-new.xlsx - 风险2.csv',
        3: 'data/risk_matching-new-new-new.xlsx - 风险3.csv'
    }

    for level, file_path in files.items():
        if not os.path.exists(file_path):
            logger.warning(f"缺失文件: {file_path}")
            continue

        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            mass = row['Exact_Mass']
            # 这里简化逻辑，Risk1 存入精确匹配，2和3存入四舍五入集合
            if level == 1:
                processed_db['positive']['risk1_precise'].append(mass)
                processed_db['positive']['risk1_rounded'].add(round(mass, 1))
            elif level == 2:
                processed_db['positive']['risk2'].add(round(mass, 1))
            elif level == 3:
                processed_db['positive']['risk3'].add(round(mass, 1))

    joblib.dump(processed_db, 'data_processed/risk_db.joblib')
    logger.info("风险数据库已保存到 data_processed/risk_db.joblib")

def convert_spectrum_library():
    """
    转换 ku.txt 谱图库
    """
    logger.info("开始转换谱图库...")
    lib_entries = []
    input_path = 'data/ku.txt'

    if not os.path.exists(input_path):
        logger.warning(f"{input_path} 不存在")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            smiles = parts[0]
            ms_str = parts[1]
            try:
                peaks = [p.split(':') for p in ms_str.split(',') if ':' in p]
                mz = np.array([float(p[0]) for p in peaks])
                intensities = np.array([float(p[1]) for p in peaks])
                # 归一化强度
                if len(intensities) > 0:
                    intensities = (intensities / intensities.max()) * 100
                lib_entries.append({'smiles': smiles, 'mz': mz, 'intensities': intensities})
            except: continue

    joblib.dump(lib_entries, 'data_processed/spectrum_db.joblib')
    logger.info(f"谱图库转换完成，共 {len(lib_entries)} 条记录。")

def save_global_stats():
    """
    保存对齐 训练.ipynb 的统计量
    """
    stats = {
        'max_mz_mean': 450.0, # 根据训练数据实际情况填写
        'max_mz_std': 150.0,
        'feature_version': '1.0'
    }
    joblib.dump(stats, 'data_processed/stats.joblib')
    logger.info("全局统计量 stats.joblib 已保存。")

if __name__ == "__main__":
    # 创建目录
    os.makedirs('data_processed', exist_ok=True)

    # 1. 转换数据库
    convert_risk_database()
    convert_spectrum_library()
    save_global_stats()

    # 2. 转换模型
    convert_model_to_safetensors('models/251229.h5', 'data_processed')