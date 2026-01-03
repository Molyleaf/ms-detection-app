# convert.py
import pandas as pd
import numpy as np
import joblib
import os

def save_global_stats(training_data_path='../data/化合物-7.xlsx'):
    """从训练集提取 MZ 统计量，确保与 训练.ipynb 逻辑一致"""
    if not os.path.exists(training_data_path):
        print("未找到训练集，无法生成统计量")
        return

    df = pd.read_excel(training_data_path)
    all_mz = []
    all_max_intensity_mz = []

    for _, row in df.iterrows():
        ms_str = str(row.get('MS', ''))
        peaks = [p.split(':') for p in ms_str.split(',') if ':' in p]
        if not peaks: continue

        mzs = [float(p[0]) for p in peaks]
        ints = [float(p[1]) for p in peaks]

        all_mz.extend(mzs)
        # 提取最大强度对应的 MZ
        max_idx = np.argmax(ints)
        all_max_intensity_mz.append(mzs[max_idx])

    stats = {
        'mz_mean': float(np.mean(all_mz)),
        'mz_std': float(np.std(all_mz)),
        'max_intensity_mz_mean': float(np.mean(all_max_intensity_mz)),
        'max_intensity_mz_std': float(np.std(all_max_intensity_mz))
    }

    os.makedirs('data_processed', exist_ok=True)
    joblib.dump(stats, '../data_processed/stats.joblib')
    print(f"统计量已保存: {stats}")

if __name__ == "__main__":
    # convert_risk_database() # 保持原有逻辑
    save_global_stats()