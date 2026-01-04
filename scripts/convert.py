# convert.py
import pandas as pd
import numpy as np
import joblib
import os
import re
from core.pipeline import MS1Cleaner

def parse_ms_string(ms_str):
    """将 mz:int,mz:int 格式的字符串解析为 (mz_arr, int_arr)"""
    try:
        peaks = [p.split(':') for p in str(ms_str).replace(';', ',').split(',') if ':' in p]
        if not peaks: return np.array([]), np.array([])
        mzs = np.array([float(p[0]) for p in peaks])
        ints = np.array([float(p[1]) for p in peaks])
        return mzs, ints
    except:
        return np.array([]), np.array([])

def clean_spectrum(mzs, ints):
    """执行与 Notebook 一致的二级质谱清洗逻辑：归一化 -> 同位素清洗 -> 强度过滤"""
    if len(mzs) == 0: return mzs, ints

    # 1. 归一化 (0-100)
    max_i = np.max(ints)
    if max_i > 0:
        ints = (ints / max_i) * 100.0

    # 2. 贪婪同位素清洗 (2Da)
    # 按强度降序排列进行抑制
    sort_idx = np.argsort(ints)[::-1]
    mzs_s, ints_s = mzs[sort_idx], ints[sort_idx]

    keep = np.ones(len(mzs_s), dtype=bool)
    for i in range(len(mzs_s)):
        if not keep[i]: continue
        for j in range(i + 1, len(mzs_s)):
            if keep[j] and abs(mzs_s[j] - mzs_s[i]) <= 2.0:
                keep[j] = False

    # 3. 过滤强度 < 1.0 且恢复质量排序
    final_mzs = mzs_s[keep]
    final_ints = ints_s[keep]

    mask = final_ints >= 1.0
    final_mzs, final_ints = final_mzs[mask], final_ints[mask]

    order = np.argsort(final_mzs)
    return final_mzs[order], final_ints[order]

def save_risk_db(excel_path='../data/risk_matching-1.xlsx', output_path='data_processed/risk_db.joblib'):
    """预处理风险数据库：提取各级风险 Mass 并转换为集合或列表"""
    print(f"正在转换风险库: {excel_path}...")
    if not os.path.exists(excel_path):
        print("❌ 未找到风险库文件")
        return

    # 定义模式映射 (根据 Notebook 逻辑)
    db = {'positive': {}, 'negative': {}}

    # 假设 Excel 中通过 Sheet 或列区分正负离子，此处演示标准逻辑
    xls = pd.ExcelFile(excel_path)
    # 映射逻辑：Risk0 和 Risk1 Precise 使用列表（精确匹配），其余使用 round(2) 集合（模糊匹配）
    sheet_map = {
        '风险0': 'risk0',
        '风险1': 'risk1',
        '风险2': 'risk2',
        '风险3': 'risk3'
    }

    for mode in ['positive', 'negative']:
        mode_data = {}
        for sheet_name, key in sheet_map.items():
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                if 'Mass' in df.columns:
                    masses = df['Mass'].dropna().tolist()
                    if key == 'risk0':
                        mode_data['risk0'] = masses
                    elif key == 'risk1':
                        mode_data['risk1_precise'] = masses
                        mode_data['risk1_rounded'] = set(round(m, 2) for m in masses)
                    else:
                        mode_data[key] = set(round(m, 2) for m in masses)
        db[mode] = mode_data

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(db, output_path)
    print(f"✅ 风险库已保存至: {output_path}")

def save_spectrum_db(training_data_path='../data/化合物-7.xlsx', output_path='data_processed/spectrum_db.joblib'):
    """预处理化合物二级质谱库：解析、清洗并存储为高效列表"""
    print(f"正在转换质谱库: {training_data_path}...")
    if not os.path.exists(training_data_path):
        print("❌ 未找到化合物数据库")
        return

    df = pd.read_excel(training_data_path)
    library = []

    for i, row in df.iterrows():
        ms_str = str(row.get('MS', ''))
        mzs, ints = parse_ms_string(ms_str)

        # 严格执行 Notebook 的清洗逻辑
        clean_mzs, clean_ints = clean_spectrum(mzs, ints)

        if len(clean_mzs) > 0:
            library.append({
                'id': i,
                'name': str(row.get('Name', f'Unknown_{i}')),
                'smiles': str(row.get('SMILES', 'N/A')),
                'mz': clean_mzs,
                'intensities': clean_ints
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(library, output_path)
    print(f"✅ 质谱库已保存 (共 {len(library)} 个条目) -> {output_path}")

def save_global_stats(training_data_path='../data/化合物-7.xlsx', output_path='data_processed/stats.joblib'):
    """从训练集提取 MZ 统计量，确保与 MS2GraphExtractor 逻辑一致"""
    print("正在生成统计量 (stats.joblib)...")
    df = pd.read_excel(training_data_path)
    all_mz = []
    all_max_intensity_mz = []

    for _, row in df.iterrows():
        mzs, ints = parse_ms_string(row.get('MS', ''))
        if len(mzs) == 0: continue

        all_mz.extend(mzs)
        all_max_intensity_mz.append(mzs[np.argmax(ints)])

    stats = {
        'mz_mean': float(np.mean(all_mz)),
        'mz_std': float(np.std(all_mz)),
        'max_intensity_mz_mean': float(np.mean(all_max_intensity_mz)),
        'max_intensity_mz_std': float(np.std(all_max_intensity_mz))
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(stats, output_path)
    print(f"✅ 统计量已更新: {stats}")

def convert_txt_to_xlsx(txt_path):
    """辅助功能：将常见的文本格式质谱数据转为标准的 Excel 格式"""
    # 识别常见的文本分隔符（空格、制表符、逗号）
    try:
        df = pd.read_csv(txt_path, sep=r'\s+|,', engine='python', names=['Mass', 'Intensity'])
        output_xlsx = txt_path.rsplit('.', 1)[0] + '.xlsx'
        df.to_excel(output_xlsx, index=False)
        print(f"已将 {txt_path} 转换为 {output_xlsx}")
        return output_xlsx
    except Exception as e:
        print(f"转换 {txt_path} 失败: {e}")
        return None

if __name__ == '__main__':
    # 确保运行环境目录存在
    os.makedirs('data_processed', exist_ok=True)

    # 1. 转换风险库 (L1匹配用)
    save_risk_db('../data/risk_matching-1.xlsx')

    # 2. 转换质谱库 (L2回溯匹配用)
    save_spectrum_db('../data/化合物-7.xlsx')

    # 3. 生成特征工程统计量 (模型推理用)
    save_global_stats('../data/化合物-7.xlsx')

    print("\n🚀 所有数据已转换为二进制格式。")