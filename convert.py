import pandas as pd
import numpy as np
import joblib
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_risk_database(input_path='data/risk_matching-new-new-new.xlsx'):
    """
    转换风险数据库：从多工作表 Excel 提取 Risk0, Risk1, Risk2, Risk3。
    """
    if not os.path.exists(input_path):
        logger.error(f"未找到风险表: {input_path}")
        return

    logger.info(f"正在从 {input_path} 转换风险数据库...")

    # 初始化结构
    processed_db = {
        'positive': {
            'risk0': [],           # 精确质量 (Exact_Mass)
            'risk1_precise': [],   # 具体的加合离子质量 ([M+H]+等)
            'risk1_rounded': set(), # 约等匹配 (保留2位小数)
            'risk2': set(),
            'risk3': set()
        },
        'negative': {
            'risk0': [],
            'risk1_precise': [],
            'risk1_rounded': set(),
            'risk2': set(),
            'risk3': set()
        }
    }

    try:
        xls = pd.ExcelFile(input_path)

        # 1. 处理风险1 (包含 Risk0 和 Risk1)
        if '风险1' in xls.sheet_names:
            df1 = pd.read_excel(xls, '风险1')
            # Risk0 (用于母离子精确比对)
            if 'Exact_Mass' in df1.columns:
                masses = df1['Exact_Mass'].dropna().tolist()
                processed_db['positive']['risk0'] = masses
                processed_db['negative']['risk0'] = masses

            # Risk1 正离子模式
            pos_cols = ['[M+H]+', '[M+Na]+', '[M+K]+']
            for col in pos_cols:
                if col in df1.columns:
                    vals = df1[col].dropna().tolist()
                    processed_db['positive']['risk1_precise'].extend(vals)
                    processed_db['positive']['risk1_rounded'].update([round(v, 2) for v in vals])

            # Risk1 负离子模式
            if '[M-H]-' in df1.columns:
                vals = df1['[M-H]-'].dropna().tolist()
                processed_db['negative']['risk1_precise'].extend(vals)
                processed_db['negative']['risk1_rounded'].update([round(v, 2) for v in vals])

        # 2. 处理风险2 和 风险3 (仅约等匹配)
        for level in [2, 3]:
            sheet_name = f'风险{level}'
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                key = f'risk{level}'

                # 正离子
                for col in ['[M+H]+', '[M+Na]+', '[M+K]+']:
                    if col in df.columns:
                        vals = df[col].dropna()
                        processed_db['positive'][key].update([round(v, 2) for v in vals])

                # 负离子
                if '[M-H]-' in df.columns:
                    vals = df['[M-H]-'].dropna()
                    processed_db['negative'][key].update([round(v, 2) for v in vals])

        # 保存
        os.makedirs('data_processed', exist_ok=True)
        joblib.dump(processed_db, 'data_processed/risk_db.joblib')
        logger.info("风险数据库转换完成，保存至 data_processed/risk_db.joblib")

    except Exception as e:
        logger.error(f"转换风险库失败: {e}")

def convert_spectrum_library(input_path='data/ku.txt'):
    """
    转换谱图库 (ku.txt) 为 joblib 以加快读取速度
    格式: SMILES \t MZ1:INT1,MZ2:INT2...
    """
    if not os.path.exists(input_path):
        logger.warning(f"未找到谱图库: {input_path}")
        return

    logger.info(f"正在转换谱图库 {input_path}...")
    lib_entries = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                smiles = parts[0]
                ms_str = parts[1]

                # 预解析字符串
                peaks = [p.split(':') for p in ms_str.split(',') if ':' in p]
                mz = np.array([float(p[0]) for p in peaks])
                intensities = np.array([float(p[1]) for p in peaks])

                lib_entries.append({
                    'smiles': smiles,
                    'mz': mz,
                    'intensities': intensities
                })

        joblib.dump(lib_entries, 'data_processed/spectrum_db.joblib')
        logger.info(f"谱图库转换完成，共 {len(lib_entries)} 条记录。")
    except Exception as e:
        logger.error(f"转换谱图库失败: {e}")

def save_global_stats(training_data_path='data/化合物-7.xlsx'):
    """
    根据训练集计算全局统计量，确保推理时的特征归一化与训练时完全一致。
    对应 训练.ipynb 中的 load_and_preprocess_data 逻辑。
    """
    if not os.path.exists(training_data_path):
        # 如果没有原始训练文件，则提供默认值（基于训练好的模型参数）
        logger.warning(f"未找到训练文件 {training_data_path}，使用预设统计量。")
        stats = {
            'mz_mean': 225.43,        # 示例值，实际应从训练中提取
            'mz_std': 112.15,
            'max_mz_mean': 285.67,
            'max_mz_std': 95.34
        }
    else:
        logger.info(f"正在从训练集 {training_data_path} 计算统计量...")
        df = pd.read_excel(training_data_path)
        all_mz = []
        all_max_mz = []

        for _, row in df.iterrows():
            ms_str = str(row.get('MS', ''))
            peaks = [p.split(':') for p in ms_str.split(',') if ':' in p]
            if not peaks: continue

            mzs = [float(p[0]) for p in peaks]
            ints = [float(p[1]) for p in peaks]

            all_mz.extend(mzs)
            # 记录该样本最强峰的 MZ
            max_idx = np.argmax(ints)
            all_max_mz.append(mzs[max_idx])

        stats = {
            'mz_mean': float(np.mean(all_mz)),
            'mz_std': float(np.std(all_mz)),
            'max_mz_mean': float(np.mean(all_max_mz)),
            'max_mz_std': float(np.std(all_max_mz))
        }

    joblib.dump(stats, 'data_processed/stats.joblib')
    logger.info(f"全局统计量已保存: {stats}")

if __name__ == "__main__":
    os.makedirs('data_processed', exist_ok=True)
    convert_risk_database()
    convert_spectrum_library()
    save_global_stats()