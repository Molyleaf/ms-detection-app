import os
import pandas as pd
import numpy as np
import joblib
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_ms_string(ms_string):
    """
    解析质谱字符串 (格式如 mass1:int1,mass2:int2)
    提取自 notebook 的 MS_SMILES_Matcher 逻辑
    """
    try:
        ms_string = str(ms_string).strip().replace('"', '').replace("'", "")
        if ';' in ms_string:
            peaks = ms_string.split(';')
        elif ',' in ms_string:
            peaks = ms_string.split(',')
        else:
            peaks = ms_string.split()

        mz_list = []
        intensity_list = []

        for peak in peaks:
            peak = peak.strip()
            if not peak: continue

            parts = peak.split(':') if ':' in peak else peak.split()
            if len(parts) >= 2:
                try:
                    mz_list.append(float(parts[0].strip()))
                    intensity_list.append(float(parts[1].strip()))
                except ValueError:
                    continue

        if not mz_list:
            return None

        # 归一化强度到 0-100 (基于最大峰)
        max_int = max(intensity_list)
        if max_int > 0:
            intensity_list = [i / max_int * 100 for i in intensity_list]

        return {"mz": np.array(mz_list), "intensities": np.array(intensity_list)}
    except Exception as e:
        logger.error(f"解析质谱字符串失败: {e}")
        return None

def convert_risk_database(input_path, output_path):
    """
    转换风险匹配数据库 (Excel -> Joblib)
    包含正/负离子模式下的精确值和两位小数近似值
    """
    logger.info(f"正在转换风险数据库: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"找不到原始风险数据库: {input_path}")
        return

    xls = pd.ExcelFile(input_path)
    risk_db = {
        'positive': {'risk1_precise': [], 'risk1_rounded': set(), 'risk2': set(), 'risk3': set()},
        'negative': {'risk1_precise': [], 'risk1_rounded': set(), 'risk2': set(), 'risk3': set()}
    }

    pos_cols = ['[M+H]+', '[M+Na]+', '[M+K]+']
    neg_cols = ['[M-H]-']

    for sheet_name in ['风险1', '风险2', '风险3']:
        if sheet_name not in xls.sheet_names:
            logger.warning(f"跳过不存在的工作表: {sheet_name}")
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)

        for mode, target_cols in [('positive', pos_cols), ('negative', neg_cols)]:
            available_cols = [c for c in target_cols if c in df.columns]
            if not available_cols: continue

            # 提取所有数值并清洗
            all_values = pd.to_numeric(df[available_cols].values.flatten(), errors='coerce')
            all_values = all_values[~np.isnan(all_values)]

            if sheet_name == '风险1':
                risk_db[mode]['risk1_precise'] = all_values.tolist()
                risk_db[mode]['risk1_rounded'] = set(np.round(all_values, 2))
            elif sheet_name == '风险2':
                risk_db[mode]['risk2'] = set(np.round(all_values, 2))
            elif sheet_name == '风险3':
                risk_db[mode]['risk3'] = set(np.round(all_values, 2))

    joblib.dump(risk_db, output_path)
    logger.info(f"风险数据库转换完成，保存至: {output_path}")

def convert_spectrum_library(input_path, output_path):
    """
    转换化合物库 (ku.txt -> Joblib)
    """
    logger.info(f"正在转换谱图库: {input_path}")
    if not os.path.exists(input_path):
        logger.error(f"找不到原始谱图库: {input_path}")
        return

    library_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue

            parts = line.split('\t')
            if len(parts) >= 2:
                smiles = parts[0].strip()
                ms_data_str = parts[1].strip()
                spec = parse_ms_string(ms_data_str)
                if spec:
                    spec['smiles'] = smiles
                    spec['id'] = i + 1
                    library_data.append(spec)

    joblib.dump(library_data, output_path)
    logger.info(f"谱图库转换完成，共加载 {len(library_data)} 条记录，保存至: {output_path}")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs('data_processed', exist_ok=True)

    # 执行转换
    convert_risk_database('data/risk_matching-new-new-new.xlsx', 'data_processed/risk_db.joblib')
    convert_spectrum_library('data/ku.txt', 'data_processed/spectrum_db.joblib')

    logger.info("所有预处理任务已完成。")