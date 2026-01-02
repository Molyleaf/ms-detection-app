import pandas as pd
import joblib
import os
import logging
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataConverter:
    def __init__(self, raw_data_dir='original_data', processed_data_dir='data_processed'):
        self.raw_dir = raw_data_dir
        self.processed_dir = processed_data_dir
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def convert_risk_db(self, filename='risk_matching-new.xlsx'):
        """转换一级质谱风险数据库"""
        logger.info(f"开始转换风险库: {filename}")
        input_path = os.path.join(self.raw_dir, filename)

        if not os.path.exists(input_path):
            logger.error(f"找不到原始文件: {input_path}")
            return

        # 读取所有 Sheet
        xlsx = pd.ExcelFile(input_path)
        risk_map = {}
        for sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            # 假设列名为 'Mass' 或 'mz'
            mz_col = 'Mass' if 'Mass' in df.columns else df.columns[0]
            # 提取 MZ 列表并去重，转换为 numpy 数组加速后续匹配
            risk_map[sheet_name] = np.sort(df[mz_col].dropna().unique())

        output_path = os.path.join(self.processed_dir, 'risk_db.joblib')
        joblib.dump(risk_map, output_path)
        logger.info(f"风险库转换完成，保存至: {output_path}")

    def convert_spectral_library(self, filename='ku.txt'):
        """转换二级质谱谱图库 (ku.txt)"""
        logger.info(f"开始转换二级谱图库: {filename}")
        input_path = os.path.join(self.raw_dir, filename)

        if not os.path.exists(input_path):
            logger.error(f"找不到原始文件: {input_path}")
            return

        library_data = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue

                    smiles = parts[0]
                    peaks_str = parts[1]

                    # 解析质谱字符串: mz1:int1,mz2:int2...
                    mz_list = []
                    int_list = []
                    for p in peaks_str.split(','):
                        if ':' in p:
                            m, i = p.split(':')
                            mz_list.append(float(m))
                            int_list.append(float(i))

                    library_data.append({
                        'smiles': smiles,
                        'mz': np.array(mz_list),
                        'intensities': np.array(int_list)
                    })
        except Exception as e:
            logger.error(f"解析库文件时出错: {e}")

        output_path = os.path.join(self.processed_dir, 'spectrum_library.joblib')
        joblib.dump(library_data, output_path)
        logger.info(f"二级谱图库转换完成，保存至: {output_path}")

if __name__ == "__main__":
    converter = DataConverter()
    converter.convert_risk_db()
    converter.convert_spectral_library()