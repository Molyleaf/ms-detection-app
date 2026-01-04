# scripts/convert.py
import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../data_processed')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_risk_db():
    """
    逻辑：读取 risk_matching-1.xlsx

    对齐 Notebook/避免 round 碰撞漏匹配：
    - risk1_precise: list[float] 保留原始精确值
    - risk1_rounded: set[float] 仅用于快速判断“是否命中两位小数”
    - risk1_rounded_to_precise: dict[rounded -> list[precise]] 用于从 rounded 反查所有候选精确值
      （关键：round 碰撞时不会被覆盖）
    """
    excel_path = os.path.join(DATA_DIR, 'risk_matching-1.xlsx')
    output_path = os.path.join(OUTPUT_DIR, 'risk_db.joblib')
    if not os.path.exists(excel_path):
        return

    db = {
        'positive': {
            'risk1_precise': [],
            'risk1_rounded': set(),
            'risk1_rounded_to_precise': defaultdict(list),
            'risk2': set(),
            'risk3': set()
        },
        'negative': {
            'risk1_precise': [],
            'risk1_rounded': set(),
            'risk1_rounded_to_precise': defaultdict(list),
            'risk2': set(),
            'risk3': set()
        }
    }

    xls = pd.ExcelFile(excel_path)
    # Notebook：风险0 实际属于 risk1 的“精确阈值命中”，因此这里合并进 risk1 数据源
    sheet_map = {'风险1': 'risk1', '风险2': 'risk2', '风险3': 'risk3', '风险0': 'risk1'}

    for sheet_name in xls.sheet_names:
        mapped_key = sheet_map.get(sheet_name)
        if not mapped_key:
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name)

        for mode, cols in [
            ('positive', ['[M+H]+', '[M+Na]+', '[M+K]+']),
            ('negative', ['[M-H]-'])
        ]:
            for col in cols:
                if col not in df.columns:
                    continue

                masses = df[col].dropna().astype(float).tolist()
                for m in masses:
                    if mapped_key == 'risk1':
                        m = float(m)
                        rm = round(m, 2)

                        db[mode]['risk1_precise'].append(m)
                        db[mode]['risk1_rounded'].add(rm)
                        db[mode]['risk1_rounded_to_precise'][rm].append(m)
                    else:
                        db[mode][mapped_key].add(round(float(m), 2))

    # defaultdict 无法直接 joblib 友好展示，这里转成普通 dict
    for mode in ['positive', 'negative']:
        db[mode]['risk1_rounded_to_precise'] = dict(db[mode]['risk1_rounded_to_precise'])

    ensure_dir(OUTPUT_DIR)
    joblib.dump(db, output_path)
    print(f"✅ 风险库已构建（risk1_rounded=set + rounded->precise 列表），共处理 {len(xls.sheet_names)} 个表单。")


def build_spectrum_db():
    """彻底取消清洗逻辑，原模原样保留 ku.txt 的所有峰"""
    txt_path = os.path.join(DATA_DIR, 'ku.txt')
    output_path = os.path.join(OUTPUT_DIR, 'spectrum_db.joblib')
    if not os.path.exists(txt_path):
        return

    library = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                peaks = [p.split(':') for p in parts[1].replace(';', ',').split(',') if ':' in p]
                mzs = np.array([float(p[0]) for p in peaks], dtype=float)
                ints = np.array([float(p[1]) for p in peaks], dtype=float)
                if len(ints) > 0:
                    ints = (ints / np.max(ints)) * 100.0
                library.append({'smiles': name, 'mz': mzs, 'intensities': ints})
            except:
                continue

    joblib.dump(library, output_path)
    print("✅ 谱图库已原模原样转换完毕。")


def update_stats():
    spec_path = os.path.join(OUTPUT_DIR, 'spectrum_db.joblib')
    output_path = os.path.join(OUTPUT_DIR, 'stats.joblib')
    if not os.path.exists(spec_path):
        return
    lib = joblib.load(spec_path)
    all_mz = [mz for entry in lib for mz in entry['mz']]
    max_mzs = [entry['mz'][np.argmax(entry['intensities'])] for entry in lib]
    stats = {
        'mz_mean': float(np.mean(all_mz)),
        'mz_std': float(np.std(all_mz)),
        'max_intensity_mz_mean': float(np.mean(max_mzs)),
        'max_intensity_mz_std': float(np.std(max_mzs))
    }
    joblib.dump(stats, output_path)
    print("✅ 统计数据更新完成。")


if __name__ == '__main__':
    build_risk_db()
    build_spectrum_db()
    update_stats()