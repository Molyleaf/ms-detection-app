# app.py
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# 核心模块导入
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 初始化全局组件
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理一级质谱上传"""
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')
    if not file: return "未选择文件", 400

    try:
        # 自动识别格式读取
        df = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)

        # [cite_start]执行 qlc.ipynb 预处理流水线 [cite: 1]
        ms1_pipeline = get_ms1_pipeline()
        df_clean = ms1_pipeline.fit_transform(df)

        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)
        return render_template('results_ms1.html', peaks=results.to_dict(orient='records'), mode=mode)
    except Exception as e:
        logger.error(f"MS1处理失败: {e}")
        return f"解析失败: {e}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """解析二级质谱文件或文本并执行深度判定"""
    parent_mz = float(request.form.get('parent_mz', 0))
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))
    ms2_data = request.form.get('ms2_data') # 文本框内容

    # 新增：优先处理上传的 MS2 Excel 文件
    if 'ms2_file' in request.files:
        file = request.files['ms2_file']
        if file and file.filename != '':
            try:
                df_ms2 = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
                # 寻找可能的列名
                m_col = 'Mass' if 'Mass' in df_ms2.columns else df_ms2.columns[0]
                i_col = 'Intensity' if 'Intensity' in df_ms2.columns else df_ms2.columns[1]

                # 转换为核心逻辑识别的文本格式: mz:int,mz:int
                peaks_list = [f"{row[m_col]}:{row[i_col]}" for _, row in df_ms2.iterrows()]
                ms2_data = ",".join(peaks_list)
            except Exception as e:
                return f"二级质谱文件解析失败: {e}", 400

    if not ms2_data:
        return "请提供二级质谱数据（上传文件或手动输入）", 400

    # 1. 触发快速旁路或模型推理
    is_risk0 = CLASSIFIER.check_risk0_bypass(risk_level, parent_mz, matched_mass)
    if is_risk0:
        prob, pred = 1.0, 1
    else:
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 2. 谱图库回溯匹配
    library_matches = []
    if pred == 1:
        try:
            raw_peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
            spec_obj = {
                'mz': np.array([float(p[0]) for p in raw_peaks]),
                'intensities': np.array([float(p[1]) for p in raw_peaks])
            }
            library_matches = SPEC_MATCHER.match(spec_obj)
        except Exception as e:
            logger.warning(f"谱图库匹配异常: {e}")

    return render_template('results_ms2.html',
                           parent_mz=parent_mz,
                           prob=round(float(prob) * 100, 2),
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=library_matches)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)