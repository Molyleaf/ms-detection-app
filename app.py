# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

app = Flask(__name__)

# 初始化全局组件
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')
    if not file: return "未选择文件", 400
    try:
        df = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
        # 执行预处理流水线
        df_clean = get_ms1_pipeline().fit_transform(df)
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)
        return render_template('results_ms1.html', peaks=results.to_dict(orient='records'), mode=mode)
    except Exception as e:
        return f"MS1 处理失败: {e}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """二级质谱深度判定路由"""
    risk_level = request.form.get('risk_level')
    matched_mass = float(request.form.get('matched_mass', 0))
    ms2_data = request.form.get('ms2_data')

    # 支持上传 MS2 Excel 或文本
    if 'ms2_file' in request.files:
        file = request.files['ms2_file']
        if file and file.filename != '':
            df_ms2 = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
            m_col = 'Mass' if 'Mass' in df_ms2.columns else df_ms2.columns[0]
            i_col = 'Intensity' if 'Intensity' in df_ms2.columns else df_ms2.columns[1]
            ms2_data = ",".join([f"{r[m_col]}:{r[i_col]}" for _, r in df_ms2.iterrows()])

    if not ms2_data: return "请提供二级质谱数据", 400

    # 1. 执行 Risk0 旁路逻辑
    if CLASSIFIER.check_risk0_bypass(risk_level, ms2_data, matched_mass):
        prob, pred = 1.0, 1
    else:
        # 2. 执行模型推理与特征峰硬规则
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 3. 阳性样本回溯谱图库匹配
    library_matches = []
    if pred == 1:
        raw_peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
        spec_obj = {
            'mz': np.array([float(p[0]) for p in raw_peaks]),
            'intensities': np.array([float(p[1]) for p in raw_peaks])
        }
        library_matches = SPEC_MATCHER.match(spec_obj)

    return render_template('results_ms2.html',
                           prob=round(prob * 100, 2),
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=library_matches)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)