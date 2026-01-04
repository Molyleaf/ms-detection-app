# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

app = Flask(__name__)

RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')
    if not file: return "未选择文件", 400
    try:
        df = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)
        df_clean = get_ms1_pipeline().fit_transform(df)
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)
        return render_template('results_ms1.html', peaks=results.to_dict(orient='records'), mode=mode)
    except Exception as e: return f"处理失败: {e}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    risk_level = request.form.get('risk_level')
    matched_mass = float(request.form.get('matched_mass', 0))
    ms2_data = request.form.get('ms2_data', '')

    if 'ms2_file' in request.files:
        f = request.files['ms2_file']
        if f and f.filename != '':
            df = pd.read_excel(f) if f.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(f)
            m_col, i_col = df.columns[0], df.columns[1]
            ms2_data = ",".join([f"{r[m_col]}:{r[i_col]}" for _, r in df.iterrows()])

    if not ms2_data: return "无数据", 400

    # 1. 提取最大峰 MZ
    peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
    mz_arr = np.array([float(p[0]) for p in peaks])
    int_arr = np.array([float(p[1]) for p in peaks])
    max_mz = mz_arr[np.argmax(int_arr)]

    # 2. 推理
    if CLASSIFIER.check_risk0_bypass(risk_level, max_mz, matched_mass):
        prob, pred = 1.0, 1
    else:
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    matches = []
    if pred == 1:
        matches = SPEC_MATCHER.match({'mz': mz_arr, 'intensities': int_arr})

    return render_template('results_ms2.html', prob=round(prob*100,2), risk_text=risk_text, risk_class=risk_class, matches=matches)

if __name__ == '__main__': app.run(host='0.0.0.0', port=5000)