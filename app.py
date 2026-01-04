# app.py
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

app = Flask(__name__)
RM = RiskMatcher(); SM = SpectrumMatcher(); CL = MS2Classifier()

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    # 获取前端传回的一级质谱匹配信息
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))
    ms2_data = ""

    # 解析 MS2 文件或文本
    if 'ms2_file' in request.files and request.files['ms2_file'].filename != '':
        file = request.files['ms2_file']
        df_ms2 = pd.read_excel(file) if file.filename.endswith('.xlsx') else pd.read_csv(file)
        m_c = 'Mass' if 'Mass' in df_ms2.columns else df_ms2.columns[0]
        i_c = 'Intensity' if 'Intensity' in df_ms2.columns else df_ms2.columns[1]
        # 预处理：归一化并转为字符串格式
        max_i = df_ms2[i_c].max()
        df_ms2['Norm_I'] = (df_ms2[i_c] / max_i * 100).round(2)
        ms2_data = ",".join([f"{r[m_c]}:{r['Norm_I']}" for _, r in df_ms2.iterrows()])
        ms2_max_mz = df_ms2[m_c].max()
    else:
        ms2_data = request.form.get('ms2_data', "")
        ms2_max_mz = float(ms2_data.split(':')[0]) if ':' in ms2_data else 0

    # 1. 执行 Risk0 旁路或模型预测
    if CL.check_risk0_bypass(risk_level, ms2_max_mz, matched_mass):
        prob, pred = 1.0, 1
    else:
        probs, preds = CL.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CL.get_risk_label(prob)

    # 2. 阳性样本回溯库匹配
    matches = []
    if pred == 1:
        raw = [p.split(':') for p in ms2_data.split(',') if ':' in p]
        spec = {'mz': np.array([float(x[0]) for x in raw]), 'intensities': np.array([float(x[1]) for x in raw])}
        matches = SM.match(spec)

    return render_template('results_ms2.html', prob=round(prob*100, 2), risk_text=risk_text, risk_class=risk_class, matches=matches)

if __name__ == '__main__':
    app.run(debug=True, port=5000)