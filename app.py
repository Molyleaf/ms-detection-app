# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

app = Flask(__name__)

# 初始化组件
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')


def get_output_risk(output_risk_level: str):
    """
    仅基于 Output_Risk 做前端展示（对齐 Notebook 展示口径）：
    - Output Risk = Risk1/Risk2 => 高风险
    - Output Risk = Risk3/Safe => 未见异常
    """
    mapping = {
        'Risk1': ('高风险', 'danger'),
        'Risk2': ('高风险', 'warning'),
        'Risk3': ('未见异常', 'success'),
        'Safe': ('未见异常', 'success')
    }
    return mapping.get(str(output_risk_level), ('未见异常', 'success'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')
    if not file:
        return "未选择文件", 400

    try:
        df = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)

        # 1) 执行严格清理（归一化、同位素窗口聚类、低强度过滤）
        df_clean = get_ms1_pipeline().fit_transform(df)

        # 2) 风险匹配（输出 Actual_Risk / Output_Risk 对齐 Notebook）
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

        # 3) 注入 Output Risk 标签（只看 Output_Risk）
        peaks_data = results.to_dict(orient='records')
        for peak in peaks_data:
            risk_text, risk_class = get_output_risk(peak.get('Output_Risk', 'Safe'))
            peak['output_risk_text'] = risk_text
            peak['output_risk_class'] = risk_class

        return render_template('results_ms1.html', peaks=peaks_data, mode=mode)

    except Exception as e:
        return f"处理失败: {e}", 500


@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    # 注意：对齐 Notebook：Risk0 仍展示为 Risk1，但我们会保留 Actual_Risk 用于旁路
    actual_risk = request.form.get('actual_risk', 'Safe')
    output_risk = request.form.get('output_risk', request.form.get('risk_level', 'Safe'))

    matched_mass = float(request.form.get('matched_mass', 0))
    parent_mz = request.form.get('parent_mz', 'Unknown')
    ms2_data = request.form.get('ms2_data', '')

    if 'ms2_file' in request.files:
        f = request.files['ms2_file']
        if f and f.filename != '':
            df = pd.read_excel(f) if f.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(f)
            m_col = next((c for c in df.columns if c.lower() in ['mass', 'm/z']), df.columns[0])
            i_col = next((c for c in df.columns if c.lower() in ['intensity', 'int']), df.columns[1])
            ms2_data = ",".join([f"{r[m_col]}:{r[i_col]}" for _, r in df.iterrows()])

    # Notebook 的 Risk0 旁路：实际是 Risk0 即“Confirmed”
    if str(actual_risk) == 'Risk0':
        return render_template(
            'results_ms2.html',
            parent_mz=parent_mz,
            prob=100.0,
            risk_text="有危险 (Confirmed)",
            risk_class="danger",
            matches=[]
        )

    if not ms2_data:
        return "无二级数据", 400

    # 模型推理 + 旁路检查（不依赖 Risk0 字符串；classifier 已改为 Risk1 + matched_mass 判定）
    if CLASSIFIER.check_risk0_bypass(str(output_risk), ms2_data, matched_mass):
        prob, pred = 1.0, 1
    else:
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 阳性结果回溯谱库
    matches = []
    if pred == 1:
        raw_peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
        target_spec = {
            'mz': np.array([float(p[0]) for p in raw_peaks], dtype=float),
            'intensities': np.array([float(p[1]) for p in raw_peaks], dtype=float)
        }
        matches = SPEC_MATCHER.match(target_spec)

    return render_template(
        'results_ms2.html',
        parent_mz=parent_mz,
        prob=round(prob * 100, 2),
        risk_text=risk_text,
        risk_class=risk_class,
        matches=matches
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)