# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

app = Flask(__name__)

# 初始化核心业务组件
# 确保数据文件路径正确，对应 Dockerfile 中的 COPY 路径
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    """首页渲染"""
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """
    一级质谱上传与筛选
    1. 执行 pipeline 清理（同位素、归一化）
    2. 执行 RiskMatcher 风险分级
    """
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')
    if not file:
        return "未选择文件", 400
    try:
        # 加载数据
        df = pd.read_excel(file) if file.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(file)

        # 执行严格的同位素清洗与强度归一化 (对齐 qlc-0103.ipynb)
        df_clean = get_ms1_pipeline().fit_transform(df)

        # 进行风险库匹配
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

        # 渲染结果页面 (前端模板将根据 Risk_Level 显示 Output Risk)
        return render_template('results_ms1.html',
                               peaks=results.to_dict(orient='records'),
                               mode=mode)
    except Exception as e:
        return f"处理失败: {e}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """
    二级质谱深度分析
    - Risk0: 直接显示“有危险”，跳过模型
    - Risk1/2: 调用模型推理并回溯谱图库
    """
    risk_level = request.form.get('risk_level')
    parent_mz = request.form.get('parent_mz', 'Unknown')
    matched_mass = float(request.form.get('matched_mass', 0))
    ms2_data = request.form.get('ms2_data', '')

    # --- 1. Risk0 旁路/拦截逻辑 ---
    # 如果一级质谱已确定为 Risk0，直接判定为阳性，无需上传文件或推理
    if risk_level == 'Risk0':
        return render_template('results_ms2.html',
                               parent_mz=parent_mz,
                               prob=100.0,
                               risk_text="有危险 (Confirmed - Risk0)",
                               risk_class="danger",
                               matches=[])

    # --- 2. 获取 MS2 数据 ---
    if 'ms2_file' in request.files:
        f = request.files['ms2_file']
        if f and f.filename != '':
            df = pd.read_excel(f) if f.filename.endswith(('.xlsx', '.xls')) else pd.read_csv(f)
            # 获取前两列（m/z 和 Intensity）
            m_col, i_col = df.columns[0], df.columns[1]
            ms2_data = ",".join([f"{r[m_col]}:{r[i_col]}" for _, r in df.iterrows()])

    if not ms2_data:
        return "无二级质谱数据，无法进一步分析", 400

    # --- 3. 模型预测与判定 ---
    # check_risk0_bypass 用于兜底校验 0.005 Da 判定
    if CLASSIFIER.check_risk0_bypass(risk_level, ms2_data, matched_mass):
        prob, pred = 1.0, 1
    else:
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    # 获取对应的显示文本和 CSS 类 (danger/warning/success)
    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # --- 4. 阳性结果回溯 ---
    matches = []
    if pred == 1:
        # 格式化数据以进行谱图匹配
        raw_peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
        target_spec = {
            'mz': np.array([float(p[0]) for p in raw_peaks]),
            'intensities': np.array([float(p[1]) for p in raw_peaks])
        }
        matches = SPEC_MATCHER.match(target_spec)

    return render_template('results_ms2.html',
                           parent_mz=parent_mz,
                           prob=round(prob*100, 2),
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=matches)

if __name__ == '__main__':
    # 启动应用
    app.run(host='0.0.0.0', port=5000)