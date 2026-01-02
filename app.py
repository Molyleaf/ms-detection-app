# app.py
import os
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# 核心模块导入
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 初始化全局组件 (路径对应 Docker 容器内路径)
# 风险库匹配器 (Risk0/1/2/3)
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
# 阳性谱图回溯匹配器
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
# ONNX 推理分类器 (替代原有 TensorFlow 模型)
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理一级质谱上传与预处理"""
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'positive')

    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    try:
        # 读取数据
        df = pd.read_excel(file)

        # 修复点：使用 fit_transform。
        # 每次请求重新获取实例以确保针对当前文件进行正确的强度归一化拟合。
        ms1_pipeline = get_ms1_pipeline()
        df_clean = ms1_pipeline.fit_transform(df)

        # 风险库匹配
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"处理 MS1 失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ms2', methods=['POST'])
def predict_ms2():
    """执行二级质谱 ONNX 推理与库回溯"""
    ms2_data = request.form.get('ms2_data')
    parent_mz = float(request.form.get('parent_mz', 0))
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))

    if not ms2_data:
        return jsonify({'error': '无数据'}), 400

    # 1. 尝试触发 Risk0 快速旁路逻辑 (对齐 qlc.ipynb)
    is_risk0_positive = CLASSIFIER.check_risk0_bypass(risk_level, parent_mz, matched_mass)

    if is_risk0_positive:
        prob, pred = 1.0, 1
    else:
        # 2. 运行 ONNX 模型预测
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 3. 阳性结果进行库回溯相似度匹配
    library_matches = []
    if pred == 1:
        try:
            peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
            spec_obj = {
                'mz': np.array([float(p[0]) for p in peaks]),
                'intensities': np.array([float(p[1]) for p in peaks])
            }
            library_matches = SPEC_MATCHER.match(spec_obj)
        except Exception as e:
            logger.warning(f"相似度匹配失败: {e}")

    return jsonify({
        'prob': round(float(prob), 4),
        'risk_text': risk_text,
        'risk_class': risk_class,
        'matches': library_matches
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)