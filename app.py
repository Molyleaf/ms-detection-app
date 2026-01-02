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
# ONNX 推理分类器
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    """渲染首页上传界面"""
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理一级质谱上传并渲染筛选结果页面"""
    if 'file' not in request.files:
        return "未上传文件", 400

    file = request.files['file']
    mode = request.form.get('mode', 'positive')

    if file.filename == '':
        return "文件名为空", 400

    try:
        # 读取上传的 Excel 数据
        df = pd.read_excel(file)

        # 核心逻辑修复：使用 fit_transform 进行动态归一化
        ms1_pipeline = get_ms1_pipeline()
        df_clean = ms1_pipeline.fit_transform(df)

        # 风险库匹配
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

        # 将结果转换为字典列表，供 results_ms1.html 中的 Jinja2 循环渲染
        peaks = results.to_dict(orient='records')

        return render_template('results_ms1.html', peaks=peaks, mode=mode)
    except Exception as e:
        logger.error(f"处理 MS1 失败: {e}")
        return f"处理失败: {str(e)}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """执行二级质谱深度分析报告渲染"""
    ms2_data = request.form.get('ms2_data')
    parent_mz = float(request.form.get('parent_mz', 0))
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))

    if not ms2_data:
        return "二级质谱谱图数据为空", 400

    # 1. 尝试触发 Risk0 快速旁路逻辑 (对齐 qlc.ipynb)
    is_risk0_positive = CLASSIFIER.check_risk0_bypass(risk_level, parent_mz, matched_mass)

    if is_risk0_positive:
        prob, pred = 1.0, 1
    else:
        # 2. 运行 ONNX 模型预测
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    # 获取风险描述和 CSS 类 (danger/warning/success)
    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 3. 阳性结果进行库回溯相似度匹配
    library_matches = []
    if pred == 1:
        try:
            # 解析 mz:intensity 字符串
            peaks_raw = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
            spec_obj = {
                'mz': np.array([float(p[0]) for p in peaks_raw]),
                'intensities': np.array([float(p[1]) for p in peaks_raw])
            }
            library_matches = SPEC_MATCHER.match(spec_obj)
        except Exception as e:
            logger.warning(f"谱图库匹配失败: {e}")

    # 4. 渲染分析报告页面，传递 Jinja2 替换变量
    return render_template('results_ms2.html',
                           parent_mz=parent_mz,
                           prob=round(float(prob) * 100, 2), # 页面显示百分制
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=library_matches)

if __name__ == '__main__':
    # 默认运行在 5000 端口，与 Dockerfile 保持一致
    app.run(host='0.0.0.0', port=5000)