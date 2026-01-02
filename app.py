# app.py
import logging
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# 核心模块导入
from core.pipeline import get_ms1_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 限制上传大小为 16MB

# 初始化全局组件
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/model.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    """渲染首页上传界面"""
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理 .xlsx/.csv 上传并自动解析一级质谱数据"""
    if 'file' not in request.files:
        return "未选择文件", 400

    file = request.files['file']
    mode = request.form.get('mode', 'positive')

    if file.filename == '':
        return "文件名为空", 400

    try:
        # 自动解析：根据文件后缀读取数据
        if file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
            df = pd.read_excel(file) # 自动读取第一个工作表
        else:
            df = pd.read_csv(file)

        # 确保列名对齐原始数据逻辑
        if 'Mass' not in df.columns or 'Intensity' not in df.columns:
            return "错误：文件必须包含 'Mass' 和 'Intensity' 列", 400

        # 执行自动化流水线 (拟合当前文件的强度分布)
        ms1_pipeline = get_ms1_pipeline()
        df_clean = ms1_pipeline.fit_transform(df) # 包含归一化和同位素清理

        # 风险库库匹配
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)
        peaks = results.to_dict(orient='records')

        # 渲染结果列表
        return render_template('results_ms1.html', peaks=peaks, mode=mode)

    except Exception as e:
        logger.error(f"解析 MS1 失败: {e}")
        return f"解析失败，请检查文件格式: {str(e)}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """二级质谱深度判定与报告生成"""
    ms2_data = request.form.get('ms2_data')
    parent_mz = float(request.form.get('parent_mz', 0))
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))

    if not ms2_data:
        return "请输入二级质谱数据 (格式 mz:int,mz:int)", 400

    # 1. 快速旁路与模型推理
    is_risk0 = CLASSIFIER.check_risk0_bypass(risk_level, parent_mz, matched_mass)
    if is_risk0:
        prob, pred = 1.0, 1
    else:
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 2. 库回溯比对
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
            logger.warning(f"谱图库匹配出错: {e}")

    # 3. 渲染报告页面
    return render_template('results_ms2.html',
                           parent_mz=parent_mz,
                           prob=round(float(prob) * 100, 2),
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=library_matches)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)