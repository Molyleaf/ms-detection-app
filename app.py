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
MS1_PIPELINE = get_ms1_pipeline()
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/251229.onnx', 'data_processed/stats.joblib')

@app.route('/')
def index():
    """主页：提供 MS1 文件上传"""
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理 MS1 数据上传与风险初步筛选"""
    file = request.files.get('file')
    mode = request.form.get('mode', 'positive')

    if not file:
        return "请选择文件", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 1. 加载数据 (支持 xlsx/csv)
        if filename.endswith('.xlsx'):
            # 默认读取第一个 Sheet
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)

        # 校验列名 (Mass, Intensity)
        if 'Mass' not in df.columns or 'Intensity' not in df.columns:
            return "文件必须包含 'Mass' 和 'Intensity' 列", 400

        # 2. 运行预处理流水线 (清理同位素、归一化等)
        df_clean = MS1_PIPELINE.transform(df)

        # 3. 风险库匹配
        results = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

        # 4. 转换结果为字典供前端渲染
        peaks = results.to_dict(orient='records')
        return render_template('results_ms1.html', peaks=peaks, mode=mode)

    except Exception as e:
        logger.error(f"MS1 处理出错: {e}")
        return f"处理失败: {str(e)}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """接收 MS2 字符串输入并运行注意力模型"""
    ms2_data = request.form.get('ms2_data')
    parent_mz = float(request.form.get('parent_mz', 0))
    risk_level = request.form.get('risk_level', 'Safe')
    matched_mass = float(request.form.get('matched_mass', 0))

    if not ms2_data:
        return jsonify({'error': '无数据'}), 400

    # 1. 尝试触发 Risk0 快速旁路逻辑
    # 如果 MS1 已发现 Risk0 物质且母离子精确匹配，直接判定
    is_risk0_positive = CLASSIFIER.check_risk0_bypass(risk_level, parent_mz, matched_mass)

    if is_risk0_positive:
        prob, pred = 1.0, 1
    else:
        # 2. 正常注意力模型预测
        probs, preds = CLASSIFIER.predict([ms2_data])
        prob, pred = probs[0], preds[0]

    risk_text, risk_class = CLASSIFIER.get_risk_label(prob)

    # 3. 如果判定为阳性 (pred == 1)，进行库回溯
    library_matches = []
    if pred == 1:
        # 解析输入字符串为 Matcher 需要的格式
        try:
            peaks = [p.split(':') for p in ms2_data.replace(';', ',').split(',') if ':' in p]
            spec_obj = {
                'mz': np.array([float(p[0]) for p in peaks]),
                'intensities': np.array([float(p[1]) for p in peaks])
            }
            library_matches = SPEC_MATCHER.match(spec_obj)
        except:
            pass

    return render_template('results_ms2.html',
                           prob=round(prob*100, 2),
                           risk_text=risk_text,
                           risk_class=risk_class,
                           matches=library_matches,
                           parent_mz=parent_mz)

if __name__ == '__main__':
    # 容器内运行建议监听 0.0.0.0
    app.run(host='0.0.0.0', port=5000)