import os
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# 导入自定义核心模块
from core.pipeline import get_preprocessing_pipeline
from core.matcher import RiskMatcher, SpectrumMatcher
from core.classifier import MS2Classifier

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "ms_analysis_secret"
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制 16MB

# 初始化全局组件
# 注意：在 Docker 中路径需对应
PREPRO_PIPELINE = get_preprocessing_pipeline()
RISK_MATCHER = RiskMatcher('data_processed/risk_db.joblib')
SPEC_MATCHER = SpectrumMatcher('data_processed/spectrum_db.joblib')
CLASSIFIER = MS2Classifier('models/251229.h5')

@app.route('/')
def index():
    """第一步：上传一级质谱页面"""
    return render_template('index.html')

@app.route('/upload_ms1', methods=['POST'])
def upload_ms1():
    """处理一级质谱并识别危险峰"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    mode = request.form.get('mode', 'positive')

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        logger.info(f"收到一级质谱上传: {filename}, 模式: {mode}")

        try:
            # 1. 读取原始数据
            raw_df = pd.read_excel(file_path)

            # 2. 运行预处理管线 (同位素清理、归一化等)
            df_clean = PREPRO_PIPELINE.fit_transform(raw_df)

            # 3. 风险库匹配
            match_df = RISK_MATCHER.match_ms1_peaks(df_clean, mode=mode)

            if match_df.empty:
                return render_template('ms1_results.html', peaks=[], message="未发现已知风险峰。")

            peaks = match_df.to_dict(orient='records')
            return render_template('ms1_results.html', peaks=peaks, mode=mode)

        except Exception as e:
            logger.error(f"处理一级质谱出错: {e}")
            return f"文件处理失败: {str(e)}", 500

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    """第二步：对特定危险峰进行二级质谱模型分析"""
    target_mz = request.form.get('mz')
    ms2_content = request.form.get('ms2_data') # 接收文本域输入的质谱数据
    mode = request.form.get('mode')

    if not ms2_content:
        return "请输入二级质谱数据", 400

    logger.info(f"开始分析 m/z {target_mz} 的二级质谱")

    # 1. 模型预测 (注意力机制)
    # 将输入封装成列表，classifier 内部会调用 pipeline 进行特征提取
    probs, preds = CLASSIFIER.predict([ms2_content])

    prob = probs[0]
    is_positive = preds[0] == 1
    risk_text, risk_level = CLASSIFIER.get_risk_label(prob)

    # 2. 如果是阳性，进行谱图库回溯匹配
    identifications = []
    if is_positive:
        # 解析当前输入以便匹配
        from core.pipeline import MS2FeatureExtractor
        # 这里复用解析逻辑
        parsed_spec = CLASSIFIER.extractor._parse_single_spec(ms2_content)
        query_spec = {'mz': parsed_spec[:, 0], 'intensities': parsed_spec[:, 1]}
        identifications = SPEC_MATCHER.search_library(query_spec)

    return render_template('final_results.html',
                           mz=target_mz,
                           prob=f"{prob:.2%}",
                           risk_text=risk_text,
                           risk_level=risk_level,
                           identifications=identifications)

if __name__ == '__main__':
    # Docker 容器内运行必须监听 0.0.0.0
    app.run(host='0.0.0.0', port=5000, debug=False)