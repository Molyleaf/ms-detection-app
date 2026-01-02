from flask import Flask, request, render_template, redirect, url_for
import os
from processor import MSProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
processor = MSProcessor('models/251229.h5', 'data_processed/risk_db.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['ms1_file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)
        # 第一步：分析危险峰
        peaks = processor.process_ms1(path)
        return render_template('ms1_results.html', peaks=peaks)
    return render_template('index.html')

@app.route('/analyze_ms2', methods=['POST'])
def analyze_ms2():
    f = request.files['ms2_file']
    target_mz = request.form['target_mz']
    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)
    # 第二步：分析二级质谱危险性
    with open(path, 'r') as file:
        result = processor.predict_ms2_risk(file.read())
    return render_template('final_results.html', mz=target_mz, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)