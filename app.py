from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache

# 尽量在导入 onnxruntime 之前禁用 GPU 探测/可见性（容器环境更稳）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DISABLE_GPU", "1")
# 如果你不想看到 onnxruntime 的 GPU discovery warning，可提高日志级别（3=ERROR）
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from core.ms1 import MS1Config, RiskConfig, process_l1_excel, risk_match_l1
from core.ms2 import MS2Config, process_l2_excel_to_peaks
from core.onnx_infer import ONNXClassifier
from core.similarity import topk_library_matches
from core.ad_checker import load_or_train_ad_checker

# 工作目录固定为 /app（仅在 /app 目录存在时切换，例如 Docker 容器环境，避免影响本地开发）
if os.path.exists("/app"):
    os.chdir("/app")

# 获取并规范化 URL_PREFIX，以斜杠开头且末尾不带斜杠。通过中间变量赋值以避免基于常量的重定义警告。
_prefix = os.environ.get("URL_PREFIX", "").strip()
if _prefix:
    if not _prefix.startswith("/"):
        _prefix = "/" + _prefix
    _prefix = _prefix.rstrip("/")
URL_PREFIX = _prefix

app = Flask(__name__, static_url_path=f"{URL_PREFIX}/static")

# 运行时依赖的资产文件（在指定工作目录下寻找）
ASSETS = {
    "risk_db": "data_processed/risk_db.joblib",
    "spectrum_db": "data_processed/spectrum_db.joblib",
    "onnx": "models/model.onnx",
    "ad_checker": "data_processed/ad_checker_model.pkl",
    "ad_train_data": "data/化合物-7-1.xlsx",
}


@lru_cache(maxsize=1)
def get_classifier() -> ONNXClassifier:
    # 懒加载，避免 worker 启动阶段 onnxruntime 初始化卡死导致 gunicorn timeout
    return ONNXClassifier(model_path=ASSETS["onnx"])


@dataclass(frozen=True)
class UploadPaths:
    # 优先从环境变量读取临时目录，无设置时默认使用 /tmp，Windows 下使用当前目录的 tmp 目录以避开权限问题
    tmp_dir: str = (
        os.environ.get("TMP_DIR")
        or os.environ.get("UPLOAD_DIR")
        or ("/tmp" if os.name != "nt" else os.path.join(os.getcwd(), "tmp"))
    )

    def ensure(self) -> None:
        os.makedirs(self.tmp_dir, exist_ok=True)

    def new_file(self, original_name: str) -> str:
        safe = secure_filename(original_name) or "upload.bin"
        stem, ext = os.path.splitext(safe)
        token = uuid.uuid4().hex[:12]
        return os.path.join(self.tmp_dir, f"{stem}_{token}{ext}")


UPLOADS = UploadPaths()


def _risk_label_for_ui(output_risk: str) -> tuple[str, str]:
    """
    对齐 templates/results_ms1.html 的展示口径：
      Risk1/Risk2 => 高风险；Risk3/Safe => 未见异常
    """
    mapping = {
        "Risk1": ("极高风险", "danger"),
        "Risk2": ("高风险", "warning"),
        "Risk3": ("未见异常", "success"),
        "Safe": ("未见异常", "success"),
        "Low Risk": ("未见异常", "success"),
    }
    return mapping.get(str(output_risk), ("未见异常", "success"))


if URL_PREFIX:
    @app.route("/")
    def root():
        # 兼容旧的根路径访问：跳转到新前缀
        return redirect(url_for("index"))

    @app.route(f"{URL_PREFIX}", strict_slashes=False)
    @app.route(f"{URL_PREFIX}/", strict_slashes=False)
    def index():
        return render_template("index.html")
else:
    @app.route("/", strict_slashes=False)
    def index():
        return render_template("index.html")
@app.route(f"{URL_PREFIX}/upload_ms1", methods=["POST"])
def upload_ms1():
    """
    Web 入口：上传 MS1 文件 -> 预处理 -> 风险匹配 -> 渲染 results_ms1.html

    性能修复：
      - 不再写入/再读回中间 xlsx（openpyxl 很慢）
      - MS1 同位素清理由 O(n^2) 改为近似 O(n log n)
      - 风险匹配直接使用 df_processed（避免重复 IO）
    """
    UPLOADS.ensure()

    f = request.files.get("file")
    mode = request.form.get("mode", "positive").strip().lower()  # positive/negative

    if not f or f.filename == "":
        return "未选择文件", 400
    if mode not in ("positive", "negative"):
        return "mode 参数错误（仅支持 positive/negative）", 400

    try:
        # 1) 保存 MS1 上传文件（已添加文件名空值回退防空保护）
        ms1_in = UPLOADS.new_file(f.filename or "upload.xlsx")
        f.save(ms1_in)

        # 2) MS1 预处理：归一化、同位素清理、低强度过滤（默认不落盘）
        df_clean = process_l1_excel(
            input_xlsx=ms1_in,
            output_xlsx=None,
            # L1 同位素/近邻峰清理窗口：±1Da
            cfg=MS1Config(mass_tolerance=1.0, min_intensity=1.0),
        )

        # 3) 风险匹配：直接用 df_processed，不写中间文件
        df_risk = risk_match_l1(
            processed_l1_xlsx=None,
            output_xlsx=None,
            cfg=RiskConfig(threshold=0.005, ion_mode=mode),
            risk_db_joblib=ASSETS["risk_db"],
            df_processed=df_clean,
        )

        # 4) 通过 Index 快速回填强度（避免 merge）
        intens = df_clean["Intensity"].to_numpy(dtype=np.float64)
        # Index 从 1 开始
        intensity_map = np.zeros(len(intens) + 1, dtype=np.float64)
        intensity_map[1:] = intens

        peaks_data = []
        for (
                idx, original_mz, actual_risk, output_risk, _matched_mz, matched_to, _match_type, _diff
        ) in df_risk.itertuples(index=False, name=None):
            original_mz_f = float(original_mz)
            intensity = float(intensity_map[int(idx)]) if 0 < int(idx) < intensity_map.size else 0.0

            matched_mass = 0.0
            if isinstance(matched_to, (int, float)) and not pd.isna(matched_to):
                matched_mass = float(matched_to)

            risk_text, risk_class = _risk_label_for_ui(str(output_risk))
            peaks_data.append(
                {
                    "Mass": original_mz_f,
                    "Intensity": intensity,
                    "Actual_Risk": str(actual_risk),
                    "Output_Risk": str(output_risk),
                    "Matched_Mass": matched_mass,
                    "output_risk_text": risk_text,
                    "output_risk_class": risk_class,
                }
            )

        # 移除了 AD 适用域判别，该步骤已被移动到 analyze_ms2 中前置串行执行
        return render_template("results_ms1.html", peaks=peaks_data, mode=mode)

    except Exception as e:
        return f"处理失败: {e}", 500


@app.route(f"{URL_PREFIX}/analyze_ms2", methods=["POST"])
def analyze_ms2():
    """
    # @ai-intent Perform MS2 classification and library match retrieval.
    # @ai-invariant Result page MUST reflect actual ONNX classifier output.
    # @ai-invariant GNN-based confidence output MUST match qlc-0103 definition without external scaling formulas.
    # @ai-boundary Flask request environment access. Temp upload file persistence.
    # @ai-context
    #   ContextData:
    #     Domain: app.py routing
    #     Flow: Upload MS2 -> AD Check -> Predict MS2 via ONNX -> Library Match

    MS2 分析入口：上传 MS2 文件（L2） -> 生成 peaks -> ONNX 推理 -> 阳性则谱库回溯

    性能修复：
      - MS2 peaks 生成不写中间 xlsx（默认）
      - onnxruntime 线程数限制避免过度订阅
      - 谱库 topk 用 heap，仅保留 top_k，避免 OOM
    """
    UPLOADS.ensure()

    parent_mz = request.form.get("parent_mz", "Unknown")
    actual_risk = request.form.get("actual_risk", "Low Risk")

    ms2_file = request.files.get("ms2_file")
    if not ms2_file or ms2_file.filename == "":
        return "未上传二级质谱文件", 400

    try:
        # 保存 MS2 上传文件（已添加文件名空值回退防空保护）
        ms2_in = UPLOADS.new_file(ms2_file.filename or "upload_ms2.xlsx")
        ms2_file.save(ms2_in)

        # AD 适用域判别：在进行 MS2 深度预测和谱库扫描前置串行运行
        ad_checker = load_or_train_ad_checker(
            model_path=ASSETS["ad_checker"],
            train_data_path=None
        )
        ad_result = ad_checker.check_ad_from_file(ms2_in, verbose=False)

        # 如果超出了模型的适用域范围，则拒绝继续执行后续昂贵的 MS2 计算和谱库回溯
        if not ad_result["within_ad"]:
            return render_template(
                "results_ms2.html",
                parent_mz=parent_mz,
                prob=None,
                risk_text=None,
                risk_class=None,
                matches=[],
                prob_symbol=None,
                ad_result=ad_result,
            )

        # 2) MS2 预处理：输出 peaks 字符串列表
        peaks_list = process_l2_excel_to_peaks(
            input_xlsx=ms2_in,
            output_xlsx=None,
            cfg=MS2Config(mass_tolerance=2.0, intensity_digits=2),
        )
        if not peaks_list:
            return "二级质谱 peaks 为空，无法判定", 400

        # 3) ONNX 推理（懒加载）
        #    支持多谱图 Batch 推理：对多个谱图概率进行平均稀释，只要有一个大于 0.5 即判为 Positive
        classifier = get_classifier()
        
        y_pred_prob = []
        positive_count = 0
        best_positive_prob = -1.0
        best_positive_peaks = None
        
        for peaks_str in peaks_list:
            pred = classifier.predict_from_peaks(peaks_str)
            prob = float(pred["probability"])
            
            # 统一提取对应“那非阳性”的原始预测概率 P_GNN
            if pred["via"] == "rule":
                p_gnn = 1.0
            else:
                if pred["label"] == "Positive":
                    p_gnn = prob
                else:
                    p_gnn = 1.0 - prob
            
            is_pos = p_gnn > 0.5 or pred["via"] == "rule"
            if is_pos:
                positive_count += 1
                if p_gnn > best_positive_prob:
                    best_positive_prob = p_gnn
                    best_positive_peaks = peaks_str
            
            y_pred_prob.append(p_gnn)
            
        if len(y_pred_prob) == 0:
            return "二级质谱 peaks 为空，无法判定", 400
            
        avg_prob = float(np.mean(y_pred_prob))
        
        # 只要这批谱图中有任意一个触发阳性，即整体判定为 Positive
        if positive_count > 0:
            label = "Positive"
            prob = avg_prob
            query_peaks = best_positive_peaks if best_positive_peaks is not None else peaks_list[0]
        else:
            label = "Negative"
            prob = 1.0 - avg_prob
            query_peaks = peaks_list[0]

        # 增加 Confidence Assessment Criteria 符号
        if prob <= 0.1:
            prob_symbol = "---"
        elif prob <= 0.3:
            prob_symbol = "--"
        elif prob <= 0.5:
            prob_symbol = "-"
        elif prob <= 0.7:
            prob_symbol = "+"
        elif prob <= 0.9:
            prob_symbol = "++"
        else:
            prob_symbol = "+++"

        # 4) 显示文本（按你模板的 class 约定）
        if label == "Positive":
            risk_text, risk_class = "有危险", "danger"
            prob_display = round(prob * 100.0, 2) if prob <= 1.0 else 100.0
        else:
            risk_text, risk_class = "未见异常", "success"
            prob_display = round(prob * 100.0, 2) if prob <= 1.0 else 0.0


        # 4.1) Risk0 的强制 Confirmed 逻辑已根据指令移除，以完全体现 ONNX 模型预测结果

        # 5) 谱库回溯（L3）：仅在预测为阳性时进行
        #    需求：预测为阴性时，不进行 L3 匹配（避免无意义的谱库扫描与耗时）
        matches = []
        if label == "Positive":
            # 结构匹配：无论最高相似度多少，都至少展示 5 个候选
            top = topk_library_matches(
                peaks=query_peaks,
                spectrum_db_joblib=ASSETS["spectrum_db"],
                tol=0.2,
                top_k=10,
            )
            # 前端不展示 similarity；仅展示最相近的结构候选（至少 5 个）
            matches = [{"smiles": r.get("smiles", "N/A")} for r in top]

            # 若谱库条目不足，做占位（保证前端至少渲染 5 行）
            while len(matches) < 5:
                matches.append({"smiles": "N/A"})

        # 返回结果并包含 AD 适用域检测结果元数据
        return render_template(
            "results_ms2.html",
            parent_mz=parent_mz,
            prob=prob_display,
            risk_text=risk_text,
            risk_class=risk_class,
            matches=matches,
            prob_symbol=prob_symbol,
            ad_result=ad_result,
        )

    except Exception as e:
        return f"二级质谱分析失败: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)