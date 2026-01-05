# app.py
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache

# 尽量在导入 onnxruntime 之前禁用 GPU 探测/可见性（容器环境更稳）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_DISABLE_GPU", "1")

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from core.ms1 import MS1Config, RiskConfig, process_l1_excel, risk_match_l1
from core.ms2 import MS2Config, process_l2_excel_to_peaks
from core.onnx_infer import ONNXClassifier
from core.similarity import topk_library_matches

app = Flask(__name__)

# 运行时依赖的资产文件（Docker 内保持这些路径存在即可）
ASSETS = {
    "risk_db": "data_processed/risk_db.joblib",
    "spectrum_db": "data_processed/spectrum_db.joblib",
    "stats": "data_processed/stats.joblib",
    "onnx": "models/model.onnx",
}

# 原来这里是启动即加载：
# CLASSIFIER = ONNXClassifier(model_path=ASSETS["onnx"], stats_joblib=ASSETS["stats"])
# 改为懒加载，避免 worker 启动阶段被 onnxruntime 初始化卡死导致 gunicorn timeout
@lru_cache(maxsize=1)
def get_classifier() -> ONNXClassifier:
    return ONNXClassifier(model_path=ASSETS["onnx"], stats_joblib=ASSETS["stats"])


@dataclass(frozen=True)
class UploadPaths:
    tmp_dir: str = "/tmp"

    def ensure(self) -> None:
        os.makedirs(self.tmp_dir, exist_ok=True)

    def new_file(self, original_name: str) -> str:
        safe = secure_filename(original_name) or "upload.bin"
        stem, ext = os.path.splitext(safe)
        token = uuid.uuid4().hex[:12]
        return os.path.join(self.tmp_dir, f"{stem}_{token}{ext}")


UPLOADS = UploadPaths()


def _is_excel(filename: str) -> bool:
    fn = filename.lower()
    return fn.endswith(".xlsx") or fn.endswith(".xls")


def _risk_label_for_ui(output_risk: str) -> tuple[str, str]:
    """
    对齐 templates/results_ms1.html 的展示口径：
      Risk1/Risk2 => 高风险；Risk3/Safe => 未见异常
    """
    mapping = {
        "Risk1": ("高风险", "danger"),
        "Risk2": ("高风险", "warning"),
        "Risk3": ("未见异常", "success"),
        "Safe": ("未见异常", "success"),
        "Low Risk": ("未见异常", "success"),
    }
    return mapping.get(str(output_risk), ("未见异常", "success"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_ms1", methods=["POST"])
def upload_ms1():
    """
    Web 入口：上传 MS1 文件 -> 预处理 -> 风险匹配 -> 渲染 results_ms1.html
    """
    UPLOADS.ensure()

    f = request.files.get("file")
    mode = request.form.get("mode", "positive").strip().lower()  # positive/negative

    if not f or f.filename == "":
        return "未选择文件", 400
    if mode not in ("positive", "negative"):
        return "mode 参数错误（仅支持 positive/negative）", 400

    try:
        # 1) 保存 MS1 上传文件
        ms1_in = UPLOADS.new_file(f.filename)
        f.save(ms1_in)

        # 2) MS1 预处理：归一化、同位素清理、低强度过滤
        ms1_processed = UPLOADS.new_file("ms1_processed.xlsx")
        df_clean = process_l1_excel(
            input_xlsx=ms1_in,
            output_xlsx=ms1_processed,
            cfg=MS1Config(mass_tolerance=2.0, min_intensity=1.0),
        )

        # 3) 风险匹配：读取处理后的文件顺序做匹配（与 notebook 行为一致）
        risk_out = UPLOADS.new_file("risk_results.xlsx")
        df_risk = risk_match_l1(
            processed_l1_xlsx=ms1_processed,
            output_xlsx=risk_out,
            cfg=RiskConfig(threshold=0.005, ion_mode=mode),
            risk_db_joblib=ASSETS["risk_db"],
        )

        # 4) 合并强度信息（df_risk 只有 m/z 风险字段；模板要显示 Intensity）
        #    这里按 Index 顺序对齐：Index 从 1 开始，df_clean 行顺序与 risk_match_l1 一致
        df_clean2 = df_clean.reset_index(drop=True).copy()
        df_clean2["Index"] = np.arange(1, len(df_clean2) + 1)

        merged = pd.merge(df_risk, df_clean2[["Index", "Intensity"]], on="Index", how="left")

        # 5) 转成模板需要的字段名
        peaks_data = []
        for _, row in merged.iterrows():
            original_mz = float(row["Original m/z"])
            intensity = row.get("Intensity", np.nan)
            actual_risk = str(row.get("Actual Risk", "Low Risk"))
            output_risk = str(row.get("Output Risk", "Low Risk"))

            matched_to = row.get("Matched to m/z", None)
            matched_mass = 0.0
            if isinstance(matched_to, (int, float)) and not pd.isna(matched_to):
                matched_mass = float(matched_to)

            peak = {
                "Mass": original_mz,
                "Intensity": float(intensity) if isinstance(intensity, (int, float)) and not pd.isna(intensity) else 0.0,
                "Actual_Risk": actual_risk,
                "Output_Risk": output_risk,
                "Matched_Mass": matched_mass,
            }
            risk_text, risk_class = _risk_label_for_ui(output_risk)
            peak["output_risk_text"] = risk_text
            peak["output_risk_class"] = risk_class
            peaks_data.append(peak)

        return render_template("results_ms1.html", peaks=peaks_data, mode=mode)

    except Exception as e:
        return f"处理失败: {e}", 500


@app.route("/analyze_ms2", methods=["POST"])
def analyze_ms2():
    """
    MS2 分析入口：上传 MS2 文件（L2） -> 生成 peaks -> ONNX 推理 -> 阳性则谱库回溯
    """
    UPLOADS.ensure()

    parent_mz = request.form.get("parent_mz", "Unknown")
    actual_risk = request.form.get("actual_risk", "Low Risk")
    matched_mass_raw = request.form.get("matched_mass", "0")

    # 兜底解析 matched_mass（用于前端展示/调试）
    try:
        matched_mass = float(matched_mass_raw)
    except Exception:
        matched_mass = 0.0

    ms2_file = request.files.get("ms2_file")
    if not ms2_file or ms2_file.filename == "":
        return "未上传二级质谱文件", 400

    try:
        ms2_in = UPLOADS.new_file(ms2_file.filename)
        ms2_file.save(ms2_in)

        # 1) 若 MS1 实际 Risk0：直接判阳（notebook 里 Risk0 是旁路）
        if str(actual_risk) == "Risk0":
            return render_template(
                "results_ms2.html",
                parent_mz=parent_mz,
                prob=100.0,
                risk_text="有危险 (Confirmed)",
                risk_class="danger",
                matches=[],
            )

        # 2) MS2 预处理：输出 peaks 字符串（mass:intensity,...）
        ms2_processed = UPLOADS.new_file("ms2_processed.xlsx")
        peaks = process_l2_excel_to_peaks(
            input_xlsx=ms2_in,
            output_xlsx=ms2_processed,
            cfg=MS2Config(mass_tolerance=2.0, intensity_digits=2),
        )
        if not peaks:
            return "二级质谱 peaks 为空，无法判定", 400

        # 3) ONNX 推理（懒加载，避免 worker 启动超时）
        classifier = get_classifier()
        pred = classifier.predict_from_peaks(peaks)
        label = pred["label"]
        prob = float(pred["probability"])

        # 4) 显示文本（按你模板的 class 约定）
        if label == "Positive":
            risk_text, risk_class = "有危险", "danger"
            prob_display = round(prob * 100.0, 2) if prob <= 1.0 else 100.0
        else:
            risk_text, risk_class = "未见异常", "success"
            prob_display = round(prob * 100.0, 2) if prob <= 1.0 else 0.0

        # 5) 阳性则谱库回溯（Top-10）
        matches = []
        if label == "Positive":
            top = topk_library_matches(
                peaks=peaks,
                spectrum_db_joblib=ASSETS["spectrum_db"],
                tol=0.2,
                top_k=10,
            )
            # 模板期望字段：match.smiles / match.score
            matches = [{"smiles": r["smiles"], "score": r["similarity"]} for r in top]

        return render_template(
            "results_ms2.html",
            parent_mz=parent_mz,
            prob=prob_display,
            risk_text=risk_text,
            risk_class=risk_class,
            matches=matches,
        )

    except Exception as e:
        return f"二级质谱分析失败: {e}", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)