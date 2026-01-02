# 化合物风险分析系统 (Compound Risk Analysis System)

本系统是一个专门用于质谱数据分析的 Web 工具，核心功能是针对**那非类化合物**（Sildenafil-like compounds）进行快速风险筛查与深度识别。系统集成了注意力机制深度学习模型、自动化特征提取流水线以及多级风险数据库匹配逻辑。

## 1. 项目架构

项目采用模块化设计，确保特征提取、风险匹配与模型推理的解耦：

```text
ms_project/
├── app.py                # Flask Web 主应用，负责路由与交互逻辑
├── convert.py            # 数据预处理脚本，将 Excel/TXT 转换为高性能 joblib 格式
├── requirements.txt      # Python 依赖列表
├── Dockerfile            # Docker 容器化构建文件
├── core/                 # 核心算法模块
│   ├── pipeline.py       # MS1 同位素清理与 MS2 10维特征提取流水线
│   ├── matcher.py        # MS1 风险库比对与 MS2 谱图库回溯
│   └── classifier.py     # 基于 Transformer 注意力机制的 MS2 分类推理
├── data/                 # 原始数据库文件（需包含 risk_matching-new.xlsx 和 ku.txt）
├── models/               # 存放预训练模型文件 (如 251229.h5)
├── data_processed/       # convert.py 生成的二进制中间数据（自动生成）
└── templates/            # Web 界面 HTML 模板

```

## 2. 核心技术逻辑

系统遵循“初筛-深挖”的两步走交互流程：

### 第一阶段：一级质谱 (MS1) 初步筛选

* **数据清洗**：自动去除零强度行，并在 2Da 范围内执行同位素峰清理。
* **风险分级**：将峰值与风险库比对，分为 **Risk0**（精确质量匹配）、**Risk1**（关键化合物）、**Risk2/3**（疑似风险）。
* **快速通道**：若触发 Risk0 且质量偏差极小，系统可直接判定为阳性。

### 第二阶段：二级质谱 (MS2) 深度判别

* **注意力模型**：使用基于 Transformer 的多头注意力模型，提取 10 个关键节点特征（如 `normalized_mz`、`position_ratio`、`is_characteristic` 等）。
* **库回溯**：对判定为阳性的样本，系统会计算其余弦相似度，并从 `ku.txt` 谱图库中回溯最可能的化学结构。

## 3. 部署说明 (Docker)

系统已完全容器化，可通过以下步骤快速部署：

### 步骤 A：准备数据

确保 `data/` 目录下存有原始风险表和谱图库，`models/` 下存有对应的 `.h5` 模型文件。

### 步骤 B：构建镜像

在项目根目录下执行，构建过程会自动运行 `convert.py` 以生成归一化统计量（`stats.joblib`）和二进制数据库：

```bash
docker build -t ms_analysis_system .

```

### 步骤 C：运行容器

```bash
docker run -d -p 5000:5000 --name ms_container ms_analysis_system

```

访问 `http://localhost:5000` 即可开始使用。

## 4. 技术特性

* **特征维度**：10 维（去除了 Intensity 直接依赖，增强了 m/z 相对分布特征）。
* **匹配精度**：特征峰匹配支持小数点后 1 位容差，Risk0 精确比对支持 0.0001 Da 容差。
* **高性能**：采用二进制索引库，二级谱图检索响应速度达毫秒级。

---

### 开发备注

* 如需更新训练模型，请同步更新 `data_processed/stats.joblib` 中的均值与标准差，以保证归一化一致性。
* 上传的文件临时存储在容器的 `/tmp` 目录中。