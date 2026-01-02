# --- 阶段 1: 构建器 ---
# 基于 Python 3.13 轻量镜像
FROM python:3.13-slim-trixie

USER root

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP="app:app" \
    PYTHONPATH="/app" \
    TF_CPP_MIN_LOG_LEVEL=2

# 更换 APT 源
RUN rm -f /etc/apt/sources.list \
    && rm -rf /etc/apt/sources.list.d/
COPY sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .

RUN pip config set global.index-url https://mirrors.zju.edu.cn/pypi/web/simple \
    && pip install --no-cache-dir -r requirements.txt

# 复制转换后的数据和模型（假设已在宿主机运行过 convert.py）
COPY data_processed/ ./data_processed/
COPY models/ ./models/

# 复制源代码
COPY core/ ./core/
COPY templates/ ./templates/
COPY app.py .

RUN mkdir -p /tmp && chmod 777 /tmp

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]