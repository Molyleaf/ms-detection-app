# Dockerfile
FROM python:3.13-slim-trixie

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 更新源并安装必要库
RUN rm -f /etc/apt/sources.list && rm -rf /etc/apt/sources.list.d/
COPY sources.list /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖
COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.zju.edu.cn/pypi/web/simple \
    && pip install --no-cache-dir -r requirements.txt

# 复制项目文件
# 注意：确保宿主机 models/ 目录下存在 model.onnx
COPY data_processed/ ./data_processed/
COPY models/model.onnx ./models/model.onnx
COPY core/ ./core/
COPY templates/ ./templates/
COPY app.py .

RUN mkdir -p /tmp && chmod 777 /tmp

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]