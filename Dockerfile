# Dockerfile
FROM python:3.13-slim-trixie

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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
CMD ["gunicorn","--bind", "0.0.0.0:5000","--worker-class", "gthread","--threads", "4","--workers", "1","--timeout", "300","--keep-alive", "2","--max-requests", "200","--max-requests-jitter", "50","app:app"]