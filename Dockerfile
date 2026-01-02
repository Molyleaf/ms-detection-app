# 使用轻量级 Python 镜像
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统库（针对 numpy/pandas）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制转换后的数据和模型（假设已在宿主机运行过 convert.py）
COPY data_processed/ ./data_processed/
COPY models/ ./models/

# 复制源代码
COPY core/ ./core/
COPY templates/ ./templates/
COPY app.py .

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app.py"]