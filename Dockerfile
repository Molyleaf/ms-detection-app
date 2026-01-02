FROM python:3.13-slim

# 安装系统依赖（RDKit 等可能需要）
RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制程序代码
COPY . .

# 暴露 Flask 端口
EXPOSE 5000

# 运行应用
CMD ["python", "app/app.py"]