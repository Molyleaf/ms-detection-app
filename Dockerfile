# Dockerfile
FROM python:3.13-slim-trixie

# 设置用户ID和组ID构建参数
ARG APP_UID=1000
ARG APP_GID=1000

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制依赖配置文件到临时目录，并使用最小化镜像构建流程
COPY requirements.txt /tmp/requirements.txt
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends binutils; \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -r /tmp/requirements.txt; \
    rm -f /tmp/requirements.txt; \
    find /usr/local/lib/python3.13/site-packages -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.py[co]" -delete 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.13/site-packages -name "*.a" -delete 2>/dev/null || true; \
    groupadd --gid "${APP_GID}" appgroup; \
    useradd --uid "${APP_UID}" --gid "${APP_GID}" --no-create-home --home-dir /tmp --shell /usr/sbin/nologin appuser; \
    apt-get purge -y --auto-remove binutils; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /root/.cache; \
    pip cache purge 2>/dev/null || true

# 复制项目文件并设置文件所有者为 appuser
# 注意：确保宿主机 models/ 目录下存在 model.onnx
COPY --chown=appuser:appgroup data_processed/ ./data_processed/
COPY --chown=appuser:appgroup models/model.onnx ./models/model.onnx
COPY --chown=appuser:appgroup core/ ./core/
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup app.py .

# 创建供非 root 用户使用的临时目录并赋予全权限
RUN mkdir -p /tmp && chmod 777 /tmp

# 切换为非 root 账户并暴露服务端口
USER appuser
EXPOSE 5000
CMD ["gunicorn","--bind", "0.0.0.0:5000","--worker-class", "gthread","--threads", "4","--workers", "1","--timeout", "300","--keep-alive", "2","--max-requests", "200","--max-requests-jitter", "50","app:app"]