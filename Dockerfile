# ---------- stage 1: build wheels ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

# ---------- stage 2: slim runtime ----------
FROM python:3.12-slim
LABEL maintainer="Tanusha Daga" description="UPI Fraud Detection - Streamlit + Prometheus metrics"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app

# libgomp is needed by catboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY src ./src
COPY artifacts/serving ./artifacts/serving

# 8501 -> Streamlit UI,  8000 -> Prometheus /metrics
EXPOSE 8501 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
