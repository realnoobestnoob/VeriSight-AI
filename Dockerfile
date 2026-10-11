# Serving image: CPU-only inference + Streamlit app. Training is NOT done in this image.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch first (versions pinned in requirements/torch.txt), then the pinned app deps.
COPY requirements/ requirements/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements/torch.txt \
 && pip install -r requirements/runtime.lock.txt

COPY pyproject.toml README.md ./
COPY src/ src/
COPY app.py ./
COPY configs/ configs/
RUN pip install --no-deps .

# Model bundle: released artifacts are copied in at build time (see docs/RELEASE.md).
# model_artifacts/current.json + model_artifacts/releases/<id>/ must exist for the scanner to work.
COPY model_artifacts/ model_artifacts/

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Hosts such as Render/Cloud Run inject PORT; default to Streamlit's 8501 locally.
ENV PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health',timeout=4)"

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
