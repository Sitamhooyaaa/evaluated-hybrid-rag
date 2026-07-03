FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY evaluation/retrieval_config.json ./evaluation/
COPY evaluation/generation_config_dev_v2.json ./evaluation/

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install .

RUN mkdir -p \
        /app/data/processed \
        /app/artifacts \
        /app/.cache/huggingface \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=180s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"]

CMD ["python", "-m", "uvicorn", "aviation_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]