# ShopSage demo image: Streamlit over precomputed artifacts. CPU-only; no torch, no raw data.
#
#   make docker-build   ->  docker build -t shopsage-demo .
#   make docker-run     ->  docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY shopsage-demo
#
# Dependencies come from requirements.txt alone; the project is not pip-installed. PYTHONPATH
# makes `import shopsage_rec` resolve to the copied src/, so core dependencies the demo does not
# use (typer, duckdb) stay out of the image. A build-time import check fails the build if the
# demo stack breaks or pulls in torch.
#
# On a fresh clone artifacts/demo/ holds only .gitkeep: the app fetches the sha256-pinned
# GitHub Release bundle into it at startup (artifacts_dir in configs/deploy/streamlit.yaml), and
# a bundle exported there locally (Day 9) is copied into the image. .dockerignore is an
# allowlist, so data/ (GroupLens licence: no redistribution), .venv/ and .streamlit/secrets.toml
# never reach the context. artifacts/demo/ is still MovieLens-derived (item embeddings, model
# weights): keep the image local and out of public registries until the demo licence decision
# (D-003) is made. Under the D-003 plan the demo ships no per-user rating rows and downloads
# movie titles from GroupLens at startup.
# Every dependency ships linux/amd64 and linux/arm64 wheels, so the image builds natively on
# Apple Silicon.

FROM python:3.12-slim

LABEL org.opencontainers.image.title="shopsage-demo" \
      org.opencontainers.image.description="ShopSage two-stage recommender demo (Streamlit, CPU-only)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1: OpenMP runtime that the LightGBM wheel links dynamically (absent from slim images).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app

WORKDIR /app

# Third-party dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The package source, importable through PYTHONPATH rather than installed. pyproject.toml is
# copied only as the repo-root marker that shopsage_rec.config.find_repo_root() looks for.
COPY pyproject.toml LICENSE ./
COPY src/ src/
ENV PYTHONPATH=/app/src

RUN python -c "import faiss, lightgbm, streamlit, rank_bm25, shopsage_rec; import sys; assert 'torch' not in sys.modules" \
    && python -c "import importlib.util as u; assert u.find_spec('torch') is None, 'torch is installed'"

COPY configs/ configs/
COPY .streamlit/config.toml .streamlit/config.toml
COPY --chown=app:app artifacts/demo/ artifacts/demo/

# /app stays root-owned and read-only for the app user, except artifacts/demo/ (the startup bundle
# fetch); anything else the app downloads or caches at runtime (e.g. the GroupLens titles) must go
# under /home/app or /tmp.
USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "src/shopsage_rec/serving/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
