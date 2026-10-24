# ShopSage Recommender: developer entry points (`make help` lists them).
# Each target names the uv extras it needs: `uv run --extra X` adds whatever is missing,
# whereas `uv sync` is exact and removes packages outside the extras it is given.
# Milvus Lite (demo and dev extras) brings faiss-cpu with it; training needs no vector index.
# macOS: LightGBM (train and demo extras) needs Homebrew's libomp; `make libomp` checks for it.
# Compatible with GNU Make 3.81 (the macOS default) and newer.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV          ?= uv
DATA_CONFIG ?= configs/data/movielens_20m_liang.yaml
PROCESSED   ?= data/processed/liang
EDA_NB      ?= notebooks/01_eda.ipynb
EDA_TIMEOUT ?= 1800
DEMO_APP    ?= src/shopsage_rec/serving/app.py
IMAGE       ?= shopsage-demo
PYTEST_ARGS ?=
# Same paths as CI; pyproject.toml also excludes notebooks, docs and *.md from ruff.
LINT_PATHS  ?= src tests

# Where LightGBM's macOS wheel looks for the OpenMP runtime (evaluated in the shell, on use).
LIBOMP_DYLIB = $$(brew --prefix 2>/dev/null)/opt/libomp/lib/libomp.dylib

.PHONY: help setup setup-train setup-gpu libomp hooks download prepare stats evaluate eda test \
        test-all lint format typecheck check demo demo-requirements docker-build docker-run clean

help: ## List targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-13s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Laptop, Day 1: core + dev extra (pytest, hypothesis, ruff, mypy, jupyter); no torch
	$(UV) sync --extra dev

setup-train: ## Laptop: dev + train extras (torch on MPS) for local training
	$(UV) sync --extra dev --extra train
	@if [ "$$(uname -s)" = Darwin ] && [ ! -f "$(LIBOMP_DYLIB)" ]; then \
	  echo "NOTE: lightgbm needs libomp on macOS: brew install libomp (check: make libomp)"; \
	fi

setup-gpu: ## Linux GPU box: train + llm extras (bootstrap_cloud.sh also adds the data)
	$(UV) sync --extra train --extra llm

libomp: ## macOS: check for the libomp.dylib LightGBM 4.7 loads (fix: brew install libomp)
	@if [ "$$(uname -s)" != Darwin ]; then \
	  echo "Not macOS: LightGBM needs libgomp1 on Debian/Ubuntu (apt-get install -y libgomp1)"; \
	elif [ -f "$(LIBOMP_DYLIB)" ]; then \
	  echo "libomp found: $(LIBOMP_DYLIB)"; \
	else \
	  echo "libomp.dylib not found: run 'brew install libomp' before importing lightgbm"; exit 1; \
	fi

hooks: ## Install the pre-commit hooks (needs a git checkout)
	$(UV) run --extra dev pre-commit install

download: ## Download, md5-verify and unzip MovieLens-20M into data/raw/
	bash scripts/download_data.sh

prepare: ## Build the Liang split in data/processed/liang/ (downloads first if missing)
	bash scripts/prepare_data.sh $(DATA_CONFIG)

stats: ## Summarise the processed artifacts
	$(UV) run shopsage stats --processed-dir $(PROCESSED)

evaluate: ## MostPop + EASE on the Liang split -> artifacts/eval/<run-id>/ (E004)
	$(UV) run shopsage evaluate --processed-dir $(PROCESSED)

eda: ## Execute notebooks/01_eda.ipynb in place (outputs committed); re-runs prepare, same split_sha256
	$(UV) run --extra eda jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.kernel_name=python3 \
	  --ExecutePreprocessor.timeout=$(EDA_TIMEOUT) $(EDA_NB)

test: ## Fast tests, as in CI (no MovieLens download)
	$(UV) run --extra dev pytest -m "not slow" $(PYTEST_ARGS)

test-all: ## Full suite, including the slow real-data tests (need data/raw/ml-20m)
	$(UV) run --extra dev pytest $(PYTEST_ARGS)

lint: ## ruff check + ruff format --check over src/ and tests/ (same as CI)
	$(UV) run --extra dev ruff check $(LINT_PATHS)
	$(UV) run --extra dev ruff format --check $(LINT_PATHS)

format: ## Apply ruff autofixes, then format (src/ and tests/)
	$(UV) run --extra dev ruff check --fix --exit-zero $(LINT_PATHS)
	$(UV) run --extra dev ruff format $(LINT_PATHS)

typecheck: ## mypy over shopsage_rec (configured in pyproject.toml)
	$(UV) run --extra dev mypy

check: lint typecheck test ## lint + typecheck + fast tests

demo: ## Streamlit app on http://localhost:8501 (deployment probe now, full demo in S6)
	@test -f $(DEMO_APP) || { echo "$(DEMO_APP) not found"; exit 1; }
	$(UV) run --extra demo streamlit run $(DEMO_APP) --client.showErrorDetails=full

demo-requirements: ## Regenerate src/shopsage_rec/serving/requirements.txt from uv.lock (CI checks it)
	$(UV) export --frozen --no-hashes --no-emit-project --extra demo -o src/shopsage_rec/serving/requirements.txt

docker-build: ## Build the CPU demo image (probe now, demo in S6; bundle fetched at app start)
	@test -f $(DEMO_APP) || { echo "$(DEMO_APP) not found: the image's CMD runs it"; exit 1; }
	docker build -t $(IMAGE) .

docker-run: ## Run the demo image on http://localhost:8501
	docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY $(IMAGE)

clean: ## Remove caches and build outputs (never data/, artifacts/ or .venv/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
