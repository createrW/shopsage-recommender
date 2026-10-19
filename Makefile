# ShopSage Recommender: developer entry points (`make help` lists them).
# Each target names the uv extras it needs: `uv run --extra X` adds whatever is missing,
# whereas `uv sync` is exact and removes packages outside the extras it is given.
# FAISS comes from exactly one extra: cpu (laptop, demo) or gpu (Linux x86_64 + NVIDIA). They
# conflict in pyproject.toml because both wheels install the same top-level `faiss` package.
# macOS: LightGBM (train and demo extras) needs Homebrew's libomp; `make libomp` checks for it.
# Compatible with GNU Make 3.81 (the macOS default) and newer.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV          ?= uv
DATA_CONFIG ?= configs/data/movielens_20m.yaml
PROCESSED   ?= data/processed
EDA_NB      ?= notebooks/01_eda.ipynb
EDA_TIMEOUT ?= 1800
DEMO_APP    ?= src/shopsage_rec/serving/app.py
IMAGE       ?= shopsage-demo
PYTEST_ARGS ?=
# Same paths as CI; pyproject.toml also excludes notebooks, docs and *.md from ruff.
LINT_PATHS  ?= src tests

# Where LightGBM's macOS wheel looks for the OpenMP runtime (evaluated in the shell, on use).
LIBOMP_DYLIB = $$(brew --prefix 2>/dev/null)/opt/libomp/lib/libomp.dylib

.PHONY: help setup setup-train setup-gpu libomp hooks download prepare stats eda test test-all \
        lint format typecheck check demo docker-build docker-run clean

help: ## List targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_-]+:.*## / { printf "  %-13s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

setup: ## Laptop, Day 1: core + dev extra (pytest, hypothesis, ruff, mypy, jupyter); no torch
	$(UV) sync --extra dev

setup-train: ## Laptop: dev + train + cpu extras (torch on MPS, faiss-cpu) for local training
	$(UV) sync --extra dev --extra train --extra cpu
	@if [ "$$(uname -s)" = Darwin ] && [ ! -f "$(LIBOMP_DYLIB)" ]; then \
	  echo "NOTE: lightgbm needs libomp on macOS: brew install libomp (check: make libomp)"; \
	fi

setup-gpu: ## Linux x86_64 GPU box: train + gpu + llm extras (bootstrap_cloud.sh adds data)
	@[ "$$(uname -sm)" = "Linux x86_64" ] || { \
	  echo "setup-gpu: faiss-gpu has Linux x86_64 wheels only; use 'make setup-train' here"; exit 1; }
	$(UV) sync --extra train --extra gpu --extra llm

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

prepare: ## Build data/processed/ from the raw data (downloads first if missing)
	bash scripts/prepare_data.sh $(DATA_CONFIG)

stats: ## Summarise the processed artifacts
	$(UV) run shopsage stats --processed-dir $(PROCESSED)

eda: ## Execute notebooks/01_eda.ipynb in place (outputs committed); re-runs prepare, same split_sha256
	$(UV) run --extra eda jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.kernel_name=python3 \
	  --ExecutePreprocessor.timeout=$(EDA_TIMEOUT) $(EDA_NB)

test: ## Fast tests, as in CI (no MovieLens download)
	$(UV) run --extra dev pytest -m "not slow" $(PYTEST_ARGS)

test-all: ## Full suite; the slow marker is reserved for future real-data tests (none yet)
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

# The demo extra conflicts with gpu, so run the demo from a laptop or cpu environment.
demo: ## Streamlit demo on http://localhost:8501 (hello-world app Day 2, full demo Day 9)
	@test -f $(DEMO_APP) || { echo "$(DEMO_APP) not found: hello-world app arrives Day 2, full demo Day 9"; exit 1; }
	$(UV) run --extra demo streamlit run $(DEMO_APP) --client.showErrorDetails=full

docker-build: ## Build the CPU demo image (hello-world app Day 2, demo Day 9; bundle fetched at app start)
	@test -f $(DEMO_APP) || { echo "$(DEMO_APP) not found: the image's CMD runs it (hello-world Day 2, demo Day 9)"; exit 1; }
	docker build -t $(IMAGE) .

docker-run: ## Run the demo image on http://localhost:8501
	docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY $(IMAGE)

clean: ## Remove caches and build outputs (never data/, artifacts/ or .venv/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov .coverage coverage.xml build dist
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
