#!/usr/bin/env bash
# Build the processed parquet artifacts (data/processed/liang/ by default) from raw MovieLens-20M,
# downloading and verifying the raw data first when it is missing.
#
# Usage: scripts/prepare_data.sh [CONFIG]   (default: configs/data/movielens_20m_liang.yaml)
#
# Runs `uv run shopsage prepare --config CONFIG`, or `python -m shopsage_rec.cli` when uv
# is absent. The split is seeded, so re-running rewrites the same artifacts. Relative paths
# inside CONFIG (raw_dir, processed_dir) are resolved from the repository root.
set -euo pipefail
unset CDPATH

log() { printf '[prepare] %s\n' "$*"; }
die() {
  printf '[prepare] ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [CONFIG]

Run 'shopsage prepare --config CONFIG' (default: configs/data/movielens_20m_liang.yaml), calling
scripts/download_data.sh first when the raw CSVs under the config's raw_dir are missing.
Uses 'uv run' when uv is installed, otherwise 'python -m shopsage_rec.cli' (\$PYTHON or python3).
USAGE
}

yaml_value() { # yaml_value KEY FILE -> value of a top-level "KEY: value" line ("" if absent)
  awk -v key="$1" 'index($0, key ":") == 1 {
    v = substr($0, length(key) + 2)
    sub(/[ \t]+#.*$/, "", v)
    gsub(/^[ \t"\047]+|[ \t"\047]+$/, "", v)
    print v
    exit
  }' "$2"
}

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/data/movielens_20m_liang.yaml}"
if [[ $# -eq 1 && -f "$1" ]]; then # an explicit path is taken relative to the caller's cwd
  CONFIG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
fi
cd "$REPO_ROOT"
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

RAW_DIR="$(yaml_value raw_dir "$CONFIG")"
RAW_DIR="${RAW_DIR:-data/raw/ml-20m}"
PROCESSED_DIR="$(yaml_value processed_dir "$CONFIG")"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"

# 1. Raw data: the loaders read ratings.csv, movies.csv and tags.csv.
missing=0
for f in ratings.csv movies.csv tags.csv; do
  [[ -s "$RAW_DIR/$f" ]] || missing=1
done
if [[ $missing -eq 0 ]]; then
  log "raw data found in $RAW_DIR"
elif [[ "$(basename "$RAW_DIR")" == "ml-20m" ]]; then
  log "raw data missing in $RAW_DIR; downloading"
  bash "$REPO_ROOT/scripts/download_data.sh" "$(dirname "$RAW_DIR")"
else
  die "raw data missing in $RAW_DIR; run scripts/download_data.sh DIR and set raw_dir to DIR/ml-20m"
fi

# 2. Processed artifacts.
if command -v uv >/dev/null 2>&1; then
  log "uv run shopsage prepare --config $CONFIG"
  uv run shopsage prepare --config "$CONFIG"
else
  py="${PYTHON:-$(command -v python3 || command -v python || true)}"
  [[ -n "$py" ]] || die "neither uv nor python found; install uv: https://docs.astral.sh/uv/"
  "$py" -c 'import shopsage_rec' 2>/dev/null ||
    die "uv not found and $py cannot import shopsage_rec; install uv, or run: pip install -e ."
  log "uv not found; $py -m shopsage_rec.cli prepare --config $CONFIG"
  "$py" -m shopsage_rec.cli prepare --config "$CONFIG"
fi

log "done: artifacts in $PROCESSED_DIR"
log "summary: uv run shopsage stats --processed-dir $PROCESSED_DIR"
