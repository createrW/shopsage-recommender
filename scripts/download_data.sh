#!/usr/bin/env bash
# Download, md5-verify and unzip MovieLens-20M from GroupLens.
#
# Usage: scripts/download_data.sh [TARGET_DIR]
#   TARGET_DIR receives ml-20m.zip, ml-20m.zip.md5 and ml-20m/ (default: <repo>/data/raw,
#   which matches raw_dir in configs/data/movielens_20m.yaml).
#
# Idempotent: a zip that passes the md5 check is not downloaded again, and an extraction
# that already completed from that same zip is not repeated.
# MOVIELENS_BASE_URL overrides the download location (a mirror, or file:// for offline tests).
# MOVIELENS_ALLOW_MD5_CHANGE=1 accepts an archive whose published md5 differs from the pinned one.
set -euo pipefail
unset CDPATH

readonly NAME="ml-20m"
readonly BASE_URL="${MOVIELENS_BASE_URL:-https://files.grouplens.org/datasets/movielens}"
# Published md5 of ml-20m.zip. Every archive must match it unless MOVIELENS_ALLOW_MD5_CHANGE=1;
# the fetched ml-20m.zip.md5 is only a cross-check.
readonly PINNED_MD5="cd245b17a1ae2cc31bb14903e1204af3"
readonly EXPECTED_FILES=(README.txt ratings.csv movies.csv tags.csv links.csv genome-scores.csv genome-tags.csv)

log() { printf '[download] %s\n' "$*"; }
warn() { printf '[download] WARNING: %s\n' "$*" >&2; }
die() {
  printf '[download] ERROR: %s\n' "$*" >&2
  exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<USAGE
Usage: $(basename "$0") [TARGET_DIR]

Download, md5-verify and unzip MovieLens-20M into TARGET_DIR (default: data/raw).
Writes TARGET_DIR/ml-20m.zip, TARGET_DIR/ml-20m.zip.md5 and TARGET_DIR/ml-20m/.
Safe to re-run: a verified zip and a completed extraction are skipped.
USAGE
}

fetch() { # fetch URL DEST [quiet]
  local url="$1" dest="$2" quiet="${3:-}"
  if have curl; then
    if [[ -n "$quiet" ]]; then
      curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 30 -o "$dest" "$url"
    else
      curl -fL --retry 3 --retry-delay 2 --connect-timeout 30 --progress-bar -o "$dest" "$url"
    fi
  elif have wget; then
    if [[ -n "$quiet" ]]; then
      wget -q -O "$dest" "$url"
    else
      wget -O "$dest" "$url"
    fi
  else
    warn "neither curl nor wget is installed"
    return 1
  fi
}

md5_of() { # md5_of FILE -> lowercase hex digest
  if have md5sum; then
    md5sum <"$1" | awk '{ print $1 }'
  elif have md5; then
    md5 -q "$1"
  else
    openssl dgst -md5 <"$1" | awk '{ print $NF }'
  fi
}

parse_md5() { # parse_md5 FILE -> digest from BSD "MD5 (f) = <hex>" or GNU "<hex>  f" lines
  awk '{
    sub(/\r$/, "")
    for (i = 1; i <= NF; i++)
      if (length($i) == 32 && $i !~ /[^0-9a-fA-F]/) { print tolower($i); exit }
  }' "$1"
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
have md5sum || have md5 || have openssl || die "need md5sum (Linux), md5 (macOS) or openssl"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$REPO_ROOT/data/raw}"
mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

ZIP="$TARGET_DIR/$NAME.zip"
MD5_FILE="$ZIP.md5"
OUT_DIR="$TARGET_DIR/$NAME"
MARKER="$OUT_DIR/.extracted_from_md5" # md5 of the archive the extraction came from
trap 'rm -f "$ZIP.part" "$MD5_FILE.part"' EXIT

# 1. Expected digest: the local .md5 file, else a fresh copy of it, else the pinned value.
expected=""
if [[ -s "$MD5_FILE" ]]; then
  expected="$(parse_md5 "$MD5_FILE")"
fi
if [[ -z "$expected" ]]; then
  log "fetching $NAME.zip.md5"
  if fetch "$BASE_URL/$NAME.zip.md5" "$MD5_FILE.part" quiet; then
    expected="$(parse_md5 "$MD5_FILE.part")"
  fi
  if [[ -n "$expected" ]]; then
    mv -f "$MD5_FILE.part" "$MD5_FILE"
  else
    warn "could not fetch or parse $NAME.zip.md5; verifying against the pinned md5 instead"
    expected="$PINNED_MD5"
  fi
fi
if [[ "$expected" != "$PINNED_MD5" ]]; then
  [[ "${MOVIELENS_ALLOW_MD5_CHANGE:-0}" == 1 ]] ||
    die "published md5 $expected ($MD5_FILE) differs from the pinned $PINNED_MD5, and ML-20M is a frozen release; delete $MD5_FILE if it is stale, or set MOVIELENS_ALLOW_MD5_CHANGE=1 to accept a changed archive"
  warn "published md5 $expected differs from the pinned $PINNED_MD5; accepted because MOVIELENS_ALLOW_MD5_CHANGE=1"
fi

# 2. Archive: reuse it when it verifies; otherwise download to .part and verify before moving.
if [[ -f "$ZIP" && "$(md5_of "$ZIP")" == "$expected" ]]; then
  log "$ZIP verified (md5 $expected); skipping download"
else
  if [[ -f "$ZIP" ]]; then
    warn "$ZIP fails the md5 check; downloading it again"
  fi
  log "downloading $BASE_URL/$NAME.zip (~199 MB)"
  fetch "$BASE_URL/$NAME.zip" "$ZIP.part" || die "download failed: $BASE_URL/$NAME.zip"
  actual="$(md5_of "$ZIP.part")"
  [[ "$actual" == "$expected" ]] || die "md5 mismatch: got $actual, expected $expected (download discarded)"
  mv -f "$ZIP.part" "$ZIP"
  log "md5 verified ($actual)"
fi

# 3. Extraction: skipped when a previous run finished extracting this same archive.
extracted_ok() {
  local f
  [[ -f "$MARKER" && "$(cat "$MARKER")" == "$expected" ]] || return 1
  for f in "${EXPECTED_FILES[@]}"; do
    [[ -s "$OUT_DIR/$f" ]] || return 1
  done
}

if extracted_ok; then
  log "$OUT_DIR already extracted from this archive; skipping unzip"
else
  log "extracting into $OUT_DIR/"
  if have unzip; then
    unzip -q -o "$ZIP" -d "$TARGET_DIR"
  elif have python3; then
    python3 -m zipfile -e "$ZIP" "$TARGET_DIR"
  else
    die "need unzip (e.g. apt-get install -y unzip) or python3 to extract $ZIP"
  fi
  for f in "${EXPECTED_FILES[@]}"; do
    [[ -s "$OUT_DIR/$f" ]] || die "extraction incomplete: $OUT_DIR/$f is missing"
  done
  printf '%s\n' "$expected" >"$MARKER"
fi

log "ready: $OUT_DIR ($(du -sh "$OUT_DIR" | awk '{ print $1 }'))"
if [[ "$TARGET_DIR" != "$REPO_ROOT/data/raw" ]]; then
  log "non-default location: set raw_dir to $OUT_DIR in a copy of configs/data/movielens_20m.yaml"
fi
log "next: scripts/prepare_data.sh"

cat <<'LICENCE'

MovieLens-20M usage licence (GroupLens Research; full text in ml-20m/README.txt):
  * No redistribution without separate permission: keep data/raw/ and data/processed/
    out of git, Docker images, demo deployments and W&B artifacts.
  * Research purposes only; no commercial or revenue-bearing use without GroupLens permission.
  * Do not state or imply endorsement by the University of Minnesota or GroupLens.
  * Acknowledge the dataset in publications by citing:
      F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
      Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19.
      https://doi.org/10.1145/2827872
LICENCE
