#!/usr/bin/env bash
# One-command setup for a fresh Linux GPU box (RunPod pod, `modal shell`, Colab/Kaggle terminal):
#   install uv -> uv sync --python 3.12 --extra train --extra gpu --extra llm -> download -> prepare,
# then show nvidia-smi, the VRAM guidance and the next command.
# Safe to re-run: every step is idempotent (uv sync, md5-verified download, seeded prepare).
#
# Usage: scripts/bootstrap_cloud.sh [--no-gpu] [--skip-data]
#   --no-gpu     sync the cpu extra (faiss-cpu) instead of the gpu extra (faiss-gpu); implied when
#                no GPU is visible, GPU 0 is outside sm_70..sm_90, or the host is not Linux x86_64
#                (faiss-gpu has no other kernels or wheels).
#                The two extras conflict in pyproject.toml (both install the `faiss` package), so
#                exactly one is synced. torch is the same build either way.
#   --skip-data  environment only: no download / prepare
# Env: SHOPSAGE_PYTHON (default 3.12); UV_LOCKED=1 makes uv fail if uv.lock is out of date.
#
# torch on Linux comes from the cu126 index pinned in pyproject.toml ([tool.uv.sources]), so it
# runs on NVIDIA drivers >= 525 (Colab, Kaggle and RunPod T4 hosts). Blackwell GPUs (RTX 50xx,
# B200) need the cu130 index instead (driver >= 580): switch the URL in pyproject.toml, run
# `uv lock`, re-run. The checks below read the index from pyproject.toml, so they follow a switch.
set -euo pipefail
unset CDPATH

log() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--no-gpu] [--skip-data]

Fresh Linux GPU box -> uv, Python env (train + gpu + llm extras), MovieLens-20M download and
prepare. Re-runnable. Env: SHOPSAGE_PYTHON (default 3.12), UV_LOCKED=1 to require uv.lock.
  --no-gpu     cpu extra (faiss-cpu) instead of gpu (faiss-gpu); implied when nvidia-smi sees
               no GPU, GPU 0 is outside sm_70..sm_90, or the host is not Linux x86_64
  --skip-data  skip download + prepare
USAGE
}

WITH_GPU=1
WITH_DATA=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-gpu) WITH_GPU=0 ;;
    --skip-data) WITH_DATA=0 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# Not PYTHON_VERSION: the official python images export it (e.g. 3.13.x), overriding the default.
SHOPSAGE_PYTHON="${SHOPSAGE_PYTHON:-3.12}"

# CUDA flavour of the torch index pinned in pyproject.toml ([[tool.uv.index]] url).
TORCH_CUDA="$(sed -n 's|^url = "https://download\.pytorch\.org/whl/\(cu[0-9]*\)".*|\1|p' pyproject.toml | head -n 1)"
case "$TORCH_CUDA" in
  cu12*) MIN_DRIVER=525 ;; # CUDA 12.x minor-version compatibility
  cu13*) MIN_DRIVER=580 ;;
  *) MIN_DRIVER="" ;;
esac

# --- 1. Host: GPU, driver and disk ---------------------------------------------------------
log "host"
OS="$(uname -s)"
ARCH="$(uname -m)"
[[ "$OS" == "Linux" ]] || warn "written for Linux GPU boxes; on macOS use 'make setup-train'"
HAS_GPU=0
DRIVER_VERSION=""
DRIVER_CUDA=""
COMPUTE_CAP=""
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
  nvidia-smi -L | sed 's/^/    /'
  DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1 || true)"
  # compute_cap needs a recent driver; an empty value only skips the architecture checks below.
  COMPUTE_CAP="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ' || true)"
  info "NVIDIA driver ${DRIVER_VERSION:-?}, supports CUDA up to ${DRIVER_CUDA:-?}; GPU 0 compute capability ${COMPUTE_CAP:-?}"
  if [[ -n "$TORCH_CUDA" ]]; then
    info "torch comes from the $TORCH_CUDA index (pyproject.toml), which needs driver >= ${MIN_DRIVER:-?}"
  else
    info "torch index: see [tool.uv.sources] in pyproject.toml"
  fi
  if [[ "$TORCH_CUDA" != cu13* ]]; then
    info "Blackwell GPUs (RTX 50xx, B200) need the cu130 index switch in pyproject.toml instead"
  fi
else
  warn "no NVIDIA GPU visible (nvidia-smi missing or failing)"
fi
# Checked before the multi-GB download, not after it.
driver_major="${DRIVER_VERSION%%.*}"
if [[ "$driver_major" =~ ^[0-9]+$ && -n "$MIN_DRIVER" ]] && ((driver_major < MIN_DRIVER)); then
  warn "driver $DRIVER_VERSION predates $MIN_DRIVER: the $TORCH_CUDA torch wheels will not see the GPU; use a host with a newer driver"
fi
cc_major="${COMPUTE_CAP%%.*}"
if [[ "$cc_major" =~ ^[0-9]+$ ]]; then
  sm="sm_${COMPUTE_CAP//./}"
  if ((cc_major >= 10)) && [[ "$TORCH_CUDA" != cu13* ]]; then
    warn "GPU 0 is Blackwell ($sm): the ${TORCH_CUDA:-cu126} torch wheels have no kernels for it. Set the pytorch index URL in pyproject.toml to https://download.pytorch.org/whl/cu130 (driver >= 580), run 'uv lock', and re-run."
  fi
  if ((cc_major < 7 || cc_major >= 10)) && ((WITH_GPU == 1)); then
    warn "GPU 0 is $sm but faiss-gpu is built for sm_70..sm_90: syncing the cpu extra (faiss-cpu) instead (exact search over 13,681 items runs on the CPU)"
    WITH_GPU=0
  fi
fi
if ((WITH_GPU == 1)); then
  if ((HAS_GPU == 0)); then
    warn "no GPU: syncing the cpu extra (faiss-cpu) instead of the gpu extra"
    WITH_GPU=0
  elif [[ "$OS" != "Linux" || "$ARCH" != "x86_64" ]]; then
    warn "faiss-gpu has Linux x86_64 wheels only (this host: $OS $ARCH): syncing the cpu extra (faiss-cpu)"
    WITH_GPU=0
  fi
fi
avail_gb="$(df -Pk . | awk 'NR == 2 { printf "%d", $4 / 1048576 }')"
info "free disk: ${avail_gb} GB"
if [[ "$avail_gb" =~ ^[0-9]+$ ]] && ((avail_gb < 15)); then
  warn "under 15 GB free: the CUDA PyTorch wheels are ~3 GB compressed and raw MovieLens-20M is ~1.1 GB"
fi

# --- 2. uv -----------------------------------------------------------------------------------
log "uv"
UV_BIN_DIR="${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
case ":$PATH:" in
  *":$UV_BIN_DIR:"*) ;;
  *) export PATH="$UV_BIN_DIR:$PATH" ;;
esac
if have uv; then
  info "found $(uv --version)"
else
  if have curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif have wget; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif have python3 && python3 -m pip --version >/dev/null 2>&1; then
    python3 -m pip install --upgrade uv
  else
    die "need curl, wget or pip to install uv (e.g. apt-get update && apt-get install -y curl)"
  fi
  have uv || die "uv was installed but is not on PATH; add $UV_BIN_DIR to PATH and re-run"
  info "installed $(uv --version)"
fi

# --- 3. Python environment -------------------------------------------------------------------
# uv sync is exact: a re-run with the other FAISS extra removes the build installed before.
if ((WITH_GPU == 1)); then FAISS_EXTRA=gpu; else FAISS_EXTRA=cpu; fi
EXTRAS=(--extra train --extra "$FAISS_EXTRA" --extra llm)
log "uv sync --python $SHOPSAGE_PYTHON ${EXTRAS[*]}"
uv sync --python "$SHOPSAGE_PYTHON" "${EXTRAS[@]}"
# Venv first on PATH: its python3 backs the unzip fallback in download_data.sh on bare images.
export PATH="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv}/bin:$PATH"

# --- 4. Sanity check: torch can use the GPU, lightgbm and faiss import -------------------------
log "sanity check"
HAS_GPU="$HAS_GPU" DRIVER_CUDA="$DRIVER_CUDA" TORCH_CUDA="$TORCH_CUDA" MIN_DRIVER="$MIN_DRIVER" \
  FAISS_EXTRA="$FAISS_EXTRA" uv run --no-sync python - <<'PY' || warn "sanity check failed; see the messages above"
import os
from importlib.metadata import PackageNotFoundError, version

import torch

ok = torch.cuda.is_available()
print(f"    torch {torch.__version__} (built for CUDA {torch.version.cuda}); CUDA usable: {ok}")
if ok:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        # Gate on the compute capability: is_bf16_supported() may also count emulated bf16.
        amp = "bf16" if p.major >= 8 else "fp16 + GradScaler (no bf16 below sm_80, e.g. T4)"
        print(
            f"    cuda:{i} {p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}; "
            f"AMP dtype: {amp}"
        )
    try:
        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum().item()
        print("    CUDA matmul OK")
    except RuntimeError as exc:
        print(f"WARNING: CUDA kernel launch failed: {exc}")
        print(f"         this torch build targets {torch.cuda.get_arch_list()}")
elif os.environ.get("HAS_GPU") == "1":
    index = os.environ.get("TORCH_CUDA") or "pinned"
    print(
        "WARNING: a GPU is visible but torch cannot use it. torch is built for CUDA "
        f"{torch.version.cuda}; this driver supports CUDA {os.environ.get('DRIVER_CUDA') or '?'}.\n"
        f"         The {index} wheels need driver >= {os.environ.get('MIN_DRIVER') or '?'}; "
        "Blackwell GPUs need the cu130 index\n"
        "         (switch the URL in pyproject.toml, run `uv lock`, re-run this script)."
    )

try:
    import lightgbm

    print(f"    lightgbm {lightgbm.__version__}")
except OSError as exc:  # the wheel links the OpenMP runtime dynamically
    print(f"WARNING: lightgbm failed to load ({exc}); apt-get install -y libgomp1")

for dist in ("faiss-gpu", "faiss-cpu"):
    try:
        print(f"    {dist} {version(dist)} installed")
    except PackageNotFoundError:
        pass
try:
    import faiss

    n_gpu = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
    print(f"    faiss {faiss.__version__}; GPUs visible to faiss: {n_gpu}")
    if os.environ.get("FAISS_EXTRA") == "gpu" and n_gpu == 0:
        print("WARNING: faiss-gpu is installed but sees no GPU; re-run with --no-gpu for faiss-cpu")
except Exception as exc:
    print(f"WARNING: faiss failed to import: {exc!r}")
PY

# --- 5. Data ---------------------------------------------------------------------------------
if [[ $WITH_DATA -eq 1 ]]; then
  log "data: download + md5 verify"
  bash scripts/download_data.sh
  log "data: prepare"
  bash scripts/prepare_data.sh
else
  log "data: skipped (--skip-data)"
fi

# --- 6. Summary ------------------------------------------------------------------------------
log "nvidia-smi"
if [[ $HAS_GPU -eq 1 ]]; then
  nvidia-smi || warn "nvidia-smi failed"
  mem_mib="$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
  # nvidia-smi reports a little under the marketed size (a 16 GB T4 shows 15360 MiB), hence the slack.
  if [[ "$mem_mib" =~ ^[0-9]+$ ]]; then
    if ((mem_mib < 7500)); then
      warn "GPU 0 has ${mem_mib} MiB, below the 8 GB minimum; smaller batches may still fit (B=4096 ~1.4 GB, B=8192 ~2.5 GB)"
    elif ((mem_mib < 15000)); then
      info "GPU 0 has ${mem_mib} MiB: above the 8 GB minimum, below the recommended 16 GB"
    else
      info "GPU 0 has ${mem_mib} MiB: meets the recommended 16 GB"
    fi
  fi
else
  info "no NVIDIA GPU on this box"
fi

cat <<'END_VRAM'

VRAM (roofline estimates, +-30%, to be measured on Day 3):
  GPU tiers         minimum 8 GB; recommended 16 GB or more (A10G 24 GB is the best value)
  Two-Tower train   136,674 x 128 user and 13,681 x 128 item tables, in-batch softmax, AMP; peak:
                      B=4096 ~1.4 GB | B=8192 ~2.5 GB | B=16384 + 4,096 sampled negatives ~6.1 GB (worst case)
  evaluation        score users in chunks of <= 8,192; one 136,674 x 13,681 fp32 score matrix is 7.5 GB
  BGE-M3 fp16       ~1.1 GB of weights (2.3 GB fp32 checkpoint); embeds the 13,681 kept items in ~1 min
  vLLM (optional)   Qwen/Qwen2.5-0.5B-Instruct reserves --gpu-memory-utilization x VRAM: 0.2 = ~3 GB on a T4,
                      ~4.5 GB on an A10G (use ~0.35 on an 8 GB card); --max-model-len 4096
                      --enforce-eager (--dtype half on T4); never at vLLM defaults next to training
  LightGBM ranker   CPU/RAM only: ~2 GB for 20K users x 500 candidates (~10M rows)
END_VRAM

log "next"
if [[ -f scripts/train.sh ]]; then
  info "bash scripts/train.sh"
else
  info "bash scripts/train.sh   (arrives on Day 3; not in the repo yet)"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  info "W&B: WANDB_API_KEY not set; run 'uv run wandb login' or export WANDB_MODE=offline"
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  info "LLM expansion (Day 8): ANTHROPIC_API_KEY not set"
fi
info "new shells: export PATH=\"$UV_BIN_DIR:\$PATH\" if uv is not found"
