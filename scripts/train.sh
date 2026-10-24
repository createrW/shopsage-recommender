#!/usr/bin/env bash
# The S2 Two-Tower sweep on this machine, one run after another (RunPod or any Linux GPU box,
# after scripts/bootstrap_cloud.sh). Modal runs the same sweep in parallel:
# `modal run scripts/modal_train.py`.
#
# Usage: scripts/train.sh [--sweep] [shopsage sweep options]   e.g. --epochs 30 --no-wandb
#        scripts/train.sh --one [shopsage train options]       e.g. --arm A1 --seed 42
# Results: artifacts/runs/<run_id>/ and artifacts/runs/sweep_summary.json.
set -euo pipefail
unset CDPATH
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode=sweep
if [[ "${1:-}" == "--sweep" ]]; then
  shift
elif [[ "${1:-}" == "--one" ]]; then
  mode=train
  shift
fi
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
  echo "W&B: no WANDB_API_KEY; logging offline (run 'wandb sync' later)" >&2
  export WANDB_MODE=offline
fi
exec uv run --no-sync shopsage "$mode" "$@"
