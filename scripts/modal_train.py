"""Run the S2 Two-Tower sweep on Modal (docs/plan.md §6.3): one command, GPUs in parallel.

    uv tool install modal && modal setup              # once
    modal secret create wandb WANDB_API_KEY=...       # once (or WANDB_MODE=offline to skip W&B)
    modal run scripts/modal_train.py                  # the whole sweep, ~1.5-2 h (estimate)
    modal volume get shopsage-data artifacts/runs artifacts/   # copy the results back

What it does, all on the `shopsage-data` Modal Volume:

1. ``prepare`` (CPU): downloads MovieLens-20M from GroupLens (md5-checked), builds the Liang split
   and the item features; skipped when they already exist.
2. ``train`` (one A10 GPU per run, in parallel): a 4-run tau / learning-rate check on A1, then
   A0, A1 x 3 seeds, A2, A4, then (after ``mine``) A3 x 3 seeds (``shopsage_rec.models.sweep``).
   A run whose ``metrics.json`` already exists is not retrained, so a failed sweep can simply be
   started again.
3. ``mine`` (CPU): ranks 101-500 of every training user under A1 seed 42, for A3.

The summary lands in ``artifacts/runs/sweep_summary.json`` on the Volume and locally. Options:
``--epochs`` (30), ``--tune-epochs`` (10); ``SHOPSAGE_GPU=L4 modal run ...`` picks another GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE = "/root/shopsage"
VOLUME_PATH = "/vol"
GPU = os.environ.get("SHOPSAGE_GPU", "A10")
PROCESSED = "data/processed/liang"
RUNS = "artifacts/runs"

sys.path.insert(0, str(REPO / "src"))  # the local entrypoint imports the (pure Python) sweep plan
from shopsage_rec.models.sweep import Job, run_sweep  # noqa: E402

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "unzip", "libgomp1")
    .uv_sync(str(REPO), extras=["train"])
    .env({"PYTHONPATH": f"{REMOTE}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(REPO / "pyproject.toml"), f"{REMOTE}/pyproject.toml")
    .add_local_dir(str(REPO / "configs"), f"{REMOTE}/configs")
    .add_local_dir(str(REPO / "scripts"), f"{REMOTE}/scripts")
    .add_local_dir(str(REPO / "src"), f"{REMOTE}/src")
)
volume = modal.Volume.from_name("shopsage-data", create_if_missing=True)
app = modal.App("shopsage-train", image=image)


def _workdir() -> None:
    """Point the checkout's data/ and artifacts/ at the Volume and work from the checkout."""
    for name in ("data", "artifacts"):
        target = Path(VOLUME_PATH) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(REMOTE) / name
        if not link.exists():
            link.symlink_to(target)
    os.chdir(REMOTE)


@app.function(volumes={VOLUME_PATH: volume}, cpu=4, memory=16384, timeout=3600)
def prepare() -> dict[str, Any]:
    _workdir()
    meta = Path(PROCESSED) / "splits" / "meta.json"
    if not meta.is_file():
        subprocess.run(["bash", "scripts/download_data.sh"], check=True)
        subprocess.run(
            [sys.executable, "-m", "shopsage_rec.cli", "prepare"], check=True
        )  # the default config is the Liang split
    from shopsage_rec.features.items import build_item_features

    data_meta = json.loads(meta.read_text())
    build_item_features(PROCESSED, data_meta["config"]["raw_dir"])  # cached for the GPU jobs
    volume.commit()
    return {k: data_meta[k] for k in ("split_sha256", "n_users", "n_items", "n_train")}


@app.function(
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=4 * 3600,
)
def train(job: dict[str, Any]) -> dict[str, Any]:
    _workdir()
    volume.reload()
    done = Path(RUNS) / job["run_id"] / "metrics.json"
    if done.is_file():
        return json.loads(done.read_text())
    from shopsage_rec.models.trainer import TrainConfig, train_two_tower

    cfg = TrainConfig(
        arm=job["arm"], seed=job["seed"], epochs=job["epochs"], tau=job["tau"], lr=job["lr"]
    )
    mined = Path(RUNS) / job["mined_from"] if job["mined_from"] else None
    summary = train_two_tower(PROCESSED, RUNS, cfg, run_id=job["run_id"], mined_from=mined)
    volume.commit()
    return summary


@app.function(volumes={VOLUME_PATH: volume}, cpu=8, memory=32768, timeout=3600)
def mine(run_id: str) -> str:
    _workdir()
    volume.reload()
    from shopsage_rec.models.mining import mine_hard_negatives

    path = mine_hard_negatives(PROCESSED, Path(RUNS) / run_id)
    volume.commit()
    return str(path)


@app.function(volumes={VOLUME_PATH: volume}, timeout=600)
def save_summary(summary: dict[str, Any]) -> None:
    _workdir()
    volume.reload()
    Path(RUNS).mkdir(parents=True, exist_ok=True)
    (Path(RUNS) / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()


@app.local_entrypoint()
def main(epochs: int = 30, tune_epochs: int = 10) -> None:
    print("data:", prepare.remote())

    def execute(jobs: list[Job]) -> list[dict[str, Any]]:
        print("training:", ", ".join(job.run_id for job in jobs))
        return list(train.map([job.as_dict() for job in jobs]))

    def mine_from(run_id: str) -> None:
        print("mined:", mine.remote(run_id))

    summary = run_sweep(execute, mine_from, epochs=epochs, tune_epochs=tune_epochs)
    save_summary.remote(summary)
    local = REPO / RUNS / "sweep_summary.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["arms"], indent=2))
    print(f"summary: {local} (and {RUNS}/sweep_summary.json on the shopsage-data Volume)")
