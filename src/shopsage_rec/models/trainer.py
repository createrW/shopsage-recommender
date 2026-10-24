"""Train one Two-Tower arm on the Liang split, select its best epoch on V_A, export, and test it.

Arms of ablation (a) (D-005), all in-batch sampled softmax with accidental-hit masking:

- ``A0``: no logQ correction;
- ``A1``: logQ correction (the reference, "without hard negatives");
- ``A2``: A1 plus ``num_uniform`` uniformly sampled negatives shared by the batch (mixed negative
  sampling, Yang et al. 2020), logQ over the mixed proposal ``log(B·p(j) + M/N)``;
- ``A3``: A1 plus ``mined_k`` negatives per row drawn from that user's ranks 101-500 under a
  reference run (``mined_from``; Huang et al. 2020), with no correction on those columns;
- ``A4``: full softmax over every item, no correction needed.

The epoch budget is fixed; the checkpoint with the best V_A Recall@50 is kept (no early
stopping). The kept model is exported (``item_embeddings.npy``, ``user_tower.npz``) and tested
on the test users through the numpy user tower, the path the demo uses, with the metric code of
``shopsage evaluate``. Everything lands in ``<runs_dir>/<run_id>/``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import platform
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

from shopsage_rec import __version__
from shopsage_rec.data.liang import load_liang, partition_val_users
from shopsage_rec.evaluation.bootstrap import bootstrap_ci
from shopsage_rec.evaluation.metrics import evaluate_topk, topk_for_users
from shopsage_rec.features.items import build_item_features
from shopsage_rec.models.losses import PositiveSet, sampled_softmax
from shopsage_rec.models.mining import MINED_FILE
from shopsage_rec.models.numpy_tower import user_vectors as np_user_vectors
from shopsage_rec.models.two_tower import TwoTower, sample_history

log = logging.getLogger(__name__)

ARMS: dict[str, dict[str, Any]] = {
    "A0": {"negatives": "inbatch", "logq": False},
    "A1": {"negatives": "inbatch", "logq": True},
    "A2": {"negatives": "uniform", "logq": True},
    "A3": {"negatives": "mined", "logq": True},
    "A4": {"negatives": "full", "logq": False},
}
EVAL_K = 100


@dataclass
class TrainConfig:
    arm: str = "A1"
    seed: int = 42
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 2e-3
    tau: float = 0.05
    weight_decay: float = 1e-6  # MLP weights only, never the embedding tables
    max_history: int = 50
    num_uniform: int = 2048
    mined_k: int = 8
    eval_every: int = 2
    user_frac: float = 1.0  # < 1 trains on a random share of training users (smoke runs)
    dim: int = 128
    id_dim: int = 64
    hist_dim: int = 64
    hidden: int = 256
    val_seed: int = 42
    frac_va: float = 0.5
    device: str = "auto"
    wandb: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; choose from {sorted(ARMS)}")
        if not 0.0 < self.user_frac <= 1.0:
            raise ValueError("user_frac must be in (0, 1]")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 where it is native (CUDA cc >= 8.0, MPS), fp16 with a GradScaler on older CUDA."""
    if device.type == "cuda":
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        return torch.bfloat16 if bf16 else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return None


def _peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**30 if sys.platform == "darwin" else peak * 1024 / 2**30


def _csr_tensors(m: sp.csr_matrix, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(m.indptr.astype(np.int64)).to(device),
        torch.from_numpy(m.indices.astype(np.int64)).to(device),
    )


@torch.no_grad()
def _recall_at_50(
    model: TwoTower, fold_in: sp.csr_matrix, targets: sp.csr_matrix, device: torch.device
) -> float:
    """Mean truncated Recall@50 over ``fold_in``'s users, fold-in items masked (GPU path)."""
    model.eval()
    items = model.item_vectors().float()
    tops = []
    for start in range(0, fold_in.shape[0], 2048):
        chunk = fold_in[start : start + 2048]
        indptr, indices = _csr_tensors(chunk, device)
        scores = model.user_vectors_csr(indptr, indices).float() @ items.T
        rows = torch.repeat_interleave(
            torch.arange(chunk.shape[0], device=device), indptr[1:] - indptr[:-1]
        )
        scores[rows, indices] = -torch.inf
        tops.append(torch.topk(scores, EVAL_K, dim=1).indices.cpu().numpy())
    model.train()
    metrics = evaluate_topk(np.vstack(tops), targets, recall_ks=(50,), ndcg_ks=(), hr_ks=())
    return float(metrics["recall@50"].mean())


def _wandb_run(cfg: TrainConfig, run_id: str) -> Any:
    if not cfg.wandb or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        return None
    if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "offline"  # no login: log locally, `wandb sync` later
    return wandb.init(
        project="shopsage-recommender", name=run_id, group=cfg.arm, config=asdict(cfg)
    )


def train_two_tower(
    processed_dir: str | Path,
    runs_dir: str | Path,
    cfg: TrainConfig,
    *,
    run_id: str | None = None,
    mined_from: str | Path | None = None,
) -> dict[str, Any]:
    """Train ``cfg.arm``, export the best epoch and return the run's summary (also written to
    ``metrics.json``)."""
    started = time.perf_counter()
    run_id = run_id or f"{cfg.arm}-s{cfg.seed}"
    out = Path(runs_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    arm = ARMS[cfg.arm]
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = pick_device(cfg.device)
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    data = load_liang(processed_dir)
    feats = build_item_features(processed_dir, data.meta["config"]["raw_dir"])
    n_items = data.n_items
    train = data.train
    if cfg.user_frac < 1.0:
        keep = np.sort(rng.choice(train.shape[0], int(train.shape[0] * cfg.user_frac), False))
        train = train[keep]
    indptr, indices = _csr_tensors(train, device)
    n_rows = int(indices.numel())
    row_user = torch.repeat_interleave(
        torch.arange(train.shape[0], device=device), indptr[1:] - indptr[:-1]
    )
    item_p = torch.from_numpy(
        (np.asarray(train.sum(axis=0), dtype=np.float64).ravel() / max(n_rows, 1)).astype(
            np.float32
        )
    ).to(device)
    positives = PositiveSet(row_user, indices, n_items).to(device)

    def logq(items: torch.Tensor) -> torch.Tensor | None:
        if not arm["logq"]:
            return None
        mixed = cfg.batch_size * item_p[items]
        if arm["negatives"] == "uniform":
            mixed = mixed + cfg.num_uniform / n_items
        return torch.log(mixed.clamp_min(1e-12)).float()

    mined: torch.Tensor | None = None
    if arm["negatives"] == "mined":
        if mined_from is None:
            raise ValueError("arm A3 needs mined_from, a run directory holding " + MINED_FILE)
        if cfg.user_frac < 1.0:
            raise ValueError("mined negatives are aligned with all training users (user_frac 1)")
        mined = torch.from_numpy(np.load(Path(mined_from) / MINED_FILE).astype(np.int64)).to(device)

    va_users, _ = partition_val_users(data.val_users, seed=cfg.val_seed, frac_va=cfg.frac_va)
    va_rows = np.searchsorted(data.val_users, va_users)
    va_fold_in, va_targets = data.val_fold_in[va_rows], data.val_targets[va_rows]

    model = TwoTower(
        feats, dim=cfg.dim, id_dim=cfg.id_dim, hist_dim=cfg.hist_dim, hidden=cfg.hidden
    ).to(device)
    embeddings = [
        p for n, p in model.named_parameters() if n.endswith(("_id.weight", "_emb.weight"))
    ]
    others = [
        p for n, p in model.named_parameters() if not n.endswith(("_id.weight", "_emb.weight"))
    ]
    groups: list[dict[str, Any]] = [
        {"params": embeddings, "weight_decay": 0.0},
        {"params": others},
    ]
    optimizer = torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=device.type in ("cuda", "mps"),
    )
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    wb = _wandb_run(cfg, run_id)

    curve: list[dict[str, float]] = []
    best = {"recall@50": -1.0, "epoch": 0}
    best_state: dict[str, torch.Tensor] = {}
    steps_per_epoch = math.ceil(n_rows / cfg.batch_size)
    log.info(
        "%s: %d rows, %d steps/epoch on %s (%s)", run_id, n_rows, steps_per_epoch, device, dtype
    )
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        total = 0.0
        perm = torch.randperm(n_rows, device=device, generator=gen)
        for step in range(steps_per_epoch):
            rows = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            users = row_user[rows]
            pos_items = indices[rows]
            hist, weight, n_hist = sample_history(
                indptr, indices, rows, users, cfg.max_history, generator=gen
            )
            with torch.autocast(
                device.type, dtype=dtype or torch.float32, enabled=dtype is not None
            ):
                u = model.user_vectors(hist, weight, n_hist)
                if arm["negatives"] == "full":
                    all_items = model.item_vectors()
                    loss, _, _ = sampled_softmax(
                        u, all_items[pos_items], pos_items, users, tau=cfg.tau,
                        positives=positives, logq=None, inbatch=False,
                        shared=(all_items, torch.arange(n_items, device=device), None),
                    )  # fmt: skip
                else:
                    shared = rowwise = None
                    if arm["negatives"] == "uniform":
                        neg = torch.randint(
                            n_items, (cfg.num_uniform,), device=device, generator=gen
                        )
                        shared = (model.item_vectors(neg), neg, logq(neg))
                    if mined is not None:
                        pick = torch.randint(
                            mined.shape[1],
                            (rows.shape[0], cfg.mined_k),
                            device=device,
                            generator=gen,
                        )
                        neg_items = mined[users[:, None], pick]
                        neg_vecs = model.item_vectors(neg_items.reshape(-1)).view(
                            rows.shape[0], cfg.mined_k, -1
                        )
                        rowwise = (neg_vecs, neg_items)
                    loss, _, _ = sampled_softmax(
                        u, model.item_vectors(pos_items), pos_items, users, tau=cfg.tau,
                        positives=positives, logq=logq(pos_items), shared=shared, rowwise=rowwise,
                    )  # fmt: skip
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        epoch_s = time.perf_counter() - t0
        point: dict[str, float] = {
            "epoch": epoch,
            "loss": total / steps_per_epoch,
            "epoch_s": epoch_s,
        }
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            point["va_recall@50"] = _recall_at_50(model, va_fold_in, va_targets, device)
            if point["va_recall@50"] > best["recall@50"]:
                best = {"recall@50": point["va_recall@50"], "epoch": epoch}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        point["peak_memory_gb"] = _peak_memory_gb(device)
        curve.append(point)
        log.info("%s epoch %d: %s", run_id, epoch, point)
        if wb is not None:
            wb.log(point)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        item_emb = model.item_vectors().float().cpu().numpy()
    weights = model.export_user_tower()
    np.save(out / "item_embeddings.npy", item_emb)
    np.savez(out / "user_tower.npz", **weights)  # type: ignore[arg-type]  # numpy stub overload
    torch.save(copy.deepcopy(best_state), out / "model.pt")

    test_vecs = np_user_vectors(data.test_fold_in, weights)
    topk = topk_for_users(
        lambda r: test_vecs[r] @ item_emb.T, data.test_fold_in, EVAL_K, chunk_size=2048
    )
    test = evaluate_topk(topk, data.test_targets)
    pl.DataFrame({"user_idx": data.test_users, **test}).write_parquet(out / "per_user.parquet")

    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "Liang-SG-strict",
        "config": asdict(cfg),
        "arm": {"name": cfg.arm, **arm},
        "mined_from": str(mined_from) if mined_from else None,
        "data": {
            "split_sha256": data.meta["split_sha256"],
            "n_items": n_items,
            "n_train_users": int(train.shape[0]),
            "n_train_rows": n_rows,
            "n_va_users": int(va_rows.size),
            "n_test_users": int(data.test_users.size),
        },
        "device": str(device),
        "amp": str(dtype),
        "best_epoch": best["epoch"],
        "va_recall@50": best["recall@50"],
        "curve": curve,
        "test": {name: bootstrap_ci(values).as_dict() for name, values in test.items()},
        "peak_memory_gb": round(max(p["peak_memory_gb"] for p in curve), 3),
        "wall_s": round(time.perf_counter() - started, 1),
        "versions": {
            "shopsage_rec": __version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(device)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wb is not None:
        wb.summary.update({f"test/{k}": v["mean"] for k, v in summary["test"].items()})
        wb.finish()
    log.info("%s: best epoch %d, test %s", run_id, best["epoch"], summary["test"]["recall@50"])
    return summary
