"""``shopsage evaluate``: MostPop and EASE on a prepared Liang split (``Liang-SG-strict``).

Both baselines are fit on the training users. EASE's lambda is chosen on V_A, one half of the
validation users (``partition_val_users``; V_B is reserved for the ranker's labels), by
Recall@50. Test metrics are computed once, over every test user with at least one target, and
reported with percentile-bootstrap CIs over those users (2,000 resamples, seed 42), plus the
paired EASE - MostPop difference. Outputs go to ``<out_root>/<run_id>/``: ``per_user.parquet``
(one row per test user, one ``model/metric`` column each) and ``summary.json``.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import scipy.sparse as sp

from shopsage_rec import __version__
from shopsage_rec.data.liang import load_liang, partition_val_users
from shopsage_rec.evaluation.bootstrap import (
    CHUNK,
    N_RESAMPLES,
    SEED,
    bootstrap_ci,
    paired_bootstrap,
)
from shopsage_rec.evaluation.metrics import evaluate_topk, topk_for_users
from shopsage_rec.models.baselines import EASE, MostPop, gram_matrix

PROTOCOL = "Liang-SG-strict"
MODELS = ("mostpop", "ease")
DEFAULT_LAMBDAS = (250.0, 500.0, 1000.0)
SELECTION_METRIC = "recall@50"
K_MAX = 100  # deep enough for recall@20/50, ndcg@10/100 and hr@50


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    out_dir: Path
    summary: dict[str, Any]
    per_user: pl.DataFrame


def default_run_id() -> str:
    """``local-YYYYMMDD-HHMM``, the experiment-log id for runs without W&B."""
    return datetime.now().strftime("local-%Y%m%d-%H%M")


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / 2**30 if sys.platform == "darwin" else peak * 1024 / 2**30)


def _metrics(
    score: Callable[[sp.csr_matrix], np.ndarray],
    fold_in: sp.csr_matrix,
    targets: sp.csr_matrix,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    topk = topk_for_users(lambda rows: score(fold_in[rows]), fold_in, K_MAX, chunk_size=chunk_size)
    return evaluate_topk(topk, targets)


def evaluate_baselines(
    processed_dir: str | Path,
    out_root: str | Path,
    *,
    models: Sequence[str] = MODELS,
    ease_lambdas: Sequence[float] = DEFAULT_LAMBDAS,
    run_id: str | None = None,
    val_seed: int = 42,
    frac_va: float = 0.5,
    chunk_size: int = 2048,
) -> EvalResult:
    """Fit, select and test the baselines; write ``per_user.parquet`` and ``summary.json``."""
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODELS)}")
    if "ease" in models and not ease_lambdas:
        raise ValueError("EASE needs at least one lambda")
    run_id = run_id or default_run_id()
    started = time.perf_counter()
    data = load_liang(processed_dir)
    va_users, vb_users = partition_val_users(data.val_users, seed=val_seed, frac_va=frac_va)
    va_rows = np.searchsorted(data.val_users, va_users)
    va_fold_in, va_targets = data.val_fold_in[va_rows], data.val_targets[va_rows]
    timing = {"load": time.perf_counter() - started}

    test: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    if "mostpop" in models:
        t = time.perf_counter()
        mostpop = MostPop().fit(data.train)
        test["mostpop"] = _metrics(mostpop.score, data.test_fold_in, data.test_targets, chunk_size)
        timing["mostpop"] = time.perf_counter() - t
    if "ease" in models:
        t = time.perf_counter()
        gram = gram_matrix(data.train)
        selection: dict[str, float] = {}
        best: EASE | None = None
        for lam in ease_lambdas:
            candidate = EASE.from_gram(gram, float(lam))
            va_metrics = _metrics(candidate.score, va_fold_in, va_targets, chunk_size)
            selection[str(float(lam))] = float(va_metrics[SELECTION_METRIC].mean())
            if best is None or selection[str(float(lam))] > selection[str(best.lam)]:
                best = candidate
            del candidate
        del gram
        assert best is not None
        test["ease"] = _metrics(best.score, data.test_fold_in, data.test_targets, chunk_size)
        summaries["ease"] = {
            "lambda": best.lam,
            "selection": {
                "users": "V_A",
                "n_users": int(va_rows.size),
                "metric": SELECTION_METRIC,
                "va": selection,
            },
        }
        timing["ease"] = time.perf_counter() - t

    for model, metrics in test.items():
        summaries.setdefault(model, {})["test"] = {
            name: bootstrap_ci(values).as_dict() for name, values in metrics.items()
        }
    paired: dict[str, dict[str, dict[str, float | int]]] = {}
    if {"ease", "mostpop"} <= test.keys():
        paired["ease-mostpop"] = {
            name: paired_bootstrap(test["ease"][name], test["mostpop"][name]).as_dict()
            for name in test["ease"]
        }

    per_user = pl.DataFrame(
        {
            "user_idx": data.test_users,
            **{
                f"{model}/{name}": values
                for model, metrics in test.items()
                for name, values in metrics.items()
            },
        }
    )
    timing["total"] = time.perf_counter() - started
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": PROTOCOL,
        "processed_dir": str(processed_dir),
        "data": {
            "split_sha256": data.meta["split_sha256"],
            "created_at": data.meta.get("created_at"),
            "n_items": data.n_items,
            "n_train_users": int(data.train.shape[0]),
            "n_test_users": int(data.test_users.size),
            "n_va_users": int(va_rows.size),
            "n_vb_users": int(vb_users.size),
            "val_seed": val_seed,
            "frac_va": frac_va,
        },
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": SEED, "chunk": CHUNK, "unit": "users"},
        "models": summaries,
        "paired": paired,
        "timing_s": {name: round(seconds, 2) for name, seconds in timing.items()},
        "peak_rss_gb": round(_peak_rss_gb(), 2),
        "versions": {
            "shopsage_rec": __version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "polars": pl.__version__,
            "python": platform.python_version(),
        },
    }
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user.write_parquet(out_dir / "per_user.parquet")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return EvalResult(run_id=run_id, out_dir=out_dir, summary=summary, per_user=per_user)


def format_summary(summary: dict[str, Any]) -> str:
    """A plain-text table of the test metrics, ``mean [95% CI]`` per model."""
    metrics = ("recall@20", "recall@50", "ndcg@10", "ndcg@100", "hr@50")
    data = summary["data"]
    lines = [
        f"{summary['protocol']}: {data['n_test_users']:,} test users, {data['n_items']:,} items "
        f"(split {data['split_sha256'][:12]})",
        f"{'model':<10}" + "".join(f"{m:>22}" for m in metrics),
    ]
    for model, info in summary["models"].items():
        cells = []
        for m in metrics:
            s = info["test"][m]
            cells.append(f"{s['mean']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]".rjust(22))
        label = f"{model} (lambda {info['lambda']:g})" if "lambda" in info else model
        lines.append(f"{label:<10}" + "".join(cells))
    return "\n".join(lines)
