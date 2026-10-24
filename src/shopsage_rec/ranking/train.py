"""``shopsage rank``: train the LightGBM LambdaRank re-ranker and test the two-stage system (D-006).

- **Data:** candidates and features (``ranking/features.py``) for V_B, the half of the validation
  users not used for model selection; labels are their held-out items. 10% of V_B users (seed 0)
  are held out for early stopping on NDCG@10. Groups whose candidates hold no target are dropped
  from training (counted, never filled with injected positives).
- **Arms:** ``ranker`` (all features) and ``ranker-no-ease`` (without ``ease_score``), each with the
  given LightGBM seeds; ``ease`` (dense EASE, λ from E004) and ``dense`` (the dense run's own order)
  as references, computed here on the same test users.
- **Test:** each ranker re-orders every test user's candidates; Recall / NDCG / HR at the usual
  cut-offs, so the system's Recall@50 counts targets the candidates missed as misses. Paired
  deltas use the per-user mean over seeds.
- **Latency:** the whole serving path for one user at a time on CPU (user tower, dense and BM25
  scoring, candidates, features, LightGBM on one thread, top 10), over ``latency_users`` test
  users.

Outputs in ``<out_root>/<run_id>/``: ``models/<arm>-s<seed>.txt``, ``per_user.parquet``,
``summary.json``.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp

from shopsage_rec import __version__
from shopsage_rec.data.liang import load_liang, partition_val_users
from shopsage_rec.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from shopsage_rec.evaluation.metrics import evaluate_topk, topk_for_users
from shopsage_rec.models.baselines import EASE
from shopsage_rec.ranking.features import FEATURES, RankerContext, Stage, build_stage, labels_for

K_EVAL = 100
PAIRED_METRICS = ("recall@50", "ndcg@10")
REPORTED = ("recall@20", "recall@50", "ndcg@10", "ndcg@100", "hr@50")
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10, 50],
    "lambdarank_truncation_level": 50,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "first_metric_only": True,
    "verbosity": -1,
}
ARMS = {"ranker": FEATURES, "ranker-no-ease": tuple(f for f in FEATURES if f != "ease_score")}


@dataclass(frozen=True)
class RankResult:
    out_dir: Path
    summary: dict[str, Any]


def _subset(stage: Stage, users: np.ndarray) -> Stage:
    """The candidates of the given user rows (ascending), re-indexed from 0."""
    starts, stops = stage.offsets[users], stage.offsets[users + 1]
    take = np.concatenate([np.arange(a, b) for a, b in zip(starts, stops, strict=True)])
    offsets = np.concatenate([[0], np.cumsum(stops - starts)]).astype(np.int64)
    return Stage(offsets, stage.items[take], stage.features[take])


def _dataset(stage: Stage, labels: np.ndarray, columns: list[int]) -> tuple[lgb.Dataset, int]:
    """A LambdaRank dataset without the groups that hold no positive; returns it and their count."""
    sizes = np.diff(stage.offsets)
    has_pos = np.add.reduceat(labels, stage.offsets[:-1]) > 0 if labels.size else np.array([])
    keep_rows = np.repeat(has_pos, sizes)
    data = lgb.Dataset(
        stage.features[keep_rows][:, columns],
        label=labels[keep_rows],
        group=sizes[has_pos],
        feature_name=[FEATURES[c] for c in columns],
        free_raw_data=False,
    )
    return data, int((~has_pos).sum())


def _ranked(stage: Stage, scores: np.ndarray, k: int) -> np.ndarray:
    """Each user's top ``k`` candidates by score (ties: lower item index)."""
    order = np.lexsort((stage.items, -scores, stage.rows))
    out = np.empty((stage.offsets.size - 1, k), dtype=np.int64)
    for u in range(out.shape[0]):
        out[u] = stage.items[order[stage.offsets[u] : stage.offsets[u] + k]]
    return out


def _ease_scores(ease: EASE, fold_in: sp.csr_matrix, rows: slice) -> np.ndarray:
    return ease.score(fold_in[rows])


def _ms(samples: list[float]) -> list[float]:
    p50, p95 = np.percentile(np.asarray(samples) * 1e3, [50, 95])
    return [round(float(p50), 2), round(float(p95), 2)]


def _latency(
    ctx: RankerContext,
    booster: lgb.Booster,
    fold_in: sp.csr_matrix,
    rows: np.ndarray,
    depth: tuple[int, int],
) -> dict[str, Any]:
    timings: dict[str, list[float]] = {"retrieve_and_features": [], "predict": [], "total": []}
    for row in rows.tolist():
        t0 = time.perf_counter()
        stage = build_stage(fold_in[row : row + 1], ctx, depth=depth)
        t1 = time.perf_counter()
        scores = np.asarray(booster.predict(stage.features, num_threads=1))
        stage.items[np.lexsort((stage.items, -scores))[:10]]  # the top 10 shown to the user
        t2 = time.perf_counter()
        timings["retrieve_and_features"].append(t1 - t0)
        timings["predict"].append(t2 - t1)
        timings["total"].append(t2 - t0)
    return {"users": int(rows.size), **{k: _ms(v) for k, v in timings.items()}}


def train_rankers(
    processed_dir: str | Path,
    runs_dir: str | Path,
    out_root: str | Path,
    *,
    dense_run: str = "A1-s42",
    seeds: Sequence[int] = (42, 43, 44),
    depth: tuple[int, int] = (500, 100),
    ease_lambda: float = 1000.0,
    neighbours: int = 100,
    params: Mapping[str, Any] | None = None,
    max_rounds: int = 1000,
    early_stopping: int = 50,
    latency_users: int = 500,
    run_id: str | None = None,
) -> RankResult:
    if depth[0] < K_EVAL:
        raise ValueError(f"the dense depth must be at least {K_EVAL}, got {depth[0]}")
    started = time.perf_counter()
    run_id = run_id or datetime.now().strftime("rank-%Y%m%d-%H%M")
    out = Path(out_root) / run_id
    (out / "models").mkdir(parents=True, exist_ok=True)
    data = load_liang(processed_dir)
    fold_in, targets = data.test_fold_in, data.test_targets

    ease = EASE.fit(data.train, ease_lambda)
    per_user: dict[str, dict[str, np.ndarray]] = {
        "ease": evaluate_topk(
            topk_for_users(partial(_ease_scores, ease, fold_in), fold_in, K_EVAL, chunk_size=2048),
            targets,
        )
    }
    ctx = RankerContext.load(
        processed_dir, data, Path(runs_dir) / dense_run, ease_weights=ease.weights,
        neighbours=neighbours,
    )  # fmt: skip
    del ease

    _, vb_users = partition_val_users(data.val_users)
    vb_rows = np.searchsorted(data.val_users, vb_users)
    vb = build_stage(data.val_fold_in[vb_rows], ctx, depth=depth)
    vb_labels = labels_for(vb, data.val_targets[vb_rows])
    rng = np.random.default_rng(0)
    stop = np.zeros(vb_rows.size, dtype=bool)
    stop[rng.choice(vb_rows.size, max(1, round(0.1 * vb_rows.size)), replace=False)] = True
    fit_users, stop_users = np.flatnonzero(~stop), np.flatnonzero(stop)
    fit_stage, stop_stage = _subset(vb, fit_users), _subset(vb, stop_users)
    fit_labels = labels_for(fit_stage, data.val_targets[vb_rows][fit_users])
    stop_labels = labels_for(stop_stage, data.val_targets[vb_rows][stop_users])

    test = build_stage(fold_in, ctx, depth=depth)
    dense_order = np.lexsort(
        (test.items, test.features[:, FEATURES.index("dense_rank")], test.rows)
    )
    dense_top = np.stack(
        [test.items[dense_order[a : a + K_EVAL]] for a in test.offsets[:-1].tolist()]
    )
    per_user["dense"] = evaluate_topk(dense_top, targets)

    lgb_params = {**DEFAULT_PARAMS, **(params or {})}
    arms: dict[str, Any] = {}
    importance: dict[str, dict[str, float]] = {}
    train_info: dict[str, Any] = {}
    boosters: dict[str, lgb.Booster] = {}
    for arm, names in ARMS.items():
        columns = [FEATURES.index(n) for n in names]
        fit_set, dropped = _dataset(fit_stage, fit_labels, columns)
        stop_set, stop_dropped = _dataset(stop_stage, stop_labels, columns)
        train_info.update(groups_without_positive=dropped + stop_dropped)
        gains, runs = [], {}
        for seed in seeds:
            booster = lgb.train(
                {**lgb_params, "seed": seed, "deterministic": True},
                fit_set,
                num_boost_round=max_rounds,
                valid_sets=[stop_set],
                callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
            )
            booster.save_model(str(out / "models" / f"{arm}-s{seed}.txt"))
            scores = np.asarray(
                booster.predict(
                    test.features[:, columns],
                    num_iteration=booster.best_iteration,
                    num_threads=lgb_params.get("num_threads", 0),  # 0: OpenMP's default
                )
            )
            metrics = evaluate_topk(_ranked(test, scores, K_EVAL), targets)
            per_user[f"{arm}:s{seed}"] = metrics
            gains.append(booster.feature_importance("gain", iteration=booster.best_iteration))
            runs[str(seed)] = {
                "best_iteration": int(booster.best_iteration),
                "stop_ndcg@10": float(booster.best_score["valid_0"]["ndcg@10"]),
                **{m: float(metrics[m].mean()) for m in REPORTED},
            }
            if arm == "ranker" and seed == seeds[0]:
                boosters[arm] = booster
        total = np.mean(gains, axis=0)
        importance[arm] = {
            n: round(float(g / total.sum()), 4) for n, g in zip(names, total, strict=True)
        }
        per_user[arm] = {
            m: np.mean([per_user[f"{arm}:s{s}"][m] for s in seeds], axis=0)
            for m in per_user["ease"]
        }
        arms[arm] = {
            "seeds": len(seeds),
            "runs": runs,
            **{
                m: {
                    "mean": float(np.mean([r[m] for r in runs.values()])),
                    "sd": float(np.std([r[m] for r in runs.values()], ddof=1))
                    if len(seeds) > 1
                    else None,
                    "ci_seed_mean": bootstrap_ci(per_user[arm][m]).as_dict(),
                }
                for m in REPORTED
            },
        }
    for ref in ("ease", "dense"):
        arms[ref] = {m: bootstrap_ci(per_user[ref][m]).as_dict() for m in REPORTED}

    def delta(a: str, b: str) -> dict[str, Any]:
        return {
            m: paired_bootstrap(per_user[a][m], per_user[b][m]).as_dict() for m in PAIRED_METRICS
        }

    paired = {
        "ranker - ease": delta("ranker", "ease"),
        "ranker - dense": delta("ranker", "dense"),
        "ranker-no-ease - ease": delta("ranker-no-ease", "ease"),
        "ranker-no-ease - dense": delta("ranker-no-ease", "dense"),
        "ranker - ranker-no-ease": delta("ranker", "ranker-no-ease"),
        "dense - ease": delta("dense", "ease"),
    }
    lat_rows = np.sort(
        np.random.default_rng(1).choice(
            fold_in.shape[0], min(latency_users, fold_in.shape[0]), replace=False
        )
    )
    latency = _latency(ctx, boosters["ranker"], fold_in, lat_rows, depth)

    table: dict[str, Any] = {"user_idx": data.test_users}
    for arm, metrics in per_user.items():
        table.update({f"{arm}/{m}": v for m, v in metrics.items()})
    pl.DataFrame(table).write_parquet(out / "per_user.parquet")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sizes = np.diff(test.offsets)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "Liang-SG-strict",
        "data": {
            "split_sha256": data.meta["split_sha256"],
            "n_items": data.n_items,
            "n_vb_users": int(vb_rows.size),
            "n_test_users": int(data.test_users.size),
        },
        "settings": {
            "dense_run": dense_run,
            "depth": list(depth),
            "ease_lambda": ease_lambda,
            "neighbours": neighbours,
            "features": list(FEATURES),
            "params": lgb_params,
            "max_rounds": max_rounds,
            "early_stopping": early_stopping,
            "seeds": list(seeds),
        },
        "train": {
            **train_info,
            "fit_users": int(fit_users.size),
            "stop_users": int(stop_users.size),
            "fit_rows": int(fit_stage.items.size),
            "fit_positive_rate": float(fit_labels.mean()),
            "vb_target_coverage": float(vb_labels.sum() / data.val_targets[vb_rows].nnz),
        },
        "test_candidates": {
            "per_user_mean": float(sizes.mean()),
            "per_user_min": int(sizes.min()),
            "target_coverage": float(labels_for(test, targets).sum() / targets.nnz),
        },
        "arms": arms,
        "paired": paired,
        "importance": importance,
        "latency": latency,
        "peak_rss_mb": round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1),
        "wall_s": round(time.perf_counter() - started, 1),
        "versions": {
            "shopsage_rec": __version__,
            "lightgbm": lgb.__version__,
            "python": platform.python_version(),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RankResult(out, summary)


def format_rank_summary(summary: dict[str, Any]) -> str:
    arms = summary["arms"]
    lines = []
    for arm in ("ease", "dense"):
        r, n = arms[arm]["recall@50"], arms[arm]["ndcg@10"]
        lines.append(f"{arm:15s} recall@50 {r['mean']:.4f}  ndcg@10 {n['mean']:.4f}")
    for arm in ARMS:
        r, n = arms[arm]["recall@50"], arms[arm]["ndcg@10"]
        lines.append(
            f"{arm:15s} recall@50 {r['mean']:.4f} ± {r['sd'] or 0:.4f}  "
            f"ndcg@10 {n['mean']:.4f} ± {n['sd'] or 0:.4f}"
        )
    for name, block in summary["paired"].items():
        r, n = block["recall@50"], block["ndcg@10"]
        lines.append(
            f"{name:24s} Δrecall@50 {r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
            f"Δndcg@10 {n['mean']:+.4f} [{n['ci_low']:+.4f}, {n['ci_high']:+.4f}]"
        )
    lat = summary["latency"]
    lines.append(f"latency p50/p95 (ms): {lat}")
    lines.append(f"importance: {summary['importance']['ranker']}")
    return "\n".join(lines)
