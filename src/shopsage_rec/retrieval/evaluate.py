"""``shopsage recall``: ablation (b), the recall paths, on a prepared Liang split (D-010).

- **B0, BM25:** k1, b and the query weighting are chosen on V_A by Recall@50 from a small grid.
- **B1, dense:** each given Two-Tower run (``item_embeddings.npy``, ``user_tower.npz``), scored
  through the numpy user tower.
- **B2, hybrid:** RRF (k = 60) of the dense top-``depth`` and the positive-score BM25
  top-``depth``, one per dense run.

Test metrics over all test users with percentile-bootstrap CIs; paired deltas (BM25 - dense,
hybrid - dense) per dense run and on the per-user mean over runs. Two diagnostics:

- ``candidates``: the share of each test user's targets inside the dense top-``candidates[0]``
  and inside its union with the BM25 top-``candidates[1]``, the ranker's candidate set (S4);
- ``milvus``: the serving index built from the first dense run and the BM25 weights, queried for
  ``parity_users`` test users. Each path's results are compared with the offline lists
  (``identical`` = same scores rank by rank, which tolerates the order of exact ties;
  ``overlap`` = shared items) and timed on this machine, next to the same answers computed
  exactly in numpy / scipy for one user at a time (``exact_ms``: dense and BM25 at ``depth``,
  hybrid = both plus RRF), the serving fallback of plan.md §8.

Outputs go to ``<out_root>/<run_id>/``: ``per_user.parquet`` and ``summary.json``.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from shopsage_rec import __version__
from shopsage_rec.data.liang import load_liang, partition_val_users
from shopsage_rec.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from shopsage_rec.evaluation.metrics import evaluate_topk, topk_for_users, topk_masked
from shopsage_rec.models.numpy_tower import user_vectors
from shopsage_rec.retrieval.bm25 import BM25, build_item_documents, fit_bm25
from shopsage_rec.retrieval.fusion import RRF_K, positive_only, rrf, rrf_scores
from shopsage_rec.retrieval.milvus_store import MilvusStore, sparse_row

DEFAULT_GRID: tuple[tuple[float, float, str], ...] = tuple(
    (k1, b, query) for k1 in (0.9, 1.5) for b in (0.4, 0.75) for query in ("binary", "log")
)
DENSE_RUNS = ("A1-s42", "A1-s43", "A1-s44")
K_EVAL = 100
K_SERVE = 50
PAIRED_METRICS = ("recall@50", "ndcg@10")


@dataclass(frozen=True)
class RecallPathsResult:
    out_dir: Path
    summary: dict[str, Any]


def _bm25_top(model: BM25, fold_in: sp.csr_matrix, k: int, chunk: int = 2048) -> np.ndarray:
    """Top-``k`` by BM25 with fold-in items masked; ``-1`` where the score is not positive."""
    out = np.empty((fold_in.shape[0], k), dtype=np.int64)
    for start in range(0, fold_in.shape[0], chunk):
        rows = slice(start, min(start + chunk, fold_in.shape[0]))
        scores = model.scores(fold_in[rows])
        out[rows] = positive_only(topk_masked(scores, fold_in[rows], k), scores)
    return out


def _filled(top: np.ndarray, fold_in: sp.csr_matrix, k: int) -> np.ndarray:
    """A ranking padded past its last match with the lowest-index unseen items, the offline tie
    rule for items that all score 0."""
    out = top[:, :k].copy()
    for row in np.flatnonzero((out < 0).any(axis=1)):
        have = out[row][out[row] >= 0]
        blocked = set(have.tolist()) | set(fold_in[row].indices.tolist())
        fill = [i for i in range(fold_in.shape[1]) if i not in blocked][: k - have.size]
        out[row] = np.concatenate([have, np.asarray(fill, dtype=np.int64)])
    return out


def _coverage(lists: Sequence[np.ndarray], targets: sp.csr_matrix) -> float:
    """Mean share of each user's targets found in the union of ``lists`` (``-1`` ignored)."""
    rows = np.concatenate([np.repeat(np.arange(lst.shape[0]), lst.shape[1]) for lst in lists])
    cols = np.concatenate([lst.ravel() for lst in lists])
    keep = cols >= 0
    union = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], cols[keep])), shape=targets.shape
    )
    union.data[:] = 1.0  # duplicates were summed
    hits = np.asarray(targets.multiply(union).sum(axis=1)).ravel()
    return float(np.mean(hits / np.diff(targets.indptr)))


def _dense_scores(users: np.ndarray, items: np.ndarray, rows: slice) -> np.ndarray:
    return users[rows] @ items.T


def _ms(samples: list[float]) -> list[float]:
    p50, p95 = np.percentile(np.asarray(samples) * 1e3, [50, 95])
    return [round(float(p50), 2), round(float(p95), 2)]


def _same_scores(got: list[tuple[int, float]], expected: np.ndarray) -> bool:
    return len(got) == expected.size and bool(
        np.allclose([s for _, s in got], expected, rtol=1e-4, atol=1e-6)
    )


def _milvus_parity(
    *,
    items: np.ndarray,
    bm25: BM25,
    users: np.ndarray,
    weights: dict[str, np.ndarray],
    fold_in: sp.csr_matrix,
    dense_top: np.ndarray,
    bm25_top: np.ndarray,
    rows: np.ndarray,
    depth: int,
) -> dict[str, Any]:
    queries = bm25.queries(fold_in[rows])
    report: dict[str, Any] = {"users": int(rows.size), "k": K_SERVE, "depth": depth}
    timings: dict[str, list[float]] = {p: [] for p in ("tower", "dense", "sparse", "hybrid")}
    exact: dict[str, list[float]] = {p: [] for p in ("dense", "sparse", "hybrid")}
    checks: dict[str, list[tuple[bool, float]]] = {p: [] for p in ("dense", "sparse", "hybrid")}
    with tempfile.TemporaryDirectory() as tmp:
        t = time.perf_counter()
        store = MilvusStore.build(Path(tmp) / "items.db", items, bm25.weights)
        report["build_s"] = round(time.perf_counter() - t, 2)
        for j, row in enumerate(rows.tolist()):
            seen = fold_in[row].indices
            t = time.perf_counter()
            vec = user_vectors(fold_in[row], weights)[0]
            query = sparse_row(queries, j)
            timings["tower"].append(time.perf_counter() - t)

            expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            dense_ids = dense_top[row, :K_SERVE]
            expected["dense"] = (dense_ids, users[row] @ items[dense_ids].T)
            sparse_ids = bm25_top[row, :K_SERVE]
            sparse_ids = sparse_ids[sparse_ids >= 0]
            expected["sparse"] = (sparse_ids, bm25.scores(fold_in[row])[0, sparse_ids])
            lists = [dense_top[row : row + 1, :depth], bm25_top[row : row + 1, :depth]]
            fused = rrf_scores(lists, items.shape[0])[0]
            hybrid_ids = rrf(lists, items.shape[0], top=K_SERVE)[0]
            expected["hybrid"] = (hybrid_ids, fused[hybrid_ids])

            calls = {
                "dense": partial(store.search_dense, vec, seen, K_SERVE),
                "sparse": partial(store.search_sparse, query, seen, K_SERVE),
                "hybrid": partial(
                    store.search_hybrid, vec, query, seen, K_SERVE, depth=(depth, depth)
                ),
            }
            # the same three answers computed exactly in numpy / scipy, for one user (S6 fallback)
            t = time.perf_counter()
            d = topk_masked((items @ vec)[None], fold_in[row], depth)
            exact["dense"].append(time.perf_counter() - t)
            t = time.perf_counter()
            sq = _bm25_top(bm25, fold_in[row : row + 1], depth)
            exact["sparse"].append(time.perf_counter() - t)
            t = time.perf_counter()
            rrf([d, sq], items.shape[0], top=K_SERVE)
            exact["hybrid"].append(
                time.perf_counter() - t + exact["dense"][-1] + exact["sparse"][-1]
            )
            for path, call in calls.items():
                t = time.perf_counter()
                got = call()
                timings[path].append(time.perf_counter() - t)
                ids, scores = expected[path]
                shared = len({i for i, _ in got} & set(ids.tolist())) / max(ids.size, 1)
                checks[path].append((_same_scores(got, scores), shared))
        store.close()
    report["tower_ms"] = _ms(timings["tower"])
    for path, results in checks.items():
        report[path] = {
            "identical": float(np.mean([ok for ok, _ in results])),
            "overlap": float(np.mean([shared for _, shared in results])),
            "ms": _ms(timings[path]),
            "exact_ms": _ms(exact[path]),
        }
    return report


def _paired(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> dict[str, dict[str, float | int]]:
    return {m: paired_bootstrap(a[m], b[m]).as_dict() for m in PAIRED_METRICS}


def evaluate_recall_paths(
    processed_dir: str | Path,
    runs_dir: str | Path,
    out_root: str | Path,
    *,
    dense_runs: Sequence[str] = DENSE_RUNS,
    grid: Sequence[tuple[float, float, str]] = DEFAULT_GRID,
    depth: int = 200,
    candidates: tuple[int, int] = (500, 100),
    parity_users: int = 1_000,
    run_id: str | None = None,
    seed: int = 0,
) -> RecallPathsResult:
    started = time.perf_counter()
    if not dense_runs:
        raise ValueError("need at least one dense run")
    run_id = run_id or datetime.now().strftime("recall-%Y%m%d-%H%M")
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    data = load_liang(processed_dir)
    docs = build_item_documents(processed_dir, data.meta["config"]["raw_dir"])

    va_users, _ = partition_val_users(data.val_users)
    va_rows = np.searchsorted(data.val_users, va_users)
    va_fold_in, va_targets = data.val_fold_in[va_rows], data.val_targets[va_rows]
    va_scores = []
    for k1, b, query in grid:
        model = fit_bm25(docs.tf, docs.vocab, k1=k1, b=b, query=query)
        top = _filled(_bm25_top(model, va_fold_in, K_SERVE), va_fold_in, K_SERVE)
        score = evaluate_topk(top, va_targets, recall_ks=(50,), ndcg_ks=(), hr_ks=())["recall@50"]
        va_scores.append(float(score.mean()))
    k1, b, query = grid[int(np.argmax(va_scores))]  # the first of equals wins
    tried = [
        {"k1": g[0], "b": g[1], "query": g[2], "va_recall@50": v}
        for g, v in zip(grid, va_scores, strict=True)
    ]
    bm25 = fit_bm25(docs.tf, docs.vocab, k1=k1, b=b, query=query)

    fold_in, targets = data.test_fold_in, data.test_targets
    n_items = data.n_items
    bm25_depth = max(depth, K_EVAL, candidates[1])
    bm25_top = _bm25_top(bm25, fold_in, bm25_depth)
    per_user: dict[str, dict[str, np.ndarray]] = {
        "bm25": evaluate_topk(_filled(bm25_top, fold_in, K_EVAL), targets)
    }
    dense_depth = max(depth, K_EVAL, candidates[0])
    first: dict[str, Any] = {}
    for name in dense_runs:
        run = Path(runs_dir) / name
        items = np.load(run / "item_embeddings.npy")
        with np.load(run / "user_tower.npz") as z:
            weights = {key: z[key] for key in z.files}
        users = user_vectors(fold_in, weights)
        dense_top = topk_for_users(
            partial(_dense_scores, users, items), fold_in, dense_depth, chunk_size=2048
        )
        per_user[f"dense:{name}"] = evaluate_topk(dense_top[:, :K_EVAL], targets)
        hybrid = rrf([dense_top[:, :depth], bm25_top[:, :depth]], n_items, top=K_EVAL)
        per_user[f"hybrid:{name}"] = evaluate_topk(hybrid, targets)
        if not first:
            first = {"items": items, "weights": weights, "users": users, "dense_top": dense_top}

    seed_mean = {
        path: {
            m: np.mean([per_user[f"{path}:{n}"][m] for n in dense_runs], axis=0)
            for m in PAIRED_METRICS
        }
        for path in ("dense", "hybrid")
    }
    paired = {
        "hybrid - dense": {
            "per_seed": {
                n: _paired(per_user[f"hybrid:{n}"], per_user[f"dense:{n}"]) for n in dense_runs
            },
            "seed_mean": _paired(seed_mean["hybrid"], seed_mean["dense"]),
        },
        "bm25 - dense": {
            "per_seed": {n: _paired(per_user["bm25"], per_user[f"dense:{n}"]) for n in dense_runs},
            "seed_mean": _paired(per_user["bm25"], seed_mean["dense"]),
        },
    }
    dense_cut = first["dense_top"][:, : candidates[0]]
    coverage = {
        f"dense@{candidates[0]}": _coverage([dense_cut], targets),
        f"bm25@{candidates[1]}": _coverage([bm25_top[:, : candidates[1]]], targets),
        f"dense@{candidates[0]} + bm25@{candidates[1]}": _coverage(
            [dense_cut, bm25_top[:, : candidates[1]]], targets
        ),
    }
    rows = np.sort(
        np.random.default_rng(seed).choice(
            fold_in.shape[0], min(parity_users, fold_in.shape[0]), replace=False
        )
    )
    milvus = _milvus_parity(
        items=first["items"],
        bm25=bm25,
        users=first["users"],
        weights=first["weights"],
        fold_in=fold_in,
        dense_top=first["dense_top"],
        bm25_top=bm25_top,
        rows=rows,
        depth=depth,
    )
    milvus["dense_run"] = dense_runs[0]

    columns: dict[str, Any] = {"user_idx": data.test_users}
    for arm, metrics in per_user.items():
        columns.update({f"{arm}/{m}": v for m, v in metrics.items()})
    pl.DataFrame(columns).write_parquet(out / "per_user.parquet")
    terms = np.array([t[:2] for t in docs.vocab])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "Liang-SG-strict",
        "data": {
            "split_sha256": data.meta["split_sha256"],
            "n_items": n_items,
            "n_va_users": int(va_rows.size),
            "n_test_users": int(data.test_users.size),
        },
        "settings": {
            "depth": depth,
            "rrf_k": RRF_K,
            "candidates": list(candidates),
            "dense_runs": list(dense_runs),
        },
        "bm25": {
            "grid": tried,
            "chosen": [k1, b, query],
            "n_terms": len(docs.vocab),
            "terms_by_kind": {kind: int((terms == kind).sum()) for kind in ("w:", "g:", "t:")},
            "nnz": int(docs.tf.nnz),
            "empty_documents": int((np.diff(docs.tf.indptr) == 0).sum()),
        },
        "arms": {
            arm: {m: bootstrap_ci(v).as_dict() for m, v in metrics.items()}
            for arm, metrics in per_user.items()
        },
        "paired": paired,
        "candidates": coverage,
        "milvus": milvus,
        "peak_rss_mb": round(peak / 2**20 if sys.platform == "darwin" else peak / 2**10, 1),
        "wall_s": round(time.perf_counter() - started, 1),
        "versions": {"shopsage_rec": __version__, "python": platform.python_version()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RecallPathsResult(out, summary)


def format_recall_summary(summary: dict[str, Any]) -> str:
    lines = [f"BM25 chosen on V_A (k1, b, query): {summary['bm25']['chosen']}"]
    for arm, metrics in summary["arms"].items():
        r, n = metrics["recall@50"], metrics["ndcg@10"]
        lines.append(
            f"{arm:22s} recall@50 {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
            f"ndcg@10 {n['mean']:.4f}"
        )
    for name, block in summary["paired"].items():
        d = block["seed_mean"]["recall@50"]
        lines.append(
            f"{name} (seed mean) recall@50 {d['mean']:+.4f} "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
        )
    lines.append(f"candidates: {summary['candidates']}")
    m = summary["milvus"]
    for path in ("dense", "sparse", "hybrid"):
        lines.append(
            f"milvus {path}: identical {m[path]['identical']:.3f}, "
            f"overlap {m[path]['overlap']:.3f}, p50/p95 {m[path]['ms'][0]} / {m[path]['ms'][1]} ms "
            f"(exact numpy/scipy {m[path]['exact_ms'][0]} / {m[path]['exact_ms'][1]} ms)"
        )
    return "\n".join(lines)
