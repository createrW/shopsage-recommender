"""Full-corpus top-k ranking and the Liang-protocol metrics (api_spec.md §8).

Every catalogue item outside a user's masked set (their fold-in items) is scored; masked items
are set to -inf *before* the top-k cut, so a user with many fold-in items still gets k candidates
(D-008). Ties are broken by ascending item index. Metrics are per-user vectors over users with at
least one target:

- truncated Recall@K = hits@K / min(K, n_u) (Liang et al. 2018), which can reach 1 for n_u > K;
- NDCG@K with binary gains, the ideal DCG taken over min(K, n_u) items;
- HR@K = 1 if any target is in the top K.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.sparse as sp

DEFAULT_RECALL_KS = (20, 50)
DEFAULT_NDCG_KS = (10, 100)
DEFAULT_HR_KS = (50,)


def topk_masked(scores: np.ndarray, exclude: sp.csr_matrix | None, k: int) -> np.ndarray:
    """Indices of each row's ``k`` highest-scoring items, best first, never an excluded item.

    ``scores`` is a dense ``(n_users, n_items)`` block; the stored entries of ``exclude`` (same
    shape) are masked. Ties go to the lower item index, including at the k-th place.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D, got shape {scores.shape}")
    _, n_items = scores.shape
    if not 1 <= k <= n_items:
        raise ValueError(f"k must be in [1, {n_items}], got {k}")
    work = np.array(scores, dtype=np.result_type(scores.dtype, np.float32), copy=True)
    if np.isnan(work).any():
        raise ValueError("scores contain NaN")
    if exclude is not None:
        if exclude.shape != scores.shape:
            raise ValueError(f"exclude has shape {exclude.shape}, scores {scores.shape}")
        rows, cols = exclude.nonzero()
        work[rows, cols] = -np.inf
    if (np.isfinite(work).sum(axis=1) < k).any():
        raise ValueError("a row has fewer than k unmasked items")

    part = np.argpartition(-work, k - 1, axis=1)[:, :k]
    kth = np.take_along_axis(work, part, axis=1).min(axis=1)
    n_above = (work > kth[:, None]).sum(axis=1)
    n_tied = (work == kth[:, None]).sum(axis=1)
    # argpartition picks arbitrarily among scores tied at the k-th value: give those slots to the
    # lowest indices instead.
    for row in np.flatnonzero(n_above + n_tied > k):
        above = np.flatnonzero(work[row] > kth[row])
        tied = np.flatnonzero(work[row] == kth[row])[: k - above.size]
        part[row] = np.concatenate([above, tied])
    values = np.take_along_axis(work, part, axis=1)
    order = np.lexsort((part, -values), axis=1)  # by score descending, then item ascending
    return np.take_along_axis(part, order, axis=1)


def topk_for_users(
    score_fn: Callable[[slice], np.ndarray],
    exclude: sp.csr_matrix,
    k: int,
    *,
    chunk_size: int = 8192,
) -> np.ndarray:
    """``topk_masked`` over row chunks of ``exclude``; ``score_fn(rows)`` scores one chunk.

    Bounds memory to one ``chunk_size x n_items`` score block (8,192 x 20,108 float32 is 659 MB).
    """
    n_users = exclude.shape[0]
    out = np.empty((n_users, k), dtype=np.int64)
    for start in range(0, n_users, chunk_size):
        rows = slice(start, min(start + chunk_size, n_users))
        out[rows] = topk_masked(score_fn(rows), exclude[rows], k)
    return out


def relevance(topk: np.ndarray, targets: sp.csr_matrix) -> np.ndarray:
    """``rel[u, r]`` is True when user ``u``'s ``r``-th recommendation is one of their targets."""
    n_users, n_items = targets.shape
    if topk.shape[0] != n_users:
        raise ValueError(f"topk has {topk.shape[0]} rows, targets {n_users}")
    coo = sp.coo_matrix(targets)
    keys = np.unique(coo.row.astype(np.int64) * n_items + coo.col)
    ranked = np.arange(n_users, dtype=np.int64)[:, None] * n_items + topk
    return np.isin(ranked, keys)


def _check(rel: np.ndarray, k: int, n_relevant: np.ndarray | None = None) -> None:
    if not 1 <= k <= rel.shape[1]:
        raise ValueError(f"k must be in [1, {rel.shape[1]}] for this ranking, got {k}")
    if n_relevant is not None and (np.asarray(n_relevant) < 1).any():
        raise ValueError("every evaluated user needs at least one target")


def recall_at_k(rel: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Truncated recall: hits in the top ``k`` divided by ``min(k, n_relevant)``."""
    _check(rel, k, n_relevant)
    return rel[:, :k].sum(axis=1) / np.minimum(k, n_relevant)


def ndcg_at_k(rel: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """NDCG@k with binary gains; the ideal ranking places ``min(k, n_relevant)`` hits first."""
    _check(rel, k, n_relevant)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel[:, :k] * discounts).sum(axis=1)
    idcg = np.cumsum(discounts)[np.minimum(k, n_relevant) - 1]
    return dcg / idcg


def hit_rate_at_k(rel: np.ndarray, k: int) -> np.ndarray:
    """1.0 when any target is in the top ``k``, else 0.0."""
    _check(rel, k)
    return rel[:, :k].any(axis=1).astype(np.float64)


def evaluate_topk(
    topk: np.ndarray,
    targets: sp.csr_matrix,
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
    hr_ks: Sequence[int] = DEFAULT_HR_KS,
) -> dict[str, np.ndarray]:
    """Per-user ``recall@K``, ``ndcg@K`` and ``hr@K`` vectors for a ranking and its targets."""
    targets = sp.csr_matrix(targets, copy=True)
    targets.eliminate_zeros()
    targets.sum_duplicates()
    n_relevant = np.diff(targets.indptr)
    if (n_relevant < 1).any():
        raise ValueError("every evaluated user needs at least one target")
    rel = relevance(topk, targets)
    out: dict[str, np.ndarray] = {}
    for k in recall_ks:
        out[f"recall@{k}"] = recall_at_k(rel, n_relevant, k)
    for k in ndcg_ks:
        out[f"ndcg@{k}"] = ndcg_at_k(rel, n_relevant, k)
    for k in hr_ks:
        out[f"hr@{k}"] = hit_rate_at_k(rel, k)
    return out
