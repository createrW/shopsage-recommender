"""Reciprocal rank fusion for offline evaluation, the twin of Milvus's ``RRFRanker`` (D-010).

An item's fused score is ``sum over lists of 1 / (k + rank)``, ranks counted from 1, and 0 for a
list that does not contain it. Ranked lists are int arrays, one row per user; ``-1`` pads a short
row (a sparse path that matched fewer items than its depth). Ties go to the lower item index, as
everywhere offline.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shopsage_rec.evaluation.metrics import topk_masked

RRF_K = 60


def positive_only(top: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """``top`` with entries whose score is not positive replaced by -1 (Milvus's sparse search
    returns matching items only)."""
    picked = np.take_along_axis(np.asarray(scores), np.asarray(top), axis=1)
    return np.where(picked > 0, top, -1)


def rrf_scores(lists: Sequence[np.ndarray], n_items: int, *, k: int = RRF_K) -> np.ndarray:
    """Dense fused scores, users x items (float64)."""
    n_users = _rows(lists)
    out = np.zeros((n_users, n_items), dtype=np.float64)
    for ranked in lists:
        ranked = np.asarray(ranked)
        valid = ranked >= 0
        rows = np.broadcast_to(np.arange(n_users)[:, None], ranked.shape)
        ranks = np.broadcast_to(np.arange(1, ranked.shape[1] + 1), ranked.shape)
        # an item appears at most once per row of a list, so plain fancy-index += is exact
        out[rows[valid], ranked[valid]] += 1.0 / (k + ranks[valid])
    return out


def rrf(
    lists: Sequence[np.ndarray],
    n_items: int,
    *,
    k: int = RRF_K,
    top: int = 100,
    chunk_size: int = 4096,
) -> np.ndarray:
    """The ``top`` items of each user by fused score, best first."""
    n_users = _rows(lists)
    out = np.empty((n_users, top), dtype=np.int64)
    for start in range(0, n_users, chunk_size):
        rows = slice(start, min(start + chunk_size, n_users))
        out[rows] = topk_masked(rrf_scores([lst[rows] for lst in lists], n_items, k=k), None, top)
    return out


def _rows(lists: Sequence[np.ndarray]) -> int:
    if not lists:
        raise ValueError("need at least one ranked list")
    sizes = {np.asarray(lst).shape[0] for lst in lists}
    if len(sizes) != 1:
        raise ValueError(f"ranked lists have different numbers of rows: {sorted(sizes)}")
    return sizes.pop()
