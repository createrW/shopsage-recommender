"""The ranker's candidate set (D-006): the dense top-``depth[0]`` united with the positive-score
BM25 top-``depth[1]``, the user's fold-in items excluded. Missed positives are never injected."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from shopsage_rec.evaluation.metrics import topk_masked
from shopsage_rec.retrieval.fusion import positive_only


def union_candidates(
    dense: np.ndarray, bm25: np.ndarray, exclude: sp.csr_matrix, *, depth: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, items, dense_rank, bm25_rank)`` for one block of users, sorted by row then item.

    Ranks count from 1 within each path's list; an item outside a list gets that list's depth + 1.
    """
    if min(depth) < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    n_users, n_items = dense.shape
    d_top = topk_masked(dense, exclude, depth[0])
    b_top = positive_only(topk_masked(bm25, exclude, depth[1]), bm25)
    d_rank = np.full((n_users, n_items), depth[0] + 1, dtype=np.int32)
    b_rank = np.full((n_users, n_items), depth[1] + 1, dtype=np.int32)
    rows = np.arange(n_users)[:, None]
    d_rank[rows, d_top] = np.arange(1, depth[0] + 1)
    valid = b_top >= 0
    b_rows = np.broadcast_to(rows, b_top.shape)[valid]
    b_rank[b_rows, b_top[valid]] = np.broadcast_to(np.arange(1, depth[1] + 1), b_top.shape)[valid]
    member = (d_rank <= depth[0]) | (b_rank <= depth[1])
    r, i = np.nonzero(member)  # row-major: sorted by row, then item
    return r.astype(np.int64), i.astype(np.int64), d_rank[r, i], b_rank[r, i]
