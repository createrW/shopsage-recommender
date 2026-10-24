"""The user tower in numpy, for the demo and for CPU evaluation (no torch import).

Mirrors ``TwoTower.user_vectors_csr``: the mean of history-item embeddings over a user's items,
concatenated with ``log1p(number of items)``, through a two-layer ReLU MLP, L2-normalised. The
weights come from ``TwoTower.export_user_tower()`` (``user_tower.npz``).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

USER_TOWER_KEYS = ("hist_emb", "w1", "b1", "w2", "b2")


def user_vectors(fold_in: sp.csr_matrix, weights: Mapping[str, np.ndarray]) -> np.ndarray:
    """Unit-norm user vectors, one row per row of ``fold_in`` (a binary user-item matrix)."""
    fold_in = sp.csr_matrix(fold_in, dtype=np.float32)
    counts = np.diff(fold_in.indptr).astype(np.float32)
    scale = sp.diags(1.0 / np.maximum(counts, 1.0))
    mean = np.asarray(scale @ fold_in @ weights["hist_emb"], dtype=np.float32)
    x = np.hstack([mean, np.log1p(counts)[:, None]])
    h = np.maximum(x @ weights["w1"].T + weights["b1"], 0.0)
    out = h @ weights["w2"].T + weights["b2"]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norm, 1e-12)).astype(np.float32)
