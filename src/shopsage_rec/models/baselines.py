"""MostPop and EASE: the calibration baselines of every results table (D-009).

Both are fit on the training users only (``Liang-SG-strict``) and score a held-out user from
their fold-in items. Their published Recall@50 on ML-20M under that protocol (popularity 0.235,
EASE 0.521; Steck 2019, Table 1) checks the split and the metric code end to end (E004).

EASE (Steck 2019, "Embarrassingly Shallow Autoencoders for Sparse Data") has a closed form:
``P = (X^T X + lambda I)^-1``, ``B = P / -diag(P)`` column-wise with ``diag(B) = 0``, and a
user's scores are ``x_u B``. At 20,108 items the Gram matrix and ``B`` are 1.6 GB each in
float32, so they are built and inverted in float32 (D-009).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp


class MostPop:
    """Ranks items by how many training users have them as positives; ignores the user."""

    def __init__(self) -> None:
        self.popularity_: np.ndarray = np.empty(0, dtype=np.float32)

    def fit(self, train: sp.csr_matrix) -> MostPop:
        self.popularity_ = np.asarray(train.sum(axis=0), dtype=np.float32).ravel()
        return self

    def score(self, fold_in: sp.csr_matrix) -> np.ndarray:
        """The same popularity row for every user (a read-only broadcast view)."""
        return np.broadcast_to(self.popularity_, (fold_in.shape[0], self.popularity_.size))


def gram_matrix(train: sp.csr_matrix) -> np.ndarray:
    """``X^T X``: item co-occurrence counts over the training users, dense float32."""
    return np.asarray((train.T @ train).toarray(), dtype=np.float32)


class EASE:
    """The EASE item-item model, ``scores = x_u B``."""

    def __init__(self, weights: np.ndarray, lam: float) -> None:
        self.weights = weights
        self.lam = lam

    @classmethod
    def from_gram(cls, gram: np.ndarray, lam: float) -> EASE:
        """Solve EASE for one ``lam`` from a precomputed Gram matrix, which is left unchanged."""
        p = np.array(gram, dtype=np.float32, copy=True)
        p[np.diag_indices_from(p)] += np.float32(lam)
        p = scipy.linalg.inv(p, overwrite_a=True, check_finite=False)
        p /= -np.diag(p).copy()  # column j divided by -P_jj, in place
        np.fill_diagonal(p, 0.0)
        return cls(p, lam)

    @classmethod
    def fit(cls, train: sp.csr_matrix, lam: float = 500.0) -> EASE:
        return cls.from_gram(gram_matrix(train), lam)

    def score(self, fold_in: sp.csr_matrix) -> np.ndarray:
        return np.asarray(fold_in @ self.weights)
