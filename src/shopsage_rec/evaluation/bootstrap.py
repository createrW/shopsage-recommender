"""Percentile bootstrap over users, paired between arms (api_spec.md §8.4, D-007).

Per-user metric vectors of two arms over the same users are resampled with the same indices:
``numpy.random.default_rng(seed)`` draws ``n`` indices with replacement per replicate, in
chunks of 100 replicates to bound memory, and the 95% CI is the 2.5th-97.5th percentile of the
replicate differences. With the same seed and users, every arm sees identical resamples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 2000
SEED = 42
CHUNK = 100


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate on the full sample and its percentile interval."""

    estimate: float
    low: float
    high: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mean": self.estimate, "ci_low": self.low, "ci_high": self.high, "n": self.n}


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    chunk: int = CHUNK,
    level: float = 0.95,
) -> BootstrapCI:
    """CI of ``mean(a) - mean(b)`` for per-user vectors over the same users, in the same order."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"paired arms must be 1-D over the same users: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    diff = a - b
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - start)
        deltas[start : start + m] = diff[rng.integers(0, diff.size, size=(m, diff.size))].mean(
            axis=1
        )
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(deltas, [tail, 100 - tail])
    return BootstrapCI(estimate=float(diff.mean()), low=float(low), high=float(high), n=diff.size)


def bootstrap_ci(
    values: np.ndarray, *, n_resamples: int = N_RESAMPLES, seed: int = SEED, chunk: int = CHUNK
) -> BootstrapCI:
    """CI of one arm's mean, drawn with the same resample indices as ``paired_bootstrap``."""
    values = np.asarray(values, dtype=np.float64)
    return paired_bootstrap(
        values, np.zeros_like(values), n_resamples=n_resamples, seed=seed, chunk=chunk
    )
