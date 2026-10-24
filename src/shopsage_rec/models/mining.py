"""Hard negatives for arm A3: each training user's ranks 101-500 under a reference run.

The reference (A1, seed 42) scores every item for every training user from the user's full
training history; training positives are excluded before ranking, and the top 100 are skipped
because they hold most of the unlabelled positives (false negatives; Huang et al. 2020). The
pool is written next to the run as ``mined_101_500.npy`` (int32, one row per training user in
ascending ``user_idx``). numpy only, so it runs on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shopsage_rec.data.liang import load_liang
from shopsage_rec.evaluation.metrics import topk_for_users
from shopsage_rec.models.numpy_tower import user_vectors

MINED_FILE = "mined_101_500.npy"


def mine_hard_negatives(
    processed_dir: str | Path, run_dir: str | Path, *, skip: int = 100, depth: int = 500
) -> Path:
    """Write the ranks ``skip+1 .. depth`` of every training user under ``run_dir``'s model."""
    if not 0 <= skip < depth:
        raise ValueError("need 0 <= skip < depth")
    run = Path(run_dir)
    data = load_liang(processed_dir)
    items = np.load(run / "item_embeddings.npy")
    with np.load(run / "user_tower.npz") as z:
        weights = {k: z[k] for k in z.files}
    users = user_vectors(data.train, weights)
    top = topk_for_users(lambda rows: users[rows] @ items.T, data.train, depth, chunk_size=4096)
    path = run / MINED_FILE
    np.save(path, top[:, skip:depth].astype(np.int32))
    return path
