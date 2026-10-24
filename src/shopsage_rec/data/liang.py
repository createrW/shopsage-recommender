"""Strong-generalisation split of Liang et al. (2018), replicated from the authors' released code.

The VAE-CF preprocessing (Liang, Krishnan, Hoffman and Jebara, "Variational Autoencoders for
Collaborative Filtering", WWW 2018) is the protocol behind the ML-20M Recall@50 anchors in
docs/plan.md (Popularity 0.235, EASE 0.521). It keeps the ratings above 3.5 (>= 4.0 on half
stars) of users with >= 5 of them and applies no item filter; ``filter_interactions`` does both.
This module does the rest, call for call:

1. users, in ascending ``userId`` order, are permuted with numpy's legacy ``RandomState(seed)``;
   the last ``2 * n_heldout_users`` become validation and test users, the others training users;
2. the items in the training users' rows define the catalogue, and the held-out users' rows are
   restricted to it;
3. for validation and for test separately, a fresh ``RandomState(seed)`` walks the held-out users
   in ascending ``userId`` order and, for each with >= 5 items, draws ``int(test_prop * n)`` of
   them (``choice(n, size, replace=False)``) as targets; the rest are the user's fold-in items.
   Users with fewer than 5 items keep them all as fold-in and get no targets.

Within a user, items are visited in ascending ``movieId`` order, the order of ML-20M's
ratings.csv and therefore of the notebook's pandas groupby over it, so the same seed reproduces
the published split; a real-data test checks its counts (136,677 users, 20,108 training items).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

LIANG_SEED = 98765  # np.random.seed(98765) in the authors' notebook
MIN_ITEMS_FOR_TARGETS = 5  # the notebook's `if n_items_u >= 5`
_SPLIT_NAMES = ("train", "val_tr", "val_te", "test_tr", "test_te")


@dataclass(frozen=True)
class LiangSplit:
    """Training users' rows, and fold-in (``*_tr``) / target (``*_te``) rows of held-out users.

    Frames keep the input columns and raw ids, sorted by ``userId`` then ``movieId``. ``users``
    gives every user's role (``train``, ``val`` or ``test``), including held-out users left with
    no rows once their items are restricted to the training catalogue.
    """

    train: pl.DataFrame
    val_tr: pl.DataFrame
    val_te: pl.DataFrame
    test_tr: pl.DataFrame
    test_te: pl.DataFrame
    users: pl.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)

    def frames(self) -> dict[str, pl.DataFrame]:
        return dict(
            zip(
                _SPLIT_NAMES,
                (self.train, self.val_tr, self.val_te, self.test_tr, self.test_te),
                strict=True,
            )
        )


def _validate(positives: pl.DataFrame, seed: int, n_heldout_users: int, test_prop: float) -> None:
    missing = [c for c in ("userId", "movieId") if c not in positives.columns]
    if missing:
        raise ValueError(f"positives are missing required columns: {missing}")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32) for numpy's legacy RandomState")
    if n_heldout_users < 1:
        raise ValueError("n_heldout_users must be >= 1")
    if not 0.0 < test_prop < 1.0:
        raise ValueError("test_prop must be strictly between 0 and 1")
    if positives["userId"].null_count() or positives["movieId"].null_count():
        raise ValueError("userId and movieId must not be null")
    if positives.select("userId", "movieId").is_duplicated().any():
        raise ValueError("positives contain duplicate (userId, movieId) pairs")


def _fold_in_and_targets(
    rows: pl.DataFrame, users: np.ndarray, items: pl.Series, seed: int, test_prop: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The notebook's ``split_train_test_proportion``: ``users``' rows within ``items``."""
    held = rows.filter(
        pl.col("userId").is_in(pl.Series(users).implode())
        & pl.col("movieId").is_in(items.implode())
    )
    uid = held["userId"].to_numpy()
    is_target = np.zeros(uid.size, dtype=bool)
    if uid.size:
        starts = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
        sizes = np.diff(np.r_[starts, uid.size])
        rng = np.random.RandomState(seed)
        for start, n_items_u in zip(starts.tolist(), sizes.tolist(), strict=True):
            if n_items_u >= MIN_ITEMS_FOR_TARGETS:
                chosen = rng.choice(n_items_u, size=int(test_prop * n_items_u), replace=False)
                is_target[start + chosen] = True
    mask = pl.Series(is_target)
    return held.filter(~mask), held.filter(mask)


def _digest(frames: dict[str, pl.DataFrame]) -> str:
    """sha256 over each frame's sorted (userId, movieId) pairs: the split's identity."""
    digest = hashlib.sha256()
    for name, frame in frames.items():
        pairs = (
            frame.select(pl.col("userId").cast(pl.Int64), pl.col("movieId").cast(pl.Int64))
            .sort("userId", "movieId")
            .to_numpy()
        )
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(pairs, dtype=np.int64).tobytes())
    return digest.hexdigest()


def liang_split(
    positives: pl.DataFrame,
    *,
    seed: int = LIANG_SEED,
    n_heldout_users: int = 10_000,
    test_prop: float = 0.2,
) -> LiangSplit:
    """Split filtered positives into training users and validation / test fold-in and targets."""
    _validate(positives, seed, n_heldout_users, test_prop)
    rows = positives.sort("userId", "movieId")
    users = rows["userId"].unique().sort().to_numpy()
    if users.size <= 2 * n_heldout_users:
        raise ValueError(
            f"need more than 2 * n_heldout_users = {2 * n_heldout_users} users, got {users.size}"
        )
    permuted = users[np.random.RandomState(seed).permutation(users.size)]
    n_train_users = users.size - 2 * n_heldout_users
    train_users = permuted[:n_train_users]
    val_users = permuted[n_train_users : n_train_users + n_heldout_users]
    test_users = permuted[n_train_users + n_heldout_users :]

    train = rows.filter(pl.col("userId").is_in(pl.Series(train_users).implode()))
    catalogue = train["movieId"].unique()
    val_tr, val_te = _fold_in_and_targets(rows, val_users, catalogue, seed, test_prop)
    test_tr, test_te = _fold_in_and_targets(rows, test_users, catalogue, seed, test_prop)
    roles = pl.DataFrame(
        {
            "userId": permuted,
            "role": ["train"] * n_train_users
            + ["val"] * n_heldout_users
            + ["test"] * n_heldout_users,
        },
        schema={"userId": rows.schema["userId"], "role": pl.String},
    ).sort("userId")
    split = LiangSplit(
        train=train, val_tr=val_tr, val_te=val_te, test_tr=test_tr, test_te=test_te, users=roles
    )

    val_with_targets = val_te["userId"].n_unique()
    test_with_targets = test_te["userId"].n_unique()
    stats: dict[str, Any] = {
        "strategy": "liang",
        "seed": int(seed),
        "n_heldout_users": int(n_heldout_users),
        "test_prop": float(test_prop),
        "n_users": int(users.size),
        "n_train_users": int(n_train_users),
        "n_val_users": int(n_heldout_users),
        "n_test_users": int(n_heldout_users),
        "n_interactions": int(rows.height),
        "n_items_input": int(rows["movieId"].n_unique()),
        "n_items": int(catalogue.len()),
        **{f"n_{name}": int(frame.height) for name, frame in split.frames().items()},
        "val_users_with_targets": int(val_with_targets),
        "test_users_with_targets": int(test_with_targets),
        "val_users_without_targets": int(n_heldout_users - val_with_targets),
        "test_users_without_targets": int(n_heldout_users - test_with_targets),
        "split_sha256": _digest(split.frames()),
    }
    return LiangSplit(**split.frames(), users=roles, stats=stats)


def partition_val_users(
    users: np.ndarray, *, seed: int = 42, frac_va: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint V_A (model selection) and V_B (ranker labels) halves of the validation users.

    The order of ``users`` does not matter; both halves come back sorted.
    """
    if not 0.0 < frac_va < 1.0:
        raise ValueError("frac_va must be strictly between 0 and 1")
    ordered = np.unique(np.asarray(users))
    perm = np.random.default_rng(seed).permutation(ordered.size)
    n_va = int(ordered.size * frac_va)
    return np.sort(ordered[perm[:n_va]]), np.sort(ordered[perm[n_va:]])


Stage = Literal["val", "test"]


@dataclass(frozen=True)
class LiangData:
    """A prepared Liang split as binary CSR matrices over the ``item_idx`` catalogue.

    ``train`` has one row per training user (ascending ``user_idx``). For each stage, rows are
    the held-out users with at least one target (``*_users``, ascending ``user_idx``), and
    ``*_fold_in`` / ``*_targets`` hold their fold-in and target items: the inputs a model scores
    from and the items it is evaluated on.
    """

    n_items: int
    train_users: np.ndarray
    train: sp.csr_matrix
    val_users: np.ndarray
    val_fold_in: sp.csr_matrix
    val_targets: sp.csr_matrix
    test_users: np.ndarray
    test_fold_in: sp.csr_matrix
    test_targets: sp.csr_matrix
    meta: dict[str, Any]

    def stage(self, name: Stage) -> tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]:
        """``(users, fold_in, targets)`` of the ``val`` or ``test`` stage."""
        if name == "val":
            return self.val_users, self.val_fold_in, self.val_targets
        if name == "test":
            return self.test_users, self.test_fold_in, self.test_targets
        raise ValueError(f"unknown stage {name!r}")


def _csr(frame: pl.DataFrame, users: np.ndarray, n_items: int) -> sp.csr_matrix:
    """Binary float32 CSR with one row per entry of ``users``; rows of other users are dropped."""
    user_idx = frame["user_idx"].to_numpy()
    keep = np.isin(user_idx, users)
    rows = np.searchsorted(users, user_idx[keep])
    cols = frame["item_idx"].to_numpy()[keep]
    data = np.ones(rows.size, dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(users.size, n_items), dtype=np.float32)
    matrix.sum_duplicates()
    return matrix


def load_liang(processed_dir: str | Path) -> LiangData:
    """Read ``shopsage prepare``'s Liang artifacts (``splits/`` and ``mappings/``) as matrices."""
    base = Path(processed_dir)
    splits = base / "splits"
    meta = json.loads((splits / "meta.json").read_text(encoding="utf-8"))
    if meta.get("strategy") != "liang":
        raise ValueError(f"{splits / 'meta.json'} is not a Liang split: {meta.get('strategy')!r}")
    n_items = pl.read_parquet(base / "mappings" / "item_map.parquet").height
    frames = {name: pl.read_parquet(splits / f"{name}.parquet") for name in _SPLIT_NAMES}
    train_users = np.sort(frames["train"]["user_idx"].unique().to_numpy())
    stages: dict[str, tuple[np.ndarray, sp.csr_matrix, sp.csr_matrix]] = {}
    for stage in ("val", "test"):
        users = np.sort(frames[f"{stage}_te"]["user_idx"].unique().to_numpy())
        stages[stage] = (
            users,
            _csr(frames[f"{stage}_tr"], users, n_items),
            _csr(frames[f"{stage}_te"], users, n_items),
        )
    return LiangData(
        n_items=n_items,
        train_users=train_users,
        train=_csr(frames["train"], train_users, n_items),
        val_users=stages["val"][0],
        val_fold_in=stages["val"][1],
        val_targets=stages["val"][2],
        test_users=stages["test"][0],
        test_fold_in=stages["test"][1],
        test_targets=stages["test"][2],
        meta=meta,
    )
