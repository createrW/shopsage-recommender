"""Leave-one-out split per user, ordered by timestamp, with a reproducible tie-break.

For every user the most recent positive becomes the *test* item, the second most recent the
*validation* item and everything else is *train*. MovieLens users frequently rate many movies
within the same second (bulk onboarding), so ties at the boundary are common (15% of users at
the test/val boundary). Ties are broken by a seeded SplitMix64 hash of ``(user, item)``:

* it is computed in numpy with explicit 64-bit wrapping arithmetic, so it does not depend on the
  Polars version (``Expr.hash`` is documented as unstable across releases);
* it is non-linear, so neither the higher nor the lower item id is favoured (an affine hash
  modulo a prime, an earlier version of this module, favoured the higher id ~81% of the time);
* the pipeline computes it over the *raw* ``(userId, movieId)`` before id remapping, so how
  two surviving tied items are ordered does not depend on the filter thresholds or on the
  contiguous id maps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

DEFAULT_TIE_KEY_COL = "tie_key"

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_SHIFT30, _SHIFT27, _SHIFT31, _SHIFT32 = (np.uint64(s) for s in (30, 27, 31, 32))
_ID_LIMIT = 2**32


@dataclass
class SplitResult:
    """The three partitions plus the summary statistics recorded in ``meta.json``."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 output function (Steele, Lea & Flood 2014) over uint64 inputs."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLDEN
        z = (z ^ (z >> _SHIFT30)) * _MIX1
        z = (z ^ (z >> _SHIFT27)) * _MIX2
        return z ^ (z >> _SHIFT31)


def compute_tie_key(frame: pl.DataFrame, user_col: str, item_col: str, seed: int) -> pl.Series:
    """UInt64 tie-break key ``splitmix64(((user << 32) | item) ^ splitmix64(seed))`` per row."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if frame.height == 0:
        return pl.Series(DEFAULT_TIE_KEY_COL, [], dtype=pl.UInt64)
    if frame[user_col].null_count() or frame[item_col].null_count():
        raise ValueError(f"{user_col} and {item_col} must not be null")
    users = frame[user_col].to_numpy()
    items = frame[item_col].to_numpy()
    for name, ids in ((user_col, users), (item_col, items)):
        if ids.min() < 0 or ids.max() >= _ID_LIMIT:
            raise ValueError(f"{name} must be in [0, 2**32) to build the tie-break key")
    base = (users.astype(np.uint64) << _SHIFT32) | items.astype(np.uint64)
    seed_mix = splitmix64(np.array([seed], dtype=np.uint64))[0]
    return pl.Series(DEFAULT_TIE_KEY_COL, splitmix64(base ^ seed_mix))


def _validate(df: pl.DataFrame, user_col: str, item_col: str, time_col: str, seed: int) -> None:
    missing = [c for c in (user_col, item_col, time_col) if c not in df.columns]
    if missing:
        raise ValueError(f"interactions are missing required columns: {missing}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if df[user_col].null_count() or df[item_col].null_count():
        raise ValueError(f"{user_col} and {item_col} must not be null")
    if df[time_col].null_count():
        raise ValueError("timestamps must not be null")
    if df.select(user_col, item_col).is_duplicated().any():
        raise ValueError("interactions contain duplicate (user, item) pairs; deduplicate first")
    too_few = df.group_by(user_col).len().filter(pl.col("len") < 3).height
    if too_few:
        raise ValueError(
            f"{too_few} users have fewer than 3 interactions; leave-one-out needs test, val "
            "and at least one train interaction per user (filter users first)"
        )


def leave_one_out_split(
    interactions: pl.DataFrame,
    *,
    seed: int = 42,
    user_col: str = "user_idx",
    item_col: str = "item_idx",
    time_col: str = "timestamp",
    tie_key_col: str = DEFAULT_TIE_KEY_COL,
) -> SplitResult:
    """Split positives into train / val / test with one held-out item per user for val and test.

    If ``tie_key_col`` is present it is used verbatim (the pipeline fills it from the raw ids);
    otherwise the same hash is computed over ``(user_col, item_col)``. The key column is never
    written to the output frames.
    """
    _validate(interactions, user_col, item_col, time_col, seed)
    if tie_key_col in interactions.columns:
        keyed = interactions.with_columns(pl.col(tie_key_col).cast(pl.UInt64).alias("_tb"))
        key_source = f"precomputed column '{tie_key_col}'"
    else:
        keyed = interactions.with_columns(
            compute_tie_key(interactions, user_col, item_col, seed).alias("_tb")
        )
        key_source = f"({user_col}, {item_col})"
    output_cols = [c for c in interactions.columns if c != tie_key_col]
    ordered = keyed.sort(
        [user_col, time_col, "_tb"], descending=[False, True, True], maintain_order=True
    ).with_columns(pl.int_range(pl.len()).over(user_col).alias("_pos"))
    test = ordered.filter(pl.col("_pos") == 0).select(output_cols)
    val = ordered.filter(pl.col("_pos") == 1).select(output_cols)
    train = ordered.filter(pl.col("_pos") >= 2).select(output_cols)
    tie_break = (
        f"splitmix64(((user << 32) | item) xor splitmix64(seed={seed})); key from {key_source}"
    )
    stats = _stats(interactions, train, val, test, user_col, item_col, time_col, seed, tie_break)
    return SplitResult(train=train, val=val, test=test, stats=stats)


def _cold_items(held_out: pl.DataFrame, train: pl.DataFrame, item_col: str) -> int:
    return (
        held_out.select(item_col)
        .unique()
        .join(train.select(item_col).unique(), on=item_col, how="anti")
        .height
    )


def _digest(val: pl.DataFrame, test: pl.DataFrame, user_col: str, item_col: str) -> str:
    """sha256 over the sorted (user, item) pairs of val and test.

    For a fixed set of positives train is the complement of val and test, so the digest
    identifies the split; a change confined to train membership is not visible to it.
    """
    digest = hashlib.sha256()
    for name, frame in (("val", val), ("test", test)):
        pairs = (
            frame.select(pl.col(user_col).cast(pl.Int64), pl.col(item_col).cast(pl.Int64))
            .sort(user_col, item_col)
            .to_numpy()
        )
        digest.update(name.encode("ascii"))
        digest.update(np.ascontiguousarray(pairs, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _stats(
    interactions: pl.DataFrame,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    user_col: str,
    item_col: str,
    time_col: str,
    seed: int,
    tie_break: str,
) -> dict[str, Any]:
    n_users = test.height
    n_items = interactions[item_col].n_unique()
    boundaries = (
        test.select(user_col, pl.col(time_col).alias("_t_test"))
        .join(val.select(user_col, pl.col(time_col).alias("_t_val")), on=user_col)
        .join(train.group_by(user_col).agg(pl.col(time_col).max().alias("_t_train")), on=user_col)
    )
    test_val_tie = boundaries["_t_test"] == boundaries["_t_val"]
    val_train_tie = boundaries["_t_val"] == boundaries["_t_train"]
    n_tv = int(test_val_tie.sum())
    n_vt = int(val_train_tie.sum())
    n_any = int((test_val_tie | val_train_tie).sum())
    return {
        "strategy": "leave_one_out",
        "seed": int(seed),
        "tie_break": tie_break,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(interactions.height),
        "n_train": int(train.height),
        "n_val": int(val.height),
        "n_test": int(test.height),
        "density": float(interactions.height / (n_users * n_items)) if n_users and n_items else 0.0,
        "users_with_boundary_tie": n_tv,
        "boundary_tie_fraction": float(n_tv / n_users) if n_users else 0.0,
        "users_with_val_train_tie": n_vt,
        "users_with_any_boundary_tie": n_any,
        "any_boundary_tie_fraction": float(n_any / n_users) if n_users else 0.0,
        "cold_test_items": int(_cold_items(test, train, item_col)),
        "cold_val_items": int(_cold_items(val, train, item_col)),
        "split_sha256": _digest(val, test, user_col, item_col),
    }


def write_splits(
    result: SplitResult, out_dir: str | Path, extra: dict[str, Any] | None = None
) -> Path:
    """Write ``train/val/test.parquet`` plus ``meta.json`` (stats merged with ``extra``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", result.train), ("val", result.val), ("test", result.test)):
        frame.write_parquet(out / f"{name}.parquet", compression="zstd")
    meta = {**result.stats, **(extra or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out
