"""Liang et al. (2018) strong-generalisation split: a replication of the authors' preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shopsage_rec.data.liang import LiangSplit, liang_split, partition_val_users

Pair = tuple[int, int]


def _reference(
    pairs: list[Pair], *, seed: int, n_heldout_users: int, test_prop: float
) -> tuple[list[Pair], list[Pair], list[Pair], list[Pair], list[Pair]]:
    """A literal transcription of the VAE-CF notebook's preprocessing, without pandas.

    ratings.csv is ordered by userId, then movieId, and pandas' groupby walks users in ascending
    order, so each user's items are visited in ascending movieId order.
    """
    by_user: dict[int, list[int]] = {}
    for user, item in sorted(pairs):
        by_user.setdefault(user, []).append(item)
    unique_uid = np.array(sorted(by_user))
    np_rng = np.random.RandomState(seed)  # np.random.seed(98765)
    unique_uid = unique_uid[np_rng.permutation(unique_uid.size)]
    n_users = unique_uid.size
    tr_users = unique_uid[: n_users - 2 * n_heldout_users]
    vd_users = unique_uid[n_users - 2 * n_heldout_users : n_users - n_heldout_users]
    te_users = unique_uid[n_users - n_heldout_users :]
    train = sorted((int(u), i) for u in tr_users for i in by_user[int(u)])
    unique_sid = {i for _, i in train}

    def split_train_test_proportion(users: np.ndarray) -> tuple[list[Pair], list[Pair]]:
        np_rng = np.random.RandomState(seed)  # reseeded inside the notebook's function
        tr_list: list[Pair] = []
        te_list: list[Pair] = []
        for user in sorted(int(u) for u in users):
            items = [i for i in by_user[user] if i in unique_sid]
            n_items_u = len(items)
            if n_items_u == 0:  # no rows left: the user is absent from the groupby
                continue
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype="bool")
                chosen = np_rng.choice(n_items_u, size=int(test_prop * n_items_u), replace=False)
                idx[chosen.astype("int64")] = True
                tr_list += [(user, i) for i, held in zip(items, idx, strict=True) if not held]
                te_list += [(user, i) for i, held in zip(items, idx, strict=True) if held]
            else:
                tr_list += [(user, i) for i in items]
        return sorted(tr_list), sorted(te_list)

    vad_tr, vad_te = split_train_test_proportion(vd_users)
    test_tr, test_te = split_train_test_proportion(te_users)
    return train, vad_tr, vad_te, test_tr, test_te


def _positives(pairs: list[Pair]) -> pl.DataFrame:
    n = len(pairs)
    return pl.DataFrame(
        {
            "userId": [u for u, _ in pairs],
            "movieId": [i for _, i in pairs],
            "rating": [4.0] * n,
            "timestamp": list(range(n)),
        },
        schema={
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )


def _random_pairs(seed: int, n_users: int = 60, n_items: int = 80) -> list[Pair]:
    """Users with 1-40 positives over a skewed catalogue, so some held-out users end with < 5
    training items and some items are rated by held-out users only."""
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, n_items + 1)
    popularity /= popularity.sum()
    pairs: list[Pair] = []
    for user in rng.choice(np.arange(1, 10 * n_users), size=n_users, replace=False):
        k = int(rng.integers(1, 41))
        items = rng.choice(np.arange(1, n_items + 1) * 7, size=k, replace=False, p=popularity)
        pairs += [(int(user), int(i)) for i in items]
    return pairs


def _pairs(frame: pl.DataFrame) -> list[Pair]:
    return sorted(zip(frame["userId"].to_list(), frame["movieId"].to_list(), strict=True))


def _as_tuple(split: LiangSplit) -> tuple[list[Pair], ...]:
    return tuple(
        _pairs(f) for f in (split.train, split.val_tr, split.val_te, split.test_tr, split.test_te)
    )


@pytest.mark.parametrize(("data_seed", "test_prop"), [(0, 0.2), (1, 0.2), (2, 0.5), (3, 0.35)])
def test_liang_split_matches_the_notebook_transcription(data_seed: int, test_prop: float) -> None:
    pairs = _random_pairs(data_seed)

    split = liang_split(_positives(pairs), seed=98765, n_heldout_users=10, test_prop=test_prop)

    assert _as_tuple(split) == _reference(
        pairs, seed=98765, n_heldout_users=10, test_prop=test_prop
    )


@settings(max_examples=25, deadline=None)
@given(
    data_seed=st.integers(0, 10_000),
    seed=st.integers(0, 2**32 - 1),
    n_heldout=st.integers(1, 12),
    test_prop=st.sampled_from([0.1, 0.2, 0.25, 0.5]),
)
def test_liang_split_matches_the_transcription_for_any_seed(
    data_seed: int, seed: int, n_heldout: int, test_prop: float
) -> None:
    pairs = _random_pairs(data_seed, n_users=30)

    split = liang_split(
        _positives(pairs), seed=seed, n_heldout_users=n_heldout, test_prop=test_prop
    )

    assert _as_tuple(split) == _reference(
        pairs, seed=seed, n_heldout_users=n_heldout, test_prop=test_prop
    )


def test_liang_split_partitions_users_and_restricts_held_out_items() -> None:
    pairs = _random_pairs(7)
    split = liang_split(_positives(pairs), seed=98765, n_heldout_users=10, test_prop=0.2)

    train_users = set(split.train["userId"].to_list())
    val_users = set(split.val_tr["userId"].to_list()) | set(split.val_te["userId"].to_list())
    test_users = set(split.test_tr["userId"].to_list()) | set(split.test_te["userId"].to_list())
    assert not train_users & val_users
    assert not train_users & test_users
    assert not val_users & test_users
    assert split.stats["n_train_users"] == 40
    assert split.stats["n_val_users"] == split.stats["n_test_users"] == 10

    catalogue = set(split.train["movieId"].to_list())
    assert split.stats["n_items"] == len(catalogue)
    by_user: dict[int, set[int]] = {}
    for user, item in pairs:
        by_user.setdefault(user, set()).add(item)
    for fold_in, targets in ((split.val_tr, split.val_te), (split.test_tr, split.test_te)):
        assert set(fold_in["movieId"].to_list()) | set(targets["movieId"].to_list()) <= catalogue
        assert not set(_pairs(fold_in)) & set(_pairs(targets))
        n_targets = targets.group_by("userId").len()
        n_fold_in = fold_in.group_by("userId").len()
        for user in set(fold_in["userId"].to_list()) | set(targets["userId"].to_list()):
            n = len(by_user[user] & catalogue)
            got_t = n_targets.filter(pl.col("userId") == user)["len"].sum()
            got_f = n_fold_in.filter(pl.col("userId") == user)["len"].sum()
            assert got_t == (int(0.2 * n) if n >= 5 else 0)
            assert got_t + got_f == n


def test_liang_split_keeps_extra_columns_and_ignores_row_order() -> None:
    frame = _positives(_random_pairs(11))
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=3)

    a = liang_split(frame, seed=98765, n_heldout_users=10)
    b = liang_split(shuffled, seed=98765, n_heldout_users=10)

    assert _as_tuple(a) == _as_tuple(b)
    assert a.stats["split_sha256"] == b.stats["split_sha256"]
    assert a.train.columns == ["userId", "movieId", "rating", "timestamp"]
    assert a.val_te.sort("userId", "movieId").equals(b.val_te.sort("userId", "movieId"))


def test_liang_split_depends_on_the_seed() -> None:
    frame = _positives(_random_pairs(5))

    a = liang_split(frame, seed=1, n_heldout_users=10)
    b = liang_split(frame, seed=2, n_heldout_users=10)

    assert a.stats["split_sha256"] != b.stats["split_sha256"]


@pytest.mark.parametrize(
    ("frame", "kwargs", "message"),
    [
        (_positives([(1, 1), (1, 1), (2, 1), (3, 1)]), {"n_heldout_users": 1}, "duplicate"),
        (_positives([(1, 1), (2, 1)]), {"n_heldout_users": 1}, "more than"),
        (_positives([(1, 1), (2, 1), (3, 1)]), {"n_heldout_users": 0}, "n_heldout_users"),
        (_positives([(1, 1), (2, 1), (3, 1)]), {"n_heldout_users": 1, "test_prop": 1.0}, "prop"),
        (_positives([(1, 1), (2, 1), (3, 1)]), {"n_heldout_users": 1, "seed": -1}, "seed"),
    ],
)
def test_liang_split_rejects_invalid_input(
    frame: pl.DataFrame, kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        liang_split(frame, **kwargs)  # type: ignore[arg-type]


def test_liang_split_rejects_null_ids() -> None:
    frame = _positives([(1, 1), (2, 1), (3, 1)]).with_columns(
        pl.when(pl.col("userId") == 2).then(None).otherwise(pl.col("userId")).alias("userId")
    )

    with pytest.raises(ValueError, match="null"):
        liang_split(frame, n_heldout_users=1)


def test_partition_val_users_is_seeded_disjoint_and_covering() -> None:
    users = np.arange(100, 200)

    va, vb = partition_val_users(users, seed=42, frac_va=0.5)
    va2, vb2 = partition_val_users(users[::-1], seed=42, frac_va=0.5)

    assert len(va) == len(vb) == 50
    assert set(va) | set(vb) == set(users.tolist())
    assert not set(va) & set(vb)
    assert np.array_equal(va, va2)  # independent of the input order
    assert np.array_equal(vb, vb2)
    assert np.all(np.diff(va) > 0)  # returned sorted


REAL_RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-20m"


@pytest.mark.slow
@pytest.mark.skipif(not (REAL_RAW / "ratings.csv").is_file(), reason="needs MovieLens-20M")
def test_liang_split_reproduces_the_published_counts() -> None:
    from shopsage_rec.data.loader import load_ratings
    from shopsage_rec.data.preprocess import filter_interactions

    positives = filter_interactions(
        load_ratings(REAL_RAW), min_rating=4.0, min_item_interactions=1, min_user_interactions=5
    )
    # "After filtering, there are 9990682 watching events from 136677 users and 20720 movies"
    assert positives.height == 9_990_682
    assert positives["userId"].n_unique() == 136_677
    assert positives["movieId"].n_unique() == 20_720

    split = liang_split(positives, seed=98765, n_heldout_users=10_000, test_prop=0.2)

    assert split.stats["n_train_users"] == 116_677
    assert split.stats["n_items"] == 20_108  # Liang et al. 2018, Table 1
