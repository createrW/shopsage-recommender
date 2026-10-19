"""Leave-one-out split invariants, checked with hypothesis-generated interaction tables."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shopsage_rec.data.split import (
    SplitResult,
    compute_tie_key,
    leave_one_out_split,
    write_splits,
)

SCHEMA = {
    "user_idx": pl.Int32,
    "item_idx": pl.Int32,
    "rating": pl.Float32,
    "timestamp": pl.Int64,
}


@st.composite
def interaction_tables(
    draw: st.DrawFn,
    min_users: int = 1,
    max_users: int = 8,
    min_items_per_user: int = 3,
    max_items_per_user: int = 7,
    n_items: int = 12,
    max_timestamp: int = 6,  # small range => frequent ties, which is the hard case
) -> pl.DataFrame:
    n_users = draw(st.integers(min_users, max_users))
    rows: list[tuple[int, int, float, int]] = []
    for user in range(n_users):
        k = draw(st.integers(min_items_per_user, max_items_per_user))
        items = draw(st.lists(st.integers(0, n_items - 1), min_size=k, max_size=k, unique=True))
        stamps = draw(st.lists(st.integers(0, max_timestamp), min_size=k, max_size=k))
        rows.extend((user, item, 4.0, ts) for item, ts in zip(items, stamps, strict=True))
    return pl.DataFrame(rows, schema=SCHEMA, orient="row")


def _canonical(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("user_idx", "item_idx", "rating", "timestamp").sort(
        "user_idx", "item_idx", "timestamp"
    )


@settings(max_examples=150, deadline=None)
@given(interaction_tables())
def test_every_user_has_exactly_one_test_and_one_val_row(table: pl.DataFrame) -> None:
    result = leave_one_out_split(table, seed=1)

    users = set(table["user_idx"].to_list())
    assert result.test["user_idx"].to_list() == sorted(users)
    assert result.val["user_idx"].to_list() == sorted(users)
    assert result.test["user_idx"].n_unique() == result.test.height
    assert result.val["user_idx"].n_unique() == result.val.height


@settings(max_examples=150, deadline=None)
@given(interaction_tables())
def test_splits_partition_the_input_exactly(table: pl.DataFrame) -> None:
    result = leave_one_out_split(table, seed=1)

    union = pl.concat([result.train, result.val, result.test])
    assert _canonical(union).equals(_canonical(table))
    assert union.height == table.height
    pairs = union.select("user_idx", "item_idx")
    assert pairs.n_unique() == pairs.height


@settings(max_examples=150, deadline=None)
@given(interaction_tables())
def test_test_is_newest_then_val_then_train(table: pl.DataFrame) -> None:
    result = leave_one_out_split(table, seed=1)

    train_max = result.train.group_by("user_idx").agg(pl.col("timestamp").max().alias("train_max"))
    joined = (
        result.test.select("user_idx", pl.col("timestamp").alias("test_ts"))
        .join(result.val.select("user_idx", pl.col("timestamp").alias("val_ts")), on="user_idx")
        .join(train_max, on="user_idx")
    )
    assert joined.height == table["user_idx"].n_unique()
    assert (joined["test_ts"] >= joined["val_ts"]).all()
    assert (joined["val_ts"] >= joined["train_max"]).all()


@settings(max_examples=60, deadline=None)
@given(interaction_tables())
def test_split_is_deterministic_for_a_given_seed(table: pl.DataFrame) -> None:
    first = leave_one_out_split(table, seed=7)
    second = leave_one_out_split(table.sample(fraction=1.0, shuffle=True, seed=0), seed=7)

    assert _canonical(first.test).equals(_canonical(second.test))
    assert _canonical(first.val).equals(_canonical(second.val))
    assert _canonical(first.train).equals(_canonical(second.train))


def test_split_output_schema_matches_input(interactions: pl.DataFrame) -> None:
    result = leave_one_out_split(interactions, seed=1)

    for frame in (result.train, result.val, result.test):
        assert frame.schema == interactions.schema


def test_users_with_fewer_than_three_interactions_are_rejected() -> None:
    table = pl.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 1],
            "item_idx": [0, 1, 0, 1, 2],
            "rating": [4.0] * 5,
            "timestamp": [1, 2, 1, 2, 3],
        },
        schema=SCHEMA,
    )

    with pytest.raises(ValueError, match="fewer than 3"):
        leave_one_out_split(table, seed=1)


def test_duplicate_user_item_pairs_are_rejected() -> None:
    table = pl.DataFrame(
        {
            "user_idx": [0, 0, 0, 0],
            "item_idx": [0, 1, 2, 2],
            "rating": [4.0] * 4,
            "timestamp": [1, 2, 3, 4],
        },
        schema=SCHEMA,
    )

    with pytest.raises(ValueError, match="duplicate"):
        leave_one_out_split(table, seed=1)


def test_tie_breaking_depends_on_seed_not_on_item_id() -> None:
    # 200 users who rated 4 items in the same second. If ties were broken by item id,
    # every user's test item would be item 3 (or item 0). With a seeded hash the chosen
    # item varies across users and across seeds.
    n_users = 200
    table = pl.DataFrame(
        {
            "user_idx": [u for u in range(n_users) for _ in range(4)],
            "item_idx": [i for _ in range(n_users) for i in range(4)],
            "rating": [4.0] * (4 * n_users),
            "timestamp": [1_000] * (4 * n_users),
        },
        schema=SCHEMA,
    )

    a = leave_one_out_split(table, seed=1)
    b = leave_one_out_split(table, seed=2)

    assert a.test["item_idx"].n_unique() > 1
    assert not a.test["item_idx"].equals(b.test["item_idx"])


def test_stats_report_boundary_ties_and_cold_test_items() -> None:
    table = pl.DataFrame(
        {
            # user 0: distinct timestamps, no tie; its test item (3) never appears in train -> cold
            # user 1: all four interactions in the same second -> boundary tie
            "user_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "item_idx": [0, 1, 2, 3, 0, 1, 2, 4],
            "rating": [4.0] * 8,
            "timestamp": [1, 2, 3, 4, 9, 9, 9, 9],
        },
        schema=SCHEMA,
    )

    result = leave_one_out_split(table, seed=1)

    assert isinstance(result, SplitResult)
    assert result.stats["n_users"] == 2
    assert result.stats["n_items"] == 5
    assert result.stats["n_train"] == 4
    assert result.stats["n_val"] == 2
    assert result.stats["n_test"] == 2
    assert result.stats["users_with_boundary_tie"] == 1
    assert result.stats["boundary_tie_fraction"] == pytest.approx(0.5)
    test_items = set(result.test["item_idx"].to_list())
    val_items = set(result.val["item_idx"].to_list())
    train_items = set(result.train["item_idx"].to_list())
    assert result.stats["cold_test_items"] == len(test_items - train_items)
    assert result.stats["cold_val_items"] == len(val_items - train_items)


def test_split_rejects_null_user_or_item_ids(interactions: pl.DataFrame) -> None:
    null_users = interactions.with_columns(
        pl.when(pl.col("user_idx") == 1).then(None).otherwise(pl.col("user_idx")).alias("user_idx")
    )
    null_items = interactions.with_columns(
        pl.when(pl.col("item_idx") == 2).then(None).otherwise(pl.col("item_idx")).alias("item_idx")
    )

    with pytest.raises(ValueError, match="null"):
        leave_one_out_split(null_users, seed=1)
    with pytest.raises(ValueError, match="null"):
        leave_one_out_split(null_items, seed=1)


def test_compute_tie_key_rejects_null_ids() -> None:
    frame = pl.DataFrame(
        {"userId": [1, None, 3], "movieId": [10, 20, 30]},
        schema={"userId": pl.Int32, "movieId": pl.Int32},
    )

    with pytest.raises(ValueError, match="null"):
        compute_tie_key(frame, "userId", "movieId", seed=1)


def test_write_splits_creates_three_parquet_files_and_meta(
    interactions: pl.DataFrame, tmp_path: Path
) -> None:
    result = leave_one_out_split(interactions, seed=1)

    write_splits(result, tmp_path / "splits")

    for name in ("train", "val", "test"):
        path = tmp_path / "splits" / f"{name}.parquet"
        assert path.exists()
        assert pl.read_parquet(path).schema == interactions.schema
    assert (tmp_path / "splits" / "meta.json").exists()


# --------------------------------------------------------------------------- tie-break quality


def _tied_table(n_users: int, n_items: int = 4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_idx": [u for u in range(n_users) for _ in range(n_items)],
            "item_idx": [i for _ in range(n_users) for i in range(n_items)],
            "rating": [4.0] * (n_items * n_users),
            "timestamp": [1_000] * (n_items * n_users),
        },
        schema=SCHEMA,
    )


def test_tie_break_is_not_biased_toward_any_item_id() -> None:
    # 2,000 users with four items rated in the same second. An unbiased tie-break picks each
    # item as the test item for ~25% of users (sd ~1 point); an affine hash would pick one id
    # far more often than the others.
    table = _tied_table(2_000)

    result = leave_one_out_split(table, seed=1)

    shares = result.test.group_by("item_idx").len().with_columns(pl.col("len") / 2_000)
    assert shares.height == 4
    assert shares["len"].is_between(0.15, 0.35, closed="none").all()


def test_splitmix64_matches_reference_vectors_and_pure_python() -> None:
    import numpy as np

    from shopsage_rec.data.split import splitmix64

    # First outputs of SplitMix64 for seeds 0 and 1 (Steele, Lea & Flood 2014; also the
    # values used to seed xoshiro generators in the reference implementations).
    out = splitmix64(np.array([0, 1], dtype=np.uint64))
    assert int(out[0]) == 0xE220A8397B1DCDAF
    assert int(out[1]) == 0x910A2DEC89025CC1

    def reference(x: int) -> int:
        mask = (1 << 64) - 1
        z = (x + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        return z ^ (z >> 31)

    rng = np.random.default_rng(0)
    xs = rng.integers(0, 2**63 - 1, size=50, dtype=np.int64).astype(np.uint64)
    assert [int(v) for v in splitmix64(xs)] == [reference(int(x)) for x in xs]


def test_precomputed_tie_key_column_is_used_and_dropped_from_outputs() -> None:
    table = _tied_table(50).with_columns(
        # force item 2 to win every tie: larger key wins
        pl.when(pl.col("item_idx") == 2)
        .then(pl.lit(10))
        .otherwise(pl.lit(1))
        .cast(pl.UInt64)
        .alias("tie_key")
    )

    result = leave_one_out_split(table, seed=1)

    assert result.test["item_idx"].unique().to_list() == [2]
    for frame in (result.train, result.val, result.test):
        assert "tie_key" not in frame.columns
        assert frame.schema == SCHEMA


def test_compute_tie_key_depends_on_raw_ids_and_seed() -> None:
    from shopsage_rec.data.split import compute_tie_key

    frame = pl.DataFrame(
        {"userId": [1, 1, 2], "movieId": [10, 20, 10]},
        schema={"userId": pl.Int32, "movieId": pl.Int32},
    )

    k1 = compute_tie_key(frame, "userId", "movieId", seed=1)
    k2 = compute_tie_key(frame, "userId", "movieId", seed=2)

    assert k1.dtype == pl.UInt64
    assert k1.n_unique() == 3
    assert not k1.equals(k2)
    assert compute_tie_key(frame, "userId", "movieId", seed=1).equals(k1)


def test_golden_split_is_pinned() -> None:
    # Characterisation test: any change to the tie-break or ordering logic changes this
    # digest and must be a deliberate, documented decision (it changes val/test for ~24% of users).
    rows = [
        (0, 0, 4.0, 5),
        (0, 1, 4.0, 5),
        (0, 2, 4.0, 5),
        (0, 3, 4.0, 1),
        (1, 1, 4.0, 9),
        (1, 2, 4.0, 9),
        (1, 3, 4.0, 9),
        (1, 4, 4.0, 9),
        (2, 0, 4.0, 3),
        (2, 2, 4.0, 2),
        (2, 4, 4.0, 1),
        (3, 5, 4.0, 7),
        (3, 6, 4.0, 7),
        (3, 7, 4.0, 6),
        (3, 8, 4.0, 6),
        (3, 9, 4.0, 6),
        (4, 0, 4.0, 1),
        (4, 1, 4.0, 1),
        (4, 5, 4.0, 1),
        (4, 9, 4.0, 1),
    ]
    table = pl.DataFrame(rows, schema=SCHEMA, orient="row")

    result = leave_one_out_split(table, seed=7)

    assert (
        result.stats["split_sha256"]
        == "4b4826124993520050abfd526c07ded76aac56249c85708f33eaf0ded7046a47"
    )


def test_stats_include_both_boundaries_and_digest() -> None:
    table = pl.DataFrame(
        {
            # user 0: no ties at all; user 1: all four in the same second (both boundaries tied);
            # user 2: val and newest train tied, test strictly newer.
            "user_idx": [0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
            "item_idx": [0, 1, 2, 0, 1, 2, 3, 0, 1, 2],
            "rating": [4.0] * 10,
            "timestamp": [1, 2, 3, 9, 9, 9, 9, 5, 5, 8],
        },
        schema=SCHEMA,
    )

    result = leave_one_out_split(table, seed=1)

    assert result.stats["users_with_boundary_tie"] == 1
    assert result.stats["users_with_val_train_tie"] == 2
    assert result.stats["users_with_any_boundary_tie"] == 2
    assert result.stats["any_boundary_tie_fraction"] == pytest.approx(2 / 3)
    assert len(result.stats["split_sha256"]) == 64
    assert "splitmix64" in result.stats["tie_break"]
