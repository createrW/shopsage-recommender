"""Movie metadata parsing, implicit-feedback filtering and contiguous id remapping."""

from __future__ import annotations

import polars as pl
import pytest

from shopsage_rec.data.preprocess import (
    build_id_maps,
    filter_interactions,
    parse_movies,
    remap_ids,
)


def _movies(*rows: tuple[int, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"movieId": pl.Int32, "title": pl.String, "genres": pl.String},
        orient="row",
    )


def _ratings(*rows: tuple[int, int, float, int]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "userId": pl.Int32,
            "movieId": pl.Int32,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
        orient="row",
    )


# --------------------------------------------------------------------------- parse_movies


def test_parse_movies_extracts_trailing_year_and_clean_title() -> None:
    out = parse_movies(_movies((1, "Toy Story (1995)", "Adventure|Comedy")))

    row = out.row(0, named=True)
    assert row["year"] == 1995
    assert row["title_clean"] == "Toy Story"
    assert out.schema["year"] == pl.Int32


def test_parse_movies_handles_trailing_space_and_nested_parentheses() -> None:
    out = parse_movies(_movies((2, "Blade Runner (Director's Cut) (1982) ", "Sci-Fi")))

    row = out.row(0, named=True)
    assert row["year"] == 1982
    assert row["title_clean"] == "Blade Runner (Director's Cut)"


def test_parse_movies_without_year_gets_null_year_and_untouched_title() -> None:
    out = parse_movies(_movies((3, "Babylon 5", "Sci-Fi")))

    row = out.row(0, named=True)
    assert row["year"] is None
    assert row["title_clean"] == "Babylon 5"


def test_parse_movies_splits_genres_into_list_and_maps_sentinel_to_empty() -> None:
    out = parse_movies(
        _movies(
            (1, "Toy Story (1995)", "Adventure|Animation|Children"),
            (6, "Doom (2005)", "(no genres listed)"),
        )
    )

    assert out.schema["genres"] == pl.List(pl.String)
    assert out.filter(pl.col("movieId") == 1)["genres"].item().to_list() == [
        "Adventure",
        "Animation",
        "Children",
    ]
    assert out.filter(pl.col("movieId") == 6)["genres"].item().to_list() == []


def test_parse_movies_keeps_raw_title_column() -> None:
    out = parse_movies(_movies((1, "Toy Story (1995)", "Adventure")))

    assert out.row(0, named=True)["title"] == "Toy Story (1995)"


# --------------------------------------------------------------------------- filter_interactions


def test_filter_interactions_keeps_ratings_at_or_above_threshold() -> None:
    ratings = _ratings((1, 10, 3.5, 1), (1, 20, 4.0, 2), (1, 30, 5.0, 3))

    out = filter_interactions(
        ratings, min_rating=4.0, min_item_interactions=1, min_user_interactions=1
    )

    assert sorted(out["movieId"].to_list()) == [20, 30]


def test_filter_interactions_drops_rare_items_before_rare_users_in_a_single_pass() -> None:
    # item counts: i1=2, i2=2, i3=1 -> i3 dropped first.
    # then user counts: u1=2 (keep), u2=1 (drop), u3=1 (drop).
    # i1 ends with a single interaction but is kept: filtering is a single pass, not iterated.
    ratings = _ratings(
        (1, 1, 4.0, 1),
        (1, 2, 4.0, 2),
        (2, 1, 4.0, 3),
        (2, 3, 4.0, 4),
        (3, 2, 4.0, 5),
    )

    out = filter_interactions(
        ratings, min_rating=4.0, min_item_interactions=2, min_user_interactions=2
    )

    assert out["userId"].unique().to_list() == [1]
    assert sorted(out["movieId"].to_list()) == [1, 2]


def test_filter_interactions_preserves_schema() -> None:
    ratings = _ratings((1, 10, 4.0, 1), (1, 20, 4.0, 2))

    out = filter_interactions(
        ratings, min_rating=4.0, min_item_interactions=1, min_user_interactions=1
    )

    assert out.schema == ratings.schema


def test_filter_interactions_rejects_nonsensical_thresholds() -> None:
    ratings = _ratings((1, 10, 4.0, 1))

    with pytest.raises(ValueError):
        filter_interactions(
            ratings, min_rating=4.0, min_item_interactions=0, min_user_interactions=1
        )


# --------------------------------------------------------------------------- id maps


def test_build_id_maps_are_contiguous_and_sorted_by_original_id() -> None:
    inter = _ratings((7, 300, 4.0, 1), (3, 100, 4.0, 2), (7, 100, 4.0, 3), (3, 200, 4.0, 4))

    user_map, item_map = build_id_maps(inter)

    assert user_map.schema == {"user_idx": pl.Int32, "userId": pl.Int32}
    assert item_map.schema == {"item_idx": pl.Int32, "movieId": pl.Int32}
    assert user_map.to_dict(as_series=False) == {"user_idx": [0, 1], "userId": [3, 7]}
    assert item_map.to_dict(as_series=False) == {
        "item_idx": [0, 1, 2],
        "movieId": [100, 200, 300],
    }


def test_remap_ids_replaces_raw_ids_and_round_trips() -> None:
    inter = _ratings((7, 300, 4.0, 1), (3, 100, 5.0, 2), (7, 100, 4.0, 3))
    user_map, item_map = build_id_maps(inter)

    out = remap_ids(inter, user_map, item_map)

    assert out.schema == {
        "user_idx": pl.Int32,
        "item_idx": pl.Int32,
        "rating": pl.Float32,
        "timestamp": pl.Int64,
    }
    assert out.height == inter.height
    back = (
        out.join(user_map, on="user_idx")
        .join(item_map, on="item_idx")
        .select("userId", "movieId", "rating", "timestamp")
        .sort("timestamp")
    )
    assert back.equals(inter.sort("timestamp"))


def test_remap_ids_rejects_maps_that_lose_rows_or_duplicate_keys() -> None:
    inter = _ratings((1, 100, 4.0, 1), (2, 100, 4.0, 2))
    user_map, item_map = build_id_maps(inter)

    with pytest.raises(ValueError, match="rows lost"):
        remap_ids(inter, user_map.filter(pl.col("userId") != 2), item_map)

    # A duplicated key keeps the row count (user 1 matches twice, user 2 is lost), so the
    # height check alone would let it through.
    duplicated = pl.DataFrame(
        {"user_idx": [0, 1], "userId": [1, 1]}, schema={"user_idx": pl.Int32, "userId": pl.Int32}
    )
    with pytest.raises(ValueError, match="not unique"):
        remap_ids(inter, duplicated, item_map)


@pytest.mark.parametrize(
    ("title", "year", "clean"),
    [
        ("Big Bang Theory, The (2007-)", 2007, "Big Bang Theory, The"),
        ("Diplomatic Immunity (2009\u2013 )", 2009, "Diplomatic Immunity"),  # en dash
        ("Fawlty Towers (1975-1979)", 1975, "Fawlty Towers"),
        (
            "Mona and the Time of Burning Love (...) (1983))",
            1983,
            "Mona and the Time of Burning Love (...)",
        ),
        ("Toy Story (1995)", 1995, "Toy Story"),
    ],
)
def test_parse_movies_handles_year_ranges_and_stray_parentheses(
    title: str, year: int, clean: str
) -> None:
    out = parse_movies(_movies((1, title, "Comedy")))

    row = out.row(0, named=True)
    assert row["year"] == year
    assert row["title_clean"] == clean


def test_remap_ids_keeps_extra_columns_after_the_indices() -> None:
    inter = _ratings((7, 300, 4.0, 1), (3, 100, 5.0, 2)).with_columns(
        pl.lit(5).cast(pl.UInt64).alias("tie_key")
    )
    user_map, item_map = build_id_maps(inter)

    out = remap_ids(inter, user_map, item_map)

    assert out.columns == ["user_idx", "item_idx", "rating", "timestamp", "tie_key"]
    assert out.schema["tie_key"] == pl.UInt64
