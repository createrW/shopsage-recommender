"""Turn raw MovieLens tables into implicit-feedback interactions with contiguous ids.

A rating at or above ``min_rating`` is a positive, and rare items, then rare users, are
dropped in a single pass. The positive threshold and the items-then-users order follow Liang
et al. (2018); the item threshold itself (5 in the Day-1 config) does not: Liang applies no
item filter (D-002), and the single pass is not an iterated k-core. Raw ratings are *not*
discarded elsewhere (low ratings are explicit negatives that the ranker can exploit later);
this module only defines what counts as a positive.
"""

from __future__ import annotations

import polars as pl

NO_GENRES_SENTINEL = "(no genres listed)"
# "(1995)", "(2007-)", "(2009<en dash> )", "(1975-1979)" and a stray trailing ")" as in "(1983))".
# The first year of a range is the release year.
_YEAR_SUFFIX = r"\((\d{4})(?:\s*[-\u2013]\s*(?:\d{4})?\s*)?\)\)?$"
_YEAR_SUFFIX_WITH_SPACE = r"\s*\(\d{4}(?:\s*[-\u2013]\s*(?:\d{4})?\s*)?\)\)?$"

INTERACTION_COLUMNS = ("userId", "movieId", "rating", "timestamp")


def parse_movies(movies: pl.DataFrame) -> pl.DataFrame:
    """Extract the release year, a year-free title and a genre list from ``movies.csv``.

    Titles look like ``"Toy Story (1995)"``; some carry trailing whitespace, nested
    parentheses (``"Blade Runner (Director's Cut) (1982) "``), year ranges for series
    (``"Fawlty Towers (1975-1979)"`` -> 1975) or a stray parenthesis, and a few have no year
    at all, which yields a null ``year``. ``"(no genres listed)"`` becomes an empty list.
    """
    title = pl.col("title").str.strip_chars()
    genres = pl.col("genres").str.split("|")
    return movies.select(
        pl.col("movieId"),
        pl.col("title"),
        title.str.replace(_YEAR_SUFFIX_WITH_SPACE, "").str.strip_chars().alias("title_clean"),
        title.str.extract(_YEAR_SUFFIX, 1).cast(pl.Int32).alias("year"),
        genres.list.eval(pl.element().filter(pl.element() != NO_GENRES_SENTINEL)).alias("genres"),
    )


def filter_interactions(
    ratings: pl.DataFrame,
    *,
    min_rating: float,
    min_item_interactions: int,
    min_user_interactions: int,
) -> pl.DataFrame:
    """Keep positives (``rating >= min_rating``) of sufficiently popular items and active users.

    The filter is a single pass — items first, then users — so an item may end up with
    fewer than ``min_item_interactions`` rows after inactive users are removed (not an
    iterated k-core). The order mirrors Liang et al. (2018) and keeps the procedure easy to
    reason about.
    """
    if min_item_interactions < 1 or min_user_interactions < 1:
        raise ValueError("min_item_interactions and min_user_interactions must both be >= 1")
    return (
        ratings.filter(pl.col("rating") >= min_rating)
        .filter(pl.len().over("movieId") >= min_item_interactions)
        .filter(pl.len().over("userId") >= min_user_interactions)
        .select(list(INTERACTION_COLUMNS))
    )


def _id_map(frame: pl.DataFrame, raw_col: str, idx_col: str) -> pl.DataFrame:
    return (
        frame.select(pl.col(raw_col).unique().sort())
        .with_row_index(idx_col)
        .select(pl.col(idx_col).cast(pl.Int32), pl.col(raw_col))
    )


def build_id_maps(interactions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Contiguous ``user_idx`` / ``item_idx`` (0..n-1, ascending by raw id) for embedding tables."""
    return _id_map(interactions, "userId", "user_idx"), _id_map(interactions, "movieId", "item_idx")


def remap_ids(
    interactions: pl.DataFrame, user_map: pl.DataFrame, item_map: pl.DataFrame
) -> pl.DataFrame:
    """Replace raw MovieLens ids with contiguous indices; every input row must map.

    Columns other than ``userId`` / ``movieId`` (``rating``, ``timestamp`` and e.g. a
    precomputed ``tie_key``) pass through unchanged, after the two index columns.
    """
    for id_map, key in ((user_map, "userId"), (item_map, "movieId")):
        if id_map[key].is_duplicated().any():
            raise ValueError(f"{key} is not unique in its id map")
    passthrough = [c for c in interactions.columns if c not in ("userId", "movieId")]
    out = (
        interactions.join(user_map, on="userId", how="inner")
        .join(item_map, on="movieId", how="inner")
        .select("user_idx", "item_idx", *passthrough)
    )
    if out.height != interactions.height:
        raise ValueError(
            f"id maps do not cover every interaction: {interactions.height - out.height} rows lost"
        )
    return out
