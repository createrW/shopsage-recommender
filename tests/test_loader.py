"""Loading the raw MovieLens-20M CSVs into typed Polars frames."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from shopsage_rec.data.loader import load_movies, load_ratings, load_tags


def test_load_ratings_has_compact_dtypes_and_all_rows(tiny_raw_dir: Path) -> None:
    ratings = load_ratings(tiny_raw_dir)

    assert ratings.schema == {
        "userId": pl.Int32,
        "movieId": pl.Int32,
        "rating": pl.Float32,
        "timestamp": pl.Int64,
    }
    assert ratings.height == 30


def test_load_movies_keeps_raw_title_and_pipe_separated_genres(tiny_raw_dir: Path) -> None:
    movies = load_movies(tiny_raw_dir)

    assert movies.schema == {"movieId": pl.Int32, "title": pl.String, "genres": pl.String}
    row = movies.filter(pl.col("movieId") == 10).row(0, named=True)
    assert row["title"] == "Toy Story (1995)"
    assert row["genres"] == "Adventure|Animation|Children|Comedy|Fantasy"


def test_load_tags_parses_quoted_tags(tiny_raw_dir: Path) -> None:
    tags = load_tags(tiny_raw_dir)

    assert tags.schema == {
        "userId": pl.Int32,
        "movieId": pl.Int32,
        "tag": pl.String,
        "timestamp": pl.Int64,
    }
    assert tags.filter(pl.col("userId") == 2)["tag"].item() == "animated, funny"


def test_missing_raw_file_points_to_download_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"scripts/download_data\.sh"):
        load_ratings(tmp_path)
