"""Read the raw MovieLens-20M CSV files into compact, explicitly typed Polars frames.

Dtypes are pinned (Int32 ids, Float32 ratings, Int64 unix-second timestamps) so that the
20M-row ratings table fits in ~400 MB and every downstream artifact shares one schema.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

RATINGS_SCHEMA: dict[str, pl.DataType] = {
    "userId": pl.Int32(),
    "movieId": pl.Int32(),
    "rating": pl.Float32(),
    "timestamp": pl.Int64(),
}
MOVIES_SCHEMA: dict[str, pl.DataType] = {
    "movieId": pl.Int32(),
    "title": pl.String(),
    "genres": pl.String(),
}
TAGS_SCHEMA: dict[str, pl.DataType] = {
    "userId": pl.Int32(),
    "movieId": pl.Int32(),
    "tag": pl.String(),
    "timestamp": pl.Int64(),
}

_DOWNLOAD_HINT = (
    "Run `scripts/download_data.sh` to download and verify MovieLens-20M into data/raw/ml-20m/."
)


def _csv_path(raw_dir: str | Path, name: str) -> Path:
    path = Path(raw_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist. {_DOWNLOAD_HINT}")
    return path


def _read(raw_dir: str | Path, name: str, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.read_csv(_csv_path(raw_dir, name), schema_overrides=schema).select(list(schema))


def load_ratings(raw_dir: str | Path) -> pl.DataFrame:
    """``ratings.csv`` -> userId, movieId, rating, timestamp (20,000,263 rows for ML-20M)."""
    return _read(raw_dir, "ratings.csv", RATINGS_SCHEMA)


def load_movies(raw_dir: str | Path) -> pl.DataFrame:
    """``movies.csv`` -> movieId, title (raw, year still embedded), genres (pipe-separated)."""
    return _read(raw_dir, "movies.csv", MOVIES_SCHEMA)


def load_tags(raw_dir: str | Path) -> pl.DataFrame:
    """``tags.csv`` -> userId, movieId, tag, timestamp (465,564 rows for ML-20M)."""
    return _read(raw_dir, "tags.csv", TAGS_SCHEMA)
