"""Item-tower inputs, one row per ``item_idx`` of a prepared split.

- **genres:** multi-hot over the 19 MovieLens genres ("(no genres listed)" is an all-zero row);
- **year bucket:** ``(year - min_year) // width + 1``, with 0 reserved for an unknown year;
- **genome:** the GroupLens tag genome (1,128 relevance scores per movie) projected on its first
  ``pca_dim`` principal components, standardised, with a flag for movies the genome does not
  cover (their row is zero). The genome is computed by GroupLens over all users, a documented mild
  global leak (README, Limitations).

The arrays are cached in ``<processed_dir>/item_features.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
CACHE_NAME = "item_features.npz"


@dataclass(frozen=True)
class ItemFeatures:
    genres: np.ndarray  # float32 [n_items, len(GENRES)]
    year_bucket: np.ndarray  # int64 [n_items], 0 = unknown year
    n_year_buckets: int
    genome: np.ndarray  # float32 [n_items, pca_dim], zero where missing
    genome_missing: np.ndarray  # float32 [n_items], 1.0 where the genome has no row

    @property
    def n_items(self) -> int:
        return int(self.genres.shape[0])


def year_buckets(years: np.ndarray, *, width: int = 5) -> tuple[np.ndarray, int]:
    """Bucket ids for release years (NaN = unknown -> 0) and the number of buckets."""
    years = np.asarray(years, dtype=np.float64)
    known = ~np.isnan(years)
    buckets = np.zeros(years.size, dtype=np.int64)
    if known.any():
        first = years[known].min()
        buckets[known] = ((years[known] - first) // width).astype(np.int64) + 1
    return buckets, int(np.max(buckets)) + 1


def genome_pca(scores: np.ndarray, *, dim: int = 64) -> np.ndarray:
    """Project rows of ``scores`` on the first ``dim`` principal components, each scaled to unit
    variance."""
    x = np.asarray(scores, dtype=np.float64)
    if not 1 <= dim <= min(x.shape):
        raise ValueError(f"dim must be in [1, {min(x.shape)}], got {dim}")
    x = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:dim].T
    proj /= proj.std(axis=0) + 1e-12
    return proj.astype(np.float32)


def _load_cache(path: Path, pca_dim: int, n_items: int) -> ItemFeatures | None:
    if not path.is_file():
        return None
    with np.load(path) as z:
        if int(z["pca_dim"]) != pca_dim or z["genres"].shape[0] != n_items:
            return None
        return ItemFeatures(
            genres=z["genres"],
            year_bucket=z["year_bucket"],
            n_year_buckets=int(z["n_year_buckets"]),
            genome=z["genome"],
            genome_missing=z["genome_missing"],
        )


def build_item_features(
    processed_dir: str | Path,
    raw_dir: str | Path,
    *,
    pca_dim: int = 64,
    year_width: int = 5,
) -> ItemFeatures:
    """Build (or load from the cache) the item features of a prepared split."""
    processed = Path(processed_dir)
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet").sort("item_idx")
    cache = processed / CACHE_NAME
    cached = _load_cache(cache, pca_dim, item_map.height)
    if cached is not None:
        return cached

    movies = item_map.join(pl.read_parquet(processed / "movies.parquet"), on="movieId", how="left")
    movies = movies.sort("item_idx")
    genres = np.zeros((movies.height, len(GENRES)), dtype=np.float32)
    column = {g: j for j, g in enumerate(GENRES)}
    for row, names in enumerate(movies["genres"].to_list()):
        for name in names or ():
            if name in column:
                genres[row, column[name]] = 1.0
    years = movies["year"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    bucket, n_buckets = year_buckets(years, width=year_width)

    scores = pl.read_csv(
        Path(raw_dir) / "genome-scores.csv",
        schema_overrides={"movieId": pl.Int32, "tagId": pl.Int32, "relevance": pl.Float32},
    ).join(item_map, on="movieId", how="inner")
    covered = np.sort(scores["item_idx"].unique().to_numpy())
    n_tags = int(scores["tagId"].to_numpy().max())
    dense = np.zeros((covered.size, n_tags), dtype=np.float32)
    dense[
        np.searchsorted(covered, scores["item_idx"].to_numpy()), scores["tagId"].to_numpy() - 1
    ] = scores["relevance"].to_numpy()
    genome = np.zeros((movies.height, pca_dim), dtype=np.float32)
    genome[covered] = genome_pca(dense, dim=pca_dim)
    missing = np.ones(movies.height, dtype=np.float32)
    missing[covered] = 0.0

    feats = ItemFeatures(
        genres=genres,
        year_bucket=bucket,
        n_year_buckets=n_buckets,
        genome=genome,
        genome_missing=missing,
    )
    np.savez(
        cache,
        genres=feats.genres,
        year_bucket=feats.year_bucket,
        n_year_buckets=np.int64(n_buckets),
        genome=feats.genome,
        genome_missing=feats.genome_missing,
        pca_dim=np.int64(pca_dim),
    )
    return feats
