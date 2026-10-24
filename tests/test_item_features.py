"""Item-tower inputs: genre multi-hot, release-year buckets, genome PCA with a missing flag."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from shopsage_rec.features.items import (
    GENRES,
    build_item_features,
    genome_pca,
    year_buckets,
)


def test_year_buckets_reserve_zero_for_unknown_years() -> None:
    years = np.array([1995.0, np.nan, 1999.0, 2000.0, 1891.0])

    buckets, n_buckets = year_buckets(years, width=5)

    assert buckets.tolist() == [21, 0, 22, 22, 1]  # (year - 1891) // 5 + 1
    assert n_buckets == 23


def test_genome_pca_projects_to_standardised_components() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((300, 40)).astype(np.float32)

    proj = genome_pca(scores, dim=8)

    assert proj.shape == (300, 8)
    assert proj.dtype == np.float32
    assert np.allclose(proj.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(proj.std(axis=0), 1.0, atol=1e-3)
    with pytest.raises(ValueError, match="dim"):
        genome_pca(scores, dim=41)


def _processed(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "liang"
    (processed / "mappings").mkdir(parents=True)
    pl.DataFrame(
        {"item_idx": [0, 1, 2], "movieId": [10, 20, 30]},
        schema={"item_idx": pl.Int32, "movieId": pl.Int32},
    ).write_parquet(processed / "mappings" / "item_map.parquet")
    pl.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "year": [1995, None, 2001, 1990],
            "genres": [["Comedy", "Drama"], [], ["Sci-Fi"], ["Action"]],
        },
        schema={"movieId": pl.Int32, "year": pl.Int32, "genres": pl.List(pl.String)},
    ).write_parquet(processed / "movies.parquet")
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = ["movieId,tagId,relevance"]
    for movie in (10, 30, 40):  # movie 20 has no genome
        rows += [f"{movie},{tag},{(movie * tag % 7) / 7:.3f}" for tag in range(1, 6)]
    (raw / "genome-scores.csv").write_text("\n".join(rows) + "\n")
    return processed, raw


def test_build_item_features_aligns_rows_with_item_idx_and_caches(tmp_path: Path) -> None:
    processed, raw = _processed(tmp_path)

    feats = build_item_features(processed, raw, pca_dim=2)

    assert feats.genres.shape == (3, len(GENRES))
    assert feats.genres[0, GENRES.index("Comedy")] == 1.0
    assert feats.genres[0, GENRES.index("Drama")] == 1.0
    assert feats.genres[1].sum() == 0.0
    assert feats.year_bucket.tolist()[1] == 0  # null year
    assert feats.genome_missing.tolist() == [0.0, 1.0, 0.0]
    assert feats.genome.shape == (3, 2)
    assert np.all(feats.genome[1] == 0.0)
    assert (processed / "item_features.npz").exists()

    again = build_item_features(processed, raw, pca_dim=2)  # served from the cache
    assert np.array_equal(again.genome, feats.genome)
    assert again.n_year_buckets == feats.n_year_buckets
