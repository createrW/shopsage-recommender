"""The S4 ranker: candidates, features (and their leak canary), and the trained LambdaRank model."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from shopsage_rec.data.liang import load_liang
from shopsage_rec.models.baselines import EASE
from shopsage_rec.ranking.candidates import union_candidates
from shopsage_rec.ranking.features import (
    FEATURES,
    RankerContext,
    build_stage,
    item_stats,
    labels_for,
    prune_neighbours,
)
from tests.conftest import write_fake_run


def test_prune_neighbours_keeps_the_largest_weights_of_each_column() -> None:
    rng = np.random.default_rng(0)
    b = rng.standard_normal((30, 30)).astype(np.float32)
    np.fill_diagonal(b, 0.0)

    pruned = prune_neighbours(b, 5)

    assert isinstance(pruned, sp.csc_matrix)
    assert pruned.shape == b.shape
    for j in range(30):
        col = pruned[:, j].toarray().ravel()
        kept = np.flatnonzero(col)
        assert kept.size == 5
        assert set(kept.tolist()) == set(np.argsort(-b[:, j], kind="stable")[:5].tolist())
        np.testing.assert_array_equal(col[kept], b[kept, j])


def test_union_candidates_merge_both_lists_and_rank_them() -> None:
    dense = np.array([[0.9, 0.1, 0.5, 0.3, 0.7, 0.2]], dtype=np.float32)
    bm25 = np.array([[0.0, 2.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    exclude = sp.csr_matrix(np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float32))

    rows, items, dense_rank, bm25_rank = union_candidates(dense, bm25, exclude, depth=(2, 3))

    # dense top-2 without item 0: 4, 2; BM25 positive top-3: 1, 3
    assert rows.tolist() == [0, 0, 0, 0]
    assert items.tolist() == [1, 2, 3, 4]
    assert dense_rank.tolist() == [3, 2, 3, 1]  # depth + 1 when outside the dense list
    assert bm25_rank.tolist() == [1, 4, 2, 4]


def test_item_stats_come_from_training_users_only(liang_dir: Path) -> None:
    data = load_liang(liang_dir)

    stats = item_stats(liang_dir, data)

    counts = np.asarray(data.train.sum(axis=0)).ravel()
    np.testing.assert_allclose(stats.log_pop, np.log1p(counts), rtol=1e-6)
    users = pl.read_parquet(liang_dir / "mappings" / "user_map.parquet")
    items = pl.read_parquet(liang_dir / "mappings" / "item_map.parquet")
    expected = (
        pl.read_parquet(liang_dir / "ratings.parquet")
        .join(users.filter(pl.col("role") == "train"), on="userId")
        .join(items, on="movieId")
        .group_by("item_idx")
        .agg(pl.col("rating").mean())
        .sort("item_idx")
    )
    got = stats.mean_rating[expected["item_idx"].to_numpy()]
    np.testing.assert_allclose(got, expected["rating"].to_numpy(), rtol=1e-5)


def _context(liang_dir: Path, tmp_path: Path) -> tuple[RankerContext, object]:
    data = load_liang(liang_dir)
    run = write_fake_run(tmp_path / "runs", "fake", data.n_items, 0)
    ease = EASE.fit(data.train, 50.0)
    ctx = RankerContext.load(liang_dir, data, run, ease_weights=ease.weights, neighbours=20)
    return ctx, data


def test_features_never_see_the_targets(liang_dir: Path, tmp_path: Path) -> None:
    ctx, data = _context(liang_dir, tmp_path)
    shuffled = sp.csr_matrix(data.test_targets[::-1].toarray())  # same users, other targets
    other = dataclasses.replace(data, test_targets=shuffled)

    a = build_stage(data.test_fold_in, ctx, depth=(100, 30))
    b = build_stage(other.test_fold_in, ctx, depth=(100, 30))

    np.testing.assert_array_equal(a.items, b.items)
    np.testing.assert_array_equal(a.features, b.features)
    assert not np.array_equal(labels_for(a, data.test_targets), labels_for(b, shuffled))


def test_stage_rows_are_candidates_with_finite_core_features(
    liang_dir: Path, tmp_path: Path
) -> None:
    ctx, data = _context(liang_dir, tmp_path)

    stage = build_stage(data.test_fold_in, ctx, depth=(100, 30))

    assert stage.features.shape == (stage.items.size, len(FEATURES))
    assert stage.offsets[0] == 0
    assert stage.offsets[-1] == stage.items.size
    for u in range(data.test_fold_in.shape[0]):
        items = stage.items[stage.offsets[u] : stage.offsets[u + 1]]
        assert np.unique(items).size == items.size >= 100
        assert not set(items.tolist()) & set(data.test_fold_in[u].indices.tolist())
    core = [FEATURES.index(n) for n in ("dense_score", "dense_rank", "bm25_score", "ease_score")]
    assert np.isfinite(stage.features[:, core]).all()


def test_rank_trains_tests_and_times_the_serving_path(liang_dir: Path, tmp_path: Path) -> None:
    from shopsage_rec.ranking.train import train_rankers

    data = load_liang(liang_dir)
    runs = tmp_path / "runs"
    write_fake_run(runs, "fake", data.n_items, 0)

    result = train_rankers(
        liang_dir,
        runs,
        tmp_path / "rank",
        dense_run="fake",
        seeds=(1, 2),
        depth=(100, 30),
        ease_lambda=50.0,
        neighbours=20,
        # one thread: this process also loads torch's and faiss's libomp, and on macOS a second
        # OpenMP runtime with its own thread pool segfaults (the `shopsage rank` process has one)
        params={"min_data_in_leaf": 5, "num_leaves": 7, "num_threads": 1},
        max_rounds=20,
        latency_users=4,
        run_id="t",
    )

    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary == result.summary
    assert {"ease", "dense", "ranker", "ranker-no-ease"} <= set(summary["arms"])
    assert summary["arms"]["ranker"]["seeds"] == 2
    for name in (
        "ranker - ease",
        "ranker - dense",
        "ranker-no-ease - ease",
        "ranker - ranker-no-ease",
    ):
        block = summary["paired"][name]
        assert {"recall@50", "ndcg@10"} <= set(block)
        assert (
            block["recall@50"]["ci_low"]
            <= block["recall@50"]["mean"]
            <= block["recall@50"]["ci_high"]
        )
    assert (result.out_dir / "models" / "ranker-s1.txt").is_file()
    assert set(summary["importance"]["ranker"]) == set(FEATURES)
    assert "ease_score" not in summary["importance"]["ranker-no-ease"]
    lat = summary["latency"]
    assert lat["users"] == 4
    for part in ("retrieve_and_features", "predict", "total"):
        p50, p95 = lat[part]
        assert 0 < p50 <= p95
    per_user = pl.read_parquet(result.out_dir / "per_user.parquet")
    assert per_user.height == data.test_users.size
    assert summary["train"]["groups_without_positive"] >= 0


@pytest.mark.parametrize("bad", [(0, 30), (100, 0)])
def test_depth_must_be_positive(liang_dir: Path, tmp_path: Path, bad: tuple[int, int]) -> None:
    ctx, data = _context(liang_dir, tmp_path)
    with pytest.raises(ValueError, match="depth"):
        build_stage(data.test_fold_in, ctx, depth=bad)
