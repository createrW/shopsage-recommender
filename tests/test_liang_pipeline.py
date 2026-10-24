"""`prepare` under the Liang protocol: artifacts, config, CLI and the sparse-matrix loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from shopsage_rec.cli import app
from shopsage_rec.config import DataConfig, SplitConfig, load_data_config
from shopsage_rec.data.liang import liang_split, load_liang
from shopsage_rec.data.pipeline import run_prepare

REPO = Path(__file__).resolve().parents[1]
SPLIT_FILES = ("train", "val_tr", "val_te", "test_tr", "test_te")


def _config(tiny_raw_dir: Path, processed: Path, **split: object) -> DataConfig:
    return DataConfig(
        dataset="movielens-tiny",
        raw_dir=tiny_raw_dir,
        processed_dir=processed,
        min_rating=4.0,
        min_item_interactions=1,
        min_user_interactions=5,
        split=SplitConfig(strategy="liang", seed=98765, n_heldout_users=1, **split),
    )


def test_run_prepare_liang_writes_the_split_contract(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "liang"

    report = run_prepare(_config(tiny_raw_dir, processed, test_prop=0.2))

    for name in SPLIT_FILES:
        frame = pl.read_parquet(processed / "splits" / f"{name}.parquet")
        assert frame.columns == ["user_idx", "item_idx", "rating", "timestamp"], name
        assert frame["user_idx"].dtype == pl.Int32
        assert frame["item_idx"].dtype == pl.Int32
    for name in ("ratings", "movies", "tags"):
        assert (processed / f"{name}.parquet").exists()
    user_map = pl.read_parquet(processed / "mappings" / "user_map.parquet")
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet")
    # users 1-5 have >= 5 positives; user 6 has 2 and is filtered out
    assert user_map["userId"].to_list() == [1, 2, 3, 4, 5]
    assert user_map["user_idx"].to_list() == [0, 1, 2, 3, 4]
    assert sorted(user_map["role"].value_counts()["role"].to_list()) == ["test", "train", "val"]
    assert user_map.filter(pl.col("role") == "train").height == 3
    assert item_map["movieId"].to_list() == sorted(item_map["movieId"].to_list())

    meta = json.loads((processed / "splits" / "meta.json").read_text())
    assert meta["strategy"] == "liang"
    assert meta["n_users"] == 5
    assert meta["n_items"] == item_map.height == report.n_items
    assert meta["config"]["split"]["n_heldout_users"] == 1
    assert len(meta["split_sha256"]) == 64
    assert "liang" in report.summary()


def test_run_prepare_liang_matches_the_split_function(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "liang"
    run_prepare(_config(tiny_raw_dir, processed))
    user_map = pl.read_parquet(processed / "mappings" / "user_map.parquet")
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet")
    written = {
        name: pl.read_parquet(processed / "splits" / f"{name}.parquet")
        .join(user_map, on="user_idx")
        .join(item_map, on="item_idx")
        for name in SPLIT_FILES
    }

    ratings = pl.read_parquet(processed / "ratings.parquet")
    positives = ratings.filter(pl.col("rating") >= 4.0).filter(pl.len().over("userId") >= 5)
    expected = liang_split(positives, seed=98765, n_heldout_users=1).frames()

    for name in SPLIT_FILES:
        got = written[name].select("userId", "movieId").sort("userId", "movieId")
        assert got.equals(expected[name].select("userId", "movieId")), name


def test_cli_prepare_runs_the_liang_config(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "liang"
    config_path = tmp_path / "liang.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tiny_raw_dir),
                "processed_dir": str(processed),
                "min_item_interactions": 1,
                "min_user_interactions": 5,
                "split": {"strategy": "liang", "seed": 98765, "n_heldout_users": 1},
            }
        )
    )

    result = CliRunner().invoke(app, ["prepare", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (processed / "splits" / "test_te.parquet").exists()
    stats = CliRunner().invoke(app, ["stats", "--processed-dir", str(processed)])
    assert stats.exit_code == 0, stats.output
    assert '"strategy": "liang"' in stats.output


def test_the_shipped_liang_config_matches_the_published_preprocessing() -> None:
    cfg = load_data_config(REPO / "configs" / "data" / "movielens_20m_liang.yaml")

    assert cfg.split.strategy == "liang"
    assert cfg.split.seed == 98765
    assert cfg.split.n_heldout_users == 10_000
    assert cfg.split.test_prop == 0.2
    assert cfg.min_rating == 4.0
    assert cfg.min_item_interactions == 1  # no item filter
    assert cfg.min_user_interactions == 5
    assert cfg.processed_dir == REPO / "data" / "processed" / "liang"


@pytest.mark.parametrize(
    "split",
    [
        {"strategy": "liang", "n_heldout_users": 0},
        {"strategy": "liang", "test_prop": 0.0},
        {"strategy": "liang", "test_prop": 1.0},
        {"strategy": "liang", "seed": 2**32},  # numpy's legacy RandomState takes < 2**32
        {"strategy": "kfold"},
    ],
)
def test_split_config_rejects_invalid_liang_settings(split: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate(split)


def test_load_liang_builds_aligned_sparse_matrices(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "liang"
    run_prepare(_config(tiny_raw_dir, processed, test_prop=0.2))

    data = load_liang(processed)

    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet")
    train = pl.read_parquet(processed / "splits" / "train.parquet")
    assert data.n_items == item_map.height
    assert data.train.shape == (train["user_idx"].n_unique(), data.n_items)
    assert data.train.nnz == train.height
    for stage in ("val", "test"):
        users, fold_in, targets = data.stage(stage)
        te = pl.read_parquet(processed / "splits" / f"{stage}_te.parquet")
        assert users.tolist() == sorted(te["user_idx"].unique().to_list())  # users with targets
        assert fold_in.shape == targets.shape == (users.size, data.n_items)
        assert targets.nnz == te.height
        assert fold_in.multiply(targets).nnz == 0  # disjoint
        assert np.all(np.asarray(targets.sum(axis=1)).ravel() >= 1)
