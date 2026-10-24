"""End-to-end `prepare` on a tiny raw directory, through the Python API and the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from shopsage_rec import __version__
from shopsage_rec.cli import app
from shopsage_rec.config import DataConfig, SplitConfig, load_data_config
from shopsage_rec.data.pipeline import run_prepare


def _config(tiny_raw_dir: Path, processed: Path) -> DataConfig:
    return DataConfig(
        dataset="movielens-tiny",
        raw_dir=tiny_raw_dir,
        processed_dir=processed,
        min_rating=4.0,
        min_item_interactions=2,
        min_user_interactions=3,
        split=SplitConfig(strategy="leave_one_out", seed=42),
    )


def test_run_prepare_writes_every_artifact_in_the_contract(
    tiny_raw_dir: Path, tmp_path: Path
) -> None:
    processed = tmp_path / "processed"

    report = run_prepare(_config(tiny_raw_dir, processed))

    for rel in (
        "ratings.parquet",
        "movies.parquet",
        "tags.parquet",
        "mappings/user_map.parquet",
        "mappings/item_map.parquet",
        "splits/train.parquet",
        "splits/val.parquet",
        "splits/test.parquet",
        "splits/meta.json",
    ):
        assert (processed / rel).exists(), rel

    ratings = pl.read_parquet(processed / "ratings.parquet")
    assert ratings.height == 30  # raw ratings are kept in full, low ratings included
    movies = pl.read_parquet(processed / "movies.parquet")
    assert set(movies.columns) == {"movieId", "title", "title_clean", "year", "genres"}

    meta = json.loads((processed / "splits" / "meta.json").read_text())
    assert meta["n_users"] == report.n_users
    assert meta["n_test"] == meta["n_users"] == meta["n_val"]
    assert meta["config"]["min_rating"] == 4.0
    assert meta["config"]["split"]["seed"] == 42
    assert 0 < meta["density"] < 1


def test_run_prepare_applies_filters_before_splitting(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"

    # min_item_interactions=2, min_user_interactions=4: movies 50 and 70 have one positive each
    # (dropped by the item filter), movie 90 is rated once; user 6 has two positives and ends
    # with one after the item filter, so the user filter drops it. With the thresholds swapped
    # movie 80 (4 positives) would also disappear; with the item filter off, 50 and 70 survive.
    run_prepare(_config(tiny_raw_dir, processed).model_copy(update={"min_user_interactions": 4}))

    user_map = pl.read_parquet(processed / "mappings/user_map.parquet")
    item_map = pl.read_parquet(processed / "mappings/item_map.parquet")
    assert user_map["userId"].to_list() == [1, 2, 3, 4, 5]
    assert item_map["movieId"].to_list() == [10, 20, 40, 60, 80]
    test = pl.read_parquet(processed / "splits/test.parquet")
    assert test.height == user_map.height


def test_cli_prepare_runs_from_yaml_config(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "movielens-tiny",
                "raw_dir": str(tiny_raw_dir),
                "processed_dir": str(processed),
                "min_rating": 4.0,
                "min_item_interactions": 2,
                "min_user_interactions": 3,
                "split": {"strategy": "leave_one_out", "seed": 3},
            }
        )
    )

    result = CliRunner().invoke(app, ["prepare", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert (processed / "splits" / "test.parquet").exists()
    assert "users" in result.output


def test_cli_stats_summarises_processed_artifacts(tiny_raw_dir: Path, tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    run_prepare(_config(tiny_raw_dir, processed))

    result = CliRunner().invoke(app, ["stats", "--processed-dir", str(processed)])

    assert result.exit_code == 0, result.output
    assert "n_users" in result.output


def test_cli_version_prints_the_package_version_and_exits() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


def test_cli_prepare_without_raw_data_fails_cleanly(tmp_path: Path) -> None:
    empty_raw = tmp_path / "ml-20m"
    empty_raw.mkdir()
    processed = tmp_path / "processed"
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump({"raw_dir": str(empty_raw), "processed_dir": str(processed)})
    )

    result = CliRunner().invoke(app, ["prepare", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "download_data.sh" in result.output  # the loader's one-line hint, not a traceback
    assert "Traceback" not in result.output
    assert not processed.exists()  # nothing is created before the raw files are read


def test_cli_prepare_rejects_a_missing_root(tiny_raw_dir: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump({"raw_dir": str(tiny_raw_dir), "processed_dir": "data/processed"})
    )

    result = CliRunner().invoke(
        app, ["prepare", "--config", str(config_path), "--root", str(tmp_path / "nowhere")]
    )

    assert result.exit_code == 2, result.output
    assert not (tmp_path / "nowhere").exists()


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'fake'\n")
    return repo


def test_cli_defaults_resolve_against_the_repo_root_from_a_subdirectory(
    tiny_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_repo(tmp_path)
    config_path = repo / "configs" / "data" / "movielens_20m_liang.yaml"  # the default config
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tiny_raw_dir),
                "processed_dir": "data/processed/liang",
                "min_item_interactions": 1,
                "min_user_interactions": 5,
                "split": {"strategy": "liang", "seed": 98765, "n_heldout_users": 1},
            }
        )
    )
    monkeypatch.chdir(repo / "scripts")

    prepared = CliRunner().invoke(app, ["prepare"])
    assert prepared.exit_code == 0, prepared.output
    assert (repo / "data" / "processed" / "liang" / "splits" / "test_te.parquet").exists()

    stats = CliRunner().invoke(app, ["stats"])
    assert stats.exit_code == 0, stats.output
    assert "n_users" in stats.output


def test_load_data_config_resolves_relative_paths_against_repo_root(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "data" / "x.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "movielens-20m",
                "raw_dir": "data/raw/ml-20m",
                "processed_dir": "data/processed",
                "min_rating": 4.0,
                "min_item_interactions": 5,
                "min_user_interactions": 5,
                "split": {"strategy": "leave_one_out", "seed": 42},
            }
        )
    )

    cfg = load_data_config(config_path, root=tmp_path)

    assert cfg.raw_dir == tmp_path / "data" / "raw" / "ml-20m"
    assert cfg.processed_dir == tmp_path / "data" / "processed"
    assert cfg.split.strategy == "leave_one_out"


def _write_relative_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"raw_dir": "data/raw/ml-20m", "processed_dir": "data/processed"})
    )


def test_load_data_config_infers_the_repo_root_from_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    config_path = tmp_path / "configs" / "data" / "x.yaml"
    _write_relative_config(config_path)
    monkeypatch.chdir(config_path.parent)  # the working directory is not the repo root

    cfg = load_data_config(config_path.name)

    assert cfg.raw_dir == tmp_path / "data" / "raw" / "ml-20m"
    assert cfg.processed_dir == tmp_path / "data" / "processed"


def test_load_data_config_resolves_a_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "x.yaml"
    _write_relative_config(config_path)
    monkeypatch.chdir(tmp_path)

    cfg = load_data_config(config_path, root=".")

    assert cfg.raw_dir.is_absolute()
    assert cfg.raw_dir == tmp_path / "data" / "raw" / "ml-20m"


def test_split_seed_must_fit_in_64_bits() -> None:
    assert SplitConfig(seed=2**64 - 1).seed == 2**64 - 1
    with pytest.raises(ValidationError):
        SplitConfig(seed=2**64)


def test_meta_records_tie_break_and_digest_and_splits_have_contract_columns(
    tiny_raw_dir: Path, tmp_path: Path
) -> None:
    processed = tmp_path / "processed"

    run_prepare(_config(tiny_raw_dir, processed))

    meta = json.loads((processed / "splits" / "meta.json").read_text())
    # The key is hashed over the raw (userId, movieId) before remapping; hashing the contiguous
    # indices instead flips user 5's 60/80 tie and changes the digest.
    assert meta["tie_break"].endswith("key from precomputed column 'tie_key'")
    assert (
        meta["split_sha256"] == "12da34c28d16573edef8be7580a4db5b076d16f48b22f9fdea4b24e3b58ab40a"
    )
    assert meta["users_with_any_boundary_tie"] >= meta["users_with_boundary_tie"]
    for name in ("train", "val", "test"):
        assert pl.read_parquet(processed / "splits" / f"{name}.parquet").columns == [
            "user_idx",
            "item_idx",
            "rating",
            "timestamp",
        ]
