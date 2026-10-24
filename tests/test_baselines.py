"""MostPop and EASE baselines, and `shopsage evaluate` on a synthetic Liang split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp
from typer.testing import CliRunner

from shopsage_rec.cli import app
from shopsage_rec.evaluation.metrics import topk_masked
from shopsage_rec.evaluation.runner import evaluate_baselines
from shopsage_rec.models.baselines import EASE, MostPop, gram_matrix


def _binary(rows: int, cols: int, density: float, seed: int) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    dense = (rng.random((rows, cols)) < density).astype(np.float32)
    return sp.csr_matrix(dense)


def test_mostpop_ranks_unseen_items_by_training_popularity() -> None:
    train = sp.csr_matrix(
        np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=np.float32)
    )
    fold_in = sp.csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))

    model = MostPop().fit(train)

    assert model.popularity_.tolist() == [4.0, 2.0, 2.0, 1.0]
    assert topk_masked(model.score(fold_in), fold_in, 3).tolist() == [[1, 2, 3], [0, 1, 2]]


def test_gram_matrix_counts_co_occurrences() -> None:
    train = _binary(30, 8, 0.4, seed=0)

    gram = gram_matrix(train)

    assert gram.dtype == np.float32
    assert np.array_equal(gram, (train.T @ train).toarray())


def test_ease_matches_the_closed_form_in_float64() -> None:
    train = _binary(200, 25, 0.2, seed=1)
    lam = 10.0

    model = EASE.fit(train, lam=lam)

    x = train.toarray().astype(np.float64)
    g = x.T @ x + lam * np.eye(25)
    p = np.linalg.inv(g)
    b = p / -np.diag(p)
    np.fill_diagonal(b, 0.0)
    assert model.lam == lam
    assert model.weights.dtype == np.float32
    assert np.allclose(model.weights, b, atol=1e-5)
    assert np.all(np.diag(model.weights) == 0.0)
    fold_in = _binary(5, 25, 0.3, seed=2)
    assert np.allclose(model.score(fold_in), fold_in.toarray() @ b, atol=1e-5)


def test_ease_from_gram_leaves_the_gram_untouched() -> None:
    train = _binary(50, 10, 0.3, seed=3)
    gram = gram_matrix(train)
    before = gram.copy()

    a = EASE.from_gram(gram, lam=5.0)
    b = EASE.from_gram(gram, lam=50.0)

    assert np.array_equal(gram, before)
    assert not np.allclose(a.weights, b.weights)


def test_evaluate_baselines_writes_per_user_metrics_and_a_summary(
    liang_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "eval"

    result = evaluate_baselines(liang_dir, out, ease_lambdas=(1.0, 10.0), run_id="test-run")

    run_dir = out / "test-run"
    assert result.out_dir == run_dir
    per_user = pl.read_parquet(run_dir / "per_user.parquet")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary == result.summary
    test_users = pl.read_parquet(liang_dir / "splits" / "test_te.parquet")["user_idx"].unique()
    assert per_user["user_idx"].to_list() == sorted(test_users.to_list())
    assert summary["protocol"] == "Liang-SG-strict"
    assert (
        summary["data"]["split_sha256"]
        == json.loads((liang_dir / "splits" / "meta.json").read_text())["split_sha256"]
    )
    for model in ("mostpop", "ease"):
        for metric in ("recall@20", "recall@50", "ndcg@10", "ndcg@100", "hr@50"):
            column = per_user[f"{model}/{metric}"].to_numpy()
            assert np.all((column >= 0) & (column <= 1))
            stats = summary["models"][model]["test"][metric]
            assert stats["mean"] == pytest.approx(float(column.mean()))
            assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    ease = summary["models"]["ease"]
    assert ease["lambda"] in (1.0, 10.0)
    assert set(ease["selection"]["va"]) == {"1.0", "10.0"}
    assert ease["selection"]["va"][str(ease["lambda"])] == max(ease["selection"]["va"].values())
    assert "recall@50" in summary["paired"]["ease-mostpop"]


def test_evaluate_baselines_is_deterministic(liang_dir: Path, tmp_path: Path) -> None:
    a = evaluate_baselines(liang_dir, tmp_path / "a", ease_lambdas=(5.0,), run_id="r")
    b = evaluate_baselines(liang_dir, tmp_path / "b", ease_lambdas=(5.0,), run_id="r")

    for key in ("models", "paired"):
        assert a.summary[key] == b.summary[key]


def test_cli_evaluate_runs_the_baselines(liang_dir: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--processed-dir",
            str(liang_dir),
            "--out-dir",
            str(tmp_path / "eval"),
            "--ease-lambdas",
            "2,20",
            "--run-id",
            "cli-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "eval" / "cli-run" / "summary.json").exists()
    assert "recall@50" in result.output
    assert "ease" in result.output


def test_cli_evaluate_rejects_an_unknown_model(liang_dir: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["evaluate", "-p", str(liang_dir), "--out-dir", str(tmp_path), "--models", "svd"]
    )

    assert result.exit_code != 0
    assert "svd" in result.output
