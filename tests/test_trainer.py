"""End-to-end Two-Tower training on a synthetic Liang split, on CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from shopsage_rec.data.liang import load_liang  # noqa: E402
from shopsage_rec.models.mining import MINED_FILE, mine_hard_negatives  # noqa: E402
from shopsage_rec.models.trainer import TrainConfig, train_two_tower  # noqa: E402


def _cfg(arm: str, **kw: object) -> TrainConfig:
    base: dict[str, object] = {
        "arm": arm,
        "epochs": 2,
        "batch_size": 256,
        "eval_every": 1,
        "dim": 16,
        "id_dim": 8,
        "hist_dim": 8,
        "hidden": 32,
        "num_uniform": 64,
        "mined_k": 4,
        "device": "cpu",
        "wandb": False,
    }
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def test_train_exports_a_tested_model(liang_dir: Path, tmp_path: Path) -> None:
    runs = tmp_path / "runs"

    summary = train_two_tower(liang_dir, runs, _cfg("A1"))

    run = runs / "A1-s42"
    emb = np.load(run / "item_embeddings.npy")
    assert emb.shape[0] == load_liang(liang_dir).n_items
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    assert {"hist_emb", "w1", "b1", "w2", "b2"} <= set(np.load(run / "user_tower.npz").files)
    assert json.loads((run / "metrics.json").read_text()) == summary
    assert summary["best_epoch"] in (1, 2)
    assert len(summary["curve"]) == 2
    per_user = pl.read_parquet(run / "per_user.parquet")
    assert per_user.height == summary["data"]["n_test_users"]
    stats = summary["test"]["recall@50"]
    assert 0.0 <= stats["ci_low"] <= stats["mean"] <= stats["ci_high"] <= 1.0
    assert summary["config"]["arm"] == "A1"
    assert _plain(summary)  # Modal unpickles it on a machine without torch


def _plain(value: object) -> bool:
    if isinstance(value, dict):
        return all(type(k) is str and _plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_plain(v) for v in value)
    return value is None or type(value) in (str, int, float, bool)


@pytest.mark.parametrize("arm", ["A0", "A2", "A4"])
def test_every_context_arm_trains(liang_dir: Path, tmp_path: Path, arm: str) -> None:
    summary = train_two_tower(liang_dir, tmp_path, _cfg(arm, epochs=1))

    assert np.isfinite(summary["curve"][0]["loss"])
    assert summary["arm"]["name"] == arm


def test_mining_then_a3(liang_dir: Path, tmp_path: Path) -> None:
    train_two_tower(liang_dir, tmp_path, _cfg("A1", epochs=1))

    path = mine_hard_negatives(liang_dir, tmp_path / "A1-s42", skip=10, depth=40)

    data = load_liang(liang_dir)
    pool = np.load(path)
    assert path.name == MINED_FILE
    assert pool.shape == (data.train.shape[0], 30)
    for row in range(0, data.train.shape[0], 17):  # no training positive of the user is mined
        assert not set(pool[row].tolist()) & set(data.train[row].indices.tolist())
    summary = train_two_tower(liang_dir, tmp_path, _cfg("A3", epochs=1), mined_from=path.parent)
    assert np.isfinite(summary["curve"][0]["loss"])


def test_a3_without_a_pool_is_rejected(liang_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mined_from"):
        train_two_tower(liang_dir, tmp_path, _cfg("A3"))
    with pytest.raises(ValueError, match="unknown arm"):
        TrainConfig(arm="A9")
