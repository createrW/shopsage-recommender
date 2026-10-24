"""Ablation (b) end to end on a synthetic Liang split, with fake dense runs (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from shopsage_rec.data.liang import load_liang
from shopsage_rec.retrieval.evaluate import evaluate_recall_paths


def _fake_run(runs: Path, run_id: str, n_items: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    out = runs / run_id
    out.mkdir(parents=True)
    emb = rng.standard_normal((n_items, 8)).astype(np.float32)
    np.save(out / "item_embeddings.npy", emb / np.linalg.norm(emb, axis=1, keepdims=True))
    np.savez(
        out / "user_tower.npz",
        hist_emb=rng.standard_normal((n_items, 4)).astype(np.float32),
        w1=rng.standard_normal((6, 5)).astype(np.float32),
        b1=np.zeros(6, np.float32),
        w2=rng.standard_normal((8, 6)).astype(np.float32),
        b2=np.zeros(8, np.float32),
    )


def test_recall_paths_writes_tables_deltas_and_milvus_parity(
    liang_dir: Path, tmp_path: Path
) -> None:
    n_items = load_liang(liang_dir).n_items
    runs = tmp_path / "runs"
    for seed, run_id in enumerate(("fake-s1", "fake-s2")):
        _fake_run(runs, run_id, n_items, seed)
    grid = ((1.2, 0.75, "log"), (1.2, 0.75, "binary"))

    result = evaluate_recall_paths(
        liang_dir,
        runs,
        tmp_path / "eval",
        dense_runs=("fake-s1", "fake-s2"),
        grid=grid,
        depth=150,
        candidates=(200, 50),
        parity_users=6,
        run_id="t",
    )

    summary = json.loads((result.out_dir / "summary.json").read_text())
    assert summary == result.summary
    assert set(summary["arms"]) == {
        "bm25",
        "dense:fake-s1",
        "dense:fake-s2",
        "hybrid:fake-s1",
        "hybrid:fake-s2",
    }
    assert summary["bm25"]["chosen"] in [list(g) for g in grid]
    assert len(summary["bm25"]["grid"]) == 2
    for arm in summary["arms"].values():
        assert (
            0.0
            <= arm["recall@50"]["ci_low"]
            <= arm["recall@50"]["mean"]
            <= arm["recall@50"]["ci_high"]
        )
    deltas = summary["paired"]
    assert {"hybrid - dense", "bm25 - dense"} <= set(deltas)
    assert set(deltas["hybrid - dense"]["per_seed"]) == {"fake-s1", "fake-s2"}
    assert "seed_mean" in deltas["hybrid - dense"]
    per_user = pl.read_parquet(result.out_dir / "per_user.parquet")
    assert per_user.height == summary["data"]["n_test_users"]
    assert "hybrid:fake-s1/recall@50" in per_user.columns
    parity = summary["milvus"]
    assert parity["users"] == 6
    for path in ("dense", "sparse", "hybrid"):
        assert parity[path]["identical"] == 1.0, path  # same scores, rank by rank, as offline
        assert parity[path]["overlap"] > 0.9, path
        p50, p95 = parity[path]["ms"]
        assert 0 < p50 <= p95
        e50, e95 = parity[path]["exact_ms"]
        assert 0 < e50 <= e95
    cand = summary["candidates"]
    assert 0.0 <= cand["dense@200"] <= cand["dense@200 + bm25@50"] <= 1.0
