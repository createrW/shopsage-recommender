"""The S2 sweep's job plan and summary, with a fake executor (no torch needed)."""

from __future__ import annotations

from typing import Any

import pytest

from shopsage_rec.models.sweep import SEEDS, Job, run_sweep, stage1_jobs, summarize


def _fake_summary(job: Job, score: float) -> dict[str, Any]:
    test = {m: {"mean": score} for m in ("recall@20", "recall@50", "ndcg@10", "ndcg@100", "hr@50")}
    return {
        "run_id": job.run_id,
        "arm": {"name": job.arm},
        "config": {"seed": job.seed, "tau": job.tau, "lr": job.lr},
        "best_epoch": job.epochs,
        "va_recall@50": score,
        "peak_memory_gb": 1.0,
        "wall_s": 10.0,
        "test": test,
    }


def test_run_sweep_tunes_then_trains_mines_and_trains_a3() -> None:
    calls: list[list[Job]] = []
    mined: list[str] = []

    def execute(jobs: list[Job]) -> list[dict[str, Any]]:
        calls.append(jobs)
        # the tuning run with tau 0.1, lr 0.001 scores best; later runs score by seed
        return [
            _fake_summary(
                j,
                0.5
                if (j.tau, j.lr) == (0.1, 1e-3) and j.run_id.startswith("tune")
                else 0.3 + j.seed / 1000,
            )
            for j in jobs
        ]

    summary = run_sweep(execute, mined.append, epochs=30, tune_epochs=10)

    assert [len(c) for c in calls] == [4, 6, 3]
    assert all(j.epochs == 10 and j.arm == "A1" for j in calls[0])
    assert {(j.tau, j.lr) for j in calls[1] + calls[2]} == {(0.1, 1e-3)}
    assert sorted(j.run_id for j in calls[1]) == [
        "A0-s42",
        "A1-s42",
        "A1-s43",
        "A1-s44",
        "A2-s42",
        "A4-s42",
    ]
    assert mined == ["A1-s42"]
    assert [j.mined_from for j in calls[2]] == ["A1-s42"] * 3
    assert [j.seed for j in calls[2]] == list(SEEDS)
    assert summary["chosen"] == {"tau": 0.1, "lr": 1e-3, "epochs": 30, "tune_epochs": 10}
    assert summary["arms"]["A1"]["seeds"] == 3
    assert summary["arms"]["A1"]["recall@50"]["mean"] == pytest.approx(0.343)
    assert summary["arms"]["A1"]["recall@50"]["sd"] == pytest.approx(0.001)
    assert summary["arms"]["A0"]["recall@50"]["sd"] is None


def test_stage1_has_three_seeds_for_the_reference_only() -> None:
    jobs = stage1_jobs(0.05, 1e-3, 30)

    seeds_per_arm = {
        arm: [j.seed for j in jobs if j.arm == arm] for arm in ("A0", "A1", "A2", "A4")
    }
    assert seeds_per_arm == {"A0": [42], "A1": [42, 43, 44], "A2": [42], "A4": [42]}


def test_summarize_keeps_every_run() -> None:
    job = Job("A1-s42", "A1", 42, 0.05, 1e-3, 30)

    out = summarize([], [_fake_summary(job, 0.4)])

    assert out["runs"]["A1-s42"]["recall@50"] == 0.4
    assert out["tuning"] == {}
