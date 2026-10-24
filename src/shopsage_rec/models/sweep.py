"""The S2 training sweep, independent of where the jobs run (Modal in parallel, or one box).

1. a short τ / learning-rate check on A1 (4 runs), the best by V_A Recall@50 is kept;
2. stage 1: A0, A1 x 3 seeds, A2, A4;
3. mining: ranks 101-500 per training user under A1 seed 42;
4. stage 2: A3 x 3 seeds.

``execute`` runs a list of jobs and returns their summaries (``train_two_tower``'s), in order;
``mine`` mines from a run id. The summary keeps each run's test means and, per arm, the mean and
sample standard deviation over seeds. Paired comparisons are made from the per-user files (S3).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

SEEDS = (42, 43, 44)
TUNING_GRID = ((0.05, 1e-3), (0.05, 3e-3), (0.1, 1e-3), (0.1, 3e-3))
REPORTED = ("recall@20", "recall@50", "ndcg@10", "ndcg@100", "hr@50")


@dataclass(frozen=True)
class Job:
    run_id: str
    arm: str
    seed: int
    tau: float
    lr: float
    epochs: int
    mined_from: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tuning_jobs(epochs: int) -> list[Job]:
    return [
        Job(f"tune-A1-tau{tau:g}-lr{lr:g}", "A1", SEEDS[0], tau, lr, epochs)
        for tau, lr in TUNING_GRID
    ]


def choose(results: Sequence[dict[str, Any]]) -> tuple[float, float]:
    """τ and learning rate of the tuning run with the best V_A Recall@50 (first wins a tie)."""
    best = max(results, key=lambda r: r["va_recall@50"])
    return float(best["config"]["tau"]), float(best["config"]["lr"])


def stage1_jobs(tau: float, lr: float, epochs: int) -> list[Job]:
    jobs = [Job(f"A1-s{seed}", "A1", seed, tau, lr, epochs) for seed in SEEDS]
    jobs += [
        Job(f"{arm}-s{SEEDS[0]}", arm, SEEDS[0], tau, lr, epochs) for arm in ("A0", "A2", "A4")
    ]
    return jobs


def stage2_jobs(tau: float, lr: float, epochs: int, mined_from: str) -> list[Job]:
    return [Job(f"A3-s{seed}", "A3", seed, tau, lr, epochs, mined_from) for seed in SEEDS]


def summarize(tuning: Sequence[dict[str, Any]], runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    per_run = {
        r["run_id"]: {
            "arm": r["arm"]["name"],
            "seed": r["config"]["seed"],
            "best_epoch": r["best_epoch"],
            "va_recall@50": r["va_recall@50"],
            "peak_memory_gb": r["peak_memory_gb"],
            "wall_s": r["wall_s"],
            **{m: r["test"][m]["mean"] for m in REPORTED},
        }
        for r in runs
    }
    arms: dict[str, dict[str, Any]] = {}
    for arm in sorted({v["arm"] for v in per_run.values()}):
        rows = [v for v in per_run.values() if v["arm"] == arm]
        arms[arm] = {"seeds": len(rows)}
        for m in REPORTED:
            values = [row[m] for row in rows]
            arms[arm][m] = {
                "mean": statistics.fmean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else None,
            }
    return {
        "tuning": {r["run_id"]: r["va_recall@50"] for r in tuning},
        "runs": per_run,
        "arms": arms,
    }


def run_sweep(
    execute: Callable[[list[Job]], list[dict[str, Any]]],
    mine: Callable[[str], None],
    *,
    epochs: int = 30,
    tune_epochs: int = 10,
) -> dict[str, Any]:
    tuning = execute(tuning_jobs(tune_epochs))
    tau, lr = choose(tuning)
    stage1 = execute(stage1_jobs(tau, lr, epochs))
    mine(f"A1-s{SEEDS[0]}")
    stage2 = execute(stage2_jobs(tau, lr, epochs, f"A1-s{SEEDS[0]}"))
    summary = summarize(tuning, [*stage1, *stage2])
    summary["chosen"] = {"tau": tau, "lr": lr, "epochs": epochs, "tune_epochs": tune_epochs}
    return summary
