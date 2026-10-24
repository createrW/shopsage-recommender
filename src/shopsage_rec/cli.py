"""``shopsage`` command-line interface (Typer).

Today: ``prepare``, ``stats``, ``evaluate`` (MostPop and EASE on the Liang split), the
Two-Tower commands ``train``, ``mine`` and ``sweep`` (they need the ``train`` extra; torch is
imported only inside them), and ``recall`` (BM25, dense and hybrid recall paths, with the Milvus
Lite parity check). Planned, by step of docs/plan.md: ``rank`` (S4), ``expand`` (S5).

Default paths (``--config``, ``--processed-dir``) are resolved against the repository root,
found by walking up from the working directory to the nearest ``pyproject.toml``, so the
commands work from any subdirectory of a checkout.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from shopsage_rec import __version__
from shopsage_rec.config import find_repo_root, load_data_config
from shopsage_rec.data.pipeline import run_prepare
from shopsage_rec.evaluation.runner import (
    DEFAULT_LAMBDAS,
    MODELS,
    evaluate_baselines,
    format_summary,
)

# Defaults follow the headline protocol, the Liang split; the Day-1 leave-one-out split is built
# with --config configs/data/movielens_20m.yaml (into data/processed/).
DEFAULT_DATA_CONFIG = Path("configs/data/movielens_20m_liang.yaml")
DEFAULT_PROCESSED_DIR = Path("data/processed/liang")
DEFAULT_EVAL_DIR = Path("artifacts/eval")
DEFAULT_RUNS_DIR = Path("artifacts/runs")

app = typer.Typer(
    help="ShopSage Recommender: Two-Tower retrieval + LightGBM ranking on MovieLens-20M.",
    no_args_is_help=True,
    add_completion=False,
)


def _repo_root() -> Path:
    """The checkout that contains the working directory, or the working directory itself."""
    return find_repo_root(Path.cwd()) or Path.cwd()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the version and exit.",
    ),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


@app.command()
def prepare(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help=f"Data pipeline YAML config (default: {DEFAULT_DATA_CONFIG} under the repo root).",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Directory that anchors the config's relative paths (default: the repo root "
        "inferred from the config file's location).",
    ),
) -> None:
    """Raw MovieLens CSVs -> parquet artifacts -> the config's split (leave-one-out or Liang)."""
    config_path = config if config is not None else _repo_root() / DEFAULT_DATA_CONFIG
    if not config_path.is_file():
        raise typer.BadParameter(f"{config_path} does not exist", param_hint="--config")
    cfg = load_data_config(config_path, root=root)
    try:
        report = run_prepare(cfg)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(report.summary())


@app.command()
def stats(
    processed_dir: Path | None = typer.Option(
        None,
        "--processed-dir",
        "-p",
        help=f"Output dir of `shopsage prepare` (default: {DEFAULT_PROCESSED_DIR} under the "
        "repo root).",
    ),
) -> None:
    """Print the dataset statistics recorded in splits/meta.json."""
    processed = processed_dir if processed_dir is not None else _repo_root() / DEFAULT_PROCESSED_DIR
    meta_path = processed / "splits" / "meta.json"
    if not meta_path.is_file():
        raise typer.BadParameter(f"{meta_path} not found; run `shopsage prepare` first.")
    typer.echo(json.dumps(json.loads(meta_path.read_text(encoding="utf-8")), indent=2))


@app.command()
def evaluate(
    processed_dir: Path | None = typer.Option(
        None,
        "--processed-dir",
        "-p",
        help=f"A prepared Liang split (default: {DEFAULT_PROCESSED_DIR} under the repo root).",
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        "-o",
        help=f"Where <run-id>/ is written (default: {DEFAULT_EVAL_DIR} under the repo root).",
    ),
    models: str = typer.Option(
        ",".join(MODELS), "--models", help=f"Comma-separated, from: {', '.join(MODELS)}."
    ),
    ease_lambdas: str = typer.Option(
        ",".join(f"{lam:g}" for lam in DEFAULT_LAMBDAS),
        "--ease-lambdas",
        help="EASE L2 grid; the best on V_A by Recall@50 is used for test.",
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run directory name (default: local-YYYYMMDD-HHMM)."
    ),
) -> None:
    """MostPop and EASE on the Liang split: per-user metrics, bootstrap CIs, summary.json."""
    root = _repo_root()
    processed = processed_dir if processed_dir is not None else root / DEFAULT_PROCESSED_DIR
    out = out_dir if out_dir is not None else root / DEFAULT_EVAL_DIR
    names = tuple(name.strip() for name in models.split(",") if name.strip())
    unknown = [name for name in names if name not in MODELS]
    if unknown or not names:
        raise typer.BadParameter(
            f"unknown model(s): {', '.join(unknown) or '(none)'}; choose from {', '.join(MODELS)}",
            param_hint="--models",
        )
    try:
        lambdas = tuple(float(value) for value in ease_lambdas.split(",") if value.strip())
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--ease-lambdas") from exc
    if not (processed / "splits" / "meta.json").is_file():
        raise typer.BadParameter(
            f"{processed / 'splits' / 'meta.json'} not found; run "
            "`shopsage prepare` (configs/data/movielens_20m_liang.yaml) first",
            param_hint="--processed-dir",
        )
    try:
        result = evaluate_baselines(
            processed, out, models=names, ease_lambdas=lambdas, run_id=run_id
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(format_summary(result.summary))
    typer.echo(f"wrote {result.out_dir}")


def _dirs(processed_dir: Path | None, runs_dir: Path | None) -> tuple[Path, Path]:
    root = _repo_root()
    processed = processed_dir if processed_dir is not None else root / DEFAULT_PROCESSED_DIR
    runs = runs_dir if runs_dir is not None else root / DEFAULT_RUNS_DIR
    if not (processed / "splits" / "meta.json").is_file():
        raise typer.BadParameter(
            f"{processed / 'splits' / 'meta.json'} not found; run `shopsage prepare` first",
            param_hint="--processed-dir",
        )
    return processed, runs


@app.command()
def train(
    arm: str = typer.Option(
        "A1", "--arm", help="A0 (no logQ), A1, A2 (uniform), A3 (mined), A4 (full)."
    ),
    seed: int = typer.Option(42, "--seed"),
    epochs: int = typer.Option(30, "--epochs"),
    tau: float = typer.Option(0.05, "--tau", help="Softmax temperature."),
    lr: float = typer.Option(2e-3, "--lr"),
    batch_size: int = typer.Option(4096, "--batch-size"),
    user_frac: float = typer.Option(
        1.0, "--user-frac", help="Share of training users (smoke runs)."
    ),
    device: str = typer.Option("auto", "--device", help="auto, cuda, mps or cpu."),
    mined_from: str | None = typer.Option(
        None, "--mined-from", help="Run id holding mined negatives (A3)."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Default: <arm>-s<seed>."),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", "-p"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    no_wandb: bool = typer.Option(False, "--no-wandb", help="Do not log to Weights & Biases."),
) -> None:
    """Train one Two-Tower arm, keep its best V_A epoch, export it and test it."""
    from shopsage_rec.models.trainer import TrainConfig, train_two_tower

    processed, runs = _dirs(processed_dir, runs_dir)
    try:
        cfg = TrainConfig(
            arm=arm, seed=seed, epochs=epochs, tau=tau, lr=lr, batch_size=batch_size,
            user_frac=user_frac, device=device, wandb=not no_wandb,
        )  # fmt: skip
        summary = train_two_tower(
            processed, runs, cfg, run_id=run_id,
            mined_from=runs / mined_from if mined_from else None,
        )  # fmt: skip
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    test = summary["test"]
    typer.echo(
        f"{summary['run_id']}: best epoch {summary['best_epoch']}, "
        f"V_A recall@50 {summary['va_recall@50']:.4f}, test recall@50 "
        f"{test['recall@50']['mean']:.4f}, ndcg@10 {test['ndcg@10']['mean']:.4f}, "
        f"peak {summary['peak_memory_gb']:.2f} GB"
    )


@app.command()
def mine(
    run_id: str = typer.Option("A1-s42", "--run-id", help="The reference run to mine from."),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", "-p"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Write each training user's ranks 101-500 under a run (the A3 negatives)."""
    from shopsage_rec.models.mining import mine_hard_negatives

    processed, runs = _dirs(processed_dir, runs_dir)
    typer.echo(f"wrote {mine_hard_negatives(processed, runs / run_id)}")


@app.command()
def sweep(
    epochs: int = typer.Option(30, "--epochs"),
    tune_epochs: int = typer.Option(10, "--tune-epochs"),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", "-p"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    no_wandb: bool = typer.Option(False, "--no-wandb"),
) -> None:
    """The whole S2 sweep on this machine, one run after another (Modal runs it in parallel)."""
    from shopsage_rec.models.mining import mine_hard_negatives
    from shopsage_rec.models.sweep import Job, run_sweep
    from shopsage_rec.models.trainer import TrainConfig, train_two_tower

    processed, runs = _dirs(processed_dir, runs_dir)

    def execute(jobs: list[Job]) -> list[dict[str, object]]:
        out = []
        for job in jobs:
            cfg = TrainConfig(
                arm=job.arm, seed=job.seed, epochs=job.epochs, tau=job.tau, lr=job.lr,
                wandb=not no_wandb,
            )  # fmt: skip
            mined = runs / job.mined_from if job.mined_from else None
            out.append(train_two_tower(processed, runs, cfg, run_id=job.run_id, mined_from=mined))
        return out

    def mine_from(run_id: str) -> None:
        mine_hard_negatives(processed, runs / run_id)

    summary = run_sweep(execute, mine_from, epochs=epochs, tune_epochs=tune_epochs)
    (runs / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary["arms"], indent=2))
    typer.echo(f"wrote {runs / 'sweep_summary.json'}")


@app.command()
def recall(
    dense_runs: str = typer.Option(
        "A1-s42,A1-s43,A1-s44", "--dense-runs", help="Comma-separated run ids under --runs-dir."
    ),
    depth: int = typer.Option(200, "--depth", help="Per-path list length fused by RRF."),
    parity_users: int = typer.Option(1000, "--parity-users", help="Test users queried in Milvus."),
    run_id: str | None = typer.Option(None, "--run-id", help="Default: recall-YYYYMMDD-HHMM."),
    processed_dir: Path | None = typer.Option(None, "--processed-dir", "-p"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
    out_dir: Path | None = typer.Option(None, "--out-dir", "-o"),
) -> None:
    """Ablation (b): BM25, dense and RRF hybrid recall, plus the Milvus Lite parity check."""
    from shopsage_rec.retrieval.evaluate import evaluate_recall_paths, format_recall_summary

    processed, runs = _dirs(processed_dir, runs_dir)
    names = tuple(n.strip() for n in dense_runs.split(",") if n.strip())
    missing = [n for n in names if not (runs / n / "item_embeddings.npy").is_file()]
    if missing or not names:
        raise typer.BadParameter(
            f"no exported run for: {', '.join(missing) or '(none)'} under {runs}",
            param_hint="--dense-runs",
        )
    out = out_dir if out_dir is not None else _repo_root() / DEFAULT_EVAL_DIR
    result = evaluate_recall_paths(
        processed, runs, out, dense_runs=names, depth=depth, parity_users=parity_users,
        run_id=run_id,
    )  # fmt: skip
    typer.echo(format_recall_summary(result.summary))
    typer.echo(f"wrote {result.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    app()
