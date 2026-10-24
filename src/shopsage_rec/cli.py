"""``shopsage`` command-line interface (Typer).

Today: ``prepare``, ``stats`` and ``evaluate`` (MostPop and EASE on the Liang split). Planned,
by step of docs/plan.md: ``train`` (S2), ``rank`` (S4), ``expand`` (S5), ``demo`` (S6).

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


if __name__ == "__main__":  # pragma: no cover
    app()
