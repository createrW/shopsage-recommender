"""``shopsage`` command-line interface (Typer).

Today: ``prepare`` and ``stats``. Planned: ``evaluate`` (Day 4; the metric functions and the
MostPop / EASE baselines land on Day 2), ``train`` (Day 3), ``build-index`` (Day 4), ``rank``
(Day 6), ``expand`` (Day 8), ``demo`` (Day 9).

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

DEFAULT_DATA_CONFIG = Path("configs/data/movielens_20m.yaml")
DEFAULT_PROCESSED_DIR = Path("data/processed")

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
    """Raw MovieLens CSVs -> parquet artifacts -> leave-one-out train/val/test splits."""
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


if __name__ == "__main__":  # pragma: no cover
    app()
