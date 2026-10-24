"""Typed configuration objects loaded from ``configs/**/*.yaml``.

Only the data-pipeline config exists today; training / ranking configs follow the same
pattern (a pydantic model per YAML file, validated eagerly, snapshotted into artifacts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitConfig(BaseModel):
    """How held-out interactions are chosen.

    ``leave_one_out`` (the Day-1 split) holds out each user's two most recent positives;
    ``liang`` is the strong-generalisation split of Liang et al. (2018), the headline protocol
    (``data/liang.py``). ``n_heldout_users`` and ``test_prop`` apply to ``liang`` only.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["leave_one_out", "liang"] = "leave_one_out"
    seed: int = Field(
        42,
        ge=0,
        lt=2**64,
        description="leave_one_out: tie-break hash seed (uint64); liang: RandomState seed.",
    )
    n_heldout_users: int = Field(
        10_000, ge=1, description="liang: validation users, and as many test users."
    )
    test_prop: float = Field(
        0.2, gt=0.0, lt=1.0, description="liang: share of a held-out user's items drawn as targets."
    )

    @model_validator(mode="after")
    def _liang_seed_fits_randomstate(self) -> Self:
        if self.strategy == "liang" and self.seed >= 2**32:
            raise ValueError("the liang split seeds numpy's legacy RandomState: seed < 2**32")
        return self


class DataConfig(BaseModel):
    """Everything needed to turn the raw MovieLens-20M CSVs into training artifacts."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = "movielens-20m"
    raw_dir: Path = Field(description="Directory holding ratings.csv / movies.csv / tags.csv.")
    processed_dir: Path = Field(description="Where parquet artifacts are written.")
    min_rating: float = Field(
        4.0, ge=0.5, le=5.0, description="rating >= min_rating is a positive."
    )
    min_item_interactions: int = Field(5, ge=1)
    min_user_interactions: int = Field(
        5,
        ge=3,
        description="Leave-one-out needs test + val + at least one train interaction; "
        "the Liang split uses 5.",
    )
    split: SplitConfig = SplitConfig()  # pydantic deep-copies defaults per instance


def find_repo_root(start: Path) -> Path | None:
    """Walk up from ``start`` until a directory containing ``pyproject.toml`` is found."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _anchor(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def load_data_config(path: str | Path, root: str | Path | None = None) -> DataConfig:
    """Load and validate a data config; relative paths are anchored at ``root``.

    ``root`` is resolved to an absolute path. It defaults to the first ancestor of the config
    file that contains a ``pyproject.toml``, so outputs land in the repo whatever the working
    directory (e.g. ``shopsage prepare --config ../configs/data/x.yaml`` from ``scripts/``),
    falling back to the current working directory.
    """
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = DataConfig.model_validate(raw)
    base = (
        Path(root).resolve()
        if root is not None
        else (find_repo_root(config_path.resolve()) or Path.cwd())
    )
    return cfg.model_copy(
        update={
            "raw_dir": _anchor(cfg.raw_dir, base),
            "processed_dir": _anchor(cfg.processed_dir, base),
        }
    )
