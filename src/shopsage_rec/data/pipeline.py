"""``shopsage prepare``: raw CSVs -> parquet artifacts -> splits + meta.json.

Two split strategies, chosen by ``split.strategy`` in the data config: ``leave_one_out``, the
Day-1 split, and ``liang``, the strong-generalisation split of Liang et al. (2018) behind the
headline results (``data/liang.py``, docs/plan.md). The whole pipeline is CPU-only and runs in
seconds on a laptop (leave-one-out: 3.3 s warm / 9.1 s cold, 2.3 GB peak RSS on an M5;
experiment_log.md E001), so a fresh cloud GPU box can bootstrap its training data with one
command instead of shipping parquet around.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from shopsage_rec import __version__
from shopsage_rec.config import DataConfig
from shopsage_rec.data.liang import liang_split
from shopsage_rec.data.loader import load_movies, load_ratings, load_tags
from shopsage_rec.data.preprocess import (
    build_id_maps,
    filter_interactions,
    parse_movies,
    remap_ids,
)
from shopsage_rec.data.split import compute_tie_key, leave_one_out_split, write_splits

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrepareReport:
    """What ``run_prepare`` produced for a leave-one-out split; also the CLI summary."""

    processed_dir: Path
    n_ratings_raw: int
    n_movies: int
    n_tags: int
    n_users: int
    n_items: int
    n_train: int
    n_val: int
    n_test: int
    density: float
    users_with_boundary_tie: int
    users_with_any_boundary_tie: int
    cold_test_items: int
    split_sha256: str
    elapsed_s: float

    def summary(self) -> str:
        lines = [
            f"prepared {self.processed_dir}",
            f"  raw ratings     : {self.n_ratings_raw:>12,}",
            f"  movies / tags   : {self.n_movies:>12,} / {self.n_tags:,}",
            f"  users / items   : {self.n_users:>12,} / {self.n_items:,} (after filtering)",
            f"  train/val/test  : {self.n_train:>12,} / {self.n_val:,} / {self.n_test:,}",
            f"  density         : {self.density:>12.5%}",
            f"  boundary ties   : {self.users_with_boundary_tie:>12,} users (test/val same second)"
            f"; {self.users_with_any_boundary_tie:,} at either boundary",
            f"  cold test items : {self.cold_test_items:>12,}",
            f"  split sha256    : {self.split_sha256[:12]}...",
            f"  elapsed         : {self.elapsed_s:>11.1f}s",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class LiangReport:
    """What ``run_prepare`` produced for a Liang split; also the CLI summary."""

    processed_dir: Path
    n_ratings_raw: int
    n_users: int
    n_items: int
    n_train_users: int
    n_heldout_users: int
    n_train: int
    n_val_tr: int
    n_val_te: int
    n_test_tr: int
    n_test_te: int
    val_users_with_targets: int
    test_users_with_targets: int
    split_sha256: str
    elapsed_s: float

    def summary(self) -> str:
        lines = [
            f"prepared {self.processed_dir} (liang split)",
            f"  raw ratings     : {self.n_ratings_raw:>12,}",
            f"  users / items   : {self.n_users:>12,} / {self.n_items:,} (training catalogue)",
            f"  train users     : {self.n_train_users:>12,} ({self.n_train:,} rows)",
            f"  val users       : {self.n_heldout_users:>12,} (fold-in {self.n_val_tr:,}, "
            f"targets {self.n_val_te:,}; {self.val_users_with_targets:,} with targets)",
            f"  test users      : {self.n_heldout_users:>12,} (fold-in {self.n_test_tr:,}, "
            f"targets {self.n_test_te:,}; {self.test_users_with_targets:,} with targets)",
            f"  split sha256    : {self.split_sha256[:12]}...",
            f"  elapsed         : {self.elapsed_s:>11.1f}s",
        ]
        return "\n".join(lines)


def _load_raw(cfg: DataConfig, out: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Read the three CSVs, then write them to ``out`` as parquet (raw, unsafe for features)."""
    log.info("loading raw CSVs from %s", cfg.raw_dir)
    ratings = load_ratings(cfg.raw_dir)
    movies = parse_movies(load_movies(cfg.raw_dir))
    tags = load_tags(cfg.raw_dir)
    (out / "mappings").mkdir(parents=True, exist_ok=True)  # nothing on disk before the inputs load
    log.info(
        "ratings=%s movies=%s tags=%s", f"{ratings.height:,}", movies.height, f"{tags.height:,}"
    )
    ratings.write_parquet(out / "ratings.parquet", compression="zstd")
    movies.write_parquet(out / "movies.parquet", compression="zstd")
    tags.write_parquet(out / "tags.parquet", compression="zstd")
    return ratings, movies, tags


def _positives(cfg: DataConfig, ratings: pl.DataFrame) -> pl.DataFrame:
    log.info(
        "filtering positives: rating>=%s, items>=%d, users>=%d",
        cfg.min_rating,
        cfg.min_item_interactions,
        cfg.min_user_interactions,
    )
    return filter_interactions(
        ratings,
        min_rating=cfg.min_rating,
        min_item_interactions=cfg.min_item_interactions,
        min_user_interactions=cfg.min_user_interactions,
    )


def _provenance(
    cfg: DataConfig, ratings: pl.DataFrame, movies: pl.DataFrame, tags: pl.DataFrame
) -> dict[str, Any]:
    return {
        "dataset": cfg.dataset,
        "config": cfg.model_dump(mode="json"),
        "n_ratings_raw": ratings.height,
        "n_movies": movies.height,
        "n_tags": tags.height,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": {
            "shopsage_rec": __version__,
            "polars": pl.__version__,
            "python": platform.python_version(),
        },
    }


def run_prepare(cfg: DataConfig) -> PrepareReport | LiangReport:
    """Execute the full data pipeline described by ``cfg`` and return a report."""
    if cfg.split.strategy == "liang":
        return _run_liang(cfg)
    t0 = time.perf_counter()
    out = Path(cfg.processed_dir)
    ratings, movies, tags = _load_raw(cfg, out)
    positives = _positives(cfg, ratings)
    # Tie-break key over the RAW ids, before remapping, so re-filtering never changes it.
    positives = positives.with_columns(
        compute_tie_key(positives, "userId", "movieId", cfg.split.seed)
    )
    user_map, item_map = build_id_maps(positives)
    user_map.write_parquet(out / "mappings" / "user_map.parquet")
    item_map.write_parquet(out / "mappings" / "item_map.parquet")
    interactions = remap_ids(positives, user_map, item_map)

    log.info("leave-one-out split (seed=%d)", cfg.split.seed)
    result = leave_one_out_split(interactions, seed=cfg.split.seed)
    write_splits(result, out / "splits", extra=_provenance(cfg, ratings, movies, tags))

    report = PrepareReport(
        processed_dir=out,
        n_ratings_raw=ratings.height,
        n_movies=movies.height,
        n_tags=tags.height,
        n_users=result.stats["n_users"],
        n_items=result.stats["n_items"],
        n_train=result.stats["n_train"],
        n_val=result.stats["n_val"],
        n_test=result.stats["n_test"],
        density=result.stats["density"],
        users_with_boundary_tie=result.stats["users_with_boundary_tie"],
        users_with_any_boundary_tie=result.stats["users_with_any_boundary_tie"],
        cold_test_items=result.stats["cold_test_items"],
        split_sha256=result.stats["split_sha256"],
        elapsed_s=time.perf_counter() - t0,
    )
    log.info("done: %s", asdict(report))
    return report


def _run_liang(cfg: DataConfig) -> LiangReport:
    t0 = time.perf_counter()
    out = Path(cfg.processed_dir)
    ratings, movies, tags = _load_raw(cfg, out)
    positives = _positives(cfg, ratings)

    log.info(
        "liang split (seed=%d, %d held-out users per stage, test_prop=%s)",
        cfg.split.seed,
        cfg.split.n_heldout_users,
        cfg.split.test_prop,
    )
    split = liang_split(
        positives,
        seed=cfg.split.seed,
        n_heldout_users=cfg.split.n_heldout_users,
        test_prop=cfg.split.test_prop,
    )
    # Contiguous ids, ascending by raw id: items over the training catalogue, users over all
    # users of the split with their role (train / val / test).
    item_map = (
        split.train.select(pl.col("movieId").unique().sort())
        .with_row_index("item_idx")
        .select(pl.col("item_idx").cast(pl.Int32), pl.col("movieId"))
    )
    user_map = split.users.with_row_index("user_idx").select(
        pl.col("user_idx").cast(pl.Int32), pl.col("userId"), pl.col("role")
    )
    user_map.write_parquet(out / "mappings" / "user_map.parquet")
    item_map.write_parquet(out / "mappings" / "item_map.parquet")

    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split.frames().items():
        remap_ids(frame, user_map.select("user_idx", "userId"), item_map).sort(
            "user_idx", "item_idx"
        ).write_parquet(splits_dir / f"{name}.parquet", compression="zstd")
    meta = {**split.stats, **_provenance(cfg, ratings, movies, tags)}
    (splits_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    stats = split.stats
    report = LiangReport(
        processed_dir=out,
        n_ratings_raw=ratings.height,
        n_users=stats["n_users"],
        n_items=stats["n_items"],
        n_train_users=stats["n_train_users"],
        n_heldout_users=stats["n_heldout_users"],
        n_train=stats["n_train"],
        n_val_tr=stats["n_val_tr"],
        n_val_te=stats["n_val_te"],
        n_test_tr=stats["n_test_tr"],
        n_test_te=stats["n_test_te"],
        val_users_with_targets=stats["val_users_with_targets"],
        test_users_with_targets=stats["test_users_with_targets"],
        split_sha256=stats["split_sha256"],
        elapsed_s=time.perf_counter() - t0,
    )
    log.info("done: %s", asdict(report))
    return report
