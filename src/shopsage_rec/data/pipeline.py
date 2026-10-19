"""``shopsage prepare``: raw CSVs -> parquet artifacts -> leave-one-out splits + meta.json.

The whole pipeline is CPU-only and runs in seconds on a laptop (3.3 s warm / 9.1 s cold at
2.3 GB peak RSS on an M5; experiment_log.md E001), so a fresh cloud GPU box can bootstrap its
training data with one command instead of shipping parquet around.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from shopsage_rec import __version__
from shopsage_rec.config import DataConfig
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
    """What ``run_prepare`` produced; also the content of the CLI summary."""

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


def run_prepare(cfg: DataConfig) -> PrepareReport:
    """Execute the full data pipeline described by ``cfg`` and return a report."""
    t0 = time.perf_counter()
    out = Path(cfg.processed_dir)

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

    log.info(
        "filtering positives: rating>=%s, items>=%d, users>=%d",
        cfg.min_rating,
        cfg.min_item_interactions,
        cfg.min_user_interactions,
    )
    positives = filter_interactions(
        ratings,
        min_rating=cfg.min_rating,
        min_item_interactions=cfg.min_item_interactions,
        min_user_interactions=cfg.min_user_interactions,
    )
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
    extra = {
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
    write_splits(result, out / "splits", extra=extra)

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
