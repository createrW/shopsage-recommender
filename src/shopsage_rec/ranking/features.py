"""Features of (user, candidate item) pairs for the LightGBM ranker (D-006).

Every user-side input is the user's fold-in items; every item statistic comes from training users
only. Nothing here reads a held-out user's targets: ``build_stage`` takes fold-in rows alone, and
labels are attached afterwards by ``labels_for`` (the canary test changes the targets and checks
that no feature moves).

Features, in ``FEATURES`` order:

- ``dense_score``, ``dense_rank``: the Two-Tower inner product; the rank in the dense list (its
  depth + 1 outside it);
- ``bm25_score``, ``bm25_rank``: the BM25 score (0 when no term matches); the rank in the BM25 list;
- ``ease_score``: sparse EASE, the sum over history items h of ``B[h, item]``, with ``B`` pruned
  to its ``neighbours`` largest weights per column (small enough to ship with the demo);
- ``item_log_pop``: ``log1p`` of the item's training positives;
- ``item_mean_rating``: the mean of all training users' ratings of the item (NaN if none);
- ``genre_affinity``: the cosine of the item's genres with the summed genres of the history;
- ``genome_sim``: the cosine of the item's genome PCA vector with the history's mean (NaN if
  either is missing);
- ``item_year``, ``year_gap``: the release year and its distance to the history's mean year
  (NaN if unknown);
- ``log_hist_len``: ``log1p`` of the number of fold-in items.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

from shopsage_rec.data.liang import LiangData
from shopsage_rec.features.items import build_item_features
from shopsage_rec.models.numpy_tower import user_vectors
from shopsage_rec.ranking.candidates import union_candidates
from shopsage_rec.retrieval.bm25 import BM25, build_item_documents, fit_bm25

FEATURES: tuple[str, ...] = (
    "dense_score",
    "dense_rank",
    "bm25_score",
    "bm25_rank",
    "ease_score",
    "item_log_pop",
    "item_mean_rating",
    "genre_affinity",
    "genome_sim",
    "item_year",
    "year_gap",
    "log_hist_len",
)
BM25_PARAMS = (0.9, 0.4, "log")  # chosen on V_A in E008


def prune_neighbours(weights: np.ndarray, k: int) -> sp.csc_matrix:
    """Keep each column's ``k`` largest entries of a dense item-item matrix (ties: lower row)."""
    w = np.asarray(weights)
    n_rows, n_cols = w.shape
    k = min(k, n_rows)
    rows = np.empty((k, n_cols), dtype=np.int64)
    for start in range(0, n_cols, 2048):
        block = w[:, start : start + 2048]
        order = np.argsort(-block, axis=0, kind="stable")[:k]
        rows[:, start : start + block.shape[1]] = order
    cols = np.broadcast_to(np.arange(n_cols), (k, n_cols))
    values = w[rows, cols]
    return sp.csc_matrix(
        (values.ravel(order="F"), (rows.ravel(order="F"), cols.ravel(order="F"))),
        shape=w.shape,
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ItemStats:
    log_pop: np.ndarray  # float32 [n_items]
    mean_rating: np.ndarray  # float32 [n_items], NaN without training ratings
    genres: np.ndarray  # float32 [n_items, 19], multi-hot
    genome: np.ndarray  # float32 [n_items, 64], unit norm, zero where missing
    genome_ok: np.ndarray  # bool [n_items]
    year: np.ndarray  # float32 [n_items], NaN if unknown


def item_stats(processed_dir: str | Path, data: LiangData) -> ItemStats:
    processed = Path(processed_dir)
    feats = build_item_features(processed, data.meta["config"]["raw_dir"])
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet")
    train_users = pl.read_parquet(processed / "mappings" / "user_map.parquet").filter(
        pl.col("role") == "train"
    )
    means = (
        pl.read_parquet(processed / "ratings.parquet")
        .join(train_users.select("userId"), on="userId", how="semi")
        .join(item_map, on="movieId", how="inner")
        .group_by("item_idx")
        .agg(pl.col("rating").mean())
    )
    mean_rating = np.full(data.n_items, np.nan, dtype=np.float32)
    mean_rating[means["item_idx"].to_numpy()] = means["rating"].to_numpy()
    years = item_map.join(pl.read_parquet(processed / "movies.parquet"), on="movieId", how="left")
    year = np.full(data.n_items, np.nan, dtype=np.float32)
    known = years.filter(pl.col("year").is_not_null())
    year[known["item_idx"].to_numpy()] = known["year"].to_numpy()
    genome = np.asarray(feats.genome, dtype=np.float32)
    ok = feats.genome_missing < 0.5
    norm = np.linalg.norm(genome, axis=1, keepdims=True)
    genome = np.where(ok[:, None], genome / np.maximum(norm, 1e-12), 0.0).astype(np.float32)
    pop = np.asarray(data.train.sum(axis=0), dtype=np.float64).ravel()
    return ItemStats(
        log_pop=np.log1p(pop).astype(np.float32),
        mean_rating=mean_rating,
        genres=np.asarray(feats.genres, dtype=np.float32),
        genome=genome,
        genome_ok=ok,
        year=year,
    )


@dataclass(frozen=True)
class RankerContext:
    items: np.ndarray  # dense item vectors of the dense run
    tower: dict[str, np.ndarray]
    bm25: BM25
    ease: sp.csc_matrix  # pruned EASE weights
    stats: ItemStats

    @classmethod
    def load(
        cls,
        processed_dir: str | Path,
        data: LiangData,
        dense_run: str | Path,
        *,
        ease_weights: np.ndarray,
        neighbours: int = 100,
    ) -> RankerContext:
        run = Path(dense_run)
        with np.load(run / "user_tower.npz") as z:
            tower = {key: z[key] for key in z.files}
        docs = build_item_documents(processed_dir, data.meta["config"]["raw_dir"])
        k1, b, query = BM25_PARAMS
        return cls(
            items=np.load(run / "item_embeddings.npy"),
            tower=tower,
            bm25=fit_bm25(docs.tf, docs.vocab, k1=k1, b=b, query=query),
            ease=prune_neighbours(ease_weights, neighbours),
            stats=item_stats(processed_dir, data),
        )


@dataclass(frozen=True)
class Stage:
    offsets: np.ndarray  # int64 [n_users + 1]; user u's rows are offsets[u]:offsets[u+1]
    items: np.ndarray  # int64 [n_rows]
    features: np.ndarray  # float32 [n_rows, len(FEATURES)]

    @property
    def rows(self) -> np.ndarray:
        """The user row of every candidate."""
        return np.repeat(np.arange(self.offsets.size - 1), np.diff(self.offsets))


def _cosine_rows(profile: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(profile, axis=1, keepdims=True)
    return np.where(norm > 0, profile / np.maximum(norm, 1e-12), 0.0).astype(np.float32)


def _block(
    fold_in: sp.csr_matrix, ctx: RankerContext, depth: tuple[int, int]
) -> tuple[np.ndarray, ...]:
    s = ctx.stats
    users = user_vectors(fold_in, ctx.tower)
    dense = users @ ctx.items.T
    bm25 = ctx.bm25.scores(fold_in)
    rows, items, d_rank, b_rank = union_candidates(dense, bm25, fold_in, depth=depth)
    ease = np.asarray((fold_in @ ctx.ease)[rows, items]).ravel()  # sparse @ csc, picked pairs

    genre_profile = _cosine_rows(np.asarray(fold_in @ s.genres))
    genre_items = _cosine_rows(s.genres)
    genre_aff = np.einsum("ij,ij->i", genre_profile[rows], genre_items[items])

    ok = s.genome_ok.astype(np.float32)
    counts = np.asarray(fold_in @ ok).ravel()
    centroid = _cosine_rows(np.asarray(fold_in @ s.genome) / np.maximum(counts, 1.0)[:, None])
    genome_sim = np.einsum("ij,ij->i", centroid[rows], s.genome[items])
    genome_sim[~s.genome_ok[items] | (counts[rows] == 0)] = np.nan

    known = ~np.isnan(s.year)
    year_sum = np.asarray(fold_in @ np.where(known, s.year, 0.0)).ravel()
    year_n = np.asarray(fold_in @ known.astype(np.float32)).ravel()
    mean_year = np.where(year_n > 0, year_sum / np.maximum(year_n, 1.0), np.nan)
    hist = np.diff(fold_in.indptr).astype(np.float32)

    features = np.column_stack(
        [
            dense[rows, items],
            d_rank,
            bm25[rows, items],
            b_rank,
            ease,
            s.log_pop[items],
            s.mean_rating[items],
            genre_aff,
            genome_sim,
            s.year[items],
            np.abs(s.year[items] - mean_year[rows]),
            np.log1p(hist)[rows],
        ]
    ).astype(np.float32)
    return rows, items, features


def build_stage(
    fold_in: sp.csr_matrix,
    ctx: RankerContext,
    *,
    depth: tuple[int, int] = (500, 100),
    chunk: int = 1024,
) -> Stage:
    """Candidates and their features for every row of ``fold_in`` (binary user x item)."""
    if min(depth) < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    fold_in = sp.csr_matrix(fold_in, dtype=np.float32)
    counts, items, blocks = [], [], []
    for start in range(0, fold_in.shape[0], chunk):
        block = fold_in[start : start + chunk]
        rows, its, feats = _block(block, ctx, depth)
        counts.append(np.bincount(rows, minlength=block.shape[0]))
        items.append(its)
        blocks.append(feats)
    offsets = np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.int64)
    return Stage(offsets, np.concatenate(items), np.vstack(blocks))


def labels_for(stage: Stage, targets: sp.csr_matrix) -> np.ndarray:
    """1 where the candidate is one of its user's targets, else 0 (int32)."""
    n_items = targets.shape[1]
    coo = sp.coo_matrix(targets)
    keys = np.unique(coo.row.astype(np.int64) * n_items + coo.col)
    return np.isin(stage.rows * n_items + stage.items, keys).astype(np.int32)
