"""The sparse recall path: BM25 over item documents, queried with a user's fold-in items (D-010).

An item's document is a bag of terms from four sources, prefixed by kind so that they never
collide:

- ``w:`` title words (``title_clean``, lower-cased, accents folded, English stopwords dropped);
- ``g:`` genres;
- ``t:`` the item's top ``genome_top`` tags by genome relevance, plus the tags that **training
  users** applied to it (one count per user). Tags of held-out users never enter a document.

Terms in fewer than ``min_df`` documents are dropped: a term found in one document can only match
that item, which is the user's own input item and is masked anyway. Document weights are BM25
(Lucene idf ``log(1 + (N - df + 0.5) / (df + 0.5))``); a user's query counts, for each term, the
fold-in items whose document contains it, as ``binary``, ``count`` or ``log`` (``log1p`` of the
count). The score of an item is the inner product of the query with its weight row, so the same
weights serve as Milvus sparse vectors (``retrieval/milvus_store.py``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp

QueryMode = Literal["binary", "count", "log"]
QUERY_MODES: tuple[str, ...] = ("binary", "count", "log")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "de",
        "do",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "into",
        "is",
        "it",
        "its",
        "la",
        "le",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "who",
        "will",
        "with",
        "you",
        "your",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+")
_SPACES = re.compile(r"\s+")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def title_terms(title: str | None) -> list[str]:
    """``w:`` terms of a title, in order, stopwords dropped."""
    return [f"w:{tok}" for tok in _TOKEN.findall(_fold(title or "")) if tok not in STOPWORDS]


def normalize_tag(tag: str) -> str:
    """The ``t:`` term of a free-text or genome tag: lower-cased, whitespace collapsed."""
    return "t:" + _SPACES.sub(" ", _fold(tag)).strip()


@dataclass(frozen=True)
class ItemDocuments:
    tf: sp.csr_matrix  # float32 [n_items, n_terms], term counts
    vocab: tuple[str, ...]


def _title_and_genre_rows(processed: Path, item_map: pl.DataFrame) -> pl.DataFrame:
    movies = item_map.join(pl.read_parquet(processed / "movies.parquet"), on="movieId", how="left")
    rows: list[tuple[int, str]] = []
    for item, title, genres in movies.select("item_idx", "title_clean", "genres").iter_rows():
        rows.extend((item, term) for term in title_terms(title))
        rows.extend((item, "g:" + g.lower()) for g in genres or () if not g.startswith("("))
    return pl.DataFrame(
        rows, schema={"item_idx": pl.Int32, "term": pl.String}, orient="row"
    ).with_columns(tf=pl.lit(1.0))


def _genome_rows(raw: Path, item_map: pl.DataFrame, top: int) -> pl.DataFrame:
    names = pl.read_csv(raw / "genome-tags.csv", schema_overrides={"tagId": pl.Int32})
    names = names.with_columns(
        term=pl.col("tag").map_elements(normalize_tag, return_dtype=pl.String)
    )
    scores = pl.read_csv(
        raw / "genome-scores.csv",
        schema_overrides={"movieId": pl.Int32, "tagId": pl.Int32, "relevance": pl.Float32},
    )
    best = (
        scores.join(item_map, on="movieId", how="inner")
        .sort(["item_idx", "relevance", "tagId"], descending=[False, True, False])
        .group_by("item_idx", maintain_order=True)
        .head(top)
    )
    return best.join(names.select("tagId", "term"), on="tagId").select(
        "item_idx", "term", tf=pl.lit(1.0)
    )


def _training_tag_rows(processed: Path, item_map: pl.DataFrame) -> pl.DataFrame:
    users = pl.read_parquet(processed / "mappings" / "user_map.parquet").filter(
        pl.col("role") == "train"
    )
    tags = (
        pl.read_parquet(processed / "tags.parquet")
        .join(users.select("userId"), on="userId", how="semi")
        .join(item_map, on="movieId", how="inner")
    )
    unique = tags["tag"].unique().to_list()
    mapping = {tag: normalize_tag(tag) for tag in unique}
    tags = tags.with_columns(term=pl.col("tag").replace_strict(mapping, return_dtype=pl.String))
    return (
        tags.filter(pl.col("term") != "t:")
        .unique(["userId", "item_idx", "term"])
        .group_by("item_idx", "term")
        .agg(tf=pl.len().cast(pl.Float64))
    )


def build_item_documents(
    processed_dir: str | Path, raw_dir: str | Path, *, genome_top: int = 20, min_df: int = 2
) -> ItemDocuments:
    """Term counts for every ``item_idx`` of a prepared Liang split."""
    processed, raw = Path(processed_dir), Path(raw_dir)
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet").sort("item_idx")
    rows = (
        pl.concat(
            [
                _title_and_genre_rows(processed, item_map),
                _genome_rows(raw, item_map, genome_top),
                _training_tag_rows(processed, item_map),
            ],
            how="vertical_relaxed",
        )
        .group_by("item_idx", "term")
        .agg(pl.col("tf").sum())
    )
    df = rows.group_by("term").agg(df=pl.len()).filter(pl.col("df") >= min_df)
    vocab = tuple(sorted(df["term"].to_list()))
    index = {term: j for j, term in enumerate(vocab)}
    rows = rows.join(df.select("term"), on="term", how="semi")
    cols = np.fromiter(
        (index[t] for t in rows["term"].to_list()), dtype=np.int64, count=rows.height
    )
    tf = sp.csr_matrix(
        (rows["tf"].to_numpy().astype(np.float32), (rows["item_idx"].to_numpy(), cols)),
        shape=(item_map.height, len(vocab)),
        dtype=np.float32,
    )
    tf.sort_indices()
    return ItemDocuments(tf=tf, vocab=vocab)


def bm25_weights(tf: sp.csr_matrix, *, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    """BM25 term weights of each document (row), float32, same sparsity as ``tf``."""
    tf = sp.csr_matrix(tf, dtype=np.float64)
    n_docs = tf.shape[0]
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
    dl = np.asarray(tf.sum(axis=1)).ravel()
    norm = k1 * (1.0 - b + b * dl / max(dl.mean(), 1e-12))
    rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
    data = idf[tf.indices] * tf.data * (k1 + 1.0) / (tf.data + norm[rows])
    return sp.csr_matrix((data.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)


@dataclass(frozen=True)
class BM25:
    weights: sp.csr_matrix  # float32 [n_items, n_terms]
    weights_t: sp.csr_matrix  # its transpose, for user x item products
    presence: sp.csr_matrix  # float32 [n_items, n_terms], 1 where the term is in the document
    vocab: tuple[str, ...]
    k1: float
    b: float
    query: str

    def queries(self, fold_in: sp.csr_matrix) -> sp.csr_matrix:
        """Query vectors (users x terms) from binary fold-in rows (users x items)."""
        q = sp.csr_matrix(sp.csr_matrix(fold_in, dtype=np.float32) @ self.presence)
        if self.query == "binary":
            q.data[:] = 1.0
        elif self.query == "log":
            q.data = np.log1p(q.data)
        return q

    def scores(self, fold_in: sp.csr_matrix) -> np.ndarray:
        """Dense float32 scores, users x items."""
        return np.asarray((self.queries(fold_in) @ self.weights_t).toarray(), dtype=np.float32)


def fit_bm25(
    tf: sp.csr_matrix,
    vocab: tuple[str, ...],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    query: str = "log",
) -> BM25:
    if query not in QUERY_MODES:
        raise ValueError(f"query must be one of {QUERY_MODES}, got {query!r}")
    weights = bm25_weights(tf, k1=k1, b=b)
    presence = sp.csr_matrix(tf, dtype=np.float32, copy=True)
    presence.data[:] = 1.0
    return BM25(weights, weights.T.tocsr(), presence, vocab, k1, b, query)
