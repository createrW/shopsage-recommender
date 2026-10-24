"""The BM25 recall path: item documents, weights and user queries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from shopsage_rec.data.liang import load_liang
from shopsage_rec.retrieval.bm25 import (
    bm25_weights,
    build_item_documents,
    fit_bm25,
    normalize_tag,
    title_terms,
)


def test_title_terms_drop_stopwords_and_punctuation() -> None:
    assert title_terms("The Lord of the Rings: The Two Towers") == [
        "w:lord",
        "w:rings",
        "w:two",
        "w:towers",
    ]
    assert title_terms("Se7en") == ["w:se7en"]
    assert title_terms("2 Fast 2 Furious") == ["w:2", "w:fast", "w:2", "w:furious"]
    assert title_terms("") == []


def test_normalize_tag() -> None:
    assert normalize_tag("  Dark   Hero ") == "t:dark hero"
    assert normalize_tag("007 (series)") == "t:007 (series)"


def test_bm25_weights_match_the_formula() -> None:
    tf = sp.csr_matrix(np.array([[2, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=np.float32))
    k1, b = 1.2, 0.75

    w = bm25_weights(tf, k1=k1, b=b).toarray()

    n, df = 3, np.array([2, 1, 2])
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    dl = np.array([3.0, 2.0, 1.0])
    avgdl = dl.mean()
    t = tf.toarray()
    expected = idf * t * (k1 + 1) / (t + k1 * (1 - b + b * dl[:, None] / avgdl))
    np.testing.assert_allclose(w, np.where(t > 0, expected, 0.0), rtol=1e-6)
    assert w.dtype == np.float32


def test_scores_are_the_query_weighted_sum_of_document_weights() -> None:
    tf = sp.csr_matrix(np.array([[2, 0, 1], [0, 1, 1], [1, 0, 0], [0, 3, 0]], dtype=np.float32))
    fold_in = sp.csr_matrix(np.array([[1, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32))

    for query in ("binary", "count", "log"):
        model = fit_bm25(tf, ("a", "b", "c"), k1=1.2, b=0.75, query=query)
        counts = (fold_in @ (tf > 0).astype(np.float32)).toarray()
        q = {"binary": (counts > 0) * 1.0, "count": counts, "log": np.log1p(counts)}[query]
        np.testing.assert_allclose(
            model.scores(fold_in), q @ model.weights.toarray().T, rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(model.queries(fold_in).toarray(), q, rtol=1e-6)
    with pytest.raises(ValueError, match="query"):
        fit_bm25(tf, ("a", "b", "c"), query="tfidf")


def test_documents_use_training_users_tags_only(liang_dir: Path) -> None:
    data = load_liang(liang_dir)
    users = pl.read_parquet(liang_dir / "mappings" / "user_map.parquet")
    train_user = users.filter(pl.col("role") == "train")["userId"][0]
    test_user = users.filter(pl.col("role") == "test")["userId"][0]
    movies = pl.read_parquet(liang_dir / "mappings" / "item_map.parquet").sort("item_idx")
    m0, m1, m2 = movies["movieId"][:3].to_list()
    pl.DataFrame(
        {
            "userId": [train_user, train_user, test_user, test_user, train_user],
            "movieId": [m0, m1, m0, m1, m2],
            "tag": ["Shared Tag", "shared tag", "Leaky Tag", "leaky tag", "solo"],
            "timestamp": [1, 2, 3, 4, 5],
        },
        schema={"userId": pl.Int32, "movieId": pl.Int32, "tag": pl.String, "timestamp": pl.Int64},
    ).write_parquet(liang_dir / "tags.parquet")

    docs = build_item_documents(liang_dir, liang_dir.parent / "raw", genome_top=5)

    vocab = list(docs.vocab)
    assert docs.tf.shape == (data.n_items, len(vocab))
    assert "t:shared tag" in vocab  # two training-user documents
    assert "t:leaky tag" not in vocab  # a held-out user's tag
    assert "t:solo" not in vocab  # in one document only: it can never link two items
    assert docs.tf[0, vocab.index("t:shared tag")] == 1.0
    assert "g:drama" in vocab
    assert {term[:2] for term in vocab} == {"w:", "g:", "t:"}
    df = np.diff(docs.tf.tocsc().indptr)
    assert df.min() >= 2
    # every genome-covered item carries at most genome_top genome tags (plus user tags)
    genome_terms = [j for j, term in enumerate(vocab) if term.startswith("t:tag ")]
    assert (docs.tf[:, genome_terms] > 0).sum(axis=1).max() <= 5
