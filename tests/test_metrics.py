"""Full-corpus ranking metrics and the paired bootstrap (api_spec.md §8)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from shopsage_rec.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from shopsage_rec.evaluation.metrics import (
    evaluate_topk,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    relevance,
    topk_for_users,
    topk_masked,
)


def _csr(rows: list[list[int]], n_items: int) -> sp.csr_matrix:
    r = [u for u, items in enumerate(rows) for _ in items]
    c = [i for items in rows for i in items]
    return sp.csr_matrix(
        (np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(rows), n_items), dtype=np.float32
    )


def test_topk_masked_orders_by_score() -> None:
    scores = np.array([[0.1, 0.9, 0.5, 0.7]], dtype=np.float32)

    assert topk_masked(scores, None, 3).tolist() == [[1, 3, 2]]


def test_topk_masked_never_returns_a_fold_in_item_even_when_it_scores_highest() -> None:
    # The user's fold-in items (0 and 1) score highest; masking happens before the top-k cut,
    # so the user still gets k = 3 items, none of them seen.
    scores = np.array([[9.0, 8.0, 0.3, 0.2, 0.5, 0.1]], dtype=np.float32)
    seen = _csr([[0, 1]], 6)

    assert topk_masked(scores, seen, 3).tolist() == [[4, 2, 3]]


def test_topk_masked_breaks_ties_by_ascending_item_index() -> None:
    scores = np.array([[1.0, 2.0, 2.0, 2.0, 0.0], [5.0, 5.0, 5.0, 5.0, 5.0]], dtype=np.float32)

    assert topk_masked(scores, None, 2).tolist() == [[1, 2], [0, 1]]
    assert topk_masked(scores, _csr([[2], [0]], 5), 3).tolist() == [[1, 3, 0], [1, 2, 3]]


def test_topk_masked_rejects_bad_input() -> None:
    scores = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="k"):
        topk_masked(scores, None, 0)
    with pytest.raises(ValueError, match="k"):
        topk_masked(scores, None, 5)
    with pytest.raises(ValueError, match="fewer than k"):
        topk_masked(scores, _csr([[0, 1, 2], []], 4), 2)
    with pytest.raises(ValueError, match="NaN"):
        topk_masked(np.array([[0.0, np.nan, 1.0]]), None, 2)
    with pytest.raises(ValueError, match="shape"):
        topk_masked(scores, _csr([[0]], 4), 2)


def test_topk_for_users_matches_one_unchunked_call() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((7, 30)).astype(np.float32)
    seen = _csr([list(rng.choice(30, size=4, replace=False)) for _ in range(7)], 30)
    expected = topk_masked(scores, seen, 5)

    for chunk_size in (1, 2, 3, 7, 100):
        got = topk_for_users(lambda rows: scores[rows], seen, 5, chunk_size=chunk_size)
        assert np.array_equal(got, expected), chunk_size


def test_relevance_marks_recommended_targets() -> None:
    topk = np.array([[3, 1, 0], [2, 4, 1]])
    targets = _csr([[1, 4], [2]], 5)

    assert relevance(topk, targets).tolist() == [[False, True, False], [True, False, False]]


def test_recall_is_truncated_at_k() -> None:
    rel = np.array([[True, True, False], [False, False, True], [False, False, False]])
    n_relevant = np.array([5, 1, 2])

    # hits / min(k, n): 2 / min(2, 5), 0 / min(2, 1), 0 / min(2, 2); then at k = 3
    assert recall_at_k(rel, n_relevant, 2).tolist() == [1.0, 0.0, 0.0]
    assert recall_at_k(rel, n_relevant, 3).tolist() == pytest.approx([2 / 3, 1.0, 0.0])


def test_ndcg_uses_binary_gains_and_the_truncated_ideal() -> None:
    rel = np.array([[True, False, True], [False, True, False]])
    n_relevant = np.array([2, 4])

    dcg0 = 1 / math.log2(2) + 1 / math.log2(4)
    idcg0 = 1 / math.log2(2) + 1 / math.log2(3)
    dcg1 = 1 / math.log2(3)
    idcg1 = sum(1 / math.log2(r + 1) for r in (1, 2, 3))
    assert ndcg_at_k(rel, n_relevant, 3) == pytest.approx([dcg0 / idcg0, dcg1 / idcg1])
    assert ndcg_at_k(rel, n_relevant, 1) == pytest.approx([1.0, 0.0])


def test_hit_rate_is_any_hit_in_the_top_k() -> None:
    rel = np.array([[False, True], [False, False]])

    assert hit_rate_at_k(rel, 1).tolist() == [0.0, 0.0]
    assert hit_rate_at_k(rel, 2).tolist() == [1.0, 0.0]


def test_evaluate_topk_reports_every_requested_metric() -> None:
    topk = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    targets = _csr([[1, 3], [0]], 6)

    out = evaluate_topk(topk, targets, recall_ks=(2,), ndcg_ks=(4,), hr_ks=(1,))

    assert set(out) == {"recall@2", "ndcg@4", "hr@1"}
    assert out["recall@2"].tolist() == [0.5, 0.0]
    assert out["hr@1"].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="at least one target"):
        evaluate_topk(topk, _csr([[1], []], 6))


def test_bootstrap_ci_of_a_constant_is_that_constant() -> None:
    ci = bootstrap_ci(np.full(50, 0.25))

    assert (ci.estimate, ci.low, ci.high, ci.n) == (0.25, 0.25, 0.25, 50)


def test_paired_bootstrap_follows_the_documented_procedure() -> None:
    rng = np.random.default_rng(1)
    a = rng.random(300)
    b = a - 0.05 + rng.normal(0, 0.01, 300)

    ci = paired_bootstrap(a, b, n_resamples=2000, seed=42)

    # api_spec.md §8.4: default_rng(seed), indices in chunks of 100 replicates, percentile CI
    gen = np.random.default_rng(42)
    deltas = np.concatenate(
        [(a - b)[gen.integers(0, 300, size=(100, 300))].mean(axis=1) for _ in range(20)]
    )
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    assert ci.estimate == pytest.approx(float((a - b).mean()))
    assert (ci.low, ci.high) == pytest.approx((lo, hi))
    assert ci.low < ci.estimate < ci.high
    assert paired_bootstrap(a, b, n_resamples=2000, seed=42) == ci  # reproducible


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="same users"):
        paired_bootstrap(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="empty"):
        bootstrap_ci(np.array([]))
