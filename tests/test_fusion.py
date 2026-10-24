"""Reciprocal rank fusion, the offline twin of Milvus's RRFRanker."""

from __future__ import annotations

import numpy as np
import pytest

from shopsage_rec.retrieval.fusion import positive_only, rrf


def test_rrf_sums_reciprocal_ranks_from_one() -> None:
    a = np.array([[3, 1, 2]])
    b = np.array([[2, 4, -1]])  # -1 pads a short list

    top = rrf([a, b], n_items=6, k=60, top=4)

    scores = {3: 1 / 61, 1: 1 / 62, 2: 1 / 63 + 1 / 61, 4: 1 / 62}
    expected = sorted(scores, key=lambda i: (-scores[i], i))
    assert top.tolist() == [expected]  # ties (1 and 4) go to the lower item index


def test_rrf_rows_are_independent_and_the_top_is_bounded() -> None:
    rng = np.random.default_rng(0)
    lists = [np.stack([rng.permutation(50)[:20] for _ in range(5)]) for _ in range(2)]

    top = rrf(lists, n_items=50, top=10, chunk_size=2)

    for row in range(5):
        alone = rrf([lst[row : row + 1] for lst in lists], n_items=50, top=10)
        assert top[row].tolist() == alone[0].tolist()
    with pytest.raises(ValueError, match="rows"):
        rrf([lists[0], lists[1][:3]], n_items=50, top=10)


def test_positive_only_pads_items_without_a_match() -> None:
    top = np.array([[4, 2, 0], [1, 3, 2]])
    scores = np.array([[0.0, 0.0, 0.5, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0, 0.0]])

    assert positive_only(top, scores).tolist() == [[4, 2, -1], [1, -1, -1]]
