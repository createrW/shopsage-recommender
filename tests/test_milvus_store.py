"""The Milvus Lite serving index returns what the exact offline paths return."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

from shopsage_rec.evaluation.metrics import topk_masked
from shopsage_rec.retrieval.fusion import positive_only, rrf, rrf_scores
from shopsage_rec.retrieval.milvus_store import MilvusStore, sparse_row


def _catalogue(seed: int = 0) -> tuple[np.ndarray, sp.csr_matrix]:
    rng = np.random.default_rng(seed)
    items = rng.standard_normal((300, 16)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    weights = sp.random(300, 80, density=0.15, random_state=seed, dtype=np.float32, format="csr")
    weights.data += 0.1
    keep = np.ones(300, dtype=np.float32)
    keep[7] = 0.0  # an item without terms
    weights = sp.csr_matrix(sp.diags(keep) @ weights)
    weights.eliminate_zeros()
    return items, weights


def test_dense_sparse_and_hybrid_match_the_offline_paths(tmp_path: Path) -> None:
    items, weights = _catalogue()
    store = MilvusStore.build(tmp_path / "items.db", items, weights)
    rng = np.random.default_rng(1)
    assert store.count() == 300

    for _ in range(10):
        seen = np.sort(rng.choice(300, size=12, replace=False))
        exclude = sp.csr_matrix((np.ones(12), (np.zeros(12, int), seen)), shape=(1, 300))
        user = rng.standard_normal(16).astype(np.float32)
        query = sp.random(1, 80, density=0.3, random_state=int(rng.integers(1 << 30)), format="csr")
        query.data += 0.1

        dense = topk_masked((items @ user)[None], exclude, 30)[0]
        sparse_scores = (query @ weights.T).toarray()
        sparse = topk_masked(sparse_scores, exclude, 40)[0]
        lists = [dense[None], positive_only(sparse[None], sparse_scores)]
        hybrid = rrf(lists, n_items=300, top=20)[0]
        fused = rrf_scores(lists, n_items=300)[0]

        got_dense = store.search_dense(user, seen, 30)
        got_sparse = store.search_sparse(sparse_row(query, 0), seen, 40)
        got_hybrid = store.search_hybrid(user, sparse_row(query, 0), seen, 20, depth=(30, 40))

        assert [i for i, _ in got_dense] == dense.tolist()
        np.testing.assert_allclose([s for _, s in got_dense], (items @ user)[dense], rtol=1e-5)
        positive = sparse[sparse_scores[0, sparse] > 0]  # Milvus returns matching items only
        assert [i for i, _ in got_sparse] == positive.tolist()
        # RRF ties are common (rank 1 in either list scores 1/61), so compare fused scores
        ids = [i for i, _ in got_hybrid]
        np.testing.assert_allclose([s for _, s in got_hybrid], fused[ids], rtol=1e-5)
        np.testing.assert_allclose(sorted(fused[ids]), sorted(fused[hybrid]), rtol=1e-5)
        assert not set(ids) & set(seen.tolist())


def test_an_item_without_terms_is_stored_but_never_matched(tmp_path: Path) -> None:
    items, weights = _catalogue()
    store = MilvusStore.build(tmp_path / "items.db", items, weights)

    query = {int(j): 1.0 for j in range(80)}
    hits = store.search_sparse(query, np.array([], dtype=np.int64), 300)

    assert 7 not in {i for i, _ in hits}
    assert 7 in {i for i, _ in store.search_dense(items[7], np.array([], dtype=np.int64), 1)}
