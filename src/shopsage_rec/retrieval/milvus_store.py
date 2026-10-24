"""The serving index (D-008): one Milvus Lite collection holding, per item, the Two-Tower item
vector (FLAT, inner product) and the BM25 weight row (sparse inverted index, inner product).

Seen items are removed with a filter (``id not in [...]``) inside each search, before the cut,
so results equal the exact offline paths: ``topk_masked`` on dense scores, positive-score BM25
items (Milvus returns matching items only), and RRF over both with the same per-path depth
(``retrieval/fusion.py``). The collection is flushed after loading: unflushed rows are scanned in
Python on every query (E004). Milvus rejects an empty sparse vector, so an item without terms gets
a placeholder term (index ``n_terms``) that no query contains.

pymilvus is imported inside the methods: it pulls in pandas, and the offline code never needs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from shopsage_rec.retrieval.fusion import RRF_K

COLLECTION = "items"
PLACEHOLDER_WEIGHT = 1e-30
_PARAMS = {"metric_type": "IP"}

Hits = list[tuple[int, float]]


def sparse_row(matrix: sp.csr_matrix, row: int) -> dict[int, float]:
    """One CSR row as Milvus's sparse-vector dict."""
    start, stop = matrix.indptr[row], matrix.indptr[row + 1]
    return dict(
        zip(matrix.indices[start:stop].tolist(), matrix.data[start:stop].tolist(), strict=True)
    )


def _filter(exclude: Sequence[int] | np.ndarray) -> str:
    ids = np.asarray(exclude, dtype=np.int64)
    return f"id not in [{', '.join(map(str, ids.tolist()))}]" if ids.size else ""


def _hits(result: Any) -> Hits:
    return [(int(h["id"]), float(h["distance"])) for h in result[0]]


class MilvusStore:
    def __init__(self, client: Any, collection: str = COLLECTION) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def build(
        cls,
        uri: str | Path,
        item_vectors: np.ndarray,
        sparse_weights: sp.csr_matrix,
        *,
        batch: int = 2_000,
    ) -> MilvusStore:
        """Create the collection at ``uri`` (a ``.db`` file) and load every item into it."""
        from pymilvus import DataType, MilvusClient

        items = np.asarray(item_vectors, dtype=np.float32)
        weights = sp.csr_matrix(sparse_weights, dtype=np.float32)
        if items.shape[0] != weights.shape[0]:
            raise ValueError(f"{items.shape[0]} item vectors but {weights.shape[0]} weight rows")
        client = MilvusClient(str(uri))
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=items.shape[1])
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        index = client.prepare_index_params()
        index.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_collection(COLLECTION, schema=schema, index_params=index)
        placeholder = {int(weights.shape[1]): PLACEHOLDER_WEIGHT}
        for start in range(0, items.shape[0], batch):
            stop = min(start + batch, items.shape[0])
            client.insert(
                COLLECTION,
                [
                    {
                        "id": i,
                        "dense": items[i].tolist(),
                        "sparse": sparse_row(weights, i) or placeholder,
                    }
                    for i in range(start, stop)
                ],
            )
        client.flush(COLLECTION)
        return cls(client)

    def count(self) -> int:
        return int(self.client.get_collection_stats(self.collection)["row_count"])

    def search_dense(self, user: np.ndarray, exclude: Sequence[int] | np.ndarray, k: int) -> Hits:
        return _hits(
            self.client.search(
                self.collection,
                data=[np.asarray(user, dtype=np.float32).tolist()],
                anns_field="dense",
                limit=k,
                filter=_filter(exclude),
                search_params=_PARAMS,
            )
        )

    def search_sparse(
        self, query: Mapping[int, float], exclude: Sequence[int] | np.ndarray, k: int
    ) -> Hits:
        if not query:
            return []
        return _hits(
            self.client.search(
                self.collection,
                data=[dict(query)],
                anns_field="sparse",
                limit=k,
                filter=_filter(exclude),
                search_params=_PARAMS,
            )
        )

    def search_hybrid(
        self,
        user: np.ndarray,
        query: Mapping[int, float],
        exclude: Sequence[int] | np.ndarray,
        k: int,
        *,
        depth: tuple[int, int] = (200, 200),
        rrf_k: int = RRF_K,
    ) -> Hits:
        """RRF of the dense top-``depth[0]`` and the sparse top-``depth[1]``."""
        from pymilvus import AnnSearchRequest, RRFRanker

        expr = _filter(exclude)
        requests = [
            AnnSearchRequest(
                [np.asarray(user, dtype=np.float32).tolist()], "dense", _PARAMS, depth[0], expr=expr
            )
        ]
        if query:
            requests.append(AnnSearchRequest([dict(query)], "sparse", _PARAMS, depth[1], expr=expr))
        return _hits(
            self.client.hybrid_search(self.collection, requests, RRFRanker(rrf_k), limit=k)
        )
