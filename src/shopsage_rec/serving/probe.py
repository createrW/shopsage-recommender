"""Deployment probe (S1): can a host run the demo's retrieval stack, in what memory and time?

It builds a Milvus Lite collection shaped like the demo's (20,108 items, each with a 128-d dense
vector and a sparse BM25-style vector) from synthetic data, flushes it, checks the dense FLAT
search against exact numpy search, and times dense, sparse and RRF hybrid queries. Milvus Lite 3.x
runs inside this process, so the reported peak RSS covers it; Streamlit Community Cloud guarantees
690 MB.
"""

from __future__ import annotations

import platform
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

COLLECTION = "probe_items"
RRF_K = 60


@dataclass(frozen=True)
class ProbeReport:
    n_items: int
    dim: int
    n_queries: int
    k: int
    import_s: float
    build_s: float
    dense_ms: tuple[float, float]  # p50, p95 per query
    sparse_ms: tuple[float, float]
    hybrid_ms: tuple[float, float]
    dense_overlap: float  # mean |Milvus top-k ∩ exact top-k| / k
    peak_rss_mb: float
    versions: dict[str, str]

    def rows(self) -> list[tuple[str, str]]:
        """Label / value pairs for display."""

        def ms(pair: tuple[float, float]) -> str:
            return f"{pair[0]:.1f} / {pair[1]:.1f} ms"

        return [
            ("catalogue", f"{self.n_items:,} items, {self.dim}-d dense + sparse vectors"),
            ("queries", f"{self.n_queries} per path, top-{self.k}"),
            ("import pymilvus", f"{self.import_s:.1f} s"),
            ("build collection", f"{self.build_s:.1f} s"),
            ("dense p50 / p95", ms(self.dense_ms)),
            ("sparse p50 / p95", ms(self.sparse_ms)),
            ("hybrid (RRF) p50 / p95", ms(self.hybrid_ms)),
            ("dense overlap with exact top-k", f"{self.dense_overlap:.3f}"),
            ("peak RSS", f"{self.peak_rss_mb:.0f} MB"),
            ("versions", ", ".join(f"{k} {v}" for k, v in self.versions.items())),
        ]


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / 2**20 if sys.platform == "darwin" else peak / 2**10)


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    p50, p95 = np.percentile(np.asarray(samples) * 1e3, [50, 95])
    return float(p50), float(p95)


def _version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return "?"


def run_probe(
    *,
    n_items: int = 20_108,
    dim: int = 128,
    n_queries: int = 100,
    k: int = 50,
    terms_per_item: int = 20,
    vocab: int = 5_000,
    seed: int = 0,
    db_dir: str | Path | None = None,
) -> ProbeReport:
    """Build the synthetic collection, query it, and report timings and memory."""
    started = time.perf_counter()
    from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

    import_s = time.perf_counter() - started
    rng = np.random.default_rng(seed)
    dense = rng.standard_normal((n_items, dim)).astype(np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    terms = [rng.choice(vocab, size=terms_per_item, replace=False) for _ in range(n_items)]
    weights = rng.random((n_items, terms_per_item)) + 0.1

    with tempfile.TemporaryDirectory(dir=db_dir) as tmp:
        t = time.perf_counter()
        client = MilvusClient(str(Path(tmp) / "probe.db"))
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        index = client.prepare_index_params()
        index.add_index(field_name="dense", index_type="FLAT", metric_type="IP")
        index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_collection(COLLECTION, schema=schema, index_params=index)
        for start in range(0, n_items, 2_000):
            stop = min(start + 2_000, n_items)
            client.insert(
                COLLECTION,
                [
                    {
                        "id": i,
                        "dense": dense[i].tolist(),
                        "sparse": dict(zip(terms[i].tolist(), weights[i].tolist(), strict=True)),
                    }
                    for i in range(start, stop)
                ],
            )
        # Seal the inserted rows: unflushed rows are scanned row by row in Python on every query
        # (~250 ms at this size on an M5), sealed segments go through the index (~9 ms).
        client.flush(COLLECTION)
        build_s = time.perf_counter() - t

        picks = rng.choice(n_items, size=n_queries, replace=False)
        dense_t: list[float] = []
        sparse_t: list[float] = []
        hybrid_t: list[float] = []
        overlap: list[float] = []
        params = {"metric_type": "IP"}
        for i in picks.tolist():
            q = dense[i] + 0.1 * rng.standard_normal(dim).astype(np.float32)
            q /= np.linalg.norm(q)
            exact = set(np.argsort(-(dense @ q), kind="stable")[:k].tolist())
            t = time.perf_counter()
            hits = client.search(
                COLLECTION, data=[q.tolist()], anns_field="dense", limit=k, search_params=params
            )[0]
            dense_t.append(time.perf_counter() - t)
            overlap.append(len(exact & {int(h["id"]) for h in hits}) / k)

            sq = {int(term): 1.0 for term in rng.choice(vocab, size=5, replace=False)}
            t = time.perf_counter()
            client.search(COLLECTION, data=[sq], anns_field="sparse", limit=k, search_params=params)
            sparse_t.append(time.perf_counter() - t)

            requests = [
                AnnSearchRequest(data=[q.tolist()], anns_field="dense", param=params, limit=k),
                AnnSearchRequest(data=[sq], anns_field="sparse", param=params, limit=k),
            ]
            t = time.perf_counter()
            client.hybrid_search(COLLECTION, reqs=requests, ranker=RRFRanker(RRF_K), limit=k)
            hybrid_t.append(time.perf_counter() - t)
        client.close()

    return ProbeReport(
        n_items=n_items,
        dim=dim,
        n_queries=n_queries,
        k=k,
        import_s=import_s,
        build_s=build_s,
        dense_ms=_p50_p95(dense_t),
        sparse_ms=_p50_p95(sparse_t),
        hybrid_ms=_p50_p95(hybrid_t),
        dense_overlap=float(np.mean(overlap)),
        peak_rss_mb=_peak_rss_mb(),
        versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pymilvus": _version("pymilvus"),
            "milvus_lite": _version("milvus-lite"),
        },
    )
