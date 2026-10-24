"""The S1 deployment probe: Milvus Lite dense, sparse and hybrid search on a synthetic catalogue."""

from __future__ import annotations

import sys
from pathlib import Path

from shopsage_rec.serving.probe import run_probe


def test_run_probe_matches_exact_search_and_times_all_three_paths(tmp_path: Path) -> None:
    report = run_probe(n_items=600, dim=16, n_queries=12, k=10, vocab=300, db_dir=tmp_path)

    assert (report.n_items, report.dim, report.n_queries, report.k) == (600, 16, 12, 10)
    assert report.dense_overlap == 1.0  # Milvus Lite's FLAT index returns the exact top-k
    for p50, p95 in (report.dense_ms, report.sparse_ms, report.hybrid_ms):
        assert 0 < p50 <= p95
    assert report.peak_rss_mb > 0
    assert {"pymilvus", "milvus_lite", "numpy"} <= set(report.versions)
    labels = [label for label, _ in report.rows()]
    assert "peak RSS" in labels
    assert "torch" not in sys.modules  # the demo stack never imports torch
