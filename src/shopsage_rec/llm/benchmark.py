"""Ablation (c): does LLM expansion help BM25 answer a free-text query? (D-010)

**Events.** Tag applications by held-out (validation and test) users: a query is the tag's text,
and its relevant items are the catalogue movies that user tagged with it. Item documents hold
training users' tags only, so no query's own tag is in any document. At most two events per user,
``n`` sampled with a fixed seed.

**Arms.** Both rank the full catalogue by BM25 (E008 settings), no personalisation:

- **C0:** the raw query's terms: its words (``w:``), the whole tag (``t:``), a genre (``g:``),
  each with weight 1 when the vocabulary has it;
- **C1:** C0 plus the expansion's grounded terms with weight ``weight`` (0.5): keywords as ``t:``
  terms (or their words), genres as ``g:`` terms, titles that match a catalogue title as that
  title's words. An expansion that grounds nothing leaves the raw query;
- **C1r (routed):** C1 when the whole tag is not a term of the vocabulary, else C0. Added after
  E010 showed expansion helping only on unknown tags; it is judged on fresh events
  (``exclude``: the events of an earlier run).

**Metrics.** HR@10, HR@50 (a relevant movie in the top k) and Recall@50 per event; events are
averaged per user and the users bootstrapped (a cluster bootstrap: users carry unequal numbers of
events), paired between arms. Cost from each live call's ``usage``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from shopsage_rec import __version__
from shopsage_rec.data.liang import load_liang
from shopsage_rec.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from shopsage_rec.evaluation.metrics import relevance, topk_masked
from shopsage_rec.llm.expand import Expansion, QueryExpander, normalize_query
from shopsage_rec.retrieval.bm25 import (
    E008_PARAMS,
    TOKEN,
    build_item_documents,
    fit_bm25,
    fold_text,
    normalize_tag,
    title_terms,
)

EXPANSION_WEIGHT = 0.5
K = 50
METRICS = ("hr@10", "hr@50", "recall@50")
_ARTICLE = ("the", "a", "an")


@dataclass(frozen=True)
class TagBenchmarkResult:
    out_dir: Path
    summary: dict[str, Any]


def normalize_title(title: str) -> str:
    """Lower-cased words of a title; a trailing article ("Matrix, The") moves to the front."""
    words = TOKEN.findall(fold_text(title))
    raw = fold_text(title).strip()
    for article in _ARTICLE:
        if raw.endswith(f", {article}") and words and words[-1] == article:
            words = [article, *words[:-1]]
    return " ".join(words)


def text_query(text: str, vocab: Mapping[str, int]) -> dict[int, float]:
    """The raw query: its words, the whole phrase as a tag and as a genre, weight 1 each."""
    terms = [*title_terms(text), normalize_tag(text), "g:" + normalize_query(text)]
    return {vocab[t]: 1.0 for t in terms if t in vocab}


def expansion_query(
    text: str,
    expansion: Expansion,
    vocab: Mapping[str, int],
    titles: Mapping[str, Sequence[int]],
    *,
    weight: float = EXPANSION_WEIGHT,
) -> tuple[dict[int, float], dict[str, Any]]:
    """The raw query plus the expansion's grounded terms, and the grounded share per field."""
    query = text_query(text, vocab)
    added: dict[int, float] = {}
    shares: dict[str, Any] = {}

    def ground(terms: list[str]) -> bool:
        hits = [vocab[t] for t in terms if t in vocab]
        for j in hits:
            added[j] = max(added.get(j, 0.0), weight)
        return bool(hits)

    kw = [ground([normalize_tag(k)]) or ground(title_terms(k)) for k in expansion.keywords]
    gen = [ground(["g:" + g.lower()]) for g in expansion.genres]
    tit = [normalize_title(t) in titles and ground(title_terms(t)) for t in expansion.titles]
    for name, flags in (("keywords", kw), ("genres", gen), ("titles", tit)):
        shares[name] = float(np.mean(flags)) if flags else None
    shares["any"] = bool(added)
    merged = dict(added)
    merged.update(query)  # a raw term keeps its full weight
    return merged, shares


def tag_events(
    processed_dir: str | Path, *, n: int = 1000, seed: int = 0, exclude: pl.DataFrame | None = None
) -> pl.DataFrame:
    """``(userId, query, term, relevant)``: held-out users' tag events, at most two per user,
    none of the ``(userId, term)`` pairs in ``exclude``."""
    processed = Path(processed_dir)
    users = pl.read_parquet(processed / "mappings" / "user_map.parquet").filter(
        pl.col("role") != "train"
    )
    item_map = pl.read_parquet(processed / "mappings" / "item_map.parquet")
    tags = (
        pl.read_parquet(processed / "tags.parquet")
        .join(users.select("userId"), on="userId", how="semi")
        .join(item_map, on="movieId", how="inner")
        .with_columns(query=pl.col("tag").str.strip_chars())
    )
    mapping = {t: normalize_tag(t) for t in tags["query"].unique().to_list()}
    tags = tags.with_columns(term=pl.col("query").replace_strict(mapping, return_dtype=pl.String))
    events = (
        tags.filter(pl.col("term") != "t:")
        .sort("userId", "term", "timestamp", "item_idx")
        .group_by("userId", "term", maintain_order=True)
        .agg(pl.col("query").first(), relevant=pl.col("item_idx").unique().sort())
    )
    if exclude is not None:
        events = events.join(exclude.select("userId", "term"), on=["userId", "term"], how="anti")
    rng = np.random.default_rng(seed)
    events = events.with_columns(order=pl.Series(rng.permutation(events.height)))
    events = events.sort("order").group_by("userId", maintain_order=True).head(2)
    return events.sort("order").head(n).drop("order")


def _queries_matrix(queries: Sequence[Mapping[int, float]], n_terms: int) -> sp.csr_matrix:
    rows = np.repeat(np.arange(len(queries)), [len(q) for q in queries])
    cols = np.fromiter((j for q in queries for j in q), dtype=np.int64, count=rows.size)
    vals = np.fromiter((v for q in queries for v in q.values()), dtype=np.float32, count=rows.size)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), n_terms), dtype=np.float32)


def _metrics(top: np.ndarray, targets: sp.csr_matrix) -> dict[str, np.ndarray]:
    rel = relevance(top, targets)
    n_rel = np.diff(targets.indptr)
    return {
        "hr@10": rel[:, :10].any(axis=1).astype(np.float64),
        "hr@50": rel[:, :50].any(axis=1).astype(np.float64),
        "recall@50": rel[:, :50].sum(axis=1) / np.minimum(n_rel, 50),
    }


def run_tag_benchmark(
    processed_dir: str | Path,
    out_root: str | Path,
    expander: QueryExpander,
    *,
    n_events: int = 1000,
    workers: int = 8,
    weight: float = EXPANSION_WEIGHT,
    seed: int = 0,
    exclude: pl.DataFrame | None = None,
    run_id: str | None = None,
) -> TagBenchmarkResult:
    started = time.perf_counter()
    run_id = run_id or datetime.now().strftime("tags-%Y%m%d-%H%M")
    out = Path(out_root) / run_id
    out.mkdir(parents=True, exist_ok=True)
    processed = Path(processed_dir)
    data = load_liang(processed)
    docs = build_item_documents(processed, data.meta["config"]["raw_dir"])
    k1, b, mode = E008_PARAMS
    bm25 = fit_bm25(docs.tf, docs.vocab, k1=k1, b=b, query=mode)
    vocab = {t: j for j, t in enumerate(docs.vocab)}
    movies = pl.read_parquet(processed / "mappings" / "item_map.parquet").join(
        pl.read_parquet(processed / "movies.parquet"), on="movieId", how="left"
    )
    titles: dict[str, list[int]] = {}
    for item, title in movies.select("item_idx", "title_clean").iter_rows():
        if title:
            titles.setdefault(normalize_title(title), []).append(item)

    events = tag_events(processed, n=n_events, seed=seed, exclude=exclude)
    unique = sorted({normalize_query(q) for q in events["query"].to_list()})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        expansions = dict(zip(unique, pool.map(expander.expand, unique), strict=True))

    raw_q, exp_q, groundings = [], [], []
    for query in events["query"].to_list():
        raw_q.append(text_query(query, vocab))
        q1, g = expansion_query(
            query, expansions[normalize_query(query)], vocab, titles, weight=weight
        )
        exp_q.append(q1)
        groundings.append(g)
    rel_rows = np.repeat(np.arange(events.height), events["relevant"].list.len().to_numpy())
    rel_cols = np.concatenate(events["relevant"].to_list()).astype(np.int64)
    targets = sp.csr_matrix(
        (np.ones(rel_rows.size, dtype=np.float32), (rel_rows, rel_cols)),
        shape=(events.height, data.n_items),
    )
    known = [t in vocab for t in events["term"].to_list()]
    routed = [r if k else e for r, e, k in zip(raw_q, exp_q, known, strict=True)]
    per_event: dict[str, dict[str, np.ndarray]] = {}
    for arm, queries in (("C0", raw_q), ("C1", exp_q), ("C1r", routed)):
        scores = (_queries_matrix(queries, len(docs.vocab)) @ bm25.weights_t).toarray()
        per_event[arm] = _metrics(topk_masked(scores, None, K), targets)

    users = events["userId"].to_numpy()
    _, user_index = np.unique(users, return_inverse=True)
    n_users = int(np.max(user_index)) + 1

    def per_user(values: np.ndarray) -> np.ndarray:
        return np.bincount(user_index, weights=values, minlength=n_users) / np.bincount(user_index)

    arms = {
        arm: {m: bootstrap_ci(per_user(v)).as_dict() for m, v in metrics.items()}
        for arm, metrics in per_event.items()
    }
    paired = {
        f"{a} - {b}": {
            m: paired_bootstrap(per_user(per_event[a][m]), per_user(per_event[b][m])).as_dict()
            for m in METRICS
        }
        for a, b in (("C1", "C0"), ("C1r", "C0"), ("C1r", "C1"))
    }
    subgroups = {}
    for name, mask in (
        ("tag_in_vocab", np.asarray(known)),
        ("tag_not_in_vocab", ~np.asarray(known)),
    ):
        if mask.sum() == 0:
            continue
        _, sub_index = np.unique(users[mask], return_inverse=True)
        counts = np.bincount(sub_index)

        def sub_user(
            values: np.ndarray,
            m: np.ndarray = mask,
            i: np.ndarray = sub_index,
            c: np.ndarray = counts,
        ) -> np.ndarray:
            return np.bincount(i, weights=values[m]) / c

        subgroups[name] = {
            "events": int(mask.sum()),
            "C0": {m: float(per_event["C0"][m][mask].mean()) for m in METRICS},
            "C1": {m: float(per_event["C1"][m][mask].mean()) for m in METRICS},
            "C1 - C0": {
                m: paired_bootstrap(
                    sub_user(per_event["C1"][m]), sub_user(per_event["C0"][m])
                ).as_dict()
                for m in ("hr@10", "hr@50")
            },
        }
    sources = [e.source for e in expansions.values()]
    live = [e for e in expansions.values() if e.source == "llm"]
    fallbacks: dict[str, int] = {}
    for s in sources:
        if s.startswith("fallback:"):
            fallbacks[s.split(":", 1)[1]] = fallbacks.get(s.split(":", 1)[1], 0) + 1
    cost = float(sum(e.cost_usd for e in live))
    latency = np.asarray([e.latency_s for e in live]) * 1e3
    field_share = {
        f: float(np.mean([g[f] for g in groundings if g[f] is not None]))
        if any(g[f] is not None for g in groundings)
        else None
        for f in ("keywords", "genres", "titles")
    }
    llm = {
        "model": expander.model,
        "queries": len(unique),
        "calls": len(live),
        "cache_hits": sources.count("cache"),
        "fallbacks": sum(fallbacks.values()),
        "fallback_reasons": fallbacks,
        "grounded_rate": float(np.mean([g["any"] for g in groundings])),
        "grounded_share": field_share,
        "input_tokens": int(sum(e.input_tokens for e in live)),
        "output_tokens": int(sum(e.output_tokens for e in live)),
        "cost_usd": round(cost, 4),
        "cost_per_1k_calls_usd": round(1000 * cost / len(live), 4) if live else None,
        "latency_ms": [round(float(x), 1) for x in np.percentile(latency, [50, 95])]
        if live
        else None,
    }
    expanded = {q: expansions[normalize_query(q)] for q in events["query"].to_list()}
    table = events.select(
        "userId", "query", "term", n_relevant=pl.col("relevant").list.len()
    ).with_columns(
        expansion_source=pl.Series([expanded[q].source for q in events["query"].to_list()]),
        grounded=pl.Series([g["any"] for g in groundings]),
        raw_terms=pl.Series([len(q) for q in raw_q]),
        tag_in_vocab=pl.Series(known),
    )
    for arm, metrics in per_event.items():
        table = table.with_columns(**{f"{arm}/{m}": pl.Series(v) for m, v in metrics.items()})
    table.write_parquet(out / "per_event.parquet")
    (out / "expansions.json").write_text(
        json.dumps(
            {q: {"keywords": e.keywords, "genres": e.genres, "titles": e.titles, "source": e.source}
             for q, e in expansions.items()},
            indent=1,
        ),
        encoding="utf-8",
    )  # fmt: skip
    summary: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "split_sha256": data.meta["split_sha256"],
            "n_events": int(events.height),
            "n_users": n_users,
            "n_queries": len(unique),
            "mean_relevant": float(np.diff(targets.indptr).mean()),
            "raw_query_matched": float(np.mean([bool(q) for q in raw_q])),
            "tag_in_vocab": float(np.mean(known)),
            "excluded_events": 0 if exclude is None else int(exclude.height),
        },
        "settings": {"weight": weight, "bm25": list(E008_PARAMS), "k": K, "seed": seed},
        "arms": arms,
        "paired": paired,
        "subgroups": subgroups,
        "llm": llm,
        "wall_s": round(time.perf_counter() - started, 1),
        "versions": {"shopsage_rec": __version__},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TagBenchmarkResult(out, summary)
