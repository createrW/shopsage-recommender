"""LLM query expansion with a fake client (no network), and the tag-as-query benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import numpy as np
import polars as pl
import pytest

from shopsage_rec.llm.benchmark import (
    expansion_query,
    run_tag_benchmark,
    tag_events,
    text_query,
)
from shopsage_rec.llm.expand import MODEL, Expansion, QueryExpander


class FakeClient:
    """Records calls; answers with a JSON text block, a refusal or an exception."""

    def __init__(self, payload: dict[str, Any] | None = None, *, stop: str = "end_turn",
                 error: Exception | None = None) -> None:  # fmt: skip
        self.payload = payload or {"keywords": ["space opera"], "genres": ["Sci-Fi"], "titles": []}
        self.stop = stop
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            stop_reason=self.stop,
            usage=SimpleNamespace(input_tokens=120, output_tokens=40),
        )


def test_expand_sends_a_structured_low_effort_request_and_caches(tmp_path: Path) -> None:
    client = FakeClient()
    expander = QueryExpander(client=client, cache_dir=tmp_path)

    first = expander.expand("  Space Battles ")
    second = expander.expand("space battles")

    assert first.keywords == ("space opera",)
    assert first.genres == ("Sci-Fi",)
    assert first.source == "llm"
    assert (first.input_tokens, first.output_tokens) == (120, 40)
    assert second.source == "cache"
    assert second.keywords == first.keywords
    assert len(client.calls) == 1  # the normalised query hits the cache
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k"} & set(call)


@pytest.mark.parametrize(
    ("client", "reason"),
    [
        (FakeClient(stop="refusal"), "refusal"),
        (FakeClient(stop="max_tokens"), "max_tokens"),
        (
            FakeClient(
                error=anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x"))
            ),
            "timeout",
        ),
        (FakeClient(payload={"keywords": "not a list"}), "invalid"),
    ],
)
def test_expand_falls_back_to_the_raw_query(
    tmp_path: Path, client: FakeClient, reason: str
) -> None:
    out = QueryExpander(client=client, cache_dir=tmp_path).expand("heist")

    assert out.source == f"fallback:{reason}"
    assert out.keywords == out.genres == out.titles == ()


def test_expand_without_a_key_never_calls_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    out = QueryExpander(cache_dir=tmp_path).expand("heist")

    assert out.source == "fallback:no_key"


def test_text_and_expansion_queries_ground_terms_in_the_vocabulary() -> None:
    vocab = {
        "w:space": 0,
        "w:battles": 1,
        "t:space battles": 2,
        "t:space opera": 3,
        "g:sci-fi": 4,
        "w:star": 5,
        "w:wars": 6,
    }
    titles = {"star wars": [7]}

    raw = text_query("Space Battles", vocab)
    assert raw == {0: 1.0, 1: 1.0, 2: 1.0}

    exp = Expansion(
        keywords=("space opera", "made-up phrase"),
        genres=("Sci-Fi",),
        titles=("Star Wars", "Not A Movie"),
        source="llm",
    )
    q, grounding = expansion_query("Space Battles", exp, vocab, titles, weight=0.5)
    assert q == {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.5, 4: 0.5, 5: 0.5, 6: 0.5}
    assert grounding == {"keywords": 0.5, "genres": 1.0, "titles": 0.5, "any": True}

    empty = Expansion(keywords=("zzz",), genres=(), titles=(), source="llm")
    q2, g2 = expansion_query("Space Battles", empty, vocab, titles, weight=0.5)
    assert q2 == raw  # nothing grounds: the raw query
    assert g2["any"] is False


def _write_heldout_tags(liang_dir: Path) -> None:
    users = pl.read_parquet(liang_dir / "mappings" / "user_map.parquet")
    held = users.filter(pl.col("role") != "train")["userId"].to_list()
    train = users.filter(pl.col("role") == "train")["userId"].to_list()
    items = (
        pl.read_parquet(liang_dir / "mappings" / "item_map.parquet")
        .sort("item_idx")["movieId"]
        .to_list()
    )
    rows = []
    for k, user in enumerate(held[:12]):
        rows += [
            (user, items[(3 * k) % 50], "Genre Tag A", k),
            (user, items[(3 * k + 1) % 50], "genre tag a", k),
        ]
        rows.append((user, items[(5 * k) % 80], f"tag {k % 7 + 1}", k))
    rows += [(train[0], items[0], "genre tag a", 1), (train[1], items[1], "genre tag a", 2)]
    pl.DataFrame(
        rows,
        schema={"userId": pl.Int32, "movieId": pl.Int32, "tag": pl.String, "timestamp": pl.Int64},
        orient="row",
    ).write_parquet(liang_dir / "tags.parquet")


def test_tag_events_come_from_held_out_users_only(liang_dir: Path) -> None:
    _write_heldout_tags(liang_dir)

    events = tag_events(liang_dir, n=1000, seed=0)

    roles = pl.read_parquet(liang_dir / "mappings" / "user_map.parquet")
    held = set(roles.filter(pl.col("role") != "train")["userId"].to_list())
    assert set(events["userId"].to_list()) <= held
    assert events.filter(pl.col("term") == "t:genre tag a")["relevant"].list.len().max() == 2
    assert events.height == events.unique(["userId", "term"]).height
    again = tag_events(liang_dir, n=1000, seed=1, exclude=events.head(5))
    assert again.join(events.head(5), on=["userId", "term"], how="semi").height == 0


def test_benchmark_compares_raw_and_expanded_queries(liang_dir: Path, tmp_path: Path) -> None:
    _write_heldout_tags(liang_dir)
    expander = QueryExpander(
        client=FakeClient({"keywords": ["tag 3"], "genres": ["Drama"], "titles": []}),
        cache_dir=tmp_path / "cache",
    )

    result = run_tag_benchmark(
        liang_dir, tmp_path / "out", expander, n_events=30, workers=2, run_id="t"
    )

    s = json.loads((result.out_dir / "summary.json").read_text())
    assert s == result.summary
    assert set(s["arms"]) == {"C0", "C1", "C1r"}
    assert set(s["paired"]) == {"C1 - C0", "C1r - C0", "C1r - C1"}
    for arm in ("C0", "C1"):
        hr = s["arms"][arm]["hr@50"]
        assert 0.0 <= hr["ci_low"] <= hr["mean"] <= hr["ci_high"] <= 1.0
    assert {"hr@10", "hr@50"} <= set(s["paired"]["C1 - C0"])
    assert (
        s["llm"]["calls"] + s["llm"]["cache_hits"] + s["llm"]["fallbacks"] == s["data"]["n_queries"]
    )
    assert 0.0 <= s["llm"]["grounded_rate"] <= 1.0
    assert s["llm"]["cost_usd"] >= 0.0
    per = pl.read_parquet(result.out_dir / "per_event.parquet")
    assert per.height == s["data"]["n_events"]
    assert np.isin(["C0/hr@50", "C1/hr@50", "expansion_source"], per.columns).all()
