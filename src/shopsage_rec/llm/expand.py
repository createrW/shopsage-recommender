"""Query expansion with Claude Haiku (D-010): a short movie-search query becomes descriptive
keywords, MovieLens genres and well-known titles, which ``llm/benchmark.py`` grounds in the BM25
vocabulary.

One ``messages.create`` call per new query: ``claude-haiku-5-5``, structured output
(``output_config.format``, a JSON schema), effort ``low`` (Haiku 5.5's default is ``medium``),
no sampling parameters (rejected by the API since 2026-10-21). Responses are cached on disk by
sha256 of the model id, the prompt and the request parameters, so a rerun costs nothing.

The raw query is always the fallback, and the reason is recorded in ``Expansion.source``:
``fallback:no_key`` (no ``ANTHROPIC_API_KEY`` and no client given), ``refusal``, ``max_tokens``,
``timeout``, ``api_error`` and ``invalid`` (a response that does not match the schema).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shopsage_rec.features.items import GENRES

MODEL = "claude-haiku-5-5"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}  # claude-haiku-5-5, prompts up to 100K tokens
MAX_TOKENS = 2048
MAX_KEYWORDS = 12
MAX_TITLES = 8
SYSTEM = (
    "You expand short movie-search queries for a keyword search engine over the MovieLens "
    "catalogue (films up to 2015). Given a query, return: keywords: up to 12 short lowercase tags "
    "a viewer might attach to matching movies (themes, moods, settings, styles, people), "
    "including close synonyms of the query; genres: the MovieLens genres that fit; titles: up "
    "to 8 well-known matching movies released before 2015, as their usual English title without "
    "the year. Use only what the query supports; return empty lists rather than guess."
)
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "genres": {"type": "array", "items": {"type": "string", "enum": list(GENRES)}},
        "titles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["keywords", "genres", "titles"],
    "additionalProperties": False,
}
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Expansion:
    keywords: tuple[str, ...] = ()
    genres: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()
    source: str = "llm"  # "llm", "cache" or "fallback:<reason>"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
        ) / 1e6


def normalize_query(query: str) -> str:
    return _SPACES.sub(" ", query).strip().lower()


def _parse(text: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out = []
    for key, limit in (("keywords", MAX_KEYWORDS), ("genres", len(GENRES)), ("titles", MAX_TITLES)):
        values = data.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            return None
        out.append(tuple(v.strip() for v in values if v.strip())[:limit])
    if any(g not in GENRES for g in out[1]):
        return None
    return out[0], out[1], out[2]


class QueryExpander:
    def __init__(
        self,
        client: Any | None = None,
        *,
        cache_dir: str | Path,
        model: str = MODEL,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if client is None and os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic

            client = anthropic.Anthropic(timeout=timeout, max_retries=2)
        self.client = client

    def _request(self, query: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": f"Query: {query}"}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        }

    def _key(self, request: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def expand(self, query: str) -> Expansion:
        query = normalize_query(query)
        request = self._request(query)
        path = self.cache_dir / f"{self._key(request)}.json"
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return Expansion(
                keywords=tuple(cached["keywords"]),
                genres=tuple(cached["genres"]),
                titles=tuple(cached["titles"]),
                source="cache",
            )
        if self.client is None:
            return Expansion(source="fallback:no_key")
        import anthropic

        started = time.perf_counter()
        try:
            response = self.client.messages.create(**request)
        except anthropic.APITimeoutError:
            return Expansion(source="fallback:timeout", latency_s=time.perf_counter() - started)
        except anthropic.APIError:
            return Expansion(source="fallback:api_error", latency_s=time.perf_counter() - started)
        latency = time.perf_counter() - started
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        if response.stop_reason in ("refusal", "max_tokens"):
            return Expansion(source=f"fallback:{response.stop_reason}", latency_s=latency, **usage)
        text = next((b.text for b in response.content if b.type == "text"), "")
        parsed = _parse(text)
        if parsed is None:
            return Expansion(source="fallback:invalid", latency_s=latency, **usage)
        out = Expansion(*parsed, source="llm", latency_s=latency, **usage)
        record = {k: v for k, v in asdict(out).items() if k in ("keywords", "genres", "titles")}
        path.write_text(
            json.dumps({"query": query, "model": self.model, **record}), encoding="utf-8"
        )
        return out
