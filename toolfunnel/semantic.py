"""Optional semantic re-rank hook.

The funnel is deterministic and lexical by default: no model, no network, no
embeddings. That is a feature -- it keeps the hot path cheap and reproducible.

But sometimes lexical recall misses a paraphrase ("my package never arrived" vs
a tool described as "logistics tracking"). This module defines a *narrow*
interface for plugging in an embedding model purely as a **re-ranker** over the
shortlist the lexica/chain layers already produced. It never sees the full
catalog, so cost stays bounded no matter how many tools exist.

If you have no embedder, do nothing: the funnel is already complete.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

from .types import ToolMatch

__all__ = ["Embedder", "SemanticRanker", "cosine", "make_openai_embedder"]

Embedder = Callable[[List[str]], List[Sequence[float]]]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class SemanticRanker:
    """Blend embedding similarity into the funnel's shortlist ranking.

    Usage::

        ranker = SemanticRanker(embedder=my_embed_fn)
        result = funnel.route(query)
        result.tools = ranker.rerank(query, result.tools, weight=0.4)

    Note the asymmetry: only the *shortlist* is embedded (a handful of tools),
    so the marginal token/dollar cost is tiny and predictable. The full 1000+
    catalog is never sent anywhere.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        *,
        cache: bool = True,
    ) -> None:
        if embedder is None:
            raise ValueError("SemanticRanker needs an embedder callable")
        self.embedder = embedder
        self.cache = cache
        self._tool_vecs: Dict[str, Sequence[float]] = {}

    def _tool_text(self, match: ToolMatch) -> str:
        t = match.tool
        return f"{t.name}. {t.description} {' '.join(t.keywords)} {t.domain} {t.stage or ''}".strip()

    def _embed_tools(self, matches: Sequence[ToolMatch]) -> Dict[str, Sequence[float]]:
        todo = [m for m in matches if not (self.cache and m.name in self._tool_vecs)]
        if todo:
            vecs = self.embedder([self._tool_text(m) for m in todo])
            for m, v in zip(todo, vecs):
                self._tool_vecs[m.name] = v
        return {m.name: self._tool_vecs[m.name] for m in matches}

    def rerank(
        self, query: str, matches: Sequence[ToolMatch], *, weight: float = 0.4
    ) -> List[ToolMatch]:
        if not matches:
            return list(matches)
        qv = self.embedder([query])[0]
        tvs = self._embed_tools(matches)
        max_base = max((m.score for m in matches), default=1.0) or 1.0
        out: List[ToolMatch] = []
        for m in matches:
            sim = cosine(qv, tvs[m.name])
            blended = (m.score / max_base) * (1 - weight) + sim * weight
            out.append(
                ToolMatch(
                    tool=m.tool,
                    score=round(blended, 4),
                    reason=(m.reason + "+semantic").lstrip("+"),
                    prefilled=dict(m.prefilled),
                )
            )
        out.sort(key=lambda x: (-x.score, x.name))
        return out


def make_openai_embedder(model: str = "text-embedding-3-small", **client_kwargs) -> Embedder:
    """Build an :data:`Embedder` from the OpenAI SDK (optional dependency).

    Kept out of the core import path so the library never requires ``openai``.
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(**client_kwargs)

    def _embed(texts: List[str]) -> List[Sequence[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return _embed
