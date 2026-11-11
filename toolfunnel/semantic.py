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
from typing import Any, Callable, Dict, List, Optional, Sequence

from .types import DomainScore, ToolMatch

__all__ = [
    "Embedder",
    "SemanticRanker",
    "SemanticDomainGate",
    "cosine",
    "make_openai_embedder",
]

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


class SemanticDomainGate:
    """Semantic fallback for layer 1, for queries no trigger word matches.

    The keyword domain gate is fast and explainable, but it fails on paraphrase:
    "我的包裹怎么还没到" contains no "物流" trigger, yet clearly belongs to the
    logistics domain. This class embeds the *domains* (a handful of them, so the
    cost is O(#domains) and stable no matter how many tools exist) and compares
    a query against them, filling in only when the lexical gate is unsure.

    It is a *fallback*, not a replacement: it runs only when the keyword scores
    are weak, so the common case stays free and deterministic.

    Usage::

        gate = SemanticDomainGate(gate=my_domain_gate, embedder=embed_fn)
        scores = gate.select_or_fallback(query)

    ``min_lexical_score`` is the threshold below which the semantic pass kicks
    in. Domains are embedded once and cached.
    """

    def __init__(
        self,
        *,
        gate: Any,
        embedder: Embedder,
        min_lexical_score: float = 0.6,
    ) -> None:
        self.gate = gate
        self.embedder = embedder
        self.min_lexical_score = min_lexical_score
        self._domain_vecs: Dict[str, Sequence[float]] = {}

    def _domain_text(self, name: str) -> str:
        spec = getattr(self.gate, "specs", {}).get(name)
        parts = [name]
        if spec is not None:
            parts.append(getattr(spec, "description", "") or "")
            parts.extend(getattr(spec, "triggers", ()) or ())
        return " ".join(p for p in parts if p).strip()

    def _ensure_domain_vecs(self) -> None:
        names = list(getattr(self.gate, "specs", {}).keys())
        todo = [n for n in names if n not in self._domain_vecs]
        if todo:
            vecs = self.embedder([self._domain_text(n) for n in todo])
            for n, v in zip(todo, vecs):
                self._domain_vecs[n] = v

    def needs_semantics(self, query: str) -> bool:
        """True when the lexical gate is too weak to trust."""
        scored = self.gate.score(query)
        if not scored:
            return True
        return scored[0].score < self.min_lexical_score

    def select_or_fallback(self, query: str, **kwargs: Any) -> List[DomainScore]:
        """Return lexical domain scores, adding semantic ones when needed.

        The semantic scores are merged *additively* and then re-ranked, so a
        strong keyword hit can never be displaced by a merely-plausible
        embedding match -- the fallback only fills the vacuum the keywords left.
        """
        lexical = self.gate.score(query)
        if not self.needs_semantics(query):
            return self.gate.select(query, **kwargs)

        self._ensure_domain_vecs()
        if not self._domain_vecs:
            return self.gate.select(query, **kwargs)

        qv = self.embedder([query])[0]
        by_domain = {d.domain: d for d in lexical}
        for name, dv in self._domain_vecs.items():
            sim = cosine(qv, dv)
            # Map cosine (-1..1) into a non-negative boost, scaled small so it
            # only matters in the low-lexical regime where we ran at all.
            boost = max(0.0, sim) * 0.8
            existing = by_domain.get(name)
            if existing is None:
                if boost > 0:
                    by_domain[name] = DomainScore(name, round(boost, 4), ["semantic"])
            else:
                existing.score = round(existing.score + boost, 4)
                existing.matched_terms = list(existing.matched_terms) + ["semantic"]

        merged = sorted(by_domain.values(), key=lambda d: (-d.score, d.domain))
        return self.gate.select(query, **kwargs) if not merged else _top(merged, self.gate, **kwargs)


def _top(scored: List[DomainScore], gate: Any, **kwargs: Any) -> List[DomainScore]:
    """Apply the gate's top-k / min-score policy to a merged score list."""
    k = kwargs.get("top_k", gate.top_k)
    kept = [d for d in scored if d.score >= gate.min_score][:k]
    if not kept:
        pool = scored or [DomainScore(n, 0.0) for n in gate.specs]
        pool.sort(key=lambda d: -gate.specs[d.domain].prior)
        kept = pool[:1]
    return kept
