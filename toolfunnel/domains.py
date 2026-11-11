"""Layer 1 -- business-domain gate.

The first funnel layer answers one question cheaply: *which business domains
does this turn plausibly belong to?* It never uses the model. It scores domains
against the query with:

* curated keyword/trigger phrase lists per domain (highest signal),
* an optional learned prior (which domains past sessions actually used for
  similar queries), and
* a lexical fallback over domain names when no curated trigger fires.

Output is a ranked list of :class:`DomainScore`. A small number of top domains
is kept; the rest are dropped. Because domains are *not* hard-isolated, the
gate also emits a set of ``bridge`` domains to keep reachable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .registry import ToolRegistry, tokenize
from .types import DomainScore

__all__ = ["DomainSpec", "DomainGate"]


@dataclass
class DomainSpec:
    """Declarative description of a business domain.

    ``triggers`` are strong phrases ("退款", "order", "采购"). ``stopwords``
    are terms that would otherwise misroute into this domain. ``bridges`` names
    sibling domains this one commonly hands off to.
    """

    name: str
    description: str = ""
    triggers: Tuple[str, ...] = ()
    stopwords: Tuple[str, ...] = ()
    bridges: Tuple[str, ...] = ()
    # Prior probability, nudges tie-breaks.
    prior: float = 1.0


class DomainGate:
    """Scores and selects candidate business domains for a query."""

    def __init__(
        self,
        specs: Optional[Iterable[DomainSpec]] = None,
        *,
        registry: Optional[ToolRegistry] = None,
        top_k: int = 2,
        min_score: float = 0.05,
    ) -> None:
        self.specs: Dict[str, DomainSpec] = {}
        self.top_k = top_k
        self.min_score = min_score
        self.registry = registry
        # Learned boost per (domain) and per (trigger-term -> domain) is
        # populated by the learner; see ``learn.py``.
        self.learned_prior: Dict[str, float] = {}
        self.learned_terms: Dict[str, Dict[str, float]] = {}
        if specs:
            for s in specs:
                self.add(s)

    def add(self, spec: DomainSpec) -> "DomainGate":
        self.specs[spec.name] = spec
        self.learned_terms.setdefault(spec.name, {})
        return self

    @property
    def names(self) -> List[str]:
        return list(self.specs)

    # ---- scoring ---------------------------------------------------------- #
    def _term_weights(self, spec: DomainSpec, terms: Sequence[str]) -> Tuple[float, List[str]]:
        score = 0.0
        hits: List[str] = []
        termset = set(terms)
        for trig in spec.triggers:
            t_terms = tokenize(trig)
            if not t_terms:
                continue
            # Multi-char CJK phrase hit if any of its bigrams matches.
            overlap = [t for t in t_terms if t in termset]
            if not overlap:
                continue
            # Full-phrase hits are worth more than partial overlap.
            frac = len(overlap) / len(t_terms)
            score += 1.4 * frac + 0.6
            hits.append(trig)
        return score, hits

    def score(self, query: str, *, extra_terms: Sequence[str] = ()) -> List[DomainScore]:
        terms = set(tokenize(query)) | set(extra_terms)
        results: List[DomainScore] = []
        for name, spec in self.specs.items():
            s, hits = self._term_weights(spec, list(terms))
            # Stopwords veto a trigger.
            if any(st in query for st in spec.stopwords):
                s *= 0.35
            # Learned prior and learned trigger evidence.
            prior = self.learned_prior.get(name, 0.0)
            s += prior
            for term in terms:
                s += self.learned_terms.get(name, {}).get(term, 0.0)
            # Lexical fallback so uncurated domains still work.
            if s <= 0 and self.registry is not None:
                dom_terms = set(tokenize(name)) | set(tokenize(spec.description))
                overlap = terms & dom_terms
                if overlap:
                    s += 0.25 * len(overlap)
            s *= math.log1p(max(spec.prior, 0.0))
            if s > 0:
                results.append(DomainScore(name, round(s, 4), hits))
        results.sort(key=lambda d: (-d.score, d.domain))
        return results

    def select(
        self, query: str, *, top_k: Optional[int] = None, extra_terms: Sequence[str] = ()
    ) -> List[DomainScore]:
        k = self.top_k if top_k is None else top_k
        scored = self.score(query, extra_terms=extra_terms)
        kept = [d for d in scored if d.score >= self.min_score][:k]
        if not kept:
            # Never return empty: fall back to the highest-prior domain so the
            # funnel always has somewhere to start. When *nothing* scored at
            # all, seed the fallback from every declared domain.
            pool = scored or [DomainScore(n, 0.0) for n in self.specs]
            pool.sort(key=lambda d: -self.specs[d.domain].prior)
            kept = pool[:1]
        return kept

    def bridges_for(self, domains: Sequence[str]) -> Set[str]:
        """Sibling domains reachable from the selected domains."""
        out: Set[str] = set()
        for d in domains:
            spec = self.specs.get(d)
            if spec:
                out.update(spec.bridges)
        return out - set(domains)

    def learn_domain(self, domain: str, weight: float) -> None:
        if domain in self.specs:
            self.learned_prior[domain] = self.learned_prior.get(domain, 0.0) + weight

    def learn_term(self, domain: str, term: str, weight: float) -> None:
        if domain in self.specs:
            bucket = self.learned_terms.setdefault(domain, {})
            bucket[term] = bucket.get(term, 0.0) + weight

    def export_state(self) -> Dict[str, object]:
        return {
            "learned_prior": dict(self.learned_prior),
            "learned_terms": {k: dict(v) for k, v in self.learned_terms.items()},
        }

    def load_state(self, state: Dict[str, object]) -> None:
        self.learned_prior = dict(state.get("learned_prior", {}))  # type: ignore[arg-type]
        self.learned_terms = {
            k: dict(v) for k, v in dict(state.get("learned_terms", {})).items()  # type: ignore[arg-type]
        }
