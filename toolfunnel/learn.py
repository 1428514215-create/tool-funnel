"""Offline learning: turn recorded sessions into a warmer funnel.

The runtime hook (:meth:`Funnel.record`) learns *as calls happen*. This module
does the heavier, batch version: given a corpus of past sessions it rebuilds the
chain graph, mines domain trigger terms, and reports what improved.

Two concrete pieces of learning:

1. **Chain graph rebuild** -- replay every session's call order with success
   weighting and time decay.
2. **Trigger mining** -- for each session we know the *query* and the *domain
   that ended up being used*. Terms that repeatedly precede a domain become
   learned triggers for that domain, so the layer-1 gate gets sharper with use
   without anyone curating keywords. This is a simple association-rule miner
   with Laplace smoothing, not a magic model.

A third, optional hook lets you plug an embedding model (``semantic.py``) for a
final re-rank; if absent, everything still works.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .chain import ChainGraph
from .funnel import Funnel
from .registry import tokenize
from .types import ToolCall

__all__ = ["SessionRecord", "Learner", "LearningReport"]


@dataclass
class SessionRecord:
    """One observed session: the query plus the ordered calls it produced."""

    query: str
    calls: List[ToolCall] = field(default_factory=list)
    domains_used: Tuple[str, ...] = ()
    success: bool = True


@dataclass
class LearningReport:
    sessions: int = 0
    calls: int = 0
    edges_before: int = 0
    edges_after: int = 0
    triggers_added: int = 0
    domains_reinforced: List[str] = field(default_factory=list)
    top_new_edges: List[Tuple[str, str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"learned from {self.sessions} sessions / {self.calls} calls; "
            f"chain edges {self.edges_before}->{self.edges_after}; "
            f"triggers+{self.triggers_added}; "
            f"top new edge: "
            + (
                f"{self.top_new_edges[0][0]}->{self.top_new_edges[0][1]}"
                if self.top_new_edges
                else "-"
            )
        )


class Learner:
    """Batch trainer over recorded sessions."""

    def __init__(
        self,
        funnel: Funnel,
        *,
        min_support: int = 2,
        min_confidence: float = 0.35,
        term_boost: float = 0.35,
    ) -> None:
        self.funnel = funnel
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.term_boost = term_boost

    def learn(self, sessions: Sequence[SessionRecord], *, rebuild: bool = True) -> LearningReport:
        report = LearningReport(
            sessions=len(sessions),
            calls=sum(len(s.calls) for s in sessions),
            edges_before=len(self.funnel.chain.edges),
        )

        if rebuild:
            # Replay all sessions with fresh decayed stats.
            self.funnel.chain = ChainGraph(
                half_life_days=self.funnel.chain.half_life_days,
                cross_domain_damping=self.funnel.chain.cross_domain_damping,
            )
            self.funnel.chain.register_domains(
                {t.name: t.domain for t in self.funnel.registry}
            )
            for t in self.funnel.registry:
                if t.next_hint:
                    self.funnel.chain.seed(t.name, t.next_hint)
            for s in sessions:
                self.funnel.chain.observe_session(
                    s.calls,
                    domain_of={t.name: t.domain for t in self.funnel.registry},
                )

        self._mine_triggers(sessions, report)
        self._reinforce_domains(sessions, report)

        report.edges_after = len(self.funnel.chain.edges)
        report.top_new_edges = [
            (e.src, e.dst, round(e.weight, 2))
            for e in self.funnel.chain.top_edges(5)
        ]
        return report

    # ------------------------------------------------------------------ #
    def _mine_triggers(self, sessions: Sequence[SessionRecord], report: LearningReport) -> None:
        """Associate query terms with the domain they reliably lead to.

        Only the session's *entry* domain is credited. A session that starts in
        ``order`` and later hands off to ``logistics`` must not teach the router
        that "订单" also means logistics -- that would blur exactly the signal
        layer 1 exists to sharpen.
        """
        term_domain: Dict[str, Counter] = defaultdict(Counter)
        domain_total: Counter = Counter()
        for s in sessions:
            if not s.success or not s.domains_used:
                continue
            primary = s.domains_used[0]
            if primary not in self.funnel.gate.specs:
                continue
            domain_total[primary] += 1
            for term in set(tokenize(s.query)):
                term_domain[term][primary] += 1

        added = 0
        for term, counts in term_domain.items():
            for d, c in counts.items():
                if c < self.min_support:
                    continue
                # Confidence = P(domain | term) among sessions mentioning it.
                mentions = sum(term_domain[term].values())
                confidence = c / max(1, mentions)
                if confidence < self.min_confidence:
                    continue
                # Laplace-smoothed, and capped so one term can never override
                # the curated triggers.
                boost = min(0.5, self.term_boost * confidence)
                self.funnel.gate.learn_term(d, term, boost)
                added += 1
        report.triggers_added = added

    def _reinforce_domains(self, sessions: Sequence[SessionRecord], report: LearningReport) -> None:
        """Nudge the domain prior toward domains that sessions actually *start* in."""
        hit: Counter = Counter()
        for s in sessions:
            if s.domains_used:
                hit[s.domains_used[0]] += 1
        for d, c in hit.items():
            if d in self.funnel.gate.specs:
                # Square-root scaling keeps big domains from steamrolling, and
                # the coefficient is small so priors only break near-ties.
                self.funnel.gate.learn_domain(d, 0.02 * (c ** 0.5))
        report.domains_reinforced = list(hit)

    # ------------------------------------------------------------------ #
    @staticmethod
    def synthetic_sessions(
        catalog_tools: Sequence,
        *,
        n: int = 200,
        seed: int = 0,
        jitter: float = 0.0,
    ) -> List[SessionRecord]:
        """Generate plausible sessions from a tool catalog.

        Handy for demos and for bootstrapping the chain graph before real
        traffic exists. Chains are sampled by following ``next_hint`` edges,
        which is exactly the structure the graph is meant to learn.

        ``jitter`` (0..1) injects *undeclared* cross-domain jumps: instead of
        following a declared hint, the walk hops to the entry tool of another
        domain. Those transitions are invisible to the seed hints, so they can
        only be predicted after the graph has observed them -- which is what
        makes a train/eval benchmark meaningful rather than tautological.
        """
        import random

        rng = random.Random(seed)
        by_name = {getattr(t, "name", t): t for t in catalog_tools}
        starts = [
            t for t in catalog_tools
            if not any(getattr(t, "name", "") in getattr(o, "next_hint", ()) for o in catalog_tools)
        ] or list(catalog_tools)
        # Entry tools per domain, for the jitter hops.
        entries: Dict[str, List] = defaultdict(list)
        for t in catalog_tools:
            if not any(getattr(t, "name", "") in getattr(o, "next_hint", ()) for o in catalog_tools):
                entries[getattr(t, "domain", "general")].append(t)

        sessions: List[SessionRecord] = []
        for _ in range(n):
            node = rng.choice(starts)
            calls: List[ToolCall] = []
            domains = []
            query_terms: List[str] = []
            for _step in range(rng.randint(1, 4)):
                name = getattr(node, "name", str(node))
                dom = getattr(node, "domain", "general")
                domains.append(dom)
                # Build a query out of the tool's own keywords, so the learner
                # sees a realistic term -> domain association rather than noise.
                kws = list(getattr(node, "keywords", ()))
                if kws:
                    query_terms.append(rng.choice(kws))
                ok = rng.random() > 0.08
                calls.append(
                    ToolCall(
                        tool=name,
                        ok=ok,
                        produced={k: f"<{k}>" for k in getattr(node, "produces", ())},
                    )
                )
                hints = list(getattr(node, "next_hint", ()))
                if jitter and rng.random() < jitter and entries:
                    # Deterministic handoff: ``dom`` always hands off to the
                    # same sibling (chosen by sorted domain order, skipping
                    # itself). Random hops would smear mass across many targets
                    # and never become learnable; a stable handoff is exactly
                    # the "these two domains always work together" pattern the
                    # graph is supposed to discover.
                    others = sorted(d for d in entries if d != dom)
                    if others:
                        # Stable index (``hash()`` on str is salted per process).
                        idx = sum(ord(c) for c in dom) % len(others)
                        # Always hop to the *same* canonical entry tool, so the
                        # (src -> dst) pair recurs and the graph can actually
                        # learn it instead of seeing a different pair each time.
                        node = sorted(entries[others[idx]], key=lambda t: getattr(t, "name", ""))[0]
                        continue
                if not hints or rng.random() < 0.35:
                    break
                node = by_name.get(rng.choice(hints), node)
            query = " ".join(query_terms) or f"{domains[0]} 处理"
            sessions.append(
                SessionRecord(
                    query=query,
                    calls=calls,
                    domains_used=tuple(dict.fromkeys(domains)),
                    success=all(c.ok for c in calls),
                )
            )
        return sessions
