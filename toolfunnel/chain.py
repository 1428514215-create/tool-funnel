"""Chain graph -- the tool-to-tool recommendation network.

This is the memory that makes the funnel *get faster over time*. Every time the
model actually calls tool ``A`` and then tool ``B`` in the same session, we add
weight to the edge ``A -> B``. Later, after the model calls ``A``, the funnel
suggests the highest-weight successors of ``A`` -- pre-surfaced, sometimes with
arguments pre-filled from what ``A`` returned.

Design choices that matter:

* **Time-decayed counts.** Edge weights use exponential decay so the graph
  tracks *recent* behaviour rather than being frozen by ancient history.
* **Domain-aware, not domain-bound.** Edges may cross domains; a cross-domain
  edge is a first-class signal (it is how "order" hands off to "logistics").
  Cross-domain edges are damped slightly so they do not drown the local chain.
* **Success-weighted.** Edges from calls that errored contribute less, so the
  graph converges on chains that actually worked.
* **Serializable.** The whole graph is plain JSON-able data: dump it, ship it
  in a repo, load it in another process.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import ToolCall

__all__ = ["ChainGraph", "ChainEdge", "ChainStep"]

_DEFAULT_RNG = random.Random()
_DEFAULT_RNG_MISSING = None

# Strength of a *declared* ``next_hint`` as a prior, before any observation.
#
# On the saturating scale an observed edge scores ``n/(n+2)`` after ``n``
# sightings: 1 -> 0.33, 3 -> 0.60, 5 -> 0.71, many -> ~1.0. A prior of 0.7
# therefore means a handful of calls (fewer than ~5) cannot dislodge a curated
# hint, while *sustained* consistent traffic (~5+ sightings of the same edge)
# legitimately overrides it -- because at that point the traffic is evidence the
# curated chain was wrong. See ``successors`` for the combination rule.
_HINT_PRIOR = 0.7


def _default_random() -> float:
    return _DEFAULT_RNG.random()


@dataclass
class ChainEdge:
    """A directed, weighted ``src -> dst`` transition."""

    src: str
    dst: str
    weight: float = 0.0
    count: int = 0
    last_seen: float = field(default_factory=time.time)
    cross_domain: bool = False
    # Argument keys that ``src`` produced and ``dst`` commonly consumes; these
    # become pre-fill candidates when the chain is suggested.
    arg_flow: Tuple[str, ...] = ()
    # Sum of success-weighted rewards, kept separately from ``count`` so the
    # exploration policy can estimate this arm's success rate.
    success: float = 0.0

    def rate(self) -> float:
        """Success-weighted mean reward of this transition (0..1)."""
        if self.count <= 0:
            return 0.0
        return self.success / self.count

    def decay(self, half_life_days: float = 30.0, now: Optional[float] = None) -> float:
        now = now or time.time()
        age_days = max(0.0, (now - self.last_seen) / 86400.0)
        return math.pow(0.5, age_days / half_life_days)


@dataclass
class ChainStep:
    """One recommended next step produced by the graph."""

    tool: str
    score: float = 0.0
    reason: str = ""
    arg_flow: Tuple[str, ...] = ()
    # How many times this transition was observed. Exposed so callers can
    # implement their own exploration policy on top of the exploit ranking.
    count: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        return f"ChainStep({self.tool!r}, {self.score:.3f})"


class ChainGraph:
    """Weighted directed graph over tool names, with decayed statistics."""

    def __init__(self, *, half_life_days: float = 30.0, cross_domain_damping: float = 0.75) -> None:
        self.edges: Dict[Tuple[str, str], ChainEdge] = {}
        self.in_degree: Dict[str, float] = {}
        self.half_life_days = half_life_days
        self.cross_domain_damping = cross_domain_damping
        self._domain_of: Dict[str, str] = {}
        # Seed hints declared statically on ToolSpec.next_hint. They act as a
        # cold-start prior: before any real traffic the funnel still has a
        # sensible chain.
        self._seed: Dict[str, List[str]] = {}
        self._observed_calls: int = 0
        self._observed_sessions: int = 0

    # ---- ingestion -------------------------------------------------------- #
    def register_domains(self, mapping: Dict[str, str]) -> None:
        self._domain_of.update(mapping)

    def seed(self, src: str, targets: Iterable[str]) -> None:
        self._seed.setdefault(src, [])
        for t in targets:
            if t not in self._seed[src]:
                self._seed[src].append(t)

    def observe_session(self, calls: Sequence[ToolCall], *, domain_of: Optional[Dict[str, str]] = None) -> None:
        """Learn from one session's ordered list of tool calls."""
        if domain_of:
            self._domain_of.update(domain_of)
        self._observed_sessions += 1
        for call in calls:
            self._observed_calls += 1
        for a, b in zip(calls, calls[1:]):
            self.observe_pair(a, b)

    def observe_pair(self, a: ToolCall, b: ToolCall) -> None:
        """Increment the ``a.tool -> b.tool`` edge."""
        key = (a.tool, b.tool)
        edge = self.edges.get(key)
        cross = self._is_cross_domain(a.tool, b.tool)
        if edge is None:
            edge = ChainEdge(src=a.tool, dst=b.tool, cross_domain=cross)
            self.edges[key] = edge
        # Success-weighted reward: failed calls still inform the topology but
        # count for less, so the graph prefers chains that worked.
        reward = 1.0 if a.ok and b.ok else (0.4 if b.ok else 0.15)
        edge.count += 1
        edge.success += reward
        edge.weight += reward
        edge.last_seen = time.time()
        if cross:
            # Record which produced keys plausibly feed the callee.
            flow = tuple(sorted(set(a.produced) & set(b.args)))
            if flow:
                edge.arg_flow = tuple(sorted(set(edge.arg_flow) | set(flow)))
        self.in_degree[b.tool] = self.in_degree.get(b.tool, 0.0) + reward

    def _is_cross_domain(self, a: str, b: str) -> bool:
        da, db = self._domain_of.get(a), self._domain_of.get(b)
        return bool(da and db and da != db)

    # ---- recommendation --------------------------------------------------- #
    def successors(
        self,
        tool: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
        now: Optional[float] = None,
    ) -> List[ChainStep]:
        """Rank the most likely next tools after ``tool`` was called.

        Blends three sources so that *both* cold-start and warmed-up behaviour
        are sensible:

        1. observed edges (time-decayed, success- and cross-domain-weighted),
        2. static seed hints (declare once, always available), and
        3. nothing else -- deliberately lexical-free so the chain stays cheap.
        """
        now = now or time.time()
        acc: Dict[str, Tuple[float, str, Tuple[str, ...]]] = {}

        for (src, dst), edge in self.edges.items():
            if src != tool:
                continue
            w = edge.weight * edge.decay(self.half_life_days, now)
            if edge.cross_domain:
                w *= self.cross_domain_damping
            cur = acc.get(dst)
            reason = "observed" + ("(cross-domain)" if edge.cross_domain else "")
            if cur is None or w > cur[0]:
                acc[dst] = (w, reason, edge.arg_flow)

        # Map observed weights onto a bounded 0..1 band with a saturating curve
        # ``w / (w + 2)``: 1 observation -> 0.33, 2 -> 0.5, 4 -> 0.67, many ->
        # ~1.0.
        acc = {k: (v[0] / (v[0] + 2.0), v[1], v[2]) for k, v in acc.items()}

        for dst in self._seed.get(tool, []):
            # A declared hint is a *prior*, and observed traffic is *evidence*.
            # They must reinforce each other, not compete: previously a hint was
            # only floored at 0.6, so any observed edge past ~4 sightings (0.67)
            # pushed the curated chain out of the top slot -- learning silently
            # degraded hand-authored routes. Combining them probabilistically
            # (``1 - (1-prior)(1-observed)``) means an observed edge can only
            # displace a hint by being genuinely consistent, and a hinted edge
            # that is also observed becomes stronger still.
            obs = acc.get(dst)
            observed_score = obs[0] if obs else 0.0
            prior = _HINT_PRIOR
            combined = 1.0 - (1.0 - prior) * (1.0 - observed_score)
            acc[dst] = (
                min(1.0, combined),
                obs[1] if obs else "seed-hint",
                obs[2] if obs else (),
            )

        steps = [
            ChainStep(dst, round(w, 4), reason, flow, self._count_of(tool, dst))
            for dst, (w, reason, flow) in acc.items()
            if w >= min_score
        ]
        steps.sort(key=lambda s: (-s.score, s.tool))
        return steps[:limit]

    def _count_of(self, src: str, dst: str) -> int:
        edge = self.edges.get((src, dst))
        return edge.count if edge else 0

    def explore(
        self,
        tool: str,
        *,
        limit: int = 5,
        epsilon: float = 0.15,
        rng: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> List[ChainStep]:
        """Return successors with a small exploration budget mixed in.

        Pure exploitation ("always follow the heaviest edge") locks the graph
        onto whatever it learned first: a shorter or better chain that is tried
        only occasionally never accumulates enough weight to overtake the
        incumbent. This introduces an *exploration* term using the upper
        confidence bound of each successor's success rate -- arms with few
        observations get a bonus proportional to how uncertain they are, so
        they are periodically re-tried and either graduate or fade.

        Concretely the returned score blends:

        * the exploit score from :meth:`successors` (weight ``1 - epsilon``),
        * a UCB exploration term ``sqrt(2 ln(total) / count)`` (weight
          ``epsilon``), which is large exactly for rarely-observed arms.

        Set ``epsilon=0`` to get pure exploitation (identical ranking to
        :meth:`successors`). ``rng`` accepts any object with ``random()`` (e.g.
        ``random.Random(seed)``) for reproducibility.
        """
        base = self.successors(tool, limit=max(limit, 5), now=now)
        if epsilon <= 0 or not base:
            return base[:limit]

        total = sum(s.count for s in base) or 1
        log_total = math.log1p(total)
        rand = rng.random() if rng is not None else _default_random()

        boosted: List[ChainStep] = []
        for s in base:
            # UCB1-style bonus: unexplored arms score highest.
            bonus = math.sqrt(2.0 * log_total / (s.count + 1.0))
            score = (1.0 - epsilon) * s.score + epsilon * bonus
            boosted.append(
                ChainStep(
                    s.tool, round(score, 4),
                    s.reason + "+explore" if s.count == 0 else s.reason,
                    s.arg_flow, s.count,
                )
            )
        boosted.sort(key=lambda s: (-s.score, s.tool))

        # When exploring, with probability ``epsilon`` swap one rarely-observed
        # arm into the visible set so it actually gets called (and thus a chance
        # to earn observations). Pure re-ranking alone would not surface it.
        if rand < epsilon:
            rare = [s for s in boosted if s.count == 0 and s not in boosted[:limit]]
            if rare:
                pick = rare[0]
                out = [s for s in boosted[:limit] if s.tool != pick.tool]
                out.append(pick)
                out.sort(key=lambda s: (-s.score, s.tool))
                return out[:limit]
        return boosted[:limit]

    def ucb_score(self, src: str, dst: str, *, now: Optional[float] = None) -> float:
        """Expose the exploit+explore score for one edge (used in tests/stats)."""
        steps = {s.tool: s for s in self.explore(src, limit=1000, now=now)}
        return steps.get(dst, ChainStep(dst)).score

    def chains(self, start: str, *, max_len: int = 3, width: int = 3) -> List[List[ChainStep]]:
        """Enumerate probable multi-step chains from ``start`` (beam search).

        Useful for *prefetching*: if the model is very likely to call
        ``A -> B -> C``, the backend can warm caches or pre-fetch data for B and
        C while the model is still thinking.
        """
        results: List[List[ChainStep]] = []
        frontier: List[Tuple[List[ChainStep], str, set]] = [([], start, {start})]
        for _ in range(max_len):
            nxt: List[Tuple[List[ChainStep], str, set]] = []
            for path, node, seen in frontier:
                for step in self.successors(node, limit=width):
                    if step.tool in seen:
                        continue
                    new_path = path + [step]
                    results.append(new_path)
                    nxt.append((new_path, step.tool, seen | {step.tool}))
            frontier = nxt
            if not frontier:
                break
        results.sort(key=lambda p: -sum(s.score for s in p))
        return results

    # ---- introspection / persistence ------------------------------------- #
    @property
    def stats(self) -> Dict[str, object]:
        return {
            "edges": len(self.edges),
            "observed_calls": self._observed_calls,
            "observed_sessions": self._observed_sessions,
            "cross_domain_edges": sum(1 for e in self.edges.values() if e.cross_domain),
        }

    def top_edges(self, n: int = 10) -> List[ChainEdge]:
        return sorted(self.edges.values(), key=lambda e: -e.weight)[:n]

    # ---- maintenance ------------------------------------------------------ #
    def distill(
        self,
        *,
        min_weight: float = 0.5,
        min_count: int = 1,
        keep_per_source: int = 8,
    ) -> Dict[str, int]:
        """Prune weak edges so the graph stays small and fast over time.

        Learned edges only accumulate: without maintenance a long-running funnel
        grows one edge per observed transition forever, and ``successors()`` --
        which scans every edge -- slowly turns from an optimisation into a cost.
        This keeps the graph bounded.

        An edge is dropped only when it is weak on *both* axes: below
        ``min_weight`` **and** seen no more than ``min_count`` times. Requiring
        both keeps a single fresh observation alive (it may be a new signal
        worth following) while sweeping away stale one-off noise. Additionally,
        any tool keeps at most ``keep_per_source`` successors, since the tail
        beyond that is never recommended anyway (``successors`` returns only a
        handful).

        Returns a report ``{edges_before, edges_after, removed, capped}``.

        This is lossy and irreversible for the dropped edges -- call it
        deliberately (e.g. on a schedule), not on every write.
        """
        before = len(self.edges)

        # 1) Drop edges that are *both* below the weight floor and unproven
        #    (seen no more than ``min_count`` times). An edge needs to be weak
        #    on BOTH axes to go, so a brand-new single observation is not
        #    discarded the moment it is learned.
        survive: Dict[Tuple[str, str], ChainEdge] = {}
        for key, edge in self.edges.items():
            if edge.weight < min_weight and edge.count <= min_count:
                continue
            survive[key] = edge

        # 2) Per-source cap: keep the strongest ``keep_per_source`` by weight,
        #    but never drop an edge proven by more than ``min_count`` sightings
        #    -- those may simply be ranked below noisier neighbours yet still be
        #    genuinely useful.
        by_src: Dict[str, List[ChainEdge]] = {}
        for edge in survive.values():
            by_src.setdefault(edge.src, []).append(edge)
        capped = 0
        final: Dict[Tuple[str, str], ChainEdge] = {}
        for src, edges in by_src.items():
            edges.sort(key=lambda e: (-e.weight, e.dst))
            for i, e in enumerate(edges):
                if i >= keep_per_source and e.count <= min_count:
                    capped += 1
                    continue
                final[(e.src, e.dst)] = e

        self.edges = final
        # Rebuild in-degree from scratch so it matches the pruned graph.
        self.in_degree = {}
        for edge in self.edges.values():
            self.in_degree[edge.dst] = self.in_degree.get(edge.dst, 0.0) + edge.weight

        return {
            "edges_before": before,
            "edges_after": len(self.edges),
            "removed": before - len(self.edges),
            "capped": capped,
        }

    def needs_distill(self, *, max_edges: int = 20000) -> bool:
        """Cheap guard so callers only pay for maintenance when it matters."""
        return len(self.edges) > max_edges

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "half_life_days": self.half_life_days,
            "cross_domain_damping": self.cross_domain_damping,
            "domain_of": dict(self._domain_of),
            "seed": {k: list(v) for k, v in self._seed.items()},
            "observed_calls": self._observed_calls,
            "observed_sessions": self._observed_sessions,
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "weight": e.weight,
                    "count": e.count,
                    "success": e.success,
                    "last_seen": e.last_seen,
                    "cross_domain": e.cross_domain,
                    "arg_flow": list(e.arg_flow),
                }
                for e in self.edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ChainGraph":
        g = cls(
            half_life_days=float(data.get("half_life_days", 30.0)),
            cross_domain_damping=float(data.get("cross_domain_damping", 0.75)),
        )
        g._domain_of = dict(data.get("domain_of", {}))  # type: ignore[arg-type]
        g._seed = {k: list(v) for k, v in dict(data.get("seed", {})).items()}  # type: ignore[arg-type]
        g._observed_calls = int(data.get("observed_calls", 0))
        g._observed_sessions = int(data.get("observed_sessions", 0))
        for e in data.get("edges", []):  # type: ignore[union-attr]
            edge = ChainEdge(
                src=e["src"],
                dst=e["dst"],
                weight=float(e.get("weight", 0.0)),
                count=int(e.get("count", 0)),
                last_seen=float(e.get("last_seen", time.time())),
                cross_domain=bool(e.get("cross_domain", False)),
                arg_flow=tuple(e.get("arg_flow", ())),
                success=float(e.get("success", e.get("weight", 0.0))),
            )
            g.edges[(edge.src, edge.dst)] = edge
        # Rebuild in-degree.
        for edge in g.edges.values():
            g.in_degree[edge.dst] = g.in_degree.get(edge.dst, 0.0) + edge.weight
        return g

    def dumps(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def loads(cls, text: str) -> "ChainGraph":
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.dumps(indent=2))

    @classmethod
    def load(cls, path: str) -> "ChainGraph":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.loads(fh.read())
