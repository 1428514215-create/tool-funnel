"""Stage auto-discovery -- infer workflow phases from observed call sequences.

The funnel's layer 2 needs to know that a domain has phases
(``locate -> inspect -> mutate -> verify``) so it can show only the relevant
tools. Until now those phases had to be declared by hand, which is the main
thing standing between this library and a purely self-configuring one.

This module infers them from data. The signal we exploit: within a domain, tools
that appear at a *similar position* in real sessions belong to the same phase.
Concretely:

1. Replay many sessions and record, for each tool, its normalised position
   (0 = first call of the session, 1 = last).
2. Cluster the tools by that position distribution -- phase 0 tends to be
   "search/locate", the middle "inspect/mutate", the end "verify/close".
3. Emit :class:`~toolfunnel.stages.StageSpec` objects, one per cluster, with
   triggers mined from the tools' own keywords.

The clustering is 1-D k-means over the mean position (with a deterministic
seeding), which is appropriate because "phase" really is one ordered dimension.
No external dependencies, and the result is reviewable: it is just a list of
stage specs you can hand-edit before committing.

This is *assistive*: discovered stages are a good default and a good starting
point, not an oracle. The docstrings say so, and the API lets you merge the
output with hand-written specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .registry import tokenize
from .stages import StageSpec
from .types import ToolCall

__all__ = [
    "DiscoveredStage",
    "discover_stages",
    "merge_stages",
    "sessions_to_positions",
]


@dataclass
class DiscoveredStage:
    """One inferred phase, with the evidence that produced it."""

    name: str
    domain: str
    position: float
    tools: List[str] = field(default_factory=list)
    triggers: Tuple[str, ...] = ()
    confidence: float = 0.0
    support: int = 0

    def to_spec(self, order: int = 0) -> StageSpec:
        return StageSpec(
            name=self.name,
            domain=self.domain,
            description=f"auto-discovered phase at position {self.position:.2f}",
            triggers=self.triggers,
            order=order,
        )


# Conventional phase names by normalised position, used to label clusters in a
# way humans recognise. Purely cosmetic -- the *clustering* is data-driven; only
# the label is chosen from this ladder.
_PHASE_NAMES = ["locate", "inspect", "mutate", "verify", "close"]


def sessions_to_positions(
    sessions: Sequence[Sequence[ToolCall]],
) -> Dict[str, List[float]]:
    """Map each tool to the list of normalised positions it occupied.

    Position is ``i / (n - 1)`` within its session (0 for the first call, 1 for
    the last). Single-call sessions contribute position 0 -- they carry no
    ordering information, but the tool's mere presence is still a signal that it
    can start a flow.
    """
    positions: Dict[str, List[float]] = {}
    for session in sessions:
        n = len(session)
        if n == 0:
            continue
        for i, call in enumerate(session):
            pos = 0.0 if n == 1 else i / (n - 1)
            positions.setdefault(call.tool, []).append(pos)
    return positions


def discover_stages(
    sessions: Sequence[Sequence[ToolCall]],
    *,
    domain_of: Dict[str, str],
    k: Optional[int] = None,
    min_support: int = 3,
    max_k: int = 4,
) -> List[DiscoveredStage]:
    """Infer workflow phases per domain from observed call sequences.

    ``k`` fixes the number of phases per domain; when omitted it is chosen from
    the data (1..``max_k``) by a simple elbow heuristic on the within-cluster
    variance. Tools seen fewer than ``min_support`` times are ignored so a
    single odd session cannot invent a phase.
    """
    positions = sessions_to_positions(sessions)
    # Group tools by domain, dropping the ones with too little evidence.
    by_domain: Dict[str, Dict[str, float]] = {}
    support: Dict[str, int] = {}
    for tool, pos_list in positions.items():
        dom = domain_of.get(tool)
        if dom is None:
            continue
        if len(pos_list) < min_support:
            continue
        by_domain.setdefault(dom, {})[tool] = sum(pos_list) / len(pos_list)
        support[tool] = len(pos_list)

    out: List[DiscoveredStage] = []
    for dom, tool_pos in by_domain.items():
        if not tool_pos:
            continue
        items = sorted(tool_pos.items(), key=lambda kv: kv[1])
        ks = [k] if k else _choose_k([p for _t, p in items], max_k=max_k)
        clusters = _kmeans_1d([p for _t, p in items], ks)
        for ci, (center, members) in enumerate(clusters):
            if not members:
                continue
            tools = [items[i][0] for i in members]
            name = _label_for(center, len(clusters))
            out.append(
                DiscoveredStage(
                    name=name,
                    domain=dom,
                    position=round(center, 3),
                    tools=sorted(tools),
                    triggers=_mine_triggers(tools),
                    confidence=round(_confidence(members, [items[i][1] for i in members]), 3),
                    support=sum(support[t] for t in tools),
                )
            )
    # Deterministic order: by domain, then position.
    out.sort(key=lambda s: (s.domain, s.position))
    return out


def _choose_k(values: Sequence[float], *, max_k: int) -> int:
    """Pick a cluster count by the elbow of within-cluster variance."""
    if len(values) <= 1:
        return 1
    best_k, prev_wcss = 1, None
    for k in range(1, min(max_k, len(values)) + 1):
        wcss = _wcss(values, k)
        if prev_wcss is not None:
            # Relative improvement; stop when a new cluster adds little.
            improvement = (prev_wcss - wcss) / (prev_wcss or 1.0)
            if improvement < 0.35:
                break
        best_k = k
        prev_wcss = wcss
    return best_k


def _wcss(values: Sequence[float], k: int) -> float:
    clusters = _kmeans_1d(list(values), k)
    total = 0.0
    for center, members in clusters:
        for i in members:
            total += (values[i] - center) ** 2
    return total


def _kmeans_1d(values: List[float], k: int, *, iters: int = 50) -> List[Tuple[float, List[int]]]:
    """1-D Lloyd's algorithm with deterministic quantile seeding.

    Returns ``[(center, [indices])]`` sorted by center. Deterministic seeding
    (equally spaced quantiles rather than random points) keeps discovery
    reproducible across runs, which matters because the output becomes config.
    """
    n = len(values)
    if n == 0:
        return []
    k = max(1, min(k, n))
    order = sorted(range(n), key=lambda i: values[i])
    # Seed centers at evenly spaced quantiles of the sorted values.
    centers = [values[order[min(n - 1, int((j + 0.5) * n / k))]] for j in range(k)]
    centers = sorted(set(centers)) or [values[order[0]]]

    assign: List[int] = [0] * n
    for _ in range(iters):
        changed = False
        for i in range(n):
            best_j = min(range(len(centers)), key=lambda j: abs(values[i] - centers[j]))
            if assign[i] != best_j:
                assign[i] = best_j
                changed = True
        # Recompute centers.
        for j in range(len(centers)):
            members = [values[i] for i in range(n) if assign[i] == j]
            if members:
                centers[j] = sum(members) / len(members)
        if not changed:
            break

    grouped: Dict[int, List[int]] = {}
    for i, j in enumerate(assign):
        grouped.setdefault(j, []).append(i)
    return sorted(
        ((centers[j], grouped[j]) for j in grouped),
        key=lambda cm: cm[0],
    )


def _label_for(center: float, total: int) -> str:
    """Human-friendly phase name from the cluster's position."""
    if total <= 1:
        return _PHASE_NAMES[0]
    idx = min(len(_PHASE_NAMES) - 1, int(round(center * (len(_PHASE_NAMES) - 1))))
    return _PHASE_NAMES[idx]


def _mine_triggers(tools: Sequence[str], *, limit: int = 6) -> Tuple[str, ...]:
    """Triggers for a stage = a few distinctive keyword tokens from its tools."""
    counts: Dict[str, int] = {}
    for t in tools:
        for tok in set(tokenize(t.replace(".", " ").replace("_", " "))):
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(tok for tok, _ in ranked[:limit])


def _confidence(members: Sequence[int], values: Sequence[float]) -> float:
    """A crude 0..1 confidence: how tightly the cluster's positions agree."""
    if len(members) <= 1:
        return 0.5
    pts = [values[i] for i in members]
    spread = max(pts) - min(pts)
    return max(0.0, 1.0 - spread)


def merge_stages(
    declared: Sequence[StageSpec],
    discovered: Sequence[DiscoveredStage],
    *,
    overwrite: bool = False,
) -> List[StageSpec]:
    """Combine hand-written and discovered stages.

    Declared stages win unless ``overwrite`` is set -- auto-discovery is an
    assistant, and a human's explicit intent should not be silently discarded.
    Discovered stages for a domain that has *no* declared stages are always
    kept, which is the whole point: a new domain gets sensible phases for free.
    """
    by_domain_declared: Dict[str, List[StageSpec]] = {}
    for s in declared:
        by_domain_declared.setdefault(s.domain, []).append(s)

    out: List[StageSpec] = list(declared)
    for dom, existing in list(by_domain_declared.items()):
        if not overwrite:
            continue
        out = [s for s in out if s.domain != dom]

    seen = {(s.domain, s.name) for s in out}
    for d in discovered:
        key = (d.domain, d.name)
        if key in seen:
            continue
        order = len([s for s in out if s.domain == d.domain])
        out.append(d.to_spec(order=order))
        seen.add(key)
    return out
