"""The funnel itself -- the orchestrator of the four layers.

One public entry point, :meth:`Funnel.route`, turns a user/agent turn into a
tiny, ranked set of tools:

    query ──▶ [1] DomainGate ──▶ [2] StageResolver ──▶ [3] chain + BM25 rank
                                                             │
                                                             ▼
                                            [4] PreFiller → tool schemas + args

Alongside ``route`` there are two feedback entry points that close the learning
loop:

* :meth:`Funnel.record` -- the model actually called a tool. Updates the chain
  graph, the per-stage success statistics and the domain prior.
* :meth:`Funnel.end_session` -- the session finished; flush aggregated evidence.

Design notes:

* The funnel *never* hides tools permanently. ``default_bridges`` plus
  :meth:`Funnel.expand` give the model a documented escape hatch, so a wrong
  layer-1 guess costs one cheap call, not a dead end.
* Everything is deterministic and offline by default. The only optional model
  contact is ``StageSpec.llm_pick`` and the embedding hook in ``semantic.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .chain import ChainGraph, ChainStep
from .domains import DomainGate, DomainSpec
from .registry import ToolRegistry, tokenize
from .stages import PreFiller, StageResolver, StageSpec
from .types import (
    RouteResult,
    SessionContext,
    ToolCall,
    ToolMatch,
    ToolSpec,
)

__all__ = ["Funnel", "FunnelConfig", "Stats"]


@dataclass
class FunnelConfig:
    """Tunables for the funnel. All optional; sane defaults for ~1000 tools."""

    top_domains: int = 2
    # Tools surfaced for the primary domain+stage.
    max_tools: int = 6
    # Extra cross-domain tools kept reachable.
    max_bridges: int = 3
    # Chain successors to append (beyond the lexical winners).
    chain_width: int = 4
    # Weight of the chain signal vs. BM25 lexical ranking.
    chain_weight: float = 1.35
    lexical_weight: float = 1.0
    # If a query token count is below this, skip BM25 bloat and trust the gate.
    always_include_stage_tools: bool = True
    # Hard cap: the funnel will never surface more than this many schemas.
    hard_cap: int = 14


@dataclass
class Stats:
    """Aggregate funnel statistics, including the savings headline."""

    routes: int = 0
    tools_considered: int = 0
    tools_surfaced: int = 0
    tokens_full: int = 0
    tokens_surfaced: int = 0
    prefilled_args: int = 0
    chain_hits: int = 0
    llm_calls: int = 0

    @property
    def token_savings(self) -> float:
        if self.tokens_full <= 0:
            return 0.0
        return 1.0 - self.tokens_surfaced / self.tokens_full

    @property
    def avg_surfaced(self) -> float:
        return self.tools_surfaced / self.routes if self.routes else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "routes": self.routes,
            "tools_considered": self.tools_considered,
            "tools_surfaced": self.tools_surfaced,
            "avg_surfaced_per_route": round(self.avg_surfaced, 2),
            "tokens_full": self.tokens_full,
            "tokens_surfaced": self.tokens_surfaced,
            "token_savings": round(self.token_savings, 4),
            "prefilled_args": self.prefilled_args,
            "chain_hits": self.chain_hits,
            "llm_calls": self.llm_calls,
        }


class Funnel:
    """Compose the four layers into one router."""

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        domains: Optional[Iterable[DomainSpec]] = None,
        stages: Optional[Iterable[StageSpec]] = None,
        chain: Optional[ChainGraph] = None,
        config: Optional[FunnelConfig] = None,
        prefiller: Optional[PreFiller] = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.config = config or FunnelConfig()
        self.gate = DomainGate(domains, registry=self.registry, top_k=self.config.top_domains)
        self.stages = StageResolver(list(stages) if stages else [])
        self.chain = chain or ChainGraph()
        self.prefiller = prefiller or PreFiller()
        self.stats = Stats()
        # Per-stage success counters, used by ``learning_bias``.
        self._stage_success: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._history: List[ToolCall] = []
        # Serialized graph is only authoritative if it carried domains.
        self.chain.register_domains({t.name: t.domain for t in self.registry})
        for t in self.registry:
            if t.next_hint:
                self.chain.seed(t.name, t.next_hint)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_tools(
        cls,
        tools: Iterable[ToolSpec],
        *,
        domains: Optional[Iterable[DomainSpec]] = None,
        stages: Optional[Iterable[StageSpec]] = None,
        config: Optional[FunnelConfig] = None,
    ) -> "Funnel":
        registry = ToolRegistry(tools)
        return cls(registry=registry, domains=domains, stages=stages, config=config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Funnel":
        """Build a funnel from a plain dict (see ``toolfunnel.compose``)."""
        from .compose import build_from_dict

        return build_from_dict(data, cls=cls)

    def add_tool(self, tool: ToolSpec) -> "Funnel":
        self.registry.add(tool)
        self.registry.finalize()
        self._catalog_token_cache = None
        self.chain.register_domains({tool.name: tool.domain})
        if tool.next_hint:
            self.chain.seed(tool.name, tool.next_hint)
        return self

    def _catalog_tokens(self) -> int:
        """Total schema tokens of the full catalog, computed once and cached."""
        cache = getattr(self, "_catalog_token_cache", None)
        if cache is None or cache[0] != len(self.registry):
            cache = (len(self.registry), sum(t.token_estimate() for t in self.registry))
            self._catalog_token_cache = cache  # type: ignore[attr-defined]
        return cache[1]

    # ------------------------------------------------------------------ #
    # Layer 1-4: route
    # ------------------------------------------------------------------ #
    def route(
        self,
        query: str,
        *,
        context: Optional[SessionContext] = None,
        last_tool: Optional[str] = None,
        history: Optional[Sequence[ToolCall]] = None,
        allow_llm_stage: bool = False,
        extra_domains: Sequence[str] = (),
    ) -> RouteResult:
        """Run the full funnel and return the narrowed tool set."""
        ctx = context or SessionContext()
        self.stats.routes += 1
        self.stats.tools_considered += len(self.registry)
        explanation: List[str] = []

        # ---- Layer 1: domains ----------------------------------------- #
        extra_terms = list(extra_domains)
        dom_scores = self.gate.select(query, extra_terms=extra_terms)
        dom_names = [d.domain for d in dom_scores]
        explanation.append(
            "L1 domains: " + (", ".join(f"{d.domain}({d.score:.2f})" for d in dom_scores) or "-")
        )

        # ---- Layer 2: stage ------------------------------------------- #
        primary = dom_names[0] if dom_names else None
        stage, stage_src = (None, "none")
        if primary:
            stage, stage_src = self.stages.resolve(
                query, primary, registry=self.registry, last_tool=last_tool,
                allow_llm=allow_llm_stage,
            )
            if stage_src == "llm":
                self.stats.llm_calls += 1
        explanation.append(f"L2 stage: {stage or '-'} via {stage_src}")

        # ---- Layer 3: candidate generation & ranking ------------------ #
        terms = tokenize(query)
        lexical = self.registry.bm25(terms) if terms else {}
        chain_scores: Dict[str, ChainStep] = {}
        if last_tool:
            for step in self.chain.successors(last_tool, limit=self.config.chain_width * 2):
                chain_scores[step.tool] = step
                self.stats.chain_hits += 1
        # The most recent call in history also seeds the chain if last_tool
        # was not passed explicitly.
        elif history:
            for step in self.chain.successors(history[-1].tool, limit=self.config.chain_width * 2):
                chain_scores[step.tool] = step

        candidates = self._gather_candidates(dom_names, stage, lexical, chain_scores)
        ranked = self._rank(candidates, lexical, chain_scores, primary, stage)

        tools = [m for m in ranked if m.tool.domain in dom_names][: self.config.max_tools]
        chosen = {m.name for m in tools}

        # Always keep the stage's own tools if we have room -- they are the
        # canonical verb-set for this phase and cheap insurance.
        if self.config.always_include_stage_tools and primary and stage:
            for t in self.registry.domain_tools(primary, stage):
                if t.name in chosen:
                    continue
                if len(tools) >= self.config.max_tools:
                    break
                tools.append(ToolMatch(t, 0.9, "stage-default"))
                chosen.add(t.name)

        # ---- Bridges: cross-domain reachability, not a hard wall ------ #
        bridge_names = self.gate.bridges_for(dom_names)
        bridges = self._pick_bridges(ranked, chosen, bridge_names)

        # ---- Layer 4: pre-fill ---------------------------------------- #
        prior_tools = self._recent_tool_specs(history)
        for m in tools:
            res = self.prefiller.fill(m.tool, {}, context=ctx, prior_tools=prior_tools)
            m.prefilled = res.filled
            self.stats.prefilled_args += len(res.filled)

        # ---- Token accounting ----------------------------------------- #
        tokens_full = self._catalog_tokens()
        surfaced = [*tools, *bridges][: self.config.hard_cap]
        tokens_surfaced = sum(m.tool.token_estimate() for m in surfaced)

        explanation.append(
            "L3 ranked: "
            + (", ".join(f"{m.name}({m.score:.2f})" for m in tools) or "-")
        )
        if bridges:
            explanation.append("bridges: " + ", ".join(m.name for m in bridges))

        result = RouteResult(
            query=query,
            domains=dom_scores,
            stage=stage,
            stage_source=stage_src,
            tools=tools,
            bridges=bridges,
            context=ctx.as_dict(),
            total_tools=len(self.registry),
            surfaced_tools=len(surfaced),
            tokens_full=tokens_full,
            tokens_surfaced=tokens_surfaced,
            explanation=explanation,
        )
        self.stats.tools_surfaced += len(surfaced)
        self.stats.tokens_full += tokens_full
        self.stats.tokens_surfaced += tokens_surfaced
        return result

    # ------------------------------------------------------------------ #
    # Candidate handling
    # ------------------------------------------------------------------ #
    def _gather_candidates(
        self,
        dom_names: Sequence[str],
        stage: Optional[str],
        lexical: Dict[str, float],
        chain_scores: Dict[str, "ChainStep"],
    ) -> set:
        cand: set = set()
        for d in dom_names:
            cand.update(t.name for t in self.registry.domain_tools(d))
        if dom_names and stage:
            cand.update(t.name for t in self.registry.domain_tools(dom_names[0], stage))
        # Lexical top-N regardless of domain (rescues a bad L1 guess cheaply
        # because it never becomes a surfaced schema unless it wins ranking).
        top_lex = sorted(lexical.items(), key=lambda kv: -kv[1])[: self.config.max_tools * 3]
        cand.update(n for n, _ in top_lex)
        cand.update(chain_scores)
        return cand

    def _rank(
        self,
        candidates: set,
        lexical: Dict[str, float],
        chain_scores: Dict[str, "ChainStep"],
        primary: Optional[str],
        stage: Optional[str],
    ) -> List[ToolMatch]:
        max_lex = max(lexical.values(), default=1.0) or 1.0
        out: List[ToolMatch] = []
        for name in candidates:
            tool = self.registry.get(name)
            if tool is None:
                continue
            score = 0.0
            reasons: List[str] = []
            lex = lexical.get(name, 0.0) / max_lex
            if lex:
                score += self.config.lexical_weight * lex
                reasons.append("lexical")
            if name in chain_scores:
                step = chain_scores[name]
                score += self.config.chain_weight * step.score
                reasons.append(f"chain:{step.reason}")
            if tool.domain == primary:
                score += 0.5
                reasons.append("domain")
            if stage and tool.stage == stage:
                score += 0.7
                reasons.append("stage")
            score += 0.2 * self._stage_success_bias(tool)
            if tool.dangerous:
                score -= 0.1
            out.append(ToolMatch(tool, round(score, 4), "+".join(reasons) or "filler"))
        out.sort(key=lambda m: (-m.score, m.name))
        return out

    def _pick_bridges(
        self, ranked: Sequence[ToolMatch], chosen: set, bridge_names: Iterable[str]
    ) -> List[ToolMatch]:
        """Pick a few cross-domain tools so the model can always reach sideways.

        Prefer tools in the declared bridge domains, but fall back to any strong
        candidate (a tool that declares its own ``bridges``, or simply ranked
        very highly) so the escape hatch is never empty.
        """
        bridges: List[ToolMatch] = []
        allowed = set(bridge_names)

        def take(predicate) -> None:
            for m in ranked:
                if len(bridges) >= self.config.max_bridges:
                    return
                if m.name in chosen or any(b.name == m.name for b in bridges):
                    continue
                if predicate(m):
                    bridges.append(ToolMatch(m.tool, m.score, "bridge"))

        take(lambda m: m.tool.domain in allowed)
        take(lambda m: bool(m.tool.bridges))
        take(lambda m: m.score > 0)
        return bridges

    def _recent_tool_specs(self, history: Optional[Sequence[ToolCall]]) -> List[ToolSpec]:
        if not history:
            return []
        out: List[ToolSpec] = []
        for c in history[-4:]:
            t = self.registry.get(c.tool)
            if t:
                out.append(t)
        return out

    def _stage_success_bias(self, tool: ToolSpec) -> float:
        if not tool.stage:
            return 0.0
        ok, total = self._stage_success.get((tool.domain, tool.stage), (0, 0))
        if total == 0:
            return 0.0
        return (ok / total) - 0.5

    # ------------------------------------------------------------------ #
    # Learning loop
    # ------------------------------------------------------------------ #
    def record(
        self,
        tool: str,
        *,
        args: Optional[Dict[str, Any]] = None,
        ok: bool = True,
        error: str = "",
        produced: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        context: Optional[SessionContext] = None,
        previous: Optional[ToolCall] = None,
    ) -> ToolCall:
        """Record one real tool call and update the learned structures."""
        call = ToolCall(
            tool=tool, args=dict(args or {}), ok=ok, error=error,
            produced=dict(produced or {}), latency_ms=latency_ms,
        )
        prev = previous if previous is not None else (self._history[-1] if self._history else None)
        if prev is not None:
            self.chain.observe_pair(prev, call)
        self._history.append(call)
        if context is not None:
            context.merge_call(call)

        tool_spec = self.registry.get(tool)
        if tool_spec and tool_spec.stage:
            ok_n, tot = self._stage_success.get((tool_spec.domain, tool_spec.stage), (0, 0))
            self._stage_success[(tool_spec.domain, tool_spec.stage)] = (ok_n + int(ok), tot + 1)
        return call

    def end_session(self, calls: Sequence[ToolCall]) -> None:
        """Flush a whole session's call list into the chain graph."""
        self.chain.observe_session(calls, domain_of={t.name: t.domain for t in self.registry})

    @property
    def history(self) -> List[ToolCall]:
        return list(self._history)

    def reset_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Escape hatch
    # ------------------------------------------------------------------ #
    def expand(self, name: str, *, context: Optional[SessionContext] = None) -> Optional[ToolMatch]:
        """Explicitly surface a tool the funnel did not rank.

        The model (or the host app) calls this when the funnel guessed wrong.
        That call is itself a learning signal: it is recorded as a strong edge
        from the current last tool, so the same correction never has to happen
        twice.
        """
        tool = self.registry.get(name)
        if tool is None:
            return None
        match = ToolMatch(tool, 1.0, "explicit-expand")
        res = self.prefiller.fill(tool, {}, context=context or SessionContext())
        match.prefilled = res.filled
        if self._history:
            self.chain.observe_pair(
                self._history[-1],
                ToolCall(tool=name, ok=True),
            )
        return match

    def prefetch(self, start: Optional[str] = None, *, depth: int = 3) -> List[List[ChainStep]]:
        """Probable chains from the current position, for warm-up/pre-fetch."""
        node = start or (self._history[-1].tool if self._history else None)
        if not node:
            return []
        return self.chain.chains(node, max_len=depth)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_learning(self, path: str) -> None:
        """Persist the *learned* state only -- a small JSON file."""
        import json

        payload = {
            "version": 1,
            "saved_at": time.time(),
            "chain": self.chain.to_dict(),
            "gate": self.gate.export_state(),
            "stage_success": {f"{d}\x1f{s}": list(v) for (d, s), v in self._stage_success.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def load_learning(self, path: str) -> None:
        import json

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.chain = ChainGraph.from_dict(payload.get("chain", {}))
        self.gate.load_state(payload.get("gate", {}))
        self._stage_success = {
            tuple(k.split("\x1f")): (int(v[0]), int(v[1]))  # type: ignore[arg-type]
            for k, v in payload.get("stage_success", {}).items()
        }

    def stats_dict(self) -> Dict[str, float]:
        d = self.stats.as_dict()
        d["chain_edges"] = len(self.chain.edges)
        return d
