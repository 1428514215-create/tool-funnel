"""Layer 2 -- stage resolution and Layer 3 -- argument pre-filling.

Layer 2 (``StageResolver``) answers "within this domain, which *stage* of the
workflow are we in?" A real business domain (orders, after-sales, marketing)
has phases -- e.g. *locate -> inspect -> mutate -> verify*. Each stage exposes
only its own handful of tools, so the funnel narrows again.

Resolution has three tiers, cheapest first:

1. ``keyword`` -- the query contains a stage trigger ("退款" -> after-sales/refund).
2. ``chain``   -- we already called a tool, so the stage is inherited from the
   tool's declared stage, or from the chain graph's recommendation.
3. ``llm``     -- an optional callback for the hard cases ("let the model pick").

Layer 3 (``PreFiller``) fills in arguments the model left out, drawing from the
session blackboard and from values earlier tools produced. This is what removes
whole round-trips: a ``order.get`` call can arrive fully specified because a
``order.search`` call already yielded the id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .registry import ToolRegistry, tokenize
from .types import ParamSpec, SessionContext, ToolSpec

__all__ = ["StageSpec", "StageResolver", "PreFiller", "PrefillResult"]


@dataclass
class StageSpec:
    """A phase within a domain."""

    name: str
    domain: str
    description: str = ""
    triggers: Tuple[str, ...] = ()
    # Typical neighbours, giving the resolver a fallback ordering.
    order: int = 0
    # Optional model-mediated picker: ``fn(query, candidates) -> stage|None``.
    # Never required; the funnel works fully deterministically without it.
    llm_pick: Optional[Callable[[str, List[str]], Optional[str]]] = None


class StageResolver:
    """Chooses the workflow stage within a domain."""

    def __init__(self, specs: Optional[Sequence[StageSpec]] = None) -> None:
        self.specs: Dict[Tuple[str, str], StageSpec] = {}
        self.by_domain: Dict[str, List[StageSpec]] = {}
        if specs:
            for s in specs:
                self.add(s)

    def add(self, spec: StageSpec) -> "StageResolver":
        self.specs[(spec.domain, spec.name)] = spec
        bucket = self.by_domain.setdefault(spec.domain, [])
        bucket.append(spec)
        bucket.sort(key=lambda s: s.order)
        return self

    def stages_of(self, domain: str) -> List[str]:
        return [s.name for s in self.by_domain.get(domain, [])]

    def resolve(
        self,
        query: str,
        domain: str,
        *,
        registry: Optional[ToolRegistry] = None,
        last_tool: Optional[str] = None,
        allow_llm: bool = False,
    ) -> Tuple[Optional[str], str]:
        """Return ``(stage, source)`` where source is keyword/chain/llm/none."""
        specs = self.by_domain.get(domain, [])
        if not specs:
            return None, "none"

        terms = set(tokenize(query))

        # 1) keyword triggers, highest overlap wins.
        best: Optional[Tuple[float, StageSpec]] = None
        for spec in specs:
            s = 0.0
            for trig in spec.triggers:
                t_terms = tokenize(trig)
                if not t_terms:
                    continue
                overlap = sum(1 for t in t_terms if t in terms)
                if overlap:
                    s += 0.6 + 0.4 * overlap / len(t_terms)
            if s > 0 and (best is None or s > best[0]):
                best = (s, spec)
        if best is not None:
            return best[1].name, "keyword"

        # 2) chain: inherit the last tool's stage when it belongs here.
        if last_tool and registry is not None:
            tool = registry.get(last_tool)
            if tool and tool.domain == domain and tool.stage:
                return tool.stage, "chain"

        # 3) optional model pick.
        if allow_llm:
            for spec in specs:
                if spec.llm_pick is None:
                    continue
                try:
                    picked = spec.llm_pick(query, [s.name for s in specs])
                except Exception:  # pragma: no cover - user callback
                    picked = None
                if picked:
                    return picked, "llm"

        # Fallback: first stage in declared order.
        return specs[0].name, "none"


# --------------------------------------------------------------------------- #
# Pre-filler
# --------------------------------------------------------------------------- #
@dataclass
class PrefillResult:
    args: Dict[str, Any] = field(default_factory=dict)
    filled: Dict[str, str] = field(default_factory=dict)  # param -> provenance
    missing_required: List[str] = field(default_factory=list)


class PreFiller:
    """Fills tool arguments from the session context and prior tool output.

    Precedence per parameter:

    1. a value the model already supplied (never overwritten),
    2. an exact context key,
    3. any declared ``aliases`` present in context,
    4. a declared literal ``default``,
    5. for required params, "produced-by" inference from the caller's
       ``produces`` sets (e.g. a prior tool that produces ``order_id``).

    The result separately reports ``missing_required`` so the caller can decide
    whether to ask the model or to short-circuit.

    **Learning the good sources.** When the funnel pre-fills a parameter and the
    model then *keeps* or *changes* that value, that outcome is a signal: a
    source (``context:order_id``, ``literal-default``, ...) whose values are
    usually accepted should be preferred, and one that is usually overridden is
    worse than useless. :meth:`observe_outcome` records these outcomes per
    ``(param, source)`` and :meth:`_lookup` consults the running acceptance rate
    to break ties between equally-specific candidates.
    """

    def __init__(self, *, use_produces: bool = True) -> None:
        self.use_produces = use_produces
        # (tool_or_param, source) -> [accepted, total]
        self._acceptance: Dict[Tuple[str, str], List[int]] = {}
        self.min_observations = 3

    # ---- learning --------------------------------------------------------- #
    def observe_outcome(
        self,
        tool: str,
        param: str,
        source: str,
        *,
        accepted: bool,
    ) -> None:
        """Record whether a pre-filled value survived contact with the model.

        ``accepted`` is True when the model used the suggested value (or left it
        untouched) and False when it supplied a different one.
        """
        if not source:
            return
        key = (f"{tool}.{param}", source)
        bucket = self._acceptance.setdefault(key, [0, 0])
        bucket[0] += int(accepted)
        bucket[1] += 1

    def acceptance_rate(self, tool: str, param: str, source: str) -> Optional[float]:
        bucket = self._acceptance.get((f"{tool}.{param}", source))
        if not bucket or bucket[1] < self.min_observations:
            return None
        return bucket[0] / bucket[1]

    def export_state(self) -> Dict[str, List[int]]:
        """JSON-safe form: ``"tool.param\x1fsource" -> [accepted, total]``."""
        return {f"{k[0]}\x1f{k[1]}": list(v) for k, v in self._acceptance.items()}

    def load_state(self, state: Dict[str, Any]) -> None:
        self._acceptance = {
            tuple(k.split("\x1f", 1)): [int(v[0]), int(v[1])]  # type: ignore[arg-type]
            for k, v in dict(state or {}).items()
        }

    def _source_bias(self, tool_name: str, param: str, source: str) -> float:
        """A small tie-break bonus/penalty from the acceptance history.

        Bounded to +/-0.25 so it can only reorder close candidates -- it must
        never override an explicit context value with a "learned" one.
        """
        rate = self.acceptance_rate(tool_name, param, source)
        if rate is None:
            return 0.0
        return (rate - 0.5) * 0.5

    def fill(
        self,
        tool: ToolSpec,
        given: Optional[Dict[str, Any]] = None,
        *,
        context: Optional[SessionContext] = None,
        prior_tools: Sequence[ToolSpec] = (),
        learned: bool = True,
    ) -> PrefillResult:
        given = dict(given or {})
        ctx = context or SessionContext()
        args: Dict[str, Any] = dict(given)
        filled: Dict[str, str] = {}
        missing: List[str] = []

        produced_keys: Dict[str, str] = {}
        if self.use_produces:
            for t in prior_tools:
                for key in t.produces:
                    produced_keys.setdefault(key, t.name)

        for pname, spec in tool.params.items():
            if pname in args and args[pname] not in (None, ""):
                continue
            value, why = self._lookup(
                tool.name, pname, spec, ctx, produced_keys, learned=learned
            )
            if value is not None:
                args[pname] = value
                filled[pname] = why
            elif spec.required:
                missing.append(pname)

        return PrefillResult(args=args, filled=filled, missing_required=missing)

    def _lookup(
        self,
        tool_name: str,
        pname: str,
        spec: ParamSpec,
        ctx: SessionContext,
        produced_keys: Dict[str, str],
        *,
        learned: bool = True,
    ) -> Tuple[Any, str]:
        # Collect every candidate source with a specificity rank. Higher rank =
        # more authoritative. Learning only perturbs within a rank band, so an
        # explicit context value can never lose to a learned default.
        candidates: List[Tuple[int, Any, str]] = []
        if pname in ctx:
            candidates.append((5, ctx[pname], f"context:{pname}"))
        for alias in spec.aliases:
            if alias in ctx:
                candidates.append((4, ctx[alias], f"context:{alias}->{pname}"))
        if spec.source == "literal" and spec.default is not None:
            candidates.append((3, spec.default, "literal-default"))
        if spec.default is not None and spec.source != "literal":
            candidates.append((2, spec.default, "default"))
        if pname in produced_keys and pname in ctx:
            candidates.append((1, ctx[pname], f"produced-by:{produced_keys[pname]}"))

        if not candidates:
            return None, ""

        if learned:
            def key(c: Tuple[int, Any, str]) -> Tuple[int, float]:
                rank, _value, source = c
                return (rank, self._source_bias(tool_name, pname, source))
            candidates.sort(key=key, reverse=True)
        else:
            candidates.sort(key=lambda c: c[0], reverse=True)

        _rank, value, source = candidates[0]
        return value, source
