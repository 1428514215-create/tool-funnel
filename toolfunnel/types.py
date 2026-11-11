"""Core data types for tool-funnel.

Everything here is deliberately dependency-free so the library can run on a bare
CPython 3.8+ interpreter, with no network and no extra packages.

The central idea: an agent may hold *hundreds or thousands* of tools. Sending
every tool schema to the model on every turn is the single largest avoidable
cost in agentic systems. ``tool-funnel`` narrows that giant flat list down to a
handful of relevant tools using a four-layer funnel, while keeping an escape
hatch so the model is never truly boxed in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ParamSpec",
    "ToolSpec",
    "ToolCall",
    "SessionContext",
    "DomainScore",
    "RouteResult",
    "ToolMatch",
]


# --------------------------------------------------------------------------- #
# Tool schema
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    """A single argument of a tool.

    ``source`` tells the pre-filler where a value may come from when the model
    omits it: a key in the :class:`SessionContext` blackboard, or a literal
    default. This is what lets the router *pre-fill* arguments and shave turns
    off the conversation.
    """

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""
    enum: Optional[List[str]] = None
    # Where to look for a value when the model omits this argument.
    # ``context``  -> read SessionContext[key or name]
    # ``literal``  -> use ``default``
    source: str = "context"
    default: Any = None
    # A value the model produced earlier that plausibly satisfies this param,
    # e.g. order_id produced by "order.search" can fill "order.get.order_id".
    aliases: Tuple[str, ...] = ()

    def as_schema(self) -> Dict[str, Any]:
        """Return a compact OpenAI-style JSON schema fragment."""
        out: Dict[str, Any] = {"type": self.type}
        if self.description:
            out["description"] = self.description
        if self.enum:
            out["enum"] = list(self.enum)
        if not self.required and self.default is not None:
            out["default"] = self.default
        return out


@dataclass
class ToolSpec:
    """A tool the agent may call.

    ``domain`` and ``stage`` are the routing coordinates. ``keywords`` are the
    cheap lexical triggers used by the layer-1 domain gate. ``produces`` and
    ``requires`` drive argument pre-filling and chain construction.
    """

    name: str
    description: str = ""
    domain: str = "general"
    stage: Optional[str] = None
    keywords: Tuple[str, ...] = ()
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    # Context keys this tool needs / yields.
    requires: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()
    # Static hints for the chain layer; the learned graph augments these.
    next_hint: Tuple[str, ...] = ()
    # Domains this tool can bridge into, enabling cross-domain hopping without
    # a hard boundary.
    bridges: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    cost: float = 1.0
    dangerous: bool = False
    handler: Optional[Any] = None  # optional callable, for the demo runner

    # ---- derived ---------------------------------------------------------- #
    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(f"{self.domain}|{self.stage}|{self.name}|{self.description}".encode())
        h.update("|".join(self.keywords).encode())
        return h.hexdigest()[:16]

    def token_estimate(self) -> int:
        """Rough token cost of shipping this tool's schema to a model.

        Uses a whitespace heuristic (~1 token / 1.3 words) that tracks the real
        number closely enough for *relative* comparisons. If ``tiktoken`` is
        installed the benchmark CLI will use it instead.

        Memoised per instance: token accounting runs over the *whole* catalog
        on every route, so recomputing JSON-serialised schemas each time would
        dominate the funnel's otherwise sub-millisecond latency.
        """
        cached = getattr(self, "_token_estimate", None)
        if cached is None:
            blob = json.dumps(self.as_schema(), ensure_ascii=False)
            cached = estimate_tokens(blob)
            object.__setattr__(self, "_token_estimate", cached)
        return cached

    def as_schema(self) -> Dict[str, Any]:
        props = {k: v.as_schema() for k, v in self.params.items()}
        required = [k for k, v in self.params.items() if v.required]
        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": props},
        }
        if required:
            schema["parameters"]["required"] = required
        return schema


@dataclass
class ToolCall:
    """A single observed invocation, used both at runtime and for learning."""

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    produced: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
# Session blackboard
# --------------------------------------------------------------------------- #
class SessionContext:
    """A tiny key/value blackboard the agent and tools share.

    Values written here (e.g. an ``order_id`` returned by a search tool) become
    candidates for argument pre-fill on later turns. It is intentionally plain:
    a dict with provenance, not a framework.

    ``remember(alias=...)`` lets a produced value be retrievable under several
    names so that a key produced by one tool can satisfy a differently named
    parameter of another.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None) -> None:
        self._data: Dict[str, Any] = dict(initial or {})
        self._provenance: Dict[str, str] = {}

    # ---- writes ----------------------------------------------------------- #
    def set(self, key: str, value: Any, *, by: str = "", aliases: Sequence[str] = ()) -> None:
        self._data[key] = value
        self._provenance[key] = by
        for a in aliases:
            if a != key:
                self._data[a] = value
                self._provenance[a] = f"{by}->{key}"

    def update(self, values: Dict[str, Any], *, by: str = "", aliases: Sequence[str] = ()) -> None:
        for k, v in values.items():
            self.set(k, v, by=by, aliases=aliases)

    def merge_call(self, call: ToolCall) -> None:
        """Absorb the outputs of a tool call into the blackboard."""
        for k, v in call.produced.items():
            self.set(k, v, by=call.tool)

    # ---- reads ------------------------------------------------------------ #
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def provenance(self, key: str) -> str:
        return self._provenance.get(key, "")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SessionContext({self._data!r})"


# --------------------------------------------------------------------------- #
# Routing results
# --------------------------------------------------------------------------- #
@dataclass
class DomainScore:
    """Layer-1 output: how strongly a query matches a business domain."""

    domain: str
    score: float
    matched_terms: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DomainScore({self.domain!r}, {self.score:.3f})"


@dataclass
class ToolMatch:
    """A tool surfaced by the funnel, with the reason it was surfaced."""

    tool: ToolSpec
    score: float = 1.0
    reason: str = ""
    prefilled: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.tool.name


@dataclass
class RouteResult:
    """The complete output of one funnel pass.

    ``tools`` is what you hand to the model. ``explanation`` records *why* so
    that the decision is debuggable -- a hard requirement for a router that
    sits between a model and real tools.
    """

    query: str
    domains: List[DomainScore] = field(default_factory=list)
    stage: Optional[str] = None
    stage_source: str = ""  # "keyword" | "chain" | "llm" | "none"
    tools: List[ToolMatch] = field(default_factory=list)
    # Tools from *other* domains the model may still reach, kept small.
    bridges: List[ToolMatch] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    # Diagnostics.
    total_tools: int = 0
    surfaced_tools: int = 0
    tokens_full: int = 0
    tokens_surfaced: int = 0
    explanation: List[str] = field(default_factory=list)
    _by_name: Dict[str, ToolMatch] = field(default_factory=dict, repr=False)

    # ---- convenience ------------------------------------------------------ #
    def __post_init__(self) -> None:
        self._by_name = {m.name: m for m in (*self.tools, *self.bridges)}

    @property
    def tool_names(self) -> List[str]:
        return [m.name for m in self.tools]

    @property
    def all_names(self) -> List[str]:
        return [m.name for m in (*self.tools, *self.bridges)]

    def get(self, name: str) -> Optional[ToolMatch]:
        return self._by_name.get(name)

    def schemas(self, include_bridges: bool = True) -> List[Dict[str, Any]]:
        """OpenAI-style tool schemas ready to pass to a chat completion call."""
        out = [m.tool.as_schema() for m in self.tools]
        if include_bridges:
            out.extend(m.tool.as_schema() for m in self.bridges)
        return out

    @property
    def token_savings(self) -> float:
        """Fraction of tool-schema tokens saved versus sending everything."""
        if self.tokens_full <= 0:
            return 0.0
        return 1.0 - (self.tokens_surfaced / self.tokens_full)

    def summary(self) -> str:
        d = ", ".join(f"{s.domain}={s.score:.2f}" for s in self.domains) or "-"
        return (
            f"domains[{d}] stage={self.stage or '-'} "
            f"tools={len(self.tools)}+{len(self.bridges)}bridge "
            f"of {self.total_tools} "
            f"tokens={self.tokens_surfaced}/{self.tokens_full} "
            f"({self.token_savings:.0%} saved)"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"RouteResult({self.summary()})"


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Good enough for *relative* comparisons between "all tools" and "funneled
    tools", which is the number this library exists to move. Roughly matches
    BPE tokenizers on mixed English/JSON within ~15%.
    """
    if not text:
        return 0
    # CJK characters are ~1 token each; latin words ~1.3 tokens each.
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    words = len(text.split())
    latin = max(0, words - cjk)
    symbolic = sum(1 for ch in text if ch in "{}[]\":,")
    return int(cjk + latin * 1.3 + symbolic * 0.25) + 1


def iter_names(items: Iterable[Any]) -> List[str]:
    names: List[str] = []
    for it in items:
        names.append(it if isinstance(it, str) else getattr(it, "name", str(it)))
    return names
