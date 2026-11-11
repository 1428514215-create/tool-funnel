"""tool-funnel -- a funnel-shaped tool router for LLM agents.

When an agent owns dozens or hundreds of tools, shipping every schema on every
turn is the biggest avoidable cost in the loop, and it clouds the model's
judgement. ``tool-funnel`` narrows the field in four cheap layers:

1. **business-domain gate** -- keywords / curated triggers pick the domain(s),
2. **stage resolver** -- within a domain, pick the workflow phase,
3. **chain + lexical rank** -- while also recommending the *next* tool after a
   call, from a graph learned across sessions,
4. **argument pre-filler** -- fill args from the session blackboard so the model
   spends fewer turns.

Domains are *not* hard-isolated: bridge tools stay reachable and ``expand()``
is an explicit escape hatch, so a wrong guess costs one cheap call, not a dead
end.

Quick start::

    from toolfunnel import Funnel, ToolSpec, ParamSpec, DomainSpec

    funnel = Funnel.from_tools([...], domains=[...])
    result = funnel.route("帮我查一下订单 12345 的物流")
    print(result.summary())        # domains[...] stage=... tokens=... (saved)
    print(result.schemas())        # pass these to your chat completion call
"""

from .adapters import (
    AgentTurn,
    ToolFunnelMiddleware,
    ToolRunner,
    anthropic_tools_payload,
    openai_tools_payload,
)
from .chain import ChainEdge, ChainGraph, ChainStep
from .compose import build_from_dict, load_document, loads, to_dict
from .domains import DomainGate, DomainSpec
from .funnel import Funnel, FunnelConfig, Stats
from .learn import Learner, LearningReport, SessionRecord
from .registry import ToolRegistry, tokenize
from .stages import PreFiller, StageResolver, StageSpec
from .types import (
    DomainScore,
    ParamSpec,
    RouteResult,
    SessionContext,
    ToolCall,
    ToolMatch,
    ToolSpec,
    estimate_tokens,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "Funnel",
    "FunnelConfig",
    "Stats",
    "RouteResult",
    "ToolSpec",
    "ParamSpec",
    "ToolCall",
    "ToolMatch",
    "SessionContext",
    "DomainScore",
    # layers
    "ToolRegistry",
    "DomainGate",
    "DomainSpec",
    "StageResolver",
    "StageSpec",
    "PreFiller",
    "ChainGraph",
    "ChainEdge",
    "ChainStep",
    # learning
    "Learner",
    "SessionRecord",
    "LearningReport",
    # composition
    "build_from_dict",
    "load_document",
    "loads",
    "to_dict",
    # adapters
    "ToolRunner",
    "AgentTurn",
    "ToolFunnelMiddleware",
    "openai_tools_payload",
    "anthropic_tools_payload",
    # utils
    "tokenize",
    "estimate_tokens",
]


def __getattr__(name: str):
    # Lazily expose the optional semantic layer so importing it never requires
    # an embedding provider.
    if name in ("SemanticRanker", "make_openai_embedder", "cosine"):
        from . import semantic

        return getattr(semantic, name)
    raise AttributeError(f"module 'toolfunnel' has no attribute {name!r}")
