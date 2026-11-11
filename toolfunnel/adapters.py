"""Adapters that plug the funnel into real agent loops.

The funnel's job ends at: *here are the ~5 relevant tool schemas, with some args
pre-filled*. These adapters show how to wire that into the tool-calling loop of
common providers -- and, crucially, how to *record* the calls so the funnel
learns.

All adapters share one invariant: the model is never given the full catalog. It
gets ``result.schemas()``; if it asks for something outside that set, the host
calls :meth:`Funnel.expand` and adds exactly that one schema next turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .funnel import Funnel
from .types import RouteResult, SessionContext, ToolCall

__all__ = [
    "AgentTurn",
    "ToolRunner",
    "openai_tools_payload",
    "anthropic_tools_payload",
    "ToolFunnelMiddleware",
]


@dataclass
class AgentTurn:
    """State carried across one agent turn."""

    query: str
    context: SessionContext = field(default_factory=SessionContext)
    route: Optional[RouteResult] = None
    last_tool: Optional[str] = None
    calls: List[ToolCall] = field(default_factory=list)
    expanded: List[str] = field(default_factory=list)


def openai_tools_payload(route: RouteResult, *, include_bridges: bool = True) -> List[Dict[str, Any]]:
    """Schemas in the OpenAI ``tools=[{"type":"function", ...}]`` shape."""
    return [
        {"type": "function", "function": schema}
        for schema in route.schemas(include_bridges=include_bridges)
    ]


def anthropic_tools_payload(route: RouteResult, *, include_bridges: bool = True) -> List[Dict[str, Any]]:
    """Schemas in the Anthropic ``tools=[{"name":...}]`` shape (identical core)."""
    return route.schemas(include_bridges=include_bridges)


class ToolRunner:
    """Look up and execute a tool by name, and feed results back to the funnel.

    ``handlers`` maps tool name -> callable(``**args``) -> ``dict`` of produced
    values. The runner merges those into the session context and records the
    call (which is what trains the chain graph).
    """

    def __init__(
        self,
        funnel: Funnel,
        handlers: Dict[str, Callable[..., Dict[str, Any]]],
        *,
        context: Optional[SessionContext] = None,
    ) -> None:
        self.funnel = funnel
        self.handlers = handlers
        self.context = context or SessionContext()
        self.turn = AgentTurn(query="", context=self.context)

    def begin(self, query: str, *, last_tool: Optional[str] = None) -> RouteResult:
        self.turn = AgentTurn(query=query, context=self.context, last_tool=last_tool)
        self.turn.route = self.funnel.route(
            query, context=self.context, last_tool=last_tool
        )
        return self.turn.route

    def call(self, name: str, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolCall:
        """Execute one tool call and record it."""
        handler = self.handlers.get(name)
        merged: Dict[str, Any] = dict(args or {})
        merged.update(kwargs)
        if handler is None:
            return self._record(name, merged, ok=False, error=f"no handler for {name!r}")
        try:
            produced = handler(**merged) or {}
            if not isinstance(produced, dict):
                produced = {"result": produced}
            return self._record(name, merged, ok=True, produced=produced)
        except Exception as exc:  # noqa: BLE001 - propagate to the model
            return self._record(name, merged, ok=False, error=repr(exc))

    def _record(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        ok: bool,
        error: str = "",
        produced: Optional[Dict[str, Any]] = None,
    ) -> ToolCall:
        prev = self.turn.calls[-1] if self.turn.calls else None
        call = self.funnel.record(
            name,
            args=args,
            ok=ok,
            error=error,
            produced=produced or {},
            context=self.context,
            previous=prev,
        )
        self.turn.calls.append(call)
        self.turn.last_tool = name
        return call

    def expand(self, name: str) -> Optional[Any]:
        """Surface an extra tool the funnel missed; learn from the correction."""
        match = self.funnel.expand(name, context=self.context)
        if match is not None:
            self.turn.expanded.append(name)
        return match


def ToolFunnelMiddleware(
    funnel: Funnel,
    *,
    handlers: Optional[Dict[str, Callable[..., Dict[str, Any]]]] = None,
) -> Dict[str, Callable[..., Any]]:
    """Return callables suitable for LangChain-style middleware or plain use.

    Provides two functions:

    * ``get_tools(query, **kw) -> List[dict]``: funnel a turn into schemas.
    * ``on_tool_end(name, args, output, ok) -> None``: record a completed call.

    Kept framework-agnostic on purpose so it can be wrapped for whichever agent
    library the host uses.
    """

    def get_tools(query: str, **kw: Any) -> List[Dict[str, Any]]:
        result = funnel.route(query, **kw)
        return result.schemas()

    def on_tool_end(name: str, args: Dict[str, Any], output: Any, ok: bool = True) -> None:
        produced = output if isinstance(output, dict) else {"result": output}
        funnel.record(name, args=args, ok=ok, produced=produced)

    return {"get_tools": get_tools, "on_tool_end": on_tool_end}
