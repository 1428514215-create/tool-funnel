"""MCP adapter -- plug a Model Context Protocol tool server into the funnel.

MCP has become the standard way to expose tools to agents, and an MCP server
easily publishes dozens or hundreds of tools. That is exactly the situation this
library exists for: shipping every MCP tool schema on every turn is the same
avoidable cost, just with a different transport.

This module turns an MCP tool listing into a funnel catalog and maps tool calls
back onto the MCP wire protocol. It is deliberately **transport-agnostic**: the
core only needs two async callables in the shape of the official
``mcp`` client's session API:

* ``list_tools() -> {"tools": [{"name", "description", "inputSchema"}, ...]}``
* ``call_tool(name, arguments) -> {"content": [...], "isError": bool}``

If you have an ``mcp.client.session.ClientSession``, its ``list_tools()`` and
``call_tool()`` already match. No hard dependency on the ``mcp`` package is
introduced: you pass the session in, and the adapter talks to it duck-typed.
That keeps the core dependency-free and lets you test with a fake session.

What the adapter does that a naive wrapper would not:

* **Flattens the schema** into the funnel's :class:`ToolSpec` model, including a
  best-effort ``domain`` derived from a name prefix (``order.search`` ->
  ``order``) and the required-parameter list, so routing works out of the box.
* **Preserves the full JSON Schema** so nothing is lost: MCP tools keep their
  exact ``inputSchema`` when handed back to the model.
* **Records every call** into the funnel's learning loop, so cross-tool chains
  that MCP servers imply (list -> get -> update) are learned from real traffic.
* **Namespaces tool names** optionally (``server__tool``) so multiple servers
  can be merged into one funnel without collisions.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .funnel import Funnel
from .types import ParamSpec, SessionContext, ToolCall, ToolSpec

__all__ = [
    "MCPToolServer",
    "MCPSessionProtocol",
    "tool_specs_from_mcp",
    "mcp_input_schema",
    "MCPToolRunner",
    "merge_mcp_tools",
    "normalize_mcp_result",
]


# A session is anything exposing these two members. We never import ``mcp``.
class MCPSessionProtocol:  # pragma: no cover - documentation only
    async def list_tools(self) -> Any: ...
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any: ...


@dataclass
class MCPToolServer:
    """A named MCP session plus the metadata the funnel needs to ingest it."""

    name: str
    session: Any
    # Optional default domain for tools whose name carries no useful prefix.
    default_domain: str = "mcp"
    # Optional domain hints: ``{"search": "order"}`` treats a tool named
    # ``search`` (or ``foo.search``) as belonging to ``order``.
    domain_hints: Dict[str, str] = field(default_factory=dict)
    # Prefix tool names with the server name to avoid cross-server collisions.
    namespace: bool = False

    def qualifier(self) -> str:
        return f"{self.name}__" if self.namespace else ""


# --------------------------------------------------------------------------- #
# Schema ingestion
# --------------------------------------------------------------------------- #
def _json_type_to_param_type(schema: Dict[str, Any]) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    if t in ("integer", "number", "boolean", "array", "object", "string"):
        return t
    if "enum" in schema:
        return "string"
    return "string"


def _params_from_schema(input_schema: Optional[Dict[str, Any]]) -> Dict[str, ParamSpec]:
    """Convert a JSON Schema ``properties`` block into funnel ParamSpecs."""
    params: Dict[str, ParamSpec] = {}
    schema = input_schema or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for name, spec in props.items():
        if not isinstance(spec, dict):
            spec = {}
        aliases: Tuple[str, ...] = ()
        # MCP servers often use ``snake_case`` here while the model says
        # ``camelCase``; record the other spelling as an alias so pre-fill can
        # bridge them.
        snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
        if snake != name:
            aliases = (snake,)
        params[name] = ParamSpec(
            name=name,
            type=_json_type_to_param_type(spec),
            required=name in required,
            description=str(spec.get("description", "") or ""),
            enum=list(spec["enum"]) if isinstance(spec.get("enum"), list) else None,
            source="context",
            default=spec.get("default"),
            aliases=aliases,
        )
    return params


def _infer_domain(
    tool_name: str, *, server: MCPToolServer, input_schema: Optional[Dict[str, Any]]
) -> str:
    """Best-effort domain for an MCP tool.

    Order of preference:

    1. an explicit ``domain_hints`` match on the tool's own name or its prefix,
    2. the name prefix before ``.``/``_``/``/`` (``order.search`` -> ``order``),
    3. the server's ``default_domain``.

    This is intentionally simple and overridable -- the funnel's domain layer is
    where real business meaning should come from, and a wrong guess costs one
    cheap ``expand()`` call, not a dead end.
    """
    lowered = tool_name.lower()
    if lowered in server.domain_hints:
        return server.domain_hints[lowered]
    for sep in (".", "_", "/", ":"):
        if sep in tool_name:
            prefix = tool_name.split(sep, 1)[0].lower()
            if prefix in server.domain_hints:
                return server.domain_hints[prefix]
            # A non-trivial prefix doubles as its own domain name.
            if len(prefix) >= 3 and prefix not in ("tool", "fn", "func", "api", "mcp"):
                return prefix
            break
    if isinstance(input_schema, dict):
        hint = input_schema.get("x-domain") or input_schema.get("domain")
        if isinstance(hint, str) and hint:
            return hint
    return server.default_domain


def tool_specs_from_mcp(
    listing: Any,
    *,
    server: Optional[MCPToolServer] = None,
    namespace: Optional[bool] = None,
) -> List[ToolSpec]:
    """Flatten an MCP ``list_tools`` result into funnel :class:`ToolSpec` objects.

    ``listing`` accepts either the raw MCP shape (``{"tools": [...]}``) or a bare
    list of tool objects/dicts, and each tool may be a dict or an object with
    ``name`` / ``description`` / ``inputSchema`` attributes. That tolerance is
    deliberate: it means this works with the official client, a fake session in
    a test, and a hand-written dict alike.
    """
    tools_raw = listing.get("tools") if isinstance(listing, dict) else listing
    if tools_raw is None:
        tools_raw = []
    ns = namespace
    if ns is None and server is not None:
        ns = server.namespace

    specs: List[ToolSpec] = []
    for raw in tools_raw:
        name = _field(raw, "name")
        if not name:
            continue
        description = _field(raw, "description") or ""
        input_schema = _field(raw, "inputSchema") or _field(raw, "input_schema") or {}
        if isinstance(input_schema, str):
            try:
                input_schema = json.loads(input_schema)
            except ValueError:
                input_schema = {}
        domain = (
            _infer_domain(name, server=server, input_schema=input_schema)
            if server is not None
            else "mcp"
        )
        qualified = f"{server.qualifier()}{name}" if server is not None and ns else name
        spec = ToolSpec(
            name=qualified,
            description=str(description),
            domain=domain,
            stage=None,
            keywords=_keywords_for(name, description),
            params=_params_from_schema(input_schema),
            tags=("mcp",),
        )
        # Stash the *original* MCP name and raw schema so execution and model
        # handoff are lossless.
        spec.handler = _MCPBinding(server_name=server.name if server else "", tool=name,
                                   input_schema=input_schema)
        specs.append(spec)
    return specs


@dataclass
class _MCPBinding:
    """Internal: remembers which server/tool a flattened ToolSpec came from."""

    server_name: str
    tool: str
    input_schema: Dict[str, Any]


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _keywords_for(name: str, description: str) -> Tuple[str, ...]:
    words = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", f"{name} {description}")
    return tuple(dict.fromkeys(w for w in words if len(w) >= 2))[:12]


def mcp_input_schema(tool: ToolSpec) -> Dict[str, Any]:
    """Return the *original* MCP ``inputSchema`` for a tool, or reconstruct one.

    Keeping the untouched schema matters: MCP servers may use constructs the
    funnel's simplified param model does not represent (``oneOf``, nested
    objects, ``$ref``). When those exist we hand the model the real thing.
    """
    binding = tool.handler
    if isinstance(binding, _MCPBinding) and binding.input_schema:
        return binding.input_schema
    # Reconstruct a minimal schema from the simplified params.
    props: Dict[str, Any] = {}
    required: List[str] = []
    for p in tool.params.values():
        entry: Dict[str, Any] = {"type": p.type}
        if p.description:
            entry["description"] = p.description
        if p.enum:
            entry["enum"] = list(p.enum)
        props[p.name] = entry
        if p.required:
            required.append(p.name)
    schema: Dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def merge_mcp_tools(
    servers: Sequence[MCPToolServer], listings: Sequence[Any]
) -> List[ToolSpec]:
    """Merge several servers' listings into one deduplicated catalog.

    Namespacing within a server is honoured; if two servers still produce the
    same name, later ones are suffixed rather than dropped, so no tool is lost.
    """
    out: List[ToolSpec] = []
    seen: Dict[str, int] = {}
    for server, listing in zip(servers, listings):
        for spec in tool_specs_from_mcp(listing, server=server):
            if spec.name in seen:
                seen[spec.name] += 1
                spec.name = f"{spec.name}__{seen[spec.name]}"
            else:
                seen[spec.name] = 0
            out.append(spec)
    return out


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def normalize_mcp_result(result: Any) -> Dict[str, Any]:
    """Turn an MCP ``call_tool`` result into a plain dict for the blackboard.

    MCP content blocks are a list of typed parts (``{"type":"text","text":...}``,
    images, embedded resources). We flatten text parts into a single ``text``
    field and keep the raw list under ``content`` so nothing is discarded. An
    ``isError`` result is marked so :meth:`Funnel.record` can weight it down.

    Critically, when the server returns ``structuredContent`` (MCP's
    machine-readable output), its top-level fields are **also lifted to the top
    of the returned dict**. That is what makes pre-filling work end-to-end: a
    search tool returning ``{"orderId": "12345"}`` puts ``orderId`` directly on
    the blackboard, so a later ``order.get(orderId=...)`` can be pre-filled
    without asking the model for an id it already produced.
    """
    if result is None:
        return {}
    if isinstance(result, dict):
        content = result.get("content")
        is_error = bool(result.get("isError") or result.get("is_error"))
        structured = result.get("structuredContent") or result.get("structured_content")
    else:
        content = getattr(result, "content", None)
        is_error = bool(getattr(result, "isError", False))
        structured = getattr(result, "structuredContent", None)

    out: Dict[str, Any] = {}
    if is_error:
        out["isError"] = True
    if content is not None:
        out["content"] = content if isinstance(content, list) else [content]
        texts: List[str] = []
        for part in out["content"]:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, dict):
                texts.append(json.dumps(part, ensure_ascii=False)[:400])
        if texts:
            out["text"] = "\n".join(texts)
    if isinstance(structured, dict):
        out["structured"] = structured
        # Lift scalar fields so they land on the blackboard for pre-filling.
        # Nested objects/arrays stay only under "structured" to avoid polluting
        # the key space with structures no tool parameter could accept.
        for k, v in structured.items():
            if k not in out and not isinstance(v, (dict, list)):
                out[k] = v
    elif structured is not None:
        out["structured"] = structured
    return out


class MCPToolRunner:
    """Drive MCP tool calls *through* the funnel, recording the chain.

    Mirrors :class:`toolfunnel.adapters.ToolRunner` but talks to one or more MCP
    servers. Both sync and async sessions are supported: if ``call_tool`` is a
    coroutine function, use :meth:`acall`; otherwise :meth:`call`.

    The value here over a raw MCP client is the learning loop: every call is fed
    to :meth:`Funnel.record`, so after a few sessions the funnel knows that
    ``order.list`` is usually followed by ``order.get`` *for this deployment*,
    and pre-surfaces it with arguments pre-filled -- with no config written by
    hand.
    """

    def __init__(
        self,
        funnel: Funnel,
        servers: Sequence[MCPToolServer],
        *,
        context: Optional[SessionContext] = None,
    ) -> None:
        self.funnel = funnel
        self.servers = {s.name: s for s in servers}
        self.context = context or SessionContext()
        self._binding: Dict[str, _MCPBinding] = {}
        for tool in funnel.registry:
            if isinstance(tool.handler, _MCPBinding):
                self._binding[tool.name] = tool.handler

    def bind_tools(self, servers: Sequence[MCPToolServer], listings: Sequence[Any]) -> int:
        """Register flattened tools (with their bindings) into the funnel.

        Returns the number of tools added. Idempotent: re-binding the same names
        replaces them rather than raising.
        """
        added = 0
        for server, listing in zip(servers, listings):
            for spec in tool_specs_from_mcp(listing, server=server):
                binding = spec.handler
                if spec.name in self.funnel.registry:
                    self.funnel.registry.remove(spec.name)
                self.funnel.add_tool(spec)
                self._binding[spec.name] = binding  # type: ignore[assignment]
                added += 1
        return added

    # ---- sync path -------------------------------------------------------- #
    def call(self, name: str, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolCall:
        """Execute a tool synchronously and record it. Use with sync sessions."""
        merged: Dict[str, Any] = dict(args or {})
        merged.update(kwargs)
        binding = self._binding.get(name)
        if binding is None:
            return self._record(name, merged, ok=False, error=f"unknown MCP tool {name!r}")
        server = self.servers.get(binding.server_name)
        if server is None:
            return self._record(name, merged, ok=False, error=f"no server {binding.server_name!r}")
        try:
            raw = server.session.call_tool(binding.tool, merged)
            if inspect.isawaitable(raw):  # pragma: no cover - guard against misuse
                raise RuntimeError(
                    "session.call_tool returned an awaitable; use MCPToolRunner.acall"
                )
            produced = normalize_mcp_result(raw)
        except Exception as exc:  # noqa: BLE001 - surface to the model
            return self._record(name, merged, ok=False, error=repr(exc))
        ok = not produced.get("isError", False)
        return self._record(name, merged, ok=ok, produced=produced)

    # ---- async path ------------------------------------------------------- #
    async def acall(self, name: str, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> ToolCall:
        """Execute a tool asynchronously and record it. Use with async sessions."""
        merged: Dict[str, Any] = dict(args or {})
        merged.update(kwargs)
        binding = self._binding.get(name)
        if binding is None:
            return self._record(name, merged, ok=False, error=f"unknown MCP tool {name!r}")
        server = self.servers.get(binding.server_name)
        if server is None:
            return self._record(name, merged, ok=False, error=f"no server {binding.server_name!r}")
        try:
            raw = server.session.call_tool(binding.tool, merged)
            if inspect.isawaitable(raw):
                raw = await raw
            produced = normalize_mcp_result(raw)
        except Exception as exc:  # noqa: BLE001
            return self._record(name, merged, ok=False, error=repr(exc))
        ok = not produced.get("isError", False)
        return self._record(name, merged, ok=ok, produced=produced)

    def _record(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        ok: bool,
        error: str = "",
        produced: Optional[Dict[str, Any]] = None,
    ) -> ToolCall:
        prev = self.funnel.history[-1] if self.funnel.history else None
        return self.funnel.record(
            name,
            args=args,
            ok=ok,
            error=error,
            produced=produced or {},
            context=self.context,
            previous=prev,
        )

    # ---- model handoff ---------------------------------------------------- #
    def tools_payload(self, query: str, **route_kwargs: Any) -> List[Dict[str, Any]]:
        """Funnel a query into MCP-flavoured tool schemas for the model.

        The schema is the *original* MCP ``inputSchema`` (see
        :func:`mcp_input_schema`) so the model sees exactly what the server
        published, just far fewer of them.
        """
        result = self.funnel.route(query, context=self.context, **route_kwargs)
        payload: List[Dict[str, Any]] = []
        for match in (*result.tools, *result.bridges):
            payload.append(
                {
                    "name": match.name,
                    "description": match.tool.description,
                    "input_schema": mcp_input_schema(match.tool),
                    "x-domain": match.tool.domain,
                    "x-prefilled": dict(match.prefilled),
                }
            )
        return payload


async def ingest_mcp_servers(
    funnel: Funnel,
    servers: Sequence[MCPToolServer],
) -> int:
    """Await ``list_tools`` on each server and load them into the funnel.

    Convenience for the common cold-start flow: connect, list, ingest. Returns
    the number of tools added.
    """
    listings: List[Any] = []
    for server in servers:
        raw = server.session.list_tools()
        if inspect.isawaitable(raw):
            raw = await raw
        listings.append(raw)
    runner = MCPToolRunner(funnel, servers)
    return runner.bind_tools(servers, listings)
