"""Tests for the MCP adapter -- schema ingestion, execution, learning, merge."""

import asyncio

from toolfunnel import Funnel
from toolfunnel.mcp import (
    MCPToolRunner,
    MCPToolServer,
    ingest_mcp_servers,
    mcp_input_schema,
    merge_mcp_tools,
    normalize_mcp_result,
    tool_specs_from_mcp,
)


# Two fake sessions: MCP servers in the wild are async, but supporting a sync
# session lets in-process/embedded servers work without an event loop. Both are
# exercised so neither path silently breaks.
class SyncSession:
    """A synchronous MCP-like session."""

    def __init__(self, tools, results=None):
        self._tools = tools
        self._results = results or {}
        self.calls = []

    def list_tools(self):
        return {"tools": self._tools}

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self._results.get(name, {"content": [{"type": "text", "text": "ok"}]})


class AsyncSession:
    """An asynchronous MCP-like session (matches the official client)."""

    def __init__(self, tools, results=None):
        self._tools = tools
        self._results = results or {}
        self.calls = []

    async def list_tools(self):
        return {"tools": self._tools}

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return self._results.get(name, {"content": [{"type": "text", "text": "ok"}]})


LISTING = {
    "tools": [
        {
            "name": "order.search",
            "description": "搜索订单",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "关键字"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["keyword"],
            },
        },
        {
            "name": "order.get",
            "description": "查看订单详情",
            "inputSchema": {
                "type": "object",
                "properties": {"orderId": {"type": "string", "description": "订单号"}},
                "required": ["orderId"],
            },
        },
        {"name": "ship.track", "description": "跟踪物流"},
    ]
}


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_tool_specs_from_mcp_basic():
    server = MCPToolServer("orders", session=None)
    specs = tool_specs_from_mcp(LISTING, server=server)
    assert len(specs) == 3
    by_name = {s.name: s for s in specs}
    assert "order.search" in by_name
    search = by_name["order.search"]
    assert search.params["keyword"].required is True
    assert search.params["limit"].type == "integer"
    assert "mcp" in search.tags


def test_tool_specs_infer_domain_from_prefix():
    server = MCPToolServer("srv", session=None)
    specs = {s.name: s for s in tool_specs_from_mcp(LISTING, server=server)}
    assert specs["order.search"].domain == "order"
    assert specs["ship.track"].domain == "ship"


def test_domain_hints_override():
    server = MCPToolServer("srv", session=None, domain_hints={"ship": "logistics"})
    specs = {s.name: s for s in tool_specs_from_mcp(LISTING, server=server)}
    assert specs["ship.track"].domain == "logistics"


def test_namespacing():
    server = MCPToolServer("orders", session=None, namespace=True)
    specs = {s.name for s in tool_specs_from_mcp(LISTING, server=server)}
    assert "orders__order.search" in specs


def test_accepts_objects_not_just_dicts():
    class T:
        def __init__(self, name, desc, schema):
            self.name = name
            self.description = desc
            self.inputSchema = schema

    server = MCPToolServer("s", session=None)
    listing = [T("a.b", "desc", {"type": "object", "properties": {}})]
    specs = tool_specs_from_mcp(listing, server=server)
    assert specs[0].name == "a.b"


def test_input_schema_preserved_exactly():
    server = MCPToolServer("s", session=None)
    specs = tool_specs_from_mcp(LISTING, server=server)
    search = next(s for s in specs if s.name == "order.search")
    schema = mcp_input_schema(search)
    # The untouched MCP schema (with default) must survive.
    assert schema["properties"]["limit"]["default"] == 20


def test_reconstructed_schema_when_no_binding():
    from toolfunnel import ParamSpec, ToolSpec

    t = ToolSpec("plain", params={"x": ParamSpec("x", required=True)})
    schema = mcp_input_schema(t)
    assert schema["required"] == ["x"]


def test_camel_case_alias_recorded():
    server = MCPToolServer("s", session=None)
    specs = {s.name: s for s in tool_specs_from_mcp(LISTING, server=server)}
    get = specs["order.get"]
    # orderId -> order_id alias so pre-fill can bridge spellings.
    assert "order_id" in get.params["orderId"].aliases


def test_merge_servers_dedupes_names():
    s1 = MCPToolServer("a", session=None, namespace=True)
    s2 = MCPToolServer("b", session=None, namespace=True)
    merged = merge_mcp_tools([s1, s2], [LISTING, LISTING])
    names = [s.name for s in merged]
    assert len(names) == len(set(names))  # no collisions


# --------------------------------------------------------------------------- #
# Result normalization
# --------------------------------------------------------------------------- #
def test_normalize_text_content():
    out = normalize_mcp_result({"content": [{"type": "text", "text": "hello"}]})
    assert out["text"] == "hello"
    assert out["content"]


def test_normalize_error_flag():
    out = normalize_mcp_result({"content": [], "isError": True})
    assert out["isError"] is True


def test_normalize_structured_content():
    out = normalize_mcp_result({"structuredContent": {"order_id": "1"}})
    assert out["structured"]["order_id"] == "1"


def test_normalize_lifts_scalar_structured_fields_for_prefill():
    """The end-to-end enabler: structured scalars land on the blackboard."""
    out = normalize_mcp_result(
        {"content": [{"type": "text", "text": "ok"}],
         "structuredContent": {"orderId": "12345", "status": "PAID",
                               "items": [1, 2, 3]}}
    )
    # Scalars lifted to the top so pre-fill can consume them...
    assert out["orderId"] == "12345"
    assert out["status"] == "PAID"
    # ...but containers stay nested (they are not valid scalar params).
    assert "items" not in out
    assert out["structured"]["items"] == [1, 2, 3]


def test_structured_content_enables_prefill_end_to_end():
    session = SyncSession(LISTING["tools"])
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    runner.bind_tools([server], [LISTING])

    # A tool that returns structured content puts orderId on the blackboard.
    funnel.record("order.search", produced={"orderId": "777"}, context=runner.context)
    assert runner.context.get("orderId") == "777"
    # A later call to order.get would now be pre-fillable from that value.
    tool = funnel.registry.get("order.get")
    res = funnel.prefiller.fill(tool, {}, context=runner.context)
    assert res.args.get("orderId") == "777"


def test_normalize_object_result():
    class R:
        content = [{"type": "text", "text": "x"}]
        isError = False

    out = normalize_mcp_result(R())
    assert out["text"] == "x"


# --------------------------------------------------------------------------- #
# Execution + learning
# --------------------------------------------------------------------------- #
def test_bind_and_call_sync_records_chain():
    session = SyncSession(LISTING["tools"])
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    added = runner.bind_tools([server], [LISTING])
    assert added == 3
    assert len(funnel.registry) == 3

    c1 = runner.call("order.search", {"keyword": "x"})
    c2 = runner.call("order.get", {"orderId": "9"})
    assert c1.ok and c2.ok
    # The transition was learned.
    assert ("order.search", "order.get") in funnel.chain.edges
    assert session.calls[0][0] == "order.search"  # original MCP name used


def test_runner_records_error_from_mcp_iserror():
    session = SyncSession(
        LISTING["tools"],
        results={"order.get": {"content": [{"type": "text", "text": "boom"}], "isError": True}},
    )
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    runner.bind_tools([server], [LISTING])
    call = runner.call("order.get", {"orderId": "1"})
    assert call.ok is False
    assert call.produced.get("isError") is True


def test_runner_unknown_tool_is_error_not_crash():
    server = MCPToolServer("orders", session=SyncSession([]))
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    call = runner.call("nope")
    assert call.ok is False and "unknown" in call.error


def test_runner_records_exception():
    class BoomSession(SyncSession):
        def call_tool(self, name, arguments=None):
            raise RuntimeError("network down")

    server = MCPToolServer("orders", session=BoomSession(LISTING["tools"]))
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    runner.bind_tools([server], [LISTING])
    call = runner.call("order.search", {"keyword": "x"})
    assert call.ok is False and "network down" in call.error


def test_runner_async_path():
    session = AsyncSession(LISTING["tools"])
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    runner.bind_tools([server], [LISTING])

    async def go():
        return await runner.acall("order.search", {"keyword": "x"})

    call = asyncio.run(go())
    assert call.ok


def test_ingest_mcp_servers():
    session = AsyncSession(LISTING["tools"])
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()

    async def go():
        return await ingest_mcp_servers(funnel, [server])

    n = asyncio.run(go())
    assert n == 3
    assert len(funnel.registry) == 3


def test_tools_payload_routes_and_preserves_schema():
    session = SyncSession(LISTING["tools"])
    server = MCPToolServer("orders", session=session)
    funnel = Funnel()
    runner = MCPToolRunner(funnel, [server])
    runner.bind_tools([server], [LISTING])
    # Give the funnel a domain so routing can narrow.
    from toolfunnel import DomainSpec

    funnel.gate.add(DomainSpec("order", triggers=("订单", "order")))
    payload = runner.tools_payload("搜索订单 keyword")
    assert payload
    assert all("input_schema" in p for p in payload)
    assert any(p["name"] == "order.search" for p in payload)
