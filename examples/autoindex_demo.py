"""Auto-index demo: hand it a raw pile of tools, get a working router back.

This is the "just point it at my tools" experience, and it needs neither an API
key nor a hand-written config. It shows the three realistic inputs and the same
one-call indexing pipeline for each:

1. an **MCP tool listing** (what a server publishes),
2. an **OpenAPI document** (what a service exposes),
3. a **bare list of tool names** (the absolute minimum),

then routes queries through the indexed funnel and prints what was inferred --
domains, mined triggers, stages and chains -- because every guess is meant to be
inspectable.

Run::

    python examples/autoinxex_demo.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolfunnel.index import (  # noqa: E402
    index_and_dump,
    tools_from_names,
    tools_from_openapi,
)

BAR = "=" * 74


def header(title: str) -> None:
    print("\n" + BAR)
    print(f"  {title}")
    print(BAR)


# --------------------------------------------------------------------------- #
# The three input shapes
# --------------------------------------------------------------------------- #
MCP_LISTING = {
    "tools": [
        {"name": "order.search", "description": "Search orders by keyword 搜索订单",
         "inputSchema": {"type": "object",
                         "properties": {"keyword": {"type": "string"}},
                         "required": ["keyword"]}},
        {"name": "order.get", "description": "Get order detail by id 查看订单详情",
         "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        {"name": "order.cancel", "description": "Cancel an order 取消订单",
         "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        {"name": "shipment.track", "description": "Track a shipment 查询物流轨迹",
         "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        {"name": "shipment.dispatch", "description": "Dispatch a shipment 发货",
         "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        {"name": "refund.create", "description": "Create a refund 创建退款单",
         "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        {"name": "refund.status", "description": "Get refund status 查询退款进度",
         "inputSchema": {"type": "object", "properties": {"refundId": {"type": "string"}}}},
        {"name": "inventory.check", "description": "Check stock level 查询库存",
         "inputSchema": {"type": "object", "properties": {"sku": {"type": "string"}}}},
    ]
}

OPENAPI_DOC = {
    "paths": {
        "/orders": {
            "get": {"operationId": "order.list", "summary": "List orders 订单列表",
                    "parameters": [{"name": "status", "schema": {"type": "string"}}]},
            "post": {"operationId": "order.create", "summary": "Create an order 创建订单"},
        },
        "/orders/{id}": {
            "get": {"operationId": "order.detail", "summary": "Order detail 订单详情",
                    "parameters": [{"name": "id", "required": True, "schema": {"type": "string"}}]},
        },
    }
}

BARE_NAMES = [
    "order.search", "order.get", "order.cancel", "order.verify",
    "shipment.track", "shipment.dispatch",
    "refund.create", "refund.get", "refund.close",
]


def _mcp_to_raws(listing):
    from toolfunnel.mcp import MCPToolServer, tool_specs_from_mcp
    from toolfunnel.index import RawTool

    specs = tool_specs_from_mcp(listing, server=MCPToolServer("demo", session=None))
    return [RawTool(s.name, s.description, params=s.params, keywords=s.keywords,
                    tags=s.tags, handler=s.handler) for s in specs]


def show(funnel, report, doc, queries) -> None:
    print(f"  {report.summary()}\n")
    print(f"  {'inferred domain':<18}{'tools':<7}{'how named':<16}triggers")
    print("  " + "-" * 68)
    for d in doc["domains"]:
        print(f"  {d['name']:<18}{report.domain_sizes.get(d['name'], 0):<7}"
              f"{report.domain_naming.get(d['name'], '?'):<16}"
              f"{', '.join(d['triggers'][:4])}")
    print()
    for q in queries:
        r = funnel.route(q)
        doms = ", ".join(f"{x.domain}={x.score:.2f}" for x in r.domains)
        print(f"  > {q}")
        print(f"      domains[{doms}] stage={r.stage} "
              f"tools={len(r.tools)}+{len(r.bridges)}bridge of {r.total_tools}")


def main() -> None:
    print("tool-funnel auto-index demo")
    print("raw tools in -> inferred routing config out (no API key, no hand-editing)")

    # ---- 1. MCP listing --------------------------------------------------- #
    header("1. From an MCP tool listing")
    print("  input: 8 MCP tools, no domain/stage config anywhere\n")
    raws = _mcp_to_raws(MCP_LISTING)
    funnel, doc, report = index_and_dump(raws)
    show(funnel, report, doc, ["帮我查一下订单 12345", "客户申请退款", "查一下物流"])

    # ---- 2. OpenAPI ------------------------------------------------------- #
    header("2. From an OpenAPI document")
    print("  input: 3 REST operations\n")
    raws = tools_from_openapi(OPENAPI_DOC)
    funnel, doc, report = index_and_dump(raws)
    show(funnel, report, doc, ["列出订单", "创建订单", "查看订单详情"])

    # ---- 3. Bare names ---------------------------------------------------- #
    header("3. From bare tool names (minimum input)")
    print("  input: just 9 names, nothing else\n")
    raws = tools_from_names(BARE_NAMES)
    funnel, doc, report = index_and_dump(raws)
    show(funnel, report, doc, ["帮我查订单", "退款进度", "物流跟踪"])

    # ---- 4. What the indexer wrote --------------------------------------- #
    header("4. The inferred config is an ordinary compose document")
    print("  (this is what --out writes; edit it, version it, reload it)\n")
    import json

    sample = {
        "domains": doc["domains"][:2],
        "stages": doc["stages"][:3],
        "tools": doc["tools"][:2],
    }
    print("  " + json.dumps(sample, ensure_ascii=False, indent=2).replace("\n", "\n  "))

    header("Summary")
    print("  you supply tools; the indexer infers domains, stages, chains and")
    print("  triggers. Every guess is written into a normal compose document,")
    print("  so you can inspect it, override any part, and reload it losslessly:")
    print()
    print("    tool-funnel index my_tools.json --out indexed.json")
    print("    tool-funnel route indexed.json \"帮我查一下订单\"")


if __name__ == "__main__":
    main()
