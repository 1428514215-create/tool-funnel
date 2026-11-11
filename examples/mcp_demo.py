"""End-to-end MCP example: route a large MCP tool catalog through the funnel.

MCP servers commonly publish 50-200+ tools. Shipping all of them every turn is
the exact cost this library exists to remove. This example shows the whole loop
**without needing an API key or a real MCP server** -- it wires up a fake session
that publishes a realistic multi-domain catalog, then demonstrates:

1. ingesting an MCP ``list_tools`` listing into the funnel,
2. routing a query down to a handful of tools,
3. calling through the runner (which records the chain so it learns),
4. the funnel then pre-surfacing + pre-filling the next tool.

To use a real server, replace :class:`FakeMCPSession` with your
``mcp.client.session.ClientSession`` -- the two methods below are the only
surface this adapter touches.

Run::

    python examples/mcp_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolfunnel import DomainSpec, Funnel  # noqa: E402
from toolfunnel.mcp import (  # noqa: E402
    MCPToolRunner,
    MCPToolServer,
    ingest_mcp_servers,
)

BAR = "=" * 74


def header(title: str) -> None:
    print("\n" + BAR)
    print(f"  {title}")
    print(BAR)


# --------------------------------------------------------------------------- #
# A fake MCP server that behaves like the real thing.
# --------------------------------------------------------------------------- #
class FakeMCPSession:
    """Publishes a multi-domain tool catalog and answers calls.

    The method signatures match ``mcp.client.session.ClientSession`` exactly, so
    swapping this for a real session is a one-line change.
    """

    def __init__(self) -> None:
        self.calls = []

    async def list_tools(self):
        def t(name, desc, **props):
            return {
                "name": name,
                "description": desc,
                "inputSchema": {
                    "type": "object",
                    "properties": props,
                    "required": [k for k, v in props.items() if v.pop("_req", False)],
                },
            }

        return {
            "tools": [
                t("order.search", "按关键字搜索订单，返回订单号与状态",
                  keyword={"type": "string", "description": "关键字", "_req": True}),
                t("order.get", "按订单号查看订单详情",
                  orderId={"type": "string", "description": "订单号", "_req": True}),
                t("order.cancel", "取消订单",
                  orderId={"type": "string", "_req": True}),
                t("logistics.track", "查询物流轨迹",
                  orderId={"type": "string", "_req": True}),
                t("refund.create", "创建退款单",
                  orderId={"type": "string", "_req": True},
                  amount={"type": "string", "default": "full"}),
                t("refund.status", "查询退款进度",
                  refundId={"type": "string", "_req": True}),
                t("inventory.query", "查询商品库存",
                  sku={"type": "string", "_req": True}),
                t("crm.profile", "查看会员画像",
                  memberId={"type": "string", "_req": True}),
                t("report.export", "导出经营报表",
                  month={"type": "string", "_req": True}),
                t("product.describe", "生成商品卖点文案",
                  productId={"type": "string", "_req": True}),
            ]
        }

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        args = arguments or {}
        if name == "order.search":
            return {"content": [{"type": "text", "text": "找到订单 12345"}],
                    "structuredContent": {"orderId": "12345", "status": "PAID"}}
        if name == "order.get":
            return {"content": [{"type": "text", "text": f"订单 {args.get('orderId')} 详情"}]}
        return {"content": [{"type": "text", "text": f"{name} ok"}]}


async def main() -> None:
    session = FakeMCPSession()
    server = MCPToolServer("ecom", session=session, default_domain="misc",
                           domain_hints={"order": "order", "logistics": "logistics",
                                         "refund": "after_sales", "inventory": "inventory",
                                         "crm": "crm", "report": "analytics",
                                         "product": "product"})

    # ---- 1. ingest the MCP catalog ---------------------------------------- #
    header("1. Ingest an MCP server's tool catalog")
    funnel = Funnel()
    n = await ingest_mcp_servers(funnel, [server])
    print(f"  ingested {n} tools from MCP server {server.name!r}")
    print(f"  inferred domains: {', '.join(funnel.registry.domains)}")

    # Register domain triggers so layer 1 has business meaning (usually you
    # would supply these; MCP alone cannot know your business vocabulary).
    for name, trig in {
        "order": ("订单", "order"), "logistics": ("物流", "包裹", "快递"),
        "after_sales": ("退款", "售后", "退货"), "inventory": ("库存", "缺货"),
        "crm": ("会员", "客户"), "analytics": ("报表", "数据"), "product": ("商品", "卖点"),
    }.items():
        funnel.gate.add(DomainSpec(name, triggers=trig))

    runner = MCPToolRunner(funnel, [server])

    # ---- 2. route a query ------------------------------------------------- #
    header("2. Route a query -- only a few tools reach the model")
    query = "帮我查一下订单 12345 走到哪了"
    payload = runner.tools_payload(query)
    print(f"  query: {query}")
    print(f"  tools sent to model: {len(payload)} of {len(funnel.registry)}")
    for p in payload[:5]:
        print(f"    - {p['name']:<20} domain={p['x-domain']:<12} "
              f"required={p['input_schema'].get('required', [])}")
    r = funnel.route(query)
    print(f"  {r.summary()}")

    # ---- 3. call through the runner (records the chain) ------------------- #
    header("3. Call through the runner -- the chain learns")
    c1 = await runner.acall("order.search", {"keyword": "12345"})
    print(f"  called order.search -> {c1.produced.get('text')}")
    # order.search produced orderId; it lands in the shared context.
    print(f"  context now holds: orderId={runner.context.get('orderId')}")

    c2 = await runner.acall("logistics.track", {"orderId": "12345"})
    print(f"  called logistics.track -> ok={c2.ok}")
    print(f"  learned edge: order.search -> logistics.track "
          f"({('order.search', 'logistics.track') in funnel.chain.edges})")

    # ---- 4. the funnel pre-surfaces and pre-fills the next tool ---------- #
    header("4. Next turn: pre-surfaced + pre-filled")
    nxt = funnel.route("那物流呢", context=runner.context, last_tool="order.search")
    print(f"  after order.search, the funnel ranks:")
    for m in nxt.tools[:3]:
        pre = f"  prefill={m.prefilled}" if m.prefilled else ""
        print(f"    - {m.name:<20} {m.score:.2f} [{m.reason}]{pre}")
    print("  (note: orderId was already in context, so nothing to ask the model)")

    header("Summary")
    print(f"  catalog: {len(funnel.registry)} MCP tools")
    print(f"  per turn: {len(payload)} tool schemas instead of {len(funnel.registry)}")
    print(f"  chain edges learned: {len(funnel.chain.edges)}")


if __name__ == "__main__":
    asyncio.run(main())
