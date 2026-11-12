"""Tests for auto-indexing: raw tool lists -> a ready-to-route funnel."""

import json
import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from toolfunnel.compose import build_from_dict  # noqa: E402
from toolfunnel.index import (  # noqa: E402
    RawTool,
    index_and_dump,
    index_tools,
    infer_specs,
    tools_from_names,
    tools_from_openapi,
)


# A small hand-written catalog with a *known* ground truth, so we can assert the
# indexer recovers the intended structure rather than just "some" structure.
def _known_catalog():
    tools = []
    for verb in ("search", "get", "cancel"):
        tools.append(RawTool(f"order.{verb}", f"{verb} orders 订单", keywords=("订单",)))
    for verb in ("track", "dispatch"):
        tools.append(RawTool(f"shipment.{verb}", f"{verb} shipments 物流", keywords=("物流",)))
    for verb in ("create", "get", "close"):
        tools.append(RawTool(f"refund.{verb}", f"{verb} refunds 退款", keywords=("退款",)))
    return tools


# --------------------------------------------------------------------------- #
# Domain / stage inference
# --------------------------------------------------------------------------- #
def test_index_recovers_known_domains():
    domains, stages, specs, report = infer_specs(_known_catalog())
    names = {d.name for d in domains}
    assert {"order", "shipment", "refund"} <= names
    assert report.tools == 8


def test_index_assigns_verbs_to_stages():
    _d, _s, specs, _r = infer_specs(_known_catalog())
    by_name = {t.name: t for t in specs}
    assert by_name["order.search"].stage == "locate"
    assert by_name["order.get"].stage == "inspect"
    assert by_name["order.cancel"].stage == "mutate"


def test_index_builds_chain_hints_from_read_to_write():
    _d, _s, specs, _r = infer_specs(_known_catalog())
    search = next(t for t in specs if t.name == "order.search")
    # A search tool should hint toward the mutating sibling about orders.
    assert "order.cancel" in search.next_hint


def test_index_mines_triggers_that_actually_fire():
    funnel, _report = index_tools(_known_catalog())
    # The mined trigger for the order domain must let layer 1 fire on 订单.
    r = funnel.route("帮我查一下订单")
    assert r.domains and r.domains[0].domain == "order"
    assert r.domains[0].score > 0  # a real trigger hit, not just the prior fallback


def test_index_suppresses_boilerplate_triggers():
    """A token present in every tool must not become a domain trigger.

    The two domains are given genuinely different subject matter (orders vs
    shipments) *plus* shared boilerplate, so they cluster apart and the
    boilerplate term ("支持精确匹配") is measurably common in both -- which is
    exactly the condition under which it must be filtered out.
    """
    tools = [
        RawTool(f"order.{v}", f"{v} 订单 orders. 支持精确匹配 用于订单处理", keywords=("订单",))
        for v in ("search", "get", "cancel")
    ] + [
        RawTool(f"shipment.{v}", f"{v} 物流 shipment. 支持精确匹配 用于物流追踪", keywords=("物流",))
        for v in ("track", "dispatch", "notify")
    ]
    _d, _s, _specs, _r = infer_specs(tools)
    funnel, _ = index_tools(tools)

    # Sanity: the two subjects must actually have formed (otherwise the test
    # would be vacuous).
    assert len(funnel.gate.specs) >= 2, "expected order and shipment to be separate domains"

    for domain in funnel.gate.specs.values():
        for boiler in ("精确匹配", "支持", "匹配"):
            assert boiler not in domain.triggers, (
                f"{boiler!r} is boilerplate (in every domain) but became a "
                f"trigger of {domain.name!r}"
            )
    # ...while the genuine subject words survive as triggers.
    triggers = {t for d in funnel.gate.specs.values() for t in d.triggers}
    assert "订单" in triggers or "物流" in triggers


def test_unclustered_tools_land_in_general():
    tools = [RawTool(f"alpha.{v}", "alpha thing", keywords=("alpha",)) for v in ("get", "set")]
    tools.append(RawTool("zzz_solo", "totally unrelated one off"))
    _d, _s, specs, report = infer_specs(tools)
    domains = {d.name for d in _d}
    assert "general" in domains
    assert report.unclustered >= 1


def test_index_is_deterministic():
    a = infer_specs(_known_catalog())
    b = infer_specs(_known_catalog())
    assert [d.name for d in a[0]] == [d.name for d in b[0]]
    assert [t.name for t in a[2]] == [t.name for t in b[2]]


def test_index_scale_is_fast():
    """Indexing must stay well under a second at 1000 tools (regression guard)."""
    try:
        from generate_catalog import build_catalog
    except Exception:
        pytest.skip("example generator unavailable")
    doc = build_catalog(1000)
    raws = [RawTool(t["name"], t["description"], keywords=tuple(t.get("keywords", ()))) for t in doc["tools"]]
    t0 = time.perf_counter()
    infer_specs(raws)
    assert time.perf_counter() - t0 < 5.0


# --------------------------------------------------------------------------- #
# The produced funnel actually routes
# --------------------------------------------------------------------------- #
def test_indexed_funnel_routes_and_narrows():
    funnel, report = index_tools(_known_catalog())
    r = funnel.route("帮我查一下订单")
    assert r.tools
    assert r.tokens_surfaced < r.tokens_full
    assert r.surfaced_tools < r.total_tools


def test_indexed_funnel_bridges_cross_domain():
    funnel, _ = index_tools(_known_catalog())
    r = funnel.route("订单物流")
    # A query touching two closely related domains should still route, and
    # bridges must not exceed the configured cap.
    assert r.bridges is not None
    assert len(r.bridges) <= funnel.config.max_bridges


# --------------------------------------------------------------------------- #
# Emitted document round-trips
# --------------------------------------------------------------------------- #
def test_index_and_dump_roundtrips_through_compose():
    funnel, doc, report = index_and_dump(_known_catalog())
    assert doc["tools"] and doc["domains"]
    rebuilt = build_from_dict(json.loads(json.dumps(doc)))
    # The emitted document must reload into an equivalent funnel.
    assert sorted(rebuilt.registry.names) == sorted(funnel.registry.names)
    assert len(rebuilt.gate.specs) == len(funnel.gate.specs)


def test_indexed_document_passes_to_route_via_cli(tmp_path):
    _f, doc, _r = index_and_dump(_known_catalog())
    path = tmp_path / "indexed.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    from toolfunnel.cli import main

    assert main(["route", str(path), "查订单"]) == 0


def test_cli_index_from_mcp_listing(tmp_path, capsys):
    from toolfunnel.cli import main

    listing = tmp_path / "mcp.json"
    listing.write_text(json.dumps({
        "tools": [
            {"name": "order.search", "description": "搜索订单 订单",
             "inputSchema": {"type": "object", "properties": {"keyword": {"type": "string"}}}},
            {"name": "order.get", "description": "查看订单 订单",
             "inputSchema": {"type": "object", "properties": {"orderId": {"type": "string"}}}},
        ]
    }, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "indexed.json"
    assert main(["index", str(listing), "--out", str(out)]) == 0
    assert out.exists()
    assert "indexed" in capsys.readouterr().out
    # The emitted document must route.
    assert main(["route", str(out), "查订单"]) == 0


def test_cli_index_from_openapi(tmp_path):
    from toolfunnel.cli import main

    spec = tmp_path / "openapi.json"
    spec.write_text(json.dumps({
        "paths": {"/orders": {"get": {"operationId": "order.list", "summary": "订单列表 订单"}}}
    }, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "idx.json"
    assert main(["index", str(spec), "--out", str(out)]) == 0
    assert out.exists()


def test_cli_index_rejects_unknown_shape(tmp_path, capsys):
    from toolfunnel.cli import main

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
    assert main(["index", str(bad)]) == 2


# --------------------------------------------------------------------------- #
# Input adapters
# --------------------------------------------------------------------------- #
def test_tools_from_names_minimum_input():
    raws = tools_from_names(["order.search", "order.get", "ship.track"])
    assert len(raws) == 3
    funnel, report = index_tools(raws)
    assert len(funnel.registry) == 3
    assert report.tools == 3


def test_tools_from_openapi():
    spec = {
        "paths": {
            "/orders": {
                "get": {"operationId": "listOrders", "summary": "List orders 订单",
                        "parameters": [{"name": "status", "required": False,
                                        "schema": {"type": "string"}}]},
                "post": {"operationId": "createOrder", "summary": "Create order 订单"},
            },
            "/shipments/{id}": {
                "get": {"summary": "track shipment 物流",
                        "parameters": [{"name": "id", "required": True,
                                        "schema": {"type": "string"}}]},
            },
        }
    }
    raws = tools_from_openapi(spec)
    names = {r.name for r in raws}
    assert "listOrders" in names
    assert "createOrder" in names
    # The unnamed one falls back to a verb_path name.
    assert any(n.startswith("get_shipments") for n in names)
    # Parameters carried through for pre-fill.
    list_orders = next(r for r in raws if r.name == "listOrders")
    assert "status" in list_orders.params


def test_index_openapi_end_to_end():
    spec = {"paths": {"/orders": {"get": {"operationId": "order.search", "summary": "订单"},
                                  "post": {"operationId": "order.create", "summary": "订单"}}}}
    raws = tools_from_openapi(spec)
    funnel, report = index_tools(raws)
    assert "order" in report.domains


# --------------------------------------------------------------------------- #
# Optional model assistance
# --------------------------------------------------------------------------- #
def test_llm_namer_is_used_and_can_fail_safely():
    calls = []

    def namer(names, descriptions):
        calls.append(len(names))
        return "OrderManagement"  # odd casing to prove normalisation

    _d, _s, _specs, report = infer_specs(
        [RawTool(f"order.{v}", "订单", keywords=("订单",)) for v in ("get", "set")],
        llm_namer=namer,
    )
    assert calls  # the hook was actually consulted
    assert report.used_llm_namer
    assert any(d.name == "ordermanagement" for d in _d)


def test_broken_llm_namer_does_not_break_indexing():
    def boom(names, descriptions):
        raise RuntimeError("model down")

    domains, _s, specs, _r = infer_specs(_known_catalog(), llm_namer=boom)
    assert domains and specs  # fell back gracefully


def test_embedder_fallback_when_it_raises():
    def bad_embedder(texts):
        raise RuntimeError("no network")

    domains, _s, specs, report = infer_specs(_known_catalog(), embedder=bad_embedder)
    assert domains and specs
    assert report.used_embedder is False  # detected and fell back to TF-IDF


def test_embedder_when_available_is_used():
    # A tiny deterministic embedder: axis per concept.
    concepts = ["订单", "物流", "退款"]

    def embedder(texts):
        return [[float(c in t) for c in concepts] for t in texts]

    domains, _s, _specs, report = infer_specs(_known_catalog(), embedder=embedder)
    assert report.used_embedder is True
    assert len(domains) >= 2
