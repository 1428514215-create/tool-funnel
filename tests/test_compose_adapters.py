"""Tests for compose (DSL/JSON), adapters, semantic re-rank and the CLI."""

import json

from toolfunnel import (
    ToolMatch,
    ToolSpec,
    build_from_dict,
    to_dict,
)
from toolfunnel.adapters import (
    ToolRunner,
    anthropic_tools_payload,
    openai_tools_payload,
)
from toolfunnel.semantic import SemanticRanker, cosine


DOC = {
    "config": {"max_tools": 3, "top_domains": 1},
    "domains": [
        {"name": "order", "triggers": ["订单"], "bridges": ["logistics"]},
        {"name": "logistics", "triggers": ["物流"]},
    ],
    "stages": [
        {"name": "locate", "domain": "order", "triggers": ["查"], "order": 0},
    ],
    "tools": [
        {"name": "order.search", "domain": "order", "stage": "locate",
         "description": "搜索订单", "keywords": ["订单"],
         "produces": ["order_id"],
         "params": [{"name": "keyword", "required": True}]},
        {"name": "ship.track", "domain": "logistics", "stage": "track",
         "description": "跟踪物流", "keywords": ["物流"]},
    ],
}


def test_build_from_dict():
    f = build_from_dict(DOC)
    assert len(f.registry) == 2
    assert f.config.max_tools == 3
    r = f.route("查订单")
    assert r.tools[0].tool.domain == "order"


def test_build_infers_domains_when_absent():
    doc = {"tools": [{"name": "a.b", "domain": "x", "keywords": ["甲"]}]}
    f = build_from_dict(doc)
    assert "x" in f.gate.names


def test_loads_json_and_roundtrip():
    f = build_from_dict(DOC)
    again = build_from_dict(json.loads(json.dumps(to_dict(f))))
    assert sorted(again.registry.names) == sorted(f.registry.names)


def test_to_dict_shape():
    f = build_from_dict(DOC)
    d = to_dict(f)
    assert {"config", "domains", "stages", "tools"} <= set(d)
    assert len(d["tools"]) == 2


def test_openai_and_anthropic_payloads():
    f = build_from_dict(DOC)
    r = f.route("查订单")
    oa = openai_tools_payload(r)
    an = anthropic_tools_payload(r)
    assert oa and oa[0]["type"] == "function"
    assert "function" in oa[0]
    assert an and "name" in an[0]


def test_tool_runner_executes_and_records():
    f = build_from_dict(DOC)
    calls = []

    def search(keyword):
        calls.append(keyword)
        return {"order_id": "42"}

    runner = ToolRunner(f, {"order.search": search})
    route = runner.begin("查订单")
    assert route.tools
    call = runner.call("order.search", {"keyword": "订单"})
    assert call.ok and call.produced["order_id"] == "42"
    assert runner.context.get("order_id") == "42"
    assert calls == ["订单"]


def test_tool_runner_records_failure():
    f = build_from_dict(DOC)

    def boom(**kw):
        raise RuntimeError("nope")

    runner = ToolRunner(f, {"order.search": boom})
    runner.begin("查订单")
    call = runner.call("order.search", {"keyword": "x"})
    assert call.ok is False
    assert "nope" in call.error


def test_semantic_rerank_changes_order():
    # A trivial "embedder" that maps known words to axes; no network needed.

    def embed(texts):
        out = []
        for t in texts:
            out.append([
                float(t.count("物流")),
                float(t.count("订单")),
            ])
        return out

    tools = [
        ToolSpec("a", description="订单工具", keywords=["订单"], domain="d"),
        ToolSpec("b", description="物流工具", keywords=["物流"], domain="d"),
    ]
    matches = [ToolMatch(tools[0], 1.0, "lexical"), ToolMatch(tools[1], 0.1, "lexical")]
    ranker = SemanticRanker(embedder=embed)
    reranked = ranker.rerank("物流 物流", matches, weight=0.9)
    assert reranked[0].name == "b"


def test_cosine_bounds():
    assert abs(cosine([1, 0], [1, 0]) - 1.0) < 1e-9
    assert abs(cosine([1, 0], [0, 1])) < 1e-9


def test_cli_route_and_bench(tmp_path, capsys):
    from toolfunnel.cli import main

    path = tmp_path / "cat.json"
    path.write_text(json.dumps(DOC, ensure_ascii=False), encoding="utf-8")

    assert main(["route", str(path), "查订单"]) == 0
    out = capsys.readouterr().out
    assert "order.search" in out
    assert "saved" in out

    assert main(["bench", str(path)]) == 0
    out = capsys.readouterr().out
    assert "average" in out

    assert main(["stats", str(path)]) == 0
