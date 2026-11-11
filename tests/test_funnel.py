"""Tests for the funnel orchestrator, pre-filling, escape hatch and learning."""

from toolfunnel import (
    Funnel,
    FunnelConfig,
    SessionContext,
    ToolCall,
)
from toolfunnel.learn import Learner, SessionRecord


def test_route_surfaces_few_tools(funnel):
    r = funnel.route("帮我查一下订单 123")
    assert r.domains[0].domain == "order"
    assert r.tools
    assert len(r.tools) <= funnel.config.max_tools
    assert r.total_tools == 4


def test_route_token_savings_positive(funnel):
    r = funnel.route("帮我查一下订单")
    assert r.tokens_full > 0
    assert r.tokens_surfaced <= r.tokens_full
    assert 0.0 <= r.token_savings <= 1.0


def test_prefill_from_context(funnel):
    ctx = SessionContext({"order_id": "999"})
    r = funnel.route("订单详情", context=ctx)
    match = r.get("order.get")
    assert match is not None
    filled = funnel.prefiller.fill(match.tool, {}, context=ctx)
    assert filled.args["order_id"] == "999"


def test_prefill_alias_resolution(funnel):
    ctx = SessionContext({"id": "abc"})
    tool = funnel.registry.get("order.get")
    res = funnel.prefiller.fill(tool, {}, context=ctx)
    assert res.args["order_id"] == "abc"
    assert "order_id" not in res.missing_required


def test_prefill_reports_missing_required(funnel):
    tool = funnel.registry.get("order.search")
    res = funnel.prefiller.fill(tool, {}, context=SessionContext())
    assert "keyword" in res.missing_required


def test_prefill_never_overwrites_given(funnel):
    tool = funnel.registry.get("order.get")
    res = funnel.prefiller.fill(tool, {"order_id": "given"}, context=SessionContext({"order_id": "ctx"}))
    assert res.args["order_id"] == "given"


def test_record_builds_chain_and_context(funnel):
    ctx = SessionContext()
    funnel.record("order.search", args={"keyword": "a"}, produced={"order_id": "1"}, context=ctx)
    funnel.record("order.get", args={"order_id": "1"}, produced={"order_status": "ok"}, context=ctx)
    assert ("order.search", "order.get") in funnel.chain.edges
    assert ctx.get("order_id") == "1"
    assert ctx.get("order_status") == "ok"


def test_chain_recommendation_after_call(funnel):
    funnel.record("order.search", produced={"order_id": "1"})
    nxt = funnel.chain.successors("order.search")
    assert any(s.tool == "order.get" for s in nxt)


def test_expand_surfaces_unranked_tool_and_learns(funnel):
    funnel.record("order.search", produced={"order_id": "1"})
    match = funnel.expand("refund.create")
    assert match is not None and match.name == "refund.create"
    # The correction is recorded as an edge from the last called tool.
    assert ("order.search", "refund.create") in funnel.chain.edges


def test_expand_unknown_tool_returns_none(funnel):
    assert funnel.expand("does.not.exist") is None


def test_bridges_are_cross_domain(funnel):
    r = funnel.route("订单物流")
    bridge_domains = {m.tool.domain for m in r.bridges}
    assert bridge_domains <= {"order", "logistics", "after_sales"}


def test_learning_improves_domain_prior(funnel):
    sessions = [
        SessionRecord(query="退货款", calls=[ToolCall("refund.create")],
                      domains_used=("after_sales",)) for _ in range(5)
    ]
    report = Learner(funnel).learn(sessions)
    assert report.sessions == 5
    assert report.triggers_added > 0
    assert funnel.gate.learned_prior.get("after_sales", 0) > 0


def test_learning_rebuilds_chain(funnel):
    sessions = [
        SessionRecord(query="q", calls=[ToolCall("order.search"), ToolCall("order.get")],
                      domains_used=("order",))
        for _ in range(3)
    ]
    Learner(funnel).learn(sessions)
    assert ("order.search", "order.get") in funnel.chain.edges


def test_hard_cap_respected(funnel):
    funnel.config = FunnelConfig(max_tools=2, max_bridges=1, hard_cap=3)
    r = funnel.route("订单物流退款都涉及")
    assert len(r.tools) <= 2
    assert r.surfaced_tools <= 3


def test_stats_accumulate(funnel):
    funnel.route("订单")
    funnel.route("物流")
    assert funnel.stats.routes == 2
    assert funnel.stats_dict()["token_savings"] > 0


def test_save_and_load_learning(funnel, tmp_path):
    funnel.record("order.search", produced={"order_id": "1"})
    funnel.record("order.get", args={"order_id": "1"})
    path = str(tmp_path / "learned.json")
    funnel.save_learning(path)

    fresh = Funnel.from_tools(
        list(funnel.registry),
        domains=None,
        stages=None,
    )
    # Rebuild with same registry to compare learned edges.
    fresh = Funnel(registry=funnel.registry)
    fresh.load_learning(path)
    assert ("order.search", "order.get") in fresh.chain.edges
