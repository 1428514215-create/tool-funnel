"""Tests for the chain graph: learning, decay, recommendation, persistence."""

import time

from toolfunnel import ChainGraph, ToolCall


def _calls(*names, ok=True):
    return [ToolCall(tool=n, ok=ok) for n in names]


def test_observe_pair_creates_edge():
    g = ChainGraph()
    g.observe_pair(ToolCall("a"), ToolCall("b"))
    steps = g.successors("a")
    assert steps and steps[0].tool == "b"
    assert steps[0].reason == "observed"


def test_repeated_pairs_rank_higher():
    g = ChainGraph()
    for _ in range(5):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    g.observe_pair(ToolCall("a"), ToolCall("c"))
    steps = g.successors("a")
    assert steps[0].tool == "b"  # more frequent -> higher


def test_failed_calls_contribute_less():
    g = ChainGraph()
    g.observe_pair(ToolCall("a"), ToolCall("b", ok=False))
    g.observe_pair(ToolCall("a"), ToolCall("c", ok=True))
    steps = {s.tool: s.score for s in g.successors("a")}
    assert steps["c"] > steps["b"]


def test_cross_domain_damping():
    g = ChainGraph(cross_domain_damping=0.5)
    g.register_domains({"a": "d1", "b": "d1", "c": "d2"})
    g.observe_pair(ToolCall("a"), ToolCall("b"))
    g.observe_pair(ToolCall("a"), ToolCall("c"))
    steps = {s.tool: s.score for s in g.successors("a")}
    # Same-domain edge wins even though both have weight 1.
    assert steps["b"] > steps["c"]


def test_time_decay_reduces_old_edges():
    g = ChainGraph(half_life_days=1.0)
    g.observe_pair(ToolCall("a"), ToolCall("b"))
    edge = g.edges[("a", "b")]
    fresh = edge.decay(g.half_life_days)
    old = edge.decay(g.half_life_days, now=time.time() + 86400 * 5)
    assert old < fresh


def test_seed_hints_available_without_observation():
    g = ChainGraph()
    g.seed("a", ["b", "c"])
    steps = {s.tool: s.reason for s in g.successors("a")}
    assert steps == {"b": "seed-hint", "c": "seed-hint"}


def test_observe_session_chains_in_order():
    g = ChainGraph()
    g.observe_session(_calls("a", "b", "c"))
    assert ("a", "b") in g.edges
    assert ("b", "c") in g.edges
    assert ("a", "c") not in g.edges


def test_arg_flow_recorded_on_cross_domain():
    g = ChainGraph()
    g.register_domains({"a": "d1", "b": "d2"})
    g.observe_pair(
        ToolCall("a", produced={"order_id": "x"}),
        ToolCall("b", args={"order_id": "x"}),
    )
    assert "order_id" in g.edges[("a", "b")].arg_flow


def test_chains_beam_search():
    g = ChainGraph()
    g.observe_session(_calls("a", "b", "c"))
    chains = g.chains("a", max_len=3)
    flat = [[s.tool for s in p] for p in chains]
    assert ["b"] in flat
    assert ["b", "c"] in flat


def test_roundtrip_serialization():
    g = ChainGraph(half_life_days=7.0, cross_domain_damping=0.6)
    g.register_domains({"a": "d1", "b": "d2"})
    g.observe_session(_calls("a", "b"))
    g.seed("x", ["y"])
    text = g.dumps()
    g2 = ChainGraph.loads(text)
    assert g2.half_life_days == 7.0
    assert ("a", "b") in g2.edges
    assert g2.edges[("a", "b")].cross_domain is True
    assert g2.successors("x")[0].tool == "y"


def test_stats_reports_counts():
    g = ChainGraph()
    g.observe_session(_calls("a", "b", "c"))
    stats = g.stats
    assert stats["edges"] == 2
    assert stats["observed_calls"] == 3
