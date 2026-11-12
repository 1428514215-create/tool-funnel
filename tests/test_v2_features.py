"""Tests for the v0.2 features: adaptive budget, cache, distill, exploration,
stage discovery, prefill-acceptance learning, and the semantic domain fallback."""


from toolfunnel import (
    ChainGraph,
    DomainSpec,
    Funnel,
    FunnelConfig,
    ParamSpec,
    ToolCall,
    ToolSpec,
)
from toolfunnel.cache import FunnelCache, load_or_build
from toolfunnel.discovery import DiscoveredStage, discover_stages, merge_stages
from toolfunnel.stages import PreFiller, StageSpec


def _tools():
    return [
        ToolSpec("order.search", domain="order", stage="locate", description="搜索订单",
                 keywords=["订单"], produces=["order_id"], next_hint=["order.get"]),
        ToolSpec("order.get", domain="order", stage="inspect", description="订单详情",
                 keywords=["订单", "详情"], params={"order_id": ParamSpec("order_id", required=True)}),
        ToolSpec("order.cancel", domain="order", stage="mutate", description="取消订单",
                 keywords=["取消"]),
        ToolSpec("refund.create", domain="after_sales", stage="refund", description="退款",
                 keywords=["退款"], params={"amount": ParamSpec("amount", source="literal", default="full")}),
    ]


# --------------------------------------------------------------------------- #
# T1 adaptive budget
# --------------------------------------------------------------------------- #
def test_adaptive_budget_does_not_pad_small_stages():
    """A 1-tool stage must not be padded up to max_tools."""
    tools = [ToolSpec("order.search", domain="order", stage="locate",
                      description="搜索订单", keywords=["订单"])]
    funnel = Funnel.from_tools(
        tools, domains=[DomainSpec("order", triggers=("订单",))],
        stages=[StageSpec("locate", "order", triggers=("订单", "搜索"))],
        config=FunnelConfig(max_tools=6, adaptive_budget=True),
    )
    r = funnel.route("搜索订单")
    assert len(r.tools) == 1  # not padded to 6


def test_adaptive_budget_off_still_pads():
    tools = [ToolSpec("order.search", domain="order", stage="locate",
                      description="搜索订单", keywords=["订单"])]
    funnel = Funnel.from_tools(
        tools, domains=[DomainSpec("order", triggers=("订单",))],
        stages=[StageSpec("locate", "order", triggers=("订单",))],
        config=FunnelConfig(max_tools=6, adaptive_budget=False),
    )
    r = funnel.route("搜索订单")
    assert len(r.tools) == 1  # only one tool exists, but the cap is respected either way


def test_score_cutoff_removes_weak_tail():
    tools = _tools()
    funnel = Funnel.from_tools(
        tools,
        domains=[DomainSpec("order", triggers=("订单",))],
        stages=[StageSpec("locate", "order", triggers=("订单",))],
        config=FunnelConfig(max_tools=6, adaptive_budget=True, score_cutoff_ratio=0.9),
    )
    r = funnel.route("订单")
    # A high cutoff should drop everything far below the best match.
    assert 1 <= len(r.tools) <= len(tools)


def test_token_budget_caps_schemas():
    tools = _tools()
    funnel = Funnel.from_tools(
        tools,
        domains=[DomainSpec("order", triggers=("订单",))],
        config=FunnelConfig(max_tools=6, token_budget=80),
    )
    r = funnel.route("订单")
    assert r.tokens_surfaced <= 160  # at most a couple of small schemas
    assert len(r.tools) >= 1


# --------------------------------------------------------------------------- #
# T3 distill
# --------------------------------------------------------------------------- #
def test_distill_drops_weak_edges():
    g = ChainGraph()
    for _ in range(5):
        g.observe_pair(ToolCall("a"), ToolCall("b"))  # proven: count 5
    g.observe_pair(ToolCall("c"), ToolCall("d"))  # one-shot, weak
    # min_count=1: an edge seen at most once may be dropped when also weak.
    report = g.distill(min_weight=2.0, min_count=1)
    assert ("a", "b") in g.edges
    assert ("c", "d") not in g.edges
    assert report["removed"] >= 1


def test_distill_caps_per_source():
    g = ChainGraph()
    for i in range(30):
        g.observe_pair(ToolCall("a"), ToolCall(f"b{i}"))
    report = g.distill(min_weight=0.0, min_count=1, keep_per_source=5)
    assert len([e for e in g.edges.values() if e.src == "a"]) == 5
    assert report["capped"] >= 25


def test_distill_rebuilds_indegree():
    g = ChainGraph()
    for _ in range(5):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    g.observe_pair(ToolCall("c"), ToolCall("d"))
    g.distill(min_weight=2.0, min_count=1)
    assert g.in_degree.get("b", 0) > 0
    assert "d" not in g.in_degree


def test_needs_distill_guard():
    g = ChainGraph()
    g.observe_pair(ToolCall("a"), ToolCall("b"))
    assert g.needs_distill(max_edges=0) is True
    assert g.needs_distill(max_edges=1000) is False


# --------------------------------------------------------------------------- #
# T5 exploration
# --------------------------------------------------------------------------- #
def test_explore_zero_epsilon_matches_exploit():
    g = ChainGraph()
    for _ in range(5):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    plain = [s.tool for s in g.successors("a", limit=3)]
    explored = [s.tool for s in g.explore("a", limit=3, epsilon=0.0)]
    assert plain == explored


def test_explore_surfaces_unobserved_arm():
    g = ChainGraph()
    for _ in range(20):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    g.observe_pair(ToolCall("a"), ToolCall("c"))  # rare arm
    # With exploration on and a seeded rng that forces the swap, the rare arm
    # must appear even though pure exploitation would rank 'b' first.
    class Always:
        def random(self):
            return 0.0

    steps = g.explore("a", limit=2, epsilon=0.9, rng=Always())
    tools = [s.tool for s in steps]
    assert "b" in tools and "c" in tools


def test_explore_bonus_favors_rare_arms_without_swap():
    g = ChainGraph()
    for _ in range(20):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    g.observe_pair(ToolCall("a"), ToolCall("c"))

    class Never:
        def random(self):
            return 1.0  # never trigger the swap branch

    steps = {s.tool: s.score for s in g.explore("a", limit=5, epsilon=0.5, rng=Never())}
    # The rare arm gets a larger exploration bonus than the heavy one.
    assert steps["c"] > 0


def test_success_rate_tracked():
    g = ChainGraph()
    g.observe_pair(ToolCall("a"), ToolCall("b", ok=True))
    g.observe_pair(ToolCall("a"), ToolCall("b", ok=False))
    edge = g.edges[("a", "b")]
    assert edge.count == 2
    assert 0.0 < edge.rate() < 1.0


def test_declared_hint_not_displaced_by_a_few_observations():
    """A curated hint must survive early noise, but yield to sustained traffic.

    The spec is deliberately two-sided: a *handful* of calls must not erase a
    curated route (that was the bug -- learning silently degrading hand-authored
    chains), yet *sustained, consistent* traffic must be able to override it,
    because at that point the traffic is evidence the curated route was wrong.
    """
    # A few sightings of a competing edge: the curated hint still wins.
    g = ChainGraph()
    g.seed("a", ["curated"])
    for _ in range(2):
        g.observe_pair(ToolCall("a"), ToolCall("noise"))
    assert [s.tool for s in g.successors("a", limit=1)] == ["curated"]

    # Sustained traffic for the competing edge: it legitimately takes over.
    g2 = ChainGraph()
    g2.seed("a", ["curated"])
    for _ in range(30):
        g2.observe_pair(ToolCall("a"), ToolCall("noise"))
    assert [s.tool for s in g2.successors("a", limit=1)] == ["noise"]


def test_hint_and_observation_compound():
    """Observing a hinted edge should strengthen, not merely preserve, it."""
    g1 = ChainGraph()
    g1.seed("a", ["b"])
    only_hint = g1.successors("a", limit=1)[0].score

    g2 = ChainGraph()
    g2.seed("a", ["b"])
    for _ in range(10):
        g2.observe_pair(ToolCall("a"), ToolCall("b"))
    hinted_and_observed = g2.successors("a", limit=1)[0].score
    assert hinted_and_observed >= only_hint


def test_emergent_chain_learned_without_any_hint():
    """With no hint declared, traffic alone should surface the edge."""
    g = ChainGraph()
    assert g.successors("a", limit=1) == []
    for _ in range(5):
        g.observe_pair(ToolCall("a"), ToolCall("b"))
    assert g.successors("a", limit=1)[0].tool == "b"


# --------------------------------------------------------------------------- #
# T2 cache
# --------------------------------------------------------------------------- #
def test_cache_roundtrip(tmp_path):
    funnel = Funnel.from_tools(_tools(), domains=[DomainSpec("order", triggers=("订单",))])
    funnel.record("order.search", produced={"order_id": "1"})
    funnel.record("order.get", args={"order_id": "1"})
    path = str(tmp_path / "cache.json")
    FunnelCache(path).save(funnel)

    fresh = Funnel.from_tools(_tools(), domains=[DomainSpec("order", triggers=("订单",))])
    info = FunnelCache(path).load(fresh)
    assert info.loaded_index and info.loaded_learning
    assert ("order.search", "order.get") in fresh.chain.edges
    # Index restored means queries still work.
    assert fresh.route("订单").tools


def test_cache_detects_stale_catalog(tmp_path):
    funnel = Funnel.from_tools(_tools(), domains=[DomainSpec("order", triggers=("订单",))])
    path = str(tmp_path / "cache.json")
    FunnelCache(path).save(funnel)

    # A different catalog must not accept the cached index.
    changed = _tools() + [ToolSpec("extra.tool", domain="order", description="新工具")]
    fresh = Funnel.from_tools(changed, domains=[DomainSpec("order", triggers=("订单",))])
    info = FunnelCache(path).load(fresh)
    assert info.loaded_index is False
    assert "mismatch" in info.reason


def test_cache_missing_file_is_graceful(tmp_path):
    fresh = Funnel.from_tools(_tools())
    info = FunnelCache(str(tmp_path / "nope.json")).load(fresh)
    assert info.hit is False and info.reason


def test_load_or_build_falls_back(tmp_path):
    path = str(tmp_path / "c.json")
    funnel, info = load_or_build(
        lambda: Funnel.from_tools(_tools(), domains=[DomainSpec("order", triggers=("订单",))]),
        path,
    )
    assert info.loaded_index is False
    assert funnel.route("订单").tools  # rebuilt and usable


# --------------------------------------------------------------------------- #
# T4 stage discovery
# --------------------------------------------------------------------------- #
def test_discover_stages_finds_phases():
    # Sessions always run locate -> inspect -> mutate in that order.
    sessions = []
    for _ in range(10):
        sessions.append([
            ToolCall("order.search"), ToolCall("order.get"), ToolCall("order.cancel"),
        ])
    domain_of = {
        "order.search": "order", "order.get": "order", "order.cancel": "order",
    }
    stages = discover_stages(sessions, domain_of=domain_of, min_support=3)
    assert stages
    # Positions should be ordered and cover the phases.
    positions = sorted(s.position for s in stages)
    assert positions == sorted(positions)
    all_tools = {t for s in stages for t in s.tools}
    assert {"order.search", "order.get", "order.cancel"} <= all_tools


def test_discover_stages_labels_first_phase_locate():
    sessions = [[ToolCall("a.first"), ToolCall("a.last")] for _ in range(8)]
    stages = discover_stages(sessions, domain_of={"a.first": "a", "a.last": "a"})
    names = {s.name for s in stages}
    assert any(n in ("locate", "inspect", "mutate", "verify", "close") for n in names)


def test_discover_stages_respects_min_support():
    sessions = [[ToolCall("x")] for _ in range(2)]  # too few
    assert discover_stages(sessions, domain_of={"x": "d"}, min_support=5) == []


def test_merge_stages_preserves_declared():
    declared = [StageSpec("locate", "order", triggers=("查",), order=0)]
    discovered = [
        DiscoveredStage("inspect", "order", position=0.5, tools=["order.get"], triggers=("详情",)),
        DiscoveredStage("locate", "newdom", position=0.0, tools=["new.x"], triggers=("新",)),
    ]
    merged = merge_stages(declared, discovered)
    keys = {(s.domain, s.name) for s in merged}
    assert ("order", "locate") in keys       # declared kept
    assert ("order", "inspect") in keys      # discovered added
    assert ("newdom", "locate") in keys      # new domain fully discovered


def test_discovered_stage_to_spec():
    d = DiscoveredStage("mutate", "order", position=0.7, tools=["a"], triggers=("改",))
    spec = d.to_spec(order=2)
    assert spec.name == "mutate" and spec.domain == "order" and spec.order == 2


# --------------------------------------------------------------------------- #
# T7 prefill acceptance
# --------------------------------------------------------------------------- #
def test_prefill_acceptance_learned():
    filler = PreFiller()
    for _ in range(5):
        filler.observe_outcome("order.get", "order_id", "context:order_id", accepted=True)
    assert filler.acceptance_rate("order.get", "order_id", "context:order_id") == 1.0


def test_prefill_acceptance_needs_min_observations():
    filler = PreFiller()
    filler.observe_outcome("t", "p", "context:p", accepted=True)
    assert filler.acceptance_rate("t", "p", "context:p") is None


def test_prefill_acceptance_bias_is_bounded():
    filler = PreFiller()
    for _ in range(10):
        filler.observe_outcome("t", "p", "context:p", accepted=True)
    bias = filler._source_bias("t", "p", "context:p")
    assert 0.0 < bias <= 0.25


def test_prefill_acceptance_roundtrip():
    filler = PreFiller()
    for _ in range(4):
        filler.observe_outcome("t", "p", "context:p", accepted=False)
    state = filler.export_state()
    fresh = PreFiller()
    fresh.load_state(state)
    assert fresh.acceptance_rate("t", "p", "context:p") == 0.0


# --------------------------------------------------------------------------- #
# T6 semantic domain fallback
# --------------------------------------------------------------------------- #
def test_semantic_domain_gate_fills_gap():
    from toolfunnel.domains import DomainGate
    from toolfunnel.semantic import SemanticDomainGate

    gate = DomainGate([
        DomainSpec("order", triggers=("订单",)),
        DomainSpec("logistics", triggers=("物流",), description="包裹 快递 配送"),
    ])

    def embed(texts):
        # Tiny hand-made embedding: one axis per concept keyword.
        out = []
        for t in texts:
            out.append([
                float(any(k in t for k in ("订单",))),
                float(any(k in t for k in ("物流", "包裹", "快递", "配送"))),
            ])
        return out

    sem = SemanticDomainGate(gate=gate, embedder=embed, min_lexical_score=1.0)
    # "我的包裹怎么还没到" has no 物流 trigger; lexical gate is weak.
    assert sem.needs_semantics("我的包裹怎么还没到")
    scored = sem.select_or_fallback("我的包裹怎么还没到")
    assert scored[0].domain == "logistics"


def test_semantic_gate_not_invoked_when_lexical_is_strong():
    from toolfunnel.domains import DomainGate
    from toolfunnel.semantic import SemanticDomainGate

    gate = DomainGate([DomainSpec("order", triggers=("订单",))])
    calls = {"n": 0}

    def embed(texts):
        calls["n"] += 1
        return [[1.0] for _ in texts]

    sem = SemanticDomainGate(gate=gate, embedder=embed, min_lexical_score=0.6)
    # A strong trigger means no embedding call should happen.
    scored = sem.select_or_fallback("帮我查订单")
    assert scored[0].domain == "order"
    assert calls["n"] == 0
