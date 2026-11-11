"""Tests for layer 1 (domains) and layer 2 (stages)."""

from toolfunnel import DomainGate, DomainSpec, StageResolver, StageSpec


def test_domain_gate_selects_relevant_domain(small_domains, small_tools):
    from toolfunnel import ToolRegistry

    gate = DomainGate(small_domains, registry=ToolRegistry(small_tools))
    picked = gate.select("帮我查一下订单 123")
    assert picked[0].domain == "order"
    assert picked[0].score > 0


def test_domain_gate_never_empty(small_domains):
    gate = DomainGate(small_domains)
    picked = gate.select("完全不相关的一句话")
    assert len(picked) == 1  # falls back to highest prior


def test_domain_stopwords_veto(small_domains):
    gate = DomainGate(
        [DomainSpec("order", triggers=("订单",), stopwords=("测试",))]
    )
    with_stop = gate.score("订单测试")
    without = gate.score("订单")
    assert with_stop[0].score < without[0].score


def test_bridges_cross_domain(small_domains):
    gate = DomainGate(small_domains)
    bridges = gate.bridges_for(["order"])
    assert "logistics" in bridges


def test_stage_resolver_by_keyword(small_stages):
    r = StageResolver(small_stages)
    stage, src = r.resolve("帮我查一下订单", "order")
    assert stage == "locate" and src == "keyword"


def test_stage_resolver_chain_inheritance(small_stages, small_tools):
    from toolfunnel import ToolRegistry

    r = StageResolver(small_stages)
    stage, src = r.resolve("随便", "order", registry=ToolRegistry(small_tools),
                           last_tool="order.get")
    assert stage == "inspect" and src == "chain"


def test_stage_resolver_fallback_order(small_stages):
    r = StageResolver(small_stages)
    stage, src = r.resolve("嗯", "order")
    assert stage == "locate" and src == "none"


def test_stage_llm_hook_used_only_when_allowed():
    def picker(query, candidates):
        return "qa" if "怎么样" in query else None

    r = StageResolver([
        StageSpec("alpha", "support", order=0),
        StageSpec("qa", "support", order=1, llm_pick=picker),
    ])
    stage, src = r.resolve("这个东西怎么样", "support", allow_llm=True)
    assert (stage, src) == ("qa", "llm")
    stage2, src2 = r.resolve("这个东西怎么样", "support", allow_llm=False)
    assert src2 == "none"
