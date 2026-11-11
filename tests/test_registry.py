"""Unit tests for the registry, indexing and tokenization."""

from toolfunnel import ToolRegistry, tokenize


def test_tokenize_english_and_cjk():
    toks = tokenize("搜索订单 order search")
    assert "订单" in toks          # CJK bigram retained
    assert "order" in toks
    assert "search" in toks
    assert "的" not in tokenize("我的订单")


def test_tokenize_cjk_bigrams():
    toks = tokenize("退款流程")
    assert "退" in toks and "款" in toks
    assert "退款" in toks


def test_registry_add_and_finalize(small_tools):
    reg = ToolRegistry(small_tools)
    assert len(reg) == 4
    assert "order.search" in reg
    assert set(reg.domains) == {"order", "logistics", "after_sales"}
    assert reg.get("order.search").stage == "locate"


def test_registry_duplicate_rejected(small_tools):
    reg = ToolRegistry()
    reg.add(small_tools[0])
    try:
        reg.add(small_tools[0])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected duplicate tool name to raise")


def test_keyword_hits_rank_by_idf(small_tools):
    reg = ToolRegistry(small_tools)
    hits = reg.keyword_hits(["订单"])
    assert "order.search" in hits and "order.get" in hits
    # 物流 is rarer than 订单 here, so track should out-score per-tool.
    track = reg.keyword_hits(["物流"])
    assert "shipment.track" in track


def test_bm25_prefers_exact_tool(small_tools):
    reg = ToolRegistry(small_tools)
    scores = reg.bm25(tokenize("物流轨迹"))
    assert scores
    assert max(scores, key=scores.get) == "shipment.track"


def test_domain_tools_filtered_by_stage(small_tools):
    reg = ToolRegistry(small_tools)
    names = [t.name for t in reg.domain_tools("order", "locate")]
    assert names == ["order.search"]


def test_registry_stats(small_tools):
    stats = ToolRegistry(small_tools).stats()
    assert stats["tools"] == 4
    assert stats["per_domain"]["order"] == 2
