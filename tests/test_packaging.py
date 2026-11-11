"""Tests that the pack is coherent: README examples run, docs match reality."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_readme_quickstart_runs():
    """The README's headline example must actually work."""
    from toolfunnel import DomainSpec, Funnel, ParamSpec, ToolSpec

    tools = [
        ToolSpec(
            name="order.search", domain="order", stage="locate",
            description="搜索订单", keywords=["订单"],
            params={"keyword": ParamSpec("keyword", required=True)},
            produces=["order_id"], next_hint=["order.get"],
        ),
        ToolSpec(
            name="order.get", domain="order", stage="inspect",
            description="查看订单详情", keywords=["订单", "详情"],
            params={"order_id": ParamSpec("order_id", required=True)},
        ),
    ]
    funnel = Funnel.from_tools(
        tools, domains=[DomainSpec("order", triggers=("订单",))]
    )
    result = funnel.route("帮我查一下订单")
    assert result.tools
    assert result.schemas()


def test_package_exports_resolve():
    import toolfunnel

    for name in toolfunnel.__all__:
        assert getattr(toolfunnel, name) is not None, name
    # Optional semantic layer is importable lazily.
    assert toolfunnel.SemanticRanker is not None


def test_readme_exists_and_mentions_key_ideas():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for idea in ["漏斗", "业务域", "阶段", "链路", "预填", "学习"]:
        assert idea in text, f"README should explain: {idea}"


def test_license_and_gitignore_present():
    assert (ROOT / "LICENSE").exists()
    assert (ROOT / ".gitignore").exists()


def test_example_catalog_file_parses():
    path = ROOT / "examples" / "compose.example.json"
    if not path.exists():
        pytest.skip("example compose file not generated")
    from toolfunnel import build_from_dict, loads

    doc = loads(path.read_text(encoding="utf-8"))
    funnel = build_from_dict(doc)
    assert len(funnel.registry) >= 1
