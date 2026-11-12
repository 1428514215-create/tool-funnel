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


def test_packaged_example_catalog_is_loadable():
    """The catalog shipped inside the package must load via importlib.resources."""
    import json
    import sys

    if sys.version_info >= (3, 9):
        from importlib.resources import files

        text = files("toolfunnel.data").joinpath("compose.example.json").read_text("utf-8")
    else:  # pragma: no cover
        from importlib.resources import read_text

        text = read_text("toolfunnel.data", "compose.example.json")
    doc = json.loads(text)
    assert doc["tools"]


def test_demo_command_runs_offline(capsys):
    from toolfunnel.cli import main

    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "tool-funnel" in out
    assert "catalog:" in out


def test_catalog_command_generates_valid_document(tmp_path, capsys):
    from toolfunnel import build_from_dict, loads
    from toolfunnel.cli import main

    out_path = tmp_path / "cat.json"
    assert main(["catalog", "--tools", "120", "--out", str(out_path)]) == 0
    doc = loads(out_path.read_text(encoding="utf-8"))
    funnel = build_from_dict(doc)
    assert len(funnel.registry) == 120


def test_mcp_import_command(tmp_path, capsys):
    import json

    from toolfunnel.cli import main

    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps({"tools": [{"name": "order.search", "description": "搜索",
                               "inputSchema": {"type": "object", "properties": {}}}]}),
        encoding="utf-8",
    )
    assert main(["mcp-import", str(listing), "--server", "ecom"]) == 0
    assert "order" in capsys.readouterr().out


def test_zipapp_builder_produces_runnable_archive(tmp_path):
    """The .pyz must be a real, self-contained, runnable artifact."""
    import subprocess
    import sys

    from tools.build_zipapp import build

    out = str(tmp_path / "tf.pyz")
    build(out, include_examples=True)
    version = subprocess.run(
        [sys.executable, out, "--version"], capture_output=True, text=True
    )
    assert version.returncode == 0
    assert "tool-funnel" in version.stdout

    # Runs the bundled demo with no other files present.
    demo = subprocess.run(
        [sys.executable, out, "demo"], capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert demo.returncode == 0
    assert "catalog:" in demo.stdout


def test_install_script_is_executable():
    import os

    path = ROOT / "install.sh"
    assert path.exists()
    assert os.access(path, os.X_OK)


def test_no_runtime_dependencies():
    """The core promise: importing the package needs nothing but the stdlib."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependencies = []" in text
