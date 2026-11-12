"""Command-line interface for tool-funnel.

    tool-funnel demo                     # self-contained demo on bundled catalog
    tool-funnel stats    FILE.json        # catalog + routing stats
    tool-funnel route    FILE.json "..."  # route one query, show the funnel
    tool-funnel bench    FILE.json        # token-savings benchmark
    tool-funnel learn    FILE.json SPEC   # learn from a session corpus
    tool-funnel discover-stages FILE.json SPEC   # infer stages from sessions
    tool-funnel distill  LEARNED.json     # prune a learned state's chain graph
    tool-funnel mcp-import LISTING.json   # flatten an MCP list_tools result

The ``FILE`` is a compose document (see :mod:`toolfunnel.compose`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .compose import build_from_dict, load_document
from .funnel import Funnel
from .learn import Learner, SessionRecord
from .types import ToolCall


def _load(path: str) -> Funnel:
    return build_from_dict(load_document(path))


def _bundled(*parts: str) -> str:
    """Path to a file shipped alongside the package (works inside the zipapp).

    The zipapp bundles ``examples/`` at the archive root, which is on
    ``sys.path`` when the archive runs -- so reconstructing the path from
    ``sys.path[0]`` finds both the source checkout and the built ``.pyz``.
    """
    base = sys.path[0] if sys.path and sys.path[0] else os.getcwd()
    return os.path.join(base, *parts)


def _read_bundled_catalog() -> Optional[dict]:
    """Load the example catalog packaged inside ``toolfunnel.data``.

    Unlike :func:`_bundled` (a filesystem path), this reads the resource through
    the import system, so it works identically from a source checkout, an
    installed wheel, and a ``.pyz`` -- the last of which has no real filesystem
    paths at all. ``importlib.resources`` is used across the 3.8/3.9+ API split.
    """
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files

            resource = files("toolfunnel.data").joinpath("compose.example.json")
            return json.loads(resource.read_text(encoding="utf-8"))
        from importlib.resources import read_text  # pragma: no cover - 3.8 path

        return json.loads(read_text("toolfunnel.data", "compose.example.json"))
    except Exception:
        return None


def cmd_stats(args: argparse.Namespace) -> int:
    funnel = _load(args.file)
    reg = funnel.registry.stats()
    # Deterministic pseudo-embedding keeps the estimate stable across runs;
    # if tiktoken is present we still report the heuristic for comparability.
    total_tokens = sum(t.token_estimate() for t in funnel.registry)
    out = {
        "registry": reg,
        "total_schema_tokens": total_tokens,
        "domains": funnel.gate.names,
        "stages": [f"{d}/{s}" for (d, s) in funnel.registry.stages],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    funnel = _load(args.file)
    result = funnel.route(args.query, allow_llm_stage=False)
    print(result.summary())
    print("-" * 60)
    for line in result.explanation:
        print("  " + line)
    print("-" * 60)
    for m in result.tools:
        pre = f"  prefill: {m.prefilled}" if m.prefilled else ""
        print(f"  * {m.name:<28} {m.score:>6.2f}  [{m.reason}]{pre}")
    if result.bridges:
        print("  bridges: " + ", ".join(m.name for m in result.bridges))
    print("-" * 60)
    print(f"  surfaced {result.surfaced_tools} / {result.total_tools} tools; "
          f"tokens {result.tokens_surfaced} vs {result.tokens_full} "
          f"({result.token_savings:.1%} saved)")
    if args.json:
        print(json.dumps(
            {"tools": result.all_names, "stage": result.stage,
             "domains": [d.domain for d in result.domains]},
            ensure_ascii=False,
        ))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Compare full-catalog token cost against funneled cost over a query set."""
    funnel = _load(args.file)
    queries = args.queries or [
        "帮我查一下订单 12345 走到哪了",
        "客户申请退款，帮我处理一下售后",
        "这批货的库存还够吗",
        "给这个商品写个卖点",
        "发一条营销短信给会员",
    ]
    full = sum(t.token_estimate() for t in funnel.registry)
    rows = []
    for q in queries:
        r = funnel.route(q)
        rows.append((q, r.surfaced_tools, r.tokens_surfaced, r.token_savings))

    print(f"catalog: {len(funnel.registry)} tools, {full} schema tokens\n")
    print(f"{'query':<30} {'tools':>6} {'tokens':>8} {'saved':>7}")
    print("-" * 56)
    for q, n, tok, sv in rows:
        label = q if len(q) <= 28 else q[:27] + "…"
        print(f"{label:<30} {n:>6} {tok:>8} {sv:>6.1%}")
    avg_saved = sum(r[3] for r in rows) / len(rows)
    avg_tools = sum(r[1] for r in rows) / len(rows)
    print("-" * 56)
    print(f"{'average':<30} {avg_tools:>6.1f} "
          f"{int(sum(r[2] for r in rows)/len(rows)):>8} {avg_saved:>6.1%}")
    if args.json:
        print(json.dumps(
            {"full_tokens": full, "avg_savings": avg_saved,
             "avg_tools": avg_tools}, ensure_ascii=False))
    return 0


def cmd_learn(args: argparse.Namespace) -> int:
    """Learn from a JSON corpus of sessions: [{query, calls:[{tool,ok,args}], domains}]."""
    funnel = _load(args.file)
    with open(args.corpus, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    sessions = [
        SessionRecord(
            query=s.get("query", ""),
            calls=[ToolCall(tool=c["tool"], ok=c.get("ok", True),
                            produced=c.get("produced", {})) for c in s.get("calls", [])],
            domains_used=tuple(s.get("domains", ())),
            success=s.get("success", True),
        )
        for s in raw
    ]
    report = Learner(funnel).learn(sessions)
    print(report.summary())
    if args.out:
        funnel.save_learning(args.out)
        print(f"saved learned state -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tool-funnel", description=__doc__)
    p.add_argument("--version", action="version", version=f"tool-funnel {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stats", help="show catalog/domain statistics")
    s.add_argument("file")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("route", help="route one query through the funnel")
    s.add_argument("file")
    s.add_argument("query")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_route)

    s = sub.add_parser("bench", help="token-savings benchmark")
    s.add_argument("file")
    s.add_argument("--queries", nargs="*")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("learn", help="learn from a session corpus")
    s.add_argument("file")
    s.add_argument("corpus")
    s.add_argument("--out", help="where to write learned state JSON")
    s.set_defaults(func=cmd_learn)

    s = sub.add_parser(
        "discover-stages",
        help="infer workflow stages from a session corpus and print stage specs",
    )
    s.add_argument("file")
    s.add_argument("corpus")
    s.add_argument("--min-support", type=int, default=3)
    s.add_argument("--k", type=int, default=0, help="fixed phase count (0 = auto)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_discover_stages)

    s = sub.add_parser("distill", help="prune a learned-state file's chain graph")
    s.add_argument("learned", help="learned-state JSON produced by `learn --out`")
    s.add_argument("--min-weight", type=float, default=0.5)
    s.add_argument("--keep-per-source", type=int, default=8)
    s.add_argument("--out", help="where to write the pruned state")
    s.set_defaults(func=cmd_distill)

    s = sub.add_parser("mcp-import", help="inspect/flatten an MCP list_tools JSON file")
    s.add_argument("listing", help="a JSON file containing an MCP list_tools result")
    s.add_argument("--server", default="mcp", help="server name used for namespacing")
    s.add_argument("--namespace", action="store_true")
    s.add_argument("--json", action="store_true", help="emit a compose document")
    s.set_defaults(func=cmd_mcp_import)

    s = sub.add_parser(
        "demo", help="run a demo on the bundled example catalog (no other files needed)"
    )
    s.add_argument("--file", help="catalog to use (default: bundled example)")
    s.add_argument("--queries", nargs="*", help="queries to route")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("catalog", help="generate a synthetic multi-domain catalog")
    s.add_argument("--tools", type=int, default=1000)
    s.add_argument("--out", help="output path (default: catalog.json)")
    s.set_defaults(func=cmd_catalog)
    return p


def cmd_discover_stages(args: argparse.Namespace) -> int:
    """Infer stages from a corpus and print them as copy-pasteable specs."""
    from .discovery import discover_stages

    funnel = _load(args.file)
    with open(args.corpus, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    sessions = [
        [ToolCall(tool=c["tool"], ok=c.get("ok", True)) for c in s.get("calls", [])]
        for s in raw
    ]
    domain_of = {t.name: t.domain for t in funnel.registry}
    stages = discover_stages(
        sessions, domain_of=domain_of, min_support=args.min_support,
        k=(args.k or None),
    )
    if args.json:
        print(json.dumps(
            [{"name": s.name, "domain": s.domain, "position": s.position,
              "triggers": list(s.triggers), "tools": s.tools,
              "confidence": s.confidence, "support": s.support} for s in stages],
            ensure_ascii=False, indent=2))
        return 0
    if not stages:
        print("no stages discovered (not enough session data)")
        return 0
    print(f"discovered {len(stages)} stages from {len(sessions)} sessions:\n")
    for s in stages:
        print(f"  {s.domain}/{s.name}  (pos={s.position:.2f} "
              f"conf={s.confidence:.2f} support={s.support})")
        print(f"    triggers: {', '.join(s.triggers) or '-'}")
        print(f"    tools:    {', '.join(s.tools[:4])}"
              + (" ..." if len(s.tools) > 4 else ""))
    print("\nadd to a compose document's \"stages\" array (use --json to emit):")
    print(json.dumps([{"name": s.name, "domain": s.domain, "order": i,
                       "triggers": list(s.triggers)}
                      for i, s in enumerate(stages)], ensure_ascii=False))
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from .chain import ChainGraph

    with open(args.learned, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    graph = ChainGraph.from_dict(payload.get("chain", {}))
    report = graph.distill(
        min_weight=args.min_weight, keep_per_source=args.keep_per_source
    )
    payload["chain"] = graph.to_dict()
    out = args.out or args.learned
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"chain distilled -> {out}")
    print(f"  edges {report['edges_before']} -> {report['edges_after']} "
          f"(removed {report['removed']}, capped {report['capped']})")
    return 0


def cmd_mcp_import(args: argparse.Namespace) -> int:
    """Flatten an MCP listing into a funnel compose document."""
    from .compose import to_dict
    from .funnel import Funnel
    from .mcp import MCPToolServer, tool_specs_from_mcp
    from .registry import ToolRegistry

    with open(args.listing, "r", encoding="utf-8") as fh:
        listing = json.load(fh)
    server = MCPToolServer(args.server, session=None, namespace=args.namespace)
    specs = tool_specs_from_mcp(listing, server=server)
    if args.json:
        funnel = Funnel(registry=ToolRegistry(specs))
        print(json.dumps(to_dict(funnel), ensure_ascii=False, indent=2))
        return 0
    print(f"flattened {len(specs)} MCP tools from server {args.server!r}:")
    by_domain: dict = {}
    for s in specs:
        by_domain.setdefault(s.domain, []).append(s.name)
    for dom, names in sorted(by_domain.items()):
        print(f"  {dom}: {len(names)} tools")
        for n in names[:3]:
            print(f"    - {n}")
    print("\nre-run with --json to emit a compose document")
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    """Generate a synthetic multi-domain catalog to a compose document.

    Useful for trying the funnel at realistic scale without wiring up real
    tools: it produces a valid compose document you can immediately feed to
    ``route`` / ``bench``.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from examples.generate_catalog import build_catalog  # type: ignore
    except Exception:
        # Inside the zipapp, examples/ is bundled at the archive root.
        base = sys.path[0] if sys.path else os.getcwd()
        sys.path.insert(0, os.path.join(base, "examples"))
        try:
            from generate_catalog import build_catalog  # type: ignore
        except Exception:
            print(
                "catalog generator not available in this distribution.\n"
                "Use the source checkout (python examples/generate_catalog.py) "
                "or write your own compose document.",
                file=sys.stderr,
            )
            return 2
    doc = build_catalog(args.tools)
    out = args.out or "catalog.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(doc['tools'])} tools, "
          f"{len(doc['domains'])} domains, {len(doc['stages'])} stages")
    print(f"try it:  tool-funnel route {out} \"帮我查一下订单\"")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a self-contained demo on the bundled catalog (no other files needed).

    This is the "just downloaded it" experience: it loads the catalog packaged
    with the library and routes a few queries, so a fresh clone, an installed
    wheel, or a standalone ``.pyz`` shows a real result without the user
    supplying anything.
    """
    if args.file:
        if not os.path.exists(args.file):
            print(f"catalog not found: {args.file}", file=sys.stderr)
            return 2
        source = args.file
        funnel = _load(args.file)
    else:
        doc = _read_bundled_catalog()
        if doc is None:
            # Fall back to the source-tree path (editable checkouts).
            path = _bundled("examples", "compose.example.json")
            if not os.path.exists(path):
                print(
                    "bundled example catalog could not be loaded.\n"
                    "Pass --file path/to/catalog.json to use your own.",
                    file=sys.stderr,
                )
                return 2
            source = path
            funnel = _load(path)
        else:
            source = "packaged example (toolfunnel.data)"
            funnel = build_from_dict(doc)

    queries = args.queries or [
        "帮我查一下订单 12345 走到哪了",
        "客户申请退款，帮我处理一下售后工单",
    ]
    print(f"tool-funnel {__version__} -- demo on {source}")
    print(f"catalog: {len(funnel.registry)} tools, {len(funnel.gate.names)} domains\n")
    for q in queries:
        r = funnel.route(q)
        print(f"> {q}")
        print(f"  {r.summary()}")
        for m in r.tools:
            pre = f"  prefill={m.prefilled}" if m.prefilled else ""
            print(f"    - {m.name:<22} {m.score:.2f} [{m.reason}]{pre}")
        print()
    print("next steps:")
    print("  tool-funnel bench --file examples/compose.example.json")
    print("  python -c \"import toolfunnel; help(toolfunnel)\"   # Python API")
    print("  see README.md for MCP, custom catalogs and the four layers")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
