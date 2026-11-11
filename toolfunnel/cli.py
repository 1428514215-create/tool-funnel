"""Command-line interface for tool-funnel.

    tool-funnel stats    FILE.json        # catalog + routing stats
    tool-funnel route    FILE.json "..."  # route one query, show the funnel
    tool-funnel bench    FILE.json        # token-savings benchmark
    tool-funnel learn    FILE.json SPEC   # learn from a session corpus

The ``FILE`` is a compose document (see :mod:`toolfunnel.compose`).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .compose import build_from_dict, load_document
from .funnel import Funnel
from .learn import Learner, SessionRecord
from .types import ToolCall


def _load(path: str) -> Funnel:
    return build_from_dict(load_document(path))


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
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
