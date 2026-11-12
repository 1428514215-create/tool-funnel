"""End-to-end demo: a simulated agent loop over the ~1000-tool catalog.

It shows the whole story without needing any API key:

1. load the catalog and build a funnel,
2. route a few realistic user turns and print what the funnel surfaced,
3. simulate the model calling the recommended chain, with argument pre-fill,
4. "train" on synthetic sessions and show that routing gets cheaper/sharper.

Run::

    python examples/demo.py
    python examples/demo.py --tools 3000
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolfunnel import Funnel, SessionContext  # noqa: E402
from toolfunnel.learn import Learner, SessionRecord  # noqa: E402
from toolfunnel.types import ToolCall, estimate_tokens  # noqa: E402

from generate_catalog import build_catalog  # noqa: E402

BAR = "=" * 72


def header(title: str) -> None:
    print("\n" + BAR)
    print(f"  {title}")
    print(BAR)


def demo_route(funnel: Funnel) -> None:
    header("1. Funnel routing (no model call)")
    queries = [
        "帮我查一下订单 12345 走到哪了",
        "客户申请退款，帮我处理一下售后工单",
        "给这个商品写三条卖点文案",
        "这个月各店铺的转化率报表导出一份",
        "发一条营销短信召回三个月未下单的会员",
    ]
    for q in queries:
        r = funnel.route(q)
        print(f"\n> {q}")
        print(f"  {r.summary()}")
        for m in r.tools[:5]:
            pre = f"  prefill={m.prefilled}" if m.prefilled else ""
            print(f"    - {m.name:<40} {m.score:.2f} [{m.reason}]{pre}")
        if r.bridges:
            print(f"    bridges: {', '.join(m.name for m in r.bridges)}")


def demo_chain(funnel: Funnel) -> None:
    header("2. Call-chain recommendation + argument pre-fill")
    q = "帮我查一下订单 12345 走到哪了"
    ctx = SessionContext({"order_id": "12345"})
    r = funnel.route(q, context=ctx)
    print(f"> {q}")
    print(f"  surfaced: {', '.join(r.all_names[:6])}")

    # Pretend the model calls a tool that has a declared chain; the funnel then
    # recommends the next step and pre-fills what it can.
    first = next(
        (m for m in r.tools if funnel.chain.successors(m.name)), r.tools[0]
    )
    args = {"order_id": ctx.get("order_id", "12345")}
    print(f"\n  model calls: {first.name}({args})  [prefilled={first.prefilled}]")
    funnel.record(first.name, args=args,
                  produced={"order_id": "12345", "status": "in_transit"}, context=ctx)

    nxt = funnel.chain.successors(first.name, limit=3)
    print("  funnel recommends next:")
    for s in nxt:
        print(f"    -> {s.tool:<40} score={s.score:.2f} ({s.reason})")

    # Show multi-step prefetch.
    chains = funnel.prefetch(first.name, depth=3)
    if chains:
        print("  probable chains (for pre-fetch):")
        for path in chains[:3]:
            print("    " + " -> ".join(s.tool for s in path))


def demo_expand(funnel: Funnel) -> None:
    header("3. Escape hatch: the model asks for a tool the funnel missed")
    ctx = SessionContext()
    q = "订单里有个字段想核对一下"
    r = funnel.route(q, context=ctx)
    print(f"> {q}")
    print(f"  funnel surfaced: {', '.join(r.all_names[:4])}")
    # Pick a tool that genuinely exists in the catalog but that the funnel did
    # not surface for this query.
    surfaced = set(r.all_names)
    missing = next(
        (
            t.name
            for t in funnel.registry
            if t.name not in surfaced and t.domain == "finance" and t.stage == "invoice"
        ),
        None,
    )
    if missing is None:
        print("  (no off-path tool available to demonstrate)")
        return
    match = funnel.expand(missing, context=ctx)
    print(f"  model: I need {missing!r} -> expand() returns it: {match is not None}")
    print("  (the funnel records this correction, so next time it ranks higher)")


def demo_learning(funnel: Funnel) -> None:
    header("4. Learning: what training actually changes")
    print("  (token count does NOT move with learning -- it is fixed by the")
    print("   budget. What moves is whether the NEXT tool is ranked correctly.)\n")

    # A clean 3-domain scenario with no duplicate tools and one hidden chain, so
    # the effect is unambiguous. Compare two regimes: a declared prior exists,
    # vs nothing declared at all.
    def build(declare):
        from toolfunnel import DomainSpec, ToolSpec

        doms = {
            "order": ("order.search", "order.get", "order.cancel"),
            "shipment": ("shipment.list", "shipment.track", "shipment.notify"),
            "refund": ("refund.create", "refund.status", "refund.close"),
        }
        stages = ("locate", "inspect", "mutate")
        specs = []
        for d, names in doms.items():
            for i, nm in enumerate(names):
                hint = (names[1],) if (declare and i == 0) else ()
                specs.append(ToolSpec(nm, domain=d, stage=stages[i],
                                      description=nm.replace(".", " "),
                                      keywords=(d,), next_hint=hint))
        return Funnel.from_tools(specs, domains=[DomainSpec(d, triggers=(d,)) for d in doms]), doms

    def sessions(doms, n, seed):
        import random
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            d = rng.choice(list(doms))
            out.append(SessionRecord(query=d,
                                     calls=[ToolCall(x) for x in doms[d]],
                                     domains_used=(d,)))
        return out

    def next_top1(f, sess):
        hit = n = 0
        for s in sess:
            names = [c.tool for c in s.calls]
            for a, b in zip(names, names[1:]):
                n += 1
                if b in [x.tool for x in f.chain.successors(a, limit=1)]:
                    hit += 1
        return hit / max(1, n)

    for declare, label in (
        (True, "with a declared prior (one hop given, rest hidden)"),
        (False, "with NO declared hints (learned purely from traffic)"),
    ):
        f, doms = build(declare)
        ev = sessions(doms, 100, seed=7)
        before = next_top1(f, ev)
        Learner(f).learn(sessions(doms, 400, seed=99))
        after = next_top1(f, ev)
        print(f"  {label}")
        print(f"    next-tool ranked #1:  {before:.0%}  ->  {after:.0%}")
        print()


def demo_tokens(funnel: Funnel, queries) -> None:
    header("5. Token-savings reality check")
    full = sum(estimate_tokens(str(t.as_schema())) for t in funnel.registry)
    total_saved = 0.0
    print(f"  full catalog: {len(funnel.registry)} tools =~ {full} schema tokens per turn\n")
    print(f"  {'query':<34}{'tools':>7}{'tokens':>9}{'saved':>8}")
    print("  " + "-" * 56)
    for q in queries:
        r = funnel.route(q)
        label = q if len(q) <= 32 else q[:31] + "…"
        print(f"  {label:<34}{r.surfaced_tools:>7}{r.tokens_surfaced:>9}{r.token_savings:>7.1%}")
        total_saved += r.token_savings
    print("  " + "-" * 56)
    print(f"  {'average':<34}{'':>7}{'':>9}{total_saved/len(queries):>7.1%}")
    print(f"\n  funnel stats: {funnel.stats_dict()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", type=int, default=1000)
    args = ap.parse_args()

    print(f"building catalog with {args.tools} tools ...")
    doc = build_catalog(args.tools)
    funnel = Funnel.from_dict(doc)
    print(f"registry: {funnel.registry.stats()}")

    demo_route(funnel)
    demo_chain(funnel)
    demo_expand(funnel)
    demo_learning(funnel)
    demo_tokens(funnel, [
        "帮我查一下订单 12345 走到哪了",
        "客户申请退款，帮我处理一下售后工单",
        "给这个商品写三条卖点文案",
        "这个月各店铺的转化率报表导出一份",
        "发一条营销短信召回三个月未下单的会员",
    ])


if __name__ == "__main__":
    main()
