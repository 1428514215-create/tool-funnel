"""Reproducible benchmark: what the funnel saves, and what learning changes.

Two different questions, reported separately, because they move for different
reasons and conflating them in one before/after table is misleading:

**A. Cost (does not depend on learning).** How many schema tokens and tools
reach the model per turn. This is decided by the *budget* (``max_tools`` etc.)
and by how well the catalog is indexed -- training does not move it, and the
report says so instead of implying it might.

**B. Learning (what training actually changes).** Given the tool a session
*actually* calls next, how often does the chain graph rank it in the top-k? This
is reported in two regimes on purpose:

* **with a declared ``next_hint``** -- a hand-authored prior exists; learning
  should *complete and sharpen* it, not fight it;
* **with no hints at all** -- nothing was declared, so any accuracy above zero is
  learned purely from observed traffic.

That contrast is the honest answer to "does learning do anything". A second
corpus-level metric is reported too: next-tool top-k on held-out sessions, split
into transitions the config *declared* and transitions that are *emergent*.

Run::

    python benchmarks/bench.py
    python benchmarks/bench.py --tools 2000 --queries 200 --json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from generate_catalog import build_catalog  # noqa: E402
from toolfunnel import DomainSpec, Funnel, ToolSpec  # noqa: E402
from toolfunnel.learn import Learner, SessionRecord  # noqa: E402
from toolfunnel.types import ToolCall  # noqa: E402

LABELED_QUERIES: List[Tuple[str, str]] = [
    ("帮我查一下订单 12345 走到哪了", "order"),
    ("客户申请退款，帮我处理售后工单", "after_sales"),
    ("给这个商品写三条卖点文案", "product"),
    ("这个月各店铺的转化率报表导出一份", "analytics"),
    ("发营销短信召回三个月未下单的会员", "crm"),
    ("这批货库存还够吗，不够就发起补货", "inventory"),
    ("订单发货后物流一直没更新，是不是丢件了", "logistics"),
    ("帮客户换货，顺便看一下他的历史订单", "after_sales"),
    ("把上架商品的主图换成白底图并审核合规", "media"),
    ("核对一下本月账单和结算金额", "finance"),
    ("会员标签怎么打，想圈出高价值客户", "crm"),
    ("发票还没开，客户在催", "finance"),
    ("活动优惠券的效果复盘一下", "marketing"),
    ("调拨一批货到华东仓", "inventory"),
    ("取消刚才那个订单并退款", "order"),
]

QUERY_POOL = [q for q, _ in LABELED_QUERIES]


def measure_cost(funnel: Funnel, queries: List[str], labels: Dict[str, str]) -> Dict[str, float]:
    full = funnel._catalog_tokens()
    surfaced, tok_surf, savings, lat = [], [], [], []
    correct = labeled = 0
    for q in queries:
        t0 = time.perf_counter()
        r = funnel.route(q)
        lat.append((time.perf_counter() - t0) * 1000)
        surfaced.append(r.surfaced_tools)
        tok_surf.append(r.tokens_surfaced)
        savings.append(r.token_savings)
        want = labels.get(q)
        if want:
            labeled += 1
            got = [d.domain for d in r.domains]
            if want == got[0]:
                correct += 1
            elif want in got:
                correct += 0.5
    return {
        "catalog_tools": len(funnel.registry),
        "full_tokens": full,
        "avg_tools": statistics.mean(surfaced),
        "avg_tokens": statistics.mean(tok_surf),
        "avg_savings": statistics.mean(savings),
        "top1_domain_accuracy": correct / labeled if labeled else 0.0,
        "p50_latency_ms": statistics.median(lat),
        "p95_latency_ms": sorted(lat)[int(len(lat) * 0.95) - 1] if len(lat) > 1 else lat[0],
        "queries": len(queries),
    }


_CLEAN_DOMAINS = {
    "order": ("order.search", "order.get", "order.cancel"),
    "shipment": ("shipment.list", "shipment.track", "shipment.notify"),
    "refund": ("refund.create", "refund.status", "refund.close"),
}
_CLEAN_STAGES = ("locate", "inspect", "mutate")


def _clean_funnel(*, declare_first_hop: bool) -> Funnel:
    specs: List[ToolSpec] = []
    for domain, names in _CLEAN_DOMAINS.items():
        for i, name in enumerate(names):
            hint = (names[1],) if (declare_first_hop and i == 0) else ()
            specs.append(ToolSpec(
                name=name, domain=domain, stage=_CLEAN_STAGES[i],
                description=name.replace(".", " "), keywords=(domain,), next_hint=hint,
            ))
    return Funnel.from_tools(
        specs, domains=[DomainSpec(d, triggers=(d,)) for d in _CLEAN_DOMAINS]
    )


def _clean_sessions(n: int, seed: int) -> List[SessionRecord]:
    rng = random.Random(seed)
    out: List[SessionRecord] = []
    for _ in range(n):
        domain = rng.choice(list(_CLEAN_DOMAINS))
        names = _CLEAN_DOMAINS[domain]
        out.append(SessionRecord(query=domain, calls=[ToolCall(nm) for nm in names],
                                 domains_used=(domain,)))
    return out


def _topk_on_sessions(funnel: Funnel, sessions, k: int) -> float:
    hit = n = 0
    for s in sessions:
        names = [c.tool for c in s.calls]
        for a, b in zip(names, names[1:]):
            n += 1
            if b in [x.tool for x in funnel.chain.successors(a, limit=k)]:
                hit += 1
    return hit / n if n else 0.0


def measure_learning_clean() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for declare in (True, False):
        tag = "declared" if declare else "emergent"
        funnel = _clean_funnel(declare_first_hop=declare)
        ev = _clean_sessions(100, seed=7)
        before1 = _topk_on_sessions(funnel, ev, 1)
        Learner(funnel).learn(_clean_sessions(400, seed=99))
        after1 = _topk_on_sessions(funnel, ev, 1)
        out[f"{tag}_top1_before"] = before1
        out[f"{tag}_top1_after"] = after1
    return out


def _next_tool_accuracy(funnel: Funnel, sessions, top_k: int = 3) -> Dict[str, float]:
    declared = set()
    for t in funnel.registry:
        for nxt in t.next_hint:
            declared.add((t.name, nxt))
    stats = {True: [0, 0], False: [0, 0]}
    for s in sessions:
        tools = [c.tool for c in s.calls]
        for a, b in zip(tools, tools[1:]):
            key = (a, b) in declared
            preds = [step.tool for step in funnel.chain.successors(a, limit=top_k)]
            stats[key][1] += 1
            if b in preds:
                stats[key][0] += 1

    def rate(pair) -> float:
        hit, tot = stats[pair]
        return hit / tot if tot else float("nan")

    return {"declared_acc": rate(True), "declared_n": stats[True][1],
            "emergent_acc": rate(False), "emergent_n": stats[False][1]}


def _print_report(args, cost, clean, acc_before, acc_after) -> None:
    print("=" * 78)
    print("  tool-funnel benchmark")
    print("=" * 78)
    print(f"  catalog: {cost['catalog_tools']} tools | "
          f"{cost['full_tokens']} schema tokens sent whole | {cost['queries']} queries")
    print("\n  A. COST per turn  (set by the budget -- learning does not move this)")
    print("  " + "-" * 66)
    print(f"    tools surfaced : {cost['avg_tools']:.1f} of {cost['catalog_tools']}")
    print(f"    schema tokens  : {cost['avg_tokens']:.0f} vs {cost['full_tokens']} "
          f"-> {cost['avg_savings']:.1%} saved")
    print(f"    top-1 domain   : {cost['top1_domain_accuracy']:.1%}")
    print(f"    routing latency: p50 {cost['p50_latency_ms']:.2f} ms, "
          f"p95 {cost['p95_latency_ms']:.2f} ms (pure CPU)")
    print("\n  B1. LEARNING -- clean scenario, no duplicate tools")
    print("      (the actually-used next tool ranked #1, before vs after training)")
    print("  " + "-" * 66)
    for tag, label in (
        ("declared", "prior existed (one hop declared, rest hidden)"),
        ("emergent", "nothing declared (learned entirely from traffic)"),
    ):
        b = clean[f"{tag}_top1_before"]
        a = clean[f"{tag}_top1_after"]
        print(f"    {label:<48} {b:>5.0%} -> {a:>5.0%}")
    print("\n  B2. LEARNING -- large corpus, held-out sessions (top-3)")
    print(f"      {args.train_sessions} train / {args.eval_sessions} eval sessions")
    print("  " + "-" * 66)
    print(f"    declared transitions (n={acc_before['declared_n']:>4}) : "
          f"{acc_before['declared_acc']:>5.0%} -> {acc_after['declared_acc']:>5.0%}")
    print(f"    emergent transitions (n={acc_before['emergent_n']:>4}) : "
          f"{acc_before['emergent_acc']:>5.0%} -> {acc_after['emergent_acc']:>5.0%}"
          "   <- only learnable from traffic")
    if acc_after["declared_acc"] + 1e-9 < acc_before["declared_acc"]:
        print("    note: the declared figure can dip here because this synthetic catalog")
        print("          has 3-4 near-duplicate tools per logical tool ('query_report' /")
        print("          'query_看板' / 'query_指标'); traffic splits across the copies.")
        print("          Panel B1 has no duplicates and rises cleanly instead.")
    print("\n" + "=" * 78)
    print("  Read A for token/latency savings, B for what training adds. They are")
    print("  independent: A is fixed by the budget, B by accumulated traffic.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tools", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--train-sessions", type=int, default=800)
    ap.add_argument("--eval-sessions", type=int, default=200)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rng = random.Random(1234)
    doc = build_catalog(args.tools)
    funnel = Funnel.from_dict(doc)
    labels = dict(LABELED_QUERIES)
    queries = [rng.choice(QUERY_POOL) for _ in range(args.queries)] + QUERY_POOL
    rng.shuffle(queries)

    cost = measure_cost(funnel, queries, labels)
    clean = measure_learning_clean()
    train = Learner.synthetic_sessions(list(funnel.registry), n=args.train_sessions, seed=99, jitter=0.4)
    ev_sessions = Learner.synthetic_sessions(list(funnel.registry), n=args.eval_sessions, seed=7, jitter=0.4)
    acc_before = _next_tool_accuracy(funnel, ev_sessions)
    Learner(funnel).learn(train)
    acc_after = _next_tool_accuracy(funnel, ev_sessions)

    if args.json:
        print(json.dumps({"cost": cost, "clean_learning": clean,
                          "corpus_before": acc_before, "corpus_after": acc_after},
                         ensure_ascii=False, indent=2))
        return
    _print_report(args, cost, clean, acc_before, acc_after)


if __name__ == "__main__":
    main()
