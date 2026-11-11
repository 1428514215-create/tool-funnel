"""Reproducible benchmark: how much does the funnel actually save?

Measures, over a synthetic-but-realistic query set against a catalog of N
tools:

* **schema tokens per turn** -- full catalog vs. funneled shortlist,
* **tools per turn** -- the raw narrowing,
* **latency** -- wall-clock time per route (the funnel is pure CPU, no model),
* **learning effect** -- savings before vs. after training on sessions.

Run::

    python benchmarks/bench.py
    python benchmarks/bench.py --tools 5000 --queries 200

Output is plain text (CI-friendly) and optionally JSON (``--json``).
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
from toolfunnel import Funnel  # noqa: E402
from toolfunnel.learn import Learner  # noqa: E402

# A pool of realistic-looking user turns paired with the domain they *should*
# route to. Deliberately mixed across domains, including cross-domain sentences
# (the hard case for a funnel). The labels let the benchmark report routing
# accuracy, which -- unlike the schema-token count -- is what learning improves.
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


def measure(funnel: Funnel, queries: List[str], labels: Dict[str, str]) -> Dict[str, float]:
    full = funnel._catalog_tokens()
    surfaced, tok_surf, savings, lat = [], [], [], []
    correct = 0
    labeled = 0
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
                correct += 0.5  # present but not top-ranked: partial credit
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


def next_tool_accuracy(funnel: Funnel, sessions, top_k: int = 3) -> Dict[str, float]:
    """Top-k accuracy at predicting the next tool from the chain graph.

    Split by whether the transition is *declared* (present in some tool's
    ``next_hint``) or *emergent* (only ever seen at runtime). The emergent set is
    the honest measure of learning: the declared set is already handled by the
    cold-start hints, so a funnel that never learned would still score on it.
    """
    declared_pairs = set()
    for t in funnel.registry:
        for nxt in t.next_hint:
            declared_pairs.add((t.name, nxt))

    stats = {True: [0, 0], False: [0, 0]}  # is_declared -> [hit, total]
    for s in sessions:
        tools = [c.tool for c in s.calls]
        for a, b in zip(tools, tools[1:]):
            key = (a, b) in declared_pairs
            preds = [step.tool for step in funnel.chain.successors(a, limit=top_k)]
            stats[key][1] += 1
            if b in preds:
                stats[key][0] += 1

    def rate(pair) -> float:
        hit, tot = stats[pair]
        return hit / tot if tot else float("nan")

    return {
        "declared_acc": rate(True), "declared_n": stats[True][1],
        "emergent_acc": rate(False), "emergent_n": stats[False][1],
    }


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

    before = measure(funnel, queries, labels)
    # Held-out sessions the funnel has never seen. They contain *undeclared*
    # cross-domain hops, so only genuine observation -- not the seed hints --
    # can predict them.
    train = Learner.synthetic_sessions(
        list(funnel.registry), n=args.train_sessions, seed=99, jitter=0.4
    )
    eval_sessions = Learner.synthetic_sessions(
        list(funnel.registry), n=args.eval_sessions, seed=7, jitter=0.4
    )
    acc_before = next_tool_accuracy(funnel, eval_sessions)

    report = Learner(funnel).learn(train)
    after = measure(funnel, queries, labels)
    acc_after = next_tool_accuracy(funnel, eval_sessions)

    if args.json:
        print(json.dumps({
            "before": before, "after": after, "learning": report.summary(),
            "next_tool_before": acc_before, "next_tool_after": acc_after,
        }, ensure_ascii=False, indent=2))
        return

    def row(label, m):
        return (f"  {label:<16}{m['catalog_tools']:>7}{m['avg_tools']:>8.1f}"
                f"{m['avg_tokens']:>9.0f}{m['full_tokens']:>10}"
                f"{m['avg_savings']:>9.1%}{m['top1_domain_accuracy']:>9.1%}"
                f"{m['p50_latency_ms']:>9.2f}{m['p95_latency_ms']:>9.2f}")

    print("=" * 96)
    print("  tool-funnel benchmark")
    print("=" * 96)
    print(f"  catalog: {before['catalog_tools']} tools | "
          f"{before['full_tokens']} schema tokens if sent whole | {args.queries} queries")
    print(f"  {report.summary()}")
    print("-" * 96)
    print(f"  {'phase':<16}{'tools':>7}{'surfaced':>8}{'tok/turn':>9}"
          f"{'tok/full':>10}{'saved':>9}{'top1 dom':>9}{'p50 ms':>9}{'p95 ms':>9}")
    print("-" * 96)
    print(row("before learn", before))
    print(row("after learn", after))
    print("-" * 96)
    print("  next-tool top-3 accuracy (held-out sessions):")
    print(f"    declared transitions  (n={acc_before['declared_n']:>4}): "
          f"{acc_before['declared_acc']:>5.1%} -> {acc_after['declared_acc']:>5.1%}")
    print(f"    emergent transitions  (n={acc_before['emergent_n']:>4}): "
          f"{acc_before['emergent_acc']:>5.1%} -> {acc_after['emergent_acc']:>5.1%}  "
          f"<- only learnable from observed traffic")
    print("=" * 96)


if __name__ == "__main__":
    main()
