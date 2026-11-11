"""Sweep tool-count and measure the funnel's advantage as a function of scale.

Answers one question with real numbers: *at what catalog size is a funnel worth
it, and where is the advantage largest?*

For each catalog size N we measure:

* **full schema tokens** -- what the naive "send everything" approach costs,
* **funnel tokens / turn** -- what the funnel actually ships,
* **token savings** -- the ratio the two imply,
* **funnel route latency (p50)** -- the funnel's own CPU cost,
* **naive select latency** -- a linear scan over every tool description, i.e.
  what you pay if you keyword-match *without* an inverted index. This is the
  CPU-side thing the funnel replaces,
* **index build time** -- one-time setup cost.

The model-prefill time is *modelled*, not measured (we cannot call a model
here); the estimate and its constant are stated explicitly so nobody mistakes it
for a measurement. Everything else is measured on this machine.

Run::

    python benchmarks/sweep.py
    python benchmarks/sweep.py --sizes 10 50 200 1000 5000 --out sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from generate_catalog import build_catalog  # noqa: E402
from toolfunnel import Funnel  # noqa: E402
from toolfunnel.registry import ToolRegistry, tokenize  # noqa: E402

QUERIES = [
    "帮我查一下订单 12345 走到哪了",
    "客户申请退款，帮我处理一下售后工单",
    "给这个商品写三条卖点文案",
    "这个月各店铺的转化率报表导出一份",
    "发营销短信召回三个月未下单的会员",
    "这批货库存还够吗，不够就发起补货",
    "订单发货后物流一直没更新，是不是丢件了",
    "核对一下本月账单和结算金额",
]

# Modelled LLM prefill throughput. Conservative mid-range for a hosted model
# (roughly 1k-4k prompt tokens/sec depending on provider and load). We use a
# single number and label it clearly as an assumption, not a measurement.
PREFILL_TOKENS_PER_SEC = 3000.0

DEFAULT_SIZES = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]


def naive_select_latency(registry: ToolRegistry, queries: List[str], repeats: int = 5) -> float:
    """Latency of keyword-matching against *every* tool, with no index.

    This is the honest baseline for "the funnel replaced a scan": same work
    (find the relevant tools from the query) done the slow way. p50 over all
    queries is returned in milliseconds.
    """
    # Pre-tokenize every tool once (a fair baseline does not re-tokenize the
    # catalog per query -- that would be a strawman).
    docs = []
    for t in registry:
        docs.append((t.name, set(tokenize(f"{t.name} {t.description} {' '.join(t.keywords)}"))))
    lats: List[float] = []
    for q in queries:
        qs = set(tokenize(q))
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = [name for name, toks in docs if qs & toks]
            lats.append((time.perf_counter() - t0) * 1000)
    return statistics.median(lats)


def measure_size(n: int, queries: List[str], build_repeats: int = 3) -> Dict[str, float]:
    doc = build_catalog(n)

    # One-time build cost (index construction), median of a few runs.
    builds = []
    for _ in range(build_repeats):
        t0 = time.perf_counter()
        funnel = Funnel.from_dict(doc)
        builds.append((time.perf_counter() - t0) * 1000)

    full_tokens = funnel._catalog_tokens()

    tok_surf, savings, lat = [], [], []
    for q in queries:
        # Warm the caches on the first pass, then measure.
        funnel.route(q)
    for q in queries:
        t0 = time.perf_counter()
        r = funnel.route(q)
        lat.append((time.perf_counter() - t0) * 1000)
        tok_surf.append(r.tokens_surfaced)
        savings.append(r.token_savings)

    avg_surf_tokens = statistics.mean(tok_surf)
    return {
        "tools": len(funnel.registry),
        "full_tokens": full_tokens,
        "surfaced_tokens": avg_surf_tokens,
        "surfaced_tools": statistics.mean([r.surfaced_tools for r in [funnel.route(q) for q in queries]]),
        "savings": statistics.mean(savings),
        "route_p50_ms": statistics.median(lat),
        "route_p95_ms": sorted(lat)[int(len(lat) * 0.95) - 1],
        "naive_select_p50_ms": naive_select_latency(funnel.registry, queries),
        "index_build_ms": statistics.median(builds),
        # Modelled, clearly labelled:
        "naive_prefill_s": full_tokens / PREFILL_TOKENS_PER_SEC,
        "funnel_prefill_s": avg_surf_tokens / PREFILL_TOKENS_PER_SEC,
        "prefill_advantage_s": (full_tokens - avg_surf_tokens) / PREFILL_TOKENS_PER_SEC,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "sweep.json"))
    args = ap.parse_args()

    rows: List[Dict[str, float]] = []
    print(f"{'tools':>6}{'full tok':>10}{'funnel tok':>11}{'saved':>8}"
          f"{'route p50':>11}{'naive ms':>10}{'build ms':>10}{'prefill win':>13}")
    print("-" * 80)
    for n in args.sizes:
        row = measure_size(n, QUERIES)
        rows.append(row)
        print(f"{row['tools']:>6}{row['full_tokens']:>10}{row['surfaced_tokens']:>11.0f}"
              f"{row['savings']:>7.1%}{row['route_p50_ms']:>10.2f}{row['naive_select_p50_ms']:>10.2f}"
              f"{row['index_build_ms']:>10.1f}{row['prefill_advantage_s']:>12.1f}s")
    print("-" * 80)
    print(f"prefill model assumes {PREFILL_TOKENS_PER_SEC:.0f} prompt tokens/sec "
          f"(modelled, not measured)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sizes": rows, "queries": QUERIES,
                   "prefill_tokens_per_sec": PREFILL_TOKENS_PER_SEC},
                  fh, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
