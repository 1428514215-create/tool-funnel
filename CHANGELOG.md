# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-11

The "operate at scale" release: everything here is about keeping the funnel
effective as the catalog, the traffic, and the tool ecosystem grow. The headline
is **first-class MCP support**.

### Added

- **MCP adapter (`toolfunnel.mcp`)** — the priority feature. Ingest one or more
  [Model Context Protocol](https://modelcontextprotocol.io) servers into the
  funnel and drive calls back through it with a closed learning loop.
  - `tool_specs_from_mcp` / `MCPToolServer` — flatten an MCP `list_tools` result
    into `ToolSpec`s, inferring a domain from the name prefix and recording the
    required parameters.
  - `MCPToolRunner` — sync (`call`) and async (`acall`) execution that records
    every call into the chain graph; `tools_payload` hands the model the
    *original* MCP `inputSchema` for a funneled subset.
  - `ingest_mcp_servers` — connect → list → ingest convenience helper.
  - `merge_mcp_tools` — merge several servers with collision-safe namespacing.
  - **`structuredContent` scalars are lifted onto the session blackboard**, which
    is what makes argument pre-fill work end-to-end (a search tool's `orderId`
    flows into the next call without a round-trip).
  - **Zero `mcp` dependency**: sessions are duck-typed, so the official async
    client, an embedded sync server, or a fake in tests all work.
- **Adaptive tool budget** — `max_tools` and `hard_cap` now behave as ceilings
  rather than targets (`FunnelConfig.adaptive_budget`); a 3-tool stage is no
  longer padded to a fixed 6. Adds `score_cutoff_ratio` (drop the weak tail) and
  `token_budget` (cap surfaced schemas by tokens, not count).
- **Chain-graph exploration (bandit)** — `ChainGraph.explore` blends the exploit
  score with a UCB1 exploration term so rarely-tried successors are periodically
  re-sampled and given a chance to earn observations. Enable in routing with
  `FunnelConfig.chain_explore` / `chain_epsilon`. `ChainEdge` now tracks a
  success-weighted `success` total (`ChainEdge.rate()`).
- **Chain-graph distillation** — `ChainGraph.distill` / `Funnel.distill` prune
  weak and stale edges (dropping only edges weak on *both* weight and count) and
  cap successors per source, so a long-running funnel does not degrade. Guarded
  by `needs_distill`.
- **Cold-start cache (`toolfunnel.cache`)** — `FunnelCache` persists the built
  index and the learned state to one JSON file; a content signature makes a
  stale cache on a changed catalog detected and ignored rather than trusted.
  `load_or_build` is the fail-soft entry point.
- **Stage auto-discovery (`toolfunnel.discovery`)** — `discover_stages` infers
  workflow phases per domain by clustering tools by their normalised position in
  observed sessions; `merge_stages` blends results with hand-written specs
  without silently discarding declared intent.
- **Semantic domain fallback** — `SemanticDomainGate` embeds the *domains* (a
  handful, cost-stable) and runs only when the lexical gate is weak, rescuing
  paraphrased queries ("我的包裹怎么还没到") that match no trigger word.
- **Pre-fill acceptance learning** — `PreFiller.observe_outcome` records whether
  a suggested value survived contact with the model; a bounded source bias
  reorders only equally-specific candidates, so an explicit context value can
  never lose to a learned default.
- CLI subcommands: `discover-stages`, `distill`, `mcp-import`.
- `examples/mcp_demo.py` — a full offline MCP end-to-end demo (ingest → route →
  call → learn → pre-fill) using a fake session.
- `benchmarks/sweep.py` — measures the funnel's advantage across catalog sizes.

### Changed

- `Funnel.route` now consults the chain through `_chain_successors` (honouring
  exploration) and applies adaptive/token budgets.
- `save_learning` / `load_learning` now include pre-fill acceptance state.
- `ChainGraph.from_dict` tolerates graphs written before `success` existed.

### Fixed

- `ToolRegistry.bm25` no longer re-tokenizes every tool per term (document
  lengths are cached at `finalize()`), turning a per-route O(terms × tools) cost
  into O(tools + terms × postings). Routing latency at 1000 tools dropped from
  ~16 ms to ~1 ms.
- Sync `MCPToolRunner.call` raises a clear error instead of silently dropping an
  un-awaited coroutine when given an async session.

## [0.1.0] - 2026-11-11

Initial release.

### Added

- Four-layer funnel: business-domain gate → stage resolver → chain + BM25
  ranking → argument pre-filler.
- `ChainGraph` with time decay, success weighting and cross-domain damping.
- Offline `Learner` (chain rebuild, trigger mining) and online `Funnel.record`.
- `ToolRegistry` with an inverted index and BM25.
- Declarative catalogs via `compose` (JSON/YAML), plus a DSL.
- OpenAI / Anthropic payload adapters and a generic `ToolRunner`.
- CLI (`stats`, `route`, `bench`, `learn`), a 1000-tool synthetic catalog
  generator, benchmarks, and 58 tests. Zero runtime dependencies.
