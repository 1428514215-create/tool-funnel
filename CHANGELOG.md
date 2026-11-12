# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-11-12

The "drop in your tools and go" release: an **auto-indexer** that turns a raw
pile of tools into a ready-to-route funnel, so the library no longer requires a
hand-written catalog to be useful. This is the codebase-memory move applied to
tool sets.

### Added

- **Auto-indexing (`toolfunnel.index`)** — infer the whole routing structure
  from raw tools, with every guess explainable and overridable:
  - `index_tools(tools)` → a ready-to-route `Funnel` + an `IndexReport`;
    `index_and_dump(tools)` → also the inferred compose document.
  - **Domain inference** decides business domains by TF-IDF cosine clustering
    over names/descriptions, with the **name prefix treated as a strong signal**
    (``order.search``/``order.get`` group even with no shared description words).
    An optional ``embedder`` swaps in embedding-based clustering and falls back
    to TF-IDF automatically if it raises.
  - **Stage inference** maps the *verb* in a tool name to a workflow phase
    (search/list → locate, get → inspect, create/update/cancel → mutate,
    verify/audit → check), in both English and Chinese.
  - **Chain inference** links read-like tools to the write/verify tools that
    share their object, seeding ``next_hint`` from structure alone.
  - **Trigger mining** derives each domain's layer-1 triggers from its own
    tools, using *outside-domain* IDF to suppress boilerplate ("支持精确匹配")
    while keeping defining vocabulary ("订单"); works even for single-domain
    catalogs.
  - **Bridges** are inferred from cross-domain vocabulary overlap.
  - **Free-form / unbounded tasks**: tools that do not cluster confidently are
    collected into a catch-all ``general`` domain, so an open-ended query always
    has somewhere to land instead of an empty candidate set.
  - Optional ``llm_namer`` hook lets a model name clusters; a broken or absent
    namer degrades to offline naming, never breaking the index.
- **Input adapters**: `tools_from_openapi` (OpenAPI/Swagger operations, with
  parameters preserved for pre-fill), `tools_from_names` (bare names — the
  minimum possible input), and MCP listings via the existing adapter.
- **CLI `index` command** that sniffs the input shape (MCP / OpenAPI / names)
  and writes an indexed compose document.
- `examples/autoindex_demo.py` — all three input shapes → inferred config →
  routing, end to end, offline.
- 22 tests for indexing (determinism, known-catalog recovery, boilerplate
  suppression, catch-all behaviour, embedder/LLM fallbacks, a scale guard).

### Fixed

- The previous clustering was O(n³) and took ~20 s for 200 tools; it is now
  sparse-candidate union-find, indexing 1000 tools in ~0.7 s and 2000 in ~1.4 s.

## [0.2.1] - 2026-11-12

The "download and use it" release: no functional changes to routing, purely
distribution and packaging so the project can be consumed without cloning or
building anything.

### Added

- **Standalone single-file zipapp** — `tools/build_zipapp.py` produces
  `tool-funnel.pyz`, a zip that runs on any Python 3.8+ interpreter with **no
  pip, no install and no network**. The archive bundles the package *and* the
  example catalog, so `python tool-funnel.pyz demo` shows a result immediately.
- **`python -m toolfunnel`** entry point (`toolfunnel/__main__.py`), equivalent
  to the installed `tool-funnel` command.
- **Bundled example catalog as package data** (`toolfunnel/data/`), loaded via
  `importlib.resources` so it works identically from a source checkout, an
  installed wheel, and the zipapp (which has no real filesystem paths).
- **CLI commands** `demo` (self-contained demo on the bundled catalog) and
  `catalog` (generate a synthetic multi-domain catalog to a compose document).
- **`install.sh`** — one-step installer/launcher: `--zipapp`, `--venv`, `--user`
  or plain pip, with a graceful fallback to building the `.pyz` when pip has no
  network or the environment is externally managed.
- **Release workflow** (`.github/workflows/release.yml`) — on a version tag,
  builds the wheel, sdist and zipapp, smoke-tests all three, and attaches them
  to a GitHub Release so users can download a runnable artifact.
- **`INSTALL.md`** — the download-and-run guide (three paths, FAQ).
- Packaging metadata: `MANIFEST.in`, fuller classifiers, `mcp`/`yaml`/`all`
  extras, project URLs.

### Changed

- CI now also builds and smoke-tests the zipapp and runs the `demo` / `catalog`
  / MCP demos, so a broken distribution is caught before release.

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
