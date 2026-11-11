# Contributing to tool-funnel

Thanks for your interest! This project aims to stay small, dependency-free at the
core, and easy to reason about. Contributions that keep it that way are very
welcome.

## Getting started

```bash
git clone <your-fork-url>
cd tool-funnel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q            # 58 tests, all should pass
python examples/demo.py        # end-to-end, no API key needed
python benchmarks/bench.py     # token-savings numbers
```

## Ground rules

1. **Core stays dependency-free.** `toolfunnel/types.py`, `registry.py`,
   `domains.py`, `stages.py`, `chain.py`, `funnel.py` must import only the
   standard library. Optional integrations go behind lazy imports and an extra
   in `pyproject.toml`.
2. **Routing stays deterministic by default.** Anything that needs a model must
   be an opt-in hook (`StageSpec.llm_pick`, `semantic.SemanticRanker`), never on
   the default path.
3. **Never hard-isolate.** Any change to domain/stage narrowing must preserve
   the bridge tools and the `expand()` escape hatch.
4. **Explain yourself.** New ranking signals should record a `reason` on
   `ToolMatch` and, where relevant, a line in `RouteResult.explanation`.
5. **Add a test.** Behaviour changes need a test in `tests/`. Numbers quoted in
   the README must be reproducible via `benchmarks/bench.py`.

## What would help most

- More adapters: LangChain, LlamaIndex, MCP.
- Better chain learning (e.g. exploration-aware / bandit-style).
- Automatic stage discovery from call sequences (currently stages are declared).
- Real-world evaluation reports on large catalogs.

## Style

- Python 3.8+ compatible (`from __future__ import annotations` is fine).
- Keep functions small; prefer explicit names over clever ones.
- Run `ruff check toolfunnel tests examples benchmarks` before opening a PR.

## Reporting issues

Please include: catalog size, how domains/stages are configured, the query, the
`RouteResult.explanation`, and what you expected instead. That output is designed
to make routing bugs diagnosable without guessing.
