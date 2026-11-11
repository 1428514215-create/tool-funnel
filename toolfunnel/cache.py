"""Cold-start cache: persist the derived index and learned state to disk.

Building the inverted index is O(N) over the whole catalog -- a one-time cost,
but one that a short-lived process (a serverless function, a CLI invocation, a
test) pays on *every* start. At 5000 tools that is ~300 ms each time, which is
pure waste when the catalog has not changed.

This module writes two things side by side:

* the **index** (postings, doc lengths) -- derived from the tools, validated by a
  content signature so a stale file is detected and ignored rather than trusted;
* the **learned state** (chain graph, domain priors, stage success, prefill
  acceptance) -- already produced by ``Funnel.save_learning``.

The result: a warm start restores both in a fraction of the rebuild time, and a
mismatch silently falls back to a normal build -- never to wrong routing.

Design rule: a cache is an *optimisation*, never a correctness dependency. Every
failure path here degrades to "just rebuild".
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Tuple

from .funnel import Funnel

__all__ = ["CacheInfo", "FunnelCache", "load_or_build"]


class CacheInfo:
    """Result of a load attempt, for logging and tests."""

    def __init__(
        self,
        *,
        loaded_index: bool = False,
        loaded_learning: bool = False,
        reason: str = "",
    ) -> None:
        self.loaded_index = loaded_index
        self.loaded_learning = loaded_learning
        self.reason = reason

    @property
    def hit(self) -> bool:
        return self.loaded_index or self.loaded_learning

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CacheInfo(index={self.loaded_index}, "
            f"learning={self.loaded_learning}, reason={self.reason!r})"
        )


class FunnelCache:
    """Save/restore a funnel's compiled artifacts to a single JSON file."""

    def __init__(self, path: str) -> None:
        self.path = path

    # ---- save ------------------------------------------------------------- #
    def save(self, funnel: Funnel) -> None:
        """Write index + learned state atomically (temp file then rename)."""
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "index": funnel.registry.export_index(),
            "learning": self._learning_payload(funnel),
        }
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, self.path)  # atomic on POSIX and Windows

    def _learning_payload(self, funnel: Funnel) -> Dict[str, Any]:
        return {
            "chain": funnel.chain.to_dict(),
            "gate": funnel.gate.export_state(),
            "stage_success": {
                f"{d}\x1f{s}": list(v) for (d, s), v in funnel._stage_success.items()
            },
            "prefill_acceptance": funnel.prefiller.export_state(),
        }

    # ---- load ------------------------------------------------------------- #
    def load(self, funnel: Funnel, *, verify: bool = True) -> CacheInfo:
        """Restore artifacts into ``funnel``. Never raises on a bad cache."""
        if not os.path.exists(self.path):
            return CacheInfo(reason="no cache file")
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            return CacheInfo(reason=f"unreadable: {exc!r}")

        info = CacheInfo()
        idx = payload.get("index")
        if isinstance(idx, dict):
            if funnel.registry.load_index(idx, verify=verify):
                info.loaded_index = True
            else:
                info.reason = "index signature mismatch (catalog changed)"

        learning = payload.get("learning")
        if isinstance(learning, dict):
            try:
                self._apply_learning(funnel, learning)
                info.loaded_learning = True
            except Exception as exc:  # noqa: BLE001 - a bad cache must not break routing
                info.reason = (info.reason + f"; learning error: {exc!r}").strip("; ")
        return info

    def _apply_learning(self, funnel: Funnel, learning: Dict[str, Any]) -> None:
        from .chain import ChainGraph

        if "chain" in learning:
            funnel.chain = ChainGraph.from_dict(learning["chain"])
            funnel.chain.register_domains({t.name: t.domain for t in funnel.registry})
        if "gate" in learning:
            funnel.gate.load_state(learning["gate"])
        funnel._stage_success = {
            tuple(k.split("\x1f")): (int(v[0]), int(v[1]))  # type: ignore[arg-type]
            for k, v in dict(learning.get("stage_success", {})).items()
        }
        if "prefill_acceptance" in learning:
            funnel.prefiller.load_state(learning["prefill_acceptance"])

    # ---- convenience ------------------------------------------------------ #
    def exists(self) -> bool:
        return os.path.exists(self.path)


def load_or_build(
    funnel_factory: Callable[[], Funnel],
    cache_path: str,
    *,
    verify: bool = True,
) -> Tuple[Funnel, CacheInfo]:
    """Try the cache first, fall back to building a fresh funnel.

    ``funnel_factory`` is a zero-arg callable returning a new :class:`Funnel`
    (the registry must already contain the tools so the index can be validated).
    """
    funnel = funnel_factory()
    info = FunnelCache(cache_path).load(funnel, verify=verify)
    if not info.loaded_index:
        funnel.registry.finalize()
    return funnel, info
