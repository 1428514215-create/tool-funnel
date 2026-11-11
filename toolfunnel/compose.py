"""Declarative composition: describe a whole funnel in a dict / JSON / YAML file.

This is the "make it universal" piece. Rather than writing Python for every
project, you describe your tool catalog, business domains and workflow stages in
one document and hand it to :func:`build_from_dict`. Parsing is strict enough to
catch typos but forgiving enough that partial specs work.

Document shape::

    {
      "config": {"max_tools": 6, "top_domains": 2},
      "domains": [
        {"name": "order", "triggers": ["订单", "order", "退款"],
         "bridges": ["logistics"], "prior": 1.2}
      ],
      "stages": [
        {"name": "locate", "domain": "order", "order": 0,
         "triggers": ["查", "查询", "找"]},
        {"name": "mutate", "domain": "order", "order": 2,
         "triggers": ["改", "取消"]}
      ],
      "tools": [
        {"name": "order_search", "domain": "order", "stage": "locate",
         "description": "...", "keywords": ["订单", "搜索"],
         "produces": ["order_id"],
         "params": [{"name": "keyword", "required": true}],
         "next_hint": ["order_get"]}
      ]
    }

YAML is supported when PyYAML is installed; JSON always works with the standard
library.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .domains import DomainSpec
from .registry import ToolRegistry
from .stages import StageSpec
from .types import ParamSpec, ToolSpec

__all__ = ["build_from_dict", "load_document", "loads", "to_dict"]


def build_from_dict(data: Dict[str, Any], *, cls: Any = None) -> Any:
    """Construct a :class:`~toolfunnel.funnel.Funnel` from a document dict."""
    from .funnel import Funnel, FunnelConfig

    if cls is None:
        cls = Funnel

    tools = [_tool(t) for t in data.get("tools", [])]
    domains = [_domain(d) for d in data.get("domains", [])]
    stages = [_stage(s) for s in data.get("stages", [])]
    cfg = FunnelConfig(**data.get("config", {})) if data.get("config") else None

    # Infer domains from tools when the author did not list them, so a bare
    # tool catalog still routes sensibly.
    if not domains:
        seen: Dict[str, List[str]] = {}
        for t in tools:
            seen.setdefault(t.domain, []).extend(t.keywords)
        domains = [
            DomainSpec(name=d, triggers=tuple(dict.fromkeys(kws)), prior=1.0)
            for d, kws in seen.items()
        ]

    funnel = cls(
        registry=ToolRegistry(tools),
        domains=domains,
        stages=stages,
        config=cfg,
    )
    return funnel


# --------------------------------------------------------------------------- #
def _tool(d: Dict[str, Any]) -> ToolSpec:
    params: Dict[str, ParamSpec] = {}
    raw_params = d.get("params", {})
    if isinstance(raw_params, dict):
        items = [{"name": k, **v} for k, v in raw_params.items()]
    else:
        items = list(raw_params)
    for p in items:
        pspec = ParamSpec(
            name=p["name"],
            type=p.get("type", "string"),
            required=bool(p.get("required", False)),
            description=p.get("description", ""),
            enum=p.get("enum"),
            source=p.get("source", "context"),
            default=p.get("default"),
            aliases=tuple(p.get("aliases", ())),
        )
        params[pspec.name] = pspec
    return ToolSpec(
        name=d["name"],
        description=d.get("description", ""),
        domain=d.get("domain", "general"),
        stage=d.get("stage"),
        keywords=tuple(d.get("keywords", ())),
        params=params,
        requires=tuple(d.get("requires", ())),
        produces=tuple(d.get("produces", ())),
        next_hint=tuple(d.get("next_hint", ())),
        bridges=tuple(d.get("bridges", ())),
        tags=tuple(d.get("tags", ())),
        cost=float(d.get("cost", 1.0)),
        dangerous=bool(d.get("dangerous", False)),
    )


def _domain(d: Dict[str, Any]) -> DomainSpec:
    return DomainSpec(
        name=d["name"],
        description=d.get("description", ""),
        triggers=tuple(d.get("triggers", ())),
        stopwords=tuple(d.get("stopwords", ())),
        bridges=tuple(d.get("bridges", ())),
        prior=float(d.get("prior", 1.0)),
    )


def _stage(d: Dict[str, Any]) -> StageSpec:
    return StageSpec(
        name=d["name"],
        domain=d["domain"],
        description=d.get("description", ""),
        triggers=tuple(d.get("triggers", ())),
        order=int(d.get("order", 0)),
    )


# --------------------------------------------------------------------------- #
def load_document(path: str) -> Dict[str, Any]:
    """Read a JSON or YAML document from disk."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        return _parse_yaml(text)
    return json.loads(text)


def loads(text: str) -> Dict[str, Any]:
    """Parse a JSON or YAML string (auto-detected)."""
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(text)
    return _parse_yaml(text)


def _parse_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "YAML support requires PyYAML: pip install tool-funnel[yaml]"
        ) from exc
    return yaml.safe_load(text)


def to_dict(funnel: Any) -> Dict[str, Any]:
    """Serialise a funnel's *definition* (config, tools, domains, stages).

    Learned state is separate -- see :meth:`Funnel.save_learning`.
    """
    return {
        "config": {
            "top_domains": funnel.config.top_domains,
            "max_tools": funnel.config.max_tools,
            "max_bridges": funnel.config.max_bridges,
            "chain_width": funnel.config.chain_width,
        },
        "domains": [
            {
                "name": s.name,
                "description": s.description,
                "triggers": list(s.triggers),
                "stopwords": list(s.stopwords),
                "bridges": list(s.bridges),
                "prior": s.prior,
            }
            for s in funnel.gate.specs.values()
        ],
        "stages": [
            {
                "name": s.name,
                "domain": s.domain,
                "description": s.description,
                "triggers": list(s.triggers),
                "order": s.order,
            }
            for s in funnel.stages.specs.values()
        ],
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "domain": t.domain,
                "stage": t.stage,
                "keywords": list(t.keywords),
                "requires": list(t.requires),
                "produces": list(t.produces),
                "next_hint": list(t.next_hint),
                "params": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "required": p.required,
                        "description": p.description,
                        "source": p.source,
                        "default": p.default,
                    }
                    for p in t.params.values()
                ],
            }
            for t in funnel.registry
        ],
    }
