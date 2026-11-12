"""Auto-indexing: turn a raw pile of tools into a ready-to-use funnel.

The rest of the library assumes someone has already said *which tools belong to
which business domain and phase*. That is fine when you hand-write a catalog,
but it is the wrong starting point for the case this module exists to serve:

    "I have 300 tools (from MCP, an OpenAPI spec, a vendor SDK, a framework
     registry). I do not want to write routing config. Just make it work."

This is the codebase-memory move applied to tools: **scan the catalog once,
infer the structure, and hand back a funnel that already routes** -- with the
inferred config emitted as an ordinary compose document you can inspect, edit,
version, and re-load. Nothing here is magic and nothing is hidden: every guess
is written down with the evidence that produced it, and every guess can be
overridden by simply editing the emitted document.

What is inferred
----------------
1. **Business domains** -- tools are grouped by lexical similarity of their
   names/descriptions (TF-IDF cosine + agglomerative merging; optionally
   embedding-based when an ``embedder`` is supplied). Groups are named from
   their own dominant token (``order.search``/``order.get`` -> ``order``).
2. **Stages** -- within a domain, tools are ordered by an inferred *verb*:
   read-like verbs (get/search/list) come before write-like verbs
   (create/update/delete) come before review-like verbs (verify/audit). This is
   a prior, not a law; it is what ``next_hint`` chains are then built on.
3. **Chains** (`next_hint`) -- within a domain, a read verb is linked to the
   write/review verbs that share its noun (``order.search`` -> ``order.cancel``
   when both are about orders). These are only *priors*; real traffic in the
   chain graph will override them.
4. **Bridges** -- domains whose tools share vocabulary often hand off to each
   other (order <-> logistics); the strongest few cross-links are recorded.
5. **Triggers** -- each domain's gate triggers are mined from its tools' own
   keywords, so layer 1 works immediately.

Free-form / unbounded tasks
---------------------------
The funnel never assumed tasks map to exactly one domain, and this module does
not change that. When the indexer cannot confidently place a query in a domain
(no trigger fires), the funnel already falls back to bridge tools and
``expand()``. For genuinely open-ended tasks the indexer additionally records a
``general`` domain holding every tool that did not clearly cluster -- so a
"just look around" query has somewhere to land instead of an empty set.

Optional model assistance
-------------------------
Naming clusters is the one part where a model genuinely helps ("these five tools
are really about after-sales"). Pass ``llm_namer`` -- a callable
``(tool_names, sample_descriptions) -> domain_name`` -- and the indexer will use
it, but it works fully offline without one (falling back to the dominant token).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .domains import DomainSpec
from .registry import tokenize
from .stages import StageSpec
from .types import ParamSpec, ToolSpec

__all__ = [
    "IndexReport",
    "RawTool",
    "index_tools",
    "infer_specs",
    "tools_from_names",
    "tools_from_openapi",
    "VERB_GROUPS",
]

# A model hook: given a cluster's tool names and a few descriptions, return a
# short business-domain name. Deliberately optional.
LlmNamer = Callable[[List[str], List[str]], Optional[str]]


# --------------------------------------------------------------------------- #
# Raw input
# --------------------------------------------------------------------------- #
@dataclass
class RawTool:
    """The minimum the indexer needs about a tool before it is 'indexed'.

    This is intentionally the lowest common denominator across MCP listings,
    OpenAPI operations, framework registries and bare name lists -- so anything
    a user has can be expressed as a list of these.
    """

    name: str
    description: str = ""
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    keywords: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    # Preserved verbatim so MCP/OpenAPI schemas survive indexing.
    raw_schema: Optional[Dict[str, Any]] = None
    handler: Any = None

    def to_spec(self, *, domain: str, stage: Optional[str]) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            domain=domain,
            stage=stage,
            keywords=self.keywords,
            params=self.params,
            tags=self.tags,
            handler=self.handler,
        )


# --------------------------------------------------------------------------- #
# Verb taxonomy -- the prior that gives an unordered tool set an order
# --------------------------------------------------------------------------- #
# Each group is (stage_name, verbs). The *order of groups* is the assumed
# workflow order; both the stage assignment and the next_hint chains use it.
VERB_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("locate", ("search", "find", "list", "query", "lookup", "discover", "resolve")),
    ("inspect", ("get", "read", "fetch", "detail", "describe", "show", "info", "status")),
    ("mutate", ("create", "add", "update", "set", "edit", "patch", "cancel",
                "delete", "remove", "submit", "apply", "execute", "run", "send",
                "publish", "assign", "import", "upload", "merge", "close")),
    ("verify", ("verify", "validate", "check", "audit", "confirm", "reconcile",
                "test", "ping", "health", "dryrun", "preview")),
)

# Chinese verbs, so a catalogue written in Chinese indexes the same way.
VERB_GROUPS_ZH: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("locate", ("查", "搜", "找", "列表", "查询", "检索")),
    ("inspect", ("详情", "查看", "读取", "获取", "状态")),
    ("mutate", ("创建", "新建", "修改", "更新", "取消", "删除", "提交",
                "执行", "发送", "发布", "导入", "上传", "合并", "关闭")),
    ("verify", ("校验", "核对", "确认", "审核", "测试", "对账", "预览")),
)

_STAGE_ORDER = {name: i for i, (name, _v) in enumerate(VERB_GROUPS)}


def _verb_of(name: str) -> Optional[str]:
    """Return the stage implied by the *verb* part of a tool name."""
    tokens = _split_name(name)
    if not tokens:
        return None
    # Prefer the last token that is a known verb (``order.cancel`` -> cancel);
    # fall back to any token.
    known: Dict[str, str] = {}
    for stage, verbs in (*VERB_GROUPS, *VERB_GROUPS_ZH):
        for v in verbs:
            known[v] = stage
    for tok in reversed(tokens):
        if tok in known:
            return known[tok]
    # Chinese names are not whitespace-split, so scan substrings too.
    for stage, verbs in VERB_GROUPS_ZH:
        for v in verbs:
            if v in name:
                return stage
    return None


def _split_name(name: str) -> List[str]:
    parts = re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", name)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        # split camelCase
        out.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", p) or [p])
    return [t.lower() for t in out if t]


def _noun_tokens(tool: RawTool) -> set:
    """Tokens that look like the *object* a tool acts on (not the verb)."""
    toks = set(_split_name(tool.name))
    for w in tokenize(tool.description):
        toks.add(w)
    verb = _verb_of(tool.name)
    if verb:
        for stage, verbs in (*VERB_GROUPS, *VERB_GROUPS_ZH):
            toks -= set(verbs)
    # Drop very short latin noise but keep CJK single chars out too (too common).
    return {t for t in toks if len(t) >= 2 or "\u4e00" <= t <= "\u9fff" and len(t) == 1}


# --------------------------------------------------------------------------- #
# Domain inference
# --------------------------------------------------------------------------- #
def _tool_text(tool: RawTool) -> str:
    return " ".join([tool.name, tool.description, " ".join(tool.keywords)])


def _tfidf_vectors(texts: Sequence[str]) -> List[Dict[str, float]]:
    """Sparse TF-IDF vectors (dicts) for a list of documents."""
    import math as _math

    docs = [tokenize(t) for t in texts]
    df: Counter = Counter()
    for toks in docs:
        df.update(set(toks))
    n = max(1, len(docs))
    vectors: List[Dict[str, float]] = []
    for toks in docs:
        tf = Counter(toks)
        vec: Dict[str, float] = {}
        for term, c in tf.items():
            idf = _math.log(1.0 + n / (1.0 + df[term]))
            vec[term] = (1.0 + _math.log(c)) * idf
        vectors.append(vec)
    return vectors


def _cosine_sparse(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # Iterate the smaller dict.
    if len(a) > len(b):
        a, b = b, a
    num = sum(w * b.get(t, 0.0) for t, w in a.items())
    if num == 0:
        return 0.0
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _agglomerative_clusters(
    n: int,
    similarity: Callable[[int, int], float],
    *,
    threshold: float,
    candidate_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[List[int]]:
    """Group items into clusters whose members are mutually similar.

    Two design choices keep this fast enough to run on a large catalog at index
    time (the previous all-pairs agglomerative loop was O(n^3) and took ~20s for
    just 200 tools):

    * **Sparse candidates.** Instead of scoring every pair, we only score pairs
      that *share vocabulary* (an inverted index supplies them). Tools with no
      words in common are, by construction, not the same domain.
    * **Union-find single linkage.** We keep a cluster merge as an O(1) union
      rather than recomputing cluster-to-cluster averages, so the whole pass is
      roughly O(pairs · α(n)).

    Each tool starts in its own cluster; a pair above ``threshold`` is unioned.
    Deterministic: candidate pairs are sorted, so the same catalog always yields
    the same clusters.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = candidate_pairs
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        if similarity(i, j) >= threshold:
            union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    # Sort by size desc, then by the smallest member index for determinism.
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def _name_cluster(
    tools: Sequence[RawTool],
    cluster: Sequence[int],
    *,
    used: set,
    llm_namer: Optional[LlmNamer] = None,
) -> Tuple[str, str]:
    """Name a cluster and say how the name was chosen."""
    if llm_namer is not None:
        try:
            names = [tools[i].name for i in cluster]
            sample = [tools[i].description for i in cluster][:5]
            picked = llm_namer(names, sample)
            if picked:
                picked = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "_", picked).lower()
                if picked and picked not in used:
                    return picked, "llm"
        except Exception:
            pass  # a broken namer must never break indexing

    # Offline naming: the most common name-prefix token across the cluster.
    prefix_counts: Counter = Counter()
    for i in cluster:
        toks = _split_name(tools[i].name)
        if toks:
            prefix_counts[toks[0]] += 1
    for token, _c in prefix_counts.most_common():
        if token in used or len(token) < 2:
            continue
        return token, "dominant-token"

    # Fall back to a distinctive content token.
    content: Counter = Counter()
    for i in cluster:
        for t in _noun_tokens(tools[i]):
            content[t] += 1
    for token, _c in content.most_common():
        if token not in used and len(token) >= 2:
            return token, "content-token"

    idx = 1
    while f"group{idx}" in used:
        idx += 1
    return f"group{idx}", "fallback"


# --------------------------------------------------------------------------- #
# The indexer
# --------------------------------------------------------------------------- #
@dataclass
class IndexReport:
    """What the indexer decided, and the evidence for each decision."""

    tools: int = 0
    domains: List[str] = field(default_factory=list)
    domain_sizes: Dict[str, int] = field(default_factory=dict)
    domain_naming: Dict[str, str] = field(default_factory=dict)  # domain -> how named
    stages: Dict[str, List[str]] = field(default_factory=dict)  # domain -> stages
    chains: int = 0
    bridges: List[Tuple[str, str]] = field(default_factory=list)
    unclustered: int = 0
    used_embedder: bool = False
    used_llm_namer: bool = False

    def summary(self) -> str:
        return (
            f"indexed {self.tools} tools into {len(self.domains)} domains "
            f"({', '.join(self.domains) or '-'}); "
            f"{sum(len(v) for v in self.stages.values())} stages; "
            f"{self.chains} chain hints; {len(self.bridges)} bridges"
            + ("; embedder" if self.used_embedder else "")
            + ("; llm-named" if self.used_llm_namer else "")
        )


def infer_specs(
    tools: Sequence[RawTool],
    *,
    embedder: Optional[Any] = None,
    llm_namer: Optional[LlmNamer] = None,
    similarity_threshold: float = 0.30,
    max_domains: int = 24,
    min_cluster: int = 2,
) -> Tuple[List[DomainSpec], List[StageSpec], List[ToolSpec], IndexReport]:
    """Infer domains, stages and chains; return specs plus a report.

    The four return values map directly onto what :class:`Funnel` needs, so this
    is the whole auto-index in one call. :func:`index_tools` wraps it to also
    build the funnel.
    """
    report = IndexReport(tools=len(tools))
    if not tools:
        return [], [], [], report

    texts = [_tool_text(t) for t in tools]

    # Global document frequency across the whole catalog: used to suppress
    # boilerplate tokens ("supports exact match") when mining domain triggers.
    global_df: Counter = Counter()
    for idx, t in enumerate(tools):
        seen = set(_split_name(t.name))
        seen.update(w for w in t.keywords if w)
        seen.update(tokenize(texts[idx]))
        for tok in seen:
            global_df[tok] += 1

    # ---- similarity, embedding-first when available --------------------- #
    embedder_ok = False
    if embedder is not None:
        try:
            embs = embedder(texts)
            embedder_ok = True
            report.used_embedder = True

            def sim(i: int, j: int) -> float:
                return _cosine_list(embs[i], embs[j])
        except Exception:
            embedder_ok = False
            report.used_embedder = False

    # The *name prefix* is the strongest domain signal a tool catalogue usually
    # carries ("order.search" and "order.get" are obviously the same domain) and
    # it is deliberately not just another lexeme -- short names give weak TF-IDF
    # vectors, so we let a shared prefix dominate the similarity.
    prefix = [_split_name(t.name)[0] if _split_name(t.name) else "" for t in tools]

    if not embedder_ok:
        vectors = _tfidf_vectors(texts)

        def sim(i: int, j: int) -> float:
            base = _cosine_sparse(vectors[i], vectors[j])
            p = prefix[i]
            if p and p == prefix[j]:
                return max(base, 0.9)  # a shared prefix is near-certain kinship
            return base

    # ---- candidate pairs: only tools that share vocabulary ---------------- #
    # Tools with no shared words are, by construction, not the same domain, so
    # scoring them is wasted work -- this is what keeps indexing fast.
    postings: Dict[str, List[int]] = defaultdict(list)
    for idx, t in enumerate(tools):
        for term in set(tokenize(texts[idx])):
            postings[term].append(idx)
    pair_set: set = set()
    for idxs in postings.values():
        if len(idxs) > 64:
            continue  # a term in almost every tool carries no grouping signal
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_set.add((idxs[a], idxs[b]))
    # Tools sharing a name prefix must be candidates even if their descriptions
    # have no words in common (the whole point of a prefix is that it groups).
    by_prefix: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(prefix):
        if p:
            by_prefix[p].append(i)
    for members in by_prefix.values():
        if len(members) > 256:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair_set.add((members[a], members[b]))
    candidate_pairs = sorted(pair_set)

    clusters = _agglomerative_clusters(
        len(tools), sim, threshold=similarity_threshold, candidate_pairs=candidate_pairs
    )

    # ---- keep large clusters; two-tool noise and the tail -> "general" ---- #
    big = [c for c in clusters if len(c) >= min_cluster]
    tiny = [c for c in clusters if len(c) < min_cluster]
    big.sort(key=lambda c: (-len(c), c[0]))
    kept, overflow = big[:max_domains], big[max_domains:]

    tool_domain: Dict[str, str] = {}
    used_names: set = set()
    domains: List[DomainSpec] = []
    stages: List[StageSpec] = []
    specs: List[ToolSpec] = []

    for cluster in kept:
        name, how = _name_cluster(tools, cluster, used=used_names, llm_namer=llm_namer)
        used_names.add(name)
        report.domain_naming[name] = how
        if how == "llm":
            report.used_llm_namer = True
        members = [tools[i] for i in cluster]
        spec_list = _build_domain(name, members, report)
        specs.extend(spec_list)
        domains.append(
            _domain_spec(name, members, spec_list, global_df=global_df, total=len(tools))
        )
        stages.extend(_stages_for(name, spec_list, report))
        for s in spec_list:
            tool_domain[s.name] = name

    # ---- everything unclustered goes to a catch-all "general" ------------- #
    folded_idx = [i for c in (*overflow, *tiny) for i in c]
    if folded_idx:
        report.unclustered = len(folded_idx)
        members = [tools[i] for i in folded_idx]
        spec_list = _build_domain("general", members, report)
        specs.extend(spec_list)
        domains.append(
            _domain_spec(
                "general", members, spec_list,
                global_df=global_df, total=len(tools), is_general=True,
            )
        )
        stages.extend(_stages_for("general", spec_list, report))
        for s in spec_list:
            tool_domain[s.name] = "general"

    report.domains = [d.name for d in domains]
    report.domain_sizes = {
        d.name: len([s for s in specs if s.domain == d.name]) for d in domains
    }

    # ---- bridges: strongest cross-domain vocabulary overlap --------------- #
    report.bridges = _infer_bridges(domains, specs)
    bridge_map: Dict[str, List[str]] = defaultdict(list)
    for a, b in report.bridges:
        bridge_map[a].append(b)
        bridge_map[b].append(a)
    for d in domains:
        if bridge_map.get(d.name):
            d.bridges = tuple(sorted(set(bridge_map[d.name])))
    report.chains = sum(len(s.next_hint) for s in specs)
    return domains, stages, specs, report


def _cosine_list(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def _build_domain(name: str, members: Sequence[RawTool], report: IndexReport) -> List[ToolSpec]:
    """Assign stages + chain hints to one domain's tools."""
    specs: List[ToolSpec] = []
    for t in members:
        stage = _verb_of(t.name) or "inspect"
        specs.append(t.to_spec(domain=name, stage=stage))

    # Chain hints: from a read stage to the write/verify stages sharing a noun.
    by_noun: Dict[str, List[ToolSpec]] = defaultdict(list)
    for s in specs:
        for n in _noun_tokens(RawTool(name=s.name, description=s.description)):
            by_noun[n].append(s)

    for s in specs:
        s_stage = _STAGE_ORDER.get(s.stage or "", 99)
        if s_stage > 0:
            continue  # only read-like tools start a chain
        targets: List[str] = []
        for n in _noun_tokens(RawTool(name=s.name, description=s.description)):
            for cand in by_noun.get(n, []):
                if cand.name == s.name:
                    continue
                if _STAGE_ORDER.get(cand.stage or "", 99) > s_stage:
                    targets.append(cand.name)
        # De-duplicate, keep a handful, deterministic order.
        uniq = sorted(set(targets))
        s.next_hint = tuple(uniq[:3])
    return specs


def _domain_spec(
    name: str,
    members: Sequence[RawTool],
    specs: Sequence[ToolSpec],
    *,
    global_df: Optional[Counter] = None,
    total: int = 0,
    is_general: bool = False,
) -> DomainSpec:
    """Triggers mined from the domain's own tools, so L1 works immediately.

    The signal we want is the domain's **object vocabulary** -- the nouns a user
    would actually type (``订单``, ``invoice``) -- not verbs (those drive stages)
    and not boilerplate. Two filters do the work:

    * **within-domain coverage** -- the token should appear in a real share of
      the domain's tools, so a one-off word is ignored; and
    * **cross-domain rarity (IDF)** -- a token that appears in *every* domain is
      boilerplate ("supports exact match") and must be suppressed even though its
      within-domain coverage is high.

    Scoring is ``coverage_fraction / (1 + global_frequency)`` so a token like
    ``订单`` (common here, absent elsewhere) beats ``匹配`` (everywhere).
    """
    coverage: Counter = Counter()
    for t in members:
        seen = set()
        seen.update(_split_name(t.name))
        seen.update(w for w in t.keywords if w)
        seen.update(tokenize(t.description))
        for tok in seen:
            # A trigger is something a *user* would type; long fragments of a
            # tool description ("用于product业务域") never are, so cap the length.
            if 2 <= len(tok) <= 12:
                coverage[tok] += 1

    n = max(1, len(members))
    df = global_df or Counter()
    outside_total = max(0, total - n)

    def outside_share(tok: str) -> float:
        inside = coverage[tok]
        outside_df = max(0, df.get(tok, 0) - inside)
        return outside_df / outside_total if outside_total else 0.0

    def score(tok: str) -> float:
        if tok == name:
            return 2.0  # always lead with the domain name itself
        cov = coverage[tok] / n
        # What matters is how common the token is *outside* this domain: a word
        # shared by every tool *here* but absent elsewhere is this domain's
        # defining vocabulary. Only tokens that are also everywhere else
        # (boilerplate) are suppressed. With a single-domain catalog the
        # outside share is 0, so nothing is wrongly penalised.
        return cov / (1.0 + 6.0 * outside_share(tok))

    ranked = sorted(coverage, key=lambda tok: (-score(tok), -len(tok), tok))
    triggers: List[str] = []
    for tok in ranked:
        if tok in triggers:
            continue
        # Two ways to qualify: a strong within-domain signal, or being the
        # domain name itself. The first is *refused* outright when the token is
        # boilerplate (in most tools outside this domain too) -- relying on a
        # score threshold alone left such tokens sitting right at the boundary.
        distinctive = outside_share(tok) < 0.5
        if (score(tok) >= 0.15 and distinctive) or tok == name:
            triggers.append(tok)
        if len(triggers) >= 10:
            break
    if name not in triggers:
        triggers.insert(0, name)

    desc = "auto-indexed domain" if not is_general else "auto-indexed catch-all for unclustered tools"
    return DomainSpec(
        name=name,
        description=desc,
        triggers=tuple(triggers[:8]),
        prior=0.8 if is_general else 1.0,
    )


def _stages_for(domain: str, specs: Sequence[ToolSpec], report: IndexReport) -> List[StageSpec]:
    present = {s.stage for s in specs if s.stage}
    out: List[StageSpec] = []
    order = 0
    for stage_name, verbs in (*VERB_GROUPS, *VERB_GROUPS_ZH):
        if stage_name not in present:
            continue
        out.append(
            StageSpec(
                name=stage_name,
                domain=domain,
                description=f"auto-inferred {stage_name} phase",
                triggers=tuple(verbs[:4]),
                order=order,
            )
        )
        order += 1
    report.stages[domain] = [s.name for s in out]
    return out


def _infer_bridges(
    domains: Sequence[DomainSpec], specs: Sequence[ToolSpec], *, max_links: int = 3
) -> List[Tuple[str, str]]:
    """Cross-domain links where the two domains share vocabulary.

    Uses the *object* tokens (nouns), not verbs -- two domains both having a
    "list" verb is not a handoff; both acting on "order" is.
    """
    dom_tokens: Dict[str, Counter] = defaultdict(Counter)
    for s in specs:
        dom_tokens[s.domain].update(_noun_tokens(RawTool(name=s.name, description=s.description)))

    pairs: List[Tuple[float, str, str]] = []
    names = [d.name for d in domains]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = set(dom_tokens[a]) & set(dom_tokens[b])
            if not shared:
                continue
            # Jaccard-ish: shared / smaller domain vocabulary.
            denom = max(1, min(len(dom_tokens[a]), len(dom_tokens[b])))
            score = len(shared) / denom
            if score > 0.08:
                pairs.append((score, a, b))
    pairs.sort(reverse=True)
    return [(a, b) for _s, a, b in pairs[: max(len(names), max_links * len(names))]]


# --------------------------------------------------------------------------- #
# One-call entry point
# --------------------------------------------------------------------------- #
def index_tools(
    tools: Sequence[RawTool],
    *,
    embedder: Optional[Any] = None,
    llm_namer: Optional[LlmNamer] = None,
    similarity_threshold: float = 0.30,
    max_domains: int = 24,
    build: bool = True,
) -> Tuple[Any, IndexReport]:
    """Index raw tools and (by default) return a ready-to-route ``Funnel``.

    Returns ``(funnel, report)``. With ``build=False`` the funnel is ``None`` and
    you get only the inferred structure via the report -- useful for inspection
    before committing to a config.
    """
    domains, stages, specs, report = infer_specs(
        tools,
        embedder=embedder,
        llm_namer=llm_namer,
        similarity_threshold=similarity_threshold,
        max_domains=max_domains,
    )
    if not build:
        return None, report

    from .funnel import Funnel

    funnel = Funnel.from_tools(specs, domains=domains, stages=stages)
    return funnel, report


def _to_document(
    domains: Sequence[DomainSpec], stages: Sequence[StageSpec], specs: Sequence[ToolSpec]
) -> Dict[str, Any]:
    """Serialise the inferred structure as a standard compose document."""
    return {
        "config": {"top_domains": 2, "max_tools": 6, "max_bridges": 3},
        "domains": [
            {
                "name": d.name,
                "description": d.description,
                "triggers": list(d.triggers),
                "bridges": list(d.bridges),
                "prior": d.prior,
            }
            for d in domains
        ],
        "stages": [
            {
                "name": s.name,
                "domain": s.domain,
                "description": s.description,
                "triggers": list(s.triggers),
                "order": s.order,
            }
            for s in stages
        ],
        "tools": [
            {
                "name": s.name,
                "domain": s.domain,
                "stage": s.stage,
                "description": s.description,
                "keywords": list(s.keywords),
                "produces": list(s.produces),
                "next_hint": list(s.next_hint),
                "params": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "required": p.required,
                        "description": p.description,
                        "source": p.source,
                        "default": p.default,
                    }
                    for p in s.params.values()
                ],
            }
            for s in specs
        ],
    }


def index_and_dump(
    tools: Sequence[RawTool],
    *,
    embedder: Optional[Any] = None,
    llm_namer: Optional[LlmNamer] = None,
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any], IndexReport]:
    """Index tools and also return the inferred compose document.

    Returns ``(funnel, document, report)`` -- the document is the same structure
    ``build_from_dict`` consumes, so it can be written to disk, reviewed, edited
    and reloaded with no loss.
    """
    domains, stages, specs, report = infer_specs(
        tools, embedder=embedder, llm_namer=llm_namer, **kwargs
    )
    doc = _to_document(domains, stages, specs)
    from .funnel import Funnel

    funnel = Funnel.from_tools(specs, domains=domains, stages=stages)
    return funnel, doc, report


# --------------------------------------------------------------------------- #
# Input adapters -- get whatever the user has into RawTool form
# --------------------------------------------------------------------------- #
def tools_from_names(names: Iterable[str], *, descriptions: Optional[Dict[str, str]] = None,
                     keywords: Optional[Dict[str, Sequence[str]]] = None) -> List[RawTool]:
    """Build RawTools from bare names (plus optional descriptions/keywords).

    The absolute minimum input: a list of tool names. Everything else is
    inferred from the name itself.
    """
    out: List[RawTool] = []
    for n in names:
        out.append(
            RawTool(
                name=n,
                description=(descriptions or {}).get(n, ""),
                keywords=tuple((keywords or {}).get(n, ())),
            )
        )
    return out


def tools_from_openapi(spec: Dict[str, Any]) -> List[RawTool]:
    """Flatten an OpenAPI/Swagger document's operations into RawTools.

    Tool name = ``operationId`` when present, else ``verb_path``. Parameters are
    taken from the path/query/body schema so argument pre-fill has something to
    work with.
    """
    out: List[RawTool] = []
    paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue
            if not isinstance(op, dict):
                continue
            op_id = op.get("operationId")
            name = op_id or f"{method.lower()}_{re.sub(r'[^0-9A-Za-z]+', '_', path).strip('_')}"
            params: Dict[str, ParamSpec] = {}
            for p in list(op.get("parameters", []) or []):
                if not isinstance(p, dict) or "name" not in p:
                    continue
                schema = p.get("schema", {}) if isinstance(p.get("schema"), dict) else {}
                params[p["name"]] = ParamSpec(
                    name=p["name"],
                    type=str(schema.get("type", "string")),
                    required=bool(p.get("required", False)),
                    description=str(p.get("description", "")),
                )
            out.append(
                RawTool(
                    name=name,
                    description=str(op.get("summary") or op.get("description") or ""),
                    params=params,
                    keywords=tuple(t for t in _split_name(name + " " + path) if t),
                    tags=tuple(op.get("tags", []) or []),
                    raw_schema={"path": path, "method": method.upper()},
                )
            )
    return out
