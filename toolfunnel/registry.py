"""Tool registry: the searchable index of every tool an agent owns.

The registry is responsible for *scale*. With 1000+ tools you cannot afford a
linear scan of all schemas in the hot path, so we maintain:

* an inverted index ``term -> {tool_name: weight}`` for fast lexical lookup, and
* per-domain / per-stage bucket indexes for O(1) narrowing.

Adding a tool is cheap; the indexes are rebuilt lazily via :meth:`finalize`.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import ToolSpec

__all__ = ["ToolRegistry", "tokenize", "CJK_STOPWORDS", "EN_STOPWORDS"]

# A light stoplist keeps the inverted index from matching on noise words.
EN_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are",
    "with", "by", "this", "that", "it", "as", "at", "from", "be", "will",
}
CJK_STOPWORDS = {
    "的", "了", "是", "在", "我", "你", "他", "它", "和", "与", "请", "帮",
    "一下", "这个", "那个", "然后", "而且", "并且", "或者", "以及",
}

_CJK = r"\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[\w]+|[{_CJK}]")


def tokenize(text: str, *, ngrams: bool = True) -> List[str]:
    """Tokenize mixed English/CJK text into index terms.

    English words are lowercased whole tokens. CJK runs are split into single
    characters *and* bi-grams, which gives decent recall on Chinese without a
    segmenter dependency.
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(lowered):
        if raw in EN_STOPWORDS or raw in CJK_STOPWORDS:
            continue
        if re.fullmatch(rf"[{_CJK}]+", raw):
            chars = list(raw)
            tokens.extend(c for c in chars if c not in CJK_STOPWORDS)
            if ngrams and len(chars) > 1:
                tokens.extend(
                    "".join(chars[i : i + 2])
                    for i in range(len(chars) - 1)
                    if "".join(chars[i : i + 2]) not in CJK_STOPWORDS
                )
        else:
            tokens.append(raw)
    return tokens


class ToolRegistry:
    """Holds every tool plus the indexes used by the funnel."""

    def __init__(self, tools: Optional[Iterable[ToolSpec]] = None) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._by_domain: Dict[str, List[str]] = defaultdict(list)
        self._by_stage: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self._by_keyword: Dict[str, List[str]] = defaultdict(list)
        self._inverted: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._df: Dict[str, int] = defaultdict(int)  # document frequency
        self._terms: Dict[str, List[str]] = {}
        self._doc_len: Dict[str, int] = {}
        if tools:
            for t in tools:
                self.add(t)
            self.finalize()

    # ---- mutation --------------------------------------------------------- #
    def add(self, tool: ToolSpec) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        self._by_domain[tool.domain].append(tool.name)
        if tool.stage:
            self._by_stage[(tool.domain, tool.stage)].append(tool.name)
        for kw in tool.keywords:
            self._by_keyword[kw.lower()].append(tool.name)
        return self

    def add_all(self, tools: Iterable[ToolSpec]) -> "ToolRegistry":
        for t in tools:
            self.add(t)
        return self

    def remove(self, name: str) -> None:
        tool = self._tools.pop(name, None)
        if tool is None:
            return
        self._by_domain[tool.domain] = [n for n in self._by_domain[tool.domain] if n != name]
        if tool.stage:
            key = (tool.domain, tool.stage)
            self._by_stage[key] = [n for n in self._by_stage[key] if n != name]
        for kw in tool.keywords:
            self._by_keyword[kw.lower()] = [
                n for n in self._by_keyword[kw.lower()] if n != name
            ]
        self.finalize()

    def extend(self, tool: ToolSpec) -> "ToolRegistry":
        """Merge/upgrade a tool in place (used when learning rewrites hints)."""
        name = tool.name
        self.remove(name)
        return self.add(tool)

    def finalize(self) -> "ToolRegistry":
        """(Re)build the inverted index. Call after bulk additions."""
        self._inverted = defaultdict(dict)
        self._df = defaultdict(int)
        self._terms = {}
        self._doc_len = {}
        for name, tool in self._tools.items():
            terms = self._tool_terms(tool)
            self._terms[name] = terms
            self._doc_len[name] = max(1, len(terms))
            # The tool's own vocabulary: name, description, keywords, domain,
            # stage. Field weights bias the router toward strong signals
            # (a keyword should outrank a word that merely appears in prose).
            weighted: Dict[str, float] = {}
            for w in tokenize(tool.name.replace("_", " ").replace(".", " ")):
                weighted[w] = max(weighted.get(w, 0.0), 3.0)
            for w in tokenize(tool.domain):
                weighted[w] = max(weighted.get(w, 0.0), 2.5)
            for w in tokenize(tool.stage or ""):
                weighted[w] = max(weighted.get(w, 0.0), 2.0)
            for kw in tool.keywords:
                for w in tokenize(kw):
                    weighted[w] = max(weighted.get(w, 0.0), 2.2)
            for w in tokenize(tool.description):
                weighted[w] = max(weighted.get(w, 0.0), 1.0)
            for term, weight in weighted.items():
                self._inverted[term][name] = weight
                self._df[term] += 1
        return self

    # ---- queries ---------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    @property
    def tools(self) -> List[ToolSpec]:
        return list(self._tools.values())

    @property
    def names(self) -> List[str]:
        return list(self._tools)

    @property
    def domains(self) -> List[str]:
        return sorted(self._by_domain)

    @property
    def stages(self) -> List[Tuple[str, str]]:
        return sorted(self._by_stage)

    def domain_tools(self, domain: str, stage: Optional[str] = None) -> List[ToolSpec]:
        if stage is not None:
            names = self._by_stage.get((domain, stage), [])
        else:
            names = self._by_domain.get(domain, [])
        return [self._tools[n] for n in names]

    def keyword_hits(self, terms: Sequence[str], *, domain: Optional[str] = None) -> Dict[str, float]:
        """Return ``{tool_name: accumulated_idf_weight}`` for the given terms.

        This is the workhorse of layer 1 & 2: instead of scanning all tools we
        only touch postings lists for the query terms.
        """
        scores: Dict[str, float] = defaultdict(float)
        n = max(1, len(self._tools))
        for term in terms:
            postings = self._inverted.get(term)
            if not postings:
                continue
            # Rare terms carry more signal than common ones.
            idf = math.log(1.0 + n / (1.0 + self._df.get(term, 1)))
            for name, weight in postings.items():
                if domain is not None and self._tools[name].domain != domain:
                    continue
                scores[name] += weight * idf
        return dict(scores)

    def bm25(self, terms: Sequence[str], *, k1: float = 1.4, b: float = 0.72) -> Dict[str, float]:
        """A small BM25 over the inverted index for tool ranking."""
        n = max(1, len(self._tools))
        dl = self._doc_len
        if not dl:  # index not built yet
            self.finalize()
            dl = self._doc_len
        avgdl = sum(dl.values()) / max(1, len(dl))
        scores: Dict[str, float] = defaultdict(float)
        for term in terms:
            postings = self._inverted.get(term)
            if not postings:
                continue
            df = self._df.get(term, 1)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for name, weight in postings.items():
                denom = weight + k1 * (1 - b + b * dl[name] / avgdl)
                scores[name] += idf * (weight * (k1 + 1)) / denom
        return dict(scores)

    def _tool_terms(self, tool: ToolSpec) -> List[str]:
        blob = " ".join(
            [tool.name, tool.domain, tool.stage or "", tool.description, *tool.keywords]
        )
        return tokenize(blob)

    # ---- arc / stats helpers --------------------------------------------- #
    def stats(self) -> Dict[str, object]:
        return {
            "tools": len(self._tools),
            "domains": len(self._by_domain),
            "stages": len(self._by_stage),
            "index_terms": len(self._inverted),
            "per_domain": {d: len(v) for d, v in sorted(self._by_domain.items())},
        }
