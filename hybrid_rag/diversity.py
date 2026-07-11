"""Query expansion + MMR diversity selection for hybrid RAG."""

from __future__ import annotations

import math
from .tokenize import tokenize


SYNONYMS = {
    "rag": ["retrieval", "augmented", "generation", "检索"],
    "agent": ["react", "tool", "智能体"],
    "mcp": ["protocol", "tools", "resources", "协议"],
    "hybrid": ["fusion", "bm25", "rrf", "混合"],
    "eval": ["metric", "precision", "recall", "mrr", "评测"],
    "chunk": ["chunking", "切块", "split"],
}


def expand_query(query: str) -> list[str]:
    """Return original query plus lightweight expanded variants (no LLM)."""
    variants = [query]
    q_toks = set(tokenize(query))
    extra: list[str] = []
    for key, syns in SYNONYMS.items():
        if key in query.lower() or key in q_toks:
            extra.extend(syns)
        for s in syns:
            if s.lower() in query.lower():
                extra.append(key)
                extra.extend(syns)
    extra = [e for e in dict.fromkeys(extra) if e.lower() not in query.lower()]
    if extra:
        variants.append(query + " " + " ".join(extra[:6]))
    return variants


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def mmr_select(
    query: str,
    candidates: list,  # list[Hit]-like with .text and .score
    top_k: int = 5,
    lambda_mult: float = 0.7,
) -> list:
    """Maximal Marginal Relevance over already-scored hits."""
    if not candidates:
        return []
    q_set = set(tokenize(query))
    selected: list = []
    remaining = list(candidates)
    tok_cache = {id(h): set(tokenize(h.text)) for h in remaining}

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_val = -1e18
        for i, h in enumerate(remaining):
            rel = h.score
            if not selected:
                div = 0.0
            else:
                div = max(_jaccard(tok_cache[id(h)], tok_cache[id(s)]) for s in selected)
            val = lambda_mult * rel - (1.0 - lambda_mult) * div
            # tiny boost for query token overlap
            val += 0.01 * _jaccard(q_set, tok_cache[id(h)])
            if val > best_val:
                best_val = val
                best_idx = i
        selected.append(remaining.pop(best_idx))
    return selected
