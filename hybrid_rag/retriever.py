"""Keyword overlap + TF-IDF hybrid retriever."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .tokenize import tokenize


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict | None = None


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    keyword_score: float
    tfidf_score: float
    meta: dict | None = None


class HybridRetriever:
    def __init__(self, docs: Iterable[Document], alpha: float = 0.45) -> None:
        self.docs = list(docs)
        self.alpha = alpha
        self._tokens = [tokenize(d.text) for d in self.docs]
        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            for t in set(toks):
                self._df[t] += 1
        self.n = max(len(self.docs), 1)

    def _keyword_score(self, q_toks: list[str], d_toks: list[str]) -> float:
        if not q_toks:
            return 0.0
        q_set = set(q_toks)
        d_set = set(d_toks)
        overlap = len(q_set & d_set)
        return overlap / len(q_set)

    def _tfidf_score(self, q_toks: list[str], d_toks: list[str]) -> float:
        if not q_toks or not d_toks:
            return 0.0
        tf = Counter(d_toks)
        dl = len(d_toks)
        score = 0.0
        for t in set(q_toks):
            if t not in tf:
                continue
            idf = math.log((self.n + 1) / (self._df[t] + 1)) + 1.0
            score += (tf[t] / dl) * idf
        # normalize by query length
        return score / math.sqrt(len(set(q_toks)))

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        q = tokenize(query)
        hits: list[Hit] = []
        for doc, toks in zip(self.docs, self._tokens):
            kw = self._keyword_score(q, toks)
            tf = self._tfidf_score(q, toks)
            hybrid = self.alpha * kw + (1.0 - self.alpha) * tf
            if hybrid <= 0:
                continue
            hits.append(
                Hit(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=hybrid,
                    keyword_score=kw,
                    tfidf_score=tf,
                    meta=doc.meta,
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
