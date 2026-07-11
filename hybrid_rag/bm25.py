"""Okapi BM25 sparse retriever."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .tokenize import tokenize


class BM25Index:
    def __init__(self, corpus_tokens: Sequence[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = list(corpus_tokens)
        self.n = len(self.corpus) or 1
        self.doc_len = [len(d) for d in self.corpus]
        self.avgdl = sum(self.doc_len) / self.n
        self.df: Counter[str] = Counter()
        for doc in self.corpus:
            for t in set(doc):
                self.df[t] += 1

    def idf(self, term: str) -> float:
        # Robertson–Sparck Jones IDF with +0.5 smoothing
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self.corpus[doc_idx]
        if not doc or not query_tokens:
            return 0.0
        tf = Counter(doc)
        dl = self.doc_len[doc_idx]
        score = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            freq = tf[t]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf(t) * (freq * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q = tokenize(query)
        scored = [(i, self.score(q, i)) for i in range(len(self.corpus))]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
