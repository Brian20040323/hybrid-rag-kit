"""Hybrid retrieval: BM25 + TF-IDF with RRF / linear fusion + optional rerank."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .bm25 import BM25Index
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
    bm25_score: float
    tfidf_score: float
    rrf_score: float
    meta: dict | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    dense_rank: int | None = None
    sparse_fusion_score: float = 0.0
    dense_fusion_score: float = 0.0


Fusion = Literal["linear", "rrf", "bm25", "tfidf"]


class HybridRetriever:
    def __init__(
        self,
        docs: Iterable[Document],
        alpha: float = 0.55,
        fusion: Fusion = "rrf",
        rrf_k: int = 60,
    ) -> None:
        self.docs = list(docs)
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k
        self._tokens = [tokenize(d.text) for d in self.docs]
        self._bm25 = BM25Index(self._tokens)
        self._df: Counter[str] = Counter()
        for toks in self._tokens:
            for t in set(toks):
                self._df[t] += 1
        self.n = max(len(self.docs), 1)

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
        return score / math.sqrt(len(set(q_toks)))

    def _normalize(self, values: list[float]) -> list[float]:
        if not values:
            return values
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.0 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def search(self, query: str, top_k: int = 5, candidate_k: int = 20) -> list[Hit]:
        q = tokenize(query)
        bm25_ranked = self._bm25.rank(query, top_k=candidate_k)
        bm25_map = {i: s for i, s in bm25_ranked}

        tfidf_raw = [self._tfidf_score(q, toks) for toks in self._tokens]
        tfidf_norm = self._normalize(tfidf_raw)
        bm25_vals = [bm25_map.get(i, 0.0) for i in range(len(self.docs))]
        bm25_norm = self._normalize(bm25_vals)

        # rank positions for RRF
        bm25_pos = {i: r for r, (i, _) in enumerate(bm25_ranked, 1)}
        tfidf_order = sorted(range(len(self.docs)), key=lambda i: tfidf_raw[i], reverse=True)
        tfidf_pos = {i: r for r, i in enumerate(tfidf_order, 1) if tfidf_raw[i] > 0}

        hits: list[Hit] = []
        for i, doc in enumerate(self.docs):
            b = bm25_norm[i]
            t = tfidf_norm[i]
            rrf = 0.0
            if i in bm25_pos:
                rrf += 1.0 / (self.rrf_k + bm25_pos[i])
            if i in tfidf_pos:
                rrf += 1.0 / (self.rrf_k + tfidf_pos[i])

            if self.fusion == "rrf":
                score = rrf
            elif self.fusion == "bm25":
                score = b
            elif self.fusion == "tfidf":
                score = t
            else:
                score = self.alpha * b + (1.0 - self.alpha) * t

            if score <= 0 and b <= 0 and t <= 0:
                continue
            hits.append(
                Hit(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=score,
                    bm25_score=bm25_vals[i],
                    tfidf_score=tfidf_raw[i],
                    rrf_score=rrf,
                    meta=doc.meta,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return self._rerank(query, hits[: max(top_k * 2, top_k)])[:top_k]

    def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Lightweight lexical reranker: boost exact phrase / title-ish overlap."""
        q = query.lower()
        q_toks = set(tokenize(query))
        rescored: list[Hit] = []
        for h in hits:
            bonus = 0.0
            text_l = h.text.lower()
            if q and q in text_l:
                bonus += 0.15
            overlap = len(q_toks & set(tokenize(h.text)))
            bonus += 0.02 * overlap
            rescored.append(
                Hit(
                    doc_id=h.doc_id,
                    text=h.text,
                    score=h.score + bonus,
                    bm25_score=h.bm25_score,
                    tfidf_score=h.tfidf_score,
                    rrf_score=h.rrf_score,
                    meta=h.meta,
                )
            )
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
