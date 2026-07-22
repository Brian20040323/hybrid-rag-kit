"""Ingest markdown/text -> chunk -> index pipeline + citation helper."""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .dense import DenseRetriever, rrf_fuse_sparse_dense
from .diversity import expand_query, mmr_select
from .retriever import Document, Hit, HybridRetriever


class RAGPipeline:
    """Chunk, index, retrieve, and format extractive citation context."""

    _warned_dense_failures: set[str] = set()

    def __init__(
        self,
        fusion: str = "rrf",
        alpha: float = 0.55,
        use_query_expansion: bool = True,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
        dense_retriever: Any | None = None,
    ) -> None:
        if fusion not in {"linear", "rrf", "bm25", "tfidf", "hybrid"}:
            raise ValueError(f"unsupported fusion mode: {fusion}")
        self.fusion = fusion
        self.alpha = alpha
        self.use_query_expansion = use_query_expansion
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.docs: list[Document] = []
        self.retriever: HybridRetriever | None = None
        self.dense_retriever = dense_retriever
        self.dense_active = False
        self.dense_fallback_reason: str | None = None

    @property
    def retrieval_mode(self) -> str:
        if self.fusion != "hybrid":
            return f"sparse_{self.fusion}"
        return "hybrid" if self.dense_active else "sparse_fallback"

    def _warn_dense_fallback(self, reason: str) -> None:
        self.dense_fallback_reason = reason
        if reason not in self._warned_dense_failures:
            warnings.warn(
                f"Hybrid retrieval is using sparse RRF fallback: {reason}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_dense_failures.add(reason)

    def ingest_dir(self, path: str | Path) -> int:
        root = Path(path)
        count = 0
        for fp in sorted(root.rglob("*")):
            if fp.suffix.lower() not in {".md", ".txt"}:
                continue
            text = fp.read_text(encoding="utf-8")
            source_id = fp.stem
            for ch in chunk_text(source_id, text):
                self.docs.append(
                    Document(
                        doc_id=ch.chunk_id,
                        text=ch.text,
                        meta={"source": source_id, "start": ch.start, "end": ch.end, "path": str(fp)},
                    )
                )
                count += 1
        self.rebuild()
        return count

    def ingest_documents(self, docs: list[Document]) -> None:
        self.docs.extend(docs)
        self.rebuild()

    def rebuild(self) -> None:
        sparse_fusion = "rrf" if self.fusion == "hybrid" else self.fusion
        self.retriever = HybridRetriever(
            self.docs,
            alpha=self.alpha,
            fusion=sparse_fusion,  # type: ignore[arg-type]
        )
        self.dense_active = False
        self.dense_fallback_reason = None
        if self.fusion != "hybrid":
            return

        if self.dense_retriever is None:
            self.dense_retriever = DenseRetriever()
        try:
            if not self.dense_retriever.available:
                reason = getattr(
                    self.dense_retriever,
                    "unavailable_reason",
                    "dense backend is unavailable",
                )
                self._warn_dense_fallback(reason or "dense backend is unavailable")
                return
            indexed = self.dense_retriever.index(
                [(doc.doc_id, doc.text) for doc in self.docs]
            )
            if indexed != len(self.docs):
                self._warn_dense_fallback(
                    f"dense backend indexed {indexed} of {len(self.docs)} documents"
                )
                return
            self.dense_active = True
        except Exception as exc:
            self._warn_dense_fallback(f"dense indexing failed: {exc}")

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.retriever:
            raise RuntimeError("pipeline has no index; call ingest_* first")

        variants = expand_query(query) if self.use_query_expansion else [query]
        pooled: dict[str, Hit] = {}
        for v in variants:
            sparse_hits = self.retriever.search(v, top_k=max(top_k * 3, 10))
            variant_hits = sparse_hits
            if self.fusion == "hybrid" and self.dense_active:
                try:
                    dense_hits = self.dense_retriever.search(
                        v, top_k=max(top_k * 3, 10)
                    )
                    sparse_dicts = [asdict(hit) for hit in sparse_hits]
                    fused = rrf_fuse_sparse_dense(
                        sparse_dicts,
                        dense_hits,
                        alpha=self.alpha,
                    )
                    docs_by_id = {doc.doc_id: doc for doc in self.docs}
                    variant_hits = []
                    for item in fused:
                        doc = docs_by_id.get(item["doc_id"])
                        if doc is None:
                            continue
                        variant_hits.append(
                            Hit(
                                doc_id=doc.doc_id,
                                text=doc.text,
                                score=item["fused_score"],
                                bm25_score=item.get("bm25_score", 0.0),
                                tfidf_score=item.get("tfidf_score", 0.0),
                                rrf_score=item.get("rrf_score", 0.0),
                                meta=doc.meta,
                                dense_score=item.get("dense_score"),
                                sparse_rank=item.get("sparse_rank"),
                                dense_rank=item.get("dense_rank"),
                                sparse_fusion_score=item["sparse_rrf_score"],
                                dense_fusion_score=item["dense_rrf_score"],
                            )
                        )
                except Exception as exc:
                    self.dense_active = False
                    self._warn_dense_fallback(f"dense search failed: {exc}")
                    variant_hits = sparse_hits

            for h in variant_hits:
                prev = pooled.get(h.doc_id)
                if prev is None or h.score > prev.score:
                    pooled[h.doc_id] = h
        candidates = sorted(pooled.values(), key=lambda x: x.score, reverse=True)
        if self.use_mmr:
            return mmr_select(query, candidates, top_k=top_k, lambda_mult=self.mmr_lambda)
        return candidates[:top_k]

    def answer_with_citations(self, query: str, top_k: int = 3) -> dict:
        hits = self.search(query, top_k=top_k)
        context = "\n\n".join(f"[{i+1}] ({h.doc_id}) {h.text}" for i, h in enumerate(hits))
        citations = [
            {
                "ref": i + 1,
                "doc_id": h.doc_id,
                "source": (h.meta or {}).get("source"),
                "score": round(h.score, 4),
                "bm25_score": round(h.bm25_score, 4),
                "tfidf_score": round(h.tfidf_score, 4),
                "rrf_score": round(h.rrf_score, 6),
                "sparse_fusion_score": round(h.sparse_fusion_score, 6),
                "dense_fusion_score": round(h.dense_fusion_score, 6),
                "dense_score": (
                    round(h.dense_score, 4) if h.dense_score is not None else None
                ),
                "sparse_rank": h.sparse_rank,
                "dense_rank": h.dense_rank,
            }
            for i, h in enumerate(hits)
        ]
        answer = (
            f"Based on retrieval: {hits[0].text if hits else 'No relevant content found.'}"
            + (f" [{1}]" if hits else "")
        )
        return {
            "query": query,
            "retrieval_mode": self.retrieval_mode,
            "dense_fallback_reason": self.dense_fallback_reason,
            "expanded": expand_query(query) if self.use_query_expansion else [query],
            "answer": answer,
            "context": context,
            "citations": citations,
            "hits": hits,
        }
