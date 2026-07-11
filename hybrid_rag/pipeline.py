"""Ingest markdown/text -> chunk -> index pipeline + citation helper."""

from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .diversity import expand_query, mmr_select
from .retriever import Document, Hit, HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        fusion: str = "rrf",
        alpha: float = 0.55,
        use_query_expansion: bool = True,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.fusion = fusion  # type: ignore[assignment]
        self.alpha = alpha
        self.use_query_expansion = use_query_expansion
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.docs: list[Document] = []
        self.retriever: HybridRetriever | None = None

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
        self.retriever = HybridRetriever(self.docs, alpha=self.alpha, fusion=self.fusion)  # type: ignore[arg-type]

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.retriever:
            raise RuntimeError("pipeline has no index; call ingest_* first")

        variants = expand_query(query) if self.use_query_expansion else [query]
        pooled: dict[str, Hit] = {}
        for v in variants:
            for h in self.retriever.search(v, top_k=max(top_k * 3, 10)):
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
            }
            for i, h in enumerate(hits)
        ]
        answer = (
            f"Based on retrieval: {hits[0].text if hits else 'No relevant content found.'}"
            + (f" [{1}]" if hits else "")
        )
        return {
            "query": query,
            "expanded": expand_query(query) if self.use_query_expansion else [query],
            "answer": answer,
            "context": context,
            "citations": citations,
            "hits": hits,
        }
