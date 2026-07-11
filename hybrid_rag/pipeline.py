"""Ingest markdown/text -> chunk -> index pipeline + citation helper."""

from __future__ import annotations

from pathlib import Path

from .chunking import chunk_text
from .retriever import Document, Hit, HybridRetriever


class RAGPipeline:
    def __init__(self, fusion: str = "rrf", alpha: float = 0.55) -> None:
        self.fusion = fusion  # type: ignore[assignment]
        self.alpha = alpha
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
        return self.retriever.search(query, top_k=top_k)

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
        # Extractive "answer": top snippet + citation markers (no LLM required)
        answer = (
            f"基于检索结果：{hits[0].text if hits else '未找到相关内容。'}"
            + (f" [{1}]" if hits else "")
        )
        return {"query": query, "answer": answer, "context": context, "citations": citations, "hits": hits}
