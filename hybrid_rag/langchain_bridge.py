"""Optional LangChain BaseRetriever adapter around RAGPipeline.

langchain-core is an optional dependency. Core hybrid_rag imports stay free of
LangChain. Install with: pip install -e ".[langchain]"
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

HAS_LANGCHAIN = find_spec("langchain_core") is not None

if HAS_LANGCHAIN:
    from langchain_core.documents import Document as LCDocument
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field, PrivateAttr

    class LangChainRetriever(BaseRetriever):
        """Wrap RAGPipeline.search as a LangChain-compatible retriever.

        ``pipeline`` is stored as a PrivateAttr so Pydantic does not try to
        serialize the in-memory indexes.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        top_k: int = Field(default=3, ge=1)
        _pipeline: Any = PrivateAttr()

        def __init__(self, pipeline: RAGPipeline, top_k: int = 3, **kwargs: Any) -> None:
            super().__init__(top_k=top_k, **kwargs)
            self._pipeline = pipeline

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[LCDocument]:
            del run_manager
            hits = self._pipeline.search(query, top_k=self.top_k)
            docs: list[LCDocument] = []
            for hit in hits:
                meta = {
                    "doc_id": getattr(hit, "doc_id", ""),
                    "score": float(getattr(hit, "score", 0.0) or 0.0),
                    "bm25_score": float(getattr(hit, "bm25_score", 0.0) or 0.0),
                    "tfidf_score": float(getattr(hit, "tfidf_score", 0.0) or 0.0),
                    "rrf_score": float(getattr(hit, "rrf_score", 0.0) or 0.0),
                }
                extra = getattr(hit, "meta", None) or {}
                if isinstance(extra, dict):
                    meta.update({k: v for k, v in extra.items() if k not in meta})
                docs.append(LCDocument(page_content=getattr(hit, "text", "") or "", metadata=meta))
            return docs

else:
    LangChainRetriever = None  # type: ignore[misc, assignment]


def rag_pipeline_to_langchain_retriever(pipeline: RAGPipeline, top_k: int = 3) -> Any:
    """Return a LangChainRetriever or raise ImportError if langchain-core missing."""
    if not HAS_LANGCHAIN or LangChainRetriever is None:
        raise ImportError(
            "langchain-core not installed. Install optional extra: pip install -e '.[langchain]'"
        )
    return LangChainRetriever(pipeline, top_k=top_k)


__all__ = [
    "HAS_LANGCHAIN",
    "LangChainRetriever",
    "rag_pipeline_to_langchain_retriever",
]
