"""Tests for optional LangChain BaseRetriever adapter."""

from __future__ import annotations

import pytest

from hybrid_rag import Document, RAGPipeline
from hybrid_rag.langchain_bridge import HAS_LANGCHAIN, rag_pipeline_to_langchain_retriever


def test_missing_langchain_raises_import_error() -> None:
    if HAS_LANGCHAIN:
        pytest.skip("langchain-core installed in this environment")
    pipe = RAGPipeline(fusion="rrf")
    pipe.ingest_documents([Document("d1", "hybrid retrieval without vectors")])
    with pytest.raises(ImportError, match="langchain-core"):
        rag_pipeline_to_langchain_retriever(pipe)


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
def test_langchain_retriever_invoke_preserves_order_and_metadata() -> None:
    pipe = RAGPipeline(fusion="rrf", use_query_expansion=False, use_mmr=False)
    pipe.ingest_documents(
        [
            Document("react", "ReAct agents interleave reasoning and tool actions."),
            Document("rag", "Hybrid retrieval combines BM25 and dense ranking."),
        ]
    )
    retriever = rag_pipeline_to_langchain_retriever(pipe, top_k=2)
    docs = retriever.invoke("ReAct agents")
    assert 1 <= len(docs) <= 2
    assert all(hasattr(d, "page_content") for d in docs)
    assert "doc_id" in docs[0].metadata
    assert "score" in docs[0].metadata
    # Top hit should prefer the ReAct document for this query.
    assert docs[0].metadata["doc_id"] == "react" or "ReAct" in docs[0].page_content
