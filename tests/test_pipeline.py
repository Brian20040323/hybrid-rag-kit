"""Tests for hybrid_rag pipeline and retriever."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_rag.pipeline import RAGPipeline
from hybrid_rag.retriever import Document, HybridRetriever


def test_ingest_and_search():
    pipe = RAGPipeline(fusion="rrf", use_query_expansion=False, use_mmr=False)
    docs = [
        Document(doc_id="d1", text="BM25 is a popular ranking function for information retrieval."),
        Document(doc_id="d2", text="TF-IDF measures word importance in document collections."),
        Document(doc_id="d3", text="RRF combines multiple ranked lists for better retrieval."),
    ]
    pipe.ingest_documents(docs)
    hits = pipe.search("BM25 ranking")
    assert len(hits) > 0
    assert hits[0].doc_id == "d1"


def test_search_returns_top_k():
    pipe = RAGPipeline(fusion="rrf", use_query_expansion=False, use_mmr=False)
    docs = [Document(doc_id=f"d{i}", text=f"document number {i} about search and retrieval") for i in range(10)]
    pipe.ingest_documents(docs)
    hits = pipe.search("search retrieval", top_k=3)
    assert len(hits) == 3


def test_answer_with_citations():
    pipe = RAGPipeline(fusion="rrf", use_query_expansion=False, use_mmr=False)
    docs = [Document(doc_id="d1", text="Python is a programming language.")]
    pipe.ingest_documents(docs)
    result = pipe.answer_with_citations("Python")
    assert result["query"] == "Python"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["doc_id"] == "d1"


def test_hybrid_retriever_bm25_mode():
    docs = [Document(doc_id="d1", text="hello world"), Document(doc_id="d2", text="goodbye moon")]
    retriever = HybridRetriever(docs, fusion="bm25")
    hits = retriever.search("hello")
    assert len(hits) > 0
    assert any(h.doc_id == "d1" for h in hits)


def test_hybrid_retriever_tfidf_mode():
    docs = [Document(doc_id="d1", text="the cat sat on the mat"), Document(doc_id="d2", text="the dog ran in the park")]
    retriever = HybridRetriever(docs, fusion="tfidf")
    hits = retriever.search("cat")
    assert len(hits) > 0
    assert any(h.doc_id == "d1" for h in hits)


def test_pipeline_raises_without_ingest():
    pipe = RAGPipeline()
    try:
        pipe.search("test")
        assert False, "should raise"
    except RuntimeError:
        pass
