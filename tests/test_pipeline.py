from __future__ import annotations

import warnings

from hybrid_rag import DenseHit, Document, RAGPipeline


class FakeDenseRetriever:
    available = True
    unavailable_reason = None

    def __init__(self) -> None:
        self.documents: list[tuple[str, str]] = []

    def index(self, documents: list[tuple[str, str]]) -> int:
        self.documents = list(documents)
        return len(documents)

    def search(self, query: str, top_k: int = 5) -> list[DenseHit]:
        del query, top_k
        return [DenseHit("semantic-id", cosine_score=0.73, rank=1)]


class MissingDenseRetriever:
    available = False
    unavailable_reason = "test backend intentionally unavailable"

    def index(self, documents: list[tuple[str, str]]) -> int:
        raise AssertionError(f"index should not run: {documents}")


def test_hybrid_pipeline_indexes_ids_and_admits_dense_only_hit() -> None:
    dense = FakeDenseRetriever()
    pipeline = RAGPipeline(
        fusion="hybrid",
        alpha=0.1,
        dense_retriever=dense,
        use_query_expansion=False,
        use_mmr=False,
    )
    pipeline.ingest_documents(
        [
            Document("lexical-id", "needle term"),
            Document("semantic-id", "unrelated wording"),
        ]
    )

    hits = pipeline.search("needle", top_k=2)

    assert dense.documents == [
        ("lexical-id", "needle term"),
        ("semantic-id", "unrelated wording"),
    ]
    assert pipeline.retriever is not None
    assert pipeline.retriever.fusion == "rrf"
    assert pipeline.retrieval_mode == "hybrid"
    assert hits[0].doc_id == "semantic-id"
    assert hits[0].dense_score == 0.73
    assert hits[0].dense_rank == 1
    assert hits[0].score == hits[0].sparse_fusion_score + hits[0].dense_fusion_score


def test_hybrid_fallback_is_visible_sparse_and_warns_once() -> None:
    pipeline = RAGPipeline(
        fusion="hybrid",
        dense_retriever=MissingDenseRetriever(),
        use_query_expansion=False,
        use_mmr=False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.ingest_documents([Document("doc-a", "retrieval term")])
        pipeline.rebuild()

    hits = pipeline.search("retrieval", top_k=1)
    result = pipeline.answer_with_citations("retrieval", top_k=1)

    assert len(caught) == 1
    assert pipeline.retrieval_mode == "sparse_fallback"
    assert pipeline.dense_active is False
    assert hits[0].doc_id == "doc-a"
    assert result["retrieval_mode"] == "sparse_fallback"
    assert "intentionally unavailable" in result["dense_fallback_reason"]
    assert result["citations"][0]["dense_score"] is None


def test_default_pipeline_remains_sparse() -> None:
    pipeline = RAGPipeline(use_query_expansion=False, use_mmr=False)
    pipeline.ingest_documents([Document("doc-a", "plain sparse retrieval")])

    assert pipeline.search("sparse", top_k=1)[0].doc_id == "doc-a"
    assert pipeline.retrieval_mode == "sparse_rrf"
    assert pipeline.dense_retriever is None
