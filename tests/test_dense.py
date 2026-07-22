from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from hybrid_rag.dense import DenseHit, DenseRetriever, rrf_fuse_sparse_dense


class Encoded(list):
    def tolist(self) -> list:
        return list(self)


class FakeModel:
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> Encoded:
        del show_progress_bar
        return Encoded([[float(len(text)), 1.0] for text in texts])


def test_dense_index_preserves_opaque_doc_ids() -> None:
    retriever = DenseRetriever(model=FakeModel())

    assert retriever.index([("opaque-id", "alpha"), ("id/with#symbols", "beta")]) == 2
    assert retriever._doc_ids == ["opaque-id", "id/with#symbols"]


def test_dense_index_rejects_duplicate_doc_ids() -> None:
    retriever = DenseRetriever(model=FakeModel())

    with pytest.raises(ValueError, match="unique doc_id"):
        retriever.index([("same", "alpha"), ("same", "beta")])


def test_dense_search_returns_ranked_doc_ids(monkeypatch) -> None:
    fake_numpy = SimpleNamespace(
        linalg=SimpleNamespace(norm=lambda vector: math.sqrt(sum(x * x for x in vector))),
        dot=lambda left, right: sum(
            value * right[index] for index, value in enumerate(left)
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    retriever = DenseRetriever(model=FakeModel())
    retriever.index([("short", "a"), ("long", "abcdef")])

    hits = retriever.search("abcde", top_k=2)

    assert [hit.doc_id for hit in hits] == ["long", "short"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert all(-1.0 <= hit.cosine_score <= 1.0 for hit in hits)


def test_weighted_rrf_uses_ranks_not_cosine() -> None:
    fused = rrf_fuse_sparse_dense(
        [{"doc_id": "a"}, {"doc_id": "b"}],
        [
            DenseHit("b", cosine_score=0.01, rank=1),
            DenseHit("a", cosine_score=0.99, rank=2),
        ],
        sparse_k=10,
        dense_k=20,
        alpha=0.1,
    )
    by_id = {hit["doc_id"]: hit for hit in fused}

    assert by_id["a"]["fused_score"] == pytest.approx(0.1 / 11 + 0.9 / 22)
    assert by_id["b"]["fused_score"] == pytest.approx(0.1 / 12 + 0.9 / 21)
    assert by_id["a"]["dense_score"] == 0.99
    assert fused[0]["doc_id"] == "b"


def test_rrf_fuses_union_and_keeps_dense_only_candidate() -> None:
    fused = rrf_fuse_sparse_dense(
        [{"doc_id": "sparse", "text": "lexical"}],
        [("dense-only", 0.8)],
        alpha=0.2,
    )

    assert [hit["doc_id"] for hit in fused] == ["dense-only", "sparse"]
    assert fused[0]["sparse_rank"] is None
    assert fused[0]["dense_rank"] == 1
