"""Optional dense retrieval and sparse/dense rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DenseHit:
    """A ranked dense result identified by the caller's stable document ID."""

    doc_id: str
    cosine_score: float
    rank: int


class DenseRetriever:
    """Optional semantic retriever using sentence-transformers.

    Install: ``pip install hybrid-rag-kit[dense]``
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model: Any = model
        self._embeddings: list[Any] = []
        self._doc_ids: list[str] = []
        self.unavailable_reason: str | None = None
        self._load_attempted = model is not None

    @property
    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.unavailable_reason = None
            return True
        except ImportError:
            self.unavailable_reason = (
                "sentence-transformers is not installed; install hybrid-rag-kit[dense]"
            )
            return False
        except Exception as exc:
            self.unavailable_reason = f"dense model {self.model_name!r} could not load: {exc}"
            return False

    def index(self, documents: list[tuple[str, str]]) -> int:
        """Encode ``(doc_id, text)`` pairs, preserving their stable identities."""
        self.clear()
        if not self.available:
            return 0
        if not documents:
            return 0

        doc_ids = [doc_id for doc_id, _ in documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("dense index requires unique doc_id values")

        self._doc_ids = doc_ids
        texts = [text for _, text in documents]
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            self._embeddings.extend(
                self._model.encode(batch, show_progress_bar=False).tolist()
            )
        return len(self._embeddings)

    def search(self, query: str, top_k: int = 5) -> list[DenseHit]:
        """Return ranked document IDs with cosine similarity metadata."""
        if not self._model or not self._embeddings:
            return []
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is unavailable; install hybrid-rag-kit[dense]"
            ) from exc

        q_vec = self._model.encode([query], show_progress_bar=False)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores: list[tuple[str, float]] = []
        for index, doc_id in enumerate(self._doc_ids):
            emb = self._embeddings[index]
            e_norm = np.linalg.norm(emb)
            if e_norm == 0:
                continue
            sim = float(np.dot(q_vec, emb) / (q_norm * e_norm))
            scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            DenseHit(doc_id=doc_id, cosine_score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(scores[:top_k], 1)
        ]

    def clear(self) -> None:
        self._embeddings.clear()
        self._doc_ids.clear()


# ---------------------------------------------------------------------------
# Fusion: combine sparse + dense results via RRF
# ---------------------------------------------------------------------------

def rrf_fuse_sparse_dense(
    sparse_hits: list[dict],
    dense_hits: list[DenseHit | tuple[str, float] | dict],
    sparse_k: int = 60,
    dense_k: int = 60,
    alpha: float = 0.5,
) -> list[dict]:
    """Fuse BM25/TF-IDF hits with dense embedding results using weighted RRF.

    Args:
        sparse_hits: Ranked ``{"doc_id": str, ...}`` values from sparse retrieval.
        dense_hits: Ranked dense values carrying stable document IDs and cosine metadata.
        sparse_k: RRF constant for sparse rank
        dense_k: RRF constant for dense rank
        alpha: Sparse rank weight (dense rank weight is ``1 - alpha``).

    Returns:
        The union of both rankings, ordered by weighted rank-based RRF.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if sparse_k < 0 or dense_k < 0:
        raise ValueError("RRF constants must be non-negative")

    by_id: dict[str, dict] = {}
    sparse_rank: dict[str, int] = {}
    for rank, hit in enumerate(sparse_hits, 1):
        doc_id = str(hit["doc_id"])
        by_id.setdefault(doc_id, {}).update(hit)
        sparse_rank.setdefault(doc_id, rank)

    dense_rank: dict[str, int] = {}
    dense_cosine: dict[str, float] = {}
    for fallback_rank, hit in enumerate(dense_hits, 1):
        if isinstance(hit, DenseHit):
            doc_id, cosine, rank = hit.doc_id, hit.cosine_score, hit.rank
        elif isinstance(hit, dict):
            doc_id = str(hit["doc_id"])
            cosine = float(hit.get("cosine_score", hit.get("score", 0.0)))
            rank = int(hit.get("rank", fallback_rank))
        else:
            doc_id, cosine = str(hit[0]), float(hit[1])
            rank = fallback_rank
        by_id.setdefault(doc_id, {"doc_id": doc_id})
        if doc_id not in dense_rank or rank < dense_rank[doc_id]:
            dense_rank[doc_id] = rank
            dense_cosine[doc_id] = cosine

    fused: list[dict] = []
    for doc_id, original in by_id.items():
        sparse_component = (
            alpha / (sparse_k + sparse_rank[doc_id])
            if doc_id in sparse_rank
            else 0.0
        )
        dense_component = (
            (1.0 - alpha) / (dense_k + dense_rank[doc_id])
            if doc_id in dense_rank
            else 0.0
        )
        fused.append(
            {
                **original,
                "doc_id": doc_id,
                "sparse_rank": sparse_rank.get(doc_id),
                "dense_rank": dense_rank.get(doc_id),
                "sparse_rrf_score": sparse_component,
                "dense_rrf_score": dense_component,
                "dense_score": dense_cosine.get(doc_id),
                "fused_score": sparse_component + dense_component,
            }
        )

    fused.sort(
        key=lambda hit: (
            -hit["fused_score"],
            hit["sparse_rank"] or float("inf"),
            hit["dense_rank"] or float("inf"),
            hit["doc_id"],
        )
    )
    return fused
