from .bm25 import BM25Index
from .chunking import Chunk, chunk_text
from .dense import DenseHit, DenseRetriever, rrf_fuse_sparse_dense
from .diversity import expand_query, mmr_select
from .pipeline import RAGPipeline
from .retriever import Document, Hit, HybridRetriever
from .tokenize import tokenize

__version__ = "0.3.0"

__all__ = [
    "BM25Index",
    "Chunk",
    "chunk_text",
    "DenseHit",
    "DenseRetriever",
    "rrf_fuse_sparse_dense",
    "expand_query",
    "mmr_select",
    "RAGPipeline",
    "Document",
    "Hit",
    "HybridRetriever",
    "tokenize",
    "__version__",
]
