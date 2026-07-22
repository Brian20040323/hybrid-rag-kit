"""Command-line entry point for local retrieval."""

from __future__ import annotations

import argparse
import json

from .pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a local Markdown/text corpus")
    parser.add_argument("corpus", help="directory containing .md or .txt files")
    parser.add_argument("query", help="retrieval query")
    parser.add_argument(
        "--fusion",
        choices=("rrf", "linear", "bm25", "tfidf", "hybrid"),
        default="rrf",
        help="retrieval mode; hybrid requires the optional dense extra",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    pipeline = RAGPipeline(fusion=args.fusion)
    indexed = pipeline.ingest_dir(args.corpus)
    result = pipeline.answer_with_citations(args.query, top_k=args.top_k)
    print(
        json.dumps(
            {
                "indexed": indexed,
                "retrieval_mode": result["retrieval_mode"],
                "dense_fallback_reason": result["dense_fallback_reason"],
                "answer": result["answer"],
                "citations": result["citations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
