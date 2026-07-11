"""End-to-end pipeline demo with citations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_rag import RAGPipeline


def main() -> None:
    pipe = RAGPipeline(fusion="rrf")
    n = pipe.ingest_dir(ROOT / "corpus")
    print(f"indexed chunks: {n}")
    q = "没有向量库时怎么做混合检索？"
    result = pipe.answer_with_citations(q, top_k=3)
    print("Q:", q)
    print("A:", result["answer"])
    print("citations:", json.dumps(result["citations"], ensure_ascii=False, indent=2))
    print("--- context ---")
    print(result["context"])


if __name__ == "__main__":
    main()
