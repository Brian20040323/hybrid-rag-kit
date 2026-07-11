"""Demo corpus + search."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_rag import Document, HybridRetriever


CORPUS = [
    Document("d1", "ReAct 智能体通过 Thought、Action、Observation 循环调用工具完成任务。"),
    Document("d2", "混合检索结合关键词匹配与 TF-IDF 向量相似度，提升 RAG 召回质量。"),
    Document("d3", "MCP（Model Context Protocol）让大模型以标准协议访问外部工具与知识源。"),
    Document("d4", "FastAPI 可用 SSE 将 Agent 推理步骤流式推送到前端。"),
    Document("d5", "评测集应包含 Precision@K、Recall@K 与 MRR，保证检索改动能量化。"),
]


def main() -> None:
    retriever = HybridRetriever(CORPUS, alpha=0.4)
    query = "如何用混合检索提升 RAG？"
    hits = retriever.search(query, top_k=3)
    print("Q:", query)
    for i, h in enumerate(hits, 1):
        print(
            f"{i}. {h.doc_id} score={h.score:.4f} "
            f"(kw={h.keyword_score:.3f}, tfidf={h.tfidf_score:.3f})"
        )
        print("  ", h.text)
    print(json.dumps([h.__dict__ for h in hits], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
