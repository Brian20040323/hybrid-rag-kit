"""Run Precision@K / Recall@K / MRR over YAML cases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from hybrid_rag import Document, HybridRetriever


CORPUS = [
    Document("d1", "ReAct 智能体通过 Thought、Action、Observation 循环调用工具完成任务。"),
    Document("d2", "混合检索结合关键词匹配与 TF-IDF 向量相似度，提升 RAG 召回质量。"),
    Document("d3", "MCP（Model Context Protocol）让大模型以标准协议访问外部工具与知识源。"),
    Document("d4", "FastAPI 可用 SSE 将 Agent 推理步骤流式推送到前端。"),
    Document("d5", "评测集应包含 Precision@K、Recall@K 与 MRR，保证检索改动能量化。"),
]


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x in relevant) / len(top)


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def main() -> None:
    cases_path = Path(__file__).with_name("cases.yaml")
    data = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    retriever = HybridRetriever(CORPUS, alpha=0.4)
    k = 3
    p_scores, r_scores, mrr_scores = [], [], []

    for case in data["cases"]:
        hits = retriever.search(case["query"], top_k=k)
        ranked = [h.doc_id for h in hits]
        relevant = set(case["relevant"])
        p = precision_at_k(ranked, relevant, k)
        r = recall_at_k(ranked, relevant, k)
        m = mrr(ranked, relevant)
        p_scores.append(p)
        r_scores.append(r)
        mrr_scores.append(m)
        print(f"Q: {case['query']}")
        print(f"  ranked={ranked} relevant={sorted(relevant)}")
        print(f"  P@{k}={p:.3f} R@{k}={r:.3f} MRR={m:.3f}")

    n = len(p_scores) or 1
    print("-" * 40)
    print(f"AVG P@{k}={sum(p_scores)/n:.3f} R@{k}={sum(r_scores)/n:.3f} MRR={sum(mrr_scores)/n:.3f}")


if __name__ == "__main__":
    main()
