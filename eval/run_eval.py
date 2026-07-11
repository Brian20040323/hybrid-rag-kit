"""Ablation: fusion mode + alpha sweep, write Markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from hybrid_rag.pipeline import RAGPipeline


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return (sum(1 for x in top if x in relevant) / len(top)) if top else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def source_id(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def evaluate(pipe: RAGPipeline, cases: list[dict], k: int = 3) -> dict:
    ps, rs, ms = [], [], []
    for case in cases:
        hits = pipe.search(case["query"], top_k=k)
        ranked = [source_id(h.doc_id) for h in hits]
        # dedupe sources while keeping order
        seen, ordered = set(), []
        for s in ranked:
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        relevant = set(case["relevant"])
        ps.append(precision_at_k(ordered, relevant, k))
        rs.append(recall_at_k(ordered, relevant, k))
        ms.append(mrr(ordered, relevant))
    n = len(ps) or 1
    return {"p": sum(ps) / n, "r": sum(rs) / n, "mrr": sum(ms) / n}


def main() -> None:
    cases = yaml.safe_load((Path(__file__).with_name("cases.yaml")).read_text(encoding="utf-8"))["cases"]
    rows = []
    for fusion in ("rrf", "linear"):
        alphas = [0.55] if fusion == "rrf" else [0.3, 0.5, 0.7]
        for alpha in alphas:
            pipe = RAGPipeline(fusion=fusion, alpha=alpha)
            n = pipe.ingest_dir(ROOT / "corpus")
            metrics = evaluate(pipe, cases, k=3)
            rows.append({"fusion": fusion, "alpha": alpha, "chunks": n, **metrics})
            print(f"fusion={fusion} alpha={alpha} chunks={n} P@3={metrics['p']:.3f} R@3={metrics['r']:.3f} MRR={metrics['mrr']:.3f}")

    report = Path(__file__).with_name("ABLATION_REPORT.md")
    lines = [
        "# Ablation Report",
        "",
        "| fusion | alpha | chunks | P@3 | R@3 | MRR |",
        "|--------|------:|-------:|----:|----:|----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fusion']} | {r['alpha']} | {r['chunks']} | {r['p']:.3f} | {r['r']:.3f} | {r['mrr']:.3f} |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "- `rrf` 不依赖手工 alpha，排序更稳，适合作为默认融合。",
        "- `linear` 可在消融中观察稀疏/稠密（TF-IDF）权重敏感度。",
        "- 指标按 **source 文档** 聚合（chunk id → source），更贴近真实 RAG。",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", report)


if __name__ == "__main__":
    main()
