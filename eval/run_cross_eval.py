"""Exploratory synthetic cross-corpus retrieval smoke check (v0.3).

This tiny set (5 synthetic documents, 12 queries) is not a generalization
benchmark. All configurations in this script are sparse.

Usage:
  python eval/run_cross_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hybrid_rag.pipeline import RAGPipeline


def source_id(chunk_id: str) -> str:
    return chunk_id.split("#", 1)[0]


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(1 for x in top if x in relevant) / len(top) if top else 0.0


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def main() -> None:
    data = yaml.safe_load(
        (Path(__file__).with_name("cross_corpus.yaml")).read_text(encoding="utf-8")
    )
    docs = data["documents"]
    cases = data["cases"]

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write documents to temp corpus
        for doc in docs:
            path = tmp / f"{doc['id']}.md"
            path.write_text(f"# {doc['title']}\n\n{doc['content']}", encoding="utf-8")

        # Evaluate sparse fusion strategies only; no model download is attempted.
        configs = [
            {"fusion": "bm25", "alpha": 1.0, "use_query_expansion": False, "use_mmr": False},
            {"fusion": "tfidf", "alpha": 0.0, "use_query_expansion": False, "use_mmr": False},
            {"fusion": "rrf", "alpha": 0.55, "use_query_expansion": True, "use_mmr": True},
            {"fusion": "linear", "alpha": 0.5, "use_query_expansion": True, "use_mmr": True},
        ]

        results = []
        for cfg in configs:
            pipe = RAGPipeline(**cfg)
            n = pipe.ingest_dir(tmp)
            ps, rs, ms = [], [], []
            per_case = []

            for case in cases:
                hits = pipe.search(case["query"], top_k=3)
                ranked = [source_id(h.doc_id) for h in hits]
                seen, ordered = set(), []
                for s in ranked:
                    if s not in seen:
                        seen.add(s)
                        ordered.append(s)
                relevant = set(case["relevant"])
                ps.append(precision_at_k(ordered, relevant, 3))
                rs.append(recall_at_k(ordered, relevant, 3))
                ms.append(mrr(ordered, relevant))
                per_case.append({
                    "query": case["query"][:60],
                    "relevant": list(case["relevant"]),
                    "hits": ordered[:3],
                    "ok": any(r in ordered[:3] for r in case["relevant"]),
                })

            n_cases = len(ps) or 1
            row = {
                "fusion": cfg["fusion"],
                "expand": cfg["use_query_expansion"],
                "mmr": cfg["use_mmr"],
                "chunks": n,
                "P@3": sum(ps) / n_cases,
                "R@3": sum(rs) / n_cases,
                "MRR": sum(ms) / n_cases,
                "detail": per_case,
            }
            results.append(row)
            print(
                f"fusion={cfg['fusion']:<7} expand={str(cfg['use_query_expansion']):<5} "
                f"mmr={str(cfg['use_mmr']):<5} P@3={row['P@3']:.3f} R@3={row['R@3']:.3f} MRR={row['MRR']:.3f}"
            )

        # Summary
        print(
            f"\nExploratory synthetic cross-corpus smoke: "
            f"{len(docs)} docs × {len(cases)} queries across 5 topics"
        )
        best = max(results, key=lambda r: r["MRR"])
        print(f"Best config: {best['fusion']} P@3={best['P@3']:.3f} R@3={best['R@3']:.3f} MRR={best['MRR']:.3f}")

        # Domain breakdown
        domains: dict[str, list] = {}
        for case in cases:
            for rel in case["relevant"]:
                domain = rel.split("_")[0]
                domains.setdefault(domain, []).append(case)

        best_config = best["detail"]
        print("\nPer-domain accuracy (best config):")
        for domain, domain_cases in sorted(domains.items()):
            d_ok = sum(1 for c in best_config if any(
                c["query"] == case["query"][:60] for case in domain_cases
            ) and c["ok"])
            print(f"  {domain}: {d_ok}/{len(domain_cases)}")


if __name__ == "__main__":
    main()
