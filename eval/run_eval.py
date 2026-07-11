"""Ablation: fusion + expansion + MMR, write Markdown + HTML report."""

from __future__ import annotations

import html
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
    configs = [
        {"fusion": "rrf", "alpha": 0.55, "use_query_expansion": False, "use_mmr": False},
        {"fusion": "rrf", "alpha": 0.55, "use_query_expansion": True, "use_mmr": False},
        {"fusion": "rrf", "alpha": 0.55, "use_query_expansion": True, "use_mmr": True},
        {"fusion": "linear", "alpha": 0.5, "use_query_expansion": True, "use_mmr": True},
    ]
    for cfg in configs:
        pipe = RAGPipeline(**cfg)
        n = pipe.ingest_dir(ROOT / "corpus")
        metrics = evaluate(pipe, cases, k=3)
        row = {**cfg, "chunks": n, **metrics}
        rows.append(row)
        print(
            f"fusion={cfg['fusion']} expand={cfg['use_query_expansion']} mmr={cfg['use_mmr']} "
            f"P@3={metrics['p']:.3f} R@3={metrics['r']:.3f} MRR={metrics['mrr']:.3f}"
        )

    report = Path(__file__).with_name("ABLATION_REPORT.md")
    lines = [
        "# Ablation Report (v0.2)",
        "",
        "| fusion | expand | mmr | chunks | P@3 | R@3 | MRR |",
        "|--------|:------:|:---:|-------:|----:|----:|----:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['fusion']} | {r['use_query_expansion']} | {r['use_mmr']} | {r['chunks']} | "
            f"{r['p']:.3f} | {r['r']:.3f} | {r['mrr']:.3f} |"
        )
    lines += [
        "",
        "## Takeaway",
        "",
        "- Query expansion + MMR are independent knobs on top of hybrid fusion.",
        "- RRF remains a strong default; expansion often helps recall on short corpora.",
        "- Metrics aggregate at **source-document** level (chunk → source).",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")

    html_path = Path(__file__).with_name("ABLATION_REPORT.html")
    rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(r['fusion']))}</td>"
        f"<td>{r['use_query_expansion']}</td>"
        f"<td>{r['use_mmr']}</td>"
        f"<td>{r['chunks']}</td>"
        f"<td>{r['p']:.3f}</td>"
        f"<td>{r['r']:.3f}</td>"
        f"<td>{r['mrr']:.3f}</td>"
        "</tr>"
        for r in rows
    )
    html_path.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Hybrid RAG Ablation</title>
<style>
body{{font-family:Segoe UI,sans-serif;background:#0f1419;color:#e7ecf3;padding:32px}}
table{{border-collapse:collapse;width:100%;max-width:900px}}
th,td{{border:1px solid #2a3a55;padding:10px;text-align:left}}
th{{background:#1a2332}}
tr:nth-child(even){{background:#141b26}}
h1{{letter-spacing:-0.02em}}
</style></head>
<body>
<h1>Hybrid RAG Kit — Ablation Report</h1>
<p>v0.2 independent retrieval stack (BM25 + TF-IDF + RRF + expansion + MMR)</p>
<table>
<thead><tr><th>fusion</th><th>expand</th><th>mmr</th><th>chunks</th><th>P@3</th><th>R@3</th><th>MRR</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body></html>
""",
        encoding="utf-8",
    )
    print("wrote", report)
    print("wrote", html_path)


if __name__ == "__main__":
    main()
