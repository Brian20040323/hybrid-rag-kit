# Hybrid RAG Kit
[![CI](https://github.com/Brian20040323/hybrid-rag-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Brian20040323/hybrid-rag-kit/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-blueviolet.svg)](#whats-new-in-v020)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-BM25%20%2B%20TF--IDF%20%2B%20RRF-purple.svg)](#features)
[![Eval](https://img.shields.io/badge/eval-P%40K%20%7C%20R%40K%20%7C%20MRR-informational.svg)](#ablation)
[![Stars](https://img.shields.io/github/stars/Brian20040323/hybrid-rag-kit?style=social)](https://github.com/Brian20040323/hybrid-rag-kit)

> **v0.2** independent hybrid retrieval stack: chunking → BM25 + TF-IDF → **RRF/linear fusion** → **query expansion** → **MMR diversity** → citations → **Markdown + HTML ablation reports**.  
> No Faiss / Chroma / paid embeddings required to learn the hard parts of RAG.

**Related:** [lite-react-agent](https://github.com/Brian20040323/lite-react-agent) · [mcp-knowledge-bridge](https://github.com/Brian20040323/mcp-knowledge-bridge)

---

## What's new in v0.2

| Addition | Independence angle |
|----------|-------------------|
| **Query expansion** | Synonym/domain expanders without an LLM |
| **MMR selection** | Diversity vs relevance trade-off you can tune |
| **Richer ablation** | fusion × expand × mmr grid |
| **HTML report** | Shareable eval artifact for READMEs / interviews |

---

## Quickstart

```bash
git clone https://github.com/Brian20040323/hybrid-rag-kit.git
cd hybrid-rag-kit
pip install -r requirements.txt
python -m examples.demo_search
python eval/run_eval.py
# opens eval/ABLATION_REPORT.md and eval/ABLATION_REPORT.html
```

```python
from hybrid_rag import RAGPipeline

pipe = RAGPipeline(fusion="rrf", use_query_expansion=True, use_mmr=True)
pipe.ingest_dir("corpus")
print(pipe.answer_with_citations("hybrid retrieval without vector DB")["citations"])
```

---

## Honest boundaries

Sparse hybrid ≠ production dense retrieval. This kit proves you understand **fusion, diversity, and measurement** — the parts most wrapper tutorials skip.

---

## License

MIT © Brian20040323
