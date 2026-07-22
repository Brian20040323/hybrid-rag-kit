# Hybrid RAG Kit v0.3.0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-blueviolet.svg)](#retrieval-modes)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![RAG](https://img.shields.io/badge/RAG-BM25%20%2B%20TF--IDF%20%2B%20RRF%20%2B%20Dense-purple.svg)](#features)
[![Eval](https://img.shields.io/badge/eval-P%40K%20%7C%20R%40K%20%7C%20MRR%20%7C%20cross--corpus-informational.svg)](#ablation)
[![Stars](https://img.shields.io/github/stars/Brian20040323/hybrid-rag-kit?style=social)](https://github.com/Brian20040323/hybrid-rag-kit)

> A small retrieval stack: chunking → BM25 + TF-IDF → rank fusion → query
> expansion → MMR diversity → extractive citation context. The default path is
> sparse and needs no vector database or embedding model.

**Related:** [lite-react-agent](https://github.com/Brian20040323/lite-react-agent) · [atlas-knowledge-agent](https://github.com/Brian20040323/atlas-knowledge-agent) · [mcp-knowledge-bridge](https://github.com/Brian20040323/mcp-knowledge-bridge)

---

## Retrieval modes

`RAGPipeline()` defaults to `fusion="rrf"`: BM25 and TF-IDF are fused with
reciprocal rank fusion. This is the light, fully local sparse path.

`fusion="hybrid"` explicitly enables a second sparse/dense RRF stage. Install it
with `pip install -e ".[dense]"`. The default sentence-transformers backend may
download `all-MiniLM-L6-v2` on first use. If the optional dependency or model is
unavailable, the pipeline emits one warning, reports `retrieval_mode` as
`sparse_fallback`, and uses sparse RRF. It does not report fallback output as
dense retrieval.

## Quickstart

```bash
git clone https://github.com/Brian20040323/hybrid-rag-kit.git
cd hybrid-rag-kit
pip install -e .
python -m examples.demo_search
python eval/run_eval.py        # Ablation on built-in corpus
python eval/run_cross_eval.py  # Exploratory synthetic cross-corpus check
```

```python
from hybrid_rag import RAGPipeline

pipe = RAGPipeline()  # sparse BM25 + TF-IDF RRF
pipe.ingest_dir("corpus")
print(pipe.answer_with_citations("hybrid retrieval without vector DB")["citations"])
```

Explicit optional dense fusion:

```python
pipe = RAGPipeline(fusion="hybrid")
pipe.ingest_dir("corpus")
result = pipe.answer_with_citations("semantic retrieval")
print(result["retrieval_mode"])  # "hybrid" or "sparse_fallback"
```

The installed CLI provides the same local flow:

```bash
hybrid-rag corpus "hybrid retrieval" --fusion rrf
```

## Honest boundaries

`answer_with_citations` is an **extractive citation helper**: it returns the
top retrieved text as an answer-shaped value and builds citation metadata. It
does not call a language model and is not generated-answer RAG.

The cross-corpus work is exploratory: it contains only **5 synthetic documents
and 12 queries** across five topics. It is useful as a smoke check, not a
credible C2 benchmark or evidence of broad generalization.

Dense retrieval is an optional in-memory reference implementation. It is not a
production vector store and has not been evaluated as a dense benchmark here.

---

## License

MIT © Brian20040323
