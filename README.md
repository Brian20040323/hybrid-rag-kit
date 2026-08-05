# Hybrid RAG Kit v0.3.0

小型检索套件：分块 → BM25 + TF-IDF → 排序融合 → 查询扩展 → MMR → 抽取式引用上下文。默认走稀疏路径，无需向量库或嵌入模型。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**关联：** [lite-react-agent](https://github.com/Brian20040323/lite-react-agent) · [atlas-knowledge-agent](https://github.com/Brian20040323/atlas-knowledge-agent) · [mcp-knowledge-bridge](https://github.com/Brian20040323/mcp-knowledge-bridge)

## 检索模式

- 默认 `fusion="rrf"`：BM25 与 TF-IDF 做 RRF 融合（全本地稀疏路径）
- `fusion="hybrid"`：可选稀疏 + dense 二段 RRF（`pip install -e ".[dense]"`）。模型不可用时回退稀疏，并如实报告 `sparse_fallback`

## 快速开始

```bash
git clone https://github.com/Brian20040323/hybrid-rag-kit.git
cd hybrid-rag-kit
pip install -e .
python -m examples.demo_search
python eval/run_eval.py        # 内置语料消融
python eval/run_cross_eval.py  # 探索性跨语料烟测
```

```python
from hybrid_rag import RAGPipeline

pipe = RAGPipeline()  # sparse BM25 + TF-IDF RRF
pipe.ingest_dir("corpus")
print(pipe.answer_with_citations("hybrid retrieval without vector DB")["citations"])
```

CLI：

```bash
hybrid-rag corpus "hybrid retrieval" --fusion rrf
```

## 诚实边界

- `answer_with_citations` 是**抽取式引用助手**，不调用 LLM，不是生成式 RAG
- 消融报告请以 `eval/ABLATION_REPORT.md` 为准。常见口径：**TF-IDF 相对纯 BM25** 在内置集上 P@3/R@3 约 +9.1%；**RRF 相对 BM25 在该集上可接近 0%**——用表选型，不喊口号
- 跨语料脚本仅为少量合成文档的烟测，不是大规模基准
- Dense 为可选内存参考实现，不是生产向量库

可选：`hybrid_rag.langchain_bridge` 提供 LangChain `BaseRetriever` 适配层；核心检索不依赖 LangChain。

## License

MIT © Brian20040323
