# Hybrid RAG Kit

> 轻量 **混合检索** 工具包：关键词 + TF-IDF 融合打分 · 可复现 Eval · 零向量库依赖。  
> 借鉴 BM25/混合检索思路，**实现与评测脚本自研**，适合简历上的「检索 / RAG 基建」条目。

## 亮点

| 亮点 | 说明 |
|------|------|
| Hybrid Score | `α * keyword + (1-α) * tfidf`，可调融合权重 |
| 可解释排序 | 返回每条命中的 keyword/tfidf/hybrid 分项 |
| Eval Harness | YAML case + Precision@K / Recall@K / MRR |
| 纯 Python | 无 Faiss / Chroma；本地即可跑通 demo 与评测 |

## 快速开始

```bash
cd hybrid-rag-kit
pip install -r requirements.txt
python -m examples.demo_search
python -m eval.run_eval
```

## 与开源的区别

- **借鉴**：稀疏检索 + 稠密/词向量混合的产品思路  
- **自研**：简易分词、TF-IDF、融合器、YAML Eval（诚实标注：非生产级向量库）

## License

MIT
