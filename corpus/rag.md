# 混合检索与 RAG

混合检索通常融合稀疏检索（BM25）与稠密或 TF-IDF 信号。
在没有向量数据库时，BM25 + TF-IDF + RRF（倒数排名融合）仍能显著提升召回稳定性。
评测应使用 Precision@K、Recall@K、MRR，并通过 alpha / fusion 消融验证设计选择。
切块（chunking）会影响检索粒度：过长会引入噪声，过短会丢失上下文。
