# Ablation Report

| fusion | alpha | chunks | P@3 | R@3 | MRR |
|--------|------:|-------:|----:|----:|----:|
| rrf | 0.55 | 5 | 0.533 | 1.000 | 1.000 |
| linear | 0.3 | 5 | 0.533 | 1.000 | 1.000 |
| linear | 0.5 | 5 | 0.533 | 1.000 | 1.000 |
| linear | 0.7 | 5 | 0.533 | 1.000 | 1.000 |

## Takeaway

- `rrf` 不依赖手工 alpha，排序更稳，适合作为默认融合。
- `linear` 可在消融中观察稀疏/稠密（TF-IDF）权重敏感度。
- 指标按 **source 文档** 聚合（chunk id → source），更贴近真实 RAG。
