# Ablation Report (v0.3)

| fusion | expand | mmr | chunks | P@3 | R@3 | MRR |
|--------|:------:|:---:|-------:|----:|----:|----:|
| bm25 | False | False | 11 | 0.458 | 0.917 | 0.875 |
| tfidf | False | False | 11 | 0.500 | 1.000 | 0.917 |
| rrf | False | False | 11 | 0.458 | 0.917 | 0.875 |
| rrf | True | False | 11 | 0.361 | 0.833 | 0.792 |
| rrf | True | True | 11 | 0.347 | 1.000 | 0.847 |
| linear | True | True | 11 | 0.417 | 0.917 | 0.833 |

## Baseline comparison (same corpus / cases, no expand / no MMR)

- Pure BM25:  P@3=0.458, R@3=0.917, MRR=0.875
- Pure TF-IDF: P@3=0.500, R@3=1.000, MRR=0.917
- Hybrid RRF: P@3=0.458, R@3=0.917, MRR=0.875
- TF-IDF vs BM25: P@3 +9.1%, R@3 +9.1%, MRR +4.8%
- RRF vs BM25: P@3 +0.0%, R@3 +0.0%, MRR +0.0%

_Note: metrics on built-in eval set (12 queries, source-doc level)._

## Takeaway

- Always report pure BM25 / TF-IDF baselines beside hybrid fusion.
- On this set, TF-IDF can beat BM25; RRF may tie BM25 — publish the table, not a slogan.
- Query expansion + MMR trade precision; use them as tunable knobs.
- Metrics aggregate at **source-document** level (chunk → source).
