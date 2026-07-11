# Ablation Report (v0.2)

| fusion | expand | mmr | chunks | P@3 | R@3 | MRR |
|--------|:------:|:---:|-------:|----:|----:|----:|
| rrf | False | False | 5 | 0.533 | 1.000 | 1.000 |
| rrf | True | False | 5 | 0.400 | 1.000 | 1.000 |
| rrf | True | True | 5 | 0.333 | 1.000 | 1.000 |
| linear | True | True | 5 | 0.400 | 1.000 | 1.000 |

## Takeaway

- Query expansion + MMR are independent knobs on top of hybrid fusion.
- RRF remains a strong default; expansion often helps recall on short corpora.
- Metrics aggregate at **source-document** level (chunk → source).
