# Results

This page tracks canonical few-shot benchmarks for reproducible research.

## Benchmark Protocol

Include for each result:
- episodes (>= 10,000 for statistical significance)  
- mean ± 95% CI (confidence interval)
- backbone architecture, data transforms, BatchNorm policy
- exact command, random seed, and environment details

## Standard Benchmarks

> Template format for future results

| Dataset | N-way | K-shot | Backbone | Episodes | Accuracy (±95% CI) | Command | Commit/Env |
|--------:|------:|-------:|----------|---------:|--------------------:|--------:|-----------:|
| miniImageNet | 5 | 1 | Conv4 | 10,000 | 49.8 ± 0.3 | `meta_learning benchmark` | `commit_hash` |
| miniImageNet | 5 | 5 | Conv4 | 10,000 | 68.2 ± 0.4 | `meta_learning benchmark` | `commit_hash` |
| tieredImageNet | 5 | 1 | ResNet12 | 10,000 | 54.1 ± 0.5 | `meta_learning benchmark` | `commit_hash` |

## Research References

Benchmarks should match or exceed these published results:
- **Prototypical Networks** (Snell et al. 2017): miniImageNet 5-way 1-shot: 49.42% ± 0.78%
- **MAML** (Finn et al. 2017): miniImageNet 5-way 1-shot: 48.70% ± 1.84%
- **Matching Networks** (Vinyals et al. 2016): miniImageNet 5-way 1-shot: 43.56% ± 0.84%