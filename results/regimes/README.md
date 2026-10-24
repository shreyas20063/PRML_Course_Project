# Do the regimes appear without labels? (GMM + PCA)

Code: `models/regimes.py`. 10,000 random arXiv-law runs. Numbers [measured]: `metrics.json`. Figure: `regimes.png`.

| view | PCA components (95 % variance) | GMM K (by BIC) | ARI vs labels | NMI | cluster purity |
|---|---|---|---|---|---|
| start features (24) | 11 | 10 | 0.08 | 0.12 | 0.56 |
| trajectory shape (r_max and ring error every 5 laps) | 3 | 10 | 0.35 | 0.57 | **0.95** |

Plain words:
- From the **start** alone, the outcome classes do not form separate clouds (ARI 0.08). The outcome depends on fine
  combinations of the start features: this is why supervised learning (Phases 2-3) is needed. The separate stripe on
  the right of the left panel is the N = 1 runs (no untracked evaders).
- From the **trajectory shape**, three arms appear in the first two principal components, one per class, and the
  unsupervised clusters are 95 % pure (each cluster is almost only one class). So success / deadlock / loss are
  natural behaviour regimes, not arbitrary thresholds. BIC prefers 10 clusters (the upper end we tried): the classes
  split into sub-regimes (e.g. fast vs slow success), which lowers the ARI although purity is high.
- Limit: escaped runs stop early; their trace after the escape is filled with the last value.
