# Novelty experiment (active learning) - IN PROGRESS, NOT VERIFIED

`active.json` is written while `models/active.py` runs. Do NOT use these numbers yet:
a GP stability problem was found (one random 4,000-run training set gives 88.5 % balanced accuracy, another 93.4 %,
with the same kernels). The cause is being checked (likely the Laplace solver stopping after 100 iterations).
This folder will be replaced by checked results.
