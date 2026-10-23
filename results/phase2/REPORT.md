# Phase 2: baseline classifiers (outcome: success / deadlock / loss)

Date: 2026-10-23. All numbers are [measured] on the held-out test set. Code: `models/features.py`, `models/baselines.py`.
Full numbers: `results/phase2/metrics.json`. Figure: `results/phase2/confusion.png`.

## Setup (plain words)

- **Input**: 26 features of the START of a run only (no information from the simulation itself). They do not depend
  on the order of the evaders and work for any N (list in `metrics.json`, "features"). Lengths are divided by kappa.
- **Test set**: 20 % of the runs, chosen at random but with the same mix of labels and N as the whole set
  ("stratified"). It is never used for choosing settings. arXiv law: 48,000 train / 12,000 test. Circle law: 8,000 / 2,000.
- **Settings** are chosen by cross-validation on the training set only (5 folds for logistic regression; 3 folds on a
  6,000-run subset for the SVM, which is then trained on 20,000 runs, because SVM cost grows with n^2).
- **Main score: balanced accuracy** = the average of the three per-class recalls. Recall of a class = the fraction of
  runs of that class that the model gets right. Plain accuracy would favour the largest class.

## Results, arXiv law (main)

| model | accuracy | balanced accuracy | recall success / deadlock / loss |
|---|---|---|---|
| always "success" (largest class) | 0.441 | 0.333 | 1.00 / 0.00 / 0.00 |
| theory rule: success if a > 1, else loss | 0.471 | 0.457 | 0.48 / 0.00 / 0.89 |
| logistic regression (C = 0.1) | 0.830 | 0.825 | 0.85 / 0.79 / 0.84 |
| **RBF-SVM** (C = 1000, gamma = 0.003) | **0.941** | **0.939** | 0.95 / 0.93 / 0.93 |

Unseen N (train on N <= 4, test on N = 5 and 6; 36,000 / 24,000 runs):

| model | accuracy | balanced accuracy | recall success / deadlock / loss |
|---|---|---|---|
| logistic regression | 0.815 | 0.811 | 0.73 / 0.79 / 0.91 |
| **RBF-SVM** | **0.927** | **0.928** | 0.94 / 0.94 / 0.91 |

SVM by N (test set, balanced accuracy): N=1 0.98 (only two classes there; no losses), N=2 0.91, N=3 0.93,
N=4 0.94, N=5 0.94, N=6 0.94.

## Results, circle law (baseline, one evader; features: log gamma, r0, cos psi0, sin psi0)

| model | accuracy | balanced accuracy | recall success / deadlock / loss |
|---|---|---|---|
| always "success" | 0.411 | 0.333 | 1.00 / 0.00 / 0.00 |
| theory rule: success if gamma < 0.385, else loss | 0.518 | 0.466 | 1.00 / 0.00 / 0.40 |
| logistic regression | 0.921 | 0.918 | 0.92 / 0.98 / 0.86 |
| **RBF-SVM** (C = 1000, gamma = 0.1) | **0.991** | **0.991** | 0.99 / 1.00 / 0.99 |

## What this shows (and what it does not)

1. **The outcome is very predictable from the start alone**: the RBF-SVM gets 94 % balanced accuracy on the arXiv law.
   The boundary is curved: the linear model (logistic regression) is about 11 points worse.
2. **It transfers to unseen N**: trained on N <= 4, it still gets 93 % on N = 5, 6. So the features capture
   something that does not depend on the flock size.
3. **The theory rule is a weak predictor** (46 %), as expected: Theorem 4 is a local result about the equilibrium,
   not a prediction of the outcome. 52 % of the successful runs have a <= 1 (outside the theorem), and 11 % of
   the lost runs have a > 1. This is the gap the project is about.
4. **The circle law is almost perfectly separable** (99 %), as expected from Phase 0: one evader, only 3 numbers.
5. Honest limits:
   - Deadlock is easy partly because it is mostly "too slow" (small beta), and beta is a feature.
   - The SVM's best settings are at the edge of the search grid (C = 1000, gamma = 0.003). A wider grid (C up to
     1000, gamma from 0.003 to 0.3) changed the test score by only 0.1 point (0.940 -> 0.939), so the score has levelled off.
   - The SVM gives labels, not calibrated probabilities. Probabilities and uncertainty are Phase 3 (GP classifier).
   - These are simulation results with our label numbers (A2); a different T_max would change the deadlock class.
6. Logistic-regression weights (features standardised), largest first:
   success: oth_r_mean (-1.04), kappa (-0.50), r_mean (+0.45), log_a (+0.42); loss: log_q_over_kappa (-1.41),
   log_beta (+1.29), log_rstar_over_kappa (+1.19), oth_r_mean (+1.12). Plain words: untracked evaders far out
   and a strong push (large beta) favour loss; a large pursuer circle compared with the flock (q/kappa) protects.

## Next (Phase 3)

GP classifier (calibrated probabilities, uncertainty near the boundary), on a stratified subset of 3,000-4,000 runs;
reliability diagram and Brier score; GP regression for herding time with censoring; then the Bayes-risk rule.
