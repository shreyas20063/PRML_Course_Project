# Viva preparation (plain words)

To be filled as the project grows. One short answer per question. The student writes the final
wording; the assistant only checks.

## The problem
- What is herding here? One pursuer circles the goal at angular speed w; N evaders run away from it (speed k/d^2).
  Main law (arXiv 2505.10048): the pursuer radius is R exp(k1 (r_0(t) - kappa)), where r_0 is the distance of the
  evader that was farthest at t = 0 (always the same evader), kappa = r_0(0).
  Baseline (CDC 2024): the same with k1 = 0, i.e. a fixed circle of radius R.
- Why only two parameters? Measure lengths in units of 1/k1 and time in units of 1/w; then only k1 R and
  beta = k k1^3 / w remain (Phase 0 report, A1). For the circle law only gamma = k/(w R^3) remains (U1).
- What are the three outcomes? success / deadlock / loss (exact rules: `docs/phase0/PHASE0_REPORT.md`, section A2).
- Why use ML if there are theorems? The paper's theorems are local (near the equilibrium), and for the evaders that are
  not tracked the paper has no estimate of the success region (only a brute-force figure). ML estimates that region
  for any N, with uncertainty, and the herding time. (Final wording: after the senior meeting.)

## Data
- How was the data made? (to fill after Phase 1)
- Why Latin hypercube sampling? It spreads samples evenly along every parameter axis with few runs.
- Why these features? (to fill) Why permutation-invariant? The order of evaders has no meaning.

## Models (to fill, one paragraph each: what it does, why chosen, what can go wrong)
- Logistic regression (baseline)
- RBF-SVM
- Gaussian process classifier (calibrated probabilities, uncertainty near the boundary)
- GP / kernel regression for herding time
- GMM + PCA (do regimes appear without labels?)
- Bayes-risk decision rule (loss costs more than deadlock)

## Evaluation (to fill)
- Stratified cross-validation
- Calibration: reliability diagram, Brier score
- Confusion matrix
- Train on N <= 4, test on N = 5, 6
- ML vs theory: how conservative are the proven conditions?

## Each plot: what it shows (to fill)

## Likely TA questions (to fill)
