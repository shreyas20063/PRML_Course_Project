# Viva preparation (plain words)

To be filled as the project grows. One short answer per question. The student writes the final
wording; the assistant only checks.

## The problem
- What is herding here? One pursuer circles the goal; N evaders run away from it (speed k/d^2).
  The pursuer radius follows the farthest evader: R_p = R exp(k1 (r_max - kappa)).
- What are the three outcomes? success / deadlock / loss (exact rules: `docs/phase0/PHASE0_REPORT.md`, Section 3).
- Why use ML if there are proofs? The proofs cover only part of the parameter space (zone A) or single points.
  ML gives an estimate of the success region everywhere, with uncertainty.

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
