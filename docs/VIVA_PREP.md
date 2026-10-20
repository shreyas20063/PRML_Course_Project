# Viva preparation (plain words)

To be filled as the project grows. One short answer per question. The student writes the final
wording; the assistant only checks.

## The problem
- What is herding here? One pursuer moves on a fixed circle (radius R, angular speed w) around the goal;
  N evaders run away from it (speed k/d^2). This is the law of the CDC 2024 paper (Singh & Chakraborty).
- Why does only gamma = k/(w R^3) matter? Measure lengths in units of R and time in units of 1/w;
  then k, w, R appear only through gamma (derivation in the Phase 0 report, U1).
- What are the three outcomes? success / deadlock / loss (exact rules: `docs/phase0/PHASE0_REPORT.md`, section U2).
- Why use ML if there are results? The paper gives an inner (conservative) estimate of the success region.
  ML estimates the whole region and the herding time, with uncertainty. (Final wording: after the senior meeting.)

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
