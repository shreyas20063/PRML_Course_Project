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
- How was the data made? Our own simulator (RK4 with a step that shrinks when the pursuer passes close to an
  evader), checked against the papers' numbers and against a high-accuracy solver (same label in 700 of 700 re-runs).
  60,000 arXiv-law runs (N = 1..6, three parameter strata, Latin hypercube) and 10,000 circle-law runs, 300 laps each.
  Details: `data/README.md`.
- Why is "deadlock" mostly "slow"? When beta = k k1^3 / w is small, the push on the evaders is weak, and 300 laps are
  not enough to settle. We call it "not settled by T_max", not "stuck".
- Why Latin hypercube sampling? It spreads samples evenly along every parameter axis with few runs.
- Why these features? They describe only the start: parameters (q, beta, kappa, a), the shape of the flock (radii,
  angular spread, pairwise distances, hull area), and where the evaders are relative to the pursuer, the tracked evader
  and the equilibrium point. Lengths are divided by kappa so flocks of different size are comparable.
- Why permutation-invariant? The order of evaders has no meaning; means, minima and spreads do not change if we
  renumber the evaders, and they exist for any N (needed for the unseen-N test).

## Models (to fill, one paragraph each: what it does, why chosen, what can go wrong)
- Logistic regression (baseline): a linear boundary between classes; it gives probabilities. Chosen as the simplest
  model. Result: 82.5 % balanced accuracy; it misses the curved boundary.
- RBF-SVM: draws a curved boundary using a similarity (kernel) between runs; gamma sets how local the similarity is,
  C how much training errors cost. Result: 93.9 % balanced accuracy, 92.8 % on unseen N = 5, 6. Weak point: no
  calibrated probabilities.
- Why balanced accuracy? It averages the recall of the three classes, so a model cannot score well by always
  guessing the largest class (that would give 33 %).
- Why is the theory rule so weak (46 %)? The theorem is local (near the equilibrium); it does not say what happens
  from far starts. That gap is the point of the project.
- Gaussian process classifier: puts a smooth random function behind the class probabilities; the kernel says that
  similar starts give similar outcomes, and one length scale per feature says how fast things change along it.
  Result: best probabilities (log-loss 0.155, Brier 0.090) with only 4,000 training runs; when it is unsure it is
  usually wrong, so answering only the most confident 80 % gives 98.9 % accuracy.
- Why only 4,000 runs for the GP? Exact GP needs an n x n matrix inverse: cost grows as n^3. Settings tuned on 1,000.
- Calibration: "80 % sure" should be right 80 % of the time. Reliability diagram = predicted vs observed frequency.
- Bayes-risk rule: choose the decision with the smallest expected cost, sum of P(class) x cost. With costs where losing
  an evader is worst, it cut "loss called success" from 174 to 12 test runs.
- GP regression for herding time: predicts log(1 + time) with a 90 % band; typical error 0.7 laps; the band holds the
  truth 89 % of the time.
- GMM + PCA (do regimes appear without labels?)
- Bayes-risk decision rule (loss costs more than deadlock)

## Evaluation (to fill)
- Stratified cross-validation
- Calibration: reliability diagram, Brier score
- Confusion matrix
- Train on N <= 4, test on N = 5, 6
- ML vs theory: how conservative are the proven conditions?

## Each plot: what it shows
- `results/phase2/confusion.png`: rows = true class, columns = predicted; the diagonal is right.
- `results/phase3/reliability.png`: dots on the diagonal = honest probabilities.
- `results/phase3/selective.png`: accuracy goes up when the model may skip the runs it is least sure about.
- `results/phase3/map.png`: one 3-evader flock; left = simulation, middle = GP, right = GP P(loss). Dashed line = arXiv
  condition a = 1. Losses also happen right of the line: the theorem is local.
- `results/phase3/time.png`: predicted vs true herding time with 90 % bands; and the censoring check.

## Likely TA questions (to fill)
