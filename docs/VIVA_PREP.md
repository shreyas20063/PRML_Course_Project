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
  Result: best probabilities (log-loss 0.162, Brier 0.092) with only 4,000 training runs; when it is unsure it is
  usually wrong, so answering only the most confident 80 % gives 98.6 % accuracy.
- Why is the GP signal variance capped at 100? Without the cap the GP became over-confident on some training sets
  (3 of 6 random sets were clearly worse). With the cap all 7 sets we tried give almost the same score (0.936-0.938).
- Why only 4,000 runs for the GP? Exact GP needs an n x n matrix inverse: cost grows as n^3. Settings tuned on 1,000.
- Calibration: "80 % sure" should be right 80 % of the time. Reliability diagram = predicted vs observed frequency.
- Bayes-risk rule: choose the decision with the smallest expected cost, sum of P(class) x cost. With costs where losing
  an evader is worst, it cut "loss called success" from 169 to 9 test runs.
- GP regression for herding time: predicts log(1 + time) with a 90 % band; typical error 0.7 laps; the band holds the
  truth 89 % of the time.
- Active learning (our novelty): the GP chooses which simulation to run next. "Entropy" = where it is least sure;
  "risk-aware" = where the best decision is most costly (Bayes risk). Risk-aware reaches random sampling's decision
  cost with 2.3x fewer runs; entropy reaches its accuracy with 2.7x fewer. Each wins on what it aims for.
- Censored herding time: a deadlock run only tells "time > 300 laps". Tobit uses that inequality; the mixture-cure
  model also allows "never settles". On 300 deadlock runs re-simulated to 3,000 laps, the cure model predicted which
  ones settle with AUC 0.98 and the right average rate (53.5 % vs 53.6 %).
- Random Fourier features: cos(w.x + b) with random w turn an RBF kernel into ordinary features, so a kernel model can
  be fitted on 27,000 runs (an exact GP cannot).
- Set kernel: compares two flocks evader by evader (average similarity over all pairs), so it works for any N.
  Trained on N <= 4 it keeps 93.8 % on N = 5, 6; the hand-made features drop to 90.0 %.
- Choosing pursuer settings: the model picks q and beta for a flock; every choice was simulated. Risk-aware choice
  (P(loss) <= 1 %) has 0 % losses like the theory rule and herds 2.3x faster. McNemar test = paired test on yes/no
  outcomes of the same flocks; Wilcoxon = paired test on times.
- GMM + PCA: from the start, the classes do not form clusters (ARI 0.08); from the trajectory, they do (purity 95 %).
  ARI = agreement of two groupings corrected for chance (0 = chance, 1 = same).
- Theory comparison: the paper's Lyapunov ellipse is a guaranteed (sufficient) region; it covers only 0.5-3.7 % of the
  starts that really converge, because near the ring the motion is an almost undamped rotation.

## Evaluation
- Stratified cross-validation
- Calibration: reliability diagram, Brier score
- Confusion matrix
- Train on N <= 4, test on N = 5, 6
- ML vs theory: how conservative are the proven conditions? (0.5-3.7 % coverage; losses where a > 1)
- New simulations check every important learned claim (map, 10x re-runs, chosen settings).

## Each plot: what it shows
- `results/phase2/confusion.png`: rows = true class, columns = predicted; the diagonal is right.
- `results/phase3/reliability.png`: dots on the diagonal = honest probabilities.
- `results/phase3/selective.png`: accuracy goes up when the model may skip the runs it is least sure about.
- `results/phase3/map.png`: one 3-evader flock; left = simulation, middle = GP, right = GP P(loss). Dashed line = arXiv
  condition a = 1. Losses also happen right of the line: the theorem is local.
- `results/phase3/time.png`: predicted vs true herding time with 90 % bands; and the censoring check.
- `results/censored/censored.png`: left = does "will it ever settle" match the 10x re-runs (diagonal = honest);
  right = predicted vs true time for the slow ones.
- `results/control/control.png`: outcome and speed of the settings chosen by each strategy (all simulated).
- `results/theory/circle.png`: green = starts that converge, blue = the region the theorem guarantees.
- `results/regimes/regimes.png`: PCA pictures; three arms in the trajectory view.
- `results/novelty/active.png`: quality vs number of simulations for three ways of choosing runs; green (risk-aware)
  has the lowest decision cost, orange (entropy) the best accuracy.

## Likely TA questions (short answers)
- "Is this just comparing models?" No. The contributions are (1) a validated dataset and calibrated predictors of a
  question the theory cannot answer (outcome and time from any start, any N); (2) a censored survival model that
  separates slow from stuck herds, checked by 10x longer simulation; (3) risk-aware active learning, which chooses
  simulations by the Bayes risk of the decision, with the finding that entropy sampling optimises the wrong target.
- "Why not deep learning?" Course rule: classical ML. Also GPs give calibrated uncertainty with 4,000 runs.
- "Why is the GP limited to 4,000 runs?" Exact GP inverts an n x n matrix: cost n^3. We tune on 1,000, fit on 4,000;
  random Fourier features (used in the time model) are the classical way to scale kernels.
- "What does a length scale mean?" How far you must move along a feature before the prediction changes. Short =
  important. The shortest are log(r*/kappa) and log(q/kappa): ring size and pursuer-circle size relative to the flock.
- "What is calibration?" If the model says 80 %, it should be right 80 % of the time. Checked by reliability diagrams,
  Brier score and ECE.
- "Why balanced accuracy?" Classes have different sizes; it averages per-class recall, so guessing the largest class
  gives only 33 %.
- "How do you know the simulator is right?" It reproduces the papers' numbers (ring radii 0.2541 and 0.1255; r* = 0.4458),
  an independent loss count (32 of 50), and an accurate adaptive solver on 700 runs (same label every time).
- "What is censoring?" A deadlock run was stopped at 300 laps; we only know its time is above 300. Dropping these runs
  biases the model; the cure model uses them correctly.
- "What is a mixture-cure model?" Two parts: the probability that a run ever settles, and how long it takes if it does.
- "Why risk-aware active learning?" Entropy spends simulations wherever the model is unsure, also on cheap mistakes
  (deadlock vs success). Risk-aware spends them where a wrong decision is expensive (calling a loss "success").
- "Does the classifier find the theory's line a = 1?" The map shows the class boundary is NOT at a = 1: losses happen
  where a > 1, because the theorem is local (near the equilibrium only).
- "What does the demo show?" Set up a flock -> probabilities, Bayes decision, predicted time -> press Simulate ->
  the real dynamics; plus a what-if map over the pursuer settings.
