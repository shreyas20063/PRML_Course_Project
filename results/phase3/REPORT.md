# Phase 3: GP classifier, calibration, uncertainty, Bayes risk, herding time

Date: 2026-10-23. All numbers are [measured] on the same held-out test set as Phase 2 (12,000 arXiv-law runs;
the shared split `models/features.py: test_split`; re-running Phase 2 with it gives identical results).
Code: `models/gp_classify.py`, `models/gp_time.py`, `models/gp_map.py`. Numbers: `gp_metrics.json`, `time_metrics.json`.

## Setup (plain words)

- **GP classifier**: one GP per class ("this class or not"), Laplace approximation, kernel = constant x RBF with
  one length scale per feature. Exact GP cost grows as n^3, so the kernel settings were tuned on 1,000 training runs
  (272 s) and the GP was then fitted on 4,000 runs with those settings fixed (54 s).
- **Compared with**: logistic regression (all 48,000 training runs), and the RBF-SVM made probabilistic with Platt
  scaling (a logistic curve fitted on the SVM's scores), trained on 4,000 runs (same as the GP) and on 20,000 runs.
- **Scores for probabilities** (lower is better):
  - log-loss: the average of -log(probability given to the true class);
  - Brier score: the average squared distance between the probability vector and the true class;
  - ECE (expected calibration error): how far "80 % sure" is from "right 80 % of the time", on average.

## 1. Probabilities: the GP is the best calibrated

| model | training runs | accuracy | balanced acc. | log-loss | Brier | ECE |
|---|---|---|---|---|---|---|
| logistic regression | 48,000 | 0.830 | 0.825 | 0.482 | 0.258 | 0.039 |
| SVM + Platt | 4,000 | 0.920 | 0.918 | 0.254 | 0.130 | 0.034 |
| SVM + Platt | 20,000 | 0.940 | 0.938 | 0.200 | 0.099 | 0.021 |
| **GP classifier** | **4,000** | 0.939 | 0.936 | **0.155** | **0.090** | **0.020** |

With the same 4,000 runs, the GP beats the SVM on every score. It even beats the SVM trained on 5 times more data on
log-loss and Brier score, at the same accuracy. Figure: `reliability.png` (points near the diagonal = honest probabilities).

## 2. Uncertainty is useful (selective prediction)

If the model answers only its most confident runs and says "not sure" for the rest:

| fraction answered | 100 % | 90 % | 80 % | 70 % | 50 % |
|---|---|---|---|---|---|
| GP accuracy | 0.939 | 0.972 | 0.989 | 0.995 | 1.000 |
| SVM + Platt (20k) | 0.940 | 0.963 | 0.972 | 0.974 | 0.980 |

The GP's uncertainty points at its mistakes: mean entropy 0.61 on wrong answers vs 0.17 on right answers.
Figure: `selective.png`.

## 3. Bayes-risk decisions (asymmetric costs)

Cost matrix [proposal; the student should choose the numbers]: calling a true **loss** a "success" costs 10;
calling a true loss "deadlock" or a true deadlock "success" costs 3; other mistakes cost 1; correct = 0.
The Bayes rule picks the decision with the lowest expected cost, sum over classes of P(class) x cost.

| model | mean cost, most-probable class | mean cost, Bayes rule | losses called "success": most probable -> Bayes |
|---|---|---|---|
| GP | 0.216 | **0.120** | 174 -> **12** |
| SVM + Platt (20k) | 0.191 | 0.141 | 144 -> 17 |
| logistic regression | 0.584 | 0.319 | 461 -> 6 |

With good probabilities, the Bayes rule almost removes the worst mistake, and the GP gives the lowest total cost.

## 4. Herding time with uncertainty (successful runs)

Target log(1 + T_h). GP regression, kernel tuned on 1,000 runs, fitted on 4,000 (out of 21,164 successful training runs).

| model | typical error (median, laps) | typical ratio error | RMSE in log | 90 % interval coverage |
|---|---|---|---|---|
| always the median | 13.0 | x 3.75 | 1.58 | - |
| linear regression (all 21,164) | 2.75 | x 1.24 | 0.43 | - |
| **GP** | **0.70** | **x 1.05** | **0.14** | **0.889** |

The 90 % bands contain the truth 88.9 % of the time (close to 90 %: honest uncertainty). Typical band: 10.6 to 15.6 laps.
Censoring check (not yet a censored model): for the 3,203 test deadlock runs (truth: T_h > 300), the GP says
P(T_h > 300) > 0.5 for 85 %; for successful runs the median of that probability is 0. Figure: `time.png`.
The other 15 % of deadlock runs get P(T_h > 300) near 0: the GP expects them to be fast. They may be truly
stuck runs (not just slow), which a success-only time model cannot know about; to check in the censoring extension.

## 5. One picture: learned map vs truth

`map.png`: a fixed 3-evader flock (kappa = 1), 40 x 40 grid over q = k1 R and beta = k k1^3 / w, each point simulated
fresh (not in the training data). The GP's most probable class agrees with the simulation on **97.3 %** of the 1,600 points.
- Loss happens also to the RIGHT of the dashed line a = 1, where arXiv Theorem 4 holds (q about 2 to 2.7, beta 0.3 to 30):
  the theorem's local result does not protect the untracked evaders. The GP finds this region.
- Deadlock (too slow) is the small-beta corner; success lies between.
- The GP's high-uncertainty line (entropy 0.6) follows the true class boundaries.
- Miss: a small deadlock patch at q about 1.1, beta 0.15 to 0.9 is predicted as loss.

## What the GP length scales say (ARD)

A short length scale = the class changes quickly with that feature = important. For all three classes the shortest
are log(r*/kappa) (0.9 to 1.6) and log(q/kappa) (2.0 to 2.4), then kappa and log(beta). Plain words: what matters most is
how big the equilibrium ring and the pursuer circle are compared with the flock, and how strong the push is.
14 to 16 of the 24 features were switched off (length scale at the upper bound).
Caution: log(a) looks unimportant here only because it is a combination of log q and kappa, which are already features;
it does not mean the theorem's quantity is irrelevant.

## Honest limits

- The GP was trained on 4,000 runs (exact GP cost); kernel settings come from 1,000 runs. A larger GP may do better.
- The cost matrix is our proposal; results depend on it.
- Herding time is modelled on successful runs only; the proper censored model is a planned extension.
- One map is one flock; other flocks give other maps.
- All results are from simulation, with our label numbers (T_max = 300 laps, eps_r = 1 %).

## Bugs found and fixed during this phase

- sklearn's `log_loss` sorts class names alphabetically; our probability columns were in another order, so the first
  log-loss numbers were wrong (e.g. 8.2). Fixed by computing log-loss directly; cross-checked with sklearn using the
  sorted order (same numbers). Accuracy, Brier, ECE and the Bayes decisions were not affected.
- One model gave a probability of -1e-13 (rounding); probabilities are now clipped to [0, 1] and renormalised.
