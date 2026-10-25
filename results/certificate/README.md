# A statistical certificate vs the Lyapunov certificate of the theory

Code: `models/certify.py`. Numbers [measured]: `metrics.json`. Figure: `certificate.png`.

## Method (classical statistics, no new model)
Safe set S = { starts x : GP probability of loss <= tau }. On calibration runs never used to fit the GP, the exact
one-sided Clopper-Pearson bound gives, with 95 % confidence, an upper bound on the loss rate inside S. We take the
largest tau whose bound is <= 1 % (increasing tau, stop at the first failure: fixed-sequence testing keeps the 95 %).
The guarantee is about the loss rate of the whole set under the sampling design (a "distribution-level" guarantee);
the Lyapunov certificate is pointwise (every single start inside it converges, if the proof is right).

## arXiv law (12,000 calibration runs; fixed 12,000-run test set)
Calibration: tau = 0.217; 7,794 calibration runs in S, 63 lost; 95 % upper bound 0.996 % (<= 1 %).

| safe set on the test set | loss rate inside the set | share of all true successes covered |
|---|---|---|
| theory: a > 1 (arXiv Theorem 4) | 8.8 % | 47.7 % |
| **data certificate** | **0.66 %** | **90.9 %** |

By N (test): loss rate in S = 0 / 1.38 / 0.78 / 0.97 / 0.89 / 0.36 % for N = 1..6; coverage 84-100 %.
The guarantee is for the mix of N; for N = 2 the rate (1.38 %) is above 1 %.

## Circle law vs the paper's Lyapunov ellipse (same fresh 90 x 90 grids of starts as `results/theory/`)
Calibration: tau = 0.309; 2,887 calibration runs in S, 17 lost; bound 0.88 %.

| gamma | Lyapunov ellipse covers (of truly converging starts) | data certificate covers | loss rate inside the data set |
|---|---|---|---|
| 0.01 | 0.5 % | **97.8 %** | 0 (0 of 5,812) |
| 0.03 | 0.6 % | **98.0 %** | 0.06 % (3 of 5,409) |
| 0.1 | 0.9 % | **97.1 %** | 0.69 % (29 of 4,223) |
| 0.2 | 1.6 % | **92.5 %** | 0.90 % (24 of 2,673) |
| 0.3 | 3.7 % | **76.7 %** | **1.69 %** (20 of 1,186) |
Pooled over the five grids: 76 lost of 19,303 certified starts = 0.39 %.

## Plain words
- The theorem-based safe set (a > 1) is neither safe (8.8 % losses) nor large (half of the successes).
- The data certificate keeps its promise on fresh data (0.66 % < 1 %) and covers 91 % of the successes.
- For the circle law, the data certificate covers 77-98 % of the starts that really converge; the Lyapunov proof covers
  0.5-3.7 %: 20 to 190 times less.
- The price: the data certificate is a statement about rates, not about every single start; at the hardest setting
  (gamma = 0.3) and for N = 2 the observed rate is above 1 %. The two certificates are complementary: the proof is exact
  but tiny; the data certificate is large but statistical.
