# Herding time with censoring: slow vs stuck herds

Code: `models/censored_time.py`. Numbers [measured]: `metrics.json`. Figure: `censored.png`. Re-simulations: `resim.npz`.

## Problem
Deadlock runs were stopped at T_max = 300 laps; we only know "time > 300" (right-censored). Dropping them (Phase 3)
biases the time model; and "deadlock" may hide two kinds of runs: slow ones and ones that never settle.

## Models (all share one kernel function class: random Fourier features (1,000) of the Phase 3 GP kernel; MAP fit)
- **successes only**: log-normal time on settled runs (deadlocks dropped).
- **Tobit**: log-normal time; deadlocks contribute P(time > 300).
- **mixture cure** (ours, kernel version of Farewell 1982): P(the run ever settles) = sigmoid(g(x)), a kernel logistic
  model; if it settles, log-normal time. Deadlocks contribute (1 - pi) + pi * P(time > 300).
Prior scale chosen on a validation split (for Tobit and cure the best was the largest value tried, x30).

## Test set (settled + deadlock test runs of the arXiv law)

| | successes only | Tobit | **cure** |
|---|---|---|---|
| test log-likelihood per run (higher better) | -24.05 | -0.27 | **+0.37** |
| time-dependent concordance (Antolini) | 0.943 | 0.979 | **0.992** |
| settled runs: median error (laps) | 0.72 | 1.95 | **0.69** |
| settled runs: 90 % band coverage | 0.917 | 0.947 | 0.916 |
| deadlock runs: mean P(time > 300) (should be high) | 0.83 | 0.95 | **0.98** |

## The real check: 300 test deadlock runs re-simulated 10x longer (3,000 laps)
Result: 158 settle, 137 still do not settle, 5 lose an evader. None of this is in the training data (which stops at 300).

| | Brier, "settles by 3,000" (lower better) | AUC | mean predicted P (observed 0.536) | settled: median ratio error | 90 % band coverage |
|---|---|---|---|---|---|
| successes only | 0.345 | 0.739 | 0.889 | x1.19 | 0.58 |
| Tobit | 0.216 | 0.853 | 0.450 | x1.77 | 0.49 |
| **cure** | **0.049** | **0.984** | **0.535** | **x1.09** | **0.83** |

Plain words:
- "Deadlock" is two things: about half of the stalled runs are only slow, the other half never settle.
- The cure model tells them apart from the START alone (AUC 0.98), and its average prediction (53.5 %) matches the
  observed rate (53.6 %). It also predicts how long the slow ones take (typical error 9 %), beyond the training horizon.
- Dropping the deadlocks (successes only) makes the model badly over-confident that every run finishes.
- Tobit (one noise level for everything) must inflate its noise to explain the stuck runs, so its times get worse.

Limits: optimiser stopped at 3,000 iterations for some fits; the prior scale is at the edge of the grid (x30);
"never settles" means "not within 3,000 laps".
