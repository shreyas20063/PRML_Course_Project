# A readable rule from the data: a conjecture to compare with the theory's a > 1

Code: `models/rule.py`. Numbers [measured]: `metrics.json`.
Method: for every 1, 2 and 3 of the 24 start features, a logistic regression on 20,000 training runs; the best
combination of each size chosen on 10,000 OTHER training runs (validation); reported once on the fixed test set.
Units: lengths divided by 1/k1 (q = k1 R, kappa = k1 r_0(0)), beta = k k1^3 / w, a = 2 q^2 exp(-2 kappa);
oth_r_max = start radius of the outermost untracked evader divided by kappa.

| question (test balanced accuracy) | theory (a vs 1) | 1 feature | 2 features | 3 features | GP classifier |
|---|---|---|---|---|---|
| will an evader be lost? | 0.672 | 0.743 | 0.817 | **0.862** | 0.938 |
| will herding succeed? | 0.614 | 0.652 | 0.692 | 0.702 | 0.962 |

The rules (as found):
- loss, 1 feature:  log(r*/kappa) > -1.10, i.e. the final ring r* is larger than about 1/3 of the flock radius.
- loss, 2 features: 0.524 log(beta) - 2.402 log(q/kappa) > -2.563, i.e. **q/kappa < 2.9 beta^0.22**:
  an evader escapes when the pursuer circle is too small compared with the flock; a stronger push needs a bigger circle.
- loss, 3 features: 0.641 log(beta) - 1.480 **log(a)** + 4.435 oth_r_max > 3.350:
  the theory's own quantity a protects (as the theorem suggests), but it must be corrected for the push strength beta and
  for an untracked evader starting near the edge of the flock.
- success, 2 features: -1.016 kappa - 2.005 oth_r_max > -2.707: small flock, no untracked evader near its edge.

Plain words: a two- or three-number rule already beats the theorem's condition as a predictor of loss (82-86 % vs 67 %),
and it says which corrections the condition needs. Success needs the full model (96 %): it depends on slow-vs-stuck
behaviour that short rules miss. These rules are conjectures from simulation, not proofs.
