# Theory vs data: how conservative is the proven region? (circle law, CDC 2024)

Code: `models/theory_circle.py`. Numbers [measured]: `circle.json`. Figure: `circle.png`.

Set-up (one evader, dimensionless R = w = 1, k = gamma; rotating frame, pursuer fixed at (1, 0)):
- **Paper's estimate** (CDC 2024, Sec. III-C, Problem 6): the largest ellipse {(w - w*)' A (w - w*) <= c} on which
  the quadratic Lyapunov function decreases. We solve the same problem: start from the Lyapunov-equation solution
  A J + J' A = -I (as the paper does), optimise the shape with Nelder-Mead, and check dV/dt on dense grids.
- **Any pursuer start** (CDC Sec. III-D): intersection of the ellipse rotated about the origin = a disk around the
  origin (exists only if the origin is inside the ellipse).
- **Truth**: a 90 x 90 grid of starts in the unit disk (CDC Assumption 1), each simulated 300 laps; "converges" = not lost.

| gamma | true converging fraction of the disk | ellipse covers (of the truly converging starts) | any-start disk radius | false positives |
|---|---|---|---|---|
| 0.01 | 97.4 % | 0.5 % | 0.061 | 0 |
| 0.03 | 90.4 % | 0.6 % | 0.042 | 0 |
| 0.1 | 70.8 % | 1.0 % | none (origin outside the ellipse) | 0 |
| 0.2 | 47.0 % | 1.6 % | none | 0 |
| 0.3 | 24.9 % | 3.7 % | none | 0 |

Plain words:
- The certificate is **always correct** (no start inside the ellipse fails), but it covers only 0.5 % to 3.7 % of the
  starts that really converge: the proven region is 27 to 190 times smaller than the true one.
- Why: near the ring the motion is an almost undamped rotation (Jacobian eigenvalues about -0.005 +- 1i at
  gamma = 0.01), so a quadratic Lyapunov function stops decreasing very close to the ring.
- The classifiers (Phase 2) learn the true region almost perfectly for this law (99 % balanced accuracy).

Honest note: our certified ellipse is smaller than the region drawn in the paper's Fig. 3/4 for R = 2, k = 1, w = 1
(gamma = 0.125). We checked our result: along the limiting direction dV/dt really turns positive about 0.08 R from the
ring (`models/theory_circle.py`, check printed during development). The paper used fmincon with its own constraint
sampling, which we cannot see; we report our dense-grid version.
