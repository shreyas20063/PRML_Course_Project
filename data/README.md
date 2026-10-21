# Dataset (Phase 1)

Generated 2026-10-21 on the cloud container (4 cores) with our own simulator `sim/herding.py`.
Plan: `docs/phase0/PHASE0_REPORT.md`, sections A2 and A4. All numbers below are [measured].

## How to rebuild

    python -m pytest -q tests                 # simulator checks (4 tests)
    python data/generate.py arxiv 0 60        # 60 chunks x 1000 runs  (~45 min, mostly root finding)
    python data/generate.py circle 0 10       # 10 chunks x 1000 runs
    python data/labels.py 0.01                # -> data/dataset_arxiv.csv, data/dataset_circle.csv (not in git)
    python data/validate.py                   # 1 % re-check with scipy DOP853 -> data/validation.json

`data/raw/*.npz` (in git) hold every run: starts, parameters, equilibria, final state, settling laps for
eps_r = 0.5 / 1 / 2 / 5 %, a trace every 5 laps, escape lap, smallest pursuer-evader distance.
So the labels can be recomputed with another eps_r without re-simulating.

## Units (dimensionless)

- arXiv law: lengths in units of 1/k1, time in units of 1/w, one lap = 2 pi.
  Parameters: q = k1 R, beta = k k1^3 / w, kappa = k1 r_0(0) (r_0 = tracked evader, the farthest at t = 0).
  a = 2 q^2 exp(-2 kappa). Strata: 1: a > 1 (arXiv Theorem 4), 2: 6/e^2 < a <= 1, 3: a <= 6/e^2.
- circle law (k1 = 0): lengths in units of R; parameter gamma = k / (w R^3). Single-evader runs
  (the flock label is "every evader succeeds", which is exact for this law).

## Label choices used (student defaults, A2)

T_max = 300 laps; eps_r = 1 %; eps_psi = 0.05 rad; r_esc = 10 max(q, kappa) (arXiv), 10 (circle);
loss = beyond r_esc at a lap end, or outside the pursuer circle at T_max.

## What is in it

| set | runs | success | deadlock | loss |
|---|---|---|---|---|
| arXiv law | 60,000 | 26,454 | 16,013 | 17,533 |
| circle law | 10,000 | 4,114 | 3,204 | 2,682 |

arXiv law, Latin-hypercube part (3,000 runs per cell), percent success / deadlock / loss:

| N | stratum 1 (a > 1) | stratum 2 | stratum 3 |
|---|---|---|---|
| 1 | 65.2 / 34.8 / 0.0 | 76.4 / 23.6 / 0.0 | 86.8 / 13.2 / 0.0 |
| 2 | 60.8 / 33.7 / 5.5 | 39.9 / 23.2 / 36.9 | 20.4 / 36.4 / 43.2 |
| 3 | 56.2 / 34.6 / 9.2 | 33.8 / 24.0 / 42.3 | 18.0 / 29.1 / 52.9 |
| 4 | 55.7 / 35.0 / 9.3 | 32.9 / 23.1 / 44.0 | 17.2 / 27.3 / 55.4 |
| 5 | 55.2 / 35.0 / 9.8 | 32.8 / 23.1 / 44.1 | 16.5 / 24.8 / 58.7 |
| 6 | 54.7 / 35.0 / 10.3 | 31.4 / 23.7 / 44.9 | 17.2 / 23.3 / 59.5 |

Boundary sets (2,000 runs each): near a = 1: 846 / 480 / 674; near a = 6/e^2: 768 / 445 / 787;
large beta (10 to 50, stratum 1): 1,708 / 0 / 292.

## First observations (data only, no model yet)

1. **One evader is never lost** (0 of 18,000 N = 1 runs, all strata). Losses are always untracked evaders.
2. **The paper's condition (stratum 1) does not prevent loss when N >= 2**: 5.5 % (N = 2) to 10.3 % (N = 6).
   This agrees with the paper's theorems being local (near the equilibrium).
3. **Deadlock is mostly "too slow"**: deadlock is 76.7 % of runs with beta < 0.01, but 0.5 % with beta >= 10.
   For small beta the push k/d^2 is weak and 300 laps are not enough. The "stuck" flag (r_max changed < 2 %
   over the last 60 laps) also fires for these slow runs, so it does not separate stuck from slow well.
4. Herding time (success runs): median 14 laps, 90 % within 166 laps, 99 % within 283 laps (close to T_max: censoring).
5. Circle law: gamma >= 0.385 gives loss in 1,000 of 1,000 runs (CDC Lemma 2 says no ring exists there).
   Every circle-law deadlock has gamma < 0.0093 (slow approach to a tiny ring).
6. 1,543 runs have more than one equilibrium radius (outside the paper's uniqueness condition); labels then use
   the root the evaders settle at.

## Accuracy check

`data/validate.py` re-ran a stratified 1 % sample (10 runs from every chunk) with scipy DOP853
(rtol 1e-10, atol 1e-12) and applied the same label rules:

| set | runs checked | same label | herding time difference |
|---|---|---|---|
| arXiv law | 600 | 600 (100 %) | 0 laps (all) |
| circle law | 100 | 100 (100 %) | 0 laps (all) |

Reference solver time: 604 s + 95 s (single core). Full result: `data/validation.json`.
Simulator tests (`tests/test_sim.py`): CDC 2024 ring radii 0.2541 and 0.1255; arXiv running example
r* = 0.4458, psi* = 1.1724; RK4 vs DOP853 state error < 1e-6; and the RnD result for the fixed-index law
vs switching (32 of 50 vs 0 of 50 runs lose an evader) is reproduced exactly.
