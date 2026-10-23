# AI-use note

This note records what the AI assistant did and what the student decided. It will go into the report.

## Phase 0 (2026-10-20)

What the AI did:
- Read the student's RnD repository read-only (simulator code, logs, theory notes). Nothing was written there.
- Audited the saved simulation data (Section 2 of `docs/phase0/PHASE0_REPORT.md`).
- Collected the label rules from the code, with file:line references, and proposed exact definitions.
- Listed the student's proven conditions (gathered with the help of a sub-agent; line references spot-checked).
- Wrote small benchmark and pilot scripts (own code, `docs/phase0/bench/`) and measured cost on the cloud container.
- Drafted the generation plan, this note, `docs/VIVA_PREP.md` (skeleton) and `README.md`.

What the student decided:
- Project topic, toolbox and rules (`BRIEF.md`).
- 2026-10-20: use only the constant-circle law of the published CDC 2024 paper (Singh & Chakraborty),
  not the switching law. The AI then updated the Phase 0 report (sections U1-U6) and ran two small circle-law pilots.
- 2026-10-21: after asking about the arXiv law, the student chose: main law = arXiv 2505.10048 (fixed tracked evader,
  no switching), baseline = CDC 2024 circle law. The AI ran a pilot comparing the arXiv and switching laws, read the
  arXiv paper's theorem list, and wrote sections A1-A6 of the Phase 0 report.
- 2026-10-21: "go ahead" with Phase 1 using the default label numbers of A2 (T_max 300 laps, eps_r 1 %,
  eps_psi 0.05 rad, r_esc 10 max(R, kappa) in scaled units, tracked evader = farthest at t = 0). No switching-law runs.
- Still open: Q-1 (overlap check with the guide), Q-5 (BRIEF wording), and whether to keep these label numbers.

## Phase 1 (2026-10-21)

What the AI did:
- Wrote our own simulator `sim/herding.py` and tests `tests/test_sim.py` (paper numbers, accurate-solver check,
  reproduction of the RnD 32/50 vs 0/50 result).
- Wrote and ran the generator (60,000 arXiv-law runs, 10,000 circle-law runs, about 50 minutes on 4 cores),
  the labelling script and the 1 % accuracy check (100 % label agreement). Wrote `data/README.md`.
- No model has been built yet.

## Phase 2 (2026-10-23)

What the student decided: title "Probabilistic Prediction of Outcome and Herding Time in Single-Pursuer Multi-Evader
Herding"; planned extensions (active learning, censored herding time); "go" for Phase 2.

What the AI did: wrote the features (`models/features.py`) and the baselines (`models/baselines.py`): majority class,
theory rule, logistic regression, RBF-SVM; fixed stratified test set; unseen-N test; circle-law baseline. Wrote
`results/phase2/REPORT.md`. All numbers are from these scripts.

## Phase 3 (2026-10-23)

What the AI did: GP classifier (one-vs-rest, kernel tuned on 1,000 runs, fitted on 4,000), comparison with logistic
regression and SVM + Platt, calibration scores and plots, selective prediction, Bayes-risk rule (cost matrix proposed by
the AI, to be chosen by the student), GP regression for herding time, a 40 x 40 map checked against fresh simulations.
Found and fixed two bugs (log-loss column order, tiny negative probability). Wrote `results/phase3/REPORT.md`.

## Novelty experiment (2026-10-23)

What the student decided: "we need novelty, do something".
What the AI did: proposed and ran risk-aware active learning (`models/active.py`), checked related work with a web
search, found and fixed a GP over-confidence problem (signal-variance cap, `models/gp_stability.py`), re-ran Phase 3
with the fix and corrected its report. Wrote `results/novelty/README.md`.
