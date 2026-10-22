# Probabilistic Prediction of Outcome and Herding Time in Single-Pursuer Multi-Evader Herding

EE604T (PRML) course project, IIT Dharwad.
One pursuer herds N evaders to a goal. Main law: Singh & Chakraborty, arXiv 2505.10048 (the pursuer radius follows the
evader that was farthest at t = 0). Baseline: their CDC 2024 fixed-circle law. We simulate many starting configurations and parameters,
label each run (success / deadlock / loss, herding time), and learn the success region with classical ML
(logistic regression, SVM, Gaussian processes, GMM/PCA, Bayes risk).
Then we compare the learned region with the papers' conditions and region-of-attraction estimates.
Status: Phase 1 (data) done: 70,000 simulated runs with labels, checked. Next: Phase 2 (baseline models).

## Folder map
- `BRIEF.md` - project scope, rules, milestones
- `docs/phase0/PHASE0_REPORT.md` - data audit, labels, theory list, generation plan, open questions
- `docs/phase0/bench/` - cost benchmark and pilot scripts with their output
- `sim/` - our simulator; `tests/` - its checks
- `data/` - generator, labels, accuracy check, raw runs (`data/README.md`)
- `docs/VIVA_PREP.md` - plain-words viva notes (grows with the project)
- `docs/AI_USE_NOTE.md` - what the AI did, what the student decided
