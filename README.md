# Probabilistic Prediction of Outcome and Herding Time in Single-Pursuer Multi-Evader Herding

EE604T (PRML) course project, IIT Dharwad.
One pursuer herds N evaders to a goal. Main law: Singh & Chakraborty, arXiv 2505.10048 (the pursuer radius follows the
evader that was farthest at t = 0). Baseline: their CDC 2024 fixed-circle law. We simulate many starting configurations and parameters,
label each run (success / deadlock / loss, herding time), and learn the success region with classical ML
(logistic regression, SVM, Gaussian processes, GMM/PCA, Bayes risk).
Then we compare the learned region with the papers' conditions and region-of-attraction estimates.
Status (2026-10-24): all parts done; see the paper `paper/main.pdf` and slides `slides/main.pdf`.

## Main results (all measured; details in each `results/*/README.md`)
| part | result |
|---|---|
| data | 60,000 arXiv-law + 10,000 circle-law runs; own simulator, 700/700 labels confirmed by an accurate solver |
| GP classifier (Phase 3) | balanced accuracy 93.7 %, best calibrated (log-loss 0.162); Bayes rule: loss called success 169 -> 9 |
| theory gap | arXiv condition a > 1 as a rule: 45.7 %; circle-law Lyapunov ellipse covers 0.5-3.7 % of the true region |
| set kernel | unseen flock sizes (train N <= 4, test N = 5, 6): 93.8 % vs 90.0 % for hand features |
| censored time (cure model) | slow vs stuck herds predicted beyond the horizon: AUC 0.98 on 10x longer re-simulation |
| risk-aware active learning | same decision cost as random sampling with 2.3x fewer simulations |
| choosing settings | risk-aware GP choice: 0 % loss like the theory rule, 2.3x faster herding (Wilcoxon p = 9e-24) |
| regimes (GMM + PCA) | invisible in start features (ARI 0.08), natural in trajectories (cluster purity 0.95) |

## Quick start
    python -m pytest -q tests                  # simulator checks
    python app/export_models.py && streamlit run app/app.py   # live demo (see app/README.md)
    cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main

## Folder map
- `BRIEF.md` - project scope, rules, milestones
- `docs/phase0/PHASE0_REPORT.md` - data audit, labels, theory list, generation plan, open questions
- `docs/phase0/bench/` - cost benchmark and pilot scripts with their output
- `sim/` - our simulator; `tests/` - its checks
- `data/` - generator, labels, accuracy check, raw runs (`data/README.md`)
- `models/` - features and models; `results/` - metrics, figures and a README per experiment
- `app/` - Streamlit demo; `paper/` - IEEE paper; `slides/` - Beamer slides
- `docs/VIVA_PREP.md` - plain-words viva notes (grows with the project)
- `docs/AI_USE_NOTE.md` - what the AI did, what the student decided
