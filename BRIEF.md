# PRML course project brief

Course: EE604T Pattern Recognition and Machine Learning, IIT Dharwad (Dr. Rahul Pandya). Individual project.
Grading: novelty + technical complexity + substantial contribution. Explicitly NOT "implement existing models and compare them".
Toolbox: CLASSICAL ML from the syllabus: Bayes decision theory / Bayes risk, SVMs and kernels, Gaussian processes, PCA, GMMs.
Deliverables: code, Beamer slides, about 5-page IEEE double-column paper (Overleaf).
Evaluation: Tue Nov 17, 2026: live demo + in-depth viva by TAs who do not follow control theory. The demo must explain itself visually.

## Project (title)
"Probabilistic Prediction of Outcome and Herding Time in Single-Pursuer Multi-Evader Herding"

Question: given the starting positions of N evaders and the pursuer, plus the controller parameters, predict the outcome
(success / deadlock / loss) with calibrated probabilities, and the herding time with uncertainty. Then compare the learned
region with the conditions and region-of-attraction estimates in the published papers.

Controller (decided Oct 21; details: docs/phase0/PHASE0_REPORT.md, sections A1-A6):
- Main law: Singh & Chakraborty, arXiv 2505.10048 (public). The pursuer circles the goal at angular speed w with radius
  R exp(k1 (r_0(t) - kappa)); evader 0 = the farthest at t = 0, tracked for the whole run (no switching); kappa = r_0(0).
- Baseline: the same authors' CDC 2024 fixed-circle law (k1 = 0).
- Evaders: velocity k/d^2 away from the pursuer, no interaction between evaders.
- Dimensionless parameters: q = k1 R and beta = k k1^3 / w (arXiv law); gamma = k / (w R^3) (circle law).
- Why ML is needed: the papers' theorems are local (near the equilibrium), and for the untracked evaders the arXiv paper has
  no estimate of the success region (only a brute-force figure). The data shows losses even where the theorems hold.

Labels (fixed in data/labels.py and data/README.md; defaults chosen by the student, Oct 21):
- loss: some evader beyond r_esc = 10 max(q, kappa) (scaled units), or outside the pursuer circle at T_max
- success: every evader within 1 % (radius) and 0.05 rad (angle) of the common equilibrium from some lap on, until T_max
- deadlock: neither ("not settled by T_max"); mostly slow runs with small beta
- herding time: the first lap from which the success test holds; T_max = 300 laps

Data (done Oct 21): 60,000 arXiv-law runs (N = 1..6, three strata of a = 2 q^2 exp(-2 kappa), Latin hypercube, plus boundary
sets) and 10,000 circle-law runs. Own simulator, checked against both papers, the RnD fixed-index vs switching result, and a
high-accuracy solver (same label in 700 of 700 re-runs). See data/README.md.

Features (permutation-invariant, any N; no raw ordered positions): N; q, beta, kappa, a; radial statistics of the evaders
(mean, std, min, as fractions of kappa); angular spread (circular variance, largest angular gap); mean/min pairwise distance;
convex-hull area; angle of each evader relative to the pursuer and to the tracked evader (summarised); theory-informed:
margin to the arXiv condition (a - 1) and to the 6/e^2 design set; distance of each untracked evader from the equilibrium point.

Models: logistic regression (baseline) -> RBF-SVM -> GP classifier (calibrated probabilities, show uncertainty near region
boundaries); GP / kernel regression for herding time (with uncertainty; runs not settled by T_max are censored); GMM + PCA
(do the regimes appear unsupervised?); Bayes-risk decision rule with asymmetric costs (loss worse than deadlock).

Evaluation: stratified CV; calibration (reliability diagram, Brier score); confusion matrices; generalisation to unseen N
(train N <= 4, test N = 5, 6); ML vs theory: does the classifier recover the arXiv condition a = 1 and the circle-law line
gamma = 0.385; how conservative are the papers' Lyapunov-ellipse estimates; how much better is the arXiv law than the circle
baseline. Optional side result (not decided): the student's switching law on the same starts.

Demo: small app (Streamlit or similar): choose N, drag/slide starting positions and parameters -> P(success/deadlock/loss) and
predicted time +- uncertainty -> press play to animate the real simulation.

Planned extensions (same title; the student decides which to keep):
- Planned now: (1) [done 2026-10-23, results/novelty/] active learning with the GP classifier: the GP chooses the next simulation where it is least sure;
  compare the number of runs needed with random sampling. (2) Censored herding time: runs not settled by T_max give only
  "time > T_max"; use a censored (survival-style) regression instead of dropping them.
- If time allows: (3) set kernels that compare two flocks as sets of points, for any N (stronger unseen-N test);
  (4) early warning: predict the outcome from the first few laps, with shrinking uncertainty (for the demo);
  (5) choosing the controller (k1, R) by Bayes risk; (6) interpretability: GP length scales vs the theory.
- Bigger change, only if there is time left: (7) noise or unknown parameters (e.g. a different k for each evader).

Milestones: title into the class sheet Oct 21 (done); data ready Oct 25 (done Oct 21); baselines Nov 1; GP + Bayes risk +
theory comparison Nov 8; demo working Nov 12; paper + slides Nov 14; viva prep Nov 15-16; evaluation Nov 17.

Open (to ask the guide): overlap check, because the PhD student's under-review paper uses the same law (we use only the
public arXiv paper).

## Working rules
- Phases: Phase 0 (audit and plan) and Phase 1 (data) are done. Before each new phase, the assistant reports and waits for
  the student's go-ahead.
- Student derives, the assistant checks and codes: the student must understand and explain every choice in the viva. Keep a
  plain-words viva-prep file (what each model does, why it was chosen, what each plot shows).
- Honest labels everywhere: measured vs assumed vs claimed. No inflated results.
- Keep a short AI-use note for the report (what the AI did, what the student did).
- Ask before any large compute.
- Separation: this repository is separate from the RnD proof repository (shreyas20063/planar-heading-extension). Read the
  simulator code and any saved run data there read-only and copy what is needed here. Never push to, branch from, or run jobs
  in the RnD repository, and never use its Kaggle slots: its certification runs have priority on compute.
- Confidential: do not use or quote the PhD student's handwritten notes or any under-review paper. Cite only published or
  public papers (CDC 2024, arXiv 2505.10048) and the student's own RnD results. If a definition exists only in confidential
  material, flag it and ask.
- Plain language for the student (not a native English speaker): short sentences, explain technical terms in one line.
