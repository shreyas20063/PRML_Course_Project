# PRML course project brief

Course: EE604T Pattern Recognition and Machine Learning, IIT Dharwad (Dr. Rahul Pandya). Individual project.
Grading: novelty + technical complexity + substantial contribution. Explicitly NOT "implement existing models and compare them".
Toolbox: CLASSICAL ML from the syllabus: Bayes decision theory / Bayes risk, SVMs and kernels, Gaussian processes, PCA, GMMs.
Deliverables: code, Beamer slides, about 5-page IEEE double-column paper (Overleaf).
Evaluation: Tue Nov 17, 2026: live demo + in-depth viva by TAs who do not follow control theory. The demo must explain itself visually.

## Project (title)
"Probabilistic Prediction of Outcome and Herding Time in Single-Pursuer Multi-Evader Herding"

Question: given the starting configuration of N evaders and the pursuer, plus controller parameters, predict whether herding
succeeds and how long it takes. Then compare the learned success region with conditions proven in the student's own RnD work.
This is a data-driven estimate of the region of attraction; the RnD work is the proofs. Different contribution, no double submission.

Labels (exact definitions/thresholds must come from the simulator and be stated in the paper):
- success: all evaders herded, r_max settles at r* within T_max
- deadlock: contained but not converged within T_max
- loss: at least one evader escapes
- herding time for successful runs

Features (permutation-invariant, any N; no raw ordered positions): N; radial statistics of the evaders from the goal
(r_max(0), mean, std, min); angular spread (circular variance, largest angular gap); mean/min pairwise distance; convex-hull
area; pursuer initial radius and angle relative to the farthest evader; controller parameters (R, k1, kappa, others);
physics-informed: sqrt(3) margin m = min_i R_p(0) / (sqrt(3) r_i(0)), R_p(0) = R exp(k1 (r_max(0) - kappa)).

Models: logistic regression (baseline) -> RBF-SVM -> GP classifier (calibrated probabilities, show uncertainty near region
boundaries); GP / kernel regression for herding time (with uncertainty); GMM + PCA (do the regimes appear unsupervised?);
Bayes-risk decision rule with asymmetric costs (loss worse than deadlock).

Evaluation: stratified CV; calibration (reliability diagram, Brier score); confusion matrices; generalisation to unseen N
(train N <= 4, test N = 5, 6); ML vs theory: how conservative are the proven conditions; does the classifier recover the
sqrt(3) boundary?

Demo: small app (Streamlit or similar): choose N, drag/slide starting positions and parameters -> P(success/deadlock/loss) and
predicted time +- uncertainty -> press play to animate the real simulation.

Milestones: title into the class sheet Oct 21; data ready Oct 25; baselines Nov 1; GP + Bayes risk + theory comparison Nov 8;
demo working Nov 12; paper + slides Nov 14; viva prep Nov 15-16; evaluation Nov 17.

## Working rules
- PHASE 0 FIRST: audit existing simulation data, confirm label definitions from the simulator code, propose a data-generation
  plan. Report in plain language. Do not build beyond Phase 0 until the student confirms (decision after the senior's advice,
  Wed Oct 21).
- Student derives, the assistant checks and codes: the student must understand and explain every choice in the viva. Keep a
  plain-words viva-prep file (what each model does, why it was chosen, what each plot shows).
- Honest labels everywhere: measured vs assumed vs claimed. No inflated results.
- Keep a short AI-use note for the report (what the AI did, what the student did).
- Ask before any large compute.
- Separation: this repository is separate from the RnD proof repository (shreyas20063/planar-heading-extension). Read the
  simulator code and any saved run data there read-only and copy what is needed here. Never push to, branch from, or run jobs
  in the RnD repository, and never use its Kaggle slots: its certification runs have priority on compute.
- Confidential: do not use or quote the PhD student's handwritten notes or any under-review paper. Cite only the student's
  own RnD results. If a definition (e.g. of zones) exists only in confidential material, flag it and ask.
- Plain language for the student (not a native English speaker): short sentences, explain technical terms in one line.
