# Where does herding succeed?

EE604T (PRML) course project, IIT Dharwad.
One pursuer moving on a fixed circle (CDC 2024 law of Singh & Chakraborty) herds N evaders to a goal. We simulate many starting configurations and parameters,
label each run (success / deadlock / loss, herding time), and learn the success region with classical ML
(logistic regression, SVM, Gaussian processes, GMM/PCA, Bayes risk).
Then we compare the learned region with the paper's conditions and region-of-attraction estimate.
Status: Phase 0 (data audit and plan) done; waiting for the student's decision before Phase 1.

## Folder map
- `BRIEF.md` - project scope, rules, milestones
- `docs/phase0/PHASE0_REPORT.md` - data audit, labels, theory list, generation plan, open questions
- `docs/phase0/bench/` - cost benchmark and pilot scripts with their output
- `docs/VIVA_PREP.md` - plain-words viva notes (grows with the project)
- `docs/AI_USE_NOTE.md` - what the AI did, what the student decided
