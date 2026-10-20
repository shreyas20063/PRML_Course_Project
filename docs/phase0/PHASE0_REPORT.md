# Phase 0 report: data audit, labels, theory, generation plan

Date: 2026-10-20. Status: waiting for the student's decision (after the senior meeting, Wed Oct 21).

How to read the labels in this report:
- [measured] = we ran it on this container; the script is in `docs/phase0/bench/`.
- [from code] = read from the RnD simulator code (file:line given).
- [RnD claim] = written in the student's RnD files; we did not re-check it.
- [proposal] = our suggestion; the student must decide.

All RnD paths are in the read-only clone of `shreyas20063/planar-heading-extension`,
branch `claude/pursuer-evader-sim-hxmdwu`, commit `b09bdfb` (2026-10-20).

---

## UPDATE (2026-10-20, later): the student chose the CDC 2024 circle law ONLY

Decision by the student: use only the **constant-circle law** of Singh & Chakraborty,
"Planar Herding of Multiple Evaders by a Single Pursuer", IEEE CDC 2024, pp. 7375-7380,
DOI 10.1109/CDC56724.2024.10886844 (a published paper, so it can be cited).
The pursuer moves on a fixed circle of radius R at constant angular speed w (paper eq. (2)).
This is the switching law with k1 = 0.

Sections U1-U6 below replace Sections 1 and 3-7. Section 2 (data audit) is still valid.
The old switching-law plan is kept below for the record, marked "superseded".

### U1. Short answer (circle law)

1. Old data: still **not usable** (Section 2). The GPU runs used the switching law, not the circle law.
   **We must generate new data.**
2. Big simplification [derived by us, please check]: scale lengths by R and time by 1/w.
   Then the equations (paper eq. (5)) have **only one parameter**, gamma = k / (w R^3).
   - Check: put u = R u', v = R v', t = s / w in eq. (5). You get
     du'/ds = gamma (u' - 1)/d'^3 + v',  dv'/ds = gamma v'/d'^3 - u'.
   - So R, k, w separately do not matter; only gamma matters (and time is counted in laps).
3. Second simplification [paper, Section IV]: with this law the evaders do not affect each other, and the
   pursuer does not react to them. So the flock succeeds **if and only if every evader succeeds alone**.
   One evader's outcome depends only on (gamma, r_i(0)/R, psi_i(0)), where psi = angle relative to the pursuer.
4. Pilot (rough) [measured]: losses are now **common**: 16-33 % of runs for N = 1, 3, 6 when gamma < 0.385,
   and **100 %** (155 of 155) when gamma >= 0.385. So the loss class is no longer a problem.
5. "Deadlock" = only "slow". In a 400-lap single-evader pilot, every run that had not settled had
   gamma < 0.0072. Then the stable ring is tiny (r2* about gamma R) and the approach is very slow.
6. Cost: about 1 core-second per single-evader run of 400 laps [measured]. A 20,000-run dataset is about
   3 core-hours, i.e. under 1 hour on this container.
7. **Main risk for the meeting**: the problem is now **low-dimensional** (one evader: 3 numbers), and the
   true success boundary for one evader can be computed directly (the stable manifold of the saddle point
   (r1*, psi1*)). This affects the grading criteria "novelty" and "technical complexity". See U6, Q-A.

### U2. Labels for the circle law

From code [from code]:
- RnD `herding_sim.py:13, 42-44`: circle mode, theta = w t, radius R, k1 = 0.
- `herding_sim.py:119-121`: r* = root of r^3 - R^2 r + k/w = 0 with 0 < r < R/sqrt3 (the paper's stable ring r2*).
- `herding_sim.py:94-107`: per evader "escaped" (r > R), "gathered" (r <= rho, default rho = 0.3,
  line 265), "in region" otherwise. Duration T = 60 time units by default (line 261). Euler, dt = 0.002 (line 268).
- RnD `SESSION_NOTES.md:51-52`: the simulator reproduced the paper's r* = 0.1255 (w = 2) and 0.2541 (w = 1)
  for R = 2, k = 1 to three decimals [RnD claim]. These are good test values for our own simulator.

Proposed exact labels [proposal]:
- **loss**: some evader has r_i > R (outside the pursuer circle) at any time before T_max
  (or at T_max only; the student chooses), or the numbers blow up.
  Always loss if gamma >= 2/(3 sqrt3) = 0.3849 (no ring exists, paper Lemma 2; pilot: 155/155 lost).
  To check in Phase 1: can an evader that left the circle ever come back? If not, "any time" and "at T_max" agree.
- **success**: every evader settles: abs(r_i - r2*)/r2* < eps_r and abs(psi_i - psi2*) < eps_psi for all lap ends
  from t_s to T_max.
- **herding time**: T_h = first such t_s, in laps. For the flock: the largest of the evaders' times.
- **deadlock (not settled)**: neither. In the pilot this happens only for very small gamma (slow approach).
  Honest name: "not settled by T_max".
- Numbers to choose: T_max (we suggest 400 laps), eps_r (1 %), eps_psi (e.g. 0.05 rad), loss rule
  (any time vs end), and the gamma range (we suggest log-uniform 1e-3 to 0.385, plus 10 % above 0.385).

### U3. Theory to compare with (published CDC 2024 paper, plus the student's checks)

| # | result | computable condition | type | where |
|---|---|---|---|---|
| P1 | ring exists | 0 < gamma = k/(w R^3) < 0.385 (exactly 2/(3 sqrt3)) | **Necessary** for success (if false, no ring, so no success) | paper Lemma 2, eq. (7) |
| P2 | two rings, ordering | R > r1* > R/sqrt3 > r2* > 0, roots of r^3 - R^2 r + k/w = 0 | Fact | paper Lemma 3, eq. (9) |
| P3 | stability | r2* ring asymptotically stable; r1* point a saddle | Local, sufficient near r2* | paper Theorem 5, Theorem 8 |
| P4 | ROA ellipse | evader start inside the Lyapunov ellipse Omega (found numerically by optimisation, for given R, k, w) | **Sufficient** (inner estimate), conservative | paper Section III-C, Problem 6 |
| P5 | ROA in the fixed frame | start inside the intersection of all rotated ellipses (a disk, Fig. 4) | Sufficient, any pursuer start | paper Section III-D, Definition 2 |
| P6 | many evaders | every evader inside the P5 set | Sufficient for flock success | paper Theorem 11 |
| P7 | small ring for fast pursuer | r2* -> 0 as w -> infinity | Fact | paper Lemma 4, Corollary 9 |

What we can compare (plain words):
- P1 is a sharp line (gamma = 0.385). The classifier should find it.
- P4/P5 are inner estimates. We can measure **how much of the true success region they cover**
  ("how conservative"), for each gamma.
- We can also compute the true one-evader boundary directly (stable manifold of the saddle, backward
  integration). This is the exact answer, and it is a strong check of the ML model.
- Important change from BRIEF.md: these results are from the **published paper**, not from the student's own
  proofs. The student's own RnD proofs (sqrt3 lemma for k1 > 0, zones, N = 1/N = 2 results) are about the
  switching law and do **not** apply here. The student's own contribution for k1 = 0 is the validated simulator.
  The BRIEF sentence "compare with conditions proven in the student's own RnD work" must be changed (Q-B).

### U4. Generation plan with cost (circle law) [proposal]

- Sample in the **dimensionless** form: gamma log-uniform in [1e-3, 0.385), plus 10 % in [0.385, 1].
  Per evader: r(0)/R uniform in area on the disk (paper Assumption 1), psi(0) uniform in [0, 2 pi).
  Optionally also report physical (R, k, w) values; they add nothing to the dynamics.
- Latin hypercube over (log gamma, plus the N starting positions), stratified by N = 1..6.
- Size: 3,000 runs per N = **18,000 runs**, plus 2,000 near gamma = 0.385 and near the true boundary = **20,000**.
- GP classifier on a subset of n = 3,000 to 4,000 (exact GP is n^3). SVM and logistic regression on all.
- Cost [measured]: about 1 core-second per single-evader run of 400 laps (400 runs in 108 s on 4 threads).
  The cost grows about linearly with N for small N, so about 1-4 core-seconds per run:
  **20,000 runs = 6 to 20 core-hours worst case, 2 to 5 hours wall** on this 4-core container.
  Cheaper option: simulate single evaders only (one run per evader start) and build the flock label as
  "all evaders succeed". That is exact for this law and costs about 1/3 of the above.
- Exact reference boundary: for each of about 50 gamma values, integrate backward from the saddle point.
  A few seconds per gamma.
- Where: this cloud container only. Not the Mac, never the RnD Kaggle slots. We ask before any run over 10 minutes.

### U5. Pilot evidence (circle law, rough) [measured]

Details in `docs/phase0/bench/OUTPUT.md`.
- `pilot_circle.py` (physical R, k, w; 100 laps; 300 runs per N): gamma < 0.385: lost 40 / 58 / 83 of about
  250 (N = 1 / 3 / 6); gamma >= 0.385: all lost.
- `pilot_circle2.py` (dimensionless, N = 1, 400 laps, 400 runs): lost 55, settled 224, not settled 121
  (all with gamma < 0.0072). Settling lap: median 68, 90 % at 309, max 387.

### U6. Open questions for the senior meeting (circle law)

- **Q-A. Is it enough for the grading?** One evader is a 3-number problem with an exactly computable answer.
  Ideas to keep it rich, for the student and the senior to choose (we do not decide):
  (1) focus on herding time and its uncertainty (GP regression), which has no closed form;
  (2) Bayes-risk decisions ("start herding now or reposition first?");
  (3) learn the flock from features only and test on unseen N (the exact AND rule is the ground truth);
  (4) measure how conservative the paper's ellipse estimate is, across all gamma.
- **Q-B. BRIEF.md change**: theory now comes from the published CDC 2024 paper, not the student's own proofs.
  The title and "contribution" sentences need new wording. Please confirm with the course instructor that this is fine.
- **Q-C. Overlap**: the student's RnD notes mention a hardware study with this law (`SESSION_NOTES.md:31-37`).
  Check with the guide that a course project on the same law does not clash with that paper.
- **Q-D. Labels**: the 5 numbers in U2.
- **Q-E. Simulator**: copying RnD files into `sim/` was blocked by the permission system (old Q6). We suggest
  writing our own small simulator here and testing it against the paper's numbers (r* = 0.1255, 0.2541).
- **Q-F. RnD-side audit**: still not in this repository; cross-check pending.

---

# Superseded: the switching-law version (kept for the record)

## 1. Short answer

1. The old 393,216 GPU runs are **not usable** for this project. They did not save the starting
   positions. They cover only N = 2 and N = 3, only zone A (the "good" zone), and they have no
   loss and no herding time. The files are also not in git (they were Kaggle outputs).
2. No other saved RnD data has per-run starting positions plus outcomes. **We must generate new data.**
3. The three labels (success / deadlock / loss) exist only in a rough form, in one RnD script
   (`stress_test.py`). There is no exact definition of success or herding time for the whole flock.
   We propose exact definitions below. The student must choose 6 numbers.
4. Simulation is cheap: about **0.1 core-second per run** for 60 laps [measured].
   A 60,000-run dataset needs about **5 to 8 core-hours**, i.e. **1.5 to 2 hours** on this container's
   4 cores. No Kaggle, no Mac needed.
5. Two big risks for the meeting:
   - **Loss is rare.** With the switching law: 0 losses in 1,440 pilot runs with kappa = r_max(0)
     (all zones), 0 in 302 zone-A runs with kappa free, and about 4 % losses outside zone A
     with kappa free [measured, rough].
   - **"Deadlock" is mostly "slow".** In zone A, 15 % of runs had not settled after 300 laps, but
     almost all of them are slow, weakly damped runs, not stuck runs [measured, rough].
6. The copy of the RnD simulator into this repository (`sim/`) was **blocked by the permission system**.
   We did not copy anything. See Section 7, question Q6.

---

## 2. Existing data (audit)

### 2.1 The 393,216 GPU runs

Script: `global/pass8/gpu/envelope_gpu_sweep.py`. Summary: `global/pass8/gpu/FINDINGS.md`.

| question | answer | where |
|---|---|---|
| Purpose | Test the "envelope conjecture" (does a running max of r_max ever rise?). Not a success/failure study. | FINDINGS.md, top |
| How many runs | 393,216, two Kaggle notebooks (seeds 0 and 1), Tesla T4 | FINDINGS.md, "Finding 2" |
| Which N | Only N = 2 and N = 3 were reached (the loop is ordered by N; time ran out). Planned N = 2, 3, 5, 10, 30. | FINDINGS.md "Coverage caveat"; script line 26 |
| Which regime | Only zone A (a > 6/e^2); the sampler throws away everything else | script lines 25, 89-94 |
| Parameters saved? | Yes: N, k1, R, kappa, a, k, w, r*, steps per lap | script lines 187-188 |
| Starting positions saved? | **No.** Only r_max(0) ("r0max"). Full positions only for a few hundred re-checked candidates (inside results.json). | script lines 187-188, 234; verify functions lines 48-50 |
| Outcomes saved? | Only "final_dist" after 50 laps (distance to the equilibrium) and envelope rises. No loss, no herding time. | script lines 224, 234 |
| Format and place | `runs.npz` and `results.json` in `/kaggle/working` (Kaggle notebook output) | script lines 30, 284-286 |
| In git? | No. `.gitignore` excludes `global/pass8/**/*.npz` and `global/pass8/gpu/results.json` | RnD `.gitignore` |
| Still on Kaggle? | **Not checked.** The rules say never touch the RnD Kaggle notebooks (slugs `pass8-*`). The student can look in the Kaggle web page (Output tab) of the two GPU notebooks. | - |

Conclusion: even if the Kaggle output still exists, it cannot train our classifier
(no starting positions, one class only, two values of N). At most it can serve as a side check:
"in zone A, every run converged" [RnD claim].

### 2.2 Other RnD data we looked at

| source | what it holds | usable? |
|---|---|---|
| `global/pass8/hunt/h1_*.json` (840,000 runs, N = 2..8) | Only summaries per batch (worst distance, median and max time into the 1e-3 ball) and the 5 slowest starting states. Mostly one parameter set (the running example R=2, k=1, w=2, k1=1, kappa=1). | No (no per-run records). Useful for herding-time scale: median about 22 time units, max about 79 at the running example. |
| `results/stress_seed0.txt`, `stress_seed1.txt` (896 runs) | Final tally per zone, plus a record line only for runs that failed. | No (only failures printed). But the runs are reproducible from the seed. |
| Large JSON files in `global/pass8/pmp_*`, `cert_*`, `mac/` | Proof certificates (interval boxes, LP results). | No (not simulation outcomes). |
| `.npy` / `.npz` files | None in the clone (git-ignored). | - |

### 2.3 A definition that changed over time: kappa

- In the latest note, kappa is a **fixed design constant** (`paper/note.tex:141-144`, Remark at line 285).
- In older scripts, kappa = r_max(0), the starting radius of the farthest evader
  (`stress_test.py:30`, `GLOBAL_attempt.md:9`).
- In the GPU sweep and the hunt, kappa is a free parameter, and r_max(0) is drawn separately
  (`envelope_gpu_sweep.py:90, 125-126`; `hunt/common.py:13`).

This matters: if kappa = r_max(0), the pursuer always starts at radius R, and the features
"pursuer start radius" and "sqrt3 margin" in BRIEF.md lose meaning. **[proposal]** use kappa as a free
design constant (as in the latest note) and sample r_max(0)/kappa separately. Student to confirm (Q2).

---

## 3. Labels

### 3.1 What the code says today [from code]

| script | label | exact rule | where |
|---|---|---|---|
| `stress_test.py` (zone hunt) | lost | at the end (T = 80 time units), some evader is outside the pursuer circle: r_i > R_p | lines 29, 40 |
| | stalled | not lost, final r_max > r* + 0.1, and r_max moved less than 0.02 over the last 20 % of time | lines 42-43 |
| | converged | **everything else** (no direct test of convergence!) | line 91 (`out = ...`) |
| | zones | A: kappa < B0, B: B0 <= kappa < B1, C: kappa >= B1, with B0 = (ln 2 + 2 ln(k1 R))/(2 k1), B1 = (1 + ln(k1 R))/k1 | lines 6-7, 80-81, 89 |
| | integrator | explicit Euler, dt = 0.005 | line 29 |
| `herding_sim.py` (old fixed-index law) | per evader: escaped / in region / gathered | escaped: outside radius R; gathered: within rho = 0.3 of the goal | lines 94-107, 265 |
| `stray_test.py` | lost | evader farther than 1.0 from the goal at T = 80 | lines 14, 52 |
| `hunt/common.py` | herding time "tin" | first time with max_i abs(x_i - x*) < 1e-3 (x* = equilibrium point) | lines 81-90, 139-140 |
| `gpu/envelope_gpu_sweep.py` | not converged | distance to the equilibrium > 1e-3 after 50 laps | line 224 |

Notes:
- The zone A test in `stress_test.py` uses the **old** bound a > 1 (B0). The latest note uses the
  larger set a > 6/e^2, i.e. B0' = (2 ln(k1 R) + 2 - ln 3)/(2 k1) (`paper/note.tex:599-608`).
- So the three-way label is **not** defined exactly in code. There is no T_max in laps, and no
  herding time for "r_max settles at r*".

### 3.2 Proposed exact definitions [proposal]

Notation: one lap = 2 pi / w time units. r* = inner root of r^3 - R_p(r)^2 r + k/w = 0 (unique in zone A).
r_max(t) = largest evader distance from the goal. R_p(t) = R exp(k1 (r_max(t) - kappa)).

1. **loss**: at some time t <= T_max, r_max(t) > r_esc, OR at T_max some evader is outside the
   pursuer circle (r_i > R_p, same as `stress_test.py:40`), OR the numbers blow up (not finite).
2. **success**: there is a time t_s <= T_max such that for every lap end from t_s to T_max,
   abs(r_max - r*) / r* < eps_r. (Option: also ask every evader to be within eps_x of x*.)
3. **herding time**: T_h = the first such t_s, reported in laps and in time units.
4. **deadlock**: neither loss nor success.
5. **Extra flag** (not a 4th class): "stuck" if, over the last W laps, r_max changed by less than
   0.02 and r_max > r* + 0.1 (same idea as `stress_test.py:43`). This separates "stuck" from "slow".

**Numbers the student must choose:**

| symbol | meaning | our suggestion | why |
|---|---|---|---|
| T_max | simulation length | 200 laps | pilot: 90 % of settling times <= 150 laps (zone A, N = 3) |
| eps_r | settling tolerance on r_max | 0.01 (1 % of r*) | used in our pilot |
| eps_x | optional: each evader near x* | off, or 1e-3 as in hunt | 1e-3 is very strict for slow runs |
| r_esc | escape radius | 10 max(R, kappa) | R_p grows like exp(k1 r_max): beyond this the pursuer has lost the evader in practice |
| W | window for the "stuck" flag | last 20 % of T_max | as in `stress_test.py` |
| law | which pursuer law | switching law only | see Q1 |

---

## 4. Theory to compare with (student's own RnD files only)

Notation: C = R^2 exp(-2 k1 kappa), a = 2 k1^2 C = 2 k1^2 R^2 exp(-2 k1 kappa),
beta = k k1^3 / w. "Sufficient" = if it holds, the outcome is guaranteed. "Necessary" = if it fails,
the outcome is impossible.

| # | result | computable condition | type | scope | status | where |
|---|---|---|---|---|---|---|
| T1 | sqrt3 (separation) lemma, zone A | a > 6/e^2 (about 0.812), i.e. kappa < B0' = (2 ln(k1 R) + 2 - ln 3)/(2 k1). Then R_p(r)^2 > 3 r^2 for all r. | Sufficient for: unique equilibrium, no collision, local stability. Sharp for the lemma itself (fails at r = 1/k1 if a < 6/e^2). **Not** sufficient for global success (see T8). | any N, all parameters | [derived, checked] | `paper/note.tex:599-638` |
| T2 | containment | kappa < B1 = (1 + ln(k1 R))/k1. Then R_p(r) > r for every r. | If and only if (for the circle enclosing every radius). As a predictor of "no loss": candidate necessary condition for loss is kappa >= B1 (zone C). | any N | [derived]; "C may lose" is [checked, open] | `paper/note.tex:726-761` |
| T3 | no collision | in zone A, d_i(t) >= (1 - 1/sqrt3) sqrt(C) for all t | Sufficient (safety bound) | any N, zone A | [derived; checked] (observed min 2.04 times the bound) | `global/pass8/C_route.md:26-52`; FINDINGS.md |
| T4 | absorbing radius | r_max(t) <= max(r_max(0), r_B + D) + D, D = (2 pi / w) k / (c^2 C), c = 1 - 1/sqrt3 | Sufficient (bound, no loss in zone A). Very crude (76.6 for the running example). | any N, zone A | [derived; checked] | `C_route.md:211-282` |
| T5 | N = 1 global convergence | parameters equal to one of 10 listed sets (e.g. (k, w, k1, R, kappa) = (1, 2, 1, 2, 1)) | Sufficient, any start | N = 1, 10 points in parameter space | computer-assisted (interval arithmetic) | `C_route.md:297-372`; `global/pass8/n1_pipeline_*.json` |
| T6 | N = 2 global convergence | running example only (R=2, k=1, w=2, k1=1, kappa=1) | Sufficient, any start | N = 2, one point | computer-assisted; marked COMPLETE on 2026-10-20 with open items (code version, student re-derivation) | `C_route.md:586-596` |
| T7 | large-w result | w > w0, with w0 = 2461 (running example; tightened), 2518, 5309, 527, 9654 (sets 2-5) | Sufficient, any N, any start (two extra hypotheses) | 5 parameter sets | [derived; constants verified] | `C_route.md:529-548`; `global/pass8/route_ii/REPORT.md` |
| T8 | counterexample S14 | zone A set (k, w, k1, R, kappa) = (2.099, 1.284, 2.972, 0.894, 0.184), a = 4.73: an attracting periodic orbit for N = 2 (never converges) | Shows zone A is **not** sufficient for success | N = 2 (also seen for 3, 6) | [checked] | `global/pass8/X2A_proof.md:21-35` |
| T9 | beta hypothesis | the S14 cycle appears only for beta above about 18-36 | Candidate: "beta small" might be needed for success | zone A | [checked; open hypothesis] | `global/pass8/hunt/REPORT.md:35-40` |
| T10 | local stability | in zone A, J_c and J_a Hurwitz (det J_c > 0 is exactly the sqrt3 lemma). Basin estimate: all evaders within 9.73e-4 of x* (running example) | Sufficient, tiny region | any N, zone A; basin for one set | [derived, checked]; basin [interim] | `paper/note.tex:840-1000`; `global/pass8/pmp_local/local_lemma.md:16-17` |

What this means for the comparison (plain words):
- Only T1, T2 (and T3, T4) are **surfaces** in parameter space. They are the right things to draw on the
  learned map: "zone A boundary a = 6/e^2" and "containment boundary kappa = B1".
- T5, T6, T7 are **points** or tiny sets. We can only check "the classifier says success there".
- T7 needs w > 527 at least. Our planned range is w <= 16, so it never applies. We must say this honestly.
- T8 and T9 are the most interesting test: does the classifier find a "deadlock" pocket at large beta
  inside zone A?
- The sqrt3 margin in BRIEF.md, m = min_i R_p(0)/(sqrt3 r_i(0)), is our own feature. It is **not** a
  proven condition on the starting state. T1 is a condition on (R, k1, kappa) only.

---

## 5. Generation plan with cost [proposal]

### 5.1 Sampling

- **Cells**: N = 1..6 times 3 zones (A: a > 6/e^2; B: B0' <= kappa < B1; C: kappa >= B1) = 18 cells.
- **Inside each cell**: Latin hypercube (each 1-D range cut into equal slices, one sample per slice)
  over
  - k1 in [0.3, 3] (log scale), R in [0.5, 4], k in [0.1, 5] (log), w in [0.5, 16] (log);
  - kappa chosen to land in the cell (through a, B0', B1);
  - rho = r_max(0)/kappa in [0.1, 1.5];
  - one categorical: starting shape (uniform disk, ring, cluster, two opposite groups),
    taken from the RnD hunt families (`hunt/h1_sweep.py:3-10`).
- **Pursuer start**: angle 0 in the rotating frame (as in all RnD code); the evaders' angles relative to it vary.
- **Extra 10 %**: near the boundaries (a close to 6/e^2, kappa close to B1, large beta) to sharpen the map.
- **Size**: 3,000 runs per cell = 54,000, plus 6,000 near boundaries = **60,000 runs**.
- Why this size: each cell then has thousands of points, enough for 5-fold cross-validation per N,
  and the rare loss class (about 4 % of zones B/C, Section 6) still gets roughly 1,000 examples
  before oversampling.

### 5.2 Model sizes

- Logistic regression and RBF-SVM: train on all data (SVM up to about 20,000 points is minutes).
- **GP classifier**: exact GP costs n^3. We propose **n = 3,000 to 4,000** training points per model
  (stratified subset; 4,000^3 = 6.4e10 operations per step, a few minutes), and report
  results on the full held-out set. If needed, a Nystrom / inducing-point version for larger n.
- GP regression for herding time: successful runs only, target log(T_h), **n about 4,000**.

### 5.3 Cost [measured on this container]

- Single-thread cost: 84 to 156 ms per run for 60 laps (N = 1..6), mean about 0.12 s.
- Four threads: 23 to 49 ms per run (wall).
- Cost grows with the number of laps. For T_max = 200 laps: about 0.3 to 0.5 core-seconds per run.
- **Total**: 60,000 runs x 0.3-0.5 s = 5 to 8 core-hours, i.e. **1.5 to 2 hours wall** on this 4-core container.
- Accuracy check: re-run 1 % (600 runs) with an adaptive high-accuracy solver (scipy DOP853):
  about 1 core-hour more.

### 5.4 Where to run

- **This cloud session's CPU** (4 cores): enough for everything. First choice.
- The student's Mac: only after its RnD jobs are finished. Not needed.
- **Never** the RnD Kaggle slots.
- We will ask before starting any run longer than about 10 minutes.

---

## 6. Pilot evidence (small, rough) [measured]

Details: `docs/phase0/bench/OUTPUT.md`.

- Kappa = r_max(0) (old style, 1,440 runs, 60 laps): **0 losses**.
- Kappa free (894 runs, 60 laps): zone A: 0 losses in 302 runs. Outside zone A: 25 losses in 592 runs (about 4 %).
- Zone A, N = 3, 300 laps (300 runs): 254 settled. Settling lap: median 24, 90 % at 150, max 278.
  The 46 runs that never settled have small beta (all but one below 2.3); r_max still moves.
  So most of them look **slow and weakly damped**, not stuck. This agrees with the RnD finding that
  damping is weak deep in zone A (`gpu/FINDINGS.md`, "Mechanism").

---

## 7. Risks and open questions for the senior meeting (Wed Oct 21)

- **Q1. Loss is rare with the switching law.** Options:
  (a) keep the switching law and oversample zone C (larger kappa);
  (b) also simulate the old fixed-index law (the published arXiv 2505.10048 law), where loss is common
      (32 of 50 runs in `stray_test.py`), and add "law" as a feature. This adds a nice story
      ("ML sees why the switching law is better"), but widens the scope.
  (c) merge to two classes (success vs not). Our suggestion: (a), and (b) only if time allows.
- **Q2. Kappa**: free design constant (latest note) or kappa = r_max(0) (old scripts)? We suggest free.
- **Q3. Deadlock vs slow**: the deadlock label depends strongly on T_max and eps_r. Suggestion: three classes
  with T_max = 200 laps, plus the "stuck" flag, and treat runs not settled as censored times in the
  herding-time model. Say honestly that "deadlock" means "not settled by T_max".
- **Q4. The 6 numbers** in Section 3.2 (T_max, eps_r, eps_x, r_esc, W, law).
- **Q5. Zone B/C accuracy**: the no-collision bound holds only in zone A, so fixed-step integration may be
  inaccurate in zones B/C. Plan: adaptive step check on 1 % of runs, and on every run that is labelled loss.
- **Q6. Copying the simulator**: the copy of RnD files into `sim/` was blocked by the permission system.
  Two options: (a) the student allows the copy (add a permission rule); (b) we write our own small
  simulator here (the equations are 10 lines; see `docs/phase0/bench/bench.py`) and test it against the
  RnD code read-only. Option (b) also keeps the two projects cleanly separate. We suggest (b).
- **Q7. Theory comparison is thin outside zone A**: only two surfaces (T1, T2) cover the whole space.
  The large-w result (T7) is outside our range of w.
- **Q8. Novelty**: learning a region of attraction with GP / SVM is a known general idea. The new parts here
  are: switched multi-agent herding, permutation-invariant features for any N, generalisation to unseen N,
  Bayes-risk decisions, and the comparison with the student's own proofs. Literature check still to do.
- **Q9. RnD-side audit**: `docs/phase0/rnd_side_audit.md` was not yet in this repository when this report
  was written. Cross-check is pending (see Section 9).

---

## 8. Confidentiality note

- We did **not** open or use the PhD student's handwritten notes or any under-review paper.
  None of the files we read quote them; they appear only in rule lines (e.g. RnD `CLAUDE.md`).
- All definitions used here (zones A/B/C, the sqrt3 lemma, B0', B1) come from the student's own files
  (`paper/note.tex`, `stress_test.py`, `MEETING_PREP.md`, `global/pass8/*`).
- The only outside source mentioned is the published law of Singh & Chakraborty (CDC 2024 / arXiv 2505.10048),
  as cited in the student's RnD files.
- Nothing was written to the RnD repository. No RnD Kaggle notebook was touched.

## 9. Cross-check with the RnD-side audit

Pending: the file `docs/phase0/rnd_side_audit.md` was not present at the time of writing (checked with
`git pull`). When it arrives, we will compare it with Sections 2-4 and list any disagreement here.
