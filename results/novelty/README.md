# Novelty experiment: risk-aware active learning of the herding outcome map

Date: 2026-10-23. All numbers [measured], mean of 3 repeats (± = standard deviation over the repeats), on the fixed
test set (12,000 arXiv-law runs). Code: `models/active.py`. Data: `active.json`. Figure: `active.png`.

## Question

Simulations cost time. If the GP may **choose** which runs to simulate next, how many runs does it need to learn the
outcome map as well as with random runs? And does it matter **what** it aims for?

## Three ways to choose the next runs

- **random**: as in Phases 1-3.
- **entropy** ("most unsure"): runs where the GP is least sure which class it is. Standard active learning.
- **risk-aware** (our proposal): runs where the best decision is most expensive in expectation,
  min over decisions j of sum_i P(class i) x COST[i, j], with the Bayes-risk cost matrix of Phase 3
  (calling a loss "success" costs 10). It aims at the boundary where a wrong decision costs most.

All three start from the same 200 random runs (per repeat), use the same fixed GP kernels (Phase 3, with the
signal-variance cap) and the same batches (+100 runs up to 1,000, then +250 up to 4,000). Each batch is the top
runs among 5,000 random candidates (so a batch is not 100 copies of almost the same run).

## Results

| at 4,000 runs | mean Bayes cost (lower better) | log-loss (lower better) | balanced accuracy |
|---|---|---|---|
| random | 0.127 ± 0.002 | 0.162 ± 0.001 | 0.936 ± 0.001 |
| entropy | 0.137 ± 0.002 | **0.155 ± 0.001** | **0.946 ± 0.000** |
| **risk-aware** | **0.099 ± 0.001** | 0.166 ± 0.005 | 0.934 ± 0.001 |

Runs needed to match random sampling at 4,000 runs (first batch size where the 3-repeat mean is at least as good):

| goal | risk-aware | entropy |
|---|---|---|
| Bayes cost | **1,750 runs (2.3x fewer)** | never (worse than random) |
| balanced accuracy | not reached | **1,500 runs (2.7x fewer)** |
| log-loss | not reached | 3,000 runs (1.3x fewer) |

## What this shows (plain words)

1. **Choosing runs pays off**: with the right choice, the same quality needs 2.3x to 2.7x fewer simulations.
2. **What you aim for is what you get.** The risk-aware choice gives a 22 % lower decision cost than random at the same
   number of runs (0.099 vs 0.127), but slightly worse accuracy and log-loss. The entropy choice gives the best accuracy,
   but its decision cost is *worse* than random. So the choice of the next simulation should follow the decision the
   model will be used for (here: "is it safe to call this start a success?").
3. The risk-aware choice picks few lost runs (13 % ± 1 % of its training runs at 4,000, vs 29 % for random): it spends
   its runs near the "success" side of the boundaries, where calling a run "success" by mistake is expensive.
4. The result is stable: the spread over the 3 repeats is small compared with the differences (figure bands).

## Relation to earlier work (honest)

GP classifiers with active learning (entropy-type choice) have been used to estimate regions of attraction of one
dynamical system (e.g. Wang et al., 2023) and missile capture regions (Lee et al., EuroGNC 2022). What we add:
three outcome classes with asymmetric costs, many agents with permutation-invariant features, and an acquisition rule
driven by the Bayes risk of the decision, with the finding that entropy and risk-aware choices win on different goals.
We have not seen this risk-aware rule compared in those works; a fuller literature check is still to do.

## Limits

- Pool-based: the runs are chosen from the 48,000 already simulated training runs. This is the same as simulating on
  demand (the simulator is deterministic), but the candidates follow the Phase 1 sampling design.
- The kernels are fixed (tuned once on 1,000 random runs); re-tuning during the loop may change the curves.
- The cost matrix is our proposal; another cost matrix moves the risk-aware boundary.
- 3 repeats only.
