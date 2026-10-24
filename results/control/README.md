# Using the model: choosing the pursuer settings for a given flock (checked by simulation)

Code: `models/control_choice.py`. Numbers [measured]: `metrics.json`. Figure: `control.png`. Choices: `choices.npz`.

Set-up: 300 multi-evader flocks from the test set (their start positions are kept). For each flock a strategy chooses
the pursuer settings q = k1 R and beta = k k1^3 / w (15 x 15 grid); every choice is then SIMULATED (300 laps).

| strategy | success | deadlock | loss | median herding time (laps) |
|---|---|---|---|---|
| random (the dataset's setting) | 38.0 % | 31.0 % | 31.0 % | 18.5 |
| fixed default (q = 4, beta = 1) | 61.7 % | 9.0 % | 29.3 % | 37 |
| theory-guided (arXiv condition with margin a = 4, beta = 1) | 83.3 % | 16.7 % | **0 %** | 15 |
| GP: max P(success), then fastest (cure model) | **86.7 %** | 10.3 % | 3.0 % | 6 |
| **GP-safe**: as GP, only where P(loss) <= 1 % | 81.7 % | 18.3 % | **0 %** | **6** |

Paired tests on the same flocks:
- GP vs theory: more successes (10 flocks only GP succeeds, 0 only theory; exact McNemar p = 0.002) but also more
  losses (9 vs 0; p = 0.004): maximising P(success) alone takes risks.
- GP-safe vs theory: same losses (0 and 0); success 81.7 % vs 83.3 % (5 flocks only theory succeeds, 0 only GP-safe;
  p = 0.06, not significant at 5 %); on the 245 flocks where both succeed, GP-safe herds in a median of 6 laps vs 14
  and is faster in 88 % of them (Wilcoxon signed-rank p = 9e-24).

Plain words:
- Using the model to choose the settings changes the outcome a lot compared with random settings (38 % -> 82-87 %).
- The theory gives a good, safe rule of thumb (choose q so that a = 4): no losses. This is a positive result for the theory.
- The risk-aware model choice keeps that safety (no losses) and herds about 2.3x faster; a choice that only maximises
  the success probability gets slightly more successes but loses evaders in 3 % of flocks. Same lesson as the active
  learning: the decision rule must include the cost of the worst mistake.

Limits: one grid of 225 settings; beta = 1 for the theory rule is our choice (the theorem says nothing about beta);
herding time in laps (a lap is 2 pi / w, so a faster w also shortens a lap in seconds).
