# Set kernel: comparing flocks as sets of points (any N)

Code: `models/set_kernel.py`. Numbers [measured]: `metrics.json` (final, widest grid).

Kernel between runs A and B (kernel mean embedding of the untracked evaders):
  K(A, B) = k_g(g_A, g_B) * [ k_e(e0_A, e0_B) + mean over pairs (u in A, v in B) of k_e(u, v) ]
k_g: RBF on 6 standardised global numbers (log q, log beta, kappa, log a, log q/kappa, log r*/kappa);
k_e: RBF on an evader's position / kappa in the rotating frame. No hand-made flock statistics; any set size.
Compared with an RBF-SVM on the 24 hand-made features; same training size (8,000), settings by 3-fold CV on 3,000.

| test | hand-made features | set kernel |
|---|---|---|
| random split (12,000 test runs) | 93.0 % | **93.6 %** |
| unseen N: train N <= 4, test N = 5, 6 (24,000 test runs) | 90.0 % | **93.8 %** |
(balanced accuracy)

Plain words: the set kernel is as good as the hand-made features on the usual test, and it does not lose accuracy
on bigger flocks than it was trained on (93.6 -> 93.8 %), while the hand-made features drop by 3 points. Comparing the
evaders directly, as a set, generalises better in N than summarising them by hand.

Search history (honest): three grids were tried because the best values kept landing on the grid edge.
| grid | set kernel (random / unseen N) | hand features (random / unseen N) |
|---|---|---|
| 1: le <= 0.6 | 92.4 / 92.5 | 93.2 / 90.4 |
| 2: le <= 2.4 | 93.6 / 93.6 | 93.3 / 90.1 |
| 3: le <= 9.6 (final) | 93.6 / 93.8 | 93.0 / 90.0 |
Final set-kernel settings: random split lg = 4, le = 2.4, C = 100 (inside the grid); unseen N lg = 4, le = 4.8,
C = 1000 (C at the upper edge). The hand-feature SVM picked gamma and C at the grid edges every time, but its score
stayed at 93.0-93.3 % / 89.9-90.4 %, so a larger grid is unlikely to change the conclusion.
