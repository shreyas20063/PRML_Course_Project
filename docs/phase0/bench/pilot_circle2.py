"""Circle law, N = 1, dimensionless form (R = 1, w = 1, k = gamma): only gamma = k/(w R^3) matters.
400 laps. Settling = |r - r2*| < 1% r2* from some lap on."""
import numpy as np, numba
from bench import batch
numba.set_num_threads(4)
rng = np.random.default_rng(4)
Xs, ps, ss, g = [], [], [], []
while len(Xs) < 400:
    gam = 10**rng.uniform(-3, np.log10(0.385))
    r = np.sqrt(rng.uniform(0.0025, 0.9)); th = rng.uniform(0, 2*np.pi)
    X = np.array([[r*np.cos(th), r*np.sin(th)]])
    if np.hypot(X[0, 0]-1, X[0, 1]) < 0.05: continue
    spl = int(min(max(400, np.ceil(24*2*np.pi/((0.05)**3/gam))), 12800))
    Xs.append(X); ps.append((1.0, gam, 1.0, 0.0, 0.0)); ss.append(spl); g.append(gam)
Xs, ps, ss, g = map(np.array, (Xs, ps, ss, g))
rl, Pf, rf = batch(Xs, ps, ss, 400)
r2 = np.array([np.sort(np.roots([1, 0, -1, x]).real)[1] for x in g])
lost = (rf > 1) | ~np.isfinite(rf)
err = np.abs(rl - r2[:, None])/r2[:, None]
first = np.array([np.argmax(e < 0.01) if (e < 0.01).any() else -1 for e in err])
sett = np.array([f >= 0 and (e[f:] < 0.01).all() for e, f in zip(err, first)]) & ~lost
other = ~lost & ~sett
print("lost", lost.sum(), "settled", sett.sum(), "other", other.sum())
print("settling lap quantiles 50/90/99/max:", np.percentile(first[sett], [50, 90, 99, 100]))
print("gamma of 'other' (sorted):", np.round(np.sort(g[other]), 4))
print("final r of 'other' vs r2*:", np.round(rl[other, -1][:15], 3), np.round(r2[other][:15], 3))
