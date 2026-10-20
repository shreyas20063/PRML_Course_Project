"""Pilot: CDC 2024 constant-circle law (k1 = 0, pursuer radius R fixed). Evaders start in disk radius R (Assumption 1)."""
import numpy as np, numba
from bench import batch
numba.set_num_threads(4)
rng = np.random.default_rng(3)
for N in (1, 3, 6):
    Xs, ps, ss, g = [], [], [], []
    while len(Xs) < 300:
        R = rng.uniform(0.5, 4); k = 10**rng.uniform(-1, np.log10(5)); w = 10**rng.uniform(np.log10(0.3), np.log10(16))
        r = R*np.sqrt(rng.uniform(0.0025, 0.9, N)); th = rng.uniform(0, 2*np.pi, N)
        X = np.stack([r*np.cos(th), r*np.sin(th)], 1)
        if np.min(np.hypot(X[:, 0]-R, X[:, 1])) < 0.05*R: continue
        tau = (0.05*R)**3/k; spl = int(min(max(400, np.ceil(24*(2*np.pi/w)/tau)), 6400))
        Xs.append(X); ps.append((R, k, w, 0.0, 0.0)); ss.append(spl); g.append(k/(w*R**3))
    Xs, ps, ss, g = np.array(Xs), np.array(ps), np.array(ss), np.array(g)
    rl, Pf, rf = batch(Xs, ps, ss, 100)
    ok = g < 0.385
    r2 = np.full(len(g), np.nan)
    for j in np.where(ok)[0]:
        rt = np.sort(np.roots([1, 0, -ps[j, 0]**2, ps[j, 1]/ps[j, 2]]).real); r2[j] = rt[1]
    lost = (rf > Pf) | ~np.isfinite(rf)
    conv = ok & ~lost & (np.abs(rl[:, -1]-r2) < 0.01*r2)
    print(f"N={N}: k/(wR^3)<0.385: {ok.sum()} -> conv {conv[ok].sum()}, lost {lost[ok].sum()}, other {(ok&~conv&~lost).sum()} | "
          f">=0.385: {(~ok).sum()} -> lost {lost[~ok].sum()}, other {(~ok&~lost).sum()}")
