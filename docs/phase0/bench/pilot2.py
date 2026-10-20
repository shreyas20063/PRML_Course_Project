"""Pilot: kappa free; r_max(0) = s * R_p-reference independent. Also fixed-index law not here."""
import numpy as np, numba, sys
from bench import batch, rstar
numba.set_num_threads(4)
rng = np.random.default_rng(1)
for N in (1, 3, 6):
    Xs, ps, ss, z = [], [], [], []
    for _ in range(300):
        k1 = 10**rng.uniform(np.log10(0.3), np.log10(3.0)); R = rng.uniform(0.5, 4.0)
        kap = rng.uniform(0, 3.0); w = 10**rng.uniform(np.log10(0.5), np.log10(16)); k = 10**rng.uniform(-1, np.log10(5))
        rho = rng.uniform(0.05, 3.0)          # r_max(0), independent of kappa
        r = rho*np.sqrt(rng.uniform(0, 1, N)); r[0] = rho; th = rng.uniform(0, 2*np.pi, N)
        X = np.stack([r*np.cos(th), r*np.sin(th)], 1)
        P0 = R*np.exp(k1*(rho - kap))
        dmin0 = np.min(np.hypot(X[:, 0]-P0, X[:, 1]))
        if dmin0 < 0.05: continue
        C = R**2*np.exp(-2*k1*kap); tau = ((1-1/np.sqrt(3))*np.sqrt(C))**3/k
        spl = int(min(max(400, np.ceil(24*(2*np.pi/w)/tau)), 6400))
        a = 2*k1**2*R**2*np.exp(-2*k1*kap)
        Xs.append(X); ps.append((R, k, w, k1, kap)); ss.append(spl); z.append(a)
    Xs, ps, ss, z = np.array(Xs), np.array(ps), np.array(ss), np.array(z)
    rl, Pf, rf = batch(Xs, ps, ss, 60)
    bad = ~np.isfinite(rf)
    lost = (rf > Pf) | bad | (rf > 10)
    rs = np.array([rstar(*p) for p in ps])
    conv = (~lost) & (np.abs(rl[:, -1]-rs) < 0.01*np.maximum(rs, 0.05))
    A = z > 6/np.e**2
    print(f"N={N} n={len(z)}: zoneA {A.sum()} (lost {lost[A].sum()}, conv {conv[A].sum()}) | "
          f"not A {(~A).sum()} (lost {lost[~A].sum()}, conv {conv[~A].sum()}, other {(~A & ~lost & ~conv).sum()}) nonfinite {bad.sum()}", flush=True)
