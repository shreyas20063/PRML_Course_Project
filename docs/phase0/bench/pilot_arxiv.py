"""Pilot: arXiv 2505.10048 law (no switching). Pursuer radius R exp(k1 (|x_0(t)| - kappa)),
x_0 = evader farthest at t = 0, kappa = |x_0(0)|. Rotating frame, fixed-step RK4, 150 laps.
Parameters sampled as RnD stress_test.py:76-89. Rough tallies only."""
import numpy as np, numba, time
from numba import njit, prange
from scipy.optimize import brentq
numba.set_num_threads(4)

@njit(cache=True)
def f(X, out, R, k, w, k1, kap, switch):
    r0 = np.sqrt(X[0, 0]**2 + X[0, 1]**2)
    if switch:
        for i in range(X.shape[0]):
            r0 = max(r0, np.sqrt(X[i, 0]**2 + X[i, 1]**2))
    P = R * np.exp(k1 * (r0 - kap))
    for i in range(X.shape[0]):
        du = X[i, 0] - P; d2 = du*du + X[i, 1]**2; s = k / (d2*np.sqrt(d2))
        out[i, 0] = s*du + w*X[i, 1]; out[i, 1] = s*X[i, 1] - w*X[i, 0]
    return P

@njit(parallel=True, cache=True)
def batch(X0s, pars, spl, laps, switch):
    B, N = X0s.shape[0], X0s.shape[1]; rf = np.empty((B, N)); Pf = np.empty(B)
    for j in prange(B):
        X = X0s[j].copy(); R, k, w, k1, kap = pars[j]; h = 2*np.pi/w/spl[j]
        a = np.empty_like(X); b = np.empty_like(X); c = np.empty_like(X); d = np.empty_like(X)
        for s in range(spl[j]*laps):
            f(X, a, R, k, w, k1, kap, switch); f(X + 0.5*h*a, b, R, k, w, k1, kap, switch)
            f(X + 0.5*h*b, c, R, k, w, k1, kap, switch); f(X + h*c, d, R, k, w, k1, kap, switch)
            X += h/6*(a + 2*b + 2*c + d)
            if not np.isfinite(X).all() or np.abs(X).max() > 1e3: break
        Pf[j] = f(X, a, R, k, w, k1, kap, switch)
        for i in range(N): rf[j, i] = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
    return rf, Pf

rng = np.random.default_rng(5)
def draw(N):
    while True:
        k1 = 10**rng.uniform(np.log10(0.3), np.log10(3)); R = rng.uniform(1, 4)
        if k1*R < 0.5: continue
        B1 = (1 + np.log(k1*R))/k1
        if B1 <= 0.05: continue
        kap = min(rng.uniform(0.3, 1.15)*B1, 0.95*R); w = 10**rng.uniform(np.log10(0.5), np.log10(16)); break
    r = kap*np.sqrt(rng.uniform(0, 1, N)); r[0] = kap; th = rng.uniform(0, 2*np.pi, N)
    C = R**2*np.exp(-2*k1*kap); tau = ((1 - 1/np.sqrt(3))*np.sqrt(C))**3
    spl = int(min(max(400, np.ceil(24*(2*np.pi/w)/tau)), 6400))
    return np.stack([r*np.cos(th), r*np.sin(th)], 1), (R, 1.0, w, k1, kap), spl
def rstar(R, k, w, k1, kap):
    G = lambda r: r**3 - (R*np.exp(k1*(r - kap)))**2*r + k/w
    rr = np.linspace(1e-6, 20, 20001); s = np.sign(G(rr)); i = np.where(np.diff(s) != 0)[0]
    return brentq(G, rr[i[0]], rr[i[0]+1]) if len(i) else np.nan
X, p, s = draw(2); batch(X[None], np.array([p]), np.array([400]), 2, False)
for N in (2, 3, 6):
    D = [draw(N) for _ in range(300)]
    Xs = np.array([d[0] for d in D]); ps = np.array([d[1] for d in D]); ss = np.array([d[2] for d in D])
    rs = np.array([rstar(*p) for p in ps])
    for switch in (False, True):
        t0 = time.time(); rf, Pf = batch(Xs, ps, ss, 150, switch); dt = time.time() - t0
        lost = (rf > Pf[:, None]).any(1) | ~np.isfinite(rf).all(1)
        nlost_ev = (rf > Pf[:, None]).sum(1)
        conv = ~lost & (np.abs(rf - rs[:, None]) < 0.01*np.maximum(rs, 0.05)[:, None]).all(1)
        name = "switching" if switch else "arXiv (fixed index)"
        print(f"N={N} {name:20s}: lost {lost.sum():3d}/300 (mean evaders lost in lost runs {nlost_ev[lost].mean() if lost.any() else 0:.1f}), "
              f"converged {conv.sum():3d}, other {(~lost & ~conv).sum():3d} | {dt:.0f} s", flush=True)
