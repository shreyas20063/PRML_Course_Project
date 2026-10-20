"""Phase 0 cost benchmark (own code, not a copy). Same ODE as the RnD switching law,
written in the frame that rotates with the pursuer (pursuer at (P, 0)):
  x_i' = k (x_i - P e1)/|x_i - P e1|^3 + w (y_i, -x_i),  P = R exp(k1 (r_max - kappa)).
Parameter and IC sampling copied in spirit from RnD stress_test.py lines 76-89 (kappa = r_max(0)).
Fixed-step RK4. Prints wall time per run and a crude outcome tally (pilot only, not labels)."""
import sys, time, numpy as np
from numba import njit, prange
from scipy.optimize import brentq


@njit(cache=True)
def f(X, out, R, k, w, k1, kap):
    rm = 0.0
    for i in range(X.shape[0]):
        rm = max(rm, np.sqrt(X[i, 0]**2 + X[i, 1]**2))
    P = R * np.exp(k1 * (rm - kap))
    for i in range(X.shape[0]):
        du = X[i, 0] - P; d2 = du * du + X[i, 1]**2; s = k / (d2 * np.sqrt(d2))
        out[i, 0] = s * du + w * X[i, 1]; out[i, 1] = s * X[i, 1] - w * X[i, 0]
    return rm, P


@njit(cache=True)
def one(X, R, k, w, k1, kap, steps_per_lap, laps):
    h = (2 * np.pi / w) / steps_per_lap
    a = np.empty_like(X); b = np.empty_like(X); c = np.empty_like(X); d = np.empty_like(X); Y = np.empty_like(X)
    rmax_lap = np.empty(laps); P = 0.0
    for L in range(laps):
        hi = 0.0
        for s in range(steps_per_lap):
            rm, P = f(X, a, R, k, w, k1, kap); hi = max(hi, rm)
            Y[:] = X + 0.5 * h * a; f(Y, b, R, k, w, k1, kap)
            Y[:] = X + 0.5 * h * b; f(Y, c, R, k, w, k1, kap)
            Y[:] = X + h * c; f(Y, d, R, k, w, k1, kap)
            X += h / 6 * (a + 2 * b + 2 * c + d)
        rmax_lap[L] = hi
    rm, P = f(X, a, R, k, w, k1, kap)
    return X, rmax_lap, P


@njit(parallel=True, cache=True)
def batch(X0s, pars, spl, laps):
    B = X0s.shape[0]; out = np.empty((B, laps)); Pf = np.empty(B); rf = np.empty(B)
    for j in prange(B):
        X, rl, P = one(X0s[j].copy(), pars[j, 0], pars[j, 1], pars[j, 2], pars[j, 3], pars[j, 4], spl[j], laps)
        out[j] = rl; Pf[j] = P; rf[j] = np.sqrt((X**2).sum(1)).max()
    return out, Pf, rf


def rstar(R, k, w, k1, kap):
    G = lambda r: r**3 - (R * np.exp(k1 * (r - kap)))**2 * r + k / w
    rr = np.linspace(1e-6, 20, 20001); s = np.sign(G(rr)); i = np.where(np.diff(s) != 0)[0]
    return brentq(G, rr[i[0]], rr[i[0] + 1]) if len(i) else np.nan


def draw(rng, N):
    while True:
        k1 = 10**rng.uniform(np.log10(0.3), np.log10(3.0)); R = rng.uniform(1.0, 4.0)
        if k1 * R < 0.5: continue
        B1 = (1 + np.log(k1 * R)) / k1
        if B1 <= 0.05: continue
        kap = min(rng.uniform(0.3, 1.15) * B1, 0.95 * R); w = 10**rng.uniform(np.log10(0.5), np.log10(16))
        break
    r = kap * np.sqrt(rng.uniform(0, 1, N)); r[0] = kap; th = rng.uniform(0, 2 * np.pi, N)
    X = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    C = R**2 * np.exp(-2 * k1 * kap); tau = ((1 - 1 / np.sqrt(3)) * np.sqrt(C))**3 / 1.0
    spl = int(min(max(400, np.ceil(24 * (2 * np.pi / w) / tau)), 12800))
    return X, (R, 1.0, w, k1, kap), spl


if __name__ == "__main__":
    B = int(sys.argv[1]); LAPS = int(sys.argv[2]); threads = sys.argv[3]
    import numba; numba.set_num_threads(int(threads))
    rng = np.random.default_rng(0)
    # warm-up compile
    X, p, s = draw(rng, 2); batch(X[None], np.array([p]), np.array([400]), 2)
    for N in range(1, 7):
        Xs, ps, ss = zip(*[draw(rng, N) for _ in range(B)])
        Xs = np.array(Xs); ps = np.array(ps); ss = np.array(ss)
        t0 = time.time(); rl, Pf, rf = batch(Xs, ps, ss, LAPS); dt = time.time() - t0
        rs = np.array([rstar(*p) for p in ps])
        lost = rf > Pf; conv = (~lost) & (np.abs(rl[:, -1] - rs) < 0.01 * np.maximum(rs, 0.05))
        stall = (~lost) & (~conv)
        print(f"N={N} B={B} laps={LAPS} threads={threads}: {dt:.2f}s total, {dt/B*1e3:.1f} ms/run "
              f"(wall), mean steps/lap {ss.mean():.0f} | pilot lost {lost.sum()} conv {conv.sum()} other {stall.sum()} "
              f"| nan {np.isnan(rl[:, -1]).sum()}", flush=True)
