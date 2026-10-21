"""Herding simulator: one pursuer, N evaders, target at the origin.

Evader model (CDC 2024, eq. (1)): velocity k / d^2, pointing away from the pursuer.
Pursuer (arXiv 2505.10048, eq. (3)): angle w t, radius
    R_p(t) = R exp(k1 (r_0(t) - kappa)),
where evader 0 is the tracked evader (the farthest at t = 0) and kappa = r_0(0).
k1 = 0 gives the CDC 2024 circle law (radius R). switch = True tracks the current
farthest evader instead (the student's RnD law; not used in the main dataset).

We integrate in the frame that rotates with the pursuer (CDC 2024, eq. (5)), so the
pursuer sits at (R_p, 0) and the equations do not depend on time:
    x_i' = k (x_i - R_p e1) / |x_i - R_p e1|^3 + w (y_i, -x_i).
Polar angle in this frame = psi in the papers.

Integrator: classical RK4 with a variable step
    h = min(lap / STEPS_MIN, CFL * d_min^3 / k, time left in the lap),
where d_min is the current smallest pursuer-evader distance (d^3 / k is the time scale
of a close pass). Lap ends are hit exactly, and every check is done at lap ends.
"""
import numpy as np
from numba import njit, prange

STEPS_MIN = 400      # at least this many steps per lap
CFL = 0.05           # step <= CFL * d_min^3 / k
MAX_STEPS = 50_000_000
EPS_R = np.array([0.005, 0.01, 0.02, 0.05])   # settling tolerances recorded for every run
EPS_PSI = 0.05                                 # rad
REC_EVERY = 5                                  # trace sample every 5 laps


@njit(cache=True)
def _rhs(X, out, R, k, w, k1, kap, switch):
    """Fills out = X'. Returns (R_p, d_min)."""
    N = X.shape[0]
    rt = np.sqrt(X[0, 0]**2 + X[0, 1]**2)
    if switch:
        for i in range(1, N):
            rt = max(rt, np.sqrt(X[i, 0]**2 + X[i, 1]**2))
    P = R * np.exp(k1 * (rt - kap))
    dmin = np.inf
    for i in range(N):
        du = X[i, 0] - P
        d2 = du * du + X[i, 1]**2
        d = np.sqrt(d2)
        dmin = min(dmin, d)
        s = k / (d2 * d)
        out[i, 0] = s * du + w * X[i, 1]
        out[i, 1] = s * X[i, 1] - w * X[i, 0]
    return P, dmin


@njit(cache=True)
def _run(X, R, k, w, k1, kap, switch, laps, roots, psis, r_esc, out_last, out_trace):
    """Integrates one run in place (X: (N, 2)). Fills:
      out_last[j, m]: last lap (1-based) at whose end the run was NOT within EPS_R[m] (radius) and
                      EPS_PSI (angle) of root j; 0 = within from the start. -1 for missing roots.
      out_trace[s, 0..2]: every REC_EVERY laps: max_i |r_i - roots[0]|/roots[0], r_max, max_i r_i/R_p.
    Returns (escape_lap or -1, R_p at the end, smallest distance seen, steps, ok flag)."""
    N = X.shape[0]
    T = 2 * np.pi / w
    a = np.empty_like(X); b = np.empty_like(X); c = np.empty_like(X); d = np.empty_like(X)
    Y = np.empty_like(X)
    nroot = roots.shape[0]
    for j in range(nroot):
        for m in range(EPS_R.shape[0]):
            out_last[j, m] = 0 if roots[j] == roots[j] else -1
    esc = -1
    dseen = np.inf
    steps = 0
    P = 0.0
    for L in range(laps):
        t = 0.0
        while t < T:
            P, dm = _rhs(X, a, R, k, w, k1, kap, switch)
            dseen = min(dseen, dm)
            h = min(T / STEPS_MIN, CFL * dm**3 / k, T - t)
            for i in range(N):
                Y[i, 0] = X[i, 0] + 0.5 * h * a[i, 0]; Y[i, 1] = X[i, 1] + 0.5 * h * a[i, 1]
            _rhs(Y, b, R, k, w, k1, kap, switch)
            for i in range(N):
                Y[i, 0] = X[i, 0] + 0.5 * h * b[i, 0]; Y[i, 1] = X[i, 1] + 0.5 * h * b[i, 1]
            _rhs(Y, c, R, k, w, k1, kap, switch)
            for i in range(N):
                Y[i, 0] = X[i, 0] + h * c[i, 0]; Y[i, 1] = X[i, 1] + h * c[i, 1]
            _rhs(Y, d, R, k, w, k1, kap, switch)
            for i in range(N):
                X[i, 0] += h / 6 * (a[i, 0] + 2 * b[i, 0] + 2 * c[i, 0] + d[i, 0])
                X[i, 1] += h / 6 * (a[i, 1] + 2 * b[i, 1] + 2 * c[i, 1] + d[i, 1])
            t += h
            steps += 1
            if steps > MAX_STEPS or not np.isfinite(X[0, 0]):
                return esc, P, dseen, steps, False
        # ---- lap end
        P, dm = _rhs(X, a, R, k, w, k1, kap, switch)
        rmax = 0.0; ratio = 0.0
        for i in range(N):
            ri = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
            rmax = max(rmax, ri); ratio = max(ratio, ri / P)
        if not np.isfinite(rmax):
            return esc, P, dseen, steps, False
        for j in range(nroot):
            rj = roots[j]
            if rj != rj:
                continue
            er = 0.0; ep = 0.0
            for i in range(N):
                ri = np.sqrt(X[i, 0]**2 + X[i, 1]**2)
                er = max(er, abs(ri - rj) / rj)
                dp = np.arctan2(X[i, 1], X[i, 0]) - psis[j]
                dp = (dp + np.pi) % (2 * np.pi) - np.pi
                ep = max(ep, abs(dp))
            for m in range(EPS_R.shape[0]):
                if er >= EPS_R[m] or ep >= EPS_PSI:
                    out_last[j, m] = L + 1
        if (L + 1) % REC_EVERY == 0:
            s = (L + 1) // REC_EVERY - 1
            r0 = roots[0]
            er0 = 0.0
            for i in range(N):
                er0 = max(er0, abs(np.sqrt(X[i, 0]**2 + X[i, 1]**2) - r0) / r0)
            out_trace[s, 0] = er0; out_trace[s, 1] = rmax; out_trace[s, 2] = ratio
        if rmax > r_esc:
            esc = L + 1
            for j in range(nroot):
                for m in range(EPS_R.shape[0]):
                    if out_last[j, m] >= 0:
                        out_last[j, m] = laps + 1      # never settled
            break
    return esc, P, dseen, steps, True


@njit(parallel=True, cache=True)
def run_batch(X0, n, pars, switch, laps, roots, psis, r_esc):
    """X0: (B, 6, 2) starts (only the first n[b] rows used; row 0 = tracked evader).
    pars: (B, 5) = (R, k, w, k1, kappa). roots/psis: (B, 3), NaN-padded. r_esc: (B,)."""
    B = X0.shape[0]
    nrec = laps // REC_EVERY
    Xf = np.full((B, 6, 2), np.nan)
    last = np.empty((B, 3, EPS_R.shape[0]), dtype=np.int64)
    trace = np.full((B, nrec, 3), np.nan, dtype=np.float64)
    esc = np.empty(B, dtype=np.int64); Pf = np.empty(B); dmin = np.empty(B)
    steps = np.empty(B, dtype=np.int64); ok = np.empty(B, dtype=np.bool_)
    for bi in prange(B):
        X = X0[bi, :n[bi]].copy()
        e, P, dm, st, good = _run(X, pars[bi, 0], pars[bi, 1], pars[bi, 2], pars[bi, 3], pars[bi, 4],
                                  switch, laps, roots[bi], psis[bi], r_esc[bi], last[bi], trace[bi])
        Xf[bi, :n[bi]] = X
        esc[bi] = e; Pf[bi] = P; dmin[bi] = dm; steps[bi] = st; ok[bi] = good
    return Xf, last, trace, esc, Pf, dmin, steps, ok


def equilibria(R, k, w, k1, kap):
    """All positive roots r of G(r) = r^3 - R_p(r)^2 r + k/w = 0 with r < R_p(r),
    R_p(r) = R exp(k1 (r - kappa)) (arXiv eq. (9); k1 = 0 gives the CDC cubic, eq. (8)).
    Returns (roots, psis) sorted, at most 3, NaN-padded; psi = arccos(r / R_p(r))."""
    from scipy.optimize import brentq
    Rp = lambda r: R * np.exp(k1 * (r - kap))
    G = lambda r: r**3 - Rp(r)**2 * r + k / w
    # roots need R_p(r) > r; for k1 = 0 that means r < R, for k1 > 0 we search up to 60 (plenty for our ranges)
    hi = R if k1 == 0 else 60.0
    rr = np.geomspace(1e-9, hi, 20001)
    with np.errstate(over="ignore", invalid="ignore"):
        g = G(rr)
    idx = np.where(np.sign(g[1:]) != np.sign(g[:-1]))[0]
    out = [brentq(G, rr[i], rr[i + 1], xtol=1e-15) for i in idx]
    out = [r for r in out if r < Rp(r)][:3]
    roots = np.full(3, np.nan); psis = np.full(3, np.nan)
    for j, r in enumerate(out):
        roots[j] = r; psis[j] = np.arccos(r / Rp(r))
    return roots, psis


def rhs_numpy(t, y, R, k, w, k1, kap, switch=False):
    """Same field for scipy (reference solver)."""
    X = y.reshape(-1, 2)
    r = np.hypot(X[:, 0], X[:, 1])
    P = R * np.exp(k1 * ((r.max() if switch else r[0]) - kap))
    du = X[:, 0] - P
    d3 = (du * du + X[:, 1]**2)**1.5
    return np.stack([k * du / d3 + w * X[:, 1], k * X[:, 1] / d3 - w * X[:, 0]], -1).ravel()
