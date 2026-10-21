"""Generate the herding dataset (Phase 1). Plan: docs/phase0/PHASE0_REPORT.md, section A4.

Everything is in dimensionless units.
  arXiv law (main): lengths in units of 1/k1, time in units of 1/w (one lap = 2 pi).
     Simulator parameters (R, k, w, k1, kappa) = (q, beta, 1, 1, kappa~), with q = k1 R,
     beta = k k1^3 / w, kappa~ = k1 r_0(0). Design-set number a = 2 q^2 exp(-2 kappa~).
  circle law (baseline, k1 = 0): lengths in units of R, time in units of 1/w.
     (R, k, w, k1, kappa) = (1, gamma, 1, 0, 0), gamma = k / (w R^3).

Strata (arXiv): i: a > 1 (arXiv Theorem 4 holds), ii: 6/e^2 < a <= 1 (RnD design set only),
iii: a <= 6/e^2. Inside each (N, stratum) cell: Latin hypercube over
(log q, log beta, kappa position in the stratum, angle of the tracked evader, start shape).

usage: python data/generate.py arxiv  CHUNK_START CHUNK_END   (chunks of 1000 runs, 0..59)
       python data/generate.py circle CHUNK_START CHUNK_END   (chunks of 1000 runs, 0..9)
       python data/generate.py pilot                          (600 arXiv runs, timing only)
Each chunk is saved to data/raw/<law>_<chunk>.npz.
"""
import os, sys, time, json
import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))
import numba
from herding import run_batch, equilibria, REC_EVERY, EPS_R, EPS_PSI

LAPS = 300                       # T_max in laps (student default, A2)
CHUNK = 1000
SEED = 2026
SHAPES = ["disk", "ring", "cluster", "two_groups"]
Q_RANGE = {1: (0.8, 12.0), 2: (0.7, 12.0), 3: (0.5, 12.0)}   # k1 R per stratum
BETA_RANGE = (1e-3, 50.0)
A2 = 6 / np.e**2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def kappa_bounds(q, stratum):
    B0 = np.log(np.sqrt(2) * q)               # a = 1
    B0p = np.log(q) + 1 - np.log(3) / 2       # a = 6/e^2
    B1 = 1 + np.log(q)                        # containment (RnD note.tex)
    cap = 0.95 * q                            # evaders start inside the pursuer circle
    if stratum == 1:
        hi = min(B0, cap); lo = 0.1 * hi
    elif stratum == 2:
        lo, hi = max(B0, 0.02), min(B0p, cap)
    else:
        lo, hi = max(B0p, 0.02), min(1.3 * B1, cap)
    return lo, hi


def place_others(rng, n, kap, shape):
    """n - 1 untracked evaders, all strictly inside radius kappa (so evader 0 stays the farthest)."""
    m = n - 1
    if m == 0:
        return np.zeros((0, 2))
    if shape == "disk":
        r = kap * np.sqrt(rng.uniform(0, 0.998, m)); th = rng.uniform(0, 2 * np.pi, m)
        P = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    elif shape == "ring":
        r = kap * rng.uniform(0.9, 0.999, m); th = rng.uniform(0, 2 * np.pi, m)
        P = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    elif shape == "cluster":
        rc = kap * np.sqrt(rng.uniform(0, 0.9)); tc = rng.uniform(0, 2 * np.pi)
        P = np.array([rc * np.cos(tc), rc * np.sin(tc)]) + rng.normal(0, kap * rng.uniform(0.03, 0.3), (m, 2))
    else:  # two_groups
        ph = rng.uniform(0, 2 * np.pi)
        th = ph + np.pi * (np.arange(m) % 2) + rng.normal(0, 0.2, m)
        r = kap * rng.uniform(0.3, 0.999, m)
        P = np.stack([r * np.cos(th), r * np.sin(th)], 1)
    rr = np.hypot(P[:, 0], P[:, 1])
    s = np.minimum(1.0, 0.999 * kap / np.maximum(rr, 1e-12))
    return P * s[:, None]


def arxiv_rows(chunk):
    """Rows of the design for one chunk. Chunks 0..53: 18 cells x 3 chunks; 54..59: boundary sets."""
    rng = np.random.default_rng([SEED, chunk])
    rows = []
    if chunk < 54:
        cell = chunk // 3
        N, stratum = cell // 3 + 1, cell % 3 + 1
        U = qmc.LatinHypercube(d=5, seed=rng).random(CHUNK)
        for u in U:
            ql, qh = Q_RANGE[stratum]
            q = np.exp(np.log(ql) + u[0] * np.log(qh / ql))
            beta = np.exp(np.log(BETA_RANGE[0]) + u[1] * np.log(BETA_RANGE[1] / BETA_RANGE[0]))
            lo, hi = kappa_bounds(q, stratum)
            tries = 0
            while lo >= hi and tries < 100:          # empty stratum at this q: redraw q (counted)
                q = np.exp(rng.uniform(np.log(ql), np.log(qh))); lo, hi = kappa_bounds(q, stratum); tries += 1
            kap = lo + u[2] * (hi - lo)
            rows.append(dict(N=N, q=q, beta=beta, kap=kap, th0=2 * np.pi * u[3],
                             shape=SHAPES[min(int(4 * u[4]), 3)], redraw=tries, set="lhs"))
    else:
        kind = ["near_a1", "near_a1", "near_a2", "near_a2", "big_beta", "big_beta"][chunk - 54]
        for _ in range(CHUNK):
            N = int(rng.integers(1, 7)); q = np.exp(rng.uniform(np.log(0.8), np.log(12.0)))
            if kind == "big_beta":
                beta = rng.uniform(10, 50); lo, hi = kappa_bounds(q, 1); kap = rng.uniform(lo, hi)
            else:
                beta = np.exp(rng.uniform(np.log(1e-3), np.log(50)))
                B = np.log(np.sqrt(2) * q) if kind == "near_a1" else np.log(q) + 1 - np.log(3) / 2
                kap = min(B * (1 + rng.uniform(-0.05, 0.05)), 0.95 * q)
            rows.append(dict(N=N, q=q, beta=beta, kap=kap, th0=rng.uniform(0, 2 * np.pi),
                             shape=SHAPES[int(rng.integers(4))], redraw=0, set=kind))
    X0 = np.full((len(rows), 6, 2), np.nan)
    for b, r in enumerate(rows):
        X0[b, 0] = r["kap"] * np.cos(r["th0"]), r["kap"] * np.sin(r["th0"])
        X0[b, 1:r["N"]] = place_others(rng, r["N"], r["kap"], r["shape"])
    pars = np.array([[r["q"], r["beta"], 1.0, 1.0, r["kap"]] for r in rows])
    r_esc = 10 * np.maximum(pars[:, 0], pars[:, 4])
    meta = dict(N=np.array([r["N"] for r in rows]), q=pars[:, 0], beta=pars[:, 1], kappa=pars[:, 4],
                a=2 * pars[:, 0]**2 * np.exp(-2 * pars[:, 4]), th0=np.array([r["th0"] for r in rows]),
                shape=np.array([r["shape"] for r in rows]), set=np.array([r["set"] for r in rows]),
                redraw=np.array([r["redraw"] for r in rows]))
    return X0, pars, r_esc, meta


def circle_rows(chunk):
    rng = np.random.default_rng([SEED, 1000 + chunk])
    U = qmc.LatinHypercube(d=3, seed=rng).random(CHUNK)
    big = U[:, 0] >= 0.9                       # 10 % with gamma >= 0.385 (no ring exists, CDC Lemma 2)
    g = np.where(big, 0.385 + (U[:, 0] - 0.9) / 0.1 * (1 - 0.385),
                 np.exp(np.log(1e-3) + U[:, 0] / 0.9 * np.log(0.385 / 1e-3)))
    r = np.sqrt(0.0025 + U[:, 1] * (0.998 - 0.0025)); th = 2 * np.pi * U[:, 2]
    X0 = np.full((CHUNK, 6, 2), np.nan)
    X0[:, 0, 0] = r * np.cos(th); X0[:, 0, 1] = r * np.sin(th)
    pars = np.stack([np.ones(CHUNK), g, np.ones(CHUNK), np.zeros(CHUNK), np.zeros(CHUNK)], 1)
    meta = dict(N=np.ones(CHUNK, int), gamma=g, r0=r, psi0=th)
    return X0, pars, np.full(CHUNK, 10.0), meta


def simulate(X0, pars, r_esc, meta, path):
    B = len(pars)
    roots = np.full((B, 3), np.nan); psis = np.full((B, 3), np.nan)
    for b in range(B):
        roots[b], psis[b] = equilibria(*pars[b])
    t0 = time.time()
    Xf, last, trace, esc, Pf, dmin, steps, ok = run_batch(X0, meta["N"].astype(np.int64), pars, False, LAPS,
                                                          roots, psis, r_esc)
    sec = time.time() - t0
    np.savez_compressed(path, X0=X0, pars=pars, r_esc=r_esc, roots=roots, psis=psis, Xf=Xf, last=last,
                        trace=trace.astype(np.float16), esc=esc, Pf=Pf, dmin=dmin, steps=steps, ok=ok,
                        laps=LAPS, eps_r=EPS_R, eps_psi=EPS_PSI, rec_every=REC_EVERY, seconds=sec, **meta)
    return sec, ok


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    numba.set_num_threads(os.cpu_count())
    law = sys.argv[1]
    if law == "pilot":
        rng = np.random.default_rng(7)
        parts = [arxiv_rows(c) for c in (0, 20, 40, 53)]   # (N,stratum) = (1,1) (3,2) (5,2) (6,3)
        take = rng.choice(CHUNK, 150, replace=False)
        X0 = np.concatenate([p[0][take] for p in parts]); pars = np.concatenate([p[1][take] for p in parts])
        r_esc = np.concatenate([p[2][take] for p in parts])
        meta = {k: np.concatenate([p[3][k][take] for p in parts]) for k in parts[0][3]}
        sec, ok = simulate(X0, pars, r_esc, meta, os.path.join(OUT, "pilot.npz"))
        print(f"pilot: {len(pars)} runs, {LAPS} laps, {sec:.1f} s wall on {os.cpu_count()} threads, "
              f"{sec * os.cpu_count() / len(pars):.2f} core-s/run, failed {int((~ok).sum())}")
        sys.exit()
    c0, c1 = int(sys.argv[2]), int(sys.argv[3])
    for c in range(c0, c1):
        path = os.path.join(OUT, f"{law}_{c:02d}.npz")
        if os.path.exists(path):
            print("skip", path, flush=True); continue
        X0, pars, r_esc, meta = (arxiv_rows if law == "arxiv" else circle_rows)(c)
        sec, ok = simulate(X0, pars, r_esc, meta, path)
        print(f"{law} chunk {c:02d}: {sec:7.1f} s, failed {int((~ok).sum())}", flush=True)
