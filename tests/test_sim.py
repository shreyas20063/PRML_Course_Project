"""Checks of sim/herding.py. Run: python -m pytest -q tests"""
import sys, os
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim"))
from herding import run_batch, equilibria, rhs_numpy


def one(X0, par, laps, switch=False, r_esc=1e3):
    n = len(X0)
    X = np.full((1, 6, 2), np.nan); X[0, :n] = X0
    roots, psis = equilibria(*par)
    out = run_batch(X, np.array([n]), np.array([par], float), switch, laps,
                    roots[None], psis[None], np.array([r_esc]))
    return out, roots, psis


def test_cdc_cubic_roots():
    # CDC 2024 Fig. 5: R = 2, k = 1: w = 1 gives r2* = 0.25 (0.2541), w = 2 gives 0.1255 (RnD SESSION_NOTES.md:51)
    assert abs(equilibria(2, 1, 1, 0, 0)[0][0] - 0.2541) < 5e-4
    assert abs(equilibria(2, 1, 2, 0, 0)[0][0] - 0.1255) < 5e-4
    # outer root r1* between R/sqrt3 and R (CDC Lemma 3)
    r = equilibria(2, 1, 1, 0, 0)[0]
    assert 2 / np.sqrt(3) < r[1] < 2


def test_arxiv_running_example_converges():
    # running example (R, k, w, k1, kappa) = (2, 1, 2, 1, 1): r* = 0.4458, psi* = 1.1724 (RnD MEETING_PREP.md)
    par = (2.0, 1.0, 2.0, 1.0, 1.0)
    roots, psis = equilibria(*par)
    assert abs(roots[0] - 0.4458) < 5e-4 and abs(psis[0] - 1.1724) < 5e-4
    # one evader at radius kappa = 1 (with N >= 2 an untracked evader may escape under this law)
    (Xf, last, trace, esc, Pf, dmin, steps, ok), _, _ = one(np.array([[np.cos(0.3), np.sin(0.3)]]), par, 60)
    assert ok[0] and esc[0] == -1
    assert last[0, 0, 1] < 60                       # settled within 1 %
    assert abs(np.hypot(*Xf[0, 0]) - roots[0]) < 1e-3


def test_rk4_matches_dop853():
    rng = np.random.default_rng(0)
    worst = 0.0
    for par in [(2.0, 1.0, 2.0, 1.0, None), (1.5, 0.3, 1.0, 2.0, None), (3.0, 2.0, 5.0, 0.5, None), (1.0, 0.05, 1.0, 0.0, 0.0)]:
        X0 = rng.uniform(-0.8, 0.8, (4, 2))
        X0 = X0[np.argsort(-np.hypot(*X0.T))]
        kap = np.hypot(*X0[0]) if par[4] is None else 0.0
        p = par[:4] + (kap,)
        laps = 5
        (Xf, *_), _, _ = one(X0, p, laps)
        T = laps * 2 * np.pi / p[2]
        ref = solve_ivp(rhs_numpy, [0, T], X0.ravel(), method="DOP853", rtol=1e-12, atol=1e-13, args=p)
        worst = max(worst, np.abs(ref.y[:, -1].reshape(-1, 2) - Xf[0, :4]).max())
    assert worst < 1e-6, worst


def test_rnd_stray_result():
    # RnD stray_test.py: R = 2, k1 = 1, w = 2, k = 1, N = 6, seeds 0..49, loss = r > 1.0 at T = 80;
    # RnD claim: 32 of 50 runs lose an evader with the fixed-index law, 0 with switching.
    par0 = (2.0, 1.0, 2.0, 1.0)
    laps = int(round(80 / np.pi))                  # T = 80 -> 25.5 laps; we use 25 laps (78.5 time units)
    lost = {False: 0, True: 0}
    for s in range(50):
        rng = np.random.default_rng(s)
        r = np.sqrt(rng.uniform(0.05, 1.0, 6)); th = rng.uniform(0, 2 * np.pi, 6)
        X0 = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        X0 = X0[np.argsort(-r)]
        p = par0 + (r.max(),)
        for sw in (False, True):
            (Xf, *_), _, _ = one(X0, p, laps, switch=sw)
            lost[sw] += bool((np.hypot(Xf[0, :, 0], Xf[0, :, 1]) > 1.0).any())
    print("lost runs (fixed, switching):", lost)
    assert abs(lost[False] - 32) <= 3 and lost[True] == 0
