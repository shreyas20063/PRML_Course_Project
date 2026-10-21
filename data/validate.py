"""Accuracy check: re-run a stratified 1 % sample with scipy DOP853 (rtol 1e-10) and compare labels.

usage: python data/validate.py  ->  data/validation.json
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
from herding import rhs_numpy
from labels import load, label

EPS_PSI = 0.05


def ref_run(job):
    X0, par, roots, psis, r_esc, L, eps_r = job
    w = par[2]; T = 2 * np.pi / w
    esc = lambda t, y, *a: np.hypot(y[0::2], y[1::2]).max() - r_esc
    esc.terminal = True
    t0 = time.time()
    sol = solve_ivp(rhs_numpy, [0, L * T], X0.ravel(), method="DOP853", rtol=1e-10, atol=1e-12,
                    t_eval=T * np.arange(1, L + 1), events=esc, args=tuple(par))
    Y = sol.y.T.reshape(len(sol.t), -1, 2)
    lost_esc = sol.status == 1
    last = np.full(3, np.inf)
    for j in range(3):
        if not np.isfinite(roots[j]):
            continue
        r = np.hypot(Y[..., 0], Y[..., 1]); ps = np.arctan2(Y[..., 1], Y[..., 0])
        dp = np.abs((ps - psis[j] + np.pi) % (2 * np.pi) - np.pi)
        bad = (np.abs(r - roots[j]).max(1) / roots[j] >= eps_r) | (dp.max(1) >= EPS_PSI)
        last[j] = (np.where(bad)[0].max() + 1) if bad.any() else 0
    yf = Y[-1]
    rt = np.hypot(*yf[0])
    P = par[0] * np.exp(par[3] * (rt - par[4]))
    outside = (np.hypot(yf[:, 0], yf[:, 1]) > P).any()
    loss = lost_esc or outside or sol.status < 0
    t_s = last.min()
    cls = "loss" if loss else ("success" if (lost_esc is False and t_s < L) else "deadlock")
    return cls, (t_s if cls == "success" else np.nan), time.time() - t0


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    report = {}
    for law, per_chunk in (("arxiv", 10), ("circle", 10)):
        d = load(law)
        cls, T_h, stuck, _ = label(d, 0.01)
        idx = np.concatenate([rng.choice(np.where(d["chunk"] == c)[0], per_chunk, replace=False)
                              for c in np.unique(d["chunk"])])
        jobs = [(d["X0"][i, :d["N"][i]], d["pars"][i], d["roots"][i], d["psis"][i], d["r_esc"][i],
                 int(d["laps"]), 0.01) for i in idx]
        with Pool(os.cpu_count()) as p:
            res = p.map(ref_run, jobs, chunksize=1)
        rc = np.array([r[0] for r in res]); rt = np.array([r[1] for r in res])
        agree = rc == cls[idx]
        both = (rc == "success") & (cls[idx] == "success")
        report[law] = dict(
            checked=int(len(idx)), label_agreement=float(agree.mean()),
            disagreements=[dict(run=int(i), fast=str(cls[i]), reference=str(c)) for i, c in zip(idx[~agree], rc[~agree])],
            herding_time_max_abs_diff_laps=float(np.nanmax(np.abs(rt[both] - T_h[idx][both]))) if both.any() else None,
            herding_time_median_abs_diff_laps=float(np.nanmedian(np.abs(rt[both] - T_h[idx][both]))) if both.any() else None,
            reference_seconds_total=float(sum(r[2] for r in res)))
        print(law, {k: v for k, v in report[law].items() if k != "disagreements"}, flush=True)
    json.dump(report, open(os.path.join(HERE, "validation.json"), "w"), indent=1)
