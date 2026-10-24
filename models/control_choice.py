"""Using the model: choose the pursuer settings for a given flock, then check by simulation.

For held-out flocks (start positions from the test set, in units of kappa), the operator may choose
q = k1 R (pursuer circle size) and beta = k k1^3 / w (push strength / speed) on a grid. Strategies:
  original : the random setting the run had in the dataset;
  fixed    : one default for every flock (q = 4, beta = 1);
  theory   : satisfy the arXiv uniqueness/stability condition with margin, a = 2 q^2 exp(-2 kappa) = 4, beta = 1
             (the theorem says nothing about beta);
  gp       : the GP classifier's best grid point: max P(success), ties (within 0.02) broken by the shortest
             predicted herding time (censored cure model, time if the run settles).
  gp_safe  : risk-aware: as gp, but only among grid points with predicted P(loss) <= 0.01 (if none, the
             smallest P(loss)).
Paired comparison of success (and of loss) between strategies on the same flocks: exact McNemar test.
Every chosen setting is SIMULATED (300 laps) and labelled with the dataset rules.

usage: python models/control_choice.py  ->  results/control/metrics.json, results/control/control.png
"""
import os, sys, json, re, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim")); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table, test_split, run_features
from gp_classify import OvRGP, CLASSES, GP_N, SEED
from gp_map import saved_kernels
from censored_time import KernelCure, gp_time_kernel, C
from labels import load, label
from herding import run_batch, equilibria

OUT = os.path.join(HERE, "..", "results", "control")
N_FLOCKS, LAPS = 300, 300
QS = np.geomspace(0.8, 12, 15)
BS = np.geomspace(1e-3, 50, 15)


def simulate(X0, n, q, beta, kap):
    B = len(q)
    pars = np.stack([q, beta, np.ones(B), np.ones(B), kap], 1)
    roots = np.full((B, 3), np.nan); psis = np.full((B, 3), np.nan)
    for b in range(B):
        roots[b], psis[b] = equilibria(*pars[b])
    Xf, last, trace, esc, Pf, dmin, steps, ok = run_batch(X0, n.astype(np.int64), pars, False, LAPS, roots, psis,
                                                          10 * np.maximum(q, kap))
    dd = dict(Xf=Xf, last=last, trace=trace, esc=esc, Pf=Pf, ok=ok, laps=LAPS, eps_r=np.array([0.005, 0.01, 0.02, 0.05]))
    cls, T_h, stuck, j = label(dd, 0.01)
    return cls, T_h


def summary(cls, T_h):
    return dict(success=float(np.mean(cls == "success")), deadlock=float(np.mean(cls == "deadlock")),
                loss=float(np.mean(cls == "loss")),
                median_time_success=float(np.nanmedian(T_h)) if np.isfinite(T_h).any() else None,
                mean_time_capped=float(np.mean(np.where(cls == "success", T_h, LAPS))))


def plot(out, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    keys = ["original", "fixed", "theory", "gp", "gp_safe"]
    labs = ["random\n(dataset)", "fixed\ndefault", "theory\n(a = 4)", "GP\nmax P(succ.)", "GP-safe\nP(loss)<=1%"]
    bottom = np.zeros(len(keys))
    for c, col in zip(CLASSES, ["#4c9f70", "#e3b23c", "#c44536"]):
        v = np.array([out[k][c] for k in keys]); axes[0].bar(labs, v, bottom=bottom, color=col, label=c); bottom += v
    axes[0].set_ylabel("fraction of the 300 flocks (simulated)"); axes[0].legend(fontsize=8, loc="lower right")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_title("Outcome when the setting is chosen by each strategy", fontsize=10)
    for k, lab in zip(keys, labs):
        Th = res[k][1]; Th = Th[np.isfinite(Th)]
        axes[1].hist(Th + 1, bins=np.geomspace(1, 301, 25), histtype="step", lw=1.5, label=lab.replace("\n", " "))
    axes[1].set_xscale("log"); axes[1].set_xlabel("herding time + 1 (laps), successful runs"); axes[1].legend(fontsize=8)
    axes[1].set_title("How fast (successful runs only)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "control.png"), dpi=130)


def paired_time_test(res, a, b):
    """Wilcoxon signed-rank test of herding times on the flocks where both strategies succeed."""
    from scipy.stats import wilcoxon
    both = (res[a][0] == "success") & (res[b][0] == "success")
    ta, tb = res[a][1][both], res[b][1][both]
    return dict(n_both=int(both.sum()), median_a=float(np.median(ta)), median_b=float(np.median(tb)),
                frac_a_faster=float(np.mean(ta < tb)), p_value=float(wilcoxon(ta, tb).pvalue))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:] == ["--plot"]:          # redraw + paired time tests from the saved results
        out = json.load(open(os.path.join(OUT, "metrics.json")))
        z = np.load(os.path.join(OUT, "choices.npz"), allow_pickle=True)
        res = {k: (z[f"cls_{k}"], z[f"T_{k}"]) for k in ("original", "fixed", "theory", "gp", "gp_safe")}
        out["paired_time"] = {f"{a}_vs_{b}": paired_time_test(res, a, b) for a, b in (("gp_safe", "theory"), ("gp", "theory"))}
        print(out["paired_time"])
        json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
        plot(out, res); sys.exit()
    X, names, y, T_h, meta = arxiv_table()
    d = load("arxiv")
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr])
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    sub, _ = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=SEED)
    gp = OvRGP().fit(sc.transform(X[sub]), y[sub], kernels=saved_kernels())
    amp, ls = gp_time_kernel()
    keep = tr[y[tr] != "loss"]; cens = y == "deadlock"
    t = np.where(cens, C, np.log1p(np.nan_to_num(T_h, nan=300)))
    mult = json.load(open(os.path.join(HERE, "..", "results", "censored", "metrics.json")))["cure"]["prior_mult"]
    tob = KernelCure(ls, amp).fit(sc.transform(X[keep]), t[keep], cens[keep], mult)

    rng = np.random.default_rng(7)
    flocks = rng.choice(te[meta["N"][te] >= 2], N_FLOCKS, replace=False)       # multi-evader flocks
    choice = {k: np.zeros((N_FLOCKS, 2)) for k in ("original", "fixed", "theory", "gp", "gp_safe")}
    pred_best = np.zeros(N_FLOCKS)
    t0 = time.time()
    for f, b in enumerate(flocks):
        n = int(d["N"][b]); X0 = d["X0"][b, :n]; kap = d["kappa"][b]
        choice["original"][f] = d["q"][b], d["beta"][b]
        choice["fixed"][f] = max(4.0, kap / 0.95 * 1.01), 1.0
        choice["theory"][f] = max(np.sqrt(2.0) * np.exp(kap), kap / 0.95 * 1.01), 1.0     # a = 4
        cand = [(q, be) for q in QS for be in BS if q > kap / 0.95]
        F = []
        for q, be in cand:
            r, p = equilibria(q, be, 1.0, 1.0, kap)
            F.append(list(run_features(X0, q, be, kap, r[0], p[0]).values()))
        Zc = sc.transform(np.array(F))
        PP = gp.predict_proba(Zc); P = PP[:, 0]; tm = tob.f(Zc)
        ok = P >= P.max() - 0.02
        i = np.flatnonzero(ok)[np.argmin(tm[ok])]
        choice["gp"][f] = cand[i]; pred_best[f] = P[i]
        safe = PP[:, 2] <= 0.01
        if not safe.any():
            safe = PP[:, 2] <= PP[:, 2].min() + 1e-12
        Ps = np.where(safe, P, -1)
        ok = Ps >= Ps.max() - 0.02
        i = np.flatnonzero(ok)[np.argmin(tm[ok])]
        choice["gp_safe"][f] = cand[i]
        if f % 50 == 0:
            print(f"flock {f}: {time.time() - t0:.0f} s", flush=True)

    out = {"n_flocks": N_FLOCKS, "grid": dict(q=QS.tolist(), beta=BS.tolist()), "gp_mean_predicted_P_success": float(pred_best.mean())}
    X0 = d["X0"][flocks]; n = d["N"][flocks]; kap = d["kappa"][flocks]
    res = {}
    for k, ch in choice.items():
        cls, Th = simulate(X0, n, ch[:, 0], ch[:, 1], kap)
        res[k] = (cls, Th); out[k] = summary(cls, Th)
        print(k, out[k], flush=True)
    from scipy.stats import binomtest
    def mcnemar(a, b):
        n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
        return dict(only_first=n10, only_second=n01,
                    p_value=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0)
    out["mcnemar"] = {}
    for a, b in (("gp", "theory"), ("gp_safe", "theory"), ("gp_safe", "gp"), ("gp", "fixed")):
        out["mcnemar"][f"{a}_vs_{b}"] = dict(success=mcnemar(res[a][0] == "success", res[b][0] == "success"),
                                             loss=mcnemar(res[a][0] == "loss", res[b][0] == "loss"))
    print(out["mcnemar"], flush=True)
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, "choices.npz"), flocks=flocks, **{f"choice_{k}": v for k, v in choice.items()},
                        **{f"cls_{k}": v[0] for k, v in res.items()}, **{f"T_{k}": v[1] for k, v in res.items()})
    out["paired_time"] = {f"{a}_vs_{b}": paired_time_test(res, a, b) for a, b in (("gp_safe", "theory"), ("gp", "theory"))}
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    plot(out, res)
