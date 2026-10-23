"""Phase 3b: herding time with uncertainty (GP regression), successful runs only.

Target: log(1 + T_h), T_h in laps. Kernel: constant x RBF (one length scale per feature) + white noise.
Settings tuned on HYPER_N successful training runs, then the GP is refitted on GP_N runs with them fixed.
Baselines: the median time; linear regression on the same features.
Censoring check (not yet a censored model, see BRIEF "planned extensions"): deadlock runs only tell us T_h > T_max;
we report the GP's predicted probability P(T_h > T_max) for them, which should be high.

usage: python models/gp_time.py  ->  results/phase3/time_metrics.json, results/phase3/time.png
"""
import os, sys, json, time
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import arxiv_table, test_split

OUT = os.path.join(HERE, "..", "results", "phase3")
HYPER_N, GP_N, SEED, T_MAX = 1000, 4000, 0, 300


def report(t_true, mu, sd=None):
    """t_true, mu in log(1 + laps)."""
    r = dict(rmse_log=float(np.sqrt(np.mean((mu - t_true)**2))),
             median_abs_err_laps=float(np.median(np.abs(np.expm1(mu) - np.expm1(t_true)))),
             median_ratio_err=float(np.median(np.exp(np.abs(mu - t_true)))))
    if sd is not None:
        lo, hi = mu - 1.645 * sd, mu + 1.645 * sd
        r.update(coverage_90=float(np.mean((t_true >= lo) & (t_true <= hi))),
                 median_interval_laps=[float(np.median(np.expm1(lo))), float(np.median(np.expm1(hi)))],
                 median_interval_width_factor=float(np.median(np.exp(hi - lo))))
    return r


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
    t = np.log1p(T_h)
    tr_s = tr[y[tr] == "success"]; te_s = te[y[te] == "success"]; te_d = te[y[te] == "deadlock"]
    sub, _ = train_test_split(tr_s, train_size=GP_N, stratify=meta["N"][tr_s], random_state=SEED)
    hyp = sub[:HYPER_N]
    k = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(0.1, (1e-4, 10))
    t0 = time.time()
    g = GaussianProcessRegressor(k, normalize_y=True, random_state=SEED).fit(Z[hyp], t[hyp])
    sec_h = time.time() - t0
    t0 = time.time()
    g = GaussianProcessRegressor(g.kernel_, normalize_y=True, optimizer=None).fit(Z[sub], t[sub])
    sec_f = time.time() - t0
    mu, sd = g.predict(Z[te_s], return_std=True)
    lin = LinearRegression().fit(Z[tr_s], t[tr_s])
    out = dict(n_train_success=int(len(tr_s)), n_gp=GP_N, n_hyper=HYPER_N, n_test_success=int(len(te_s)),
               seconds_hyper=sec_h, seconds_fit=sec_f, kernel=str(g.kernel_),
               median_baseline=report(t[te_s], np.full(len(te_s), np.median(t[tr_s]))),
               linear=report(t[te_s], lin.predict(Z[te_s])),
               gp=report(t[te_s], mu, sd))
    mu_d, sd_d = g.predict(Z[te_d], return_std=True)
    p_over = 1 - norm.cdf((np.log1p(T_MAX) - mu_d) / sd_d)
    p_over_s = 1 - norm.cdf((np.log1p(T_MAX) - mu) / sd)
    out["censoring_check"] = dict(
        n_test_deadlock=int(len(te_d)),
        deadlock_median_P_T_over_Tmax=float(np.median(p_over)),
        deadlock_fraction_P_over_half=float(np.mean(p_over > 0.5)),
        success_median_P_T_over_Tmax=float(np.median(p_over_s)))
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open(os.path.join(OUT, "time_metrics.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]; o = np.random.default_rng(0).choice(len(te_s), 600, replace=False)
    ax.errorbar(np.expm1(t[te_s][o]) + 1, np.expm1(mu[o]) + 1,
                yerr=[np.expm1(mu[o]) - np.expm1(mu[o] - 1.645 * sd[o]), np.expm1(mu[o] + 1.645 * sd[o]) - np.expm1(mu[o])],
                fmt="o", ms=2, lw=0.4, alpha=0.5, color="tab:blue")
    ax.plot([1, 400], [1, 400], "k:", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true herding time + 1 (laps)"); ax.set_ylabel("GP prediction + 1 (laps), 90 % band")
    ax.set_title(f"Herding time, test runs (600 shown); coverage {out['gp']['coverage_90']:.2f}")
    ax = axes[1]
    ax.hist(p_over_s, bins=20, range=(0, 1), alpha=0.6, density=True, label="successful runs")
    ax.hist(p_over, bins=20, range=(0, 1), alpha=0.6, density=True, label="deadlock runs (T > 300)")
    ax.set_xlabel("GP: P(herding time > 300 laps)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    ax.set_title("Censoring check")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "time.png"), dpi=130)
