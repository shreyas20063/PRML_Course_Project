"""Censored herding time: kernel Tobit and kernel mixture-cure models.

Deadlock runs were stopped at T_max = 300 laps: for them we only know T_h > 300 ("right-censored").
All models share  f(x) = b + w . z(x),  z = random Fourier features (RFF) of the RBF kernel with the Phase 3 GP
length scales, w ~ N(0, s^2 I): f is approximately a GP; we fit the MAP. y = log(1 + T_h).

  success_only : y = f + sigma eps on settled runs only (drops the deadlocks; what Phase 3 did).
  tobit        : same, plus the censored likelihood P(y > c) = 1 - Phi((c - f)/sigma) for deadlocks.
  cure         : mixture cure model (Farewell 1982): a run EVER settles with probability pi(x) = sigmoid(g(x)),
                 g = b' + v . z(x) (kernel logistic regression); if it settles, y ~ N(f, sigma^2).
                 settled run: pi * N(y; f, sigma);  deadlock: (1 - pi) + pi * (1 - Phi((c - f)/sigma)).
Ground truth for the hidden part: 300 test deadlock runs re-simulated for 3,000 laps (10 x T_max).

usage: python models/censored_time.py  ->  results/censored/metrics.json, results/censored/censored.png
"""
import os, sys, json, re, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr, expit
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim")); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table, test_split
from labels import load, label
from herding import run_batch

OUT = os.path.join(HERE, "..", "results", "censored")
T_MAX, LONG, D, SEED, N_RESIM = 300, 3000, 1000, 0, 300
C = np.log1p(T_MAX)
PRIOR_MULTS = (1.0, 3.0, 10.0, 30.0)


def gp_time_kernel():
    """Amplitude and ARD length scales of the Phase 3 GP regression kernel."""
    k = json.load(open(os.path.join(HERE, "..", "results", "phase3", "time_metrics.json")))["kernel"]
    amp = float(re.match(r"\s*([0-9.eE+-]+)\*\*2", k).group(1))
    ls = np.array([float(v) for v in re.search(r"length_scale=\[([^\]]*)\]", k).group(1).split(",")])
    return amp, ls


class RFF:
    def __init__(self, ls, D=D, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(size=(len(ls), D)) / ls[:, None]
        self.b = rng.uniform(0, 2 * np.pi, D)

    def __call__(self, X):
        return np.sqrt(2.0 / self.W.shape[1]) * np.cos(X @ self.W + self.b)


class KernelTobit:
    """mode = 'tobit' or 'success_only'."""
    def __init__(self, ls, amp, mode="tobit"):
        self.z, self.s2, self.mode = RFF(ls), amp**2, mode

    def fit(self, X, y, cens, mult=1.0):
        Z = self.z(X); Dn = Z.shape[1]
        if self.mode == "success_only":
            Z, y, cens = Z[~cens], y[~cens], cens[~cens]
        obs = ~cens; s2 = self.s2 * mult

        def nll(th):
            w, b0, ls = th[:Dn], th[Dn], th[Dn + 1]
            sig = np.exp(ls); f = Z @ w + b0; r = (y - f) / sig
            L = 0.5 * np.sum(r[obs]**2) + obs.sum() * ls
            g_f = np.zeros_like(f); g_ls = obs.sum() - np.sum(r[obs]**2); g_f[obs] = -r[obs] / sig
            if cens.any():
                u = (C - f[cens]) / sig; lp = log_ndtr(-u); L -= lp.sum()
                h = np.exp(norm.logpdf(u) - lp)
                g_f[cens] = -h / sig; g_ls += np.sum(-h * u)
            L += 0.5 * w @ w / s2
            return L, np.r_[Z.T @ g_f + w / s2, g_f.sum(), g_ls]

        th0 = np.r_[np.zeros(Dn), np.mean(y[obs]), np.log(np.std(y[obs]) + 1e-3)]
        r = minimize(nll, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
        self.w, self.b0, self.sigma = r.x[:Dn], r.x[Dn], float(np.exp(r.x[Dn + 1]))
        self.opt = dict(success=bool(r.success), nit=int(r.nit))
        return self

    def f(self, X):
        return self.z(X) @ self.w + self.b0

    def settle_prob(self, X):
        return np.ones(len(X))

    def predict(self, X):
        return self.f(X)

    def survival(self, X, t):
        """P(T_h > t) for each row (t scalar or array of the same length)."""
        return ndtr((self.f(X) - np.log1p(t)) / self.sigma)


class KernelCure(KernelTobit):
    def __init__(self, ls, amp):
        super().__init__(ls, amp, "cure")

    def fit(self, X, y, cens, mult=1.0):
        Z = self.z(X); Dn = Z.shape[1]; obs = ~cens; s2 = self.s2 * mult

        def nll(th):
            w, b0, ls = th[:Dn], th[Dn], th[Dn + 1]
            v, c0 = th[Dn + 2:2 * Dn + 2], th[2 * Dn + 2]
            sig = np.exp(ls); f = Z @ w + b0; eta = Z @ v + c0; pi = expit(eta)
            logpi = -np.logaddexp(0, -eta); log1mpi = -np.logaddexp(0, eta)
            g_f = np.zeros_like(f); g_eta = np.zeros_like(f)
            r = (y[obs] - f[obs]) / sig
            L = np.sum(-logpi[obs] + 0.5 * r**2) + obs.sum() * ls
            g_f[obs] = -r / sig; g_eta[obs] = -(1 - pi[obs]); g_ls = obs.sum() - np.sum(r**2)
            if cens.any():
                u = (C - f[cens]) / sig; logS = log_ndtr(-u)
                A = log1mpi[cens]; Bv = logpi[cens] + logS; Lc = np.logaddexp(A, Bv)
                wA, wB = np.exp(A - Lc), np.exp(Bv - Lc)
                L -= Lc.sum()
                h = np.exp(norm.logpdf(u) - logS)
                g_eta[cens] = -(wA * (-pi[cens]) + wB * (1 - pi[cens]))
                g_f[cens] = -wB * h / sig
                g_ls += np.sum(-wB * h * u)
            L += 0.5 * (w @ w + v @ v) / s2
            return L, np.r_[Z.T @ g_f + w / s2, g_f.sum(), g_ls, Z.T @ g_eta + v / s2, g_eta.sum()]

        th0 = np.r_[np.zeros(Dn), np.mean(y[obs]), np.log(np.std(y[obs]) + 1e-3), np.zeros(Dn), 2.0]
        r = minimize(nll, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000))
        self.w, self.b0, self.sigma = r.x[:Dn], r.x[Dn], float(np.exp(r.x[Dn + 1]))
        self.v, self.c0 = r.x[Dn + 2:2 * Dn + 2], r.x[2 * Dn + 2]
        self.opt = dict(success=bool(r.success), nit=int(r.nit))
        return self

    def settle_prob(self, X):
        return expit(self.z(X) @ self.v + self.c0)

    def survival(self, X, t):
        pi = self.settle_prob(X)
        return (1 - pi) + pi * ndtr((self.f(X) - np.log1p(t)) / self.sigma)


def val_loglik(m, X, y, cens):
    """Per-run log-likelihood of the censored data under the model (for choosing the prior scale)."""
    S = np.clip(m.survival(X, T_MAX), 1e-300, 1)
    dens = m.settle_prob(X) * norm.pdf(y, m.f(X), m.sigma)
    return float(np.mean(np.where(cens, np.log(S), np.log(np.clip(dens, 1e-300, None)))))


def time_c_index(t_laps, event, model, X):
    """Antolini's time-dependent concordance: for pairs (i settled at t_i, j still running at t_i),
    concordant if the model gives j the larger survival probability at t_i."""
    num = den = 0.0
    ev = np.where(event)[0]
    for i in ev:
        later = t_laps > t_laps[i]
        if not later.any():
            continue
        Si = model.survival(X[i:i + 1], t_laps[i])[0]
        Sj = model.survival(X[later], t_laps[i])
        den += later.sum(); num += np.sum(Sj > Si) + 0.5 * np.sum(Sj == Si)
    return float(num / den)


def resimulate(d, idx, laps):
    out = run_batch(d["X0"][idx], d["N"][idx].astype(np.int64), d["pars"][idx], False, laps,
                    d["roots"][idx], d["psis"][idx], d["r_esc"][idx])
    Xf, last, trace, esc, Pf, dmin, steps, ok = out
    cls, T_h, stuck, j = label(dict(Xf=Xf, last=last, trace=trace, esc=esc, Pf=Pf, ok=ok, laps=laps, eps_r=d["eps_r"]), 0.01)
    return cls, T_h


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    Z = StandardScaler().fit(X[tr]).transform(X)
    keep_tr = tr[y[tr] != "loss"]; keep_te = te[y[te] != "loss"]
    cens = y == "deadlock"
    t = np.where(cens, C, np.log1p(np.nan_to_num(T_h, nan=T_MAX)))
    t_laps = np.where(cens, T_MAX, np.nan_to_num(T_h, nan=T_MAX))
    amp, ls = gp_time_kernel()
    out = dict(T_max=T_MAX, D=D, n_train_success=int((y[keep_tr] == "success").sum()), n_train_censored=int(cens[keep_tr].sum()))

    rng = np.random.default_rng(SEED)
    va = rng.choice(keep_tr, len(keep_tr) // 5, replace=False); fi = np.setdiff1d(keep_tr, va)
    models = {}
    for name, make in (("success_only", lambda: KernelTobit(ls, amp, "success_only")),
                       ("tobit", lambda: KernelTobit(ls, amp, "tobit")), ("cure", lambda: KernelCure(ls, amp))):
        best = None
        for mult in (PRIOR_MULTS if name != "success_only" else (1.0,)):
            m = make().fit(Z[fi], t[fi], cens[fi], mult)
            ll = val_loglik(m, Z[va], t[va], cens[va])
            print(f"{name} prior x{mult}: validation log-lik {ll:.4f} (sigma {m.sigma:.3f}, {m.opt})", flush=True)
            if best is None or ll > best[1]:
                best = (mult, ll)
        t0 = time.time(); models[name] = make().fit(Z[keep_tr], t[keep_tr], cens[keep_tr], best[0])
        out[name] = dict(prior_mult=best[0], val_loglik=best[1], sigma=models[name].sigma, seconds=time.time() - t0,
                         prior_mult_at_grid_edge=bool(best[0] in (PRIOR_MULTS[0], PRIOR_MULTS[-1])))

    # ---- standard test set (settled + censored)
    st = te[y[te] == "success"]; dl = te[y[te] == "deadlock"]
    for name, m in models.items():
        mu = m.f(Z[st]); s = m.sigma
        out[name].update(
            test_loglik=val_loglik(m, Z[keep_te], t[keep_te], cens[keep_te]),
            success_median_abs_err_laps=float(np.median(np.abs(np.expm1(mu) - T_h[st]))),
            success_coverage_90=float(np.mean(np.abs(t[st] - mu) <= 1.645 * s)),
            deadlock_mean_P_over_Tmax=float(np.mean(m.survival(Z[dl], T_MAX))),
            c_index_time=time_c_index(t_laps[keep_te], ~cens[keep_te], m, Z[keep_te]))
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)

    # ---- 10 x longer re-simulation of test deadlock runs (cached)
    d = load("arxiv")
    cache = os.path.join(OUT, "resim.npz")
    pick = np.random.default_rng(1).choice(dl, N_RESIM, replace=False)
    if os.path.exists(cache) and np.array_equal(np.load(cache)["idx"], pick):
        cz = np.load(cache, allow_pickle=True); cls_long, T_long = cz["cls_long"], cz["T_long"]
    else:
        cls_long, T_long = resimulate(d, pick, LONG)
    nl = cls_long != "loss"; settled = cls_long == "success"
    res = dict(n=N_RESIM, laps=LONG, settled=int(settled.sum()), still_deadlock=int((cls_long == "deadlock").sum()),
               lost=int((cls_long == "loss").sum()))
    ev = settled[nl].astype(float)
    for name, m in models.items():
        p_by_long = 1 - m.survival(Z[pick][nl], LONG)           # P(settles within 3,000 laps)
        f = m.f(Z[pick][settled]); yl = np.log1p(T_long[settled])
        res[name] = dict(
            brier_settle_by_3000=float(np.mean((p_by_long - ev)**2)),
            auc_settle_by_3000=float(roc_auc_score(ev, p_by_long)),
            mean_pred_P_settle_by_3000=float(p_by_long.mean()),
            settled_median_ratio_err=float(np.median(np.exp(np.abs(f - yl)))),
            settled_coverage_90=float(np.mean(np.abs(yl - f) <= 1.645 * m.sigma)))
    res["observed_frac_settle_by_3000_among_not_lost"] = float(ev.mean())
    out["resim_3000"] = res
    print(json.dumps(res, indent=1), flush=True)
    np.savez_compressed(cache, idx=pick, cls_long=cls_long, T_long=T_long)
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"success_only": "tab:red", "tobit": "tab:blue", "cure": "tab:green"}
    lab = {"success_only": "successes only", "tobit": "Tobit (censored)", "cure": "cure model (censored + never-settle)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for name, m in models.items():
        p = 1 - m.survival(Z[pick][nl], LONG); o = np.argsort(p); bins = np.array_split(o, 6)
        ax.plot([p[b].mean() for b in bins], [ev[b].mean() for b in bins], "o-", color=col[name],
                label=f"{lab[name]} (Brier {res[name]['brier_settle_by_3000']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P(settles within 3,000 laps)"); ax.set_ylabel("observed fraction (10x re-simulation)")
    ax.set_title(f"300 test deadlock runs re-simulated 10x longer: {settled.sum()} settle, "
                 f"{(cls_long == 'deadlock').sum()} never", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for name, m in models.items():
        ax.scatter(T_long[settled] + 1, np.expm1(m.f(Z[pick][settled])) + 1, s=9, alpha=0.6, color=col[name], label=lab[name])
    ax.plot([1, 5000], [1, 5000], "k:", lw=1); ax.axvline(T_MAX + 1, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=7)
    ax.set_xlabel("true herding time + 1 (laps), runs that settle in the re-run"); ax.set_ylabel("predicted median + 1")
    ax.set_title("Herding time beyond T_max (the deadlocks that do settle)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "censored.png"), dpi=130)
