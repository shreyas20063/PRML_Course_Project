"""Statistical certificate of a safe set, compared with the Lyapunov certificate of the theory.

Idea (a risk-controlling prediction set; classical binomial statistics):
  S_tau = { starts x : P_GP(loss | x) <= tau }   ("the model says this start is safe").
On CALIBRATION runs never used to fit the GP, count the starts inside S_tau and how many of them lose an evader.
The exact one-sided Clopper-Pearson bound gives, with confidence 1 - delta, an upper bound on the loss rate inside
S_tau. We take the largest tau (the largest set) whose bound is <= eps. Fixed-sequence testing over increasing tau
(stop at the first failure) keeps the overall confidence at 1 - delta (Bates et al. / Angelopoulos et al.
"learn then test" idea; here with a single ordered family).
Then on FRESH test data: is the loss rate inside S really <= eps, and how much of the true success region is covered?

Two set-ups:
  arXiv law : GP classifier of Phase 3 (fitted on 4,000 training runs); calibration = 12,000 other training runs;
              test = the fixed 12,000-run test set.
  circle law: GP classifier on 4,000 circle-law runs; calibration = the other 4,000 training runs; test = the
              90 x 90 start grids of results/theory (fresh simulations) - so the learned certificate can be put
              next to the paper's Lyapunov ellipse on exactly the same starts.

usage: python models/certify.py  ->  results/certificate/metrics.json, results/certificate/certificate.png
"""
import os, sys, json
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim")); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table, circle_table, test_split
from gp_classify import OvRGP, CLASSES, GP_N, SEED
from gp_map import saved_kernels

OUT = os.path.join(HERE, "..", "results", "certificate")
EPS, DELTA = 0.01, 0.05
TAUS = np.r_[np.geomspace(1e-6, 0.5, 300)]


def cp_upper(k, n, delta=DELTA):
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    return 1.0 if n == 0 else float(beta_dist.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0


def calibrate(p_loss, is_loss, eps=EPS):
    """Largest tau (fixed-sequence over increasing tau) whose CP bound on the loss rate in {p_loss <= tau} is <= eps."""
    best = None
    for tau in TAUS:
        m = p_loss <= tau
        n, k = int(m.sum()), int((is_loss & m).sum())
        if n == 0:
            continue
        if cp_upper(k, n) <= eps:
            best = (float(tau), n, k, cp_upper(k, n))
        else:
            break
    return best


def evaluate(p_loss, y, tau):
    m = p_loss <= tau
    succ = y == "success"; conv = y != "loss"
    return dict(n_in_set=int(m.sum()), frac_of_all_in_set=float(m.mean()),
                loss_rate_in_set=float(np.mean(y[m] == "loss")) if m.any() else None,
                coverage_of_true_success=float(np.mean(m[succ])), coverage_of_not_lost=float(np.mean(m[conv])))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = dict(eps=EPS, delta=DELTA)
    # ---------------- arXiv law
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr])
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    sub, rest = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=SEED)
    cal, _ = train_test_split(rest, train_size=12000, stratify=strat[rest], random_state=SEED)
    gp = OvRGP().fit(sc.transform(X[sub]), y[sub], kernels=saved_kernels())
    pc = gp.predict_proba(sc.transform(X[cal]))[:, 2]; pt = gp.predict_proba(sc.transform(X[te]))[:, 2]
    tau, n, k, ub = calibrate(pc, y[cal] == "loss")
    out["arxiv"] = dict(tau=tau, calibration=dict(n_in_set=n, losses=k, cp_upper=ub), test=evaluate(pt, y[te], tau))
    # the theory's rule as a "safe set" on the same test runs: a > 1
    th = meta["a"][te] > 1
    out["arxiv"]["theory_set_a_gt_1"] = dict(loss_rate_in_set=float(np.mean(y[te][th] == "loss")),
                                             coverage_of_true_success=float(np.mean(th[y[te] == "success"])))
    out["arxiv"]["test_by_N"] = {int(n_): evaluate(pt[meta["N"][te] == n_], y[te][meta["N"][te] == n_], tau) for n_ in range(1, 7)}
    print("arxiv", json.dumps(out["arxiv"], indent=1), flush=True)

    # ---------------- circle law vs the Lyapunov ellipse, on the theory grids
    Xc, namesc, yc, Tc, metac = circle_table()
    trc, tec = test_split(yc, None, SEED)
    scc = StandardScaler().fit(Xc[trc])
    subc, calc = train_test_split(trc, train_size=4000, stratify=yc[trc], random_state=SEED)
    gpc = OvRGP().fit(scc.transform(Xc[subc[:1000]]), yc[subc[:1000]])
    gpc = OvRGP().fit(scc.transform(Xc[subc]), yc[subc], kernels=gpc.kernels)
    pcc = gpc.predict_proba(scc.transform(Xc[calc]))[:, 2]
    tau_c, n, k, ub = calibrate(pcc, yc[calc] == "loss")
    out["circle"] = dict(tau=tau_c, calibration=dict(n_in_set=n, losses=k, cp_upper=ub), per_gamma={})
    from theory_circle import truth, ellipse_estimate, GAMMAS
    panels = []
    for g in GAMMAS:
        U, Vv, inside, P, cls, (rs, ps) = truth(g)
        F = np.stack([np.full(len(P), np.log(g)), np.hypot(*P.T), np.cos(np.arctan2(P[:, 1], P[:, 0])),
                      np.sin(np.arctan2(P[:, 1], P[:, 0]))], 1)
        pl = gpc.predict_proba(scc.transform(F))[:, 2]
        ws = np.array([rs * np.cos(ps), rs * np.sin(ps)])
        A, c, area, J = ellipse_estimate(g, ws)
        dw = P - ws; in_ell = np.einsum("ni,ij,nj->n", dw, A, dw) <= c
        conv = cls != "loss"; m = pl <= tau_c
        out["circle"]["per_gamma"][str(g)] = dict(
            learned_coverage_of_converging=float(np.mean(m[conv])),
            learned_loss_rate_in_set=float(np.mean(~conv[m])) if m.any() else None,
            learned_false_positives=int((m & ~conv).sum()), n_in_set=int(m.sum()),
            lyapunov_coverage_of_converging=float(np.mean(in_ell[conv])))
        print(g, out["circle"]["per_gamma"][str(g)], flush=True)
        panels.append((g, P, cls, m, A, c, ws))
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.7))
    th_ = np.linspace(0, 2 * np.pi, 400)
    for ax, (g, P, cls, m, A, c, ws) in zip(axes, panels):
        conv = cls != "loss"
        ax.scatter(P[~conv, 0], P[~conv, 1], s=2, color="#c44536", label="lost")
        ax.scatter(P[conv & ~m, 0], P[conv & ~m, 1], s=2, color="#b8d8c4", label="converges, not certified")
        ax.scatter(P[conv & m, 0], P[conv & m, 1], s=2, color="#2e7d4f", label="certified by data (loss <= 1 %, 95 %)")
        ev, evec = np.linalg.eigh(A)
        bnd = ws[:, None] + evec @ (np.sqrt(c / ev)[:, None] * np.stack([np.cos(th_), np.sin(th_)]))
        ax.plot(*bnd, "b-", lw=1.5, label="certified by Lyapunov (theory)")
        ax.plot(1, 0, "k^", ms=7); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        r = out["circle"]["per_gamma"][str(g)]
        ax.set_title(f"gamma = {g}\ndata {r['learned_coverage_of_converging']:.0%} vs theory {r['lyapunov_coverage_of_converging']:.1%}", fontsize=9)
    axes[0].legend(fontsize=6, loc="lower left", markerscale=3)
    fig.suptitle("How much of the true safe region is certified: statistical certificate (green) vs Lyapunov proof (blue)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "certificate.png"), dpi=130)
