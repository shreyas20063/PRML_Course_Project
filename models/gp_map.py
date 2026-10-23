"""Phase 3c: one picture of the learned map vs the truth.

A fixed 3-evader flock (tracked evader at radius kappa = 1, two untracked evaders at fixed places), and a 40 x 40 grid
over q = k1 R and beta = k k1^3 / w. For every grid point:
  truth  = a new simulation with the same rules as the dataset (sim/herding.py, data/labels.py);
  model  = the Phase 3 GP classifier (refitted on the same 4,000 training runs with the kernels saved in
           results/phase3/gp_metrics.json).
The arXiv condition a = 2 q^2 exp(-2 kappa) > 1 is the vertical line q = e / sqrt(2).

usage: python models/gp_map.py  ->  results/phase3/map.png, results/phase3/map.npz
"""
import os, sys, json, re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "sim")); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table, test_split, run_features
from gp_classify import OvRGP, CLASSES, GP_N, SEED
from herding import run_batch, equilibria
from labels import label

OUT = os.path.join(HERE, "..", "results", "phase3")
LAPS = 300
FLOCK = np.array([[np.cos(2.0), np.sin(2.0)],                    # tracked evader, radius 1, angle 2 rad
                  [0.6 * np.cos(-1.0), 0.6 * np.sin(-1.0)],
                  [0.8 * np.cos(0.5), 0.8 * np.sin(0.5)]])


def saved_kernels():
    m = json.load(open(os.path.join(OUT, "gp_metrics.json")))
    ks = []
    for c, s in zip(CLASSES, m["gp_kernels"]):
        const = float(re.match(r"\s*([0-9.eE+-]+)\*\*2", s).group(1))**2
        ls = np.array([m["gp_length_scales"][c][f] for f in m["features"]])
        ks.append(ConstantKernel(const, "fixed") * RBF(ls, "fixed"))
    return ks


if __name__ == "__main__":
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr])
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    sub, _ = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=SEED)
    gp = OvRGP().fit(sc.transform(X[sub]), y[sub], kernels=saved_kernels())

    qs = np.geomspace(1.1, 12, 40); bs = np.geomspace(1e-3, 50, 40)
    Q, Bt = np.meshgrid(qs, bs)
    pars = np.stack([Q.ravel(), Bt.ravel(), np.ones(Q.size), np.ones(Q.size), np.ones(Q.size)], 1)
    B = len(pars)
    X0 = np.full((B, 6, 2), np.nan); X0[:, :3] = FLOCK
    roots = np.full((B, 3), np.nan); psis = np.full((B, 3), np.nan)
    for b in range(B):
        roots[b], psis[b] = equilibria(*pars[b])
    Xf, last, trace, esc, Pf, dmin, steps, ok = run_batch(X0, np.full(B, 3), pars, False, LAPS, roots, psis,
                                                          10 * np.maximum(pars[:, 0], 1.0))
    d = dict(Xf=Xf, last=last, trace=trace, esc=esc, Pf=Pf, ok=ok, laps=LAPS, eps_r=np.array([0.005, 0.01, 0.02, 0.05]))
    truth, _, _, _ = label(d, 0.01)
    F = np.array([[v for v in run_features(FLOCK, p[0], p[1], 1.0, roots[i, 0], psis[i, 0]).values()]
                  for i, p in enumerate(pars)])
    P = gp.predict_proba(sc.transform(F))
    pred = CLASSES[P.argmax(1)]
    agree = float((pred == truth).mean())
    print(f"map: GP agrees with the simulation on {agree:.1%} of {B} grid points", flush=True)
    np.savez_compressed(os.path.join(OUT, "map.npz"), q=qs, beta=bs, truth=truth.reshape(Q.shape), P=P.reshape(*Q.shape, 3))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    code = {c: i for i, c in enumerate(CLASSES)}
    cmap = ListedColormap(["#4c9f70", "#e3b23c", "#c44536"])
    qline = np.e / np.sqrt(2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, Zc, title in ((axes[0], np.vectorize(code.get)(truth.reshape(Q.shape)), "simulation (truth)"),
                          (axes[1], np.vectorize(code.get)(pred.reshape(Q.shape)), f"GP most probable class ({agree:.0%} agree)")):
        ax.pcolormesh(Q, Bt, Zc, cmap=cmap, vmin=-0.5, vmax=2.5, shading="nearest")
        ax.set_title(title, fontsize=10)
    im = axes[2].pcolormesh(Q, Bt, P[:, 2].reshape(Q.shape), cmap="Reds", vmin=0, vmax=1, shading="nearest")
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1).reshape(Q.shape)
    axes[2].contour(Q, Bt, ent, levels=[0.6], colors="k", linewidths=0.8)
    axes[2].set_title("GP P(loss); black line: high uncertainty", fontsize=10)
    fig.colorbar(im, ax=axes[2])
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(qline, color="k", ls="--", lw=1)
        ax.set_xticks([1.5, 2, 3, 5, 8, 12], ["1.5", "2", "3", "5", "8", "12"]); ax.minorticks_off()
        ax.set_xlabel("q = k1 R  (dashed: a = 1, arXiv condition holds to the right)"); ax.set_ylabel("beta = k k1^3 / w")
    for c, col in zip(CLASSES, cmap.colors):
        axes[0].plot([], [], "s", color=col, label=c)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("One 3-evader flock (kappa = 1): where herding succeeds, stalls or loses an evader", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "map.png"), dpi=130)
