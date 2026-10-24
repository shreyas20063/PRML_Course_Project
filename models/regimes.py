"""Unsupervised check (BRIEF: GMM + PCA): do the three regimes appear without labels?

Two views, each on 10,000 random arXiv-law runs:
  start   : the 24 start features (what the classifiers use);
  dynamics: how r_max evolves (log of r_max/kappa every 5 laps for 300 laps) plus the radial error to the
            equilibrium ring (log, every 5 laps) - the shape of the trajectory, no label used.
For each view: standardise, PCA (keep 95 % of the variance), Gaussian mixture with full covariances for K = 1..10,
K chosen by BIC; agreement with the labels by the adjusted Rand index (ARI, 0 = chance, 1 = identical) and
normalised mutual information (NMI); class mix of every cluster.

usage: python models/regimes.py  ->  results/regimes/metrics.json, results/regimes/regimes.png
"""
import os, sys, json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table
from labels import load

OUT = os.path.join(HERE, "..", "results", "regimes")
CLASSES = ["success", "deadlock", "loss"]


def analyse(F, y, name):
    Z = StandardScaler().fit_transform(F)
    pca = PCA(0.95, random_state=0).fit(Z); P = pca.transform(Z)
    fits = [GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(P) for k in range(1, 11)]
    bic = [m.bic(P) for m in fits]; m = fits[int(np.argmin(bic))]
    lab = m.predict(P)
    mix = {int(c): {cl: int(((lab == c) & (y == cl)).sum()) for cl in CLASSES} for c in np.unique(lab)}
    r = dict(n_components_pca=int(pca.n_components_), explained=pca.explained_variance_ratio_[:3].tolist(),
             bic=bic, K=int(m.n_components), ari=float(adjusted_rand_score(y, lab)),
             nmi=float(normalized_mutual_info_score(y, lab)), clusters=mix)
    # majority-vote purity
    r["purity"] = float(sum(max(v.values()) for v in mix.values()) / len(y))
    print(name, {k: r[k] for k in ("n_components_pca", "K", "ari", "nmi", "purity")}, flush=True)
    return r, P, lab


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    d = load("arxiv")
    idx = np.random.default_rng(0).choice(len(y), 10000, replace=False)
    ok = idx[d["esc"][idx] < 0]                                   # escaped runs stop early: trace incomplete
    tr = d["trace"][idx].astype(float)
    rm = np.log(np.nan_to_num(tr[:, :, 1], nan=np.nanmax(tr[:, :, 1])) / d["kappa"][idx, None])
    er = np.log(np.nan_to_num(tr[:, :, 0], nan=10.0) + 1e-6)
    # escaped runs: fill the missing part of the trace with their last value (they are far out)
    dyn = np.hstack([rm, er])
    out = {"n": int(len(idx)), "note": "escaped runs: trace after escape filled with the last/max value"}
    out["start"], Ps, ls = analyse(X[idx], y[idx], "start features")
    out["dynamics"], Pd, ld = analyse(dyn, y[idx], "trajectory shape")
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"success": "#4c9f70", "deadlock": "#e3b23c", "loss": "#c44536"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, P, title in ((axes[0], Ps, f"start features, PCA (GMM: K={out['start']['K']}, ARI {out['start']['ari']:.2f})"),
                         (axes[1], Pd, f"trajectory shape, PCA (GMM: K={out['dynamics']['K']}, ARI {out['dynamics']['ari']:.2f})")):
        for c in CLASSES:
            m = y[idx] == c; ax.scatter(P[m, 0], P[m, 1], s=2, alpha=0.4, color=col[c], label=c)
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_title(title, fontsize=9)
    axes[0].legend(markerscale=5, fontsize=8)
    mix = out["dynamics"]["clusters"]; ks = sorted(mix, key=int); bottom = np.zeros(len(ks))
    for c in CLASSES:
        v = np.array([mix[k][c] for k in ks]); axes[2].bar([str(k) for k in ks], v, bottom=bottom, color=col[c], label=c); bottom += v
    axes[2].set_xlabel("GMM cluster (trajectory shape)"); axes[2].set_ylabel("runs"); axes[2].legend(fontsize=8)
    axes[2].set_title("Class mix of each unsupervised cluster", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "regimes.png"), dpi=130)
