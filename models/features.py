"""Permutation-invariant features of the START of a run (BRIEF.md, "Features"). Works for any N.

All lengths are divided by kappa (the start radius of the tracked evader), so the evader cloud is
compared at the same scale. Angles are measured in the rotating frame, where the pursuer starts at angle 0.
Evader 0 is the tracked evader; the other N - 1 evaders are treated as a set (order does not matter).
"""
import os, sys
import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data"))
from labels import load, label


def _gap(th):
    """Largest angular gap between the angles th (radians), in [0, 2 pi]."""
    if len(th) < 2:
        return 2 * np.pi
    s = np.sort(np.mod(th, 2 * np.pi))
    return np.max(np.diff(np.r_[s, s[0] + 2 * np.pi]))


def run_features(X0, q, beta, kap, r_star, psi_star):
    """X0: (N, 2) start positions, row 0 = tracked evader. Returns a dict of features."""
    N = len(X0)
    P0 = np.array([q, 0.0])                         # pursuer start (R_p(0) = R since kappa = r_0(0))
    r = np.hypot(X0[:, 0], X0[:, 1]) / kap
    th = np.arctan2(X0[:, 1], X0[:, 0])
    th0 = th[0]
    oth = th[1:] - th0                               # untracked angles relative to the tracked evader
    xs = r_star * np.array([np.cos(psi_star), np.sin(psi_star)])
    dstar = np.hypot(*(X0 - xs).T) / kap             # distance of each evader to the equilibrium point
    dpur = np.hypot(*(X0 - P0).T) / kap              # distance of each evader to the pursuer
    a = 2 * q**2 * np.exp(-2 * kap)
    f = dict(
        # log_a > 0 <=> a > 1 (arXiv Theorem 4); log_a > log(6/e^2) <=> RnD design set
        N=N, log_q=np.log(q), log_beta=np.log(beta), kappa=kap, log_a=np.log(a),
        log_q_over_kappa=np.log(q / kap), log_rstar_over_kappa=np.log(r_star / kap),
        cos_th0=np.cos(th0), sin_th0=np.sin(th0),
        r_mean=r.mean(), r_std=r.std(), r_min=r.min(),
        circ_var=1 - np.hypot(np.cos(th).mean(), np.sin(th).mean()), max_gap=_gap(th),
        log_dpur_min=np.log(dpur.min()), log_dstar_max=np.log(dstar.max()), log_dstar_mean=np.log(dstar.mean()),
    )
    if N > 1:
        f.update(rel_cos_mean=np.cos(oth).mean(), rel_cos_min=np.cos(oth).min(),
                 oth_r_max=r[1:].max(), oth_r_mean=r[1:].mean())
        D = np.hypot(*(X0[:, None] - X0[None]).transpose(2, 0, 1)) / kap
        iu = np.triu_indices(N, 1)
        f.update(pair_mean=D[iu].mean(), pair_min=D[iu].min())
    else:                                            # no untracked evaders: neutral values
        f.update(rel_cos_mean=1.0, rel_cos_min=1.0, oth_r_max=0.0, oth_r_mean=0.0, pair_mean=0.0, pair_min=0.0)
    f["hull_area"] = ConvexHull(X0).volume / (np.pi * kap**2) if N >= 3 else 0.0
    return f


def arxiv_table(eps_r=0.01):
    """Returns (X, names, y, T_h, meta) for the arXiv-law dataset."""
    d = load("arxiv")
    y, T_h, stuck, j = label(d, eps_r)
    rows = []
    for b in range(len(y)):
        n = int(d["N"][b])
        rows.append(run_features(d["X0"][b, :n], d["q"][b], d["beta"][b], d["kappa"][b],
                                 d["roots"][b, 0], d["psis"][b, 0]))
    names = list(rows[0])
    X = np.array([[r[k] for k in names] for r in rows], float)
    meta = dict(N=d["N"].astype(int), a=d["a"], set=d["set"], chunk=d["chunk"])
    return X, names, y, T_h, meta


def circle_table(eps_r=0.01):
    """Single-evader circle-law runs: features (log gamma, r0, cos psi0, sin psi0)."""
    d = load("circle")
    y, T_h, stuck, j = label(d, eps_r)
    X = np.stack([np.log(d["gamma"]), d["r0"], np.cos(d["psi0"]), np.sin(d["psi0"])], 1)
    return X, ["log_gamma", "r0", "cos_psi0", "sin_psi0"], y, T_h, dict(gamma=d["gamma"])


def test_split(y, N=None, seed=0):
    """The fixed 20 % test set used by every phase (stratified by label, and by N if given)."""
    from sklearn.model_selection import train_test_split
    strat = y.astype(str) if N is None else np.char.add(y.astype(str), np.asarray(N).astype(str))
    return train_test_split(np.arange(len(y)), test_size=0.2, stratify=strat, random_state=seed)
