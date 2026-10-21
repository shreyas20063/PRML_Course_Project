"""Labels from the raw runs (rules: docs/phase0/PHASE0_REPORT.md, section A2).

  loss     : some evader beyond r_esc at a lap end (simulation stopped), OR some evader outside the
             pursuer circle (r_i > R_p) at T_max, OR the integration failed.
  success  : not loss, and from some lap t_s on, at every lap end, every evader is within eps_r
             (relative, radius) and EPS_PSI (angle) of the same equilibrium (r*, psi*).
  deadlock : neither ("not settled by T_max").
  T_h      : t_s in laps (success only).
  stuck    : deadlock runs whose r_max moved less than 2 % over the last 60 laps (else "slow").

eps_r is one of the recorded tolerances EPS_R = 0.005, 0.01, 0.02, 0.05 (default 0.01).
usage: python data/labels.py [eps_r]   ->  writes data/dataset_arxiv.csv and data/dataset_circle.csv
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(law):
    files = sorted(glob.glob(os.path.join(HERE, "raw", f"{law}_*.npz")))
    ds = [dict(np.load(f, allow_pickle=True)) for f in files]
    out = {}
    for k in ds[0]:
        v = [d[k] for d in ds]
        out[k] = np.concatenate(v) if np.ndim(v[0]) > 0 else v[0]
    out["chunk"] = np.concatenate([np.full(len(d["pars"]), i) for i, d in enumerate(ds)])
    return out


def label(d, eps_r=0.01):
    L = int(d["laps"])
    m = int(np.argmin(np.abs(d["eps_r"] - eps_r)))
    r = np.hypot(d["Xf"][..., 0], d["Xf"][..., 1])
    outside = np.nanmax(r / d["Pf"][:, None], 1) > 1
    loss = (d["esc"] > 0) | outside | ~d["ok"]
    last = d["last"][:, :, m].astype(float)
    last[last < 0] = np.inf                      # missing root
    j = np.argmin(last, 1)
    t_s = last[np.arange(len(j)), j]
    success = ~loss & (t_s < L)
    cls = np.where(loss, "loss", np.where(success, "success", "deadlock"))
    rm = d["trace"][:, -12:, 1].astype(float)    # last 60 laps (trace every 5 laps)
    stuck = (cls == "deadlock") & ((np.nanmax(rm, 1) - np.nanmin(rm, 1)) < 0.02 * np.nanmean(rm, 1))
    T_h = np.where(success, t_s, np.nan)
    return cls, T_h, stuck, j


def table(law, eps_r=0.01):
    d = load(law)
    cls, T_h, stuck, j = label(d, eps_r)
    cols = {"chunk": d["chunk"], "N": d["N"]}
    if law == "arxiv":
        for k in ("q", "beta", "kappa", "a", "th0", "shape", "set"):
            cols[k] = d[k]
        cols["stratum"] = np.where(d["a"] > 1, 1, np.where(d["a"] > 6 / np.e**2, 2, 3))
    else:
        for k in ("gamma", "r0", "psi0"):
            cols[k] = d[k]
    cols["n_roots"] = np.sum(np.isfinite(d["roots"]), 1)
    cols["r_star"] = d["roots"][np.arange(len(j)), j]
    for i in range(6):
        cols[f"x{i}"] = d["X0"][:, i, 0]; cols[f"y{i}"] = d["X0"][:, i, 1]
    cols.update(label=cls, T_h_laps=T_h, stuck=stuck, n_lost=np.nansum(
        np.hypot(d["Xf"][..., 0], d["Xf"][..., 1]) > d["Pf"][:, None], 1),
        escape_lap=d["esc"], min_dist=d["dmin"], steps=d["steps"])
    return cols


def write_csv(cols, path):
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            f.write(",".join(f"{cols[k][i]:.6g}" if isinstance(cols[k][i], (float, np.floating)) else str(cols[k][i])
                             for k in keys) + "\n")


if __name__ == "__main__":
    eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    for law in ("arxiv", "circle"):
        cols = table(law, eps)
        write_csv(cols, os.path.join(HERE, f"dataset_{law}.csv"))
        lab = cols["label"]
        print(law, len(lab), {c: int((lab == c).sum()) for c in ("success", "deadlock", "loss")})
