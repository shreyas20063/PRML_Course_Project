"""Set kernel for flocks of any size (kernel mean embedding), compared with hand-made summary features.

A run = (global parameters g, tracked evader e0, set S of untracked evaders). Kernel between runs A and B:
  K(A, B) = k_g(g_A, g_B) * [ k_e(e0_A, e0_B) + mean_{u in S_A, v in S_B} k_e(u, v) ]
  k_g: RBF on the standardised global features (log q, log beta, kappa, log a, log q/kappa, log r*/kappa)
  k_e: RBF on an evader's position in the rotating frame, divided by kappa: (x/kappa, y/kappa)
The mean over pairs is the inner product of the two sets' kernel mean embeddings, so K is a valid kernel for any
set sizes (an empty set embeds to zero). No ordering of evaders and no hand-made summary statistics are needed.

Compared on the same training sizes with an RBF-SVM on the 24 hand-made features (Phase 2):
  (1) random split (the fixed Phase 2/3 test set), (2) unseen N (train N <= 4, test N = 5, 6).
Settings (length scales, C) for both models chosen by 3-fold CV on a 3,000-run subset of the training data.

usage: python models/set_kernel.py  ->  results/set_kernel/metrics.json
"""
import os, sys, json, time, itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from features import arxiv_table, test_split
from labels import load

OUT = os.path.join(HERE, "..", "results", "set_kernel")
GLOBAL = ["log_q", "log_beta", "kappa", "log_a", "log_q_over_kappa", "log_rstar_over_kappa"]
N_TRAIN, N_TUNE, SEED = 8000, 3000, 0
CLASSES = ["success", "deadlock", "loss"]


def run_sets(d):
    """Tracked evader (n, 2) and untracked set (n, 5, 2) with mask, in units of kappa."""
    kap = d["kappa"][:, None]
    E0 = d["X0"][:, 0] / kap
    S = d["X0"][:, 1:6] / kap[:, :, None]
    M = np.isfinite(S[..., 0])
    return E0, np.nan_to_num(S), M


def rbf(A, B, ls):
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-np.maximum(d2, 0) / (2 * ls**2))


def set_gram(ia, ib, G, E0, S, M, lg, le):
    """K for runs ia x ib."""
    Kg = rbf(G[ia], G[ib], lg)
    Ke = rbf(E0[ia], E0[ib], le)
    Ms_a, Ms_b = M[ia].astype(float), M[ib].astype(float)
    na, nb = np.maximum(Ms_a.sum(1), 1), np.maximum(Ms_b.sum(1), 1)
    Kset = np.zeros_like(Kg)
    for i in range(5):
        for j in range(5):
            w = np.outer(Ms_a[:, i], Ms_b[:, j])
            if w.any():
                Kset += w * rbf(S[ia, i], S[ib, j], le)
    Kset /= np.outer(na, nb)
    return Kg * (Ke + Kset)


def gram_blocks(ia, ib, *args, block=3000):
    return np.vstack([set_gram(ia[s:s + block], ib, *args) for s in range(0, len(ia), block)])


def cv_score(K, y, C):
    sk = StratifiedKFold(3, shuffle=True, random_state=SEED); s = []
    for a, b in sk.split(K, y):
        m = SVC(C=C, kernel="precomputed").fit(K[np.ix_(a, a)], y[a])
        s.append(balanced_accuracy_score(y[b], m.predict(K[np.ix_(b, a)])))
    return float(np.mean(s))


def experiment(name, tr_idx, te_idx, y, Zh, G, E0, S, M, out):
    strat = y[tr_idx]
    tr, _ = train_test_split(tr_idx, train_size=min(N_TRAIN, len(tr_idx) - 3), stratify=strat, random_state=SEED)
    tu, _ = train_test_split(tr, train_size=N_TUNE, stratify=y[tr], random_state=SEED)
    res = {"n_train": int(len(tr)), "n_test": int(len(te_idx))}
    # --- set kernel: tune (lg, le, C)
    best = None; t0 = time.time()
    for lg, le in itertools.product((1.0, 2.0, 4.0, 8.0), (0.3, 0.6, 1.2, 2.4)):
        K = set_gram(tu, tu, G, E0, S, M, lg, le)
        for C in (1, 10, 100, 1000):
            s = cv_score(K, y[tu], C)
            if best is None or s > best[0]:
                best = (s, lg, le, C)
    _, lg, le, C = best
    Ktr = gram_blocks(tr, tr, G, E0, S, M, lg, le)
    m = SVC(C=C, kernel="precomputed").fit(Ktr, y[tr])
    p = m.predict(gram_blocks(te_idx, tr, G, E0, S, M, lg, le))
    res["set_kernel"] = dict(balanced_accuracy=float(balanced_accuracy_score(y[te_idx], p)),
                             confusion=confusion_matrix(y[te_idx], p, labels=CLASSES).tolist(),
                             params=dict(lg=lg, le=le, C=C, cv=best[0]), seconds=time.time() - t0)
    # --- hand features: RBF-SVM tuned the same way
    best = None; t0 = time.time()
    for g, C in itertools.product((0.001, 0.003, 0.01, 0.03, 0.1), (10, 100, 1000, 10000)):
        K = np.exp(-g * np.maximum((Zh[tu]**2).sum(1)[:, None] + (Zh[tu]**2).sum(1)[None] - 2 * Zh[tu] @ Zh[tu].T, 0))
        s = cv_score(K, y[tu], C)
        if best is None or s > best[0]:
            best = (s, g, C)
    _, g, C = best
    m = SVC(C=C, gamma=g).fit(Zh[tr], y[tr]); p = m.predict(Zh[te_idx])
    res["hand_features"] = dict(balanced_accuracy=float(balanced_accuracy_score(y[te_idx], p)),
                                confusion=confusion_matrix(y[te_idx], p, labels=CLASSES).tolist(),
                                params=dict(gamma=g, C=C, cv=best[0]), seconds=time.time() - t0)
    out[name] = res
    print(name, {k: (v["balanced_accuracy"], v["params"]) for k, v in res.items() if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    d = load("arxiv")
    tr, te = test_split(y, meta["N"], SEED)
    Zh = StandardScaler().fit(X[tr]).transform(X)
    gi = [names.index(f) for f in GLOBAL]
    G = StandardScaler().fit(X[tr][:, gi]).transform(X[:, gi])
    E0, S, M = run_sets(d)
    N = meta["N"]
    out = {"global_features": GLOBAL, "n_train_cap": N_TRAIN, "n_tune": N_TUNE}
    experiment("random_split", tr, te, y, Zh, G, E0, S, M, out)
    experiment("unseen_N", np.where(N <= 4)[0], np.where(N >= 5)[0], y, Zh, G, E0, S, M, out)
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
