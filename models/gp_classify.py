"""Phase 3a: GP classifier with calibrated probabilities, compared with logistic regression and SVM + Platt.
Also: uncertainty vs errors, and the Bayes-risk decision rule.

GP: one binary GP classifier per class (one-vs-rest, Laplace approximation), kernel = constant x RBF with one
length scale per feature (ARD). Exact GP cost grows as n^3, so:
  1. kernel settings are tuned (marginal likelihood) on HYPER_N training runs;
  2. the GP is then fitted on GP_N training runs with those settings fixed.
Probabilities of the three binary GPs are normalised to sum to 1 (as sklearn does for one-vs-rest).
A length scale at its upper bound means the GP found that feature irrelevant for that class (ARD).

usage: python models/gp_classify.py  ->  results/phase3/gp_metrics.json, reliability.png, selective.png,
                                          results/phase3/test_probs.npz
"""
import os, sys, json, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import arxiv_table, test_split

OUT = os.path.join(HERE, "..", "results", "phase3")
CLASSES = np.array(["success", "deadlock", "loss"])
HYPER_N, GP_N, SVM_N, SEED = 1000, 4000, 20000, 0
# Bayes-risk cost matrix COST[true, decided], order success / deadlock / loss  [proposal: the student chooses]
COST = np.array([[0, 1, 1],     # true success: calling it deadlock or loss wastes an attempt
                 [3, 0, 1],     # true deadlock called success: time wasted waiting
                 [10, 3, 0]])   # true loss called success: an evader is lost (worst)


class OvRGP:
    """Three binary GP classifiers with their own tuned kernels (fixed when refitting on more data)."""
    def fit(self, X, y, kernels=None):
        from joblib import Parallel, delayed
        def one(i, c):
            if kernels is None:
                k = ConstantKernel(1.0, (1e-2, 1e5)) * RBF(np.ones(X.shape[1]), (1e-2, 1e4))
                m = GaussianProcessClassifier(k, random_state=SEED)
            else:
                m = GaussianProcessClassifier(kernels[i], optimizer=None, random_state=SEED)
            return m.fit(X, (y == c).astype(int))
        self.models = Parallel(n_jobs=3)(delayed(one)(i, c) for i, c in enumerate(CLASSES))
        self.kernels = [m.kernel_ for m in self.models]
        return self

    def predict_proba(self, X):
        P = np.stack([m.predict_proba(X)[:, 1] for m in self.models], 1)
        return P / P.sum(1, keepdims=True)


def prob_scores(y, P):
    """P columns in CLASSES order."""
    P = np.clip(P, 0, 1); P = P / P.sum(1, keepdims=True)   # remove rounding errors like -1e-13
    Y = (y[:, None] == CLASSES[None]).astype(float)
    pred = CLASSES[P.argmax(1)]
    conf = P.max(1); right = pred == y
    bins = np.linspace(0, 1, 16); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(right[m].mean() - conf[m].mean())
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                log_loss=float(-np.mean(np.log(np.clip((P * Y).sum(1), 1e-15, 1)))), brier=float(((P - Y)**2).sum(1).mean()),
                ece=float(ece))


def bayes_decide(P):
    return CLASSES[np.argmin(P @ COST, 1)]


def mean_cost(y, dec):
    idx = {c: i for i, c in enumerate(CLASSES)}
    return float(COST[[idx[v] for v in y], [idx[v] for v in dec]].mean())


def reorder(P, classes):
    """Columns of P (in the model's class order) -> CLASSES order."""
    return P[:, [list(classes).index(c) for c in CLASSES]]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr])
    Z = sc.transform(X)
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    sub, _ = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=SEED)
    hyp, _ = train_test_split(sub, train_size=HYPER_N, stratify=strat[sub], random_state=SEED)
    out = dict(n_hyper=HYPER_N, n_gp=GP_N, n_svm=SVM_N, n_test=int(len(te)), cost_matrix=COST.tolist(), features=names)

    t0 = time.time(); gp = OvRGP().fit(Z[hyp], y[hyp]); out["seconds_hyper"] = time.time() - t0
    print(f"GP kernels tuned on {HYPER_N}: {out['seconds_hyper']:.0f} s", flush=True)
    t0 = time.time(); gp = OvRGP().fit(Z[sub], y[sub], kernels=gp.kernels); out["seconds_fit"] = time.time() - t0
    print(f"GP fitted on {GP_N}: {out['seconds_fit']:.0f} s", flush=True)
    out["gp_kernels"] = [str(k) for k in gp.kernels]
    out["gp_length_scales"] = {c: dict(zip(names, map(float, k.k2.length_scale))) for c, k in zip(CLASSES, gp.kernels)}
    probs = {"gp": gp.predict_proba(Z[te])}

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
    probs["logreg"] = reorder(lr.predict_proba(X[te]), lr.classes_)
    for name, idx in (("svm_platt_20k", train_test_split(tr, train_size=SVM_N, stratify=strat[tr], random_state=SEED)[0]),
                      ("svm_platt_4k", sub)):
        t0 = time.time()
        m = CalibratedClassifierCV(make_pipeline(StandardScaler(), SVC(C=1000, gamma=0.003)), cv=3,
                                   method="sigmoid").fit(X[idx], y[idx])
        probs[name] = reorder(m.predict_proba(X[te]), m.classes_)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)

    yt = y[te]
    out["scores"] = {k: prob_scores(yt, P) for k, P in probs.items()}
    # Bayes risk vs plain "most probable class"
    out["bayes"] = {}
    for k, P in probs.items():
        am, bd = CLASSES[P.argmax(1)], bayes_decide(P)
        cm = confusion_matrix(yt, bd, labels=list(CLASSES))
        out["bayes"][k] = dict(mean_cost_argmax=mean_cost(yt, am), mean_cost_bayes=mean_cost(yt, bd),
                               loss_called_success_argmax=int(((yt == "loss") & (am == "success")).sum()),
                               loss_called_success_bayes=int(((yt == "loss") & (bd == "success")).sum()),
                               bayes_confusion=cm.tolist())
    # uncertainty vs errors: accuracy on the most confident fraction of runs (selective prediction)
    out["selective"] = {}
    for k, P in probs.items():
        conf = P.max(1); right = CLASSES[P.argmax(1)] == yt; o = np.argsort(-conf)
        ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
        out["selective"][k] = dict(
            accuracy_at_coverage={f"{c:.1f}": float(right[o[:int(c * len(o))]].mean()) for c in (1.0, 0.9, 0.8, 0.7, 0.5)},
            mean_entropy_right=float(ent[right].mean()), mean_entropy_wrong=float(ent[~right].mean()))
    for k in probs:
        print(k, {m: round(v, 4) for m, v in out["scores"][k].items()}, "| cost argmax %.3f bayes %.3f" %
              (out["bayes"][k]["mean_cost_argmax"], out["bayes"][k]["mean_cost_bayes"]), flush=True)
    json.dump(out, open(os.path.join(OUT, "gp_metrics.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(OUT, "test_probs.npz"), test_index=te, y=yt, N=meta["N"][te],
                        **{k: v for k, v in probs.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("gp", "GP classifier (4k)"), ("svm_platt_4k", "SVM + Platt (4k)"),
            ("svm_platt_20k", "SVM + Platt (20k)"), ("logreg", "logistic regression (48k)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for k, lab in show:
        P = probs[k]; conf = P.max(1); right = CLASSES[P.argmax(1)] == yt
        bins = np.linspace(0.3, 1, 15); mid, acc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 30:
                mid.append(conf[m].mean()); acc.append(right[m].mean())
        axes[0].plot(mid, acc, "o-", ms=3, label=f"{lab}, ECE {out['scores'][k]['ece']:.3f}")
        pl = P[:, 2]; isl = yt == "loss"; b2 = np.linspace(0, 1, 11); x2, f2 = [], []
        for lo, hi in zip(b2[:-1], b2[1:]):
            m = (pl >= lo) & (pl < hi)
            if m.sum() >= 30:
                x2.append(pl[m].mean()); f2.append(isl[m].mean())
        axes[1].plot(x2, f2, "o-", ms=3, label=lab)
    for ax, t, xl, yl in ((axes[0], "Top-class reliability", "predicted confidence", "fraction correct"),
                          (axes[1], "Reliability of P(loss)", "predicted P(loss)", "observed fraction lost")):
        ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title(t); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "reliability.png"), dpi=130)
    fig, ax = plt.subplots(figsize=(5, 4))
    for k, lab in show:
        P = probs[k]; right = CLASSES[P.argmax(1)] == yt; o = np.argsort(-P.max(1))
        cov = np.linspace(0.05, 1, 40)
        ax.plot(cov, [right[o[:max(1, int(c * len(o)))]].mean() for c in cov], label=lab)
    ax.set_xlabel("fraction of runs answered (most confident first)"); ax.set_ylabel("accuracy on those runs")
    ax.set_title("Uncertainty is useful: abstain on the least confident"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "selective.png"), dpi=130)
