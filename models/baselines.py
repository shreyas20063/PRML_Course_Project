"""Phase 2 baselines: outcome classification (success / deadlock / loss).

Models: majority class, theory rule (arXiv Theorem 4: success iff a > 1, else loss),
logistic regression, RBF-SVM. One fixed test set (20 %, stratified by label and N), never used for tuning.
Also: unseen N (train N <= 4, test N = 5, 6) and the circle-law baseline.

usage: python models/baselines.py   ->  results/phase2/metrics.json, results/phase2/confusion.png
"""
import os, sys, json, time
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import arxiv_table, circle_table

OUT = os.path.join(HERE, "..", "results", "phase2")
CLASSES = ["success", "deadlock", "loss"]
SVM_TRAIN = 20000          # SVM training cost grows ~ n^2; use a stratified subset of the training set
SEED = 0


def scores(y, p):
    cm = confusion_matrix(y, p, labels=CLASSES)
    return dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, labels=CLASSES, average="macro"),
                recall={c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(CLASSES)},
                confusion=cm.tolist())


def fit_lr(X, y):
    g = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                     {"logisticregression__C": [0.01, 0.1, 1, 10]},
                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="balanced_accuracy", n_jobs=-1)
    g.fit(X, y)
    return g.best_estimator_, g.best_params_


def fit_svm(X, y, rng):
    if len(y) > SVM_TRAIN:
        X, _, y, _ = train_test_split(X, y, train_size=SVM_TRAIN, stratify=y, random_state=SEED)
    Xs, _, ys, _ = train_test_split(X, y, train_size=min(6000, len(y) - 3), stratify=y, random_state=SEED)
    g = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                     {"svc__C": [1, 10, 100, 1000], "svc__gamma": [0.003, 0.01, 0.03, 0.1, 0.3]},
                     cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="balanced_accuracy", n_jobs=-1)
    g.fit(Xs, ys)                                   # tune on a 6,000 subset (cheap), then refit on SVM_TRAIN
    m = g.best_estimator_.fit(X, y)
    return m, g.best_params_


def evaluate(X, y, strat, rng, theory=None):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=strat, random_state=SEED)
    res = {"n_train": int(len(tr)), "n_test": int(len(te)),
           "test_class_counts": {c: int((y[te] == c).sum()) for c in CLASSES}}
    vals, cnt = np.unique(y[tr], return_counts=True)
    res["majority"] = scores(y[te], np.full(len(te), vals[np.argmax(cnt)]))
    if theory is not None:
        res["theory_rule"] = scores(y[te], theory[te])
    t0 = time.time(); lr, p = fit_lr(X[tr], y[tr])
    res["logreg"] = scores(y[te], lr.predict(X[te]))
    res["logreg"].update(params=p, log_loss=log_loss(y[te], lr.predict_proba(X[te]), labels=lr.classes_),
                         seconds=time.time() - t0)
    t0 = time.time(); sv, p = fit_svm(X[tr], y[tr], rng)
    res["svm"] = scores(y[te], sv.predict(X[te])); res["svm"].update(params=p, seconds=time.time() - t0)
    return res, (lr, sv, te)


def unseen_N(X, y, N, rng):
    tr = np.where(N <= 4)[0]; te = np.where(N >= 5)[0]
    res = {"n_train": int(len(tr)), "n_test": int(len(te))}
    lr, p = fit_lr(X[tr], y[tr]); res["logreg"] = scores(y[te], lr.predict(X[te]))
    sv, p = fit_svm(X[tr], y[tr], rng); res["svm"] = scores(y[te], sv.predict(X[te]))
    return res


def plot_confusions(panels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    for ax, (title, cm) in zip(axes, panels):
        cm = np.array(cm, float); frac = cm / cm.sum(1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{frac[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                        color="white" if frac[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X, names, y, T_h, meta = arxiv_table()
    theory = np.where(meta["a"] > 1, "success", "loss")
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    out = {"features": names}
    out["arxiv"], (lr, sv, te) = evaluate(X, y, strat, rng, theory)
    print("arxiv", {k: (v["balanced_accuracy"] if isinstance(v, dict) and "balanced_accuracy" in v else v)
                    for k, v in out["arxiv"].items()}, flush=True)
    out["arxiv_by_N_svm"] = {int(n): scores(y[te][meta["N"][te] == n], sv.predict(X[te][meta["N"][te] == n]))
                             for n in range(1, 7)}
    out["arxiv_unseen_N"] = unseen_N(X, y, meta["N"], rng)
    print("unseen N", {k: out["arxiv_unseen_N"][k]["balanced_accuracy"] for k in ("logreg", "svm")}, flush=True)
    coef = lr.named_steps["logisticregression"].coef_
    out["logreg_coefficients"] = {c: dict(zip(names, map(float, coef[i]))) for i, c in enumerate(lr.classes_)}
    Xc, namesc, yc, Tc, metac = circle_table()
    theory_c = np.where(metac["gamma"] < 0.385, "success", "loss")
    out["circle"], _ = evaluate(Xc, yc, yc, rng, theory_c)
    print("circle", {k: (v["balanced_accuracy"] if isinstance(v, dict) and "balanced_accuracy" in v else v)
                     for k, v in out["circle"].items()}, flush=True)
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
    plot_confusions([("theory rule (a > 1), arXiv", out["arxiv"]["theory_rule"]["confusion"]),
                     ("logistic regression, arXiv", out["arxiv"]["logreg"]["confusion"]),
                     ("RBF-SVM, arXiv", out["arxiv"]["svm"]["confusion"]),
                     ("RBF-SVM, train N<=4, test N=5,6", out["arxiv_unseen_N"]["svm"]["confusion"])],
                    os.path.join(OUT, "confusion.png"))
