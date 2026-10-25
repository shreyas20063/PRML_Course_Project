"""Discover a short, readable rule for the outcome: a data-driven conjecture to compare with the theory's a > 1.

Two yes/no questions: "will an evader be lost?" and "will herding succeed?".
For every single feature, pair and triple of the 24 start features, fit a logistic regression on a 20,000-run part of
the training set and score it (balanced accuracy) on a separate 10,000-run validation part. The best rule of each size
is chosen on validation only, then reported ONCE on the fixed test set, next to the theory rule and the GP classifier.
The chosen rule is written back in the original feature units as  w1 x1 + w2 x2 (+ w3 x3) > c.

usage: python models/rule.py  ->  results/rule/metrics.json
"""
import os, sys, json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import arxiv_table, test_split

OUT = os.path.join(HERE, "..", "results", "rule")


def fit_rule(X, y, idx):
    mu, sd = X[:, idx].mean(0), X[:, idx].std(0) + 1e-12
    m = LogisticRegression(class_weight="balanced", max_iter=2000).fit((X[:, idx] - mu) / sd, y)
    w = m.coef_[0] / sd; c = -(m.intercept_[0] - np.sum(m.coef_[0] * mu / sd))
    return w, c


def predict(X, idx, w, c):
    return X[:, idx] @ w > c


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], 0)
    fit_i, val_i = train_test_split(tr, train_size=20000, test_size=10000, random_state=0, stratify=y[tr])
    gp_probs = np.load(os.path.join(HERE, "..", "results", "phase3", "test_probs.npz"), allow_pickle=True)
    assert np.array_equal(gp_probs["test_index"], te)
    out = {}
    for question, target, theory_pred, gp_col in (("loss", "loss", lambda s: meta["a"][s] <= 1, 2),
                                                 ("success", "success", lambda s: meta["a"][s] > 1, 0)):
        yb = y == target
        res = {}
        for size in (1, 2, 3):
            best = None
            for idx in itertools.combinations(range(len(names)), size):
                idx = list(idx)
                w, c = fit_rule(X[fit_i], yb[fit_i], idx)
                s = balanced_accuracy_score(yb[val_i], predict(X[val_i], idx, w, c))
                if best is None or s > best[0]:
                    best = (s, idx, w, c)
            s, idx, w, c = best
            test = balanced_accuracy_score(yb[te], predict(X[te], idx, w, c))
            terms = " + ".join(f"{wi:+.3f} * {names[i]}" for wi, i in zip(w, idx))
            res[f"{size}_features"] = dict(features=[names[i] for i in idx], weights=w.tolist(), threshold=float(c),
                                           rule=f"{target} if  {terms}  > {c:.3f}", val_balanced_accuracy=float(s),
                                           test_balanced_accuracy=float(test))
            print(question, size, res[f"{size}_features"]["rule"], f"val {s:.3f} test {test:.3f}", flush=True)
        res["theory_rule"] = dict(rule=("loss if a <= 1" if target == "loss" else "success if a > 1"),
                                  test_balanced_accuracy=float(balanced_accuracy_score(yb[te], theory_pred(te))))
        gp_pred = gp_probs["gp"][:, gp_col] > 0.5
        res["gp_classifier"] = dict(test_balanced_accuracy=float(balanced_accuracy_score(yb[te], gp_pred)))
        out[question] = res
    json.dump(out, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    for q in out:
        print(q, {k: round(v["test_balanced_accuracy"], 3) for k, v in out[q].items()})
