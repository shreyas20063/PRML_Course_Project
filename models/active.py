"""Novelty experiment: active learning of the herding outcome map, with a RISK-AWARE choice of the next simulation.

Question: how many simulations are needed to learn the outcome map, if the GP chooses which runs to simulate?
Strategies for choosing the next batch of runs from the pool (pool = the 48,000 training runs; a run's label is
only "revealed" when the strategy picks it, which is the same as simulating it on demand):
  random  : random runs (what Phases 1-3 did)
  entropy : runs where the GP is most unsure about the class (largest entropy of P)
  risk    : runs where the best decision is most costly in expectation: min_j sum_i P_i COST[i, j]
            (Bayes risk of the decision; it focuses on success/loss confusion, which costs most)
Every strategy starts from the same 200 random runs (per seed). The GP kernels are fixed (the Phase 3 kernels,
tuned once on 1,000 random training runs) so only the choice of runs differs. Batches: +100 up to 1,000 runs,
then +250 up to 4,000. To keep batches diverse, each batch is the top runs among 5,000 random pool candidates.
Scores on the fixed test set (12,000 runs) after every batch: balanced accuracy, log-loss, mean Bayes cost.

usage: python models/active.py  ->  results/novelty/active.json, results/novelty/active.png
       python models/active.py --plot   (redraw the figure only)
"""
import os, sys, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import arxiv_table, test_split
from gp_classify import OvRGP, CLASSES, COST, SEED
from gp_map import saved_kernels

OUT = os.path.join(HERE, "..", "results", "novelty")
SIZES = list(range(200, 1001, 100)) + list(range(1250, 4001, 250))
CAND = 5000
SEEDS = (0, 1, 2)
STRATEGIES = ("random", "entropy", "risk")


def evaluate(P, y):
    Y = (y[:, None] == CLASSES[None]).astype(float)
    P = np.clip(P, 1e-15, 1); P = P / P.sum(1, keepdims=True)
    idx = {c: i for i, c in enumerate(CLASSES)}
    dec = np.argmin(P @ COST, 1)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, CLASSES[P.argmax(1)])),
                log_loss=float(-np.mean(np.log((P * Y).sum(1)))),
                bayes_cost=float(COST[[idx[v] for v in y], dec].mean()))


def score(P, how):
    if how == "entropy":
        return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    return np.min(P @ COST, 1)                       # risk: expected cost of the best decision


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    lab = {"random": "random runs", "entropy": "most unsure (entropy)", "risk": "risk-aware (Bayes cost)"}
    for ax, metric, title in zip(axes, ("bayes_cost", "log_loss", "balanced_accuracy"),
                                 ("mean Bayes cost (lower is better)", "log-loss (lower is better)",
                                  "balanced accuracy (higher is better)")):
        for how in STRATEGIES:
            A = np.array([[r[metric] for r in res[how][str(s)]] for s in SEEDS])
            ns = [r["n"] for r in res[how][str(SEEDS[0])]]
            ax.plot(ns, A.mean(0), "o-", ms=3, label=lab[how])
            ax.fill_between(ns, A.min(0), A.max(0), alpha=0.2)
        ax.set_xscale("log"); ax.set_xticks([200, 500, 1000, 2000, 4000], ["200", "500", "1000", "2000", "4000"]); ax.minorticks_off(); ax.set_xlabel("number of simulations used for training"); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Choosing which simulations to run (3 repeats; band = min to max)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "active.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:] == ["--plot"]:            # redraw the figure from the saved results
        plot(json.load(open(os.path.join(OUT, "active.json")))["results"]); sys.exit()
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
    kernels = saved_kernels()
    res = {s: {str(seed): [] for seed in SEEDS} for s in STRATEGIES}
    for seed in SEEDS:
        rng0 = np.random.default_rng(100 + seed)
        start = rng0.choice(tr, SIZES[0], replace=False)
        for how in STRATEGIES:
            rng = np.random.default_rng(200 + seed)
            lab = list(start); pool = np.setdiff1d(tr, start)
            t0 = time.time()
            for n in SIZES:
                if len(lab) < n:
                    need = n - len(lab)
                    if how == "random":
                        pick = rng.choice(pool, need, replace=False)
                    else:
                        cand = rng.choice(pool, min(CAND, len(pool)), replace=False)
                        pick = cand[np.argsort(-score(gp.predict_proba(Z[cand]), how))[:need]]
                    lab += list(pick); pool = np.setdiff1d(pool, pick)
                L = np.array(lab)
                if len(np.unique(y[L])) < 3:                 # a class is missing: cannot fit one-vs-rest
                    continue
                gp = OvRGP().fit(Z[L], y[L], kernels=kernels)
                r = evaluate(gp.predict_proba(Z[te]), y[te]); r["n"] = n
                r["frac_loss_picked"] = float(np.mean(y[L] == "loss"))
                res[how][str(seed)].append(r)
            print(f"seed {seed} {how:7s}: n={n} bal {r['balanced_accuracy']:.4f} logloss {r['log_loss']:.4f} "
                  f"cost {r['bayes_cost']:.4f} ({time.time() - t0:.0f} s)", flush=True)
            json.dump(dict(sizes=SIZES, seeds=list(SEEDS), cost=COST.tolist(), results=res),
                      open(os.path.join(OUT, "active.json"), "w"), indent=1)

    # summary: runs needed by each strategy to match random sampling at 4,000 runs
    summ = {}
    for metric, better in (("log_loss", "low"), ("bayes_cost", "low"), ("balanced_accuracy", "high")):
        ref = np.mean([res["random"][str(s)][-1][metric] for s in SEEDS])
        summ[metric] = {"random_at_4000": float(ref)}
        for how in STRATEGIES:
            curve = np.mean([[r[metric] for r in res[how][str(s)]] for s in SEEDS], 0)
            ns = [r["n"] for r in res[how][str(SEEDS[0])]]
            ok = [n for n, v in zip(ns, curve) if (v <= ref if better == "low" else v >= ref)]
            summ[metric][how] = dict(at_4000=float(curve[-1]), runs_to_match_random_4000=(ok[0] if ok else None))
    print(json.dumps(summ, indent=1))
    json.dump(dict(sizes=SIZES, seeds=list(SEEDS), cost=COST.tolist(), results=res, summary=summ),
              open(os.path.join(OUT, "active.json"), "w"), indent=1)
    plot(res)
