"""Stability check of the GP classifier: tune the kernels once (signal variance capped at AMP_MAX, argv[1]),
then fit on 6 random 4,000-run training sets and on the Phase 3 subset; print per-class and 3-class test scores.
usage: python models/gp_stability.py 100  ->  results/phase3/stability.txt"""
import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from joblib import Parallel, delayed
from features import arxiv_table, test_split
from gp_classify import CLASSES, HYPER_N, GP_N
from active import evaluate
X, names, y, T, meta = arxiv_table()
tr, te = test_split(y, meta['N'], 0)
Z = StandardScaler().fit(X[tr]).transform(X)
strat = np.char.add(y.astype(str), meta['N'].astype(str))
sub, _ = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=0)
hyp, _ = train_test_split(sub, train_size=HYPER_N, stratify=strat[sub], random_state=0)
CMAX = float(sys.argv[1])
def tune(c):
    k = ConstantKernel(1.0, (1e-2, CMAX)) * RBF(np.ones(X.shape[1]), (1e-2, 1e4))
    return GaussianProcessClassifier(k, random_state=0).fit(Z[hyp], (y[hyp] == c).astype(int)).kernel_
t0 = time.time(); ks = Parallel(n_jobs=3)(delayed(tune)(c) for c in CLASSES)
print(f'CMAX {CMAX}: tuned in {time.time()-t0:.0f} s; amplitudes', [round(k.k1.constant_value, 1) for k in ks], flush=True)
def fitc(i, L):
    return GaussianProcessClassifier(ks[i], optimizer=None).fit(Z[L], (y[L] == CLASSES[i]).astype(int)).predict_proba(Z[te])[:, 1]
for s in list(range(6)) + ['phase3']:
    L = sub if s == 'phase3' else np.random.default_rng(300 + s).choice(tr, 4000, replace=False)
    P = np.stack(Parallel(n_jobs=3)(delayed(fitc)(i, L) for i in range(3)), 1)
    acc = [np.mean((P[:, i] > .5) == (y[te] == c)) for i, c in enumerate(CLASSES)]
    P = P / P.sum(1, keepdims=True); e = evaluate(P, y[te])
    print(f'set {s}: binary acc {np.round(acc, 4)} | bal {e["balanced_accuracy"]:.4f} logloss {e["log_loss"]:.4f} cost {e["bayes_cost"]:.4f}', flush=True)
