"""Fit the final models once and save them for the demo app (app/models.joblib).

arXiv law: feature scaler, GP classifier (Phase 3 kernels, 4,000 training runs), kernel cure model for herding time.
circle law: GP classifier on (log gamma, r0, cos psi0, sin psi0), kernels tuned on 1,000 runs, fitted on 4,000.
usage: python app/export_models.py
"""
import os, sys, json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "models"))
from features import arxiv_table, circle_table, test_split
from gp_classify import OvRGP, GP_N, SEED, CLASSES, COST
from gp_map import saved_kernels
from censored_time import KernelCure, gp_time_kernel, C

if __name__ == "__main__":
    X, names, y, T_h, meta = arxiv_table()
    tr, te = test_split(y, meta["N"], SEED)
    sc = StandardScaler().fit(X[tr])
    strat = np.char.add(y.astype(str), meta["N"].astype(str))
    sub, _ = train_test_split(tr, train_size=GP_N, stratify=strat[tr], random_state=SEED)
    gp = OvRGP().fit(sc.transform(X[sub]), y[sub], kernels=saved_kernels())
    amp, ls = gp_time_kernel()
    keep = tr[y[tr] != "loss"]; cens = y == "deadlock"
    t = np.where(cens, C, np.log1p(np.nan_to_num(T_h, nan=300)))
    mult = json.load(open(os.path.join(HERE, "..", "results", "censored", "metrics.json")))["cure"]["prior_mult"]
    cure = KernelCure(ls, amp).fit(sc.transform(X[keep]), t[keep], cens[keep], mult)

    Xc, namesc, yc, Tc, metac = circle_table()
    trc, tec = test_split(yc, None, SEED)
    scc = StandardScaler().fit(Xc[trc])
    subc, _ = train_test_split(trc, train_size=4000, stratify=yc[trc], random_state=SEED)
    hypc = subc[:1000]
    gpc = OvRGP().fit(scc.transform(Xc[hypc]), yc[hypc])
    gpc = OvRGP().fit(scc.transform(Xc[subc]), yc[subc], kernels=gpc.kernels)
    Pc = gpc.predict_proba(scc.transform(Xc[tec]))
    acc_c = float(np.mean(CLASSES[Pc.argmax(1)] == yc[tec]))
    print("circle GP test accuracy", acc_c)
    joblib.dump(dict(names=names, scaler=sc, gp=gp, cure=cure, names_circle=namesc, scaler_circle=scc, gp_circle=gpc,
                     classes=CLASSES, cost=COST, circle_test_accuracy=acc_c),
                os.path.join(HERE, "models.joblib"), compress=3)
    print("saved", os.path.join(HERE, "models.joblib"))
