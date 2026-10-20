import numpy as np, numba, time
from bench import batch, rstar
numba.set_num_threads(4)
rng = np.random.default_rng(2)
Xs, ps, ss = [], [], []
N = 3
while len(Xs) < 300:
    k1 = 10**rng.uniform(np.log10(0.3), np.log10(3.0)); R = rng.uniform(0.5, 4.0)
    kap = rng.uniform(0, 3.0); w = 10**rng.uniform(np.log10(0.5), np.log10(16)); k = 10**rng.uniform(-1, np.log10(5))
    if 2*k1**2*R**2*np.exp(-2*k1*kap) <= 6/np.e**2: continue
    rho = rng.uniform(0.05, 1.5)*max(kap, 0.3)
    r = rho*np.sqrt(rng.uniform(0, 1, N)); r[0] = rho; th = rng.uniform(0, 2*np.pi, N)
    C = R**2*np.exp(-2*k1*kap); tau = ((1-1/np.sqrt(3))*np.sqrt(C))**3/k
    spl = int(min(max(400, np.ceil(24*(2*np.pi/w)/tau)), 6400))
    Xs.append(np.stack([r*np.cos(th), r*np.sin(th)], 1)); ps.append((R, k, w, k1, kap)); ss.append(spl)
Xs, ps, ss = np.array(Xs), np.array(ps), np.array(ss)
t0 = time.time(); rl, Pf, rf = batch(Xs, ps, ss, 300); print("time", time.time()-t0)
rs = np.array([rstar(*p) for p in ps])
err = np.abs(rl - rs[:, None])/rs[:, None]
first = np.array([np.argmax(e < 0.01) if (e < 0.01).any() else -1 for e in err])
settled = np.array([(e[f:] < 0.01).all() if f >= 0 else False for e, f in zip(err, first)])
print("settled (1% of r*, stays) by 300 laps:", settled.sum(), "/300; never:", (~settled).sum())
print("lap of settling quantiles 50/90/99/max:", np.percentile(first[settled], [50, 90, 99, 100]))
beta = ps[:, 1]*ps[:, 3]**3/ps[:, 2]
print("beta of never-settled:", np.round(np.sort(beta[~settled]), 1))
print("tail range of r_max (last 50 laps) for never-settled:", np.round((rl[~settled, -50:].max(1)-rl[~settled, -50:].min(1)), 3))
print("median beta all:", np.median(beta))
