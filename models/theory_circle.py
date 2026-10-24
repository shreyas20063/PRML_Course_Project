"""How conservative is the theory? Circle law (CDC 2024), one evader.

Dimensionless form (R = 1, w = 1, k = gamma). Rotating frame: pursuer fixed at (1, 0); evader w = (u, v):
  u' = gamma (u - 1)/d^3 + v,  v' = gamma v / d^3 - u,  d = |w - (1, 0)|.
1. Paper's estimate (CDC 2024, Sec. III-C, Problem 6): a quadratic Lyapunov function V = (w - w*)' A (w - w*);
   the largest ellipse {V <= c} on which dV/dt < 0 (except at w*) is a SUFFICIENT region of attraction. We search the
   ellipse shape (axis ratio and angle, det A = 1) starting from the Lyapunov-equation solution (as the paper does),
   and for each shape take the largest c allowed by a dense grid check of dV/dt. (The paper used fmincon; this is
   the same problem solved with Nelder-Mead.)
2. Paper's estimate for ANY pursuer start (Sec. III-D, Fig. 4): the intersection of the ellipse rotated about the
   origin by every angle = the disk of radius min |w| over the ellipse boundary (if the origin is inside the ellipse).
3. Truth: simulate a 90 x 90 grid of starts inside the unit disk (CDC Assumption 1); "converges" = not lost in
   300 laps (slow runs still converge; we also report success within 300 laps).
Coverage = area(estimate inside the disk and truly converging) / area(truly converging starts).

usage: python models/theory_circle.py  ->  results/theory/circle.json, results/theory/circle.png
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim")); sys.path.insert(0, os.path.join(HERE, "..", "data"))
from herding import run_batch, equilibria
from labels import label

OUT = os.path.join(HERE, "..", "results", "theory")
GAMMAS = [0.01, 0.03, 0.1, 0.2, 0.3]
GRID = 90


def field(u, v, g):
    du = u - 1.0; d3 = (du * du + v * v)**1.5
    return g * du / d3 + v, g * v / d3 - u


def _c_for(A, pts, f):
    """Largest c such that dV/dt < 0 at every checked point with V <= c (points: offsets from w*)."""
    du, dv = pts; fu, fv = f
    V = A[0, 0] * du**2 + 2 * A[0, 1] * du * dv + A[1, 1] * dv**2
    Vd = 2 * ((A[0, 0] * du + A[0, 1] * dv) * fu + (A[0, 1] * du + A[1, 1] * dv) * fv)
    bad = ~(Vd < 0)
    return V[bad].min() if bad.any() else V.max()


def _shape(p):
    ratio, ang = np.exp(p[0]), p[1]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return R @ np.diag([ratio, 1 / ratio]) @ R.T                       # det A = 1


def ellipse_estimate(g, ws):
    """Largest Lyapunov ellipse around ws (det A = 1, so area = pi c). Start from the Lyapunov-equation solution
    A J + J' A = -I (as the paper does), then optimise the shape (log axis ratio, angle) with Nelder-Mead.
    dV/dt is checked on a fine grid near w* and a coarse grid further out (points closer than 1e-4 are skipped)."""
    from scipy.linalg import solve_continuous_lyapunov
    from scipy.optimize import minimize
    grids = []
    for L, n in ((0.05, 250), (0.4, 350), (1.6, 350)):
        uu, vv = np.meshgrid(np.linspace(-L, L, n), np.linspace(-L, L, n))
        keep = np.hypot(uu, vv) > 1e-4
        du, dv = uu[keep], vv[keep]
        grids.append(((du, dv), field(ws[0] + du, ws[1] + dv, g), L))
    def c_of(A):
        c = min(_c_for(A, pts, f) for pts, f, L in grids)
        # the level set must stay inside the largest checked box
        lam = np.linalg.eigvalsh(A)
        return min(c, lam.min() * grids[-1][2]**2)
    h = 1e-6
    J = np.column_stack([(np.array(field(ws[0] + h * e[0], ws[1] + h * e[1], g)) -
                          np.array(field(ws[0] - h * e[0], ws[1] - h * e[1], g))) / (2 * h) for e in np.eye(2)])
    A0 = solve_continuous_lyapunov(J.T, -np.eye(2)); A0 = A0 / np.sqrt(np.linalg.det(A0))
    lam, vec = np.linalg.eigh(A0)
    p0 = np.array([0.5 * np.log(lam[1] / lam[0]), np.arctan2(vec[1, 1], vec[0, 1])])
    starts = [p0] + [np.array([np.log(r), a]) for r in (2.0, 6.0) for a in np.linspace(0, np.pi, 3, endpoint=False)]
    best = (0.0, None)
    for s0 in starts:
        r = minimize(lambda p: -c_of(_shape(p)), s0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10, maxiter=400))
        if -r.fun > best[0]:
            best = (-r.fun, _shape(r.x))
    c, A = best
    return A, c, np.pi * c, J


def truth(g):
    xs = np.linspace(-1, 1, GRID); U, Vv = np.meshgrid(xs, xs)
    inside = (U**2 + Vv**2 < 0.98) & (np.hypot(U - 1, Vv) > 0.05)
    P = np.stack([U[inside], Vv[inside]], 1)
    B = len(P)
    X0 = np.full((B, 6, 2), np.nan); X0[:, 0] = P
    pars = np.tile([1.0, g, 1.0, 0.0, 0.0], (B, 1))
    r, p = equilibria(1.0, g, 1.0, 0.0, 0.0)
    roots = np.tile(r, (B, 1)); psis = np.tile(p, (B, 1))
    Xf, last, trace, esc, Pf, dmin, steps, ok = run_batch(X0, np.ones(B, np.int64), pars, False, 300, roots, psis, np.full(B, 10.0))
    cls, *_ = label(dict(Xf=Xf, last=last, trace=trace, esc=esc, Pf=Pf, ok=ok, laps=300,
                         eps_r=np.array([0.005, 0.01, 0.02, 0.05])), 0.01)
    return U, Vv, inside, P, cls, (r[0], p[0])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    out = {}; panels = []
    for g in GAMMAS:
        U, Vv, inside, P, cls, (rs, ps) = truth(g)
        ws = np.array([rs * np.cos(ps), rs * np.sin(ps)])
        A, c, area, J = ellipse_estimate(g, ws)
        dw = P - ws
        in_ell = np.einsum("ni,ij,nj->n", dw, A, dw) <= c
        # rotated-ellipse intersection = disk of radius rho about the origin
        th = np.linspace(0, 2 * np.pi, 2000)
        ev, evec = np.linalg.eigh(A)
        bnd = ws[:, None] + evec @ (np.sqrt(c / ev)[:, None] * np.stack([np.cos(th), np.sin(th)]))
        origin_in = ws @ A @ ws <= c
        rho = float(np.hypot(*bnd).min()) if origin_in else 0.0
        in_disk = np.hypot(*P.T) <= rho
        conv = cls != "loss"; succ = cls == "success"
        cell = (2 / (GRID - 1))**2
        out[str(g)] = dict(
            r_star=rs, jacobian_eigs=[str(e) for e in np.linalg.eigvals(J)], ellipse_c=float(c), area_disk_checked=float(len(P) * cell), area_true_converging=float(conv.sum() * cell),
            area_true_success_300=float(succ.sum() * cell),
            ellipse_area_inside_disk=float(in_ell.sum() * cell), ellipse_false_positive_points=int((in_ell & ~conv).sum()),
            coverage_ellipse=float((in_ell & conv).sum() / max(conv.sum(), 1)),
            any_start_disk_radius=rho, coverage_any_start_disk=float((in_disk & conv).sum() / max(conv.sum(), 1)),
            frac_disk_converging=float(conv.mean()))
        print(g, out[str(g)], flush=True)
        panels.append((g, U, Vv, inside, P, cls, bnd, rho, ws))
    json.dump(out, open(os.path.join(OUT, "circle.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6))
    col = {"success": "#4c9f70", "deadlock": "#e3b23c", "loss": "#c44536"}
    for ax, (g, U, Vv, inside, P, cls, bnd, rho, ws) in zip(axes, panels):
        for k in col:
            m = cls == k; ax.scatter(P[m, 0], P[m, 1], s=2, color=col[k], label=k)
        ax.plot(*bnd, "b-", lw=1.5, label="paper's ellipse")
        if rho > 0:
            t = np.linspace(0, 2 * np.pi, 200); ax.plot(rho * np.cos(t), rho * np.sin(t), "k--", lw=1, label="any-start disk")
        ax.plot(1, 0, "k^", ms=7); ax.plot(*ws, "w*", ms=8, mec="k")
        ax.set_aspect("equal"); ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        o = out[str(g)]
        ax.set_title(f"gamma = {g}: ellipse covers {o['coverage_ellipse']:.0%}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=6, loc="lower left", markerscale=3)
    fig.suptitle("Circle law, rotating frame (pursuer = triangle): true outcome of every start vs the Lyapunov estimate", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "circle.png"), dpi=130)
