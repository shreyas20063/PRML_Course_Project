"""Demo: probabilistic prediction of outcome and herding time for one pursuer and N evaders.

run:  streamlit run app/app.py      (models: python app/export_models.py first)
"""
import os, sys
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
for p in ("models", "sim", "data"):
    sys.path.insert(0, os.path.join(ROOT, p))
from features import run_features
from herding import run_batch, equilibria, rhs_numpy
from labels import label

COLORS = {"success": "#4c9f70", "deadlock": "#e3b23c", "loss": "#c44536"}
WORDS = {"success": "all evaders herded to the ring", "deadlock": "not settled within 300 laps",
         "loss": "at least one evader escapes"}
st.set_page_config(page_title="Herding outcome predictor", layout="wide")


@st.cache_resource
def models():
    return joblib.load(os.path.join(HERE, "models.joblib"))


M = models()
CLASSES = list(M["classes"])


# ------------------------------------------------------------------ inputs
st.sidebar.title("Set up the herd")
law = st.sidebar.radio("Pursuer law", ["arXiv law (radius follows the tracked evader)", "circle law (fixed radius)"])
circle = law.startswith("circle")
N = st.sidebar.slider("number of evaders N", 1, 6, 3)
R = st.sidebar.slider("R: pursuer circle radius", 0.5, 4.0, 2.0, 0.1)
k = st.sidebar.slider("k: repulsion strength", 0.1, 5.0, 1.0, 0.1)
w = st.sidebar.slider("w: pursuer angular speed", 0.5, 16.0, 2.0, 0.1)
k1 = 0.0 if circle else st.sidebar.slider("k1: radius feedback gain", 0.3, 3.0, 1.0, 0.05)
spread = st.sidebar.slider("flock size: farthest evader / R", 0.1, 0.95, 0.5, 0.05)
shape = st.sidebar.selectbox("flock shape", ["random disk", "ring", "cluster", "two groups"])
seed = st.sidebar.number_input("random seed (positions)", 0, 9999, 3)


def make_flock(N, kap, shape, seed):
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(0, 2 * np.pi)
    X = [[kap * np.cos(th0), kap * np.sin(th0)]]
    for i in range(N - 1):
        if shape == "random disk":
            r, th = kap * np.sqrt(rng.uniform(0, 0.998)), rng.uniform(0, 2 * np.pi)
        elif shape == "ring":
            r, th = kap * rng.uniform(0.9, 0.999), rng.uniform(0, 2 * np.pi)
        elif shape == "cluster":
            r, th = kap * rng.uniform(0.6, 0.999), th0 + rng.normal(0, 0.4)
        else:
            r, th = kap * rng.uniform(0.3, 0.999), th0 + np.pi * (i % 2) + rng.normal(0, 0.2)
        X.append([r * np.cos(th), r * np.sin(th)])
    return np.array(X)


kap = spread * R
X0 = make_flock(N, kap, shape, int(seed))                       # physical units, rotating frame (pursuer at (R, 0))
par = (R, k, w, k1, kap if not circle else 0.0)


# ------------------------------------------------------------------ prediction
def predict_many(X0, settings):
    """Class probabilities (n, 3) for one flock under many settings (R, k, w, k1, kap): one model call."""
    if circle:
        out = []
        for R, k, w, k1, kap in settings:
            g = k / (w * R**3)
            th = np.arctan2(X0[:, 1], X0[:, 0])
            out.append(np.stack([np.full(len(X0), np.log(g)), np.hypot(X0[:, 0], X0[:, 1]) / R, np.cos(th), np.sin(th)], 1))
        Pe = M["gp_circle"].predict_proba(M["scaler_circle"].transform(np.vstack(out))).reshape(len(settings), len(X0), 3)
        ps = np.prod(Pe[:, :, 0], 1); pl = 1 - np.prod(1 - Pe[:, :, 2], 1)
        P = np.stack([ps, np.clip(1 - ps - pl, 0, None), pl], 1)
        return P / P.sum(1, keepdims=True), None
    F = []
    for R, k, w, k1, kap in settings:
        q, beta, kt = k1 * R, k * k1**3 / w, k1 * kap
        r, p = equilibria(q, beta, 1.0, 1.0, kt)
        F.append(list(run_features(X0 * k1, q, beta, kt, r[0], p[0]).values()))
    Z = M["scaler"].transform(np.array(F))
    return M["gp"].predict_proba(Z), Z


def predict(X0, R, k, w, k1, kap):
    """Returns P (3,), Bayes decision, expected costs, herding-time info (or None)."""
    P, Z = predict_many(X0, [(R, k, w, k1, kap)])
    P = P[0]
    time_info = None
    if Z is not None:
        cure = M["cure"]
        mu = cure.f(Z)[0]; s = cure.sigma; pi = cure.settle_prob(Z)[0]
        time_info = dict(pi=pi, med=np.expm1(mu), lo=np.expm1(mu - 1.645 * s), hi=np.expm1(mu + 1.645 * s))
    cost = P @ M["cost"]
    return P, CLASSES[int(np.argmin(cost))], cost, time_info


@st.cache_data(show_spinner="computing the what-if map ...")
def whatif_map(X0_bytes, n, k, k1, kap, law_circle):
    X0 = np.frombuffer(X0_bytes).reshape(n, 2)
    Rs = np.geomspace(max(kap / 0.95 * 1.02, 0.5), 4.0, 22); ws = np.geomspace(0.5, 16, 22)
    settings = [(RR, k, ww, k1, kap) for ww in ws for RR in Rs]
    P, _ = predict_many(X0, settings)
    return Rs, ws, P.reshape(len(ws), len(Rs), 3)


P, decision, cost, tinfo = predict(X0, *par)

st.title("Where does herding succeed, and how fast?")
st.caption("One pursuer circles the goal; each evader runs away from it with speed k/d^2. The models only look at the "
           "START (positions + parameters) and predict the outcome before anything is simulated.")
c1, c2 = st.columns([1.1, 1])

with c1:
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=R * np.cos(t), y=R * np.sin(t), mode="lines", line=dict(color="gray", dash="dot"),
                             name="pursuer circle at start"))
    fig.add_trace(go.Scatter(x=[R], y=[0], mode="markers", marker=dict(symbol="triangle-up", size=16, color="black"), name="pursuer"))
    fig.add_trace(go.Scatter(x=X0[:, 0], y=X0[:, 1], mode="markers", marker=dict(size=12, color="#1f77b4"), name="evaders"))
    if not circle:
        fig.add_trace(go.Scatter(x=[X0[0, 0]], y=[X0[0, 1]], mode="markers", marker=dict(size=18, color="rgba(0,0,0,0)",
                                 line=dict(color="red", width=2)), name="tracked evader (farthest)"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12, color="green"), name="goal"))
    lim = 1.15 * R
    fig.update_layout(height=430, xaxis=dict(range=[-lim, lim], scaleanchor="y"), yaxis=dict(range=[-lim, lim]),
                      margin=dict(l=10, r=10, t=30, b=10), title="Start (frame rotating with the pursuer)")
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Prediction (no simulation yet)")
    bar = go.Figure(go.Bar(x=[f"{c}<br>{100 * p:.0f}%" for c, p in zip(CLASSES, P)], y=P,
                           marker_color=[COLORS[c] for c in CLASSES]))
    bar.update_layout(height=230, yaxis=dict(range=[0, 1], title="probability"), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, width="stretch")
    st.markdown(f"**Bayes decision:** call it **{decision}** "
                f"(expected cost: success {cost[0]:.2f}, deadlock {cost[1]:.2f}, loss {cost[2]:.2f}; "
                "calling a real loss 'success' costs 10).")
    if tinfo:
        lap = 2 * np.pi / w
        st.markdown(f"**Herding time** (if it settles, probability {tinfo['pi']:.0%}): about **{tinfo['med']:.0f} laps** "
                    f"({tinfo['med'] * lap:.0f} time units), 90 % band {tinfo['lo']:.0f} to {tinfo['hi']:.0f} laps.")
    unc = -np.sum(P * np.log(np.clip(P, 1e-12, 1)))
    st.markdown(f"**Model uncertainty (entropy):** {unc:.2f} " + ("- confident" if unc < 0.3 else "- unsure: close to a boundary"))

tab1, tab2, tab3 = st.tabs(["Simulate and compare", "What-if map", "What we found"])

with tab1:
    laps_show = st.slider("laps to animate", 5, 80, 25)
    if st.button("Simulate", type="primary"):
        # true outcome: the dataset rules, 300 laps
        r_, p_ = equilibria(*par) if not circle else equilibria(R, k, w, 0.0, 0.0)
        X = np.full((1, 6, 2), np.nan); X[0, :N] = X0
        out = run_batch(X, np.array([N]), np.array([par], float), False, 300, r_[None], p_[None],
                        np.array([10 * max(R, kap)]))
        Xf, last, trace, esc, Pf, dmin, steps, ok = out
        cls, Th, _, _ = label(dict(Xf=Xf, last=last, trace=trace, esc=esc, Pf=Pf, ok=ok, laps=300,
                                   eps_r=np.array([0.005, 0.01, 0.02, 0.05])), 0.01)
        msg = f"Simulation result (300 laps): **{cls[0]}** - {WORDS[cls[0]]}"
        if cls[0] == "success":
            msg += f", herding time {Th[0]:.0f} laps"
        st.markdown(msg + f". The model said **{CLASSES[int(np.argmax(P))]}** ({P.max():.0%}).")
        # animation (fixed frame: the pursuer moves around the goal)
        T = laps_show * 2 * np.pi / w
        fr = np.linspace(0, T, laps_show * 12 + 1)
        sol = solve_ivp(rhs_numpy, [0, T], X0.ravel(), method="DOP853", rtol=1e-8, atol=1e-10, t_eval=fr, args=par)
        Y = sol.y.T.reshape(len(sol.t), -1, 2)
        ang = w * sol.t
        rot = lambda P2, a: np.stack([P2[..., 0] * np.cos(a) - P2[..., 1] * np.sin(a), P2[..., 0] * np.sin(a) + P2[..., 1] * np.cos(a)], -1)
        Yi = rot(Y, ang[:, None])
        rt = np.hypot(Y[:, 0, 0], Y[:, 0, 1]) if not circle else np.zeros(len(sol.t))
        Rp = R * np.exp(k1 * (rt - (kap if not circle else 0.0)))
        pur = np.stack([Rp * np.cos(ang), Rp * np.sin(ang)], 1)
        lim = 1.3 * max(R, np.nanmax(np.abs(Yi[:, :, :])) if np.isfinite(Yi).all() else R)
        lim = min(lim, 4 * R)
        frames = [go.Frame(data=[go.Scatter(x=Yi[i, :, 0], y=Yi[i, :, 1]), go.Scatter(x=[pur[i, 0]], y=[pur[i, 1]]),
                                 go.Scatter(x=Yi[max(0, i - 24):i + 1, :, 0].ravel(), y=Yi[max(0, i - 24):i + 1, :, 1].ravel())],
                           name=str(i)) for i in range(len(sol.t))]
        fig = go.Figure(data=[go.Scatter(x=Yi[0, :, 0], y=Yi[0, :, 1], mode="markers", marker=dict(size=11, color="#1f77b4"), name="evaders"),
                              go.Scatter(x=[pur[0, 0]], y=[pur[0, 1]], mode="markers", marker=dict(symbol="triangle-up", size=15, color="black"), name="pursuer"),
                              go.Scatter(x=[], y=[], mode="markers", marker=dict(size=3, color="lightblue"), name="recent path")],
                        frames=frames)
        fig.update_layout(height=520, xaxis=dict(range=[-lim, lim], scaleanchor="y"), yaxis=dict(range=[-lim, lim]),
                          updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate",
                                       args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)])])],
                          title=f"Real simulation, first {laps_show} laps (fixed frame; goal at the centre)")
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(symbol="x", size=12, color="green"), name="goal"))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Press **Simulate** to run the real dynamics and compare with the prediction above.")

with tab2:
    st.markdown("Keep this flock, change two pursuer settings. Colour = predicted probability; the star is your setting.")
    which = st.radio("show", ["P(success)", "P(loss)"], horizontal=True)
    Rs, ws, Pm = whatif_map(X0.tobytes(), N, k, k1, kap, circle)
    zi = 0 if which == "P(success)" else 2
    hm = go.Figure(go.Heatmap(x=Rs, y=ws, z=Pm[:, :, zi], zmin=0, zmax=1,
                              colorscale="Greens" if zi == 0 else "Reds", colorbar=dict(title=which)))
    hm.add_trace(go.Scatter(x=[R], y=[w], mode="markers", marker=dict(symbol="star", size=18, color="blue"), name="your setting"))
    if not circle:
        a_line = np.sqrt(np.exp(2 * k1 * kap) / (2 * k1**2))       # a = 2 k1^2 R^2 exp(-2 k1 kappa) = 1
        hm.add_vline(x=a_line, line=dict(color="black", dash="dash"), annotation_text="arXiv condition a = 1")
    hm.update_layout(height=480, xaxis=dict(type="log", title="R (pursuer radius)"), yaxis=dict(type="log", title="w (angular speed)"))
    st.plotly_chart(hm, width="stretch")

with tab3:
    figs = [("results/phase3/map.png", "The learned map agrees with 1,600 fresh simulations on 97 %; evaders are lost even where the paper's condition holds (right of the dashed line)."),
            ("results/novelty/active.png", "Choosing which simulations to run: the risk-aware choice reaches the same decision cost with 2.3x fewer runs."),
            ("results/phase3/reliability.png", "The GP's probabilities are honest: 80 % sure means right about 80 % of the time."),
            ("results/regimes/regimes.png", "Without labels, trajectory shapes fall into three arms: the three outcomes are natural regimes.")]
    for f, cap in figs:
        path = os.path.join(ROOT, f)
        if os.path.exists(path):
            st.image(path, caption=cap)
