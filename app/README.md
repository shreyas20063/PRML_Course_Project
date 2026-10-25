# Live demo

    python -m pip install numpy scipy scikit-learn numba streamlit plotly joblib
    python app/export_models.py          # once (about 7 minutes): fits the models -> app/models.joblib (208 MB, not in git)
    streamlit run app/app.py             # opens in the browser

What it shows (it explains itself on screen):
1. **Set up the herd** (sidebar): law (arXiv / circle), N, R, k, w, k1, flock size and shape.
2. **Prediction before any simulation**: P(success / deadlock / loss) from the GP classifier, the Bayes-risk decision
   (losing an evader costs most), predicted herding time with a 90 % band and P(ever settles) (cure model), and the
   model's uncertainty.
3. **Simulate and compare**: runs the real dynamics (300 laps) and prints the true outcome next to the prediction;
   animated trajectories (Play button) in the fixed frame.
4. **What-if map**: same flock, P(success) or P(loss) over the pursuer radius R and speed w; the star is your setting,
   the dashed line is the arXiv condition a = 1.
5. **What we found**: the main result figures with one-line captions.

Checked headless with `streamlit.testing` (2026-10-24): no exceptions for the arXiv law (N = 3, 6), the circle law,
the map and Simulate; first load 13 s (model file), Simulate 2 s, map about 2 s (one batched GP call).
Demo tip: the default setting (seed 3) is a good opener: the most likely class is "success" (54 %) but the Bayes rule
says "loss", and the simulation shows a loss.

Certificate badge (added 2026-10-25): the app says whether the start lies inside the statistically certified safe set
(results/certificate): among such starts at most 1 % lose an evader, with 95 % confidence. For the circle law the
certificate was calibrated on single-evader runs, so for a flock it is a per-evader statement.
