# Novelty experiment (active learning) - RE-RUNNING

The first partial run was removed: it used the GP kernels before the over-confidence fix (commit "Fix GP
over-confidence"), so its numbers were not reliable. `models/active.py` is re-running with the fixed kernels;
checked results will appear here.
