"""Claim 5 synthetic sweep: does the weighting-neutral result depend on config?

The boundary-coincidence argument says that when Δ is a symmetric linear-Gaussian
function of features, the cost-weighted-optimal and error-optimal boundaries both
sit at Δ=0, so |Δ|-weighted CE cannot beat standard CE for a well-specified model
at scale — regardless of noise sigma, dimension d, regularization strength C, or
(above the small-sample regime) N. This sweep tests that: it varies one factor at
a time around the run_synthetic.py base config (N=20000, d=20, sigma=0.35, C=1)
and reports the weighted-minus-standard NEC difference (negative = weighting
helps), mean over 5 seeds with a t-based 95% CI.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import nec, sign_label, split_80_10_10

BASE = {"N": 20_000, "d": 20, "sigma": 0.35, "C": 1.0}
SWEEP = [
    {"N": 2_000},
    {"N": 20_000},
    {"sigma": 0.2},
    {"sigma": 0.7},
    {"d": 10},
    {"d": 50},
    {"C": 0.01},
    {"C": 100.0},
]
SEEDS = range(5)

rows = []
for override in SWEEP:
    cfg = {**BASE, **override}
    diffs = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        X = rng.standard_normal((cfg["N"], cfg["d"]))
        w = rng.standard_normal(cfg["d"])
        w /= np.linalg.norm(w)
        delta = X @ w + cfg["sigma"] * rng.standard_normal(cfg["N"])
        tr, _va, te = split_80_10_10(cfg["N"], delta, seed)
        y_tr = sign_label(delta[tr])
        necs = {}
        for strat in ("standard", "weighted"):
            clf = LogisticRegression(max_iter=1000, C=cfg["C"], random_state=seed)
            weight = np.abs(delta[tr]) if strat == "weighted" else None
            clf.fit(X[tr], y_tr, sample_weight=weight)
            necs[strat] = nec(delta[te], clf.predict(X[te]))
        diffs.append((necs["weighted"] - necs["standard"]) * 100)
    d_arr = np.array(diffs)
    mean = float(d_arr.mean())
    half = float(stats.t.ppf(0.975, len(d_arr) - 1) * d_arr.std(ddof=1) / np.sqrt(len(d_arr)))
    varied = ", ".join(f"{k}={v}" for k, v in override.items())
    rows.append(
        {
            "config": varied,
            "wt_minus_std_nec_pts": round(mean, 3),
            "ci95_lo": round(mean - half, 3),
            "ci95_hi": round(mean + half, 3),
            "helps": bool(mean + half < 0),
        }
    )

out = pd.DataFrame(rows)
print("Synthetic sweep: weighted - standard NEC (pts); negative = weighting helps")
print(f"base config: {BASE}, 5 seeds per row, one factor varied per row")
print(out.to_string(index=False))
out.to_csv(Path(__file__).resolve().parents[1] / "outputs" / "synthetic" / "sweep.csv", index=False)
print("wrote outputs/synthetic/sweep.csv")
