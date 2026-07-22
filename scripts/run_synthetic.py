"""Synthetic control (§4.1/§4.2 of arXiv:2605.03135).

Δ is a linear function of isotropic-Gaussian features plus Gaussian noise, so
costs are linearly predictable from features — the paper's idealized case where
cost-weighted training should help. The paper does not specify N, d, the weight
vector, or the noise scale; we choose N=20000, d=20, |w|=1, sigma=0.35 to land
near the paper's reported error-rate scale (~4.6%) and report that choice.

Paper targets (Table 2, Synthetic LogReg): standard CE 0.5 NEC / 4.6 error;
|Δ|-weighted CE 0.4 NEC / 4.1 error (%); error/NEC gap ~9x.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import run_strategies, summarize

N, D, SIGMA = 20_000, 20, 0.35
SEEDS = range(10)

rows = []
for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    X = rng.standard_normal((N, D))
    w = rng.standard_normal(D)
    w /= np.linalg.norm(w)
    delta = X @ w + SIGMA * rng.standard_normal(N)
    rows += run_strategies(
        X,
        delta,
        make_clf=lambda s: LogisticRegression(max_iter=1000, random_state=s),
        make_reg=lambda s: Ridge(random_state=s),
        seed=seed,
    )

summary = summarize(rows)
summary["ratio_err_over_nec"] = (summary["error_mean"] / summary["nec_mean"]).round(2)
print(f"Synthetic control: N={N}, d={D}, sigma={SIGMA}, {len(list(SEEDS))} seeds")
print(summary.to_string(index=False))

out = Path(__file__).resolve().parents[1] / "outputs" / "synthetic"
out.mkdir(parents=True, exist_ok=True)
import pandas as pd

pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
summary.to_csv(out / "summary.csv", index=False)
print(f"wrote {out}/per_seed.csv and summary.csv")
