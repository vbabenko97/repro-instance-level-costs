"""Claim 6: paired per-seed test of regression NEC vs classification NEC.

The seeds share train/test splits within a dataset, so the honest comparison is
the PAIRED per-seed difference (regression - standard), not overlapping marginal
CIs. Reports a descriptive nominal t interval for the overlapping seeded splits;
it is not a population-significance claim.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "Jigsaw TF-IDF": ROOT / "outputs" / "jigsaw_tfidf" / "per_seed.csv",
    "Turkey (any_injury)": ROOT / "outputs" / "turkey" / "per_seed_any_injury.csv",
    "NHANES": ROOT / "outputs" / "nhanes" / "per_seed.csv",
}

rows = []
for name, path in SOURCES.items():
    df = pd.read_csv(path)
    piv = df.pivot(index="seed", columns="strategy", values="nec")
    d = (piv["regression"] - piv["standard"]).to_numpy() * 100  # NEC points (%)
    n = len(d)
    mean = float(np.mean(d))
    half = float(stats.t.ppf(0.975, n - 1) * np.std(d, ddof=1) / np.sqrt(n))
    rows.append(
        {
            "dataset": name,
            "n_seeds": n,
            "paired_diff_mean_pts": round(mean, 3),
            "ci95_lo": round(mean - half, 3),
            "ci95_hi": round(mean + half, 3),
            "excludes_zero": bool(mean - half > 0 or mean + half < 0),
        }
    )

out = pd.DataFrame(rows)
print("Paired per-seed NEC difference: regression - standard (percentage points)")
print(out.to_string(index=False))
out.to_csv(ROOT / "outputs" / "claim6_paired.csv", index=False)
print("wrote outputs/claim6_paired.csv")
