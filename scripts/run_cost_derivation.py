"""Claim 3: the three cost-derivation sources (§3.5) + Δ histograms (Fig. 2).

Demonstrates Eq. 3 and Eq. 4 on real obtainable data and the rating source
synthetically:
  Eq. 3  vote-margin log-odds     -> Jigsaw (toxicity votes), Turkey (injury votes)
  Eq. 4  threshold distance       -> NHANES (SBP - 130)
  rating direct confidence scale  -> synthetic 7-point demo (Δ = score - midpoint),
         the mechanism the paper applied to iNaturalist Gemini ratings
Also emits the signed-Δ histograms (paper Fig. 2), oriented minority-on-right,
with the decision boundary at Δ=0.
"""

import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import threshold_distance, vote_log_odds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Eq. 3 on Jigsaw (already recovered) ---
jig = pd.read_parquet(ROOT / "data" / "jigsaw" / "jigsaw_delta.parquet")
d_jig = jig["delta"].to_numpy()
# sanity: recompute Δ from stored votes to confirm Eq. 3 implementation
assert np.allclose(d_jig, vote_log_odds(jig["n_yes"].to_numpy(), jig["n_no"].to_numpy()))

# --- Eq. 3 on Turkey (any-injury votes) ---
z = zipfile.ZipFile(ROOT / "data" / "turkey" / "Turkey.zip")
import json

ann = json.loads(z.read("Turkey/annotations.json"))
votes = defaultdict(lambda: defaultdict(int))
for rec in ann:
    for a in rec["annotations"]:
        votes[a["image_path"]][a["class_label"]] += 1
paths = sorted(votes)
n_inj = np.array([votes[p]["head_injury"] + votes[p]["plumage_injury"] for p in paths], float)
n_not = np.array([votes[p]["not_injured"] for p in paths], float)
d_turk = vote_log_odds(n_inj, n_not)

# --- Eq. 4 on NHANES ---
nh = pd.read_csv(ROOT / "outputs" / "nhanes" / "nhanes_prepared.csv")
d_nh = threshold_distance(nh["SBP"].to_numpy(), 130.0)

# --- rating demo (iNaturalist mechanism): 7-point scale, Δ = score - 4 ---
rng = np.random.default_rng(0)
scores = rng.integers(1, 8, size=9956)  # 1..7
d_rate = scores - 4.0

stats = []
for name, d in (("Jigsaw (Eq.3 votes)", d_jig), ("Turkey (Eq.3 votes)", d_turk),
                ("NHANES (Eq.4 SBP-130)", d_nh), ("Rating demo (score-4)", d_rate)):
    stats.append(
        {
            "dataset": name,
            "n": len(d),
            "mean_abs_delta": float(np.mean(np.abs(d))),
            "frac_pos": float(np.mean(d >= 0)),
            "frac_low_cost_|d|<0.5": float(np.mean(np.abs(d) < 0.5)),
        }
    )
stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))
stats_df.to_csv(ROOT / "outputs" / "cost_derivation_stats.csv", index=False)

# --- Fig. 2 histograms ---
fig = make_subplots(rows=2, cols=2, subplot_titles=[s["dataset"] for s in stats])
series = [d_jig, d_turk, d_nh, d_rate]
for i, d in enumerate(series):
    r, c = divmod(i, 2)
    fig.add_histogram(x=d, nbinsx=60, marker_color="#2563eb", showlegend=False,
                      row=r + 1, col=c + 1)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=r + 1, col=c + 1)
fig.update_layout(
    title="Claim 3 — Signed Δ distributions (paper Fig. 2); red line = decision boundary Δ=0",
    template="plotly_white",
    height=560,
    margin=dict(t=80, b=40),
)
fig.write_html(OUT / "delta_hist.html", include_plotlyjs="cdn")
stats_df.to_json(OUT / "delta_hist.json", orient="records", indent=2)
print("wrote outputs/figures/delta_hist.html and delta_hist.json")
