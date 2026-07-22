"""Build the NEC-vs-error-rate bar figure (paper Fig. 1) from summary CSVs.

Usage: uv run python scripts/plot_nec_bars.py OUT.html OUT.json "Title" \
           label1=path/to/summary.csv[:strategy] label2=...

Each summary CSV must have columns strategy, nec_mean, error_mean. Picks the
`standard` strategy row unless an explicit :strategy suffix is given.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def pick(spec: str) -> tuple[str, float, float]:
    label, rhs = spec.split("=", 1)
    path, _, strat = rhs.partition(":")
    strat = strat or "standard"
    df = pd.read_csv(path)
    row = df[df["strategy"] == strat].iloc[0]
    return label, float(row["nec_mean"]), float(row["error_mean"])


out_html, out_json, title, *specs = sys.argv[1:]
labels, nec, err, ratios = [], [], [], []
for s in specs:
    lab, n, e = pick(s)
    labels.append(lab)
    nec.append(round(n, 2))
    err.append(round(e, 2))
    ratios.append(round(e / n, 2))

fig = go.Figure()
fig.add_bar(name="NEC (cost-weighted)", x=labels, y=nec, marker_color="#2563eb",
            text=[f"{v:.1f}" for v in nec], textposition="outside")
fig.add_bar(name="Error rate", x=labels, y=err, marker_color="#f97316",
            text=[f"{v:.1f}" for v in err], textposition="outside")
fig.update_layout(
    title=title,
    barmode="group",
    yaxis_title="%",
    template="plotly_white",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=70, r=20, l=50, b=40),
)
Path(out_html).parent.mkdir(parents=True, exist_ok=True)
fig.write_html(out_html, include_plotlyjs="cdn")
pd.DataFrame({"label": labels, "nec": nec, "error": err, "ratio_err_over_nec": ratios}).to_json(out_json, orient="records", indent=2)
print(f"wrote {out_html} and {out_json}")
