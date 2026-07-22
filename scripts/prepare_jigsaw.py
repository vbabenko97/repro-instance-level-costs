"""Download the Jigsaw Unintended Bias data and recover per-example vote margins.

Source (verified public, no credentials): HF dataset
TheMrguiller/jigsaw-unintended-bias-in-toxicity-classification, which carries the
Civil Comments columns `toxicity` (fraction of annotators calling the comment
toxic) and `toxicity_annotator_count`. The paper derives Δ from raw votes
(Eq. 3); we recover them:

    n_yes = round(toxicity * count),  n_no = count - n_yes,
    Δ     = log((n_yes+1)/(n_no+1)).

Writes a compact parquet (text, n_yes, n_no, delta) so the classifier script
does not re-download 1.8M rows each run.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "jigsaw"
OUT.mkdir(parents=True, exist_ok=True)

BASE = (
    "https://huggingface.co/datasets/TheMrguiller/"
    "jigsaw-unintended-bias-in-toxicity-classification/resolve/"
    "refs%2Fconvert%2Fparquet/default/train"
)
parts = [f"{BASE}/0000.parquet", f"{BASE}/0001.parquet"]

frames = []
for url in parts:
    print(f"reading {url}")
    frames.append(pd.read_parquet(url, columns=["text", "toxicity", "toxicity_annotator_count"]))
df = pd.concat(frames, ignore_index=True)
print(f"loaded {len(df):,} rows (paper cites ~1.8M)")

df = df.dropna(subset=["text", "toxicity", "toxicity_annotator_count"])
df = df[df["toxicity_annotator_count"] > 0].reset_index(drop=True)

count = df["toxicity_annotator_count"].to_numpy()
n_yes = np.round(df["toxicity"].to_numpy() * count).astype(int)
n_no = count.astype(int) - n_yes
delta = np.log((n_yes + 1.0) / (n_no + 1.0))

out = pd.DataFrame({"text": df["text"].to_numpy(), "n_yes": n_yes, "n_no": n_no, "delta": delta})
print(f"toxic (delta>=0, minority) fraction: {float(np.mean(delta >= 0)):.4f}")
print(f"delta range [{delta.min():.3f}, {delta.max():.3f}]; N={len(out):,}")

dest = OUT / "jigsaw_delta.parquet"
out.to_parquet(dest, index=False)
print(f"wrote {dest} ({dest.stat().st_size/1e6:.1f} MB)")
