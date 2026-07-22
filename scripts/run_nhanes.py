"""NHANES 2013–2014 hypertension experiment (§4.1/§4.2 of arXiv:2605.03135).

Task: hypertension classification at the 2017 ACC/AHA threshold of 130 mmHg
systolic blood pressure, cost Δᵢ = SBPᵢ − 130 (Eq. 4). Features: age, gender,
race/ethnicity, BMI. Model: HistGradientBoosting. Paper reports N=7,455.

Paper targets (Table 2, NHANES HistGBM): standard CE 14.8 NEC / 21.7 error;
|Δ|-weighted CE 14.5 NEC / 21.5 error (%); error/NEC ratio 1.5x.

SBP is the mean of the available BPX readings (BPXSY1..BPXSY3); rows with no
valid SBP, BMI, or demographics are dropped.
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import run_strategies, summarize, threshold_distance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nhanes"
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "DEMO_H.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/DEMO_H.xpt",
    "BMX_H.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BMX_H.xpt",
    "BPX_H.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/BPX_H.xpt",
}
for name, url in FILES.items():
    dest = DATA / name
    if not dest.exists():
        print(f"downloading {url}")
        urlretrieve(url, dest)

demo = pd.read_sas(DATA / "DEMO_H.xpt")[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3"]]
bmx = pd.read_sas(DATA / "BMX_H.xpt")[["SEQN", "BMXBMI"]]
bpx = pd.read_sas(DATA / "BPX_H.xpt")[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3"]]

df = demo.merge(bmx, on="SEQN").merge(bpx, on="SEQN")
df["SBP"] = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)
df = df.dropna(subset=["SBP", "BMXBMI", "RIDAGEYR", "RIAGENDR", "RIDRETH3"])
print(f"N after merge + dropna = {len(df)} (paper reports 7,455)")

delta = threshold_distance(df["SBP"].to_numpy(), 130.0)
X = df[["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI"]].to_numpy()
frac_pos = float(np.mean(delta >= 0))
print(f"hypertensive (SBP>=130) fraction: {frac_pos:.3f}")

CAT = [1, 2]  # RIAGENDR, RIDRETH3 columns are categorical

rows = []
for seed in range(10):
    rows += run_strategies(
        X,
        delta,
        make_clf=lambda s: HistGradientBoostingClassifier(
            categorical_features=CAT, random_state=s
        ),
        make_reg=lambda s: HistGradientBoostingRegressor(
            categorical_features=CAT, random_state=s
        ),
        seed=seed,
    )

summary = summarize(rows)
summary["ratio_err_over_nec"] = (summary["error_mean"] / summary["nec_mean"]).round(2)
print(summary.to_string(index=False))

out = ROOT / "outputs" / "nhanes"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
summary.to_csv(out / "summary.csv", index=False)
df[["SEQN", "SBP", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI"]].to_csv(
    out / "nhanes_prepared.csv", index=False
)
print(f"wrote {out}/per_seed.csv, summary.csv, nhanes_prepared.csv")
