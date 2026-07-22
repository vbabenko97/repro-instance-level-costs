"""Jigsaw TF-IDF experiment (Claim 1 + Claims 5/6 on text).

Reproduces the Table 2 "Jigsaw TF-IDF" row: standard CE vs |Δ|-weighted CE, plus
sampling strategies and Δ-regression, on the real recovered vote margins.

Paper targets (Table 2, Jigsaw TF-IDF): standard CE 1.8 NEC / 5.3 error;
|Δ|-weighted 1.8 / 5.6 (%); error/NEC ratio > 3x. Δ-regression MAE ≈ 0.30.

Full data is 1.8M rows; TF-IDF + liblinear LogReg on that is memory-heavy on a
16 GB laptop, so a stratified subsample is drawn (default 300k). The paper's own
sample-size scaling (Fig. 3) shows the error/NEC ratio is ~constant from
N=1,000 to 100,000, so the headline ratio is preserved at this scale. Scale is
recorded in the output. Set JIGSAW_N=0 to use all rows.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nec import (
    error_rate,
    nec,
    p_up_indices,
    sign_label,
    split_80_10_10,
    summarize,
    tdown_indices,
)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_parquet(ROOT / "data" / "jigsaw" / "jigsaw_delta.parquet")
delta_all = df["delta"].to_numpy()

N = int(os.environ.get("JIGSAW_N", "300000"))
if N and N < len(df):
    y_all = sign_label(delta_all)
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(y_all == 1)
    neg = np.flatnonzero(y_all == -1)
    frac = N / len(df)
    keep = np.concatenate(
        [
            rng.choice(pos, size=int(len(pos) * frac), replace=False),
            rng.choice(neg, size=int(len(neg) * frac), replace=False),
        ]
    )
    keep.sort()
    df = df.iloc[keep].reset_index(drop=True)
print(f"using N={len(df):,} rows (full set is 1,804,868)")

texts = df["text"].to_numpy()
delta = df["delta"].to_numpy()

STRATS = ("standard", "weighted", "p_up", "tdown30", "tdown50", "tdown70", "regression")
rows = []
for seed in range(10):
    tr, _va, te = split_80_10_10(len(delta), delta, seed)
    # TF-IDF (vocabulary, min_df pruning, IDF) is fit on the train split only —
    # fitting on the full subsample would leak test-document statistics into the
    # representation (sklearn "common pitfalls": fit learned preprocessing on
    # train only). One fit per seed, shared by all strategies.
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = csr_matrix(vec.fit_transform(texts[tr]))
    Xte = csr_matrix(vec.transform(texts[te]))
    y_tr = sign_label(delta[tr])
    srng = np.random.default_rng(seed)
    for strat in STRATS:
        if strat == "regression":
            reg = Ridge(alpha=1.0, random_state=seed)
            reg.fit(Xtr, delta[tr])
            pd_delta = reg.predict(Xte)
            y_pred = np.where(pd_delta >= 0, 1, -1)
            mae = float(np.mean(np.abs(pd_delta - delta[te])))
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=seed)
            if strat == "standard":
                clf.fit(Xtr, y_tr)
            elif strat == "weighted":
                clf.fit(Xtr, y_tr, sample_weight=np.abs(delta[tr]))
            elif strat == "p_up":
                sub = p_up_indices(delta[tr], srng)
                clf.fit(Xtr[sub], sign_label(delta[tr][sub]))
            elif strat.startswith("tdown"):
                keep = tdown_indices(delta[tr], int(strat.removeprefix("tdown")))
                clf.fit(Xtr[keep], sign_label(delta[tr][keep]))
            y_pred = clf.predict(Xte)
            mae = float("nan")
        rows.append(
            {
                "seed": seed,
                "strategy": strat,
                "nec": nec(delta[te], y_pred),
                "error": error_rate(delta[te], y_pred),
                "mae": mae,
                "n_train": len(tr),
                "n_test": len(te),
            }
        )
    print(f"seed {seed} done")

summary = summarize(rows)
summary["ratio_err_over_nec"] = (summary["error_mean"] / summary["nec_mean"]).round(2)
print(f"\nJigsaw TF-IDF, N={len(df):,}, 10 seeds")
print(summary.to_string(index=False))

out = ROOT / "outputs" / "jigsaw_tfidf"
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
summary.to_csv(out / "summary.csv", index=False)
print("wrote outputs/jigsaw_tfidf/per_seed.csv and summary.csv")
