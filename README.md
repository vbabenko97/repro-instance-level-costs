# Reproduction — "Instance-Level Costs for Nuanced Classifier Evaluation"

Independent reproduction of ICML 2026 paper #31878 (OpenReview `qMI1xD8O3x`,
arXiv:2605.03135), Kang & Mussmann. Part of the HF × AlphaXiv ICML-2026 agent
reproduction challenge.

The paper proposes **Normalized Excess Cost (NEC)**, a metric that weights
classification errors by per-example costs `|Δ|` and reduces to error rate when
costs are uniform. This reproduction verifies the metric's mechanism and the
training-strategy results on the datasets that are publicly obtainable.

- **Live logbook (Trackio Space):** https://huggingface.co/spaces/vbabenko97/repro-instance-level-costs
- **Paper:** OpenReview [`qMI1xD8O3x`](https://openreview.net/forum?id=qMI1xD8O3x) · arXiv:2605.03135 (Kang & Mussmann)

## What reproduces

| Claim | Result | Scale |
|---|---|---|
| 1. Jigsaw NEC ≈ ⅓ error rate | 2.02 NEC / 5.91 error, **2.92x** (paper ~3x); TF-IDF fit on train splits only | real 1.8M data, 300k stratified subsample, 10 seeds |
| 2. Divergence across modalities | Turkey 1.65x (any-injury; head-only 1.51x), NHANES 1.37x (paper 1.6x, 1.5x); iNat blocked | real data |
| 3. Three cost sources (Eq. 3/4/rating) | partial — Eq. 3/4 verified on real data; rating transform synthetic demo only | real + synthetic |
| 5. Cost-weighting inconsistent | partial — Turkey −11.6%, NHANES −1.9% match; Jigsaw +11.4% (harms; paper neutral); synthetic control neutral (paper −18.6%) | real data |
| 6. Δ-regression = calibration not accuracy | MAE 0.325 (paper 0.30); reg NEC slightly lower everywhere, paired diff significant on Jigsaw (−0.06 pts) | real data |
| 4. Fine-tuned Table 2 numbers | NOT reproduced — toy MPS probe only (exploratory) | reduced scale, no GPU |

Blocked: iNaturalist (Gemini annotations unreleased), GPU fine-tuning (HF Jobs 402 — no credits).
Integrity: `manifest.sha256` lists SHA-256 hashes of every payload file in this bundle (all files except the manifest itself).

## Verification

Produced as an agent-driven entry to the Hugging Face × AlphaXiv ICML-2026
reproduction challenge. The published logbook was scored by the challenge's automated
**Logbook Judge** (model `GLM-5.2`): the two headline claims graded **`verified`**
(NEC ≪ error rate on real Jigsaw — 2.92x) and **`inconclusive`** (cost-weighted
training benefits are inconsistent), overall quality **`high`**
([public verdicts](https://huggingface.co/datasets/ICML-2026-agent-repro/verdicts)).
This is automated challenge verification, not peer review.

## Layout

```
src/nec.py                 core metric + cost derivations + training strategies
scripts/prepare_jigsaw.py  download Jigsaw, recover votes -> data/jigsaw/jigsaw_delta.parquet
scripts/run_jigsaw_tfidf.py  Claim 1/5/6 on text
scripts/run_nhanes.py      Claim 2/5 on NHANES tabular (auto-downloads CDC XPT)
scripts/run_turkey.py      Claim 2/5 on Turkey images (frozen ResNet-50 features)
scripts/run_turkey_finetune.py  Claim 4 toy fine-tune (MPS)
scripts/run_synthetic.py   Claim 5 synthetic control
scripts/run_synthetic_sweep.py  Claim 5 sweep: weighting neutrality across N/sigma/d/C
scripts/run_cost_derivation.py  Claim 3 cost equations + Fig. 2 histograms
scripts/paired_regression_stats.py  Claim 6 paired per-seed regression-vs-standard test
scripts/plot_nec_bars.py   figure helper
outputs/                   per-seed CSVs, summaries, figures (committed)
data/                      downloaded/derived inputs (NOT committed; regenerable)
```

## Rerun

Requires `uv`. Data downloads are ~930 MB (Jigsaw parquet + Turkey.zip + NHANES XPT).

```bash
uv sync
uv run python scripts/prepare_jigsaw.py          # -> data/jigsaw/jigsaw_delta.parquet
JIGSAW_N=300000 uv run python scripts/run_jigsaw_tfidf.py
uv run python scripts/run_nhanes.py              # auto-downloads CDC XPT
uv run python scripts/run_turkey.py              # auto-downloads Turkey.zip, extracts ResNet features
uv run python scripts/run_synthetic.py
uv run python scripts/run_synthetic_sweep.py     # Claim 5 config sweep
uv run python scripts/run_cost_derivation.py
uv run python scripts/paired_regression_stats.py # Claim 6 paired test (needs the runs above)
FT_N=2500 FT_EPOCHS=12 uv run python scripts/run_turkey_finetune.py  # toy, MPS (reported config)
```

All metric/mechanism claims (1, 2-NHANES/Turkey, 3, 5, 6) run on CPU/MPS in
minutes each. Claim 4 is a reduced-scale toy (no GPU available).

## Provenance of data (all public, no credentials)

- Jigsaw: HF `TheMrguiller/jigsaw-unintended-bias-in-toxicity-classification`
- Turkey: Zenodo `10.5281/zenodo.8115942` (DCIC benchmark)
- NHANES 2013–2014: CDC `wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/`

## Quick check

A fast, data-free smoke test verifies the metric's defining property (NEC reduces
to error rate under uniform costs) and seed-determinism of the resampling strategy:

```bash
pip install numpy pandas scikit-learn scipy pytest && pytest -q
```

## License and attribution

Reproduction code: MIT (see `LICENSE`). The paper, the datasets (Jigsaw, Turkey/DCIC,
NHANES), and their sources retain their own terms; attribution and provenance are in
`NOTICE` and the "Provenance of data" section above. This reproduction is not
affiliated with or endorsed by the paper's authors.
