# Reproducing the recorded experiments

The repository keeps outputs from one completed agent-driven reproduction. They
are evidence for the reported run, not an assertion that all environments reproduce
the same bytes. The experiment scripts write into `outputs/`; preserve the current
record by working in a copy before rerunning them.

## Data sources

| Experiment | Script | Input created or downloaded | Versioned output |
| --- | --- | --- | --- |
| Jigsaw text | `prepare_jigsaw.py`, `run_jigsaw_tfidf.py` | Hugging Face Jigsaw data; `data/jigsaw/jigsaw_delta.parquet` | `outputs/jigsaw_tfidf/` |
| NHANES tabular | `run_nhanes.py` | CDC 2013–2014 `DEMO_H`, `BMX_H`, and `BPX_H` XPT files | `outputs/nhanes/` |
| Turkey image | `run_turkey.py` | Zenodo DCIC `Turkey.zip`; cached ResNet features | `outputs/turkey/` |
| Synthetic control | `run_synthetic.py`, `run_synthetic_sweep.py` | generated Gaussian features and margins | `outputs/synthetic/` |
| Cost distributions | `run_cost_derivation.py` | Jigsaw, Turkey, and NHANES inputs above | `outputs/cost_derivation_stats.csv`, `outputs/figures/delta_hist.*` |
| Regression comparison | `paired_regression_stats.py` | the three real-data per-seed output files | `outputs/claim6_paired.csv` |

The source URLs and exclusion policy are in [`../DATA_NOT_INCLUDED.md`](../DATA_NOT_INCLUDED.md).
No raw dataset archive is committed. `outputs/nhanes/nhanes_prepared.csv` is a
small derived table used by the cost-distribution script.

## Protocol details

`src/nec.py` defines labels as `sign(Δ)`, maps a zero margin to `+1`, and makes
each `80/10/10` split stratified by that label. `run_jigsaw_tfidf.py` records a
300k stratified sample from the 1,804,868-row Jigsaw dataset; its current protocol
fits the vectorizer on each training split before transforming the test split.

Jigsaw does not expose individual votes in the used fields. The preparation script
reconstructs integer counts as `round(toxicity × toxicity_annotator_count)` and
derives the margin with `log((n_yes + 1) / (n_no + 1))`. This approximation is a
property of this reproduction's data pipeline.

Turkey is evaluated with ImageNet-frozen ResNet-50 features and logistic
regression. Both `head_only` and `any_injury` definitions are reported because the
binary mapping is ambiguous. The fine-tuning script trains a top-block MPS probe
on a reduced subset; its results are exploratory and do not reproduce the paper's
full GPU fine-tuning setting.

The frozen feature cache currently has no source checksum or ordered-image-path
metadata. Reuse it only with the unchanged original archive and extraction order;
regenerate it when either changes. The fine-tuning seed fix does not validate an
existing cache's provenance or alignment.

## Reading uncertainty correctly

The 10 seed-specific evaluations reuse rows across random train/test splits.
Consequently, the t-based intervals in `outputs/claim6_paired.csv` describe the
observed paired split differences only. They do not establish population-level
significance or statistical equivalence. The logbook's early vocabulary-leakage
run is retained only in the original local Trackio export; the committed Jigsaw
outputs use the train-only vectorizer protocol.

iNaturalist is omitted from the runnable set because the paper's graded annotations
that define its cost signal are not included with this repository. The synthetic
rating example therefore demonstrates the transformation, not that dataset.
