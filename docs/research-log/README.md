# Experiment-log index

The original working folder contains a Trackio logbook with command transcripts,
intermediate attempts, and embedded Plotly payloads. Its large embedded pages are
kept locally in `../repro-31878-instance-level-costs/.trackio/`; they are not copied
here because the data URI and script payloads add about 24 MB without adding a
portable research artifact.

This repository keeps the source code, final CSVs, figures, and the following
portable record of what the logbook established:

| Topic | Final record | Source in this repository |
| --- | --- | --- |
| Claim 1 — Jigsaw gap | 2.02% NEC, 5.91% error, 2.92× on a 300k sample | `outputs/jigsaw_tfidf/`, `scripts/run_jigsaw_tfidf.py` |
| Claim 2 — modalities | Turkey 1.65× and NHANES 1.37×; iNaturalist blocked | `outputs/turkey/`, `outputs/nhanes/` |
| Claim 3 — cost derivation | Eq. 3 and Eq. 4 run on real data; rating is synthetic | `scripts/run_cost_derivation.py`, `outputs/cost_derivation_stats.csv` |
| Claim 4 — fine tuning | Reduced MPS probe only; not reproduced | `outputs/turkey_ft/`, `scripts/run_turkey_finetune.py` |
| Claim 5 — weighting | Partial and dataset-specific; Jigsaw/synthetic differ from paper | `outputs/master_results.csv`, `outputs/synthetic/` |
| Claim 6 — Δ regression | Jigsaw MAE 0.325; paired differences are descriptive only | `outputs/claim6_paired.csv`, `scripts/paired_regression_stats.py` |
| Feasibility | Jigsaw, Turkey, and NHANES are scripted; iNaturalist annotations unavailable | [`../reproducing.md`](../reproducing.md) |

The source Trackio pages also include two superseded Jigsaw runs and generated
interactive Plotly HTML. The committed result files and scripts above are the
canonical record for this repository. The full local export is intentionally
retained in the old working folder rather than published as a second source of
truth.
