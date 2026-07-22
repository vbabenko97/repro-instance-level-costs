"""Core metric + training strategies from arXiv:2605.03135 (Instance-Level Costs).

Implements:
- NEC (Eq. 1), error rate (Eq. 2)
- cost derivations: vote-margin log-odds (Eq. 3), threshold distance (Eq. 4)
- training strategies of §3.6: standard CE, |Δ|-weighted CE, per-class
  probability-proportional upsampling (P_up), top-k% filtering (Tdown k),
  Δ-regression (Eq. 7, classify by thresholding at 0)
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sign_label(delta: np.ndarray) -> np.ndarray:
    """y = sign(Δ); Δ == 0 maps to +1 (e.g. SBP exactly at the ≥130 threshold)."""
    return np.where(delta >= 0, 1, -1)


def nec(delta: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 1: NEC = Σ|Δᵢ|·1[ŷᵢ ≠ sign(Δᵢ)] / Σ|Δᵢ|."""
    y = sign_label(delta)
    return float(np.sum(np.abs(delta) * (y_pred != y)) / np.sum(np.abs(delta)))


def error_rate(delta: np.ndarray, y_pred: np.ndarray) -> float:
    """Eq. 2: plain misclassification rate against y = sign(Δ)."""
    return float(np.mean(y_pred != sign_label(delta)))


def vote_log_odds(n_yes: np.ndarray, n_no: np.ndarray) -> np.ndarray:
    """Eq. 3: Δᵢ = log((n_yes+1)/(n_no+1)) — Laplace-smoothed vote margin."""
    return np.log((n_yes + 1.0) / (n_no + 1.0))


def threshold_distance(z: np.ndarray, tau: float) -> np.ndarray:
    """Eq. 4: Δᵢ = zᵢ − τ."""
    return z - tau


def split_80_10_10(
    n: int, delta: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 train/val/test indices, stratified on sign(Δ) (§4.1)."""
    idx = np.arange(n)
    y = sign_label(delta)
    train, rest = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=y[rest]
    )
    return train, val, test


def p_up_indices(delta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-class resample with replacement, P(select) ∝ |Δ|, class sizes preserved."""
    y = sign_label(delta)
    out = []
    for cls in (-1, 1):
        cls_idx = np.flatnonzero(y == cls)
        w = np.abs(delta[cls_idx])
        if w.sum() == 0:
            w = np.ones_like(w)
        out.append(rng.choice(cls_idx, size=len(cls_idx), replace=True, p=w / w.sum()))
    return np.concatenate(out)


def tdown_indices(delta: np.ndarray, keep_pct: int) -> np.ndarray:
    """Keep examples with |Δ| at or above the per-class (100−k)th percentile."""
    y = sign_label(delta)
    out = []
    for cls in (-1, 1):
        cls_idx = np.flatnonzero(y == cls)
        cut = np.percentile(np.abs(delta[cls_idx]), 100 - keep_pct)
        out.append(cls_idx[np.abs(delta[cls_idx]) >= cut])
    return np.concatenate(out)


def run_strategies(
    X,
    delta: np.ndarray,
    make_clf: Callable[[int], object],
    make_reg: Callable[[int], object] | None,
    seed: int,
    strategies: tuple[str, ...] = (
        "standard",
        "weighted",
        "p_up",
        "tdown30",
        "tdown50",
        "tdown70",
        "regression",
    ),
) -> list[dict]:
    """Train each §3.6 strategy on one seed's split; return test NEC/error rows."""
    tr, _va, te = split_80_10_10(len(delta), delta, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for strat in strategies:
        if strat == "regression":
            if make_reg is None:
                continue
            model = make_reg(seed)
            model.fit(X[tr], delta[tr])
            pred_delta = model.predict(X[te])
            y_pred = np.where(pred_delta >= 0, 1, -1)
            mae = float(np.mean(np.abs(pred_delta - delta[te])))
        else:
            model = make_clf(seed)
            y_tr = sign_label(delta[tr])
            if strat == "standard":
                model.fit(X[tr], y_tr)
            elif strat == "weighted":
                model.fit(X[tr], y_tr, sample_weight=np.abs(delta[tr]))
            elif strat == "p_up":
                sub = tr[p_up_indices(delta[tr], rng)]
                model.fit(X[sub], sign_label(delta[sub]))
            elif strat.startswith("tdown"):
                keep = int(strat.removeprefix("tdown"))
                sub = tr[tdown_indices(delta[tr], keep)]
                model.fit(X[sub], sign_label(delta[sub]))
            else:
                raise ValueError(f"unknown strategy {strat}")
            y_pred = model.predict(X[te])
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
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    """Mean ± 95% CI over seeds per strategy, in percent (matches Table 2 format)."""
    df = pd.DataFrame(rows)
    out = []
    for strat, g in df.groupby("strategy", sort=False):
        rec = {"strategy": strat, "n_seeds": len(g)}
        for m in ("nec", "error"):
            vals = g[m].to_numpy() * 100
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            rec[m] = f"{vals.mean():.2f} ± {ci:.2f}"
            rec[f"{m}_mean"] = vals.mean()
        if not np.isnan(g["mae"]).all():
            rec["mae"] = f"{g['mae'].mean():.3f}"
        out.append(rec)
    return pd.DataFrame(out)
