"""Data-free smoke test for the core metric (src/nec.py).

Checks the paper's defining property — NEC reduces to error rate when costs are
uniform, and is dominated by errors on high-|Δ| examples — plus seed-determinism of
the probability-proportional resampling. No downloads required."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import nec  # noqa: E402


def test_nec_equals_error_rate_under_uniform_costs():
    # |Δ| constant -> Eq. 1 collapses to Eq. 2.
    delta = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    y = nec.sign_label(delta)
    y_pred = y.copy()
    y_pred[:2] = -y_pred[:2]  # flip two -> error rate 0.2
    assert abs(nec.error_rate(delta, y_pred) - 0.2) < 1e-12
    assert abs(nec.nec(delta, y_pred) - nec.error_rate(delta, y_pred)) < 1e-12


def test_nec_below_error_when_mistakes_are_low_cost():
    # Errors fall on the smallest-|Δ| examples -> NEC << error rate.
    delta = np.array([1.0, -1.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
    y = nec.sign_label(delta)
    y_pred = y.copy()
    y_pred[:2] = -y_pred[:2]  # two errors, both |Δ| = 1
    assert abs(nec.error_rate(delta, y_pred) - 0.25) < 1e-12
    # NEC = (1 + 1) / sum(|Δ|) = 2 / 62
    assert abs(nec.nec(delta, y_pred) - (2.0 / 62.0)) < 1e-12
    assert nec.nec(delta, y_pred) < nec.error_rate(delta, y_pred)


def test_pup_resampling_is_seed_deterministic():
    delta = np.linspace(-5, 5, 200)
    a = nec.p_up_indices(delta, np.random.default_rng(0))
    b = nec.p_up_indices(delta, np.random.default_rng(0))
    assert np.array_equal(a, b)
