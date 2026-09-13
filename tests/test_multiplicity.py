"""Multiplicity corrections must be step procedures, not per-hypothesis filters."""

from __future__ import annotations

import numpy as np
import pytest

from ab_engine.multiplicity import benjamini_hochberg, holm_bonferroni


def test_bh_is_a_step_up_procedure():
    """The classic bug: rejecting only hypotheses that pass their own threshold.

    With m=3 and alpha=0.05 the thresholds are 0.0167, 0.0333 and 0.05. The middle
    p-value (0.04) fails its own threshold, but the largest (0.045) passes its own, so
    a correct step-up procedure rejects all three.
    """
    rejected, _ = benjamini_hochberg([0.001, 0.04, 0.045], alpha=0.05)
    assert rejected.tolist() == [True, True, True]


def test_bh_rejects_nothing_when_all_p_values_are_large():
    rejected, _ = benjamini_hochberg([0.4, 0.6, 0.9, 0.95], alpha=0.05)
    assert not rejected.any()


def test_bh_preserves_input_order():
    rejected, adjusted = benjamini_hochberg([0.9, 0.0001, 0.8], alpha=0.05)
    assert rejected.tolist() == [False, True, False]
    assert adjusted[1] < adjusted[0]


def test_bh_adjusted_values_are_monotone_in_the_sorted_order():
    p_values = [0.001, 0.008, 0.02, 0.04, 0.2, 0.7]
    _, adjusted = benjamini_hochberg(p_values, alpha=0.05)
    order = np.argsort(p_values)
    ordered = adjusted[order]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(ordered, ordered[1:]))
    assert np.all(adjusted <= 1.0)


def test_bh_controls_the_error_rate_under_a_global_null():
    """With every null true, FDR control coincides with control of any-rejection."""
    rng = np.random.default_rng(17)
    trials = 400
    any_rejection = 0
    for _ in range(trials):
        p_values = rng.uniform(0.0, 1.0, 50)
        rejected, _ = benjamini_hochberg(p_values, alpha=0.05)
        any_rejection += bool(rejected.any())
    assert any_rejection / trials < 0.10


def test_holm_is_a_subset_of_bh():
    """Controlling the family-wise rate is strictly stronger than controlling FDR."""
    rng = np.random.default_rng(23)
    for _ in range(50):
        p_values = np.concatenate(
            [rng.uniform(0.0, 0.02, 5), rng.uniform(0.0, 1.0, 25)]
        )
        bh_rejected, _ = benjamini_hochberg(p_values, alpha=0.05)
        holm_rejected, _ = holm_bonferroni(p_values, alpha=0.05)
        assert np.all(bh_rejected[holm_rejected])


def test_holm_is_at_least_as_powerful_as_bonferroni():
    p_values = [0.001, 0.009, 0.03, 0.5]
    holm_rejected, _ = holm_bonferroni(p_values, alpha=0.05)
    bonferroni_rejected = np.asarray(p_values) <= 0.05 / len(p_values)
    assert holm_rejected.sum() >= bonferroni_rejected.sum()


def test_single_hypothesis_needs_no_correction():
    rejected, adjusted = benjamini_hochberg([0.04], alpha=0.05)
    assert rejected.tolist() == [True]
    assert abs(adjusted[0] - 0.04) < 1e-12


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        benjamini_hochberg([])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5], alpha=1.0)
    with pytest.raises(ValueError):
        holm_bonferroni([np.nan])
