"""The SRM gate must fire on broken allocation and stay quiet otherwise."""

from __future__ import annotations

import pytest

from ab_engine.diagnostics import srm_check


def test_balanced_allocation_passes():
    result = srm_check([50000, 50120])
    assert result.mismatch is False
    assert "No sample ratio mismatch" in result.explain()


def test_clearly_skewed_allocation_fires():
    """A 5000/4500 split on a 50/50 design is a bug, not bad luck."""
    result = srm_check([5000, 4500])
    assert result.mismatch is True
    assert result.p_value < 0.001
    assert "Sample ratio mismatch" in result.explain()


def test_designed_imbalance_is_not_a_mismatch():
    """A 90/10 ramp must not be reported as broken when it matches the design."""
    result = srm_check([9000, 1000], expected_ratios=[0.9, 0.1])
    assert result.mismatch is False


def test_designed_imbalance_evaluated_against_the_wrong_design_does_fire():
    result = srm_check([9000, 1000], expected_ratios=[0.5, 0.5])
    assert result.mismatch is True


def test_degrees_of_freedom_is_arms_minus_one():
    assert srm_check([100, 100]).degrees_of_freedom == 1
    assert srm_check([100, 100, 100]).degrees_of_freedom == 2


def test_expected_counts_sum_to_the_observed_total():
    result = srm_check([300, 700], expected_ratios=[0.5, 0.5])
    assert abs(sum(result.expected) - 1000) < 1e-9


def test_three_arm_experiment_detects_one_broken_arm():
    result = srm_check([10000, 10000, 8000])
    assert result.mismatch is True


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        srm_check([100])
    with pytest.raises(ValueError):
        srm_check([0, 0])
    with pytest.raises(ValueError):
        srm_check([10, -1])
    with pytest.raises(ValueError):
        srm_check([10, 10], expected_ratios=[1.0])
    with pytest.raises(ValueError):
        srm_check([10, 10], expected_ratios=[0.0, 0.0])
