"""Fixed-horizon tests: the baseline the sequential methods are judged against."""

from __future__ import annotations

import numpy as np
import pytest

from ab_engine.fixed_horizon import two_proportion_z_test, welch_t_test


def test_welch_finds_a_large_effect():
    rng = np.random.default_rng(1)
    control = rng.normal(0.0, 1.0, 2000)
    treatment = rng.normal(0.5, 1.0, 2000)
    result = welch_t_test(treatment, control)
    assert result.p_value < 1e-10
    assert result.ci_lower > 0.0
    assert result.significant is True


def test_welch_finds_nothing_under_a_true_null():
    rng = np.random.default_rng(2)
    control = rng.normal(0.0, 1.0, 2000)
    treatment = rng.normal(0.0, 1.0, 2000)
    result = welch_t_test(treatment, control)
    assert result.p_value > 0.05
    assert result.significant is False


def test_welch_estimate_sign_means_treatment_minus_control():
    treatment = np.array([1.0, 1.1, 0.9, 1.0])
    control = np.array([0.0, 0.1, -0.1, 0.0])
    assert welch_t_test(treatment, control).estimate > 0
    assert welch_t_test(control, treatment).estimate < 0


def test_welch_handles_unequal_variances_between_arms():
    """Unequal variance is the normal case, not an exception."""
    rng = np.random.default_rng(4)
    control = rng.normal(0.0, 1.0, 1000)
    treatment = rng.normal(0.0, 4.0, 1000)
    result = welch_t_test(treatment, control)
    assert np.isfinite(result.p_value)
    assert 0.0 <= result.p_value <= 1.0


def test_welch_returns_a_degenerate_result_for_constant_arms():
    result = welch_t_test(np.ones(5), np.ones(5))
    assert result.standard_error == 0.0
    assert result.p_value == 1.0


def test_welch_requires_two_observations_per_arm():
    with pytest.raises(ValueError):
        welch_t_test(np.array([1.0]), np.array([0.0, 1.0]))


def test_proportion_test_finds_a_real_lift():
    result = two_proportion_z_test(1200, 10000, 1000, 10000)
    assert result.estimate > 0
    assert result.p_value < 0.001
    assert result.ci_lower > 0.0


def test_proportion_test_finds_nothing_on_equal_rates():
    result = two_proportion_z_test(1000, 10000, 1000, 10000)
    assert result.estimate == 0.0
    assert result.p_value == 1.0


def test_proportion_test_confidence_interval_uses_unpooled_error():
    """The interval must not be built from the pooled null variance."""
    result = two_proportion_z_test(1200, 10000, 1000, 10000)
    p_t, p_c = 0.12, 0.10
    expected_se = np.sqrt(p_t * (1 - p_t) / 10000 + p_c * (1 - p_c) / 10000)
    assert abs(result.standard_error - expected_se) < 1e-12


def test_proportion_test_rejects_impossible_counts():
    with pytest.raises(ValueError):
        two_proportion_z_test(11, 10, 5, 10)
    with pytest.raises(ValueError):
        two_proportion_z_test(5, 10, -1, 10)
    with pytest.raises(ValueError):
        two_proportion_z_test(5, 0, 5, 10)
