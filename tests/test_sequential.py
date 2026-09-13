"""The sequential tests must keep their guarantee under repeated looking.

The two simulation tests here are the point of the whole repository: they *measure*
the peeking failure rather than asserting it. Sample sizes are kept modest so the
suite stays quick; tolerances are set wide enough that Monte Carlo noise cannot flip
the result, and the seed is fixed.
"""

from __future__ import annotations

import math

import pytest
from scipy import stats

from ab_engine.sequential import (
    msprt,
    msprt_confidence_sequence,
    msprt_log_likelihood_ratio,
    obf_critical_values,
    obf_nominal_alphas,
    obf_spending,
)
from ab_engine.simulate import simulate


def test_log_likelihood_ratio_is_negative_with_no_observed_effect():
    """At zero estimate the mixture ratio is below 1, so its log is negative."""
    assert msprt_log_likelihood_ratio(0.0, 0.01, 0.1) < 0.0


def test_log_likelihood_ratio_grows_with_the_effect():
    small = msprt_log_likelihood_ratio(0.01, 0.01, 0.1)
    large = msprt_log_likelihood_ratio(0.10, 0.01, 0.1)
    assert large > small


def test_confidence_sequence_brackets_the_estimate():
    lower, upper = msprt_confidence_sequence(0.05, 0.01, 0.1, alpha=0.05)
    assert lower < 0.05 < upper


def test_confidence_sequence_is_wider_than_the_fixed_horizon_interval():
    """Being allowed to look continuously has to cost something, and this is it."""
    variance = 0.0025
    lower, upper = msprt_confidence_sequence(0.0, variance, tau=0.1, alpha=0.05)
    sequential_half_width = (upper - lower) / 2.0
    fixed_half_width = stats.norm.ppf(0.975) * math.sqrt(variance)
    assert sequential_half_width > fixed_half_width


def test_spending_function_endpoints_and_monotonicity():
    alpha = 0.05
    assert obf_spending(0.0, alpha) == 0.0
    assert abs(obf_spending(1.0, alpha) - alpha) < 1e-9
    values = [obf_spending(t / 10.0, alpha) for t in range(1, 11)]
    assert all(later >= earlier for earlier, later in zip(values, values[1:]))


def test_spending_increments_sum_to_alpha():
    alpha = 0.05
    increments = obf_nominal_alphas([0.2, 0.4, 0.6, 0.8, 1.0], alpha)
    assert abs(sum(increments) - alpha) < 1e-9


def test_early_looks_have_stricter_boundaries():
    """O'Brien-Fleming spends little early, so early critical values are larger."""
    criticals = obf_critical_values([0.25, 0.5, 0.75, 1.0], 0.05)
    assert all(later <= earlier for earlier, later in zip(criticals, criticals[1:]))


def test_spending_rejects_non_increasing_information():
    with pytest.raises(ValueError):
        obf_nominal_alphas([0.5, 0.25, 1.0], 0.05)


def test_msprt_result_reports_rejection_for_a_clear_effect():
    result = msprt(estimate=0.5, variance=0.0025, tau=0.1, alpha=0.05)
    assert result.reject_null is True
    assert result.ci_lower > 0.0


def test_peeking_inflates_the_false_positive_rate_above_a_single_look():
    """The failure this repository exists to fix, measured directly.

    Under a true null, checking a fixed-horizon test at five looks and stopping at the
    first significant result must reject noticeably more often than testing once.
    """
    report = simulate(
        n_experiments=800,
        n_per_arm=800,
        n_looks=5,
        true_effect=0.0,
        alpha=0.05,
        tau=0.1,
        seed=2024,
    )
    peeking = report.outcome("fixed_peeking").rejection_rate
    single_look = report.outcome("fixed_final").rejection_rate
    assert peeking > single_look
    assert peeking > 0.08


def test_sequential_methods_hold_their_level_under_the_same_peeking():
    """Both always-valid methods must stay at or below the nominal level.

    The bound is alpha; the allowance added here is Monte Carlo slack only.
    """
    report = simulate(
        n_experiments=800,
        n_per_arm=800,
        n_looks=5,
        true_effect=0.0,
        alpha=0.05,
        tau=0.1,
        seed=2024,
    )
    assert report.outcome("msprt").rejection_rate <= 0.08
    assert report.outcome("alpha_spending").rejection_rate <= 0.08


def test_sequential_methods_retain_power_on_a_real_effect():
    report = simulate(
        n_experiments=400,
        n_per_arm=800,
        n_looks=5,
        true_effect=0.25,
        alpha=0.05,
        tau=0.1,
        seed=99,
    )
    assert report.outcome("msprt").rejection_rate > 0.5
    assert report.outcome("msprt").mean_stop_fraction < 1.0


def test_simulate_rejects_too_few_units_per_look():
    with pytest.raises(ValueError):
        simulate(n_experiments=2, n_per_arm=4, n_looks=5)
