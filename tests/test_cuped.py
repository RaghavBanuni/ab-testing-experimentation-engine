"""CUPED must reduce variance without moving the estimate."""

from __future__ import annotations

import numpy as np

from ab_engine.cuped import cuped_adjust, cuped_theta, variance_reduction_factor


def _correlated_pair(n: int = 20000, rho: float = 0.8, seed: int = 3):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    outcome = rho * covariate + np.sqrt(1.0 - rho**2) * noise
    return outcome, covariate


def test_theta_recovers_the_generating_slope():
    rng = np.random.default_rng(11)
    covariate = rng.normal(0.0, 2.0, 50000)
    outcome = 3.0 * covariate + rng.normal(0.0, 1.0, 50000)
    assert abs(cuped_theta(outcome, covariate) - 3.0) < 0.05


def test_adjustment_reduces_variance_when_covariate_is_correlated():
    outcome, covariate = _correlated_pair()
    adjusted = cuped_adjust(outcome, covariate)
    assert adjusted.var(ddof=1) < outcome.var(ddof=1)


def test_adjustment_preserves_the_mean_exactly():
    """Unbiasedness: the subtracted term has mean zero by construction."""
    outcome, covariate = _correlated_pair()
    adjusted = cuped_adjust(outcome, covariate)
    assert np.isclose(adjusted.mean(), outcome.mean())


def test_predicted_reduction_matches_the_achieved_reduction():
    outcome, covariate = _correlated_pair(rho=0.7)
    predicted = variance_reduction_factor(outcome, covariate)
    achieved = cuped_adjust(outcome, covariate).var(ddof=1) / outcome.var(ddof=1)
    assert abs(predicted - achieved) < 0.01


def test_uncorrelated_covariate_leaves_variance_essentially_unchanged():
    rng = np.random.default_rng(5)
    outcome = rng.normal(0.0, 1.0, 20000)
    covariate = rng.normal(0.0, 1.0, 20000)
    assert variance_reduction_factor(outcome, covariate) > 0.99


def test_constant_covariate_is_a_no_op():
    outcome = np.array([1.0, 2.0, 3.0, 4.0])
    covariate = np.array([5.0, 5.0, 5.0, 5.0])
    assert cuped_theta(outcome, covariate) == 0.0
    assert np.allclose(cuped_adjust(outcome, covariate), outcome)
    assert variance_reduction_factor(outcome, covariate) == 1.0


def test_shared_theta_adjusts_both_arms_identically():
    """Per-arm theta would let the adjustment absorb the treatment effect."""
    outcome, covariate = _correlated_pair(n=10000)
    theta = cuped_theta(outcome, covariate)
    mean = float(covariate.mean())
    first_half = cuped_adjust(outcome[:5000], covariate[:5000], theta=theta, covariate_mean=mean)
    expected = outcome[:5000] - theta * (covariate[:5000] - mean)
    assert np.allclose(first_half, expected)
