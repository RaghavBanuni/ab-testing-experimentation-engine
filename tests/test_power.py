"""Power and MDE must be consistent inverses of one another."""

from __future__ import annotations

import pytest

from ab_engine.power import (
    mde_for_total_n,
    sample_size_per_arm,
    sample_size_proportion,
    total_sample_size,
)


def test_sample_size_and_mde_are_inverses():
    sigma, mde = 1.0, 0.05
    total = total_sample_size(sigma, mde, alpha=0.05, power=0.8)
    recovered = mde_for_total_n(sigma, total, alpha=0.05, power=0.8)
    assert abs(recovered - mde) / mde < 0.01


def test_more_power_costs_more_traffic():
    small = total_sample_size(1.0, 0.05, power=0.8)
    large = total_sample_size(1.0, 0.05, power=0.95)
    assert large > small


def test_smaller_effects_cost_quadratically_more_traffic():
    """Halving the MDE should roughly quadruple the required sample size."""
    coarse = total_sample_size(1.0, 0.10)
    fine = total_sample_size(1.0, 0.05)
    assert 3.9 < fine / coarse < 4.1


def test_balanced_allocation_is_the_cheapest():
    balanced = total_sample_size(1.0, 0.05, allocation=0.5)
    skewed = total_sample_size(1.0, 0.05, allocation=0.2)
    assert skewed > balanced


def test_per_arm_is_half_of_the_balanced_total():
    total = total_sample_size(1.0, 0.05)
    per_arm = sample_size_per_arm(1.0, 0.05)
    assert abs(per_arm - total / 2) <= 1


def test_proportion_sample_size_grows_as_the_lift_shrinks():
    big_lift = sample_size_proportion(0.10, 0.20)
    small_lift = sample_size_proportion(0.10, 0.05)
    assert small_lift > big_lift


def test_proportion_sample_size_is_symmetric_in_lift_direction():
    """Detecting a 10% drop is about as expensive as detecting a 10% rise."""
    up = sample_size_proportion(0.20, 0.10)
    down = sample_size_proportion(0.20, -0.10)
    assert abs(up - down) / up < 0.05


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        total_sample_size(0.0, 0.05)
    with pytest.raises(ValueError):
        total_sample_size(1.0, 0.0)
    with pytest.raises(ValueError):
        total_sample_size(1.0, 0.05, allocation=0.0)
    with pytest.raises(ValueError):
        mde_for_total_n(1.0, 0)
    with pytest.raises(ValueError):
        sample_size_proportion(1.5, 0.1)
    with pytest.raises(ValueError):
        sample_size_proportion(0.1, 0.0)
    with pytest.raises(ValueError):
        sample_size_proportion(0.9, 0.5)
