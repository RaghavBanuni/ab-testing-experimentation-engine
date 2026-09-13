"""Assignment must be stable, uniform, and independent across experiments."""

from __future__ import annotations

import pytest

from ab_engine.bucketing import assign, bucket_fraction, in_experiment


def test_bucket_fraction_is_deterministic_across_calls():
    first = bucket_fraction("user-42", "exp-a")
    second = bucket_fraction("user-42", "exp-a")
    assert first == second


def test_bucket_fraction_stays_in_unit_interval():
    for index in range(2000):
        fraction = bucket_fraction(f"user-{index}", "exp-a")
        assert 0.0 <= fraction < 1.0


def test_assignment_is_stable_for_the_same_unit():
    variants = ["control", "treatment"]
    first = assign("user-7", "checkout-v2", variants)
    second = assign("user-7", "checkout-v2", variants)
    assert first.variant == second.variant


def test_balanced_split_is_close_to_even():
    """A 50/50 design should land near 50/50; wide tolerance keeps this deterministic."""
    n_units = 20000
    treatment = sum(
        assign(f"user-{i}", "exp-balance", ["control", "treatment"]).variant == "treatment"
        for i in range(n_units)
    )
    assert 0.48 < treatment / n_units < 0.52


def test_weighted_split_respects_weights():
    n_units = 20000
    treatment = sum(
        assign(
            f"user-{i}", "exp-weighted", ["control", "treatment"], weights=[0.9, 0.1]
        ).variant
        == "treatment"
        for i in range(n_units)
    )
    assert 0.08 < treatment / n_units < 0.12


def test_two_experiments_assign_independently():
    """Concurrent experiments must not be confounded.

    If both experiments hashed the unit id alone, agreement would be 100%. Independent
    assignment puts agreement near 50%.
    """
    n_units = 20000
    agree = 0
    for index in range(n_units):
        unit = f"user-{index}"
        first = assign(unit, "exp-one", ["control", "treatment"]).variant
        second = assign(unit, "exp-two", ["control", "treatment"]).variant
        agree += first == second
    assert 0.47 < agree / n_units < 0.53


def test_raising_exposure_admits_a_superset_of_units():
    """Ramping traffic up must never reshuffle who was already in the experiment."""
    low = {
        f"user-{i}" for i in range(5000) if in_experiment(f"user-{i}", "exp-ramp", 0.2)
    }
    high = {
        f"user-{i}" for i in range(5000) if in_experiment(f"user-{i}", "exp-ramp", 0.5)
    }
    assert low <= high
    assert len(low) < len(high)


def test_variant_split_is_balanced_within_a_partial_ramp():
    """Exposure and variant use separate salts, so the split inside a ramp stays even."""
    exposed_treatment = 0
    exposed_total = 0
    for index in range(30000):
        assignment = assign(
            f"user-{index}", "exp-ramp-split", ["control", "treatment"], exposure=0.1
        )
        if assignment.exposed:
            exposed_total += 1
            exposed_treatment += assignment.variant == "treatment"
    assert exposed_total > 2000
    assert 0.46 < exposed_treatment / exposed_total < 0.54


def test_unexposed_units_get_no_variant():
    assignment = assign("user-1", "exp-zero", ["control", "treatment"], exposure=0.0)
    assert assignment.exposed is False
    assert assignment.variant is None


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        assign("user-1", "exp", [])
    with pytest.raises(ValueError):
        assign("user-1", "exp", ["a", "b"], weights=[1.0])
    with pytest.raises(ValueError):
        assign("user-1", "exp", ["a", "b"], weights=[0.0, 0.0])
    with pytest.raises(ValueError):
        in_experiment("user-1", "exp", 1.5)
