"""Segment analysis must correct for multiplicity and catch aggregate reversals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ab_engine.segments import analyse_segments, detect_simpson_reversal


def _frame_with_effect_in_one_segment(seed: int = 31) -> pd.DataFrame:
    """Three segments; only ``ios`` carries a real effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for segment in ("android", "ios", "web"):
        effect = 0.6 if segment == "ios" else 0.0
        for variant in ("control", "treatment"):
            shift = effect if variant == "treatment" else 0.0
            for value in rng.normal(shift, 1.0, 1500):
                rows.append({"segment": segment, "variant": variant, "metric": value})
    return pd.DataFrame(rows)


def _simpson_frame() -> pd.DataFrame:
    """Every segment positive, aggregate negative.

    Treatment is concentrated in the low-baseline segment and control in the
    high-baseline segment, so the size-weighted aggregate points the other way from
    the within-segment effects.
    """
    rows = []
    for _ in range(100):
        rows.append({"segment": "low", "variant": "treatment", "metric": 1.0})
    for _ in range(10):
        rows.append({"segment": "low", "variant": "control", "metric": 0.0})
    for _ in range(10):
        rows.append({"segment": "high", "variant": "treatment", "metric": 11.0})
    for _ in range(100):
        rows.append({"segment": "high", "variant": "control", "metric": 10.0})
    return pd.DataFrame(rows)


def test_only_the_segment_with_a_real_effect_survives_correction():
    effects = analyse_segments(_frame_with_effect_in_one_segment(), segment_column="segment")
    by_segment = {effect.segment: effect for effect in effects}
    assert set(by_segment) == {"android", "ios", "web"}
    assert by_segment["ios"].significant_after_correction is True
    assert by_segment["android"].significant_after_correction is False
    assert by_segment["web"].significant_after_correction is False


def test_adjusted_p_values_are_never_smaller_than_raw_ones():
    effects = analyse_segments(_frame_with_effect_in_one_segment(), segment_column="segment")
    for effect in effects:
        assert effect.adjusted_p_value >= effect.p_value - 1e-12


def test_segments_with_too_little_data_are_skipped_not_reported():
    frame = pd.DataFrame(
        [
            {"segment": "tiny", "variant": "treatment", "metric": 1.0},
            {"segment": "tiny", "variant": "control", "metric": 0.0},
            {"segment": "big", "variant": "treatment", "metric": 1.0},
            {"segment": "big", "variant": "treatment", "metric": 1.5},
            {"segment": "big", "variant": "control", "metric": 0.0},
            {"segment": "big", "variant": "control", "metric": 0.4},
        ]
    )
    effects = analyse_segments(frame, segment_column="segment")
    assert [effect.segment for effect in effects] == ["big"]


def test_reversal_is_detected_when_aggregate_contradicts_every_segment():
    reversal, explanation = detect_simpson_reversal(_simpson_frame(), segment_column="segment")
    assert reversal is True
    assert "Simpson" in explanation


def test_no_reversal_when_aggregate_agrees_with_the_segments():
    reversal, explanation = detect_simpson_reversal(
        _frame_with_effect_in_one_segment(), segment_column="segment"
    )
    assert reversal is False
    assert "Simpson" not in explanation


def test_mixed_segment_directions_are_reported_as_heterogeneity_not_reversal():
    frame = pd.DataFrame(
        [
            {"segment": "a", "variant": "treatment", "metric": 2.0},
            {"segment": "a", "variant": "treatment", "metric": 2.2},
            {"segment": "a", "variant": "control", "metric": 1.0},
            {"segment": "a", "variant": "control", "metric": 1.1},
            {"segment": "b", "variant": "treatment", "metric": 1.0},
            {"segment": "b", "variant": "treatment", "metric": 0.9},
            {"segment": "b", "variant": "control", "metric": 2.0},
            {"segment": "b", "variant": "control", "metric": 2.1},
        ]
    )
    reversal, explanation = detect_simpson_reversal(frame, segment_column="segment")
    assert reversal is False
    assert "disagree in direction" in explanation
