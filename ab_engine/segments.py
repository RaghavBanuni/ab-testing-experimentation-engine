"""Segment-level effects, and the two ways they mislead.

Slicing an experiment by country, platform or user tenure is the most common source
of false wins in experimentation, for two independent reasons.

**Multiplicity.** Ten segments is ten more tests. Segment analysis is almost always
exploratory, so :func:`ab_engine.multiplicity.benjamini_hochberg` is applied across
the segments here by default. A segment that survives correction is a hypothesis worth
re-testing in a dedicated experiment -- not a result to ship on.

**Simpson's paradox.** An effect can be positive in every segment and negative
overall, or the reverse, whenever the segments differ in both their allocation and
their baseline. The aggregate is a weighted average whose weights are the segment
sizes, so if the treatment shifts the *mix* of segments, the aggregate moves for a
reason that has nothing to do with the within-segment effects.

The practical rule this module encodes: when the aggregate direction disagrees with
the consistent direction of the segments, the aggregate is the number that needs
explaining, not the segments. A reversal is usually a sign that segment membership is
itself affected by the treatment, which means it is not a legitimate covariate to
slice on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .fixed_horizon import welch_t_test
from .multiplicity import benjamini_hochberg


@dataclass(frozen=True)
class SegmentEffect:
    """The effect within one segment."""

    segment: str
    estimate: float
    p_value: float
    adjusted_p_value: float
    significant_after_correction: bool
    n_treatment: int
    n_control: int


def analyse_segments(
    frame: pd.DataFrame,
    segment_column: str,
    variant_column: str = "variant",
    metric_column: str = "metric",
    treatment_label: str = "treatment",
    control_label: str = "control",
    alpha: float = 0.05,
    min_per_arm: int = 2,
) -> list[SegmentEffect]:
    """Estimate the effect within each segment, corrected for multiplicity.

    Segments with fewer than ``min_per_arm`` units in either arm are skipped rather
    than reported with a meaningless interval.
    """
    for column in (segment_column, variant_column, metric_column):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} not present in frame")

    names: list[str] = []
    estimates: list[float] = []
    p_values: list[float] = []
    sizes: list[tuple[int, int]] = []

    for segment, group in frame.groupby(segment_column, sort=True):
        treatment = group.loc[group[variant_column] == treatment_label, metric_column]
        control = group.loc[group[variant_column] == control_label, metric_column]
        if treatment.size < min_per_arm or control.size < min_per_arm:
            continue
        result = welch_t_test(treatment.to_numpy(), control.to_numpy(), alpha=alpha)
        names.append(str(segment))
        estimates.append(result.estimate)
        p_values.append(result.p_value)
        sizes.append((result.n_treatment, result.n_control))

    if not names:
        return []

    rejected, adjusted = benjamini_hochberg(p_values, alpha=alpha)
    return [
        SegmentEffect(
            segment=name,
            estimate=estimate,
            p_value=p_value,
            adjusted_p_value=float(adjusted_p),
            significant_after_correction=bool(is_rejected),
            n_treatment=n_t,
            n_control=n_c,
        )
        for name, estimate, p_value, adjusted_p, is_rejected, (n_t, n_c) in zip(
            names, estimates, p_values, adjusted, rejected, sizes
        )
    ]


def detect_simpson_reversal(
    frame: pd.DataFrame,
    segment_column: str,
    variant_column: str = "variant",
    metric_column: str = "metric",
    treatment_label: str = "treatment",
    control_label: str = "control",
    min_per_arm: int = 2,
) -> tuple[bool, str]:
    """Detect a sign disagreement between the aggregate and every segment.

    Returns ``(reversal_present, explanation)``. A reversal is reported only when every
    qualifying segment agrees on a direction and the aggregate points the other way,
    which is the unambiguous case; mixed segment directions are reported as ordinary
    heterogeneity instead.
    """
    treatment_all = frame.loc[frame[variant_column] == treatment_label, metric_column]
    control_all = frame.loc[frame[variant_column] == control_label, metric_column]
    if treatment_all.size < min_per_arm or control_all.size < min_per_arm:
        return False, "Not enough data to compare aggregate against segments."

    aggregate = float(treatment_all.mean() - control_all.mean())

    segment_estimates: list[float] = []
    for _, group in frame.groupby(segment_column, sort=True):
        treatment = group.loc[group[variant_column] == treatment_label, metric_column]
        control = group.loc[group[variant_column] == control_label, metric_column]
        if treatment.size < min_per_arm or control.size < min_per_arm:
            continue
        segment_estimates.append(float(treatment.mean() - control.mean()))

    if len(segment_estimates) < 2:
        return False, "Fewer than two usable segments; no reversal check performed."

    all_positive = all(estimate > 0 for estimate in segment_estimates)
    all_negative = all(estimate < 0 for estimate in segment_estimates)

    if all_positive and aggregate < 0:
        return True, (
            "Simpson's paradox: the effect is positive in every segment but negative "
            "overall. The aggregate is a size-weighted average, so the treatment has "
            "most likely changed the segment mix. Check whether segment membership is "
            "itself downstream of the treatment."
        )
    if all_negative and aggregate > 0:
        return True, (
            "Simpson's paradox: the effect is negative in every segment but positive "
            "overall. The aggregate is a size-weighted average, so the treatment has "
            "most likely changed the segment mix. Check whether segment membership is "
            "itself downstream of the treatment."
        )
    if not (all_positive or all_negative):
        return False, (
            "Segments disagree in direction. This is ordinary heterogeneity, not a "
            "reversal, but the aggregate alone will not describe it."
        )
    return False, "Aggregate direction agrees with the segments."
