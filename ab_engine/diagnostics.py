"""Invalidation gates -- checks that decide whether a readout may be believed at all.

These run *before* any effect is reported. The distinction matters: a significance
test answers "did the metric move", whereas these answer "is this experiment
trustworthy". A failing gate is not a negative result, it is a broken experiment, and
the correct response is to fix the instrumentation and discard the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scipy import stats


@dataclass(frozen=True)
class SRMResult:
    """Outcome of a sample ratio mismatch check."""

    observed: tuple[int, ...]
    expected: tuple[float, ...]
    chi_square: float
    degrees_of_freedom: int
    p_value: float
    mismatch: bool

    def explain(self) -> str:
        if not self.mismatch:
            return "No sample ratio mismatch detected; allocation is consistent with design."
        return (
            "Sample ratio mismatch: observed allocation is inconsistent with the "
            "designed split. Do not interpret the treatment effect. The usual causes "
            "are a bug in assignment, differential bot filtering, redirect or "
            "latency loss on one arm, or logging that drops events for one variant."
        )


def srm_check(
    observed_counts: Sequence[int],
    expected_ratios: Sequence[float] | None = None,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test on arm sizes.

    ``alpha`` defaults to 0.001, far stricter than a metric test, and deliberately so.
    With large samples this test has enormous power, so a 5% level would fire
    constantly on trivial imbalances; conversely a genuine bucketing bug produces a
    p-value many orders of magnitude below any threshold. The consequence of firing is
    also asymmetric: a false alarm costs an investigation, while a missed mismatch
    invalidates every conclusion drawn from the experiment.

    A mismatch is *not* evidence about the treatment effect. It is evidence that the
    units being compared are not the units that were assigned, which breaks the
    randomisation the whole analysis rests on.
    """
    observed = [int(c) for c in observed_counts]
    if len(observed) < 2:
        raise ValueError("need at least two arms")
    if any(c < 0 for c in observed):
        raise ValueError("counts must be non-negative")
    total = sum(observed)
    if total == 0:
        raise ValueError("no units observed")

    if expected_ratios is None:
        expected_ratios = [1.0 / len(observed)] * len(observed)
    if len(expected_ratios) != len(observed):
        raise ValueError("expected_ratios and observed_counts must be the same length")
    ratio_total = float(sum(expected_ratios))
    if ratio_total <= 0:
        raise ValueError("expected_ratios must sum to a positive number")
    expected = [total * r / ratio_total for r in expected_ratios]
    if any(e <= 0 for e in expected):
        raise ValueError("every arm must have a positive expected count")

    chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
    dof = len(observed) - 1
    p_value = float(stats.chi2.sf(chi_square, dof))
    return SRMResult(
        observed=tuple(observed),
        expected=tuple(expected),
        chi_square=float(chi_square),
        degrees_of_freedom=dof,
        p_value=p_value,
        mismatch=p_value < alpha,
    )
