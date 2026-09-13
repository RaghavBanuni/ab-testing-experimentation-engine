"""Correcting for testing many metrics at once.

An experiment readout is rarely one number. Twenty metrics tested at the 5% level
give roughly a 64% chance of at least one false positive if all nulls are true
(``1 - 0.95**20``), which is why a dashboard of many metrics almost always contains
something that looks like a win.

Which correction to use depends on the decision:

- The **primary metric** (the OEC) is one pre-committed test. It needs no correction,
  because there is no multiplicity in a single pre-specified comparison.
- **Guardrail metrics** ask "did we break anything". Here a false *negative* is the
  expensive error, so correcting aggressively is wrong; some teams deliberately test
  guardrails at a looser level.
- **Exploratory metrics** are where multiplicity bites. Benjamini-Hochberg is the
  right default: it controls the expected *proportion* of false discoveries among the
  rejections, which is the quantity a triage process actually cares about, and it is
  far more powerful than controlling the probability of *any* false positive.
- Use **Holm-Bonferroni** when even one false positive is unacceptable -- for example
  a launch gate. It controls the family-wise error rate and is uniformly more
  powerful than plain Bonferroni, so there is no reason to use the latter.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def benjamini_hochberg(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up procedure controlling the false discovery rate.

    Sort the ``m`` p-values ascending and find the largest ``k`` with
    ``p_(k) <= k/m * alpha``; reject every hypothesis up to that rank. Rejecting
    everything below the largest passing rank -- rather than only those individually
    passing -- is what makes it a *step-up* procedure, and omitting that step is the
    most common implementation bug.

    Returns ``(rejected, adjusted)`` in the original input order. Adjusted values are
    the monotone-enforced ``m/k * p_(k)`` clipped to 1.0, so they can be compared
    directly against ``alpha``.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-D sequence")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("p_values must all lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Step-up: cumulative minimum from the largest rank downwards enforces monotonicity.
    adjusted_sorted = np.minimum.accumulate((m / ranks * sorted_p)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    passing = np.nonzero(sorted_p <= ranks / m * alpha)[0]
    rejected_sorted = np.zeros(m, dtype=bool)
    if passing.size:
        rejected_sorted[: passing[-1] + 1] = True

    rejected = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted
    return rejected, adjusted


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down procedure controlling the family-wise error rate.

    Compare the ``i``-th smallest p-value against ``alpha / (m - i + 1)`` and stop at
    the first failure; everything from there on is retained. Strictly more powerful
    than Bonferroni at the same guarantee.
    """
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("p_values must be a non-empty 1-D sequence")
    if np.any((p < 0.0) | (p > 1.0)) or np.any(np.isnan(p)):
        raise ValueError("p_values must all lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = np.arange(m, 0, -1, dtype=float)

    adjusted_sorted = np.maximum.accumulate(np.clip(sorted_p * multipliers, 0.0, 1.0))
    failures = np.nonzero(sorted_p > alpha / multipliers)[0]
    rejected_sorted = np.zeros(m, dtype=bool)
    stop = failures[0] if failures.size else m
    rejected_sorted[:stop] = True

    rejected = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted
    return rejected, adjusted
