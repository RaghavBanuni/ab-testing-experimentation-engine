"""Sample size and minimum detectable effect.

This module exists to be run *before* an experiment starts. A test that is
underpowered for the effect the team hopes to find cannot succeed: it will most
likely produce a non-significant result regardless of whether the feature works, and
the team will then either ship on a hunch or run it again and peek.

All formulas assume the difference in means is approximately normal, which the
central limit theorem supplies at the sample sizes online experiments run at. For a
two-sided test at level ``alpha`` with power ``1 - beta``, detecting a true
difference ``d`` requires

    d = (z_{1-alpha/2} + z_{1-beta}) * sqrt(Var(delta_hat))

and for two independent arms with per-arm sizes ``n_t`` and ``n_c`` and common
standard deviation ``sigma``,

    Var(delta_hat) = sigma**2 * (1/n_t + 1/n_c).

Writing ``n_t = w * N`` and ``n_c = (1 - w) * N`` for total ``N`` and treatment
allocation ``w`` gives

    Var(delta_hat) = (sigma**2 / N) * (1/w + 1/(1-w)),

which is minimised at ``w = 0.5``. That is why a 50/50 split is the default: any
other allocation needs more total traffic for the same precision.
"""

from __future__ import annotations

import math

from scipy import stats


def _z(p: float) -> float:
    """Standard normal quantile."""
    return float(stats.norm.ppf(p))


def _allocation_factor(allocation: float) -> float:
    """``1/w + 1/(1-w)``, the price paid for an unbalanced split."""
    if not 0.0 < allocation < 1.0:
        raise ValueError(f"allocation must be in (0, 1), got {allocation}")
    return 1.0 / allocation + 1.0 / (1.0 - allocation)


def total_sample_size(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> int:
    """Total units needed across both arms to detect ``mde`` with ``power``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if mde <= 0:
        raise ValueError("mde must be positive")
    z_sum = _z(1.0 - alpha / 2.0) + _z(power)
    n = _allocation_factor(allocation) * (sigma**2) * (z_sum / mde) ** 2
    return int(math.ceil(n))


def sample_size_per_arm(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Per-arm sample size for a balanced two-arm test."""
    return int(math.ceil(total_sample_size(sigma, mde, alpha, power, 0.5) / 2.0))


def mde_for_total_n(
    sigma: float,
    total_n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> float:
    """The smallest effect detectable with ``total_n`` units -- the inverse problem.

    Use this when traffic is fixed, which it usually is. If the returned MDE is larger
    than any effect the change could plausibly produce, the experiment is not worth
    running as designed.
    """
    if total_n <= 0:
        raise ValueError("total_n must be positive")
    z_sum = _z(1.0 - alpha / 2.0) + _z(power)
    return z_sum * sigma * math.sqrt(_allocation_factor(allocation) / total_n)


def sample_size_proportion(
    baseline_rate: float,
    relative_lift: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> int:
    """Total sample size for a binary metric, expressed as a *relative* lift.

    Product teams state targets relatively ("a 2% lift in conversion"), so that is the
    input. The variance is taken at the average of the two rates, which is the usual
    convention and is slightly conservative for a lift.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be in (0, 1)")
    if relative_lift == 0:
        raise ValueError("relative_lift must be non-zero")
    treatment_rate = baseline_rate * (1.0 + relative_lift)
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError("implied treatment rate falls outside (0, 1)")
    absolute_mde = abs(treatment_rate - baseline_rate)
    pooled = (baseline_rate + treatment_rate) / 2.0
    sigma = math.sqrt(pooled * (1.0 - pooled))
    return total_sample_size(sigma, absolute_mde, alpha, power, allocation)
