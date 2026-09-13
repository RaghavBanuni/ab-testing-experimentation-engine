"""Tests that remain valid when you look more than once.

The problem
-----------
A fixed-horizon test controls the false-positive rate at ``alpha`` on the assumption
that it is evaluated exactly once, at a sample size committed to in advance. Nobody
works that way. Dashboards are refreshed daily, and the experiment is stopped when it
looks significant. Each additional look is another opportunity to cross the
threshold by chance, so the realised false-positive rate is strictly greater than
``alpha`` and grows with the number of looks. ``ab_engine.simulate`` measures this
directly rather than asserting it.

Approach 1: the mixture SPRT
----------------------------
Let ``delta_hat`` be the estimated difference in means with variance ``V``, and treat
``delta_hat ~ Normal(delta, V)``. Under the null ``delta = 0``. Rather than testing
against a single point alternative, mix over alternatives with a
``Normal(0, tau**2)`` prior. The resulting mixture likelihood ratio has a closed form:

    Lambda = sqrt(V / (V + tau**2)) * exp( delta_hat**2 * tau**2 / (2 * V * (V + tau**2)) )

Under the null this is a non-negative martingale with expectation 1, so Ville's
inequality bounds the probability that it *ever* exceeds ``1/alpha``:

    P( exists n : Lambda_n >= 1/alpha | H0 ) <= alpha

That bound holds simultaneously at every sample size, which is exactly the licence to
stop whenever you like. This is the mSPRT of Robbins (1970), in the form popularised
for online experiments by Johari, Koomen, Pekelis and Walsh (2017).

``tau`` is the analyst's guess at the scale of effects worth detecting. It trades
sensitivity across the range of possible effects: small ``tau`` detects small effects
sooner and large ones later. It does *not* affect validity -- any fixed ``tau``
chosen before looking at the data keeps the guarantee.

The caveat, stated plainly: the martingale argument assumes ``V`` is known. In
practice ``V`` is estimated from the sample, so the guarantee is asymptotic rather
than exact, and is unreliable at very small samples or on heavy-tailed metrics.

Approach 2: alpha spending
--------------------------
If the number of looks is known in advance, the alternative is to divide ``alpha``
across them. The O'Brien-Fleming style spending function of Lan and DeMets (1983)

    alpha(t) = 2 * (1 - Phi( z_{1-alpha/2} / sqrt(t) ))

gives the cumulative error permitted by information fraction ``t`` in ``(0, 1]``. It
spends almost nothing early and most of ``alpha`` at the end, which is what you want:
stopping early should require overwhelming evidence.

This module converts the spending function into a per-look nominal level by taking the
*increment* ``alpha(t_k) - alpha(t_{k-1})``. This is an approximation and it is
conservative: exact group-sequential boundaries account for the positive correlation
between successive looks (the recursive numerical integration of Armitage, McPherson
and Rowe, 1969) and are therefore slightly *less* strict than these. Using the
increments spends at most the intended total, so the realised false-positive rate is
at or below ``alpha`` -- never above it. Correctness is preserved; a little power is
given up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy import stats


@dataclass(frozen=True)
class SequentialResult:
    """An always-valid readout that may be consulted at any time."""

    estimate: float
    log_likelihood_ratio: float
    threshold: float
    ci_lower: float
    ci_upper: float

    @property
    def reject_null(self) -> bool:
        return self.log_likelihood_ratio >= self.threshold


def msprt_log_likelihood_ratio(estimate: float, variance: float, tau: float) -> float:
    """``log(Lambda)`` for the normal-mixture SPRT.

    Computed in logs because ``Lambda`` overflows quickly once the effect is real.
    """
    if variance <= 0:
        raise ValueError("variance must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    tau_sq = tau**2
    total = variance + tau_sq
    return 0.5 * math.log(variance / total) + (estimate**2 * tau_sq) / (
        2.0 * variance * total
    )


def msprt_confidence_sequence(
    estimate: float,
    variance: float,
    tau: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """An always-valid confidence interval.

    Inverting the test gives the set of ``theta`` not rejected. Solving

        0.5*log(V/(V+tau^2)) + (estimate-theta)^2 * tau^2 / (2 V (V+tau^2)) < log(1/alpha)

    for ``theta`` yields a symmetric interval of half-width

        sqrt( 2 V (V+tau^2)/tau^2 * ( log(1/alpha) + 0.5*log((V+tau^2)/V) ) ).

    Unlike a fixed-horizon interval, this one may be inspected continuously: the
    probability that it *ever* fails to cover the truth is at most ``alpha``. It is
    correspondingly wider.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if variance <= 0:
        raise ValueError("variance must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    tau_sq = tau**2
    total = variance + tau_sq
    radicand = (2.0 * variance * total / tau_sq) * (
        math.log(1.0 / alpha) + 0.5 * math.log(total / variance)
    )
    half_width = math.sqrt(radicand)
    return estimate - half_width, estimate + half_width


def msprt(
    estimate: float,
    variance: float,
    tau: float,
    alpha: float = 0.05,
) -> SequentialResult:
    """Evaluate the mSPRT at the current sample size."""
    log_lr = msprt_log_likelihood_ratio(estimate, variance, tau)
    lower, upper = msprt_confidence_sequence(estimate, variance, tau, alpha)
    return SequentialResult(
        estimate=estimate,
        log_likelihood_ratio=log_lr,
        threshold=math.log(1.0 / alpha),
        ci_lower=lower,
        ci_upper=upper,
    )


def obf_spending(information_fraction: float, alpha: float = 0.05) -> float:
    """Cumulative alpha permitted by information fraction ``t``.

    ``alpha(0)`` is 0 and ``alpha(1)`` is ``alpha``; in between it is convex, spending
    very little early.
    """
    if not 0.0 <= information_fraction <= 1.0:
        raise ValueError("information_fraction must be in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if information_fraction == 0.0:
        return 0.0
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(2.0 * stats.norm.sf(z_alpha / math.sqrt(information_fraction)))


def obf_nominal_alphas(
    information_fractions: Sequence[float],
    alpha: float = 0.05,
) -> list[float]:
    """Per-look nominal two-sided levels from the incremental spend.

    ``information_fractions`` must be strictly increasing and end at 1.0 for the full
    budget to be spent.
    """
    if not information_fractions:
        raise ValueError("at least one look is required")
    previous = 0.0
    increments: list[float] = []
    for fraction in information_fractions:
        if fraction <= previous - 1e-12:
            raise ValueError("information_fractions must be strictly increasing")
        cumulative = obf_spending(fraction, alpha)
        increments.append(max(cumulative - previous, 0.0))
        previous = cumulative
    return increments


def obf_critical_values(
    information_fractions: Sequence[float],
    alpha: float = 0.05,
) -> list[float]:
    """Two-sided critical ``|z|`` values corresponding to :func:`obf_nominal_alphas`.

    A look whose incremental spend rounds to zero gets ``inf``, meaning "cannot stop
    here" -- which is the correct reading, not an error.
    """
    criticals: list[float] = []
    for nominal in obf_nominal_alphas(information_fractions, alpha):
        if nominal <= 0.0:
            criticals.append(math.inf)
        else:
            criticals.append(float(stats.norm.ppf(1.0 - nominal / 2.0)))
    return criticals
