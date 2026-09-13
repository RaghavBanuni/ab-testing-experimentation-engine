"""CUPED: variance reduction using pre-experiment data.

The idea (Deng, Xu, Kohavi and Walker, 2013) is that most of the variance in a
per-user metric is not caused by the experiment at all -- it is the user being who
they already were. If a covariate ``X`` measured *before* assignment correlates with
the outcome ``Y``, then

    Y_adjusted = Y - theta * (X - mean(X))

has the same expectation as ``Y`` but smaller variance. Expanding,

    Var(Y_adj) = Var(Y) - 2*theta*Cov(Y, X) + theta**2 * Var(X)

which is minimised at ``theta = Cov(Y, X) / Var(X)``, and substituting back gives

    Var(Y_adj) = Var(Y) * (1 - rho**2)

where ``rho`` is the correlation between ``Y`` and ``X``. So a covariate correlated
at 0.7 removes about half the variance, which by the sample-size formula in
:mod:`ab_engine.power` roughly halves the traffic needed. This is usually the single
largest practical win available to an experimentation platform.

Two conditions matter, and both are easy to get wrong:

1. ``X`` must be measured **before** assignment. A covariate that the treatment can
   influence is not a covariate, and adjusting for it biases the estimate -- this is
   conditioning on a post-treatment variable.
2. ``theta`` must be estimated on the **pooled** sample, not per arm. Fitting a
   separate ``theta`` to each arm lets the adjustment absorb part of the treatment
   effect itself.

The estimator stays unbiased because ``E[X - mean(X)] = 0`` by construction, so
subtracting a multiple of it changes no expectation -- only the variance.
"""

from __future__ import annotations

import numpy as np


def cuped_theta(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """The variance-minimising coefficient ``Cov(Y, X) / Var(X)``.

    Estimate this on the pooled data from both arms. Returns 0.0 for a constant
    covariate, which correctly reduces the adjustment to a no-op.
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if outcome.shape != covariate.shape:
        raise ValueError("outcome and covariate must have the same shape")
    if outcome.size < 2:
        raise ValueError("need at least 2 observations")
    var_x = float(covariate.var(ddof=1))
    if var_x == 0.0:
        return 0.0
    covariance = float(np.cov(outcome, covariate, ddof=1)[0, 1])
    return covariance / var_x


def cuped_adjust(
    outcome: np.ndarray,
    covariate: np.ndarray,
    theta: float | None = None,
    covariate_mean: float | None = None,
) -> np.ndarray:
    """Return the CUPED-adjusted outcome.

    Pass ``theta`` and ``covariate_mean`` computed on the pooled sample when adjusting
    one arm at a time, so both arms are adjusted identically. Omitting them estimates
    both from the array given, which is only correct when that array *is* the pooled
    sample.
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if outcome.shape != covariate.shape:
        raise ValueError("outcome and covariate must have the same shape")
    if theta is None:
        theta = cuped_theta(outcome, covariate)
    if covariate_mean is None:
        covariate_mean = float(covariate.mean())
    return outcome - theta * (covariate - covariate_mean)


def variance_reduction_factor(outcome: np.ndarray, covariate: np.ndarray) -> float:
    """The predicted remaining variance fraction, ``1 - rho**2``.

    A return value of 1.0 means the covariate is useless; 0.0 would mean it explains
    the outcome perfectly. This is a prediction from the correlation, so compare it
    against the variance actually achieved -- they agree only up to estimation error
    in ``theta``.
    """
    outcome = np.asarray(outcome, dtype=float)
    covariate = np.asarray(covariate, dtype=float)
    if outcome.size < 2:
        raise ValueError("need at least 2 observations")
    if float(covariate.var(ddof=1)) == 0.0 or float(outcome.var(ddof=1)) == 0.0:
        return 1.0
    rho = float(np.corrcoef(outcome, covariate)[0, 1])
    return 1.0 - rho**2
