"""The ordinary single-look tests, kept explicitly as a baseline.

Nothing here is novel; it is included so the sequential methods have something to be
compared against, and so the simulation harness can demonstrate what happens when a
fixed-horizon test is checked repeatedly.

Welch's t-test is used rather than Student's because equal variances between arms is
an assumption nobody checks and the treatment frequently changes the variance as well
as the mean. Welch costs almost nothing when variances *are* equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    """A single-look readout.

    ``estimate`` is treatment minus control, so a positive number always means the
    treatment moved the metric up, whatever the metric is.
    """

    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_treatment: int
    n_control: int

    @property
    def significant(self) -> bool:
        """Only meaningful against a pre-committed alpha; see :mod:`ab_engine.sequential`."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0


def welch_t_test(
    treatment: np.ndarray,
    control: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided Welch t-test on a continuous metric."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    n_t, n_c = treatment.size, control.size
    if n_t < 2 or n_c < 2:
        raise ValueError("each arm needs at least 2 observations")

    mean_t, mean_c = float(treatment.mean()), float(control.mean())
    var_t = float(treatment.var(ddof=1))
    var_c = float(control.var(ddof=1))
    se = math.sqrt(var_t / n_t + var_c / n_c)
    estimate = mean_t - mean_c
    if se == 0.0:
        return TestResult(estimate, 0.0, 0.0, 1.0, estimate, estimate, n_t, n_c)

    # Welch-Satterthwaite degrees of freedom.
    df_num = (var_t / n_t + var_c / n_c) ** 2
    df_den = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    df = df_num / df_den
    statistic = estimate / se
    p_value = float(2.0 * stats.t.sf(abs(statistic), df))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return TestResult(
        estimate=estimate,
        standard_error=se,
        statistic=statistic,
        p_value=p_value,
        ci_lower=estimate - crit * se,
        ci_upper=estimate + crit * se,
        n_treatment=n_t,
        n_control=n_c,
    )


def two_proportion_z_test(
    successes_treatment: int,
    n_treatment: int,
    successes_control: int,
    n_control: int,
    alpha: float = 0.05,
) -> TestResult:
    """Two-sided z-test on a binary metric.

    The test statistic uses the *pooled* proportion, because under the null the two
    rates are equal and pooling is the more powerful estimator of the common rate. The
    confidence interval uses the *unpooled* standard error, because under the
    alternative the rates differ and pooling would misstate the precision. Mixing the
    two this way is deliberate and is the standard convention.
    """
    if n_treatment <= 0 or n_control <= 0:
        raise ValueError("both arms need at least one unit")
    if not 0 <= successes_treatment <= n_treatment:
        raise ValueError("successes_treatment out of range")
    if not 0 <= successes_control <= n_control:
        raise ValueError("successes_control out of range")

    p_t = successes_treatment / n_treatment
    p_c = successes_control / n_control
    estimate = p_t - p_c

    pooled = (successes_treatment + successes_control) / (n_treatment + n_control)
    se_pooled = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_treatment + 1.0 / n_control))
    se_unpooled = math.sqrt(
        p_t * (1.0 - p_t) / n_treatment + p_c * (1.0 - p_c) / n_control
    )

    if se_pooled == 0.0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = estimate / se_pooled
        p_value = float(2.0 * stats.norm.sf(abs(statistic)))

    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return TestResult(
        estimate=estimate,
        standard_error=se_unpooled,
        statistic=statistic,
        p_value=p_value,
        ci_lower=estimate - crit * se_unpooled,
        ci_upper=estimate + crit * se_unpooled,
        n_treatment=n_treatment,
        n_control=n_control,
    )
