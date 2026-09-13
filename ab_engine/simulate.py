"""A harness that measures the analysis methods instead of trusting them.

This is the centrepiece of the repository. Every claim made in
:mod:`ab_engine.sequential` is checkable here, on synthetic data where the true effect
is known by construction:

- Set ``true_effect=0`` and the rejection rate *is* the realised false-positive rate.
- Set ``true_effect`` non-zero and the rejection rate *is* the realised power.

Four analysis strategies are compared on identical data:

``fixed_final``
    One test, at the pre-committed sample size. The correct use of a fixed-horizon
    test, and the baseline everything else is judged against.
``fixed_peeking``
    The same test evaluated at every look, stopping at the first significant result.
    This is what teams actually do, and its false-positive rate should exceed the
    nominal level -- increasingly so with more looks.
``msprt``
    The mixture SPRT, valid at every stopping time.
``alpha_spending``
    O'Brien-Fleming style incremental spending across a fixed number of looks.

The simulation draws normal outcomes, so it isolates the effect of the *stopping rule*
from any question about distributional robustness. That is the intent: the peeking
problem is not a heavy-tails problem, and it does not go away on well-behaved data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .sequential import msprt_log_likelihood_ratio, obf_critical_values

METHODS = ("fixed_final", "fixed_peeking", "msprt", "alpha_spending")


@dataclass(frozen=True)
class MethodOutcome:
    """How one analysis strategy behaved across many simulated experiments."""

    name: str
    rejections: int
    n_experiments: int
    mean_stop_fraction: float

    @property
    def rejection_rate(self) -> float:
        """False-positive rate when the true effect is zero; power otherwise."""
        return self.rejections / self.n_experiments


@dataclass(frozen=True)
class SimulationReport:
    """The result of a full sweep."""

    true_effect: float
    sigma: float
    n_per_arm: int
    n_looks: int
    alpha: float
    tau: float
    n_experiments: int
    outcomes: tuple[MethodOutcome, ...]

    @property
    def is_null_scenario(self) -> bool:
        return self.true_effect == 0.0

    def outcome(self, name: str) -> MethodOutcome:
        for candidate in self.outcomes:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no outcome named {name!r}")

    def format_table(self) -> str:
        label = "false positive rate" if self.is_null_scenario else "power"
        header = (
            f"true_effect={self.true_effect}  sigma={self.sigma}  "
            f"n_per_arm={self.n_per_arm}  looks={self.n_looks}  "
            f"alpha={self.alpha}  tau={self.tau}  "
            f"experiments={self.n_experiments}"
        )
        lines = [
            header,
            "",
            f"{'method':<16}{label:>22}{'mean stop fraction':>22}",
            "-" * 60,
        ]
        for outcome in self.outcomes:
            lines.append(
                f"{outcome.name:<16}{outcome.rejection_rate:>22.4f}"
                f"{outcome.mean_stop_fraction:>22.4f}"
            )
        return "\n".join(lines)


def _look_sizes(n_per_arm: int, n_looks: int) -> list[int]:
    """Equally spaced cumulative per-arm sample sizes, ending exactly at ``n_per_arm``."""
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if n_per_arm < 2 * n_looks:
        raise ValueError("n_per_arm must be at least 2 * n_looks so every look has data")
    sizes = [int(round(n_per_arm * (k + 1) / n_looks)) for k in range(n_looks)]
    sizes[-1] = n_per_arm
    return sizes


def _prefix_mean_and_var(values: np.ndarray, sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and unbiased variance of each prefix, computed once via cumulative sums.

    Recomputing the variance from scratch at every look would make the sweep
    quadratic in the number of looks for no reason.
    """
    cumulative_sum = np.cumsum(values)
    cumulative_square = np.cumsum(values**2)
    index = np.asarray(sizes) - 1
    n = np.asarray(sizes, dtype=float)
    total = cumulative_sum[index]
    total_square = cumulative_square[index]
    mean = total / n
    variance = (total_square - n * mean**2) / (n - 1.0)
    return mean, np.maximum(variance, 0.0)


def simulate(
    n_experiments: int = 2000,
    n_per_arm: int = 2000,
    n_looks: int = 5,
    true_effect: float = 0.0,
    sigma: float = 1.0,
    alpha: float = 0.05,
    tau: float = 0.1,
    seed: int = 12345,
) -> SimulationReport:
    """Run the sweep and return realised rejection rates for every method.

    ``tau`` is the mSPRT mixing scale and should be set to the order of magnitude of
    effects worth detecting, in the same units as the metric.
    """
    if n_experiments < 1:
        raise ValueError("n_experiments must be at least 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    sizes = _look_sizes(n_per_arm, n_looks)
    information_fractions = [size / n_per_arm for size in sizes]
    spending_criticals = obf_critical_values(information_fractions, alpha)
    fixed_critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    log_threshold = float(np.log(1.0 / alpha))

    rng = np.random.default_rng(seed)
    rejections = {name: 0 for name in METHODS}
    stop_fractions: dict[str, list[float]] = {name: [] for name in METHODS}

    for _ in range(n_experiments):
        control = rng.normal(0.0, sigma, n_per_arm)
        treatment = rng.normal(true_effect, sigma, n_per_arm)
        mean_c, var_c = _prefix_mean_and_var(control, sizes)
        mean_t, var_t = _prefix_mean_and_var(treatment, sizes)

        n = np.asarray(sizes, dtype=float)
        estimate = mean_t - mean_c
        variance = var_t / n + var_c / n
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(variance > 0, estimate / np.sqrt(variance), 0.0)

        # fixed_final: the single pre-committed look.
        if abs(z[-1]) > fixed_critical:
            rejections["fixed_final"] += 1
        stop_fractions["fixed_final"].append(1.0)

        # fixed_peeking: stop at the first look that crosses the uncorrected threshold.
        stopped = False
        for k in range(n_looks):
            if abs(z[k]) > fixed_critical:
                rejections["fixed_peeking"] += 1
                stop_fractions["fixed_peeking"].append(information_fractions[k])
                stopped = True
                break
        if not stopped:
            stop_fractions["fixed_peeking"].append(1.0)

        # msprt: always-valid, so stopping at the first crossing is legitimate.
        stopped = False
        for k in range(n_looks):
            if variance[k] <= 0:
                continue
            log_lr = msprt_log_likelihood_ratio(float(estimate[k]), float(variance[k]), tau)
            if log_lr >= log_threshold:
                rejections["msprt"] += 1
                stop_fractions["msprt"].append(information_fractions[k])
                stopped = True
                break
        if not stopped:
            stop_fractions["msprt"].append(1.0)

        # alpha_spending: per-look boundaries from the incremental spend.
        stopped = False
        for k in range(n_looks):
            if abs(z[k]) > spending_criticals[k]:
                rejections["alpha_spending"] += 1
                stop_fractions["alpha_spending"].append(information_fractions[k])
                stopped = True
                break
        if not stopped:
            stop_fractions["alpha_spending"].append(1.0)

    outcomes = tuple(
        MethodOutcome(
            name=name,
            rejections=rejections[name],
            n_experiments=n_experiments,
            mean_stop_fraction=float(np.mean(stop_fractions[name])),
        )
        for name in METHODS
    )
    return SimulationReport(
        true_effect=true_effect,
        sigma=sigma,
        n_per_arm=n_per_arm,
        n_looks=n_looks,
        alpha=alpha,
        tau=tau,
        n_experiments=n_experiments,
        outcomes=outcomes,
    )
