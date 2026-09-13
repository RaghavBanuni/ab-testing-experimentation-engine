"""Generate a synthetic experiment CSV for the ``analyse`` subcommand.

Kept separate from :mod:`ab_engine.simulate` because the purpose is different: this
produces one experiment's worth of rows to exercise the readout path end to end,
including a pre-period covariate for CUPED and a segment column.

Usage::

    python -m ab_engine.make_dataset --out experiment.csv --true-effect 0.05
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .bucketing import assign

SEGMENTS = ("android", "ios", "web")


def make_experiment_frame(
    n_units: int = 20000,
    true_effect: float = 0.05,
    sigma: float = 1.0,
    covariate_correlation: float = 0.7,
    experiment_key: str = "demo-experiment",
    seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic experiment with a correlated pre-period covariate.

    The covariate is generated first and the outcome is built from it, so the
    correlation is a property of the construction rather than something fitted
    afterwards. The covariate is untouched by the treatment, which is what makes it a
    legitimate CUPED covariate.
    """
    if not -1.0 < covariate_correlation < 1.0:
        raise ValueError("covariate_correlation must be in (-1, 1)")
    rng = np.random.default_rng(seed)

    pre_metric = rng.normal(0.0, 1.0, n_units)
    noise = rng.normal(0.0, 1.0, n_units)
    rho = covariate_correlation
    base = rho * pre_metric + np.sqrt(1.0 - rho**2) * noise

    unit_ids = [f"user-{index}" for index in range(n_units)]
    variants = [
        assign(unit_id, experiment_key, ["control", "treatment"]).variant
        for unit_id in unit_ids
    ]
    is_treatment = np.array([variant == "treatment" for variant in variants])

    segment_index = rng.integers(0, len(SEGMENTS), n_units)
    segment_offset = np.array([0.0, 0.15, -0.1])[segment_index]

    metric = sigma * base + segment_offset + true_effect * is_treatment

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "variant": variants,
            "metric": metric,
            "pre_metric": pre_metric,
            "segment": [SEGMENTS[i] for i in segment_index],
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic experiment CSV.")
    parser.add_argument("--out", default="experiment.csv")
    parser.add_argument("--n-units", type=int, default=20000)
    parser.add_argument("--true-effect", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--covariate-correlation", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    frame = make_experiment_frame(
        n_units=args.n_units,
        true_effect=args.true_effect,
        sigma=args.sigma,
        covariate_correlation=args.covariate_correlation,
        seed=args.seed,
    )
    frame.to_csv(args.out, index=False)
    print(f"wrote {args.out} with {len(frame)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
