"""Command line interface.

Four subcommands, matching the four things you do in order:

``power``     size the experiment before running it
``bucket``    inspect the assignment function
``analyse``   read out a finished (or running) experiment from a CSV
``simulate``  score the analysis methods themselves

Run ``python -m ab_engine.cli <subcommand> --help`` for the options of each.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .bucketing import assign
from .cuped import cuped_adjust, cuped_theta, variance_reduction_factor
from .diagnostics import srm_check
from .fixed_horizon import welch_t_test
from .power import mde_for_total_n, sample_size_proportion, total_sample_size
from .segments import analyse_segments, detect_simpson_reversal
from .sequential import msprt
from .simulate import simulate


def _cmd_power(args: argparse.Namespace) -> int:
    if args.baseline_rate is not None:
        if args.relative_lift is None:
            print("--relative-lift is required with --baseline-rate", file=sys.stderr)
            return 2
        total = sample_size_proportion(
            args.baseline_rate, args.relative_lift, args.alpha, args.power, args.allocation
        )
        print(f"binary metric, baseline rate {args.baseline_rate}, "
              f"relative lift {args.relative_lift}")
        print(f"total units needed: {total}")
        print(f"per arm (balanced):  {total // 2}")
        return 0

    if args.sigma is None:
        print("--sigma is required for a continuous metric", file=sys.stderr)
        return 2

    if args.total_n is not None:
        mde = mde_for_total_n(args.sigma, args.total_n, args.alpha, args.power, args.allocation)
        print(f"continuous metric, sigma {args.sigma}, total n {args.total_n}")
        print(f"minimum detectable effect: {mde:.6g}")
        print("If no plausible change moves the metric by at least this much, the "
              "experiment cannot succeed as designed.")
        return 0

    if args.mde is None:
        print("provide either --mde or --total-n", file=sys.stderr)
        return 2
    total = total_sample_size(args.sigma, args.mde, args.alpha, args.power, args.allocation)
    print(f"continuous metric, sigma {args.sigma}, mde {args.mde}")
    print(f"total units needed: {total}")
    return 0


def _cmd_bucket(args: argparse.Namespace) -> int:
    variants = args.variants.split(",")
    counts: Counter[str] = Counter()
    for index in range(args.n_units):
        assignment = assign(
            unit_id=f"{args.unit_prefix}{index}",
            experiment_key=args.experiment_key,
            variants=variants,
            exposure=args.exposure,
        )
        counts[assignment.variant if assignment.variant is not None else "<unexposed>"] += 1

    print(f"experiment_key={args.experiment_key!r} exposure={args.exposure}")
    print(f"units hashed: {args.n_units}")
    for name, count in sorted(counts.items()):
        print(f"  {name:<16}{count:>10}{count / args.n_units:>12.4f}")
    exposed = [c for name, c in counts.items() if name != "<unexposed>"]
    if len(exposed) >= 2:
        result = srm_check(exposed)
        print()
        print(f"SRM chi-square={result.chi_square:.4f} p={result.p_value:.6g}")
        print(result.explain())
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.csv)
    for column in (args.variant_column, args.metric_column):
        if column not in frame.columns:
            print(f"column {column!r} not found in {args.csv}", file=sys.stderr)
            return 2

    treatment_mask = frame[args.variant_column] == args.treatment_label
    control_mask = frame[args.variant_column] == args.control_label
    n_t, n_c = int(treatment_mask.sum()), int(control_mask.sum())
    if n_t < 2 or n_c < 2:
        print("each arm needs at least 2 rows", file=sys.stderr)
        return 2

    print("=== invalidation gates ===")
    srm = srm_check([n_t, n_c])
    print(f"arm sizes: treatment={n_t} control={n_c}")
    print(f"SRM chi-square={srm.chi_square:.4f} p={srm.p_value:.6g}")
    print(srm.explain())
    if srm.mismatch:
        print("\nStopping: the effect is not reported when the allocation gate fails.")
        return 1

    outcome_t = frame.loc[treatment_mask, args.metric_column].to_numpy(dtype=float)
    outcome_c = frame.loc[control_mask, args.metric_column].to_numpy(dtype=float)

    if args.covariate_column and args.covariate_column in frame.columns:
        print("\n=== CUPED ===")
        pooled_outcome = frame[args.metric_column].to_numpy(dtype=float)
        pooled_covariate = frame[args.covariate_column].to_numpy(dtype=float)
        theta = cuped_theta(pooled_outcome, pooled_covariate)
        covariate_mean = float(pooled_covariate.mean())
        predicted = variance_reduction_factor(pooled_outcome, pooled_covariate)
        print(f"theta (pooled) = {theta:.6g}")
        print(f"predicted remaining variance fraction (1 - rho^2) = {predicted:.6g}")
        outcome_t = cuped_adjust(
            outcome_t,
            frame.loc[treatment_mask, args.covariate_column].to_numpy(dtype=float),
            theta=theta,
            covariate_mean=covariate_mean,
        )
        outcome_c = cuped_adjust(
            outcome_c,
            frame.loc[control_mask, args.covariate_column].to_numpy(dtype=float),
            theta=theta,
            covariate_mean=covariate_mean,
        )
        print("Metric replaced by its CUPED-adjusted form for the tests below.")

    print("\n=== fixed-horizon readout (valid only at a pre-committed sample size) ===")
    fixed = welch_t_test(outcome_t, outcome_c, alpha=args.alpha)
    print(f"estimate={fixed.estimate:.6g} se={fixed.standard_error:.6g}")
    print(f"t={fixed.statistic:.4f} p={fixed.p_value:.6g}")
    print(f"{100 * (1 - args.alpha):.0f}% CI = [{fixed.ci_lower:.6g}, {fixed.ci_upper:.6g}]")

    print("\n=== always-valid readout (safe to inspect at any time) ===")
    sequential = msprt(
        estimate=fixed.estimate,
        variance=fixed.standard_error**2,
        tau=args.tau,
        alpha=args.alpha,
    )
    print(f"log likelihood ratio={sequential.log_likelihood_ratio:.4f} "
          f"threshold={sequential.threshold:.4f}")
    print(f"reject null: {sequential.reject_null}")
    print(f"always-valid CI = [{sequential.ci_lower:.6g}, {sequential.ci_upper:.6g}]")
    print("This interval is wider than the fixed-horizon one; that width is the price "
          "of being allowed to look whenever you like.")

    if args.segment_column and args.segment_column in frame.columns:
        print("\n=== segments (exploratory, BH-corrected) ===")
        effects = analyse_segments(
            frame,
            segment_column=args.segment_column,
            variant_column=args.variant_column,
            metric_column=args.metric_column,
            treatment_label=args.treatment_label,
            control_label=args.control_label,
            alpha=args.alpha,
        )
        if not effects:
            print("no segment had enough units in both arms")
        for effect in effects:
            flag = "*" if effect.significant_after_correction else " "
            print(f" {flag} {effect.segment:<20} estimate={effect.estimate:>12.6g} "
                  f"p={effect.p_value:.4g} p_adj={effect.adjusted_p_value:.4g} "
                  f"n_t={effect.n_treatment} n_c={effect.n_control}")
        reversal, explanation = detect_simpson_reversal(
            frame,
            segment_column=args.segment_column,
            variant_column=args.variant_column,
            metric_column=args.metric_column,
            treatment_label=args.treatment_label,
            control_label=args.control_label,
        )
        print()
        print(f"reversal detected: {reversal}")
        print(explanation)
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    report = simulate(
        n_experiments=args.n_experiments,
        n_per_arm=args.n_per_arm,
        n_looks=args.n_looks,
        true_effect=args.true_effect,
        sigma=args.sigma,
        alpha=args.alpha,
        tau=args.tau,
        seed=args.seed,
    )
    print(report.format_table())
    print()
    if report.is_null_scenario:
        print("With true_effect=0 the column above is the realised false-positive rate. "
              "Compare fixed_peeking against alpha, and against fixed_final.")
    else:
        print("With a non-zero true_effect the column above is realised power. "
              "Compare the mean stop fraction: sequential methods buy early stopping.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ab_engine",
        description="Experimentation engine with always-valid inference.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    power = subparsers.add_parser("power", help="sample size and minimum detectable effect")
    power.add_argument("--sigma", type=float, help="metric standard deviation (continuous)")
    power.add_argument("--mde", type=float, help="absolute effect to detect")
    power.add_argument("--total-n", type=int, help="solve for MDE at this total sample size")
    power.add_argument("--baseline-rate", type=float, help="baseline conversion rate (binary)")
    power.add_argument("--relative-lift", type=float, help="relative lift to detect (binary)")
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--power", type=float, default=0.8)
    power.add_argument("--allocation", type=float, default=0.5)
    power.set_defaults(func=_cmd_power)

    bucket = subparsers.add_parser("bucket", help="inspect deterministic assignment")
    bucket.add_argument("--experiment-key", default="demo-experiment")
    bucket.add_argument("--variants", default="control,treatment")
    bucket.add_argument("--n-units", type=int, default=100000)
    bucket.add_argument("--exposure", type=float, default=1.0)
    bucket.add_argument("--unit-prefix", default="user-")
    bucket.set_defaults(func=_cmd_bucket)

    analyse = subparsers.add_parser("analyse", help="read out an experiment from a CSV")
    analyse.add_argument("csv")
    analyse.add_argument("--variant-column", default="variant")
    analyse.add_argument("--metric-column", default="metric")
    analyse.add_argument("--covariate-column", default="pre_metric")
    analyse.add_argument("--segment-column", default="segment")
    analyse.add_argument("--treatment-label", default="treatment")
    analyse.add_argument("--control-label", default="control")
    analyse.add_argument("--alpha", type=float, default=0.05)
    analyse.add_argument("--tau", type=float, default=0.1)
    analyse.set_defaults(func=_cmd_analyse)

    sim = subparsers.add_parser("simulate", help="score the analysis methods themselves")
    sim.add_argument("--n-experiments", type=int, default=2000)
    sim.add_argument("--n-per-arm", type=int, default=2000)
    sim.add_argument("--n-looks", type=int, default=5)
    sim.add_argument("--true-effect", type=float, default=0.0)
    sim.add_argument("--sigma", type=float, default=1.0)
    sim.add_argument("--alpha", type=float, default=0.05)
    sim.add_argument("--tau", type=float, default=0.1)
    sim.add_argument("--seed", type=int, default=12345)
    sim.set_defaults(func=_cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
