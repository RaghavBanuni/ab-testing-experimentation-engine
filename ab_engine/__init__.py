"""An experimentation engine that treats the *analysis* as the thing most likely to be wrong.

The modules are deliberately small and independent so each statistical decision is
readable on its own:

- ``bucketing``    deterministic, stable, mutually-independent variant assignment
- ``power``        sample size and minimum detectable effect, computed *before* the test
- ``fixed_horizon``the ordinary one-look test, kept as an explicit baseline
- ``sequential``   always-valid tests (mSPRT, alpha spending) that survive peeking
- ``cuped``        pre-period variance reduction
- ``diagnostics``  sample ratio mismatch and other invalidation gates
- ``multiplicity`` Benjamini-Hochberg and Holm across a metric family
- ``segments``     heterogeneous effects, with a Simpson's-paradox guard
- ``simulate``     a harness that scores the analysis methods themselves
"""

from .bucketing import Assignment, assign, bucket_fraction, in_experiment
from .cuped import cuped_adjust, cuped_theta, variance_reduction_factor
from .diagnostics import SRMResult, srm_check
from .fixed_horizon import TestResult, two_proportion_z_test, welch_t_test
from .multiplicity import benjamini_hochberg, holm_bonferroni
from .power import mde_for_total_n, sample_size_per_arm, sample_size_proportion
from .sequential import (
    SequentialResult,
    msprt,
    msprt_confidence_sequence,
    obf_nominal_alphas,
    obf_spending,
)
from .segments import SegmentEffect, analyse_segments, detect_simpson_reversal

__all__ = [
    "Assignment",
    "SRMResult",
    "SegmentEffect",
    "SequentialResult",
    "TestResult",
    "analyse_segments",
    "assign",
    "benjamini_hochberg",
    "bucket_fraction",
    "cuped_adjust",
    "cuped_theta",
    "detect_simpson_reversal",
    "holm_bonferroni",
    "in_experiment",
    "mde_for_total_n",
    "msprt",
    "msprt_confidence_sequence",
    "obf_nominal_alphas",
    "obf_spending",
    "sample_size_per_arm",
    "sample_size_proportion",
    "srm_check",
    "two_proportion_z_test",
    "variance_reduction_factor",
    "welch_t_test",
]
