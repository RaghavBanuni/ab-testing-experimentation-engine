"""Deterministic variant assignment.

Why hashing rather than a random number generator: assignment must be a *pure
function* of the unit and the experiment. A unit that returns tomorrow must land in
the same variant, without the assignment being stored anywhere, and without any
shared state between the servers doing the assigning. Hashing gives all of that for
free; ``random.choice`` gives none of it.

Why a per-experiment salt: two experiments running at once must assign
independently. If both hashed only the unit id, every unit in variant A of
experiment 1 would also be in variant A of experiment 2, and the two experiments
would be perfectly confounded. Salting with the experiment key decorrelates them.

Why a *separate* salt for the exposure decision: ramping an experiment to 10% of
traffic and then splitting that 10% between variants are two different decisions. If
both used the same hash, the variant split would be taken from the *low* end of the
hash range and would not be balanced. Deriving them from different salts keeps them
independent.

SHA-256 is used because it is uniformly distributed enough for this purpose and is
stable across processes, machines and Python versions -- unlike :func:`hash`, which
is randomised per interpreter run by default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

_UINT64_SPAN = 2**64


def bucket_fraction(unit_id: str, salt: str) -> float:
    """Map ``(unit_id, salt)`` to a stable fraction in ``[0, 1)``.

    The first 8 bytes of the digest are read as a big-endian unsigned integer and
    divided by ``2**64``, so the result is uniform on ``[0, 1)`` to the precision of a
    float and is identical on every machine.
    """
    digest = hashlib.sha256(f"{salt}:{unit_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / _UINT64_SPAN


def in_experiment(unit_id: str, experiment_key: str, exposure: float) -> bool:
    """Whether a unit is exposed to the experiment at all, at the given ramp.

    ``exposure`` is the fraction of traffic admitted, in ``[0, 1]``. Because this uses
    its own salt namespace, ramping exposure up admits a *superset* of the units
    previously admitted, so a ramp never reshuffles anybody.
    """
    if not 0.0 <= exposure <= 1.0:
        raise ValueError(f"exposure must be in [0, 1], got {exposure}")
    return bucket_fraction(unit_id, f"{experiment_key}:exposure") < exposure


@dataclass(frozen=True)
class Assignment:
    """The outcome of assigning one unit."""

    unit_id: str
    variant: str | None
    fraction: float
    exposed: bool


def assign(
    unit_id: str,
    experiment_key: str,
    variants: Sequence[str],
    weights: Sequence[float] | None = None,
    exposure: float = 1.0,
) -> Assignment:
    """Assign one unit to a variant.

    ``weights`` need not sum to 1; they are normalised. A unit not admitted by the
    exposure ramp gets ``variant=None`` and must be excluded from analysis entirely --
    not folded into control, which would dilute the effect.
    """
    if not variants:
        raise ValueError("variants must not be empty")
    if weights is None:
        weights = [1.0] * len(variants)
    if len(weights) != len(variants):
        raise ValueError("weights and variants must be the same length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    exposed = in_experiment(unit_id, experiment_key, exposure)
    fraction = bucket_fraction(unit_id, f"{experiment_key}:variant")
    if not exposed:
        return Assignment(unit_id=unit_id, variant=None, fraction=fraction, exposed=False)

    threshold = 0.0
    for variant, weight in zip(variants, weights):
        threshold += weight / total
        if fraction < threshold:
            return Assignment(unit_id, variant, fraction, True)
    # Only reachable through floating-point accumulation at fraction ~= 1.0.
    return Assignment(unit_id, variants[-1], fraction, True)
