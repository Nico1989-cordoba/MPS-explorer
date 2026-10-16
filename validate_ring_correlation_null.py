# -*- coding: utf-8 -*-
"""
Calibration of the rotation null of the ring correlation
(``tools.mps_gaps.profile_correlation``) on synthetic rings: the public
validator its p-value is published with (gaps-rotation-null-pvalue).

What is checked
---------------
1. Level. Two INDEPENDENT coverage profiles of a stationary patch process
   around the ring (patches of random length at random places, the
   published magnitudes: ~20 % occupancy, patches of tens of nm on
   perimeters of micrometres) are correlated and tested; the share of
   p < 0.05 is the false-positive rate, printed with its Wilson 95 %
   interval. Under a stationary process the rotation of one profile is an
   exact symmetry of the null, so the rate must not exceed 5 % beyond its
   sampling error. The same with the bins masked out where both profiles
   share an unmeasured stretch, and in the exploratory angle frame's worst
   case, a stretch interpolated identically in both profiles (shared
   geometry, which the null cannot remove: the rate it reaches is printed,
   not judged).
2. Power. Two profiles of the same patches with independent jitter and
   independent missing patches give p < 0.05 nearly always.
3. The p-value's arithmetic: one-sided, (1 + b) / (1 + n_rot), the floor
   1 / (1 + n_rot), exhaustive over every shift when n_rot >= n - 1, and
   deterministic for a seed.

Synthetic only (R8): no data are read.

Run:  python validate_ring_correlation_null.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.mps_gaps import DEFAULT_N_ROTATIONS, profile_correlation  # noqa: E402

PASSED = 0
FAILED = 0

N_BINS = 3600
PERIMETER_NM = 20_000.0
MEAN_PATCH_NM = 60.0
OCCUPANCY = 0.20
REPLICATES = 400
ALPHA = 0.05


def check(name: str, fn) -> None:
    global PASSED, FAILED
    try:
        detail = fn()
    except Exception:  # noqa: BLE001 - a harness
        FAILED += 1
        print(f"  FAIL  {name}")
        for line in traceback.format_exc().strip().splitlines()[-3:]:
            print(f"        {line}")
        return
    PASSED += 1
    print(f"  ok    {name}" + (f"   [{detail}]" if detail else ""))


def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def patches(rng: np.random.Generator, centres_nm: np.ndarray, lengths_nm: np.ndarray) -> np.ndarray:
    """A binary coverage profile of the given patches on the circle."""
    bins = np.arange(N_BINS) * (PERIMETER_NM / N_BINS)
    out = np.zeros(N_BINS)
    for c, L in zip(centres_nm, lengths_nm):
        d = np.abs(((bins - c + PERIMETER_NM / 2) % PERIMETER_NM) - PERIMETER_NM / 2)
        out[d <= L / 2] = 1.0
    return out


def random_patches(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = rng.poisson(OCCUPANCY * PERIMETER_NM / MEAN_PATCH_NM)
    return rng.uniform(0, PERIMETER_NM, n), rng.exponential(MEAN_PATCH_NM, n) + 10.0


def level_independent() -> str:
    rng = np.random.default_rng(1)
    hits = 0
    for rep in range(REPLICATES):
        a = patches(rng, *random_patches(rng))
        b = patches(rng, *random_patches(rng))
        c = profile_correlation(a, b, seed=rep)
        hits += c.p_rotation < ALPHA
    lo, hi = wilson(hits, REPLICATES)
    rate = hits / REPLICATES
    assert lo <= ALPHA, (rate, lo, hi)
    return f"false positives {rate:.3f} (Wilson 95 % {lo:.3f}-{hi:.3f}) at nominal {ALPHA:g}, {REPLICATES} pairs"


def level_masked() -> str:
    rng = np.random.default_rng(2)
    hits = 0
    for rep in range(REPLICATES):
        a = patches(rng, *random_patches(rng))
        b = patches(rng, *random_patches(rng))
        start = int(rng.integers(0, N_BINS))
        hole = (np.arange(N_BINS) - start) % N_BINS < N_BINS // 10
        c = profile_correlation(a, b, seed=rep, measured_a=~hole, measured_b=~hole)
        hits += c.p_rotation < ALPHA
    lo, hi = wilson(hits, REPLICATES)
    assert lo <= ALPHA, (hits / REPLICATES, lo, hi)
    return f"masked 10 % unmeasured stretch: {hits / REPLICATES:.3f} (Wilson {lo:.3f}-{hi:.3f})"


def shared_interpolation() -> str:
    """A stretch filled identically in both profiles: what the angle frame's
    interpolation does where a contour does not reach an angle. Printed."""
    rng = np.random.default_rng(3)
    hits = 0
    for rep in range(REPLICATES):
        a = patches(rng, *random_patches(rng))
        b = patches(rng, *random_patches(rng))
        start = int(rng.integers(0, N_BINS))
        hole = (np.arange(N_BINS) - start) % N_BINS < N_BINS // 10
        ramp = np.linspace(0.0, 1.0, int(hole.sum()))
        a[hole] = ramp
        b[hole] = ramp
        hits += profile_correlation(a, b, seed=rep).p_rotation < ALPHA
    lo, hi = wilson(hits, REPLICATES)
    return (f"10 % interpolated identically in both: {hits / REPLICATES:.3f} (Wilson {lo:.3f}-{hi:.3f}); "
            "shared geometry inflates the null, which is why the arc frame is the default")


def power_aligned() -> str:
    rng = np.random.default_rng(4)
    hits = 0
    n = 100
    for rep in range(n):
        centres, lengths = random_patches(rng)
        keep_a = rng.random(centres.size) < 0.8
        keep_b = rng.random(centres.size) < 0.8
        a = patches(rng, centres[keep_a] + rng.normal(0, 10, keep_a.sum()), lengths[keep_a])
        b = patches(rng, centres[keep_b] + rng.normal(0, 10, keep_b.sum()), lengths[keep_b])
        hits += profile_correlation(a, b, seed=rep).p_rotation < ALPHA
    assert hits / n >= 0.9, hits / n
    return f"aligned patches (80 % kept each, 10 nm jitter): detected in {hits}/{n}"


def arithmetic() -> str:
    rng = np.random.default_rng(5)
    a = patches(rng, *random_patches(rng))
    c = profile_correlation(a, a)
    assert c.n_rotations == DEFAULT_N_ROTATIONS
    assert abs(c.p_rotation - 1.0 / (1 + DEFAULT_N_ROTATIONS)) < 1e-12, c.p_rotation
    b = patches(rng, *random_patches(rng))
    c1, c2 = profile_correlation(a, b, seed=7), profile_correlation(a, b, seed=7)
    assert c1.p_rotation == c2.p_rotation
    small = np.sin(np.linspace(0, 6 * np.pi, 64, endpoint=False))
    ce = profile_correlation(small, small, n_rotations=DEFAULT_N_ROTATIONS)
    assert ce.n_rotations == 63
    # every shift: the count over the 63 non-zero shifts plus rotation 0, over 64
    b_exact = int(np.count_nonzero(ce.correlation[1:] >= ce.r_at_zero))
    assert abs(ce.p_rotation - (1 + b_exact) / 64.0) < 1e-12
    return "floor 1/(1 + 1999); same seed, same p; exhaustive below n - 1 rotations"


def main() -> int:
    print("=" * 72)
    print("ROTATION NULL OF THE RING CORRELATION (synthetic)")
    print("=" * 72)
    check("level on independent stationary profiles", level_independent)
    check("level with a masked unmeasured stretch", level_masked)
    check("shared interpolated stretch (reported)", shared_interpolation)
    check("power on aligned patches", power_aligned)
    check("p-value arithmetic", arithmetic)
    print("\n" + "=" * 72)
    print(f"{PASSED} passed, {FAILED} failed")
    print("=" * 72)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
