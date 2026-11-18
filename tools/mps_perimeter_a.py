"""
Question A of the perimeter density analysis: how is the density arranged around its maximum? (package P7)

The question
------------
Along the membrane of one axon slice the localization density has a maximum ("the large cluster"). Is the
rest of the density (a) piled up next to it (an ATTRACTOR, e.g. a Gaussian halo), (b) kept away from it and/or
laid out at regular intervals counted from it (a periodic REPULSOR: an inhibition zone, regular modes), or
(c) indifferent to where the maximum is (FLAT: no structure)?

The trap
--------
Every profile is aligned on its own maximum. Lag 0 is then 1 by construction and its neighbours are high only
because a clump spans more than one bin. Pooling aligned profiles shows a peak at 0 even when nothing is there.
A test that compares the aligned profile against anything that was NOT aligned the same way finds that peak
every time. Three things are done about it, and all are needed:

1. The maximum and its flanks (the EXCLUSION ZONE) enter no statistic. The zone runs from the maximum down to
   the BASELINE of the profile and half as far again (``exclusion_zone``, rule "baseline"); stopping at half
   height leaves the flank of any cluster wider than one bin in the remainder, and that flank reads as
   attraction every time once axons are pooled (review R2: 18 pooled axons, cluster of sd 150 nm, no attraction,
   "attractor" in 20 of 20 sets).
2. The null REPEATS THE ALIGNMENT. Null hypothesis: the position of the rest of the density is independent
   of the position of the maximum. A replicate keeps the maximum and its zone where they are and shifts the
   REMAINDER (the arc outside the zone) circularly inside its own arc by a number of bins (toroidal shift,
   Lotwick & Silverman 1982 [lit, not re-read for this module]). It keeps the number of localizations, the
   perimeter, the maximum, its shape and the autocorrelation of the remainder (blinking clumps included); it
   breaks only the relative position. No remainder value exceeds the maximum, so the maximum is still a
   maximum in every replicate: the alignment is repeated by construction and the selection bias is identical
   in the observed profile and in the null.
3. A pooled reading is only given when it SURVIVES WIDENING THE ZONE (2 and 3 times, ``ROBUST_WIDEN``). An
   excess that is there with the default zone and gone with a wider one is the flank of the cluster.

   A null made of uniformly re-thrown localizations (``uniform_realigned_null``) is provided as a DIAGNOSTIC
   of the artefact only. It is not used for any readout because it also destroys the clumping of the
   remainder, and clumping alone would then read as structure.

The two statistics (the same code on the observed profile and on every replicate)
---------------------------------------------------------------------------------
Both are computed on the remainder DIVIDED BY ITS OWN MEAN (a constant under the shift, so the null is
untouched). The relative density (counts / maximum) of an axon with a small cluster is high and that of an axon
with a large one is low; without this step the pooled statistic follows whichever axons hold each distance and
its null is not centred on zero (review R2).

``contrast``  mean density within ``near_nm`` of the zone minus the mean of the rest of the remainder, in units
              of the remainder's mean (+0.2 = 20 % more density near than far). Two-sided. Its SIGN is always
              read against the null (``value - null_mean``), never from the raw value.
              Pooled: the mean over axons of the per-axon contrast, each measured from its own zone.
``period``    largest PHASE-LOCKED periodogram power of the folded remainder over the scanned periods: the
              variance carried by a cosine whose crests sit at multiples of the period counted from the
              maximum. It is NOT divided by the total variance: under the pooled null the total variance falls
              together with the locked part, and the ratio loses the contrast between them (measured: z 2.6
              instead of p at the floor on the regular control). One-sided. The maximum over periods is taken
              in every replicate too, so there is no hidden multiple comparison.
              Pooled: on the mean over axons of the folded remainders, on the nm axis.

              Why phase-locked, and not the plain periodogram of the remainder the plan first wrote: the
              modulus of the discrete Fourier transform of a sequence is INVARIANT under a circular shift of
              that sequence. Against the shift null the plain periodogram peak equals itself in every
              replicate and p is 1 whatever the data (``test_plain_periodogram_is_blind_to_the_shift_null``).
              What the shift null can test is whether the modes are placed relative to the maximum, which is
              the question asked. Whether the remainder is periodic regardless of the maximum is a different
              question with a different null, and is not answered here.

              Consequence for ONE axon: a remainder of period T looks almost the same after a shift by T,
              so a good share of the replicates reproduce the observed statistic and the per-axon test has
              little power. The per-axon ``period`` is a descriptive z (the null of a maximum is skewed: do
              not feed it to a t-test without looking); the test with power is the pooled one.

Readings (descriptive labels; the numbers are what is reported)
---------------------------------------------------------------
"attractor"        pooled contrast above its null, significant after Holm, AND still above and significant
                   (raw p) with the zone widened 2 and 3 times.
"inhibition zone"  pooled contrast below its null, significant after Holm.
"regular"          pooled period significant after Holm, the peak not at the longest scanned period, AND still
                   so with the widened zones. A maximum at the low-frequency edge is a trend (a halo or a
                   depleted zone), not a repeat; it is already read by the contrast.
"... (not robust to the exclusion zone: cluster flank)"  significant with the default zone only.
"no structure detectable"  none of the above, always with the number of axons.
A periodic repulsor reads "inhibition zone + regular" or "regular".

What "attractor" means, and what it does not
--------------------------------------------
"attractor" means only: more density near the edge of the exclusion zone than far from it, compared with
shifting the rest of the profile. A cluster wider than the zone gives this same reading with no attraction at
all. The reading holds only if it is kept when the zone is widened beyond the width of the cluster (the
sensitivity table in ``PooledDetails.sensitivity``). A halo and the shape of the cluster differ ONLY BY SCALE:
what the programme calls an attractor is excess BEYOND the edge of the cluster as the zone rule defines it. A
cluster with a heavy tail (e.g. a Student t with 3 degrees of freedom) still holds a few per cent of its
localizations a micron away; that is, by any measurable property, a halo, and it reads as one. It is a test
within these N axons; it does not generalise to animals. Exploratory.

Scope of the pooled p
---------------------
It is a randomization test WITHIN these axons (each axon is its own stratum). It does not generalise to the
population of animals: the unit of inference between groups is the animal, and that lives in
``mps_perimeter_pool``. With no animal the group is labelled "SIN ANIMAL"; with no animal or one animal the
reading STARTS with "EXPLORATORY, within-sample". No p between axons, animals or genotypes is produced here.
The per-axon p values are descriptive and carry no correction.

The global band is for display
------------------------------
``PooledProfile.global_lo/hi`` is a valid simultaneous band, but "the profile leaves the band" is a third test
outside the Holm family of the two statistics (and a fourth one on the fraction axis). The decision is the two
statistics; the band is drawn so the reader sees where the profile departs. It is given only at distances held
by at least ``GLOBAL_MIN_AXONS`` axons.

Analyst's choices to freeze, with a date, before new data (rule of the statistical plan): ``NEAR_NM``,
``PERIOD_MIN_NM``, the zone rule and its three numbers, ``ROBUST_WIDEN``, the axis and the weights.

Known limits
------------
* Zone versus halo, above.
* The toroidal shift joins the two ends of the remainder. The seam makes the null of ``contrast`` slightly
  wider than the observed statistic for a clumpy remainder (conservative direction).
* If the remainder holds a bin tied with the maximum (``DensityProfile.max_tie``) a replicate can put that tie
  ahead of the maximum; the maximum bin is still a maximum, not necessarily the first one.
* On the nm grid the effective bins (99-101 nm) of different axons do not line up exactly: now and then two
  lags of one axon share a cell and the next cell is empty for it (``n_axons`` can dip by one). Cosmetic.

Pure functions, numpy only, deterministic given the seed. No Qt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from tools.mps_perimeter_types import DensityProfile, PooledProfile, PooledResult, TestReadout

# ----------------------------------------------------------------------------------------------------
# Constants (origin tags as in the contract)
# ----------------------------------------------------------------------------------------------------
NEAR_NM: float = 1000.0          # [ad hoc] "near the large cluster" = within 1 um of the zone; to be swept in sim
PERIOD_MIN_NM: float = 300.0     # [ad hoc] shortest scanned period = 3 nominal bins (Nyquist of a 100 nm bin is 200)
EXCLUSION_MIN_NM: float = 200.0  # [ad hoc] = contract A_EXCLUSION_MIN_NM: at least +-2 nominal bins excluded
MIN_REMAINDER_BINS: int = 8      # [ad hoc] fewer remainder bins than this: no statistic (as the 8-bin floor of the grid)
ALPHA: float = 0.05              # [lit] conventional level; Holm (1979) between the two pooled statistics
POINTWISE_PCT: Tuple[float, float] = (2.5, 97.5)   # [lit] conventional 95 % pointwise band
GLOBAL_MIN_AXONS: int = 3        # [ad hoc] the global band is drawn where at least this many axons contribute
NO_ANIMAL_GROUP: str = "SIN ANIMAL"   # label of the statistical plan for axons with no animal assigned

# Zone rule "baseline": the zone reaches the first bin below median + max(FRAC * height, K * noise) and then
# WIDEN times as far. [own sim, 2026-11-16: the sweep is in test_perimeter_a.py and in the package report]
BASELINE_HEIGHT_FRAC: float = 0.10   # of the height of the maximum above the median
BASELINE_NOISE_K: float = 2.0        # robust sd of the profile (1.4826 MAD): never chase the tail into the noise
ZONE_WIDEN: float = 1.5              # the tail below the threshold is not zero: half as far again
ROBUST_WIDEN: Tuple[float, ...] = (2.0, 3.0)   # [own sim] zones a pooled reading has to survive
DEFAULT_RULE: str = "baseline"
ZONE_RULES: Tuple[str, ...] = ("baseline", "half_max", "fixed")
POOL_AXES: Tuple[str, ...] = ("nm", "fraction")
POOL_WEIGHTS: Tuple[str, ...] = ("equal", "animal")

# Vocabulary of PooledResult.reading (the label; see ``reading_label``)
READING_ATTRACTOR: str = "attractor"
READING_INHIBITION: str = "inhibition zone"
READING_REGULAR: str = "regular"
READING_NONE: str = "no structure detectable"
READING_NOT_COMPUTABLE: str = "not computable"
READING_NOT_ROBUST: str = " (not robust to the exclusion zone: cluster flank)"   # suffix of attractor / regular
READING_JOIN: str = " + "
PREFIX_EXPLORATORY: str = "EXPLORATORY, within-sample: "       # no animal, or a single animal
PREFIX_WITHIN_SAMPLE: str = "WITHIN-SAMPLE (not an animal-level test): "
READING_SEPARATOR: str = " | "


# ----------------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------------
def _relative(p: DensityProfile) -> NDArray[np.float64]:
    """Relative density with NaN where the bin was not measured."""
    rel = np.asarray(p.relative, dtype=np.float64).ravel().copy()
    measured = np.asarray(p.measured, dtype=bool).ravel()
    if measured.size == rel.size:
        rel[~measured] = np.nan
    return cast(NDArray[np.float64], rel)


def _zone_bins(zone: Tuple[int, int], n: int) -> NDArray[np.int64]:
    """Bin indices (modulo n) of an unwrapped zone (first, last)."""
    first, last = int(zone[0]), int(zone[1])
    return np.arange(first, last + 1, dtype=np.int64) % n


def _remainder_bins(zone: Tuple[int, int], n: int) -> NDArray[np.int64]:
    """Bin indices of the remainder, in arc order from just after the zone to just before it."""
    first, last = int(zone[0]), int(zone[1])
    m = n - (last - first + 1)
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.arange(last + 1, last + 1 + m, dtype=np.int64) % n


def _fold(rows: NDArray[np.float64], max_bin: int) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fold rows (R, n) about ``max_bin``: value and number of contributing bins (0, 1 or 2) at lags 0..n//2."""
    n = rows.shape[-1]
    k = np.arange(n // 2 + 1)
    ia = (max_bin + k) % n
    ib = (max_bin - k) % n
    same = ia == ib                              # lag 0, and the antipode when n is even
    a = rows[:, ia]
    b = rows[:, ib]
    fa = np.isfinite(a)
    fb = np.isfinite(b) & ~same[None, :]
    count = fa.astype(np.float64) + fb.astype(np.float64)
    total = np.where(fa, a, 0.0) + np.where(fb, b, 0.0)
    value = np.full(total.shape, np.nan)
    np.divide(total, count, out=value, where=count > 0)
    return value, count


def _contrast_rows(folded: NDArray[np.float64], weights: NDArray[np.float64],
                   lags_nm: NDArray[np.float64], near_nm: float) -> NDArray[np.float64]:
    """Near-minus-far weighted mean of every row (R, H). NaN when either side is empty."""
    ok = np.isfinite(folded) & (weights > 0)
    lag = np.broadcast_to(lags_nm[None, :], folded.shape)
    first = np.min(np.where(ok, lag, np.inf), axis=1)          # first lag outside the zone
    near = ok & (lag < first[:, None] + near_nm)
    far = ok & ~near
    x = np.where(ok, folded, 0.0) * weights
    w_near = np.sum(weights * near, axis=1)
    w_far = np.sum(weights * far, axis=1)
    out = np.full(folded.shape[0], np.nan)
    good = (w_near > 0) & (w_far > 0)
    out[good] = np.sum(x * near, axis=1)[good] / w_near[good] - np.sum(x * far, axis=1)[good] / w_far[good]
    return out


def _period_grid(span_nm: float, period_range_nm: Tuple[float, Optional[float]]) -> NDArray[np.float64]:
    """Scanned periods, longest first: 2 * span / k (the cosine basis of the folded interval [0, span])."""
    lo = float(period_range_nm[0])
    hi = float(span_nm) if period_range_nm[1] is None else float(period_range_nm[1])
    if not (span_nm > 0 and lo > 0 and hi >= lo):
        return np.zeros(0)
    k_min = max(1, int(np.ceil(2.0 * span_nm / hi - 1e-9)))
    k_max = int(np.floor(2.0 * span_nm / lo + 1e-9))
    if k_max < k_min:
        return np.zeros(0)
    periods: NDArray[np.float64] = 2.0 * span_nm / np.arange(k_min, k_max + 1, dtype=np.float64)
    return periods


def _power_rows(folded: NDArray[np.float64], weights: NDArray[np.float64],
                lags_nm: NDArray[np.float64], periods_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Phase-locked power (R, P): variance of each row carried by cos(2 pi lag / period) (weighted least squares:
    amplitude squared times the variance of the cosine over the lags present)."""
    ok = np.isfinite(folded) & (weights > 0)
    w = np.where(ok, weights, 0.0)
    x = np.where(ok, folded, 0.0)
    sw = np.sum(w, axis=1, keepdims=True)
    sw_safe = np.where(sw > 0, sw, 1.0)
    xc = (x - np.sum(w * x, axis=1, keepdims=True) / sw_safe) * ok
    basis = np.cos(2.0 * np.pi * lags_nm[:, None] / periods_nm[None, :])          # (H, P)
    num = ((w * xc) @ basis) ** 2
    var_b = (w @ (basis * basis)) - ((w @ basis) ** 2) / sw_safe
    den = var_b * sw_safe
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 1e-300)
    return out


def _peak_rows(power: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Row-wise maximum of the power and its index; NaN / -1 for rows with no finite power."""
    if power.shape[1] == 0:
        return np.full(power.shape[0], np.nan), np.full(power.shape[0], -1, dtype=np.int64)
    any_ok = np.any(np.isfinite(power), axis=1)
    filled = np.where(np.isfinite(power), power, -np.inf)
    idx = np.argmax(filled, axis=1).astype(np.int64)
    peak = filled[np.arange(power.shape[0]), idx]
    return np.where(any_ok, peak, np.nan), np.where(any_ok, idx, -1)


def _nan_readout(sided: str) -> TestReadout:
    nan = float("nan")
    return TestReadout(value=nan, null_mean=nan, null_sd=nan, z=nan, p=nan, n_null=0, sided=sided)


def _readout(observed: float, null: NDArray[np.float64], sided: str) -> TestReadout:
    """Monte Carlo readout: p = (1 + #{null at least as extreme}) / (1 + n) (Phipson & Smyth 2010)."""
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return _nan_readout(sided)
    n = int(null.size)
    mean = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if n > 1 else float("nan")
    z = (observed - mean) / sd if (np.isfinite(sd) and sd > 0) else float("nan")
    tol = 1e-9 * max(1.0, abs(observed))       # a replicate equal to the observed up to rounding counts as a tie
    if sided == "two":
        centre = (observed + float(np.sum(null))) / (n + 1)      # symmetric in observed and replicates
        b = int(np.sum(np.abs(null - centre) >= abs(observed - centre) - tol))
    else:
        b = int(np.sum(null >= observed - tol))
    return TestReadout(value=float(observed), null_mean=mean, null_sd=sd, z=float(z), p=(1 + b) / (1 + n),
                       n_null=n, sided=sided)


def holm_two(p_a: float, p_b: float) -> Tuple[float, float]:
    """Holm (1979) adjusted p-values of a family of two, in the order given. NaN stays NaN (family of one)."""
    a_ok, b_ok = bool(np.isfinite(p_a)), bool(np.isfinite(p_b))
    if not (a_ok and b_ok):
        return (float(p_a), float(p_b))
    lo, hi = min(p_a, p_b), max(p_a, p_b)
    adj_lo = min(1.0, 2.0 * lo)
    adj_hi = min(1.0, max(adj_lo, hi))
    return (adj_lo, adj_hi) if p_a <= p_b else (adj_hi, adj_lo)


# ----------------------------------------------------------------------------------------------------
# Exclusion zone, folded profile, null
# ----------------------------------------------------------------------------------------------------
def _run_above(rel: NDArray[np.float64], k: int, n: int, threshold: float, direction: int) -> int:
    """Number of consecutive bins next to ``k`` (one direction) that are measured and at or above ``threshold``."""
    run = 0
    while run < n - 1:
        v = rel[(k + direction * (run + 1)) % n]
        if np.isfinite(v) and v >= threshold:
            run += 1
        else:
            break
    return run


def _cap_sides(left: int, right: int, n: int) -> Tuple[int, int]:
    while left + right + 1 > n:                     # the zone cannot be longer than the curve
        if right >= left and right > 0:
            right -= 1
        elif left > 0:
            left -= 1
        else:
            break
    return left, right


def exclusion_zone(p: DensityProfile, *, rule: str = DEFAULT_RULE, min_nm: float = 200.0) -> Tuple[int, int]:
    """Bins (first, last) around the maximum excluded from every statistic.

    The pair is UNWRAPPED: ``first <= p.max_bin <= last``; ``first`` can be negative and ``last`` can reach
    ``n_bins`` or more (the curve is closed). Take the indices modulo ``n_bins``. Its width in nm is
    ``(last - first + 1) * p.bin_nm_effective`` (export the width, not the indices).

    ``rule="baseline"`` (default): from the maximum to each side, the REACH is the distance to the first bin
    below ``median + max(BASELINE_HEIGHT_FRAC * height, BASELINE_NOISE_K * noise)``, where height is the
    maximum above the median of the profile and noise is 1.4826 MAD of the profile; the zone goes
    ``ZONE_WIDEN`` times the reach, and never less than ``min_nm`` per side. A Gaussian cluster reaches the
    10 % threshold at 2.15 sd, so the zone ends at about 3.2 sd, where less than 0.1 % of the cluster is left.
    The noise term stops the run when the tail of a low, broad cluster sinks into the fluctuations of the
    profile instead of walking on along them. [own sim]

    ``rule="half_max"``: the run at or above ``median + height / 2`` plus one bin (the first design). Kept for
    the sensitivity table; it leaves the flank of every resolved cluster in the remainder. Do not use it for a
    reading.

    ``rule="fixed"``: ``min_nm`` on each side, nothing else.

    Every rule looks only at the maximum, at the bins inside the zone, and at the median and MAD of the whole
    profile, which a shift of the remainder cannot change: recomputing the zone in a replicate gives the same
    zone, and the remainder starts on bins the rule never looked at.

    ``min_nm`` is converted with ``round(min_nm / bin)``: 200 nm are 2 bins of the nominal 100 nm grid whatever
    the effective bin (99-101 nm) turns out to be.
    """
    if rule not in ZONE_RULES:
        raise ValueError(f"unknown exclusion rule {rule!r}; expected one of {ZONE_RULES}")
    n = int(p.n_bins)
    k = int(p.max_bin)
    if n <= 0:
        return (k, k)
    bin_nm = float(p.bin_nm_effective)
    side_min = max(1, int(round(float(min_nm) / bin_nm))) if bin_nm > 0 else 1
    left = right = side_min
    if rule != "fixed":
        rel = _relative(p)
        finite = rel[np.isfinite(rel)]
        if finite.size and np.isfinite(rel[k % n]):
            base = float(np.median(finite))
            height = float(rel[k % n]) - base
            if rule == "half_max":
                threshold, widen = base + 0.5 * height, 1.0
            else:
                noise = 1.4826 * float(np.median(np.abs(finite - base)))
                threshold = base + max(BASELINE_HEIGHT_FRAC * height, BASELINE_NOISE_K * noise)
                widen = ZONE_WIDEN
            sides = []
            for direction in (+1, -1):
                reach = _run_above(rel, k, n, threshold, direction) + 1      # the bin that ended the run
                sides.append(max(int(np.ceil(widen * reach - 1e-9)), side_min))
            right, left = sides[0], sides[1]
    left, right = _cap_sides(left, right, n)
    return (k - left, k + right)


def _widen_zone(zone: Tuple[int, int], max_bin: int, factor: float, n: int) -> Tuple[int, int]:
    """The zone with both half-widths multiplied by ``factor`` (rounded up)."""
    left = int(np.ceil(factor * (max_bin - zone[0]) - 1e-9))
    right = int(np.ceil(factor * (zone[1] - max_bin) - 1e-9))
    left, right = _cap_sides(left, right, n)
    return (max_bin - left, max_bin + right)


def folded_profile(p: DensityProfile, zone: Tuple[int, int]) -> NDArray[np.float64]:
    """Relative density against |d|, the arc distance to the maximum, with the zone left out.

    Entry j is the mean of the (one or two) bins at j bins from the maximum, j = 0 .. n_bins // 2, i.e. at
    ``j * p.bin_nm_effective`` nm. Bins inside ``zone`` and unmeasured bins do not contribute; a lag with no
    contributing bin is NaN (lag 0 always is). The sign of d means nothing without the orientation of the
    section, hence the fold. Values are relative to the maximum, as in the profile; the readouts divide the
    remainder by its own mean before using it.
    """
    n = int(p.n_bins)
    rel = _relative(p)
    if n <= 0 or rel.size != n:
        return np.zeros(0)
    rel[_zone_bins(zone, n)] = np.nan
    value, _ = _fold(rel[None, :], int(p.max_bin))
    return cast(NDArray[np.float64], value[0])


def _shifted_rows(rel: NDArray[np.float64], zone: Tuple[int, int], shifts: NDArray[np.int64]) -> NDArray[np.float64]:
    """One row per shift: ``rel`` with its remainder rolled by that many bins inside its own arc."""
    n = rel.size
    rem = _remainder_bins(zone, n)
    rows = np.tile(rel, (shifts.size, 1))
    m = rem.size
    if m:
        src = (np.arange(m)[None, :] - shifts[:, None]) % m
        rows[:, rem] = rel[rem][src]
    return rows


def remainder_shift_null(p: DensityProfile, zone: Tuple[int, int], *, n_null: int,
                         rng: np.random.Generator) -> NDArray[np.float64]:
    """Replicates of the remainder shifted circularly inside its own arc, one row each.

    Each row is the whole relative profile (``n_bins`` values, NaN where unmeasured) with the zone untouched
    and the remainder rolled by a non-zero number of bins. With m remainder bins there are m - 1 distinct
    non-zero shifts: when ``m - 1 <= n_null`` EVERY one is returned once (fewer rows than ``n_null``; the
    Monte Carlo p is then exact), otherwise ``n_null`` of them drawn without replacement. Unmeasured bins of the
    remainder travel with it.
    """
    n = int(p.n_bins)
    rel = _relative(p)
    m = _remainder_bins(zone, n).size if rel.size == n else 0
    if n <= 0 or m < 2 or n_null <= 0:
        return np.zeros((0, max(n, 0)))
    if m - 1 <= n_null:
        shifts: NDArray[np.int64] = np.arange(1, m, dtype=np.int64)
    else:
        shifts = rng.choice(np.arange(1, m, dtype=np.int64), size=int(n_null), replace=False)
    return _shifted_rows(rel, zone, shifts)


def uniform_realigned_null(p: DensityProfile, *, n_null: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """DIAGNOSTIC of the alignment artefact; not used by any readout.

    Each row: the profile's ``n_locs`` thrown uniformly over its measured bins (same number of localizations,
    same perimeter, no structure), normalised by its maximum and rolled so that its own maximum lands on
    ``p.max_bin`` -- the alignment, repeated. The height of the pooled peak these rows show at lag 0 is what
    aligning alone produces. It destroys the clumping of the real remainder as well, so it must not be used to
    test the remainder (see the module docstring).
    """
    n = int(p.n_bins)
    measured = np.asarray(p.measured, dtype=bool).ravel()
    counts = np.asarray(p.counts).ravel()
    if n <= 0 or measured.size != n or not measured.any() or n_null <= 0:
        return np.zeros((0, max(n, 0)))
    total = int(counts[measured].sum())
    prob = measured / measured.sum()
    out = np.empty((int(n_null), n))
    for i in range(int(n_null)):
        c = rng.multinomial(total, prob).astype(np.float64)
        top = float(c.max())
        row = c / top if top > 0 else c
        row[~measured] = np.nan
        peak = int(np.argmax(c))
        out[i] = np.roll(row, int(p.max_bin) - peak)
    return out


# ----------------------------------------------------------------------------------------------------
# The two statistics
# ----------------------------------------------------------------------------------------------------
def near_far_contrast(folded: NDArray[np.float64], lags_nm: NDArray[np.float64], *, near_nm: float = 1000.0,
                      weights: Optional[NDArray[np.float64]] = None) -> float:
    """Mean density within ``near_nm`` of the zone minus that of the rest.

    ``folded`` is a folded profile with NaN inside the zone; the zone ends where the first finite value is, and
    "near" are the finite lags from there up to ``near_nm`` further (10 bins with the defaults). ``weights``
    (optional, per lag) weight the two means; the readouts pass the number of bins behind each lag.
    NaN when either side is empty. The value is in the units of ``folded``; the readouts pass the remainder
    divided by its own mean.
    """
    f = np.asarray(folded, dtype=np.float64).ravel()
    lags = np.asarray(lags_nm, dtype=np.float64).ravel()
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if f.size == 0 or f.size != lags.size or w.size != f.size:
        return float("nan")
    return float(_contrast_rows(f[None, :], w[None, :], lags, float(near_nm))[0])


def periodogram_peak(remainder: NDArray[np.float64], bin_nm: float, *,
                     period_range_nm: Tuple[float, Optional[float]] = (300.0, None),
                     weights: Optional[NDArray[np.float64]] = None) -> Tuple[float, float]:
    """Return (period_nm, power) of the phase-locked periodogram maximum; upper limit None = L / 2.

    ``remainder`` is the FOLDED remainder (``folded_profile``): entry j at ``j * bin_nm`` from the maximum, NaN
    inside the zone. The power at a period T is the variance of the remainder carried by cos(2 pi |d| / T)
    (weighted least squares): modes at T, 2T, 3T ... from the maximum, or exactly in between. The cosines of
    the half range are a complete basis, so modes at another fixed phase are seen too, at neighbouring
    periods. Scanned periods are 2 S / k with S the folded span, k integer, from ``period_range_nm[0]`` up
    to the upper limit; ``None`` means S, which is half the perimeter (one full cycle on each side). The plain
    periodogram is not used because it cannot see a circular shift (module docstring).
    """
    f = np.asarray(remainder, dtype=np.float64).ravel()
    w = np.ones_like(f) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if f.size < 2 or w.size != f.size or not bin_nm > 0:
        return (float("nan"), float("nan"))
    lags = np.arange(f.size, dtype=np.float64) * float(bin_nm)
    periods = _period_grid(float(lags[-1]), period_range_nm)
    if periods.size == 0:
        return (float("nan"), float("nan"))
    peak, idx = _peak_rows(_power_rows(f[None, :], w[None, :], lags, periods))
    if idx[0] < 0:
        return (float("nan"), float("nan"))
    return (float(periods[idx[0]]), float(peak[0]))


# ----------------------------------------------------------------------------------------------------
# One axon
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class _Prepared:
    """One profile ready for the null: relative density, zone, and whether it can be tested."""

    rel: NDArray[np.float64]
    n: int
    bin_nm: float
    max_bin: int
    zone: Tuple[int, int]
    m: int                      # remainder bins
    scale: float                # mean of the remainder (relative to the maximum); the statistics divide by it
    usable: bool

    @property
    def zone_nm(self) -> float:
        return (self.zone[1] - self.zone[0] + 1) * self.bin_nm


def _prepare(p: DensityProfile, rule: str, min_nm: float, widen: float = 1.0) -> _Prepared:
    n = int(p.n_bins)
    rel = _relative(p)
    zone = exclusion_zone(p, rule=rule, min_nm=min_nm)
    if widen != 1.0 and n > 0:
        zone = _widen_zone(zone, int(p.max_bin), float(widen), n)
    m = int(_remainder_bins(zone, n).size) if n > 0 else 0
    usable = bool(n > 0 and rel.size == n and p.bin_nm_effective > 0 and m >= MIN_REMAINDER_BINS
                  and np.isfinite(rel[int(p.max_bin) % n]) and int(p.n_locs) > 0)
    scale = float("nan")
    if usable:
        rest = rel[_remainder_bins(zone, n)]
        rest = rest[np.isfinite(rest)]
        scale = float(np.mean(rest)) if rest.size else float("nan")
        usable = bool(rest.size >= MIN_REMAINDER_BINS and scale > 0)
    return _Prepared(rel=rel, n=n, bin_nm=float(p.bin_nm_effective), max_bin=int(p.max_bin), zone=zone, m=m,
                     scale=scale, usable=usable)


def _masked_fold(rows: NDArray[np.float64], prep: _Prepared) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """What every statistic sees: the zone left out, the remainder divided by its own mean, folded."""
    masked = rows / prep.scale
    masked[:, _zone_bins(prep.zone, prep.n)] = np.nan
    return _fold(masked, prep.max_bin)


@dataclass(frozen=True)
class AxonADetails:
    """Everything question A yields for one axon. NaN readouts when ``usable`` is False."""

    contrast: TestReadout           # two-sided; value in units of the remainder's mean
    period: TestReadout             # one-sided; value = phase-locked power at the peak
    zone: Tuple[int, int]           # unwrapped bins (first, last)
    exclusion_nm: float             # width of the zone
    exclusion_rule: str
    period_nm: float                # period of the observed peak
    period_at_edge: bool            # the peak is the longest scanned period: a trend, not a repeat
    n_remainder_bins: int
    usable: bool


def axon_a_details(p: DensityProfile, *, n_null: int = 1999, seed: int = 0, rule: str = DEFAULT_RULE,
                   min_nm: float = EXCLUSION_MIN_NM, near_nm: float = NEAR_NM,
                   period_range_nm: Tuple[float, Optional[float]] = (PERIOD_MIN_NM, None),
                   widen: float = 1.0) -> AxonADetails:
    """Question A for one axon: the two readouts, the zone, its width in nm and the period of the peak.

    ``widen`` multiplies the half-widths of the zone (sensitivity). An empty profile (``max_bin = -1``, nothing
    measured) or a remainder shorter than ``MIN_REMAINDER_BINS`` gives NaN readouts, never an exception.
    """
    prep = _prepare(p, rule, min_nm, widen)
    contrast_readout, period_readout = _nan_readout("two"), _nan_readout("one")
    period_nm, at_edge = float("nan"), False
    if prep.usable and n_null > 0:
        rng = np.random.default_rng(seed)
        null_rows = remainder_shift_null(p, prep.zone, n_null=n_null, rng=rng)
        rows = np.vstack([prep.rel[None, :], null_rows])
        folded, count = _masked_fold(rows, prep)
        lags = np.arange(folded.shape[1], dtype=np.float64) * prep.bin_nm
        contrast = _contrast_rows(folded, count, lags, float(near_nm))
        contrast_readout = _readout(float(contrast[0]), contrast[1:], "two")
        periods = _period_grid(float(lags[-1]), period_range_nm)
        if periods.size:
            peak, idx = _peak_rows(_power_rows(folded, count, lags, periods))
            period_readout = _readout(float(peak[0]), peak[1:], "one")
            if idx[0] >= 0:
                period_nm, at_edge = float(periods[idx[0]]), bool(idx[0] == 0)
    return AxonADetails(contrast=contrast_readout, period=period_readout, zone=prep.zone,
                        exclusion_nm=prep.zone_nm, exclusion_rule=rule, period_nm=period_nm,
                        period_at_edge=at_edge, n_remainder_bins=prep.m, usable=prep.usable)


def axon_a_readouts(p: DensityProfile, *, n_null: int = 1999, seed: int = 0, rule: str = DEFAULT_RULE,
                    min_nm: float = EXCLUSION_MIN_NM, near_nm: float = NEAR_NM,
                    period_range_nm: Tuple[float, Optional[float]] = (PERIOD_MIN_NM, None)) -> Dict[str, TestReadout]:
    """Per-axon "contrast" and "period" against their nulls.

    ``contrast`` is two-sided, ``period`` one-sided; ``z`` is signed against the axon's own null (the type-B
    readout that is averaged per animal). The null is every non-zero shift of the remainder when there are no
    more than ``n_null`` of them (``n_null`` in the readout is the number actually used). The period of the
    peak and the width of the zone are in ``axon_a_details``. These p values are descriptive.
    """
    d = axon_a_details(p, n_null=n_null, seed=seed, rule=rule, min_nm=min_nm, near_nm=near_nm,
                       period_range_nm=period_range_nm)
    return {"contrast": d.contrast, "period": d.period}


def zone_sensitivity(p: DensityProfile, *, n_null: int = 1999, seed: int = 0,
                     near_nm: float = NEAR_NM) -> Dict[str, Dict[str, TestReadout]]:
    """The per-axon readouts under six exclusion zones: baseline, baseline widened 2 and 3 times, half height,
    fixed 200 nm and fixed 400 nm. The pooled counterpart is ``PooledDetails.sensitivity``.

    A contrast that is positive with the narrow zones and gone with the wide ones is the flank of the large
    cluster, not an attractor.
    """
    settings = (("baseline", "baseline", EXCLUSION_MIN_NM, 1.0), ("baseline_x2", "baseline", EXCLUSION_MIN_NM, 2.0),
                ("baseline_x3", "baseline", EXCLUSION_MIN_NM, 3.0), ("half_max", "half_max", EXCLUSION_MIN_NM, 1.0),
                ("fixed_200nm", "fixed", 200.0, 1.0), ("fixed_400nm", "fixed", 400.0, 1.0))
    out: Dict[str, Dict[str, TestReadout]] = {}
    for name, rule, width, widen in settings:
        d = axon_a_details(p, n_null=n_null, seed=seed, rule=rule, min_nm=width, near_nm=near_nm, widen=widen)
        out[name] = {"contrast": d.contrast, "period": d.period}
    return out


# ----------------------------------------------------------------------------------------------------
# Pooled across axons
# ----------------------------------------------------------------------------------------------------
def _clean_animals(animals: Optional[Sequence[str]], n: int) -> List[str]:
    """One label per profile; a blank label is no animal."""
    if animals is None:
        return [""] * n
    labels = ["" if a is None else str(a).strip() for a in animals]
    if len(labels) != n:
        raise ValueError(f"animals has {len(labels)} entries for {n} profiles")
    return labels


def _grid(preps: Sequence[_Prepared], axis: str) -> Tuple[List[NDArray[np.int64]], NDArray[np.float64]]:
    """Common lag grid: for every axon the grid cell of each folded lag, and the grid values.

    "nm": cells of the median effective bin (100 nm nominal), cell j at j * bin. "fraction": |d| / perimeter in
    steps of 1 / (fewest bins among the axons), so the shortest axon leaves no hole; values in [0, 0.5].
    """
    if axis == "nm":
        step = float(np.median([q.bin_nm for q in preps]))
        pos = [np.arange(q.n // 2 + 1) * q.bin_nm / step for q in preps]
    else:
        step = 1.0 / min(q.n for q in preps)
        pos = [np.arange(q.n // 2 + 1) / q.n / step for q in preps]
    cells = [np.rint(x).astype(np.int64) for x in pos]
    size = max(int(c[-1]) for c in cells) + 1
    return cells, np.arange(size, dtype=np.float64) * step


def _to_grid(value: NDArray[np.float64], count: NDArray[np.float64], cells: NDArray[np.int64],
             size: int) -> NDArray[np.float64]:
    """Rows (R, H) of one axon on the common grid (R, size); NaN where the axon has nothing."""
    onehot = np.zeros((cells.size, size))
    onehot[np.arange(cells.size), cells] = 1.0
    ok = np.isfinite(value) & (count > 0)
    c = np.where(ok, count, 0.0)
    num = (np.where(ok, value, 0.0) * c) @ onehot
    den = c @ onehot
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


@dataclass(frozen=True, eq=False)
class _Pooled:
    """Pooled curves, row 0 observed and one row per replicate."""

    mean: NDArray[np.float64]            # (R + 1, K)
    units: NDArray[np.float64]           # (R + 1, K) axons (or animals) behind each cell: the weights of the period
    n_axons: NDArray[np.int64]  # (K,) observed
    n_animals: NDArray[np.int64]


def _mean_over_units(sums: Dict[str, NDArray[np.float64]], counts: Dict[str, NDArray[np.float64]],
                     by_animal: bool) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One axon one weight (a single unit "all"), or one animal one weight (mean of the animals' means)."""
    if by_animal:
        first = next(iter(sums.values()))
        total = np.zeros_like(first)
        units = np.zeros_like(first)
        for unit, s in sums.items():
            k = counts[unit]
            total += np.divide(s, k, out=np.zeros_like(s), where=k > 0)
            units += k > 0
    else:
        total, units = sums["all"], counts["all"]
    mean = np.full(total.shape, np.nan)
    np.divide(total, units, out=mean, where=units > 0)
    return mean, units


def _pool(preps: Sequence[_Prepared], rows: Sequence[NDArray[np.float64]], animals: Sequence[str], weights: str,
          cells: Sequence[NDArray[np.int64]], size: int, masked: bool) -> _Pooled:
    """Fold every axon's rows, put them on the grid and average: one axon one weight, or one animal one weight."""
    n_rows = rows[0].shape[0]
    sums: Dict[str, NDArray[np.float64]] = {}
    counts: Dict[str, NDArray[np.float64]] = {}
    present: Dict[str, NDArray[np.bool_]] = {}
    n_axons = np.zeros(size, dtype=np.int64)
    for i, (prep, r) in enumerate(zip(preps, rows)):
        value, count = _masked_fold(r, prep) if masked else _fold(r, prep.max_bin)
        curve = _to_grid(value, count, cells[i], size)
        has = np.isfinite(curve)
        unit = animals[i] if weights == "animal" else "all"
        if unit not in sums:
            sums[unit] = np.zeros((n_rows, size))
            counts[unit] = np.zeros((n_rows, size))
        sums[unit] += np.where(has, curve, 0.0)
        counts[unit] += has
        n_axons += has[0]
        if animals[i]:
            seen = present.setdefault(animals[i], np.zeros(size, dtype=bool))
            seen |= has[0]
    mean, units = _mean_over_units(sums, counts, weights == "animal")
    n_animals = np.zeros(size, dtype=np.int64)
    for seen in present.values():
        n_animals += seen
    return _Pooled(mean=mean, units=units, n_axons=n_axons, n_animals=n_animals)


def _pooled_contrast(preps: Sequence[_Prepared], rows: Sequence[NDArray[np.float64]], animals: Sequence[str],
                     weights: str, near_nm: float) -> NDArray[np.float64]:
    """Mean over axons (or over animals) of the per-axon contrast, each from its own zone. One value per row."""
    n_rows = rows[0].shape[0]
    sums: Dict[str, NDArray[np.float64]] = {}
    counts: Dict[str, NDArray[np.float64]] = {}
    for i, (prep, r) in enumerate(zip(preps, rows)):
        folded, count = _masked_fold(r, prep)
        lags = np.arange(folded.shape[1], dtype=np.float64) * prep.bin_nm
        c = _contrast_rows(folded, count, lags, near_nm)
        has = np.isfinite(c)
        unit = animals[i] if weights == "animal" else "all"
        if unit not in sums:
            sums[unit] = np.zeros(n_rows)
            counts[unit] = np.zeros(n_rows)
        sums[unit] += np.where(has, c, 0.0)
        counts[unit] += has
    mean, _ = _mean_over_units(sums, counts, weights == "animal")
    return mean


def _global_envelope(curves: NDArray[np.float64], alpha: float,
                     allowed: Optional[NDArray[np.bool_]] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Global envelope of the rows of ``curves`` (row 0 observed): directional quantile envelope of
    Myllymaki et al. (2017) [lit, not re-read for this module].

    Every curve is measured by its largest pointwise departure from the mean curve, in units of the distance
    from the mean to the pointwise 97.5 % quantile (above) or 2.5 % quantile (below). The envelope is the mean
    plus/minus those distances times the (1 - alpha) quantile of that measure over all curves, the observed
    included: under the null the observed curve leaves it ANYWHERE with probability <= alpha.

    The extreme-rank envelope of the same paper, which the plan named, is not used: with 1999 replicates and
    about 100 lags its rank collapses to 1 (the envelope becomes the minimum and maximum of all curves, the
    observed included, and nothing can ever leave it); it needs tens of thousands of replicates.

    Lags outside ``allowed``, lags that are not finite in every row, and constant lags (inside every zone the
    replicates equal the observed) take no part; constant allowed lags get that constant, the others NaN.
    """
    s, k_lags = curves.shape
    lo = np.full(k_lags, np.nan)
    hi = np.full(k_lags, np.nan)
    finite = np.all(np.isfinite(curves), axis=0)
    if allowed is not None:
        finite &= allowed
    spread = np.zeros(k_lags)
    spread[finite] = np.ptp(curves[:, finite], axis=0)
    constant = finite & (spread <= 1e-12)
    lo[constant] = hi[constant] = curves[0, constant]
    use = finite & ~constant
    if s < 2 or not use.any():
        return lo, hi
    sub = curves[:, use]
    centre = np.mean(sub, axis=0)
    q_lo, q_hi = np.percentile(sub, POINTWISE_PCT, axis=0)
    up = np.maximum(q_hi - centre, 1e-12)
    down = np.maximum(centre - q_lo, 1e-12)
    dev = sub - centre[None, :]
    measure = np.max(np.where(dev >= 0, dev / up[None, :], -dev / down[None, :]), axis=1)
    rank = min(s, int(np.ceil((1.0 - alpha) * s)))
    u = float(np.sort(measure)[rank - 1])
    lo[use] = centre - u * down
    hi[use] = centre + u * up
    return lo, hi


def _group_label(group: str, labels: Sequence[str]) -> str:
    known = sorted({a for a in labels if a})
    if group:
        return group if known else f"{group} [{NO_ANIMAL_GROUP}]"
    return NO_ANIMAL_GROUP if not known else ""


def _check_pool_args(axis: str, weights: str, labels: Sequence[str]) -> None:
    if axis not in POOL_AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {POOL_AXES}")
    if weights not in POOL_WEIGHTS:
        raise ValueError(f"unknown weights {weights!r}; expected one of {POOL_WEIGHTS}")
    if weights == "animal" and not all(labels):
        raise ValueError('weights="animal" needs an animal for every profile')


def _empty_profile(group: str, axis: str) -> PooledProfile:
    z = np.zeros(0)
    zi = np.zeros(0, dtype=np.int64)
    return PooledProfile(group=group, lag_nm=z, mean_relative=z, n_axons=zi, n_animals=zi, null_lo=z, null_hi=z,
                         global_lo=z, global_hi=z, axis=axis)


def pool_aligned(profiles: Sequence[DensityProfile], *, axis: str = "nm", weights: str = "equal",
                 animals: Optional[Sequence[str]] = None, group: str = "") -> PooledProfile:
    """Mean of the folded profiles (one axon one weight) with the number of axons at each distance.

    This is the RAW pooled profile, zone included, relative to each axon's maximum: it shows the peak at lag 0
    that aligning produces, on purpose. It carries no bands (NaN): only ``pooled_test`` can say what part of it
    is more than alignment.

    ``axis="nm"`` (default, decision of 2026-11-16): distance in nm on the common 100 nm grid; long distances
    are held by the few long axons, see ``n_axons``. ``axis="fraction"``: distance as a fraction of each
    axon's perimeter (0 to 0.5); ``lag_nm`` then holds fractions.
    ``weights="equal"``: simple mean over axons. ``weights="animal"``: one curve per animal (simple mean of its
    axons), then the mean over animals; every profile needs an animal.
    ``animals`` (one label per profile, blank = none) only fills ``n_animals`` and the group label: with no
    animal the group reads "SIN ANIMAL".
    """
    labels = _clean_animals(animals, len(profiles))
    _check_pool_args(axis, weights, labels)
    label = _group_label(group, labels)
    keep = [i for i, p in enumerate(profiles) if int(p.n_bins) > 1 and np.asarray(p.relative).size == int(p.n_bins)
            and p.bin_nm_effective > 0]
    if not keep:
        return _empty_profile(label, axis)
    preps = [_Prepared(rel=_relative(profiles[i]), n=int(profiles[i].n_bins),
                       bin_nm=float(profiles[i].bin_nm_effective), max_bin=int(profiles[i].max_bin),
                       zone=(int(profiles[i].max_bin), int(profiles[i].max_bin)), m=0, scale=1.0, usable=True)
             for i in keep]
    cells, lag = _grid(preps, axis)
    pooled = _pool(preps, [q.rel[None, :] for q in preps], [labels[i] for i in keep], weights, cells, lag.size,
                   masked=False)
    nan = np.full(lag.size, np.nan)
    return PooledProfile(group=label, lag_nm=lag, mean_relative=pooled.mean[0], n_axons=pooled.n_axons,
                         n_animals=pooled.n_animals, null_lo=nan, null_hi=nan.copy(), global_lo=nan.copy(),
                         global_hi=nan.copy(), axis=axis)


def _scope(n_axons: int, labels: Sequence[str]) -> str:
    known = sorted({a for a in labels if a})
    base = f"n = {n_axons} axons; randomization within each axon"
    if not known:
        return f"{base}; {NO_ANIMAL_GROUP}: no inference across animals or genotypes"
    if len(known) == 1:
        return f"{base}; n_animals = 1 ({known[0]}): no comparison between animals is possible"
    return f"{base}; n_animals = {len(known)}: this p is not an animal-level test"


@dataclass(frozen=True)
class ZoneRun:
    """The two pooled statistics under one exclusion zone (one row of the pooled sensitivity table)."""

    widen: float                    # factor on the half-widths of the default zone (1 = the zone of the reading)
    zone_nm_median: float           # median width of the zone over the axons that entered
    n_axons: int
    contrast: TestReadout
    period: TestReadout
    contrast_minus_null: float      # value - null_mean: the sign that is read
    period_nm: float
    period_at_edge: bool


@dataclass(frozen=True, eq=False)
class PooledDetails:
    """The pooled question A of one group, structured. ``result`` is the contract object."""

    result: PooledResult
    label: str                      # the reading alone, e.g. "attractor" or "inhibition zone + regular"
    exploratory: bool               # no animal or a single animal
    contrast_p_holm: float
    period_p_holm: float
    contrast_minus_null: float
    period_nm: float
    period_at_edge: bool
    attractor_robust: bool          # contrast above its null and significant under every widened zone
    regular_robust: bool
    scope: str
    n_animals: int
    exclusion_rule: str
    exclusion_min_nm: float
    zone_nm_median: float
    zone_nm_min: float
    zone_nm_max: float
    sensitivity: Tuple[ZoneRun, ...]   # the default zone first, then the widened ones
    alpha: float


def reading_label(result: PooledResult) -> str:
    """The label of ``result.reading`` ALONE (no prefix, no numbers), built from the ``READING_*`` vocabulary."""
    text = str(result.reading)
    for prefix in (PREFIX_EXPLORATORY, PREFIX_WITHIN_SAMPLE):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.split(READING_SEPARATOR)[0]


def _zone_run(preps: Sequence[_Prepared], labels: Sequence[str], weights: str, n_null: int,
              rng: np.random.Generator, near_nm: float, period_range_nm: Tuple[float, Optional[float]],
              widen: float) -> Tuple[ZoneRun, List[NDArray[np.float64]]]:
    """The two pooled statistics for one set of zones, and the rows (observed first) they were computed on."""
    rows: List[NDArray[np.float64]] = []
    for q in preps:                                  # row 0 observed (shift 0), then n_null independent shifts
        shifts = np.concatenate([np.zeros(1, dtype=np.int64), rng.integers(0, q.m, size=int(n_null))])
        rows.append(_shifted_rows(q.rel, q.zone, shifts))
    contrast = _pooled_contrast(preps, rows, labels, weights, float(near_nm))
    contrast_readout = _readout(float(contrast[0]), contrast[1:], "two")
    cells_nm, lag_nm = _grid(preps, "nm")
    stat = _pool(preps, rows, labels, weights, cells_nm, lag_nm.size, masked=True)
    periods = _period_grid(float(lag_nm[-1]), period_range_nm)
    period_readout = _nan_readout("one")
    period_nm, at_edge = float("nan"), False
    if periods.size:
        peak, idx = _peak_rows(_power_rows(stat.mean, stat.units, lag_nm, periods))
        period_readout = _readout(float(peak[0]), peak[1:], "one")
        if idx[0] >= 0:
            period_nm, at_edge = float(periods[idx[0]]), bool(idx[0] == 0)
    run = ZoneRun(widen=float(widen), zone_nm_median=float(np.median([q.zone_nm for q in preps])),
                  n_axons=len(preps), contrast=contrast_readout, period=period_readout,
                  contrast_minus_null=float(contrast_readout.value - contrast_readout.null_mean),
                  period_nm=period_nm, period_at_edge=at_edge)
    return run, rows


def pooled_details(profiles: Sequence[DensityProfile], *, n_null: int = 1999, seed: int = 0, axis: str = "nm",
                   weights: str = "equal", animals: Optional[Sequence[str]] = None, group: str = "",
                   rule: str = DEFAULT_RULE, min_nm: float = EXCLUSION_MIN_NM, near_nm: float = NEAR_NM,
                   period_range_nm: Tuple[float, Optional[float]] = (PERIOD_MIN_NM, None),
                   alpha: float = ALPHA, robust_widen: Sequence[float] = ROBUST_WIDEN) -> PooledDetails:
    """Pooled question A of one group, with everything ``PooledResult.reading`` only says in words.

    Null: in every replicate the remainder of EACH axon is shifted on its own (independent uniform shifts,
    the identity included) and the axons are pooled exactly as the observed ones. The zone of each axon stays
    put, so the alignment artefact is in every replicate as it is in the data.

    * ``result.profile``: the raw pooled profile (zone included, relative to the maximum) with the pointwise
      2.5-97.5 % band and the global directional quantile envelope of the null at level ``alpha``. Inside the
      zones the band collapses onto the profile: the peak at 0 is inside its own null. The band is for display
      (module docstring) and is drawn where at least ``GLOBAL_MIN_AXONS`` axons contribute.
    * ``result.contrast`` / ``result.period``: the two pooled statistics (module docstring) under the default
      zone. ``p`` is the RAW Monte Carlo p; the Holm pair is in ``contrast_p_holm`` / ``period_p_holm``.
    * ``sensitivity``: the same two statistics with every zone widened by ``robust_widen`` (own shifts, own
      null). Axons whose remainder becomes too short drop out of that run (``ZoneRun.n_axons``).
    * ``label`` and ``result.reading``: decided on the Holm p at ``alpha`` and on the sign of
      ``contrast_minus_null``; "attractor" and "regular" also need every widened run (raw p <= alpha, same
      sign, peak not at the edge), otherwise they carry ``READING_NOT_ROBUST``.
      The reading starts with "EXPLORATORY, within-sample: " when there is no animal or a single one.
    * ``result.n_axons``: axons that entered the statistics (a profile that is empty or whose remainder is
      shorter than ``MIN_REMAINDER_BINS`` is left out).

    ``animals`` / ``group`` / ``weights`` / ``axis``: as in ``pool_aligned``; the statistics are always on the
    nm axis. No p between animals or genotypes is computed here, whatever is passed.
    """
    labels_all = _clean_animals(animals, len(profiles))
    _check_pool_args(axis, weights, labels_all)
    group_label = _group_label(group, labels_all)
    prepared = [_prepare(p, rule, min_nm) for p in profiles]
    keep = [i for i, q in enumerate(prepared) if q.usable]
    labels = [labels_all[i] for i in keep]
    known = sorted({a for a in labels if a})
    exploratory = len(known) < 2
    prefix = PREFIX_EXPLORATORY if exploratory else PREFIX_WITHIN_SAMPLE
    scope = _scope(len(keep), labels)
    nan = float("nan")
    if not keep or n_null <= 0:
        reading = f"{prefix}{READING_NOT_COMPUTABLE}{READING_SEPARATOR}{scope}"
        result = PooledResult(profile=_empty_profile(group_label, axis), contrast=_nan_readout("two"),
                              period=_nan_readout("one"), n_axons=len(keep), reading=reading)
        return PooledDetails(result=result, label=READING_NOT_COMPUTABLE, exploratory=exploratory,
                             contrast_p_holm=nan, period_p_holm=nan, contrast_minus_null=nan, period_nm=nan,
                             period_at_edge=False, attractor_robust=False, regular_robust=False, scope=scope,
                             n_animals=len(known), exclusion_rule=rule, exclusion_min_nm=float(min_nm),
                             zone_nm_median=nan, zone_nm_min=nan, zone_nm_max=nan, sensitivity=(), alpha=float(alpha))
    preps = [prepared[i] for i in keep]
    main, rows = _zone_run(preps, labels, weights, n_null, np.random.default_rng([int(seed), 0]), near_nm,
                           period_range_nm, 1.0)

    # --- sensitivity: the same statistics with every zone widened ---
    runs: List[ZoneRun] = [main]
    for j, factor in enumerate(robust_widen):
        wide_all = [_prepare(profiles[i], rule, min_nm, float(factor)) for i in keep]
        ok = [k for k, q in enumerate(wide_all) if q.usable]
        if not ok:
            continue
        run, _ = _zone_run([wide_all[k] for k in ok], [labels[k] for k in ok], weights, n_null,
                           np.random.default_rng([int(seed), j + 1]), near_nm, period_range_nm, float(factor))
        runs.append(run)
    wide = runs[1:]

    # --- display: raw profile (zones included) with the bands of the default run's replicates ---
    cells, lag = _grid(preps, axis)
    raw = _pool(preps, rows, labels, weights, cells, lag.size, masked=False)
    null = raw.mean[1:]
    some = np.any(np.isfinite(null), axis=0)
    null_lo = np.full(lag.size, np.nan)
    null_hi = np.full(lag.size, np.nan)
    if some.any():
        null_lo[some], null_hi[some] = np.nanpercentile(null[:, some], POINTWISE_PCT, axis=0)
    enough = raw.n_axons >= min(GLOBAL_MIN_AXONS, int(raw.n_axons.max()))
    global_lo, global_hi = _global_envelope(raw.mean, float(alpha), enough)
    profile = PooledProfile(group=group_label, lag_nm=lag, mean_relative=raw.mean[0], n_axons=raw.n_axons,
                            n_animals=raw.n_animals, null_lo=null_lo, null_hi=null_hi, global_lo=global_lo,
                            global_hi=global_hi, axis=axis)

    # --- reading ---
    p_contrast, p_period = holm_two(main.contrast.p, main.period.p)
    above = main.contrast_minus_null > 0
    contrast_sig = bool(np.isfinite(p_contrast) and p_contrast <= alpha)
    period_sig = bool(np.isfinite(p_period) and p_period <= alpha and not main.period_at_edge)
    attractor_robust = bool(wide) and all(r.contrast_minus_null > 0 and r.contrast.p <= alpha for r in wide)
    regular_robust = bool(wide) and all(r.period.p <= alpha and not r.period_at_edge for r in wide)
    found: List[str] = []
    if contrast_sig and above:
        found.append(READING_ATTRACTOR + ("" if attractor_robust else READING_NOT_ROBUST))
    elif contrast_sig:
        found.append(READING_INHIBITION)
    if period_sig:
        found.append(READING_REGULAR + ("" if regular_robust else READING_NOT_ROBUST))
    label = READING_JOIN.join(found) if found else READING_NONE
    edge = " (edge: a trend, not a repeat)" if main.period_at_edge else ""
    widened = "; ".join(f"x{r.widen:g}: contrast - null {r.contrast_minus_null:+.3g} (p {r.contrast.p:.3g}), "
                        f"period p {r.period.p:.3g}" for r in wide)
    zone_nm = [q.zone_nm for q in preps]
    reading = READING_SEPARATOR.join([
        f"{prefix}{label}",
        f"contrast - null {main.contrast_minus_null:+.3g} (Holm p {p_contrast:.3g})",
        f"period peak {main.period_nm:.0f} nm{edge} (Holm p {p_period:.3g})",
        f"zone {rule}, median {float(np.median(zone_nm)):.0f} nm; widened {widened if widened else 'not run'}",
        scope])
    result = PooledResult(profile=profile, contrast=main.contrast, period=main.period, n_axons=len(preps),
                          reading=reading)
    return PooledDetails(result=result, label=label, exploratory=exploratory, contrast_p_holm=float(p_contrast),
                         period_p_holm=float(p_period), contrast_minus_null=main.contrast_minus_null,
                         period_nm=main.period_nm, period_at_edge=main.period_at_edge,
                         attractor_robust=attractor_robust, regular_robust=regular_robust, scope=scope,
                         n_animals=len(known), exclusion_rule=rule, exclusion_min_nm=float(min_nm),
                         zone_nm_median=float(np.median(zone_nm)), zone_nm_min=float(np.min(zone_nm)),
                         zone_nm_max=float(np.max(zone_nm)), sensitivity=tuple(runs), alpha=float(alpha))


def pooled_test(profiles: Sequence[DensityProfile], *, n_null: int = 1999, seed: int = 0, axis: str = "nm",
                weights: str = "equal", animals: Optional[Sequence[str]] = None, group: str = "",
                rule: str = DEFAULT_RULE, min_nm: float = EXCLUSION_MIN_NM, near_nm: float = NEAR_NM,
                period_range_nm: Tuple[float, Optional[float]] = (PERIOD_MIN_NM, None),
                alpha: float = ALPHA) -> PooledResult:
    """Pooled bands (pointwise and global) and the two pooled statistics, Holm between them.

    The contract object of ``pooled_details`` (see there for the null, the statistics, the zone sensitivity and
    the reading). ``reading`` is one line:
    "EXPLORATORY, within-sample: attractor | contrast - null +0.21 (Holm p 0.001) | period peak 9800 nm (edge:
    a trend, not a repeat) (Holm p 0.001) | zone baseline, median 500 nm; widened x2: ...; x3: ... | n = 18
    axons; ...". ``reading_label`` returns the label alone.
    """
    return pooled_details(profiles, n_null=n_null, seed=seed, axis=axis, weights=weights, animals=animals,
                          group=group, rule=rule, min_nm=min_nm, near_nm=near_nm, period_range_nm=period_range_nm,
                          alpha=alpha).result
