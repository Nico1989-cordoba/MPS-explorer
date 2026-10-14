# -*- coding: utf-8 -*-
"""
Axial precision as the data actually achieved it, by depth, and what a
cluster leaks into the neighbouring ring's slab because of it.

Why this exists
---------------
Every ring boundary in the column analysis is a cut in z, and every cut
in z leaks: a localization whose true depth lies inside one ring lands
in the neighbouring slab whenever its axial error exceeds its distance
to the boundary. How often that happens is governed by the axial
precision -- and the only axial precision the file reports, Picasso's
``lpz``, propagates the fit uncertainty of the PSF widths sx and sy
through the astigmatism calibration and nothing else. It knows nothing of calibration error, of focal drift
between calibration and acquisition, or of a z assigned to the wrong
branch of the calibration curve. Fitted ring components can come out
much wider than lpz, and nothing in a table of lpz values says which of
the two is the precision.

The data can say. A molecule seen in two consecutive frames is measured
twice, with independent axial errors of the same sigma; the difference
dz of its two z values has variance 2 sigma^2 whatever the structure
around it, because it is the same molecule in the same place. That is
NeNA (Endesfelder et al. 2014) turned to the axial coordinate, and it
is done here per depth stratum, because an astigmatic z precision is
not a constant: the calibration's slope, and with it the precision,
changes across the +-400 nm range.

What is here
------------
``axial_nena``                    sigma_z per z stratum from consecutive-
                                  frame pairs, next to the reported lpz
                                  and their ratio
``leak_fraction``                 fraction of a cluster expected beyond a
                                  boundary, from its own lpz values
``spurious_cluster_probability``  P[leaked localizations reach min_samples]
``structural_width_nm``           sqrt(sigma_component^2 - lpz^2), floored
                                  at zero
``z_quality``                     the first filter: per ring pair the
                                  separation d, valley, analytic leak,
                                  expected leak copies, events vs the
                                  Delta BIC requirement, and the verdict
                                  viable / marginal / not viable
``viability_v2``                  ring-pair viability rule v2: SiZer
                                  peak + valley (tools.mps_sizer), the
                                  localization minimum vs the reference
                                  ring, the pair's leak-copy tolerance,
                                  and the limiting factors of the axon
``ViabilityParams``               every editable value of the above
``calibrated_lpz_nm``             lpz x the NeNA ratio of the stratum of
                                  each localization's lab z

Design decisions
----------------
* Pairs are consecutive frames of ONE link event (``max_dark_time = 0``).
  A dark frame in between doubles the time over which drift and a real
  unbinding-rebinding can act, and the whole argument rests on "same
  molecule, same place".
* A pair belongs to one stratum, the one containing its mean z. Placing
  each member separately would count the pair twice and could put its
  two halves on either side of an edge; the mean is the best estimate
  of the depth the pair measures.
* Two estimates of sigma_z per stratum: the sd (the NeNA definition) and
  1.4826 * MAD (robust). A z on the wrong branch of the astigmatism
  calibration is a gross outlier of hundreds of nm; it inflates a sd and
  leaves a MAD almost untouched, so a gap between the two is itself a
  diagnostic.
* ``ratio = sigma_z / median lpz`` per stratum. Above 1 the file's lpz
  is optimistic and the leak estimates must use sigma_z; that is the
  number this module exists to put on the table.
* The leak fraction averages 1 - Phi over the cluster's own lpz values
  instead of plugging in their mean. The tail 1 - Phi(Delta / s) is
  convex in the width s only where the boundary is more than sqrt(2)
  widths away (Delta / s > sqrt 2): there the mean-lpz shortcut
  underestimates the leak (Jensen 1906); closer to the boundary the
  tail is concave and the shortcut overestimates it.

Nothing here draws random numbers; two runs on the same input are
bit-identical.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, norm

from tools.mps_paint import link_localizations

__all__ = [
    "AxialNena",
    "DEFAULT_MIN_PAIRS",
    "DEFAULT_Z_BIN_NM",
    "MAD_TO_SIGMA",
    "axial_nena",
    "calibrated_lpz_nm",
    "calibration_range_flags",
    "uncalibrated_label_pattern",
    "UNCALIBRATED_LABELS_ENV",
    "boundary_verdict",
    "BoundaryQuality",
    "RingQuality",
    "ZQuality",
    "ZQualityThresholds",
    "z_quality",
    "z_selection_key",
    "AxonViabilityV2",
    "EdgeTruncation",
    "FMin",
    "LimitingFactor",
    "PairViabilityV2",
    "RingViabilityV2",
    "VIABILITY_RULE_V2",
    "central_ring_position",
    "f_min_of_x",
    "verdict_v2",
    "viability_v2",
    "viability_v2_key",
    "z_edge_truncation",
    "leak_fraction",
    "spurious_cluster_probability",
    "structural_width_nm",
    "AGGREGATIONS",
    "CopyTolerance",
    "DELTA_MARGINAL_SD",
    "DELTA_VIABLE_SD",
    "FMinCalibration",
    "P_REF_DEFAULT_NM",
    "P_REF_PRESETS",
    "P_REF_RANGE_NM",
    "P_REF_SOURCE",
    "REFERENCE_RINGS",
    "SIZER_MAP_H_OVER_P",
    "SizerMapRow",
    "VIABILITY_RANGES",
    "ViabilityParams",
    "X_STATISTICS",
    "axon_period_nm",
    "axon_verdict",
    "calibrate_count_floor",
    "column_test_tau_nm",
    "copy_tolerance",
    "d_equivalents",
    "group_z_scale_of",
    "guard_star_nm",
    "guard_z_star",
    "peak_location_sd_nm",
    "reference_ring_position",
    "scale_period_nm",
    "silverman_reading",
    "valley_depth_equal",
]

# Stratum width. The astigmatic precision changes over hundreds of nm,
# so 50 nm resolves its depth dependence while a typical picked axon
# (thousands of localizations over ~+-400 nm) still puts hundreds of
# pairs into each stratum.
DEFAULT_Z_BIN_NM = 50.0
# The sd of a sd estimate on n independent values is sigma / sqrt(2 n):
# 10 % at n = 50, the coarsest a per-stratum number is still worth
# printing. Below that the stratum reports NaN rather than a guess.
DEFAULT_MIN_PAIRS = 50
# 1.4826 * MAD estimates sigma for Gaussian residuals.
MAD_TO_SIGMA = 1.4826
# Both localizations of a pair carry an independent error of the same
# sigma, so var(dz) = 2 sigma^2 and sigma = sd(dz) / sqrt(2).
_SQRT2 = math.sqrt(2.0)
# Relative tolerance for "the range is an exact multiple of the bin":
# floating-point division noise is ~1e-15, and the harness compares the
# last edge to the range end at 1e-9 nm absolute, so 1e-12 of a bin
# (5e-11 nm at 50 nm) sits safely between the two.
_EDGE_TOLERANCE = 1e-12


@dataclass
class AxialNena:
    """
    Axial precision per depth stratum, from consecutive-frame pairs.

    ``sigma_z_nm`` is the NeNA-type estimate sd(dz) / sqrt(2) over the
    pairs whose mean z falls in the stratum, ``sigma_z_robust_nm`` its
    MAD-based counterpart, and ``ratio`` the first divided by the median
    lpz the file reports for the localizations of those pairs. A ratio
    above 1 means the reported precision is optimistic.

    ``n_pairs_total`` and ``n_events_total`` count every consecutive-
    frame pair and every link event (single-frame events included) in
    the input, whatever ``z_range_nm`` was; the per-stratum counts only
    cover the range. The ``*_overall`` numbers are over all pairs.
    """

    z_bin_edges_nm: NDArray[np.float64]      # (S+1,)
    z_bin_centre_nm: NDArray[np.float64]     # (S,)
    n_pairs: NDArray[np.int64]               # (S,) consecutive-frame pairs per stratum
    sigma_z_nm: NDArray[np.float64]          # (S,) sd(dz)/sqrt(2); NaN where n_pairs < min_pairs
    sigma_z_robust_nm: NDArray[np.float64]   # (S,) 1.4826*MAD(dz)/sqrt(2); NaN likewise
    lpz_median_nm: NDArray[np.float64]       # (S,) median reported lpz of the pair members; NaN if lpz absent
    ratio: NDArray[np.float64]               # (S,) sigma_z_nm / lpz_median_nm
    sigma_z_overall_nm: float                # over all pairs
    lpz_median_overall_nm: float
    n_pairs_total: int
    n_events_total: int
    link_radius_nm: float
    min_pairs: int
    warnings: List[str] = field(default_factory=list)


# ===================================================================
#  Helpers
# ===================================================================
def _stratum_edges(
    z_nm: NDArray[np.float64],
    z_bin_nm: float,
    z_range_nm: Optional[Tuple[float, float]],
    warnings: List[str],
) -> NDArray[np.float64]:
    """
    Edges of ``z_bin_nm``-wide strata covering the range.

    The first edge is the range start and the last edge is EXACTLY the
    range end, so that on the default range (min z, max z) every pair's
    mean z lies inside some stratum. When the range is not a multiple of
    the bin the last stratum is the narrower remainder: a partial edge
    stratum with its own count is more honest than silently extending
    the range or dropping the tail.
    """
    if not np.isfinite(z_bin_nm) or z_bin_nm <= 0.0:
        raise ValueError(f"z_bin_nm must be a positive width, got {z_bin_nm!r}")
    if z_range_nm is None:
        finite = z_nm[np.isfinite(z_nm)]
        if finite.size == 0:
            warnings.append("no finite z values: a single empty stratum is reported")
            lo, hi = 0.0, 0.0
        else:
            lo, hi = float(finite.min()), float(finite.max())
    else:
        lo, hi = float(z_range_nm[0]), float(z_range_nm[1])
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
            raise ValueError(
                f"z_range_nm must be finite with zmin <= zmax, got {z_range_nm!r}"
            )
    if hi <= lo:
        warnings.append(
            f"z range has zero width at {lo:.1f} nm: one stratum of "
            f"{z_bin_nm:g} nm from there"
        )
        return np.array([lo, lo + z_bin_nm], dtype=float)

    n_full = int(math.floor((hi - lo) / z_bin_nm + _EDGE_TOLERANCE))
    edges = lo + z_bin_nm * np.arange(n_full + 1, dtype=float)
    if hi - edges[-1] > _EDGE_TOLERANCE * z_bin_nm:
        edges = np.append(edges, hi)
    else:
        # An exact multiple up to rounding: snap rather than add a sliver.
        edges[-1] = hi
    return np.asarray(edges, dtype=float)


def _consecutive_pairs(
    event: NDArray[np.int64],
    frame: NDArray[np.int64],
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    Indices (first, second) of every pair of localizations of one event
    whose frames differ by exactly 1, in input indices.

    Sorting by (event, frame) makes the members of each pair adjacent
    rows, so the whole extraction is one vectorised comparison. With
    ``max_dark_time = 0`` the link never puts two localizations of the
    same frame, or a gap, inside one event; the frame test is kept
    anyway so the rule stated in the docstring is the rule enforced.
    """
    if event.size < 2:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    order = np.lexsort((frame, event))
    e = event[order]
    f = frame[order]
    adjacent = (e[:-1] == e[1:]) & (e[:-1] >= 0) & ((f[1:] - f[:-1]) == 1)
    where = np.nonzero(adjacent)[0]
    return order[where].astype(np.intp), order[where + 1].astype(np.intp)


def _sd_sigma_nm(dz_nm: NDArray[np.float64]) -> float:
    """sd(dz) / sqrt(2). Population sd: its bias is 1/(4n), 0.5 % at n = 50."""
    return float(np.std(dz_nm)) / _SQRT2


def _robust_sigma_nm(dz_nm: NDArray[np.float64]) -> float:
    """1.4826 * MAD(dz) / sqrt(2), the MAD taken about the median of dz."""
    centre = float(np.median(dz_nm))
    return MAD_TO_SIGMA * float(np.median(np.abs(dz_nm - centre))) / _SQRT2


def _median_or_nan(values: NDArray[np.float64]) -> float:
    """Median of the finite values; NaN when there are none."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


# ===================================================================
#  Axial NeNA per stratum
# ===================================================================
def axial_nena(
    frame: NDArray[np.int64],
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    *,
    lpz_nm: Optional[NDArray[np.float64]] = None,
    link_radius_nm: float,
    z_bin_nm: float = DEFAULT_Z_BIN_NM,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    z_range_nm: Optional[Tuple[float, float]] = None,
) -> AxialNena:
    """
    Axial localization precision per depth stratum, from the data.

    Localizations are linked into events with
    ``mps_paint.link_localizations`` at ``max_dark_time = 0``, so an
    event is a run of consecutive frames within ``link_radius_nm`` of
    each other laterally. Within each event every pair of localizations
    whose frames differ by exactly 1 contributes dz = z(t+1) - z(t), and
    the pair is assigned to the stratum containing its mean z. Both
    members carry independent axial errors of the same sigma, so
    var(dz) = 2 sigma^2 and sigma_z = sd(dz) / sqrt(2); the structural
    width of the ring does not enter, because it is the same molecule.

    Parameters
    ----------
    frame : (N,) frame index of each localization.
    x_nm, y_nm, z_nm : (N,) coordinates in nanometres. Non-finite z
        values cannot form a pair and are dropped with a warning.
    lpz_nm : (N,) axial precision the file reports, or None. Without
        it ``lpz_median_nm``, ``ratio`` and ``lpz_median_overall_nm``
        are NaN; nothing is guessed.
    link_radius_nm : maximum lateral step between the two localizations
        of a pair. A consecutive-frame step of one molecule is Rayleigh
        with scale sqrt(2) * lp, so a few lateral precisions keep every
        genuine pair while staying far below the distance to another
        molecule in the next frame.
    z_bin_nm : stratum width; see DEFAULT_Z_BIN_NM.
    min_pairs : strata with fewer pairs report NaN (and are listed in a
        warning) instead of a number whose sampling error would exceed
        what it is meant to resolve; see DEFAULT_MIN_PAIRS. The same
        rule applies to the overall estimate. Values below 1 count as 1.
    z_range_nm : (zmin, zmax) covered by the strata; defaults to the
        min and max of the finite z values. Pairs whose mean z is
        outside it are counted in ``n_pairs_total`` and in the overall
        numbers but in no stratum.

    Returns
    -------
    AxialNena
        Per-stratum counts and estimates, the overall numbers and the
        parameters used. ``lpz_median_nm`` is the median over both
        members of every pair in the stratum (a localization in two
        pairs of the same stratum enters twice, so each pair weighs the
        same as in sigma_z); it is reported from a single pair onwards,
        because it describes what the file says rather than estimating
        anything, whereas ``sigma_z_nm`` honours ``min_pairs``.
    """
    warnings: List[str] = []
    frame_arr = np.asarray(frame, dtype=np.int64).ravel()
    x = np.asarray(x_nm, dtype=float).ravel()
    y = np.asarray(y_nm, dtype=float).ravel()
    z = np.asarray(z_nm, dtype=float).ravel()
    n = frame_arr.size
    if not (x.size == y.size == z.size == n):
        raise ValueError(
            f"frame, x_nm, y_nm and z_nm must have the same length, got "
            f"{n}, {x.size}, {y.size}, {z.size}"
        )
    lpz: Optional[NDArray[np.float64]] = None
    if lpz_nm is not None:
        lpz = np.asarray(lpz_nm, dtype=float).ravel()
        if lpz.size != n:
            raise ValueError(f"lpz_nm has length {lpz.size}, expected {n}")
    if not np.isfinite(link_radius_nm) or link_radius_nm <= 0.0:
        raise ValueError(f"link_radius_nm must be positive, got {link_radius_nm!r}")
    min_pairs_int = int(min_pairs)
    effective_min = max(min_pairs_int, 1)

    edges = _stratum_edges(z, float(z_bin_nm), z_range_nm, warnings)
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_strata = int(centres.size)

    # --- events and pairs -------------------------------------------
    if n > 0:
        event = link_localizations(frame_arr, x, y, float(link_radius_nm),
                                   max_dark_time=0)
        n_events_total = int(np.unique(event[event >= 0]).size)
    else:
        event = np.empty(0, dtype=np.int64)
        n_events_total = 0
    first, second = _consecutive_pairs(event, frame_arr)
    dz = z[second] - z[first]
    finite = np.isfinite(dz)
    if not np.all(finite):
        warnings.append(
            f"{int((~finite).sum())} consecutive-frame pairs with a "
            f"non-finite z were dropped"
        )
        first, second, dz = first[finite], second[finite], dz[finite]
    mean_z = 0.5 * (z[first] + z[second])
    n_pairs_total = int(first.size)

    # --- stratum of each pair: the bin holding its mean z --------------
    stratum = np.digitize(mean_z, edges) - 1
    stratum[mean_z == edges[-1]] = n_strata - 1     # top edge inclusive
    in_range = (stratum >= 0) & (stratum < n_strata)
    n_pairs = np.bincount(stratum[in_range], minlength=n_strata).astype(np.int64)

    sigma_z = np.full(n_strata, np.nan)
    sigma_z_robust = np.full(n_strata, np.nan)
    lpz_median = np.full(n_strata, np.nan)
    for s in range(n_strata):
        sel = in_range & (stratum == s)
        if n_pairs[s] >= effective_min:
            sigma_z[s] = _sd_sigma_nm(dz[sel])
            sigma_z_robust[s] = _robust_sigma_nm(dz[sel])
        if lpz is not None and n_pairs[s] > 0:
            members = np.concatenate([lpz[first[sel]], lpz[second[sel]]])
            lpz_median[s] = _median_or_nan(members)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sigma_z / lpz_median

    # --- overall ---------------------------------------------------------
    if n_pairs_total >= effective_min:
        sigma_z_overall = _sd_sigma_nm(dz)
    else:
        sigma_z_overall = float("nan")
    if lpz is not None and n_pairs_total > 0:
        lpz_median_overall = _median_or_nan(
            np.concatenate([lpz[first], lpz[second]])
        )
    else:
        lpz_median_overall = float("nan")

    # --- warnings ----------------------------------------------------------
    if n_pairs_total == 0:
        warnings.append(
            f"no consecutive-frame pairs within {link_radius_nm:g} nm: "
            f"axial NeNA unavailable (every estimate is NaN)"
        )
    elif n_pairs_total < effective_min:
        warnings.append(
            f"only {n_pairs_total} consecutive-frame pairs in total (need "
            f"{effective_min}): the overall sigma_z is NaN"
        )
    sparse = np.nonzero(n_pairs < effective_min)[0]
    if sparse.size:
        listed = ", ".join(f"{centres[s]:.0f}" for s in sparse)
        warnings.append(
            f"{sparse.size} of {n_strata} strata have fewer than "
            f"{effective_min} consecutive-frame pairs and report NaN "
            f"(centres {listed} nm)"
        )
    n_outside = int((~in_range).sum())
    if n_outside:
        warnings.append(
            f"{n_outside} of {n_pairs_total} pairs have a mean z outside "
            f"[{edges[0]:.0f}, {edges[-1]:.0f}] nm and fall in no stratum"
        )
    if lpz is None:
        warnings.append(
            "no lpz given: lpz_median_nm and ratio are NaN (sigma_z_nm "
            "stands on its own)"
        )

    return AxialNena(
        z_bin_edges_nm=edges,
        z_bin_centre_nm=centres,
        n_pairs=n_pairs,
        sigma_z_nm=sigma_z,
        sigma_z_robust_nm=sigma_z_robust,
        lpz_median_nm=lpz_median,
        ratio=ratio,
        sigma_z_overall_nm=sigma_z_overall,
        lpz_median_overall_nm=lpz_median_overall,
        n_pairs_total=n_pairs_total,
        n_events_total=n_events_total,
        link_radius_nm=float(link_radius_nm),
        min_pairs=min_pairs_int,
        warnings=warnings,
    )


# ===================================================================
#  Leak across a ring boundary
# ===================================================================
def leak_fraction(
    z_centre_nm: float,
    boundary_z_nm: float,
    lpz_nm: NDArray[np.float64],
    sigma_struct_nm: float = 0.0,
) -> float:
    """
    Expected fraction of a cluster's localizations beyond a z boundary.

    Each localization i sits at the cluster centre with a Gaussian
    spread of sqrt(sigma_struct^2 + lpz_i^2) -- the structural width of
    the ring convolved with its own axial precision -- so its chance of
    being measured on the far side of the boundary is
    1 - Phi(|boundary - centre| / sqrt(sigma_struct^2 + lpz_i^2)). The
    cluster's leak is the mean of those chances. Averaging AFTER
    evaluating the tail matters where the spread of the precisions is
    large: the tail is convex in the width where |boundary - centre| /
    width > sqrt 2, and there the value at the mean lpz is a lower bound
    (Jensen 1906); nearer the boundary the tail is concave and the
    mean-lpz value is an upper bound.

    Parameters
    ----------
    z_centre_nm : axial centre of the cluster.
    boundary_z_nm : the ring boundary; only |boundary - centre| matters.
    lpz_nm : (n,) axial precision of each localization of the cluster.
        The empirical sigma_z from ``axial_nena`` belongs here whenever
        its ratio to lpz is above 1. Non-finite entries are ignored; a
        scalar is accepted.
    sigma_struct_nm : structural axial width of the ring around its
        centre, see ``structural_width_nm``. Zero treats the cluster as
        a point.

    Returns
    -------
    float
        Fraction in [0, 0.5]; 0.5 with the boundary at the centre; NaN
        when no finite lpz is available.
    """
    lpz = np.atleast_1d(np.asarray(lpz_nm, dtype=float)).ravel()
    lpz = lpz[np.isfinite(lpz)]
    if lpz.size == 0:
        return float("nan")
    delta = abs(float(boundary_z_nm) - float(z_centre_nm))
    scale = np.sqrt(float(sigma_struct_nm) ** 2 + lpz ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = delta / scale
    # A zero width leaks nothing unless the boundary is at the centre.
    t = np.where(scale > 0.0, t, np.inf if delta > 0.0 else 0.0)
    return float(np.mean(norm.sf(t)))


def spurious_cluster_probability(
    n_locs: int,
    leak_fraction: float,
    min_samples: int,
) -> float:
    """
    Probability that the leaked part of a cluster is itself a cluster.

    Each of the n localizations leaks independently with probability F,
    so the number that cross is Binomial(n, F); DBSCAN can promote them
    to a cluster of their own in the neighbouring slab once they reach
    ``min_samples``. P[Binomial(n, F) >= min_samples] is
    ``scipy.stats.binom.sf(min_samples - 1, n, F)``.

    Parameters
    ----------
    n_locs : localizations in the cluster.
    leak_fraction : F from ``leak_fraction``; NaN gives NaN.
    min_samples : the DBSCAN parameter in the neighbouring slab.

    Returns
    -------
    float
        Exactly 0.0 when n_locs < min_samples or F == 0, exactly 1.0
        when F == 1 (the tail is then closed-form and scipy agrees).
    """
    n = int(n_locs)
    m = int(min_samples)
    f = float(leak_fraction)
    if n < 0:
        raise ValueError(f"n_locs must be non-negative, got {n_locs!r}")
    if math.isnan(f):
        return float("nan")
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"leak_fraction must lie in [0, 1], got {leak_fraction!r}")
    if n < m or f == 0.0:
        return 0.0
    if f == 1.0:
        return 1.0
    return float(binom.sf(m - 1, n, f))


def structural_width_nm(sigma_component_nm: float, lpz_nm: float) -> float:
    """
    Structural axial width of a ring: what remains of a fitted component
    width once the localization precision is removed in quadrature.

    A fitted component is the true ring profile convolved with the
    axial error, so sigma_component^2 = sigma_struct^2 + lpz^2. When the
    precision is as wide as the component or wider, the ring is
    unresolved and its structural width is reported as 0, not as an
    imaginary number or NaN: 0 is the value every downstream formula can
    take, and it says "no structure resolved" honestly.

    Parameters
    ----------
    sigma_component_nm : sd of the fitted Gaussian component.
    lpz_nm : axial precision, ideally the empirical sigma_z.

    Returns
    -------
    float
        sqrt(max(sigma_component^2 - lpz^2, 0)); NaN if either is NaN.
    """
    sigma = float(sigma_component_nm)
    lpz = float(lpz_nm)
    if math.isnan(sigma) or math.isnan(lpz):
        return float("nan")
    excess = sigma * sigma - lpz * lpz
    return math.sqrt(excess) if excess > 0.0 else 0.0


# ===================================================================
#  The D-19 calibration per localization
# ===================================================================
def calibrated_lpz_nm(
    nena: AxialNena,
    z_lab_nm: NDArray[np.float64],
    lpz_nm: NDArray[np.float64],
    *,
    warnings: Optional[List[str]] = None,
) -> NDArray[np.float64]:
    """
    The reported lpz of each localization multiplied by the NeNA ratio
    sigma_z / lpz of the depth stratum holding its laboratory z (D-19,
    H5): the per-localization axial width the leak diagnostics should
    use instead of a single ``lpz_scale``, because the astigmatic
    precision can be honest on one side of focus and optimistic on the
    other (H1's pooled table), and a ring's leak into its neighbour
    depends on the depth it sits at.

    Parameters
    ----------
    nena : AxialNena
        The strata (``z_bin_edges_nm``, laboratory z) and their ``ratio``.
    z_lab_nm : (n,) laboratory z of each localization (the coordinate
        the strata were built on; NOT the axon-frame z').
    lpz_nm : (n,) reported axial precision of each localization.
    warnings : list, optional
        Receives one line per stratum whose ratio is NaN (its factor is
        1.0: the reported lpz taken at face value) and one when
        localizations fall outside the edges (nearest stratum).

    Returns
    -------
    (n,) float64
        lpz_nm x ratio[stratum]; a z below the first edge or above the
        last takes the nearest stratum (the calibration is not
        extrapolated, only held constant beyond the range); a z exactly
        on an inner edge belongs to the stratum above it
        (``numpy.digitize``), the last edge to the last stratum (not
        counted as outside); a ratio that is NaN, infinite or <= 0 (a
        degenerate stratum: ``sigma_z_robust_nm`` 0) maps to the factor
        1.0 -- a factor 0 would hand the leak diagnostics zero tail
        widths and NaN log ratios; a localization whose lab z is NaN
        (a filtered Picasso file can carry them) keeps its reported lpz
        (factor 1.0) and is counted in its own warning (review of
        2026-09-25). Nothing is drawn; the inputs are not modified.
    """
    z = np.asarray(z_lab_nm, dtype=float).ravel()
    lpz = np.asarray(lpz_nm, dtype=float).ravel()
    if z.size != lpz.size:
        raise ValueError(f"z_lab_nm and lpz_nm must have the same length, got {z.size} and {lpz.size}")
    edges = np.asarray(nena.z_bin_edges_nm, dtype=float).ravel()
    ratio = np.asarray(nena.ratio, dtype=float).ravel()
    n_strata = int(ratio.size)
    if edges.size != n_strata + 1 or n_strata < 1:
        raise ValueError(f"nena has {edges.size} edges for {n_strata} ratios")
    usable = np.isfinite(ratio) & (ratio > 0.0)
    factor = np.where(usable, ratio, 1.0)
    z_ok = np.isfinite(z)
    stratum = np.digitize(np.where(z_ok, z, edges[0]), edges) - 1
    stratum = np.where(z_ok & (z == edges[-1]), n_strata - 1, stratum)
    outside = int(np.count_nonzero(z_ok & ((stratum < 0) | (stratum > n_strata - 1))))
    stratum = np.clip(stratum, 0, n_strata - 1)
    per_loc = factor[stratum]
    per_loc = np.where(z_ok, per_loc, 1.0)
    n_nan_z = int(np.count_nonzero(~z_ok))
    if warnings is not None:
        bad_strata = np.nonzero(~usable)[0]
        if bad_strata.size:
            centres = 0.5 * (edges[:-1] + edges[1:])
            listed = ", ".join(f"{centres[s]:.0f}" for s in bad_strata)
            n_hit = int(np.count_nonzero(np.isin(stratum, bad_strata) & z_ok))
            warnings.append(
                f"{bad_strata.size} of {n_strata} strata have no usable NeNA ratio (NaN or <= 0; centres {listed} nm): "
                f"factor 1.0 for the {n_hit} localizations there (reported lpz taken at face value)"
            )
        if outside:
            warnings.append(
                f"{outside} of {z.size} localizations have a lab z outside [{edges[0]:.0f}, {edges[-1]:.0f}] nm "
                "and take the nearest stratum's ratio"
            )
        if n_nan_z:
            warnings.append(f"{n_nan_z} of {z.size} localizations have a NaN lab z: factor 1.0 (reported lpz kept)")
    return np.asarray(lpz * per_loc, dtype=np.float64)


# ===================================================================
#  z-information quality of every ring pair (D-39, the first filter)
# ===================================================================
# The thresholds of the first z-quality filter (separation, leak copies,
# information ratio) were chosen from a simulation study whose generator was
# parameterised on a pilot acquisition (one contour, its period, cluster
# sizes and precisions); no column statistic of a real axon entered their
# choice. They are suggested defaults, labelled with that origin and
# editable (``ViabilityParams``). The separation index
# d = |mu2 - mu1| / sqrt(s1^2 + s2^2) is the ring gap over the combined
# axial width: two equal rings are bimodal only above d = sqrt 2 (Schilling
# et al. 2002); 2.2 / 2.0 are sigma/P = 0.32 / 0.35 at equal widths, valley
# depth 41 % / 28 % (``d_equivalents``). The information ratio asks for twice
# the events that give an expected Delta BIC of 10 (Kass & Raftery 1995).
ZQ_D_VIABLE = 2.2
ZQ_D_MARGINAL = 2.0
ZQ_SPUR_VIABLE = 0.02
ZQ_SPUR_MARGINAL = 0.05
ZQ_INFO_RATIO_MIN = 2.0
ZQ_DELTA_BIC = 10.0
ZQ_EXTRA_PARAMS = 3
# A component wider than P/2 is unresolved (possibly two merged rings);
# a spacing over 1.3 P suggests a ring lost in between.
ZQ_MERGED_SIGMA_OVER_P = 0.5
ZQ_MERGED_GAP_OVER_P = 1.3
# A localization is "ambiguous" when its two-component posterior is
# below this.
ZQ_POSTERIOR_MIN = 0.85
# The reference period P_ref: the external yardstick of the merge flags
# (a component wider than 0.5 P_ref cannot be one ring of a bimodal pair; a
# gap over 1.3 P_ref suggests a lost ring), which must not depend on the axon
# they check. Default: 170 nm, the mean inter-peak distance along z reported
# for mouse sciatic nerve axons (Gazal et al. 2026, 170 +/- 15 nm); presets for
# other preparations (``P_REF_PRESETS``) and a free value in 150-210 nm. The
# scale-type quantities (SiZer bandwidth, peak window, width over period) use
# the axon's own measured period (``axon_period_nm``) and fall back to P_ref.
P_REF_DEFAULT_NM = 170.0
P_REF_SOURCE = "literature: Gazal et al. 2026, sciatic nerve axial period 170 nm"
P_REF_RANGE_NM = (150.0, 210.0)
# (label, value nm, source words); the first is the default.
P_REF_PRESETS: Tuple[Tuple[str, float, str], ...] = (
    ("sciatic nerve, axial (170 nm, Gazal et al. 2026)", 170.0,
     "literature: Gazal et al. 2026, sciatic nerve axial period 170 nm"),
    ("sciatic nerve, lateral STED, mean (186 nm, D'Este et al. 2016)", 186.0,
     "literature: D'Este et al. 2016, sciatic nerve lateral STED period 186 nm (mean)"),
    ("cultured neurons (182 nm, Xu et al. 2013)", 182.0,
     "literature: Xu et al. 2013, cultured hippocampal neurons 182 nm"),
    ("cultured neurons (190 nm, Zhong et al. 2014)", 190.0,
     "literature: Zhong et al. 2014, cultured neurons ~190 nm"),
)
VERDICT_VIABLE = "viable"
VERDICT_MARGINAL = "marginal"
VERDICT_NOT_VIABLE = "not viable"
AXON_FEWER_THAN_TWO_RINGS = "not viable (fewer than 2 rings)"
Z_SELECTION_MODES = ("viable", "viable+marginal")
# Q-28 / D-38(f): inputs from an acquisition known to lie outside the
# calibrated range (closure defect) carry a flag that never blocks and is
# shown next to the result. Which inputs those are is site knowledge, not
# code: a regular expression matched against the label / source path, from
# ``ROI2_PATTERN`` when a caller sets it, else from the environment
# variable ``UNCALIBRATED_LABELS_ENV`` when it is set (even empty), else
# from ``uncalibrated_labels`` in the local settings file
# (tools.mps_settings); none of them (the default): no flag.
UNCALIBRATED_LABELS_ENV = "MPS_UNCALIBRATED_LABELS"
ROI2_PATTERN: Optional["re.Pattern[str]"] = None
ROI2_FLAG = "outside the calibrated range (flagged acquisition)"
# The (d_sep, exp_spur_frac) range of the accepted S4 originals, filled
# in H5-E S5 as {"d_sep_min": ..., "exp_spur_frac_max": ...}; None until
# then (no envelope flag is raised).
CALIBRATED_ENVELOPE: Optional[Dict[str, float]] = None


def uncalibrated_label_pattern() -> Optional["re.Pattern[str]"]:
    """The pattern of ``calibration_range_flags``' acquisition flag: ``ROI2_PATTERN`` when set, else the regular
    expression in the ``MPS_UNCALIBRATED_LABELS`` environment variable when it is set (an empty value flags nothing),
    else ``uncalibrated_labels`` of the local settings file (``tools.mps_settings``, next to the program), else None
    (no input is flagged). Both are read at call time, so worker processes see them. An invalid expression in the
    variable raises ValueError naming it; one in the settings file was already dropped, with a warning, on loading."""
    if ROI2_PATTERN is not None:
        return ROI2_PATTERN
    if UNCALIBRATED_LABELS_ENV in os.environ:
        text = os.environ[UNCALIBRATED_LABELS_ENV].strip()
    else:
        from tools.mps_settings import load_settings     # lazy: no settings import for callers that set the pattern
        text = load_settings().uncalibrated_labels.strip()
    if not text:
        return None
    try:
        return re.compile(text)
    except re.error as exc:
        raise ValueError(f"{UNCALIBRATED_LABELS_ENV} is not a valid regular expression: {exc}") from exc

_MIN_SIGMA_NM = 1e-6
# The keep floor of the expected-copies count: n0 = n / keep is capped at 20 n,
# a numerical safeguard that does not act on realistic data (editable,
# ``ViabilityParams.keep_floor``).
_MIN_KEEP = 0.05
# The tolerated bias of the column excess, in null standard deviations, of the
# per-pair leak-copy tolerance c = delta sd* / (2 (1 - p0)) (viable / marginal).
DELTA_VIABLE_SD = 0.25
DELTA_MARGINAL_SD = 0.5
# The axial-profile SiZer bandwidths of the map shown beside the verdict.
SIZER_MAP_H_OVER_P: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)
# Silverman's critical-bandwidth reading: smoothed-bootstrap replicates and the
# largest sample read (deterministic subsample).
SILVERMAN_N_BOOT = 199
SILVERMAN_MAX_N = 4000
REFERENCE_RINGS: Tuple[str, ...] = ("central", "most populated", "pick")
X_STATISTICS: Tuple[str, ...] = ("mean", "median")
AGGREGATIONS: Tuple[str, ...] = ("any", "all")
# The documented range of every editable value (SR-8: outside it a value is
# flagged, never clamped). Keys are ``ViabilityParams`` fields.
VIABILITY_RANGES: Dict[str, Tuple[float, float]] = {
    "p_ref_nm": P_REF_RANGE_NM,
    "h_over_p": (0.05, 0.25),
    "peak_window_over_p": (0.2, 0.5),
    "d_viable": (1.41, 3.0),
    "d_marginal": (1.41, 3.0),
    "delta_viable_sd": (0.1, 0.5),
    "delta_marginal_sd": (0.25, 1.0),
    "recall_tolerance": (0.80, 0.95),
    "group_z_scale": (0.25, 1.0),
    "keep_floor": (0.01, 0.2),
    "v2c_h_over_p": (0.10, 0.30),
    "v2c_cut_flag_over_h": (0.5, 2.0),
    "v2c_sigma_floor_nm": (1.0, 20.0),
}


@dataclass(frozen=True)
class ViabilityParams:
    """Every editable value of the ring-pair viability (rule v2, the first z-quality filter, rule v2-clusters), one
    object read from one store (``tools.mps_settings.MPSSettings.viability_params``). The defaults are the declared
    ones; a value outside ``VIABILITY_RANGES`` is kept and flagged (``out_of_range``), never clamped; an impossible one
    (a non-positive period, a fraction outside [0, 1]) raises ValueError.

    p_ref_nm / p_ref_source: the reference period and its source words. h_over_p, peak_window_over_p: SiZer
    bandwidth and peak window as fractions of the axon's period. d_viable / d_marginal: separation thresholds.
    delta_viable_sd / delta_marginal_sd: the tolerated column-excess bias, in null sd, of the per-pair copy
    tolerance. tau_nm: the eclipse tolerance of the column test that tolerance refers to (None: the column test's
    tau_0, ``column_test_tau_nm``). recall_tolerance: the count floor's cluster recall (and v2c's k_min).
    reference_ring / reference_ring_index: the denominator of the count criterion. x_statistic: mean or median
    localizations per cluster. group_z_scale: the axial scale of the 3D variance groups (None: sigma_xy / sigma_z of
    the file, 0.5 without precisions). small_sample_correction: G / (G - 1) on the cluster-robust variance.
    keep_floor: the slab-retention floor of the copy count. v2c_*: rule v2-clusters' bandwidth, cut-flag width (in
    bandwidths) and sigma floor. aggregation: the axon verdict from 'any' pair or 'all' pairs."""

    p_ref_nm: float = P_REF_DEFAULT_NM
    p_ref_source: str = P_REF_SOURCE
    h_over_p: float = 0.10
    peak_window_over_p: float = 1.0 / 3.0
    d_viable: float = 2.2
    d_marginal: float = 2.0
    delta_viable_sd: float = DELTA_VIABLE_SD
    delta_marginal_sd: float = DELTA_MARGINAL_SD
    tau_nm: Optional[float] = None
    recall_tolerance: float = 0.90
    reference_ring: str = "central"
    reference_ring_index: Optional[int] = None
    x_statistic: str = "mean"
    group_z_scale: Optional[float] = None
    small_sample_correction: bool = False
    keep_floor: float = _MIN_KEEP
    v2c_h_over_p: float = 0.10
    v2c_cut_flag_over_h: float = 1.0
    v2c_sigma_floor_nm: float = 5.0
    aggregation: str = "any"

    def __post_init__(self) -> None:
        for name in ("p_ref_nm", "h_over_p", "peak_window_over_p", "d_viable", "d_marginal", "delta_viable_sd",
                     "delta_marginal_sd", "v2c_h_over_p", "v2c_cut_flag_over_h", "v2c_sigma_floor_nm"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be a positive number, got {getattr(self, name)!r}")
        for name in ("recall_tolerance", "keep_floor"):
            v = float(getattr(self, name))
            if not (0.0 < v <= 1.0):
                raise ValueError(f"{name} must lie in (0, 1], got {v!r}")
        if self.tau_nm is not None and not (math.isfinite(float(self.tau_nm)) and float(self.tau_nm) > 0.0):
            raise ValueError(f"tau_nm must be positive or None, got {self.tau_nm!r}")
        if self.group_z_scale is not None and not (math.isfinite(float(self.group_z_scale))
                                                   and float(self.group_z_scale) > 0.0):
            raise ValueError(f"group_z_scale must be positive or None, got {self.group_z_scale!r}")
        if self.reference_ring not in REFERENCE_RINGS:
            raise ValueError(f"reference_ring must be one of {REFERENCE_RINGS}, got {self.reference_ring!r}")
        if self.reference_ring == "pick" and self.reference_ring_index is None:
            raise ValueError("reference_ring 'pick' needs reference_ring_index")
        if self.x_statistic not in X_STATISTICS:
            raise ValueError(f"x_statistic must be one of {X_STATISTICS}, got {self.x_statistic!r}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")

    def thresholds(self) -> "ZQualityThresholds":
        """The first filter's thresholds (the fixed copy tiers stay the suggestion used without counts)."""
        return ZQualityThresholds(d_viable=float(self.d_viable), d_marginal=float(self.d_marginal))

    def out_of_range(self) -> Tuple[str, ...]:
        """The fields whose value lies outside its documented range (flagged, never clamped; SR-8), plus
        'd_marginal' when it exceeds d_viable."""
        out: List[str] = []
        for name, (lo, hi) in VIABILITY_RANGES.items():
            v = getattr(self, name)
            if v is None:
                continue
            if not (lo - 1e-12 <= float(v) <= hi + 1e-12):
                out.append(name)
        if float(self.d_marginal) > float(self.d_viable) and "d_marginal" not in out:
            out.append("d_marginal")
        return tuple(out)

    def to_store(self) -> Dict[str, Any]:
        """The fields that differ from the defaults (what the settings file keeps)."""
        base = ViabilityParams()
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    @classmethod
    def from_store(cls, data: Any) -> "ViabilityParams":
        """The parameters a settings entry describes: unknown keys and values of the wrong kind are dropped (one at a
        time, so one bad entry never loses the others); out-of-range values are KEPT (flagged by ``out_of_range``)."""
        if not isinstance(data, dict):
            return cls()
        names = {f.name for f in dataclasses.fields(cls)}
        kept: Dict[str, Any] = {}
        for k, v in data.items():
            if k not in names:
                continue
            try:
                cls(**dict(kept, **{k: v}))
            except (TypeError, ValueError):
                continue
            kept[k] = v
        return cls(**kept)

    def with_(self, **changes: Any) -> "ViabilityParams":
        return dataclasses.replace(self, **changes)


def axon_period_nm(rings: Sequence[Any]) -> float:
    """The axon's period: the median over consecutive listed rings (sorted by index) of centre spacing / index step,
    so a missing ring (index step 2) counts as one period, not two; NaN with fewer than 2 rings. ``rings``: objects
    with ``index`` and ``centre_z_nm``."""
    ordered = sorted(rings, key=lambda r: int(r.index))
    if len(ordered) < 2:
        return float("nan")
    centres = np.array([float(r.centre_z_nm) for r in ordered])
    idx = np.array([float(r.index) for r in ordered])
    return float(np.median(np.diff(centres) / np.diff(idx)))


def scale_period_nm(rings: Sequence[Any], p_ref_nm: float) -> Tuple[float, str]:
    """(P, source) of the period-scaled quantities: the axon's measured period when it has two rings or more (and it
    is positive), else the reference period (fallback)."""
    p = axon_period_nm(rings)
    if math.isfinite(p) and p > 0.0:
        return float(p), "measured on this axon"
    return float(p_ref_nm), "fallback: the reference period (fewer than two rings)"


def valley_depth_equal(sigma_over_p: float) -> float:
    """Valley depth of an equal-weight pair of Gaussian rings one period apart, relative to the density at a ring
    centre: 1 - phi(P / (2 s)) / (0.5 [phi(0) + phi(P / s)]); 0 when there is no valley (s / P >= 0.5)."""
    s = float(sigma_over_p)
    if not (math.isfinite(s) and s > 0.0):
        return float("nan")
    centre = 0.5 * (norm.pdf(0.0) + norm.pdf(1.0 / s))
    mid = norm.pdf(0.5 / s)
    return float(max(0.0, 1.0 - mid / centre))


def d_equivalents(d_sep: float) -> Tuple[float, float]:
    """(sigma / P, valley depth) of an equal-width pair one period apart at separation index d: sigma / P =
    1 / (d sqrt 2)."""
    d = float(d_sep)
    if not (math.isfinite(d) and d > 0.0):
        return float("nan"), float("nan")
    sp = 1.0 / (d * math.sqrt(2.0))
    return sp, valley_depth_equal(sp)


def guard_z_star(n0: int = 200, m: int = 10, prob: float = 0.05) -> float:
    """z* such that a cluster of n0 localizations leaking a fraction 1 - Phi(z*) makes a copy (>= m leaked) with
    probability ``prob``: F* solves P[Bin(n0, F*) >= m] = prob (bisection), z* = Phi^-1(1 - F*)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(binom.sf(m - 1, n0, mid)) > prob:
            hi = mid
        else:
            lo = mid
    return float(norm.isf(0.5 * (lo + hi)))


def guard_star_nm(s_eff_nm: float, period_nm: float, n0: int = 200, m: int = 10) -> float:
    """The guard g* = 2 (z* s_eff - P/2) (total width of the dead zone centred on a boundary) beyond which the
    residual leak of an n0-localization cluster no longer makes a copy with 5 % probability; <= 0: no guard needed."""
    s, p = float(s_eff_nm), float(period_nm)
    if not (math.isfinite(s) and math.isfinite(p)):
        return float("nan")
    return float(2.0 * (guard_z_star(n0, m) * s - 0.5 * p))


def column_test_tau_nm(path: Optional[str] = None) -> Optional[float]:
    """The eclipse tolerance the column test runs at: tau_0 of the column parameters file
    (``config/columns_params.yaml``), or None when it cannot be read."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                            "columns_params.yaml")
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return None
    hit = _TAU_CACHE.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        from tools.mps_columns import load_columns_params
        tau = float(load_columns_params(path).tau0_nm)
    except Exception:  # noqa: BLE001 - an unreadable file means "tau unknown": the fixed suggestions are used
        tau = float("nan")
    value = tau if math.isfinite(tau) and tau > 0.0 else None
    _TAU_CACHE[path] = (stamp, value)
    return value


_TAU_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}


@dataclass(frozen=True)
class CopyTolerance:
    """The tolerated leak-copy fraction of one ring pair: c = delta sd* / (2 (1 - p0)) at the viable and the marginal
    delta, with the null match rate p0 = 1 - exp(-2 tau K_other / L_other) (mean of the two directions) and sd* =
    sqrt(p0 (1 - p0) / min K) (binomial approximation before the null is run); ``source`` in words. When the counts,
    the ring lengths or tau are unknown, the fixed suggestions (2 % / 5 %) with ``per_pair`` False."""

    viable: float
    marginal: float
    p0: float
    sd_star: float
    per_pair: bool
    source: str


def copy_tolerance(k_a: int, k_b: int, length_a_nm: float, length_b_nm: float, tau_nm: Optional[float], *,
                   delta_viable_sd: float = DELTA_VIABLE_SD,
                   delta_marginal_sd: float = DELTA_MARGINAL_SD) -> CopyTolerance:
    """The per-pair copy tolerance (see ``CopyTolerance``). A leak copy sits at its parent's lateral position and
    adds one match unless the parent was already matched by chance (p0): copies = c (K_a + K_b) shift the
    standardised excess by about 2 c (1 - p0) / sd*, so tolerating delta null sd gives c = delta sd* / (2 (1 - p0))
    (toroidal-shift null, Lotwick & Silverman 1982)."""
    ka, kb = int(k_a), int(k_b)
    la, lb = float(length_a_nm), float(length_b_nm)
    if (tau_nm is None or not (math.isfinite(float(tau_nm)) and float(tau_nm) > 0.0) or min(ka, kb) < 1
            or not (math.isfinite(la) and math.isfinite(lb) and la > 0.0 and lb > 0.0)):
        why = ("tau unknown" if tau_nm is None or not math.isfinite(float(tau_nm)) else
               "a ring without clusters" if min(ka, kb) < 1 else "ring length unknown")
        return CopyTolerance(ZQ_SPUR_VIABLE, ZQ_SPUR_MARGINAL, float("nan"), float("nan"), False,
                             f"fixed suggestion {100 * ZQ_SPUR_VIABLE:g} % / {100 * ZQ_SPUR_MARGINAL:g} % ({why})")
    tau = float(tau_nm)
    p_ab = 1.0 - math.exp(-2.0 * tau * kb / lb)
    p_ba = 1.0 - math.exp(-2.0 * tau * ka / la)
    p0 = 0.5 * (p_ab + p_ba)
    min_k = min(ka, kb)
    sd = math.sqrt(max(p0 * (1.0 - p0), 0.0) / min_k)
    if 1.0 - p0 <= 1e-12:
        cv = cm = float("inf")
    else:
        cv = float(delta_viable_sd) * sd / (2.0 * (1.0 - p0))
        cm = float(delta_marginal_sd) * sd / (2.0 * (1.0 - p0))
    return CopyTolerance(cv, cm, p0, sd, True,
                         f"per pair: {float(delta_viable_sd):g} / {float(delta_marginal_sd):g} null sd at tau "
                         f"{tau:.1f} nm (p0 {p0:.2f}, min K {min_k})")


@dataclass(frozen=True)
class ZQualityThresholds:
    """The first filter's verdict thresholds (suggestions from a pilot-parameterised simulation, editable through
    ``ViabilityParams``). ``spur_viable`` / ``spur_marginal`` are the fixed copy tiers, used only when a pair's own
    tolerance cannot be computed (``copy_tolerance``)."""

    d_viable: float = ZQ_D_VIABLE
    d_marginal: float = ZQ_D_MARGINAL
    spur_viable: float = ZQ_SPUR_VIABLE
    spur_marginal: float = ZQ_SPUR_MARGINAL
    info_ratio_min: float = ZQ_INFO_RATIO_MIN
    merged_sigma_over_p: float = ZQ_MERGED_SIGMA_OVER_P
    merged_gap_over_p: float = ZQ_MERGED_GAP_OVER_P


@dataclass(frozen=True)
class RingQuality:
    """One ring's axial description: the fitted width of its component,
    the median reported lpz, the structural width left once lpz is
    removed in quadrature, its size in localizations, events and
    clusters, sigma_z / P_ref and the merged-ring flags."""

    index: int
    centre_z_nm: float
    sigma_z_nm: float
    lpz_median_nm: float
    sigma_struct_nm: float
    n_locs: int
    n_events: Optional[int]
    k_clusters: int
    sigma_over_p: float
    flags: Tuple[str, ...]


@dataclass(frozen=True)
class BoundaryQuality:
    """One consecutive ring pair (ring_a below ring_b).

    d_sep is the separation index of the two components (Ashman's D /
    sqrt 2); f_lo and f_hi the analytic leak of each past the cut;
    exp_spur_frac the expected fraction of the pair's clusters that are
    leak copies (D-27c: sum over clusters of P[Bin(n0, F) >= min_samples],
    n0 = n / kept fraction); kl_nats the KL divergence of the mixture from
    its moment-matched Gaussian; n_req_events the events for an expected
    Delta BIC >= 10; info_ratio = n_events / n_req_events (NaN without
    events). ``verdict`` is VIABLE / MARGINAL / NOT_VIABLE, ``reasons``
    says why it is not better."""

    ring_a: int
    ring_b: int
    d_sep: float
    ashman_d: float
    valley_depth: float
    gap_nm: float
    cut_nm: float
    f_lo: float
    f_hi: float
    misassign: float
    ambiguous_frac: float
    k_a: int
    k_b: int
    exp_spur_frac: float
    kl_nats: float
    n_req_events: float
    n_events: Optional[int]
    info_ratio: float
    verdict: str
    reasons: Tuple[str, ...]
    # the pair's tolerated copy fraction (viable / marginal), the null match rate and sd behind it, and its source
    spur_viable: float = ZQ_SPUR_VIABLE
    spur_marginal: float = ZQ_SPUR_MARGINAL
    tolerance_p0: float = float("nan")
    tolerance_sd: float = float("nan")
    tolerance_source: str = ""


@dataclass(frozen=True)
class ZQuality:
    """``z_quality``'s answer for one axon: rings (bottom to top), their
    boundaries, the axon verdict (the best pair's), the viable and
    marginal pairs, the P reference with its source, the min_samples and
    thresholds used, and warnings."""

    rings: Tuple[RingQuality, ...]
    boundaries: Tuple[BoundaryQuality, ...]
    axon_verdict: str
    viable_pairs: Tuple[Tuple[int, int], ...]
    marginal_pairs: Tuple[Tuple[int, int], ...]
    p_ref_nm: float
    p_ref_source: str
    min_samples: int
    thresholds: ZQualityThresholds
    warnings: Tuple[str, ...]
    params: Optional["ViabilityParams"] = None
    tau_nm: Optional[float] = None


def kl_mixture_vs_gauss(mu: NDArray[np.float64], sd: NDArray[np.float64], w: NDArray[np.float64],
                        n_grid: int = 4001) -> float:
    """
    KL(p || q) in nats of a Gaussian mixture p = sum w N(mu, sd) from q,
    the Gaussian with p's mean and variance (the best single Gaussian in
    the KL(p || .) sense), by the trapezoid rule on +-8 sd.

    It is the expected log-likelihood gain per observation of two
    components over one, the quantity the information ratio needs.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64) / float(np.sum(w))
    m = float(np.sum(w * mu))
    v = float(np.sum(w * (sd ** 2 + mu ** 2)) - m * m)
    s = math.sqrt(max(v, _MIN_SIGMA_NM))
    lo, hi = float(np.min(mu - 8 * sd)), float(np.max(mu + 8 * sd))
    z = np.linspace(lo, hi, n_grid)
    p = np.sum(w[:, None] * norm.pdf(z[None, :], mu[:, None], np.maximum(sd, _MIN_SIGMA_NM)[:, None]), axis=0)
    q = norm.pdf(z, m, s)
    ok = p > 1e-300
    f = np.zeros_like(z)
    f[ok] = p[ok] * (np.log(p[ok]) - np.log(np.maximum(q[ok], 1e-300)))
    return float(np.trapezoid(f, z))


def n_required(kl: float, delta_bic: float = ZQ_DELTA_BIC, extra_params: int = ZQ_EXTRA_PARAMS) -> float:
    """Smallest n with 2 n KL - extra_params ln n >= delta_bic: the events
    that give an expected BIC gain of two components over one of
    ``delta_bic``. Geometric search (x1.02) then 40 bisections; inf when
    KL is not positive."""
    if not math.isfinite(kl) or kl <= 0:
        return math.inf
    n = 2.0
    while n < 1e8:
        if 2.0 * n * kl - extra_params * math.log(n) >= delta_bic:
            break
        n *= 1.02
    a, b = n / 1.02, n
    for _ in range(40):
        c = 0.5 * (a + b)
        if 2.0 * c * kl - extra_params * math.log(c) >= delta_bic:
            b = c
        else:
            a = c
    return float(b)


def boundary_verdict(*, d_sep: float, exp_spur_frac: float, info_ratio: float, valley_depth: float, gap_nm: float,
                     ring_flags: Sequence[Sequence[str]], rings: Tuple[int, int], p_ref_nm: float,
                     thresholds: ZQualityThresholds = ZQualityThresholds(), spur_viable: Optional[float] = None,
                     spur_marginal: Optional[float] = None) -> Tuple[str, Tuple[str, ...]]:
    """
    The first filter's verdict of one pair, with its reasons.

    VIABLE: an interior valley, no merge block, d >= d_viable, expected
    copies <= the viable tolerance and (without events, not applied) info
    ratio >= info_ratio_min. MARGINAL: valley, no block, d >= d_marginal
    and copies <= the marginal tolerance. Otherwise NOT VIABLE. A block is
    a spacing over merged_gap_over_p x P_ref (a lost ring) or a ring of the
    pair wider than merged_sigma_over_p x P_ref (``ring_flags``, one tuple
    per ring). ``spur_viable`` / ``spur_marginal``: the pair's own copy
    tolerance (``copy_tolerance``); None -> the fixed tiers of ``thresholds``.
    """
    t = thresholds
    sv = float(t.spur_viable if spur_viable is None else spur_viable)
    sm = float(t.spur_marginal if spur_marginal is None else spur_marginal)
    reasons: List[str] = []
    valley = math.isfinite(valley_depth) and valley_depth > 0
    if not valley:
        reasons.append("no interior valley")
    gap_block = gap_nm > t.merged_gap_over_p * p_ref_nm
    if gap_block:
        reasons.append(f"gap {gap_nm:.0f} nm > {t.merged_gap_over_p:g} P_ref (lost ring?)")
    for k, flags in zip(rings, ring_flags):
        reasons += [f"ring {k}: {m}" for m in flags]
    blocked = gap_block or any(len(f) > 0 for f in ring_flags)
    info_ok = not math.isfinite(info_ratio) or info_ratio >= t.info_ratio_min
    if valley and not blocked and d_sep >= t.d_viable and exp_spur_frac <= sv and info_ok:
        verdict = VERDICT_VIABLE
    elif valley and not blocked and d_sep >= t.d_marginal and exp_spur_frac <= sm:
        verdict = VERDICT_MARGINAL
    else:
        verdict = VERDICT_NOT_VIABLE
    if d_sep < t.d_marginal:
        reasons.append(f"d_sep {d_sep:.2f} < {t.d_marginal:g}")
    elif d_sep < t.d_viable:
        reasons.append(f"d_sep {d_sep:.2f} < {t.d_viable:g}")
    if exp_spur_frac > sm:
        reasons.append(f"expected leak copies {100 * exp_spur_frac:.1f} % > {_tol_text(sm)}")
    elif exp_spur_frac > sv:
        reasons.append(f"expected leak copies {100 * exp_spur_frac:.1f} % > {_tol_text(sv)}")
    if verdict != VERDICT_VIABLE and math.isfinite(info_ratio) and info_ratio < t.info_ratio_min:
        reasons.append(f"events {info_ratio:.2f} x n_req < {t.info_ratio_min:g}")
    return verdict, tuple(reasons)


def _tol_text(c: float) -> str:
    """A tolerance in percent, three significant digits ("2 %", "1.37 %")."""
    if not math.isfinite(c):
        return "any"
    return f"{100 * c:.3g} %"


def _expected_copies(ring: Any, f_to: float, mu_all: NDArray[np.float64], sd_all: NDArray[np.float64],
                     min_samples: int, keep_floor: float = _MIN_KEEP) -> float:
    """The expected leak copies of one ring: sum over its clusters of P[Bin(n0, F) >= min_samples] (Ester et al.
    1996: the DBSCAN core count), n0 = n_locs / kept fraction of the ring's component inside its own slab (an
    inclusion-probability correction, Horvitz & Thompson 1952), the kept fraction floored at ``keep_floor``. Every
    cluster is evaluated at its ring's centre (flat-ring assumption): off-centre clusters can make it
    under-estimate."""
    c = int(ring.component_index)
    m, s = float(mu_all[c]), max(float(sd_all[c]), _MIN_SIGMA_NM)
    keep = float(norm.cdf((float(ring.z_hi_nm) - m) / s) - norm.cdf((float(ring.z_lo_nm) - m) / s))
    keep = min(max(keep, float(keep_floor)), 1.0)
    return float(sum(float(binom.sf(min_samples - 1, int(round(int(cl.n_locs) / keep)), f_to)) for cl in ring.clusters))


def z_quality(res: Any, *, p_ref_nm: Optional[float] = None, min_samples: Optional[int] = None,
              posterior_min: float = ZQ_POSTERIOR_MIN,
              thresholds: Optional[ZQualityThresholds] = None,
              params: Optional[ViabilityParams] = None) -> ZQuality:
    """
    How much axial information separates every consecutive ring pair of
    a ``build_rings`` result, and the first filter's verdict of each pair.

    Reads only what ``build_rings`` returns: the axial mixture
    (``res.ms.z_result``), each ring's component, slab and valley depth,
    its localizations, clusters and link events. Port of the research
    prototype ``zq_core.z_quality`` + ``verdict`` (H5-E, same numbers);
    the cut of a pair is the midpoint of the lower ring's top and the
    upper ring's bottom. Deterministic: nothing is drawn.

    Parameters
    ----------
    res : a ``RingsResult`` (duck-typed; nothing is imported from
        mps_columns).
    p_ref_nm : the reference period the merge rules compare with; None ->
        ``params.p_ref_nm`` (170 nm, Gazal et al. 2026, by default); the
        source is recorded.
    min_samples : DBSCAN min_samples of the copy count; None ->
        ``res.params.min_samples``.
    posterior_min : a localization with a two-component posterior below
        this counts as ambiguous.
    thresholds : the verdict thresholds; None -> ``params.thresholds()``.
    params : the viability parameters (``ViabilityParams``; None ->
        defaults): the copy tolerance's delta and tau (None -> the column
        test's tau_0, ``column_test_tau_nm``), the keep floor.

    Returns
    -------
    ZQuality
        Rings bottom to top, one ``BoundaryQuality`` per consecutive pair
        and the axon verdict (the best pair's; ``AXON_FEWER_THAN_TWO_RINGS``
        with fewer than two rings).
    """
    vp = params if params is not None else ViabilityParams()
    if p_ref_nm is None:
        p_ref, p_src = float(vp.p_ref_nm), str(vp.p_ref_source)
    else:
        p_ref, p_src = float(p_ref_nm), "argument"
    ms_min = int(res.params.min_samples) if min_samples is None else int(min_samples)
    t = thresholds if thresholds is not None else vp.thresholds()
    tau = float(vp.tau_nm) if vp.tau_nm is not None else column_test_tau_nm()
    warnings: List[str] = []
    rings = sorted(res.rings, key=lambda r: int(r.index))
    zr = res.ms.z_result
    mu_all = np.asarray(zr.means_nm, dtype=np.float64).ravel()
    sd_all = np.asarray(zr.sigmas_nm, dtype=np.float64).ravel()
    w_all = np.asarray(zr.weights, dtype=np.float64).ravel()
    zp = np.asarray(res.z_p, dtype=np.float64)
    ev = None if res.event_id is None else np.asarray(res.event_id, dtype=np.int64)

    ring_q: List[RingQuality] = []
    for r in rings:
        sig = float(r.sigma_z_nm)
        lpz = float(r.lpz_median_nm)
        flags: Tuple[str, ...] = ()
        if sig > t.merged_sigma_over_p * p_ref:
            flags = (f"sigma_z {sig:.0f} > {t.merged_sigma_over_p:g} P_ref",)
        ring_q.append(RingQuality(
            index=int(r.index), centre_z_nm=float(r.centre_z_nm), sigma_z_nm=sig, lpz_median_nm=lpz,
            sigma_struct_nm=(structural_width_nm(sig, lpz) if math.isfinite(lpz) else float("nan")),
            n_locs=int(np.asarray(r.loc_index).size), n_events=(None if r.n_events is None else int(r.n_events)),
            k_clusters=len(r.clusters), sigma_over_p=sig / p_ref, flags=flags))
    flags_of = {q.index: q.flags for q in ring_q}

    bounds: List[BoundaryQuality] = []
    for ra, rb in zip(rings[:-1], rings[1:]):
        ia, ib = int(ra.index), int(rb.index)
        ca, cb = int(ra.component_index), int(rb.component_index)
        cut = 0.5 * (float(ra.z_hi_nm) + float(rb.z_lo_nm))
        k_a, k_b = len(ra.clusters), len(rb.clusters)
        if not (0 <= ca < mu_all.size and 0 <= cb < mu_all.size):
            warnings.append(f"rings {ia}-{ib}: no axial mixture component, pair not viable")
            nan = float("nan")
            bounds.append(BoundaryQuality(ia, ib, nan, nan, nan, nan, cut, nan, nan, nan, nan, k_a, k_b, nan, nan,
                                          math.inf, None, nan, VERDICT_NOT_VIABLE, ("no axial mixture component",)))
            continue
        mu = np.array([mu_all[ca], mu_all[cb]])
        sd = np.maximum(np.array([sd_all[ca], sd_all[cb]]), _MIN_SIGMA_NM)
        w = np.array([w_all[ca], w_all[cb]])
        w = w / w.sum()
        d_sep = float(abs(mu[1] - mu[0]) / math.sqrt(float(np.sum(sd ** 2))))
        f_lo = float(norm.sf((cut - mu[0]) / sd[0]))
        f_hi = float(norm.sf((mu[1] - cut) / sd[1]))
        depth = ra.relative_depth_hi
        valley = float(depth) if depth is not None else float("nan")
        li = np.concatenate([np.asarray(ra.loc_index, np.intp), np.asarray(rb.loc_index, np.intp)])
        z = zp[li]
        if z.size:
            lp = np.log(w)[None, :] - np.log(sd)[None, :] - 0.5 * ((z[:, None] - mu[None, :]) / sd[None, :]) ** 2
            lp -= lp.max(axis=1, keepdims=True)
            post = np.exp(lp)
            post /= post.sum(axis=1, keepdims=True)
            amb = float(np.mean(post.max(axis=1) < posterior_min))
        else:
            amb = float("nan")
        e_ab = _expected_copies(ra, f_lo, mu_all, sd_all, ms_min, vp.keep_floor)
        e_ba = _expected_copies(rb, f_hi, mu_all, sd_all, ms_min, vp.keep_floor)
        spur = float((e_ab + e_ba) / (k_a + k_b)) if (k_a + k_b) else float("nan")
        kl = kl_mixture_vs_gauss(mu, sd, w)
        n_req = n_required(kl)
        n_ev: Optional[int] = None
        if ev is not None:
            e = ev[li]
            n_ev = int(np.unique(e[e >= 0]).size) + int(np.count_nonzero(e < 0))
        info = float(n_ev / n_req) if (n_ev and math.isfinite(n_req)) else float("nan")
        gap = float(mu[1] - mu[0])
        tol = copy_tolerance(k_a, k_b, float(getattr(ra, "length_nm", float("nan"))),
                             float(getattr(rb, "length_nm", float("nan"))), tau,
                             delta_viable_sd=vp.delta_viable_sd, delta_marginal_sd=vp.delta_marginal_sd)
        sv, sm = ((tol.viable, tol.marginal) if tol.per_pair else (t.spur_viable, t.spur_marginal))
        verdict, reasons = boundary_verdict(d_sep=d_sep, exp_spur_frac=spur, info_ratio=info, valley_depth=valley,
                                            gap_nm=gap, ring_flags=(flags_of[ia], flags_of[ib]), rings=(ia, ib),
                                            p_ref_nm=p_ref, thresholds=t, spur_viable=sv, spur_marginal=sm)
        bounds.append(BoundaryQuality(
            ring_a=ia, ring_b=ib, d_sep=d_sep, ashman_d=d_sep * math.sqrt(2.0), valley_depth=valley, gap_nm=gap,
            cut_nm=cut, f_lo=f_lo, f_hi=f_hi, misassign=float(w[0] * f_lo + w[1] * f_hi), ambiguous_frac=amb,
            k_a=k_a, k_b=k_b, exp_spur_frac=spur, kl_nats=kl, n_req_events=n_req, n_events=n_ev, info_ratio=info,
            verdict=verdict, reasons=reasons, spur_viable=float(sv), spur_marginal=float(sm), tolerance_p0=tol.p0,
            tolerance_sd=tol.sd_star, tolerance_source=tol.source))

    axon = axon_verdict([b.verdict for b in bounds], vp.aggregation)
    return ZQuality(
        rings=tuple(ring_q), boundaries=tuple(bounds), axon_verdict=axon,
        viable_pairs=tuple((b.ring_a, b.ring_b) for b in bounds if b.verdict == VERDICT_VIABLE),
        marginal_pairs=tuple((b.ring_a, b.ring_b) for b in bounds if b.verdict == VERDICT_MARGINAL),
        p_ref_nm=p_ref, p_ref_source=p_src, min_samples=ms_min, thresholds=t, warnings=tuple(warnings),
        params=vp, tau_nm=tau)


def axon_verdict(verdicts: Sequence[str], aggregation: str = "any") -> str:
    """The axon's verdict from its pairs': 'any' (default) is the best pair's (viable when at least one pair is, since
    the column comparison is made pair by pair); 'all' the worst pair's. ``AXON_FEWER_THAN_TWO_RINGS`` without
    pairs."""
    order = {VERDICT_VIABLE: 2, VERDICT_MARGINAL: 1, VERDICT_NOT_VIABLE: 0}
    if not verdicts:
        return AXON_FEWER_THAN_TWO_RINGS
    ranks = [order[v] for v in verdicts]
    best = max(ranks) if aggregation == "any" else min(ranks)
    return {2: VERDICT_VIABLE, 1: VERDICT_MARGINAL, 0: VERDICT_NOT_VIABLE}[best]


def z_selection_key(zq: ZQuality, mode: str) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """(number of rings, selected pairs sorted): the identity a simulated
    replicate must share with the observed axon to be accepted (Q-29 +
    D-39 d). ``mode`` "viable" selects the viable pairs, "viable+marginal"
    both."""
    if mode not in Z_SELECTION_MODES:
        raise ValueError(f"mode must be one of {Z_SELECTION_MODES}, got {mode!r}")
    pairs = zq.viable_pairs if mode == "viable" else zq.viable_pairs + zq.marginal_pairs
    return len(zq.rings), tuple(sorted(pairs))


def calibration_range_flags(label: str, zq: Optional[ZQuality]) -> Tuple[str, ...]:
    """
    Warnings that a result lies outside the calibrated range (Q-28,
    D-38 f). Never blocks.

    An acquisition whose closure defect was not calibrated: the label or
    source path matches ``uncalibrated_label_pattern()`` (``ROI2_PATTERN``,
    the ``MPS_UNCALIBRATED_LABELS`` regular expression or the settings
    file's ``uncalibrated_labels``; none by default). After H5-E S5 fills
    ``CALIBRATED_ENVELOPE``, also a viable or marginal pair with d_sep
    below, or exp_spur_frac above, the range of the accepted S4
    originals.
    """
    out: List[str] = []
    pat = uncalibrated_label_pattern()
    if pat is not None and pat.search(label or ""):
        out.append(ROI2_FLAG)
    env = CALIBRATED_ENVELOPE
    if env is not None and zq is not None:
        sel = set(zq.viable_pairs) | set(zq.marginal_pairs)
        bad = [f"{b.ring_a}-{b.ring_b}" for b in zq.boundaries if (b.ring_a, b.ring_b) in sel
               and (b.d_sep < env["d_sep_min"] or b.exp_spur_frac > env["exp_spur_frac_max"])]
        if bad:
            out.append("pairs " + ", ".join(bad) + ": outside the calibrated range (d_sep / expected copies outside the "
                       "range the synthetic calibration accepted)")
    return tuple(out)


# ===================================================================
#  Ring-pair viability rule v2: SiZer peak + valley, localization
#  minimum vs the reference ring, the pair's leak-copy tolerance
# ===================================================================
# The rule the user adopted on 2026-10-02 (the declared default):
#   (1) every ring of the pair has a significant SiZer PEAK within P/3 of
#       its centre and (2) there is a significant VALLEY between the two
#       ring centres (cluster-robust SiZer, h = 0.10 P, alpha 0.05,
#       simultaneous quantile; groups = 3D DBSCAN clusters), P the axon's
#       own period (the reference period with fewer than two rings);
#   (3) both rings hold f = n_locs(ring) / n_locs(reference ring) >= f_min:
#       the fraction at which a thinned ring keeps the recall tolerance (90 %)
#       of its clusters, calibrated on the axon by Bernoulli thinning
#       (``calibrate_count_floor``) or, until that is run, read from a line
#       fitted to a simulation study, f_min(x) = 1.46 - 0.336 ln x, x =
#       localizations per cluster of the reference ring / min_samples (fitted
#       on x in [5.8, 13.3]; above the range f_min(13.3), below it the line
#       capped at 100 %, both flagged);
#   (4) the expected leak copies <= the pair's viable tolerance -> VIABLE;
#       <= its marginal tolerance with (1)-(3) -> MARGINAL (sensitivity
#       only); anything else NOT VIABLE.
# Every threshold is an editable suggestion (``ViabilityParams``); the
# composition is a conjunction of necessary conditions built from published
# components (SiZer, Chaudhuri & Marron 1999; DBSCAN, Ester et al. 1996;
# Gaussian tails).
VIABILITY_RULE_V2 = ("viability rule v2: cluster-robust SiZer peak + valley (h 0.10 P, alpha 0.05), localizations "
                     ">= f_min of the reference ring, expected leak copies within the pair's tolerance")
V2_PEAK_WINDOW_OVER_P = 1.0 / 3.0
FMIN_INTERCEPT = 1.46
FMIN_SLOPE_PER_LN_X = -0.336
FMIN_X_RANGE = (5.8, 13.3)
FMIN_CAP = 1.0
# The fixed copy tiers, shown as the suggestion when a pair's tolerance cannot be computed (``copy_tolerance``).
V2_SPUR_VIABLE = ZQ_SPUR_VIABLE
V2_SPUR_MARGINAL = ZQ_SPUR_MARGINAL
# Reference marks of the ring width / period summary (descriptive; never a criterion), each with its meaning: at
# sigma_z / P = 0.30 the valley of an equal pair is 50 % deep, at 0.36 25 % deep, and d = 2.2 is 0.32 (41 %).
LF_SIGMA_OVER_P_VALLEY = 0.30
LF_SIGMA_OVER_P_NONE = 0.36
LF_SIGMA_OVER_P_D39 = 0.32
# z-filter truncation flag (descriptive): an edge of the lab z range is
# "cut" when the TRUNC_BIN_NM next to it still hold >= TRUNC_EDGE_FRAC of
# the fullest bin and >= TRUNC_MIN_COUNT localizations (a natural tail
# ends at ~0): a z filter can cut the profile where the density is still a
# sizeable fraction of the maximum.
TRUNC_BIN_NM = 20.0
TRUNC_EDGE_FRAC = 0.05
TRUNC_MIN_COUNT = 10
LF_LIMITING = "limiting"
LF_OK = "ok"
LF_INFO = "info"
LF_WARNING = "warning"
LF_OUTSIDE = "outside calibrated range"
# The count-floor calibration by thinning: the retained fractions and the replicates per fraction.
THINNING_F_GRID: Tuple[float, ...] = tuple(round(1.0 - 0.05 * i, 2) for i in range(15))   # 1.00 ... 0.30
THINNING_REPLICATES = 20
FMIN_SOURCE_FORMULA = "suggestion (simulation): the fitted line f_min(x)"
FMIN_SOURCE_THINNING = "calibrated on this axon by thinning"


@dataclass(frozen=True)
class FMin:
    """f_min of criterion 3: the minimum fraction of the reference ring's
    localizations, the x it came from, whether x is inside the fitted
    range and, when not, what was done ("" inside)."""

    f_min: float
    x: float
    in_range: bool
    note: str


@dataclass(frozen=True)
class FMinCalibration:
    """The count floor calibrated on one axon by Bernoulli thinning of its reference ring's detected clusters: the
    retained fractions, the mean (and sd) cluster recall at each over the replicates, f_min (the fraction at which
    the mean recall falls to the tolerance, linearly interpolated; 1.0 when even the full ring is below it; NaN
    without clusters), the tolerance, the replicates, the clusters thinned and the DBSCAN parameters."""

    f_grid: Tuple[float, ...]
    recall_mean: Tuple[float, ...]
    recall_sd: Tuple[float, ...]
    f_min: float
    tolerance: float
    replicates: int
    k_clusters: int
    ring_index: int
    eps_nm: float
    min_samples: int
    seed: int
    note: str


@dataclass(frozen=True)
class EdgeTruncation:
    """One edge of the z range: where it is, how many localizations the
    TRUNC_BIN_NM next to it hold, that count over the fullest bin's, and
    whether that looks like a filter cut rather than a tail."""

    side: str
    z_nm: float
    edge_count: int
    edge_frac: float
    truncated: bool


@dataclass(frozen=True)
class RingViabilityV2:
    """One ring under rule v2 (bottom to top): size, fraction of the
    reference ring, clusters, localizations per cluster, axial width and
    precision, the SiZer peak (and where), and its roles (``is_central``:
    the reference ring of the count criterion; ``is_in_focus``: the ring
    with the best axial precision)."""

    index: int
    centre_z_nm: float
    n_locs: int
    k_clusters: int
    f_of_central: float
    locs_per_cluster: float
    sigma_z_nm: float
    sigma_over_p: float
    lpz_median_nm: float
    peak: bool
    peak_z_nm: Optional[float]
    is_central: bool
    is_in_focus: bool
    truncated_edge: str


@dataclass(frozen=True)
class PairViabilityV2:
    """One consecutive ring pair under rule v2: the four criteria with
    their numbers, the first filter's verdict next to it, the verdict and
    the plain reasons it is not better; the pair's copy tolerance and, as
    reference readings, Silverman's critical bandwidth and its bootstrap p
    and the valley at every bandwidth of the SiZer map."""

    ring_a: int
    ring_b: int
    peak_a: bool
    peak_b: bool
    valley: bool
    f_a: float
    f_b: float
    f_min: float
    x: float
    x_in_range: bool
    count_ok: bool
    exp_spur_frac: float
    d_sep: float
    d39_verdict: str
    verdict: str
    reasons: Tuple[str, ...]
    spur_viable: float = V2_SPUR_VIABLE
    spur_marginal: float = V2_SPUR_MARGINAL
    tolerance_source: str = ""
    silverman_h_crit_nm: float = float("nan")
    silverman_p: float = float("nan")
    valley_by_h: Tuple[bool, ...] = ()


@dataclass(frozen=True)
class SizerMapRow:
    """The axial-profile SiZer at one bandwidth of the map: h as a fraction of P and in nm, the significant peaks and
    valleys, which ring has a peak within the window and which consecutive pair a valley."""

    h_over_p: float
    h_nm: float
    peaks: Tuple[float, ...]
    valleys: Tuple[float, ...]
    ring_peak: Tuple[bool, ...]
    pair_valley: Tuple[bool, ...]


@dataclass(frozen=True)
class LimitingFactor:
    """One line of the axon's limiting-factor summary (for the microscopy
    team): what was measured, what the rule needs, and whether it limits."""

    key: str
    label: str
    value: str
    needed: str
    status: str
    source: str


@dataclass(frozen=True)
class AxonViabilityV2:
    """``viability_v2``'s answer for one axon: the rule, the reference period (``p_ref_nm``, with its source) and the
    period that scales h and the peak window (``p_nm``: the axon's own, or the reference as a fallback), the rings
    and pairs, the reference (``central_index``) and best-precision (``in_focus_index``) rings, criterion 3's x and
    f_min (with its source), the SiZer profile and map, the variance groups (with their z scale), the z-edge flags,
    the limiting factors and warnings."""

    rule: str
    p_ref_nm: float
    h_nm: float
    alpha: float
    min_samples: int
    rings: Tuple[RingViabilityV2, ...]
    pairs: Tuple[PairViabilityV2, ...]
    central_index: Optional[int]
    in_focus_index: Optional[int]
    x: float
    f_min: float
    x_in_range: bool
    x_note: str
    groups_source: str
    n_groups: int
    profile: Optional[Any]
    z_edges: Tuple[EdgeTruncation, ...]
    z_edges_source: str
    limiting_factors: Tuple[LimitingFactor, ...]
    viable_pairs: Tuple[Tuple[int, int], ...]
    marginal_pairs: Tuple[Tuple[int, int], ...]
    warnings: Tuple[str, ...]
    p_nm: float = float("nan")
    p_source: str = ""
    p_ref_source: str = P_REF_SOURCE
    params: Optional[ViabilityParams] = None
    sizer_map: Tuple[SizerMapRow, ...] = ()
    group_z_scale: float = float("nan")
    group_z_scale_source: str = ""
    peak_sd_nm: float = float("nan")
    fmin_source: str = FMIN_SOURCE_FORMULA
    fmin_calibration: Optional[FMinCalibration] = None
    reference_ring: str = "central"

    @property
    def p_scale_nm(self) -> float:
        """The period of the scale-type quantities (the reference period for a result without ``p_nm``)."""
        return float(self.p_nm) if math.isfinite(float(self.p_nm)) else float(self.p_ref_nm)


def f_min_of_x(x: float) -> FMin:
    """
    Criterion 3's suggested threshold from a simulation study (shown until
    the per-axon calibration is run), f_min(x) = 1.46 - 0.336 ln x, the
    fraction that kept >= 90 % of the central ring's cluster recall on the
    simulated design points x in [5.8, 13.3].

    Outside the fitted range it stays conservative and says so: x above
    13.3 takes f_min(13.3) (the formula keeps falling there), x below 5.8
    takes the extrapolated formula capped at 100 %. A non-finite or
    non-positive x gives NaN (the criterion then fails).
    """
    lo, hi = FMIN_X_RANGE
    if not (isinstance(x, (int, float, np.floating)) and math.isfinite(float(x)) and float(x) > 0):
        return FMin(float("nan"), float("nan"), False, "x not computable (reference ring without clusters)")
    x = float(x)
    if x > hi:
        f = FMIN_INTERCEPT + FMIN_SLOPE_PER_LN_X * math.log(hi)
        return FMin(f, x, False, f"x {x:.1f} outside calibrated range ({lo:g}-{hi:g}): f_min({hi:g}) used (conservative)")
    f = FMIN_INTERCEPT + FMIN_SLOPE_PER_LN_X * math.log(x)
    if x < lo:
        capped = f >= FMIN_CAP
        f = min(f, FMIN_CAP)
        return FMin(f, x, False, f"x {x:.1f} outside calibrated range ({lo:g}-{hi:g}): formula extrapolated"
                    + (", capped at 100 %" if capped else ""))
    return FMin(f, x, True, "")


def central_ring_position(n_locs: Sequence[int]) -> Optional[int]:
    """
    The central ring of a stack (rings bottom to top, given by
    their localization counts): the middle ring of an odd stack; with 2
    rings the more populated; with an even count >= 4 the more populated
    of the two middle rings (ties: the lower). One ring is its own centre
    (it has no pairs); no ring -> None.
    """
    n = len(n_locs)
    if n == 0:
        return None
    if n == 1:
        return 0
    if n % 2 == 1:
        return n // 2
    a, b = n // 2 - 1, n // 2
    if n == 2:
        a, b = 0, 1
    return a if int(n_locs[a]) >= int(n_locs[b]) else b


def reference_ring_position(n_locs: Sequence[int], ring_indices: Sequence[int], mode: str = "central",
                            pick: Optional[int] = None) -> Optional[int]:
    """The position (bottom to top) of the count criterion's reference ring: 'central' (``central_ring_position``,
    the ring expected nearest focus), 'most populated' (ties: the lower) or 'pick' (the ring whose index is ``pick``;
    the central ring when that ring does not exist). None without rings."""
    if not n_locs:
        return None
    if mode == "most populated":
        best = max(int(v) for v in n_locs)
        return next(i for i, v in enumerate(n_locs) if int(v) == best)
    if mode == "pick" and pick is not None:
        for i, k in enumerate(ring_indices):
            if int(k) == int(pick):
                return i
    return central_ring_position(n_locs)


def _pct(f: float) -> str:
    return f"{100.0 * f:.0f} %"


def verdict_v2(*, rings: Tuple[int, int], peak_a: bool, peak_b: bool, valley: bool, f_a: float, f_b: float,
               f_min: float, exp_spur_frac: float, x: float = float("nan"), x_note: str = "",
               p_ref_nm: float = P_REF_DEFAULT_NM, spur_viable: Optional[float] = None,
               spur_marginal: Optional[float] = None, window_over_p: float = V2_PEAK_WINDOW_OVER_P,
               reference_word: str = "central") -> Tuple[str, Tuple[str, ...]]:
    """
    The rule-v2 verdict of one pair with its plain-language reasons.

    Criteria (1)-(3) must all hold; then the leak decides: expected copies
    <= the viable tolerance VIABLE, <= the marginal tolerance MARGINAL
    (sensitivity only), else (or not finite) NOT VIABLE. ``p_ref_nm`` is
    the period of the peak window (the axon's own when known);
    ``spur_viable`` / ``spur_marginal`` the pair's tolerance (None: the
    fixed 2 % / 5 % suggestion).
    """
    a, b = rings
    sv = float(V2_SPUR_VIABLE if spur_viable is None else spur_viable)
    sm = float(V2_SPUR_MARGINAL if spur_marginal is None else spur_marginal)
    reasons: List[str] = []
    win = float(window_over_p) * p_ref_nm
    for k, pk in ((a, peak_a), (b, peak_b)):
        if not pk:
            reasons.append(f"ring {k}: no significant peak within {win:.0f} nm of its centre (SiZer)")
    if not valley:
        reasons.append(f"no significant valley between rings {a} and {b} (the profile between them is flat or one-sided)")
    count_ok = math.isfinite(f_min) and all(math.isfinite(f) and f >= f_min for f in (f_a, f_b))
    if not math.isfinite(f_min):
        reasons.append(f"localization minimum not computable ({reference_word} ring without clusters)")
    else:
        for k, f in ((a, f_a), (b, f_b)):
            if not (math.isfinite(f) and f >= f_min):
                reasons.append(f"ring {k} has {_pct(f) if math.isfinite(f) else 'an unknown fraction'} of the "
                               f"{reference_word} ring's localizations; the minimum is {_pct(f_min)}"
                               + (f" (x = {x:.1f})" if math.isfinite(x) else "") + (f"; {x_note}" if x_note else ""))
    crit = peak_a and peak_b and valley and count_ok
    spur = float(exp_spur_frac)
    if not math.isfinite(spur):
        reasons.append("expected leak copies unknown (no axial mixture component)")
    elif spur > sm:
        reasons.append(f"expected leak copies {100 * spur:.1f} % > {_tol_text(sm)}")
    elif spur > sv:
        # "marginal" only when it is the verdict: a pair failing (1)-(3) is NOT VIABLE whatever its leak
        reasons.append(f"expected leak copies {100 * spur:.1f} % > {_tol_text(sv)} "
                       + ("(marginal: sensitivity only)" if crit else f"(VIABLE needs <= {_tol_text(sv)})"))
    if crit and math.isfinite(spur) and spur <= sv:
        return VERDICT_VIABLE, tuple(reasons)
    if crit and math.isfinite(spur) and spur <= sm:
        return VERDICT_MARGINAL, tuple(reasons)
    return VERDICT_NOT_VIABLE, tuple(reasons)


def z_edge_truncation(z: NDArray[np.float64], *, bin_nm: float = TRUNC_BIN_NM, edge_frac: float = TRUNC_EDGE_FRAC,
                      min_count: int = TRUNC_MIN_COUNT) -> Tuple[EdgeTruncation, EdgeTruncation]:
    """
    (bottom, top): is either end of the z range a hard cut (a z filter)
    rather than a tail? The ``bin_nm`` next to the edge holding >=
    ``edge_frac`` of the fullest ``bin_nm`` bin and >= ``min_count``
    localizations is a cut. Descriptive only.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        nan = float("nan")
        return (EdgeTruncation("bottom", nan, 0, nan, False), EdgeTruncation("top", nan, 0, nan, False))
    lo, hi = float(z.min()), float(z.max())
    n_bins = max(1, int(math.ceil((hi - lo) / bin_nm)))
    counts, _ = np.histogram(z, bins=n_bins, range=(lo, lo + n_bins * bin_nm))
    peak = max(int(counts.max()), 1)
    c_lo = int(np.count_nonzero(z < lo + bin_nm))
    c_hi = int(np.count_nonzero(z > hi - bin_nm))
    out = []
    for side, edge, c in (("bottom", lo, c_lo), ("top", hi, c_hi)):
        frac = c / peak
        out.append(EdgeTruncation(side, edge, c, frac, bool(c >= min_count and frac >= edge_frac)))
    return out[0], out[1]


# ------------------------------------------------------------------- reference multimodality readings
_SILVERMAN_BINS = 1024


def _n_modes(z: NDArray[np.float64], h: float, lo: float, hi: float) -> int:
    """Modes of the Gaussian kernel density of ``z`` at bandwidth ``h`` on a binned grid over [lo, hi] (the counts
    smoothed by a Gaussian filter; a mode is a strict rise followed by a fall, ripples below 1e-9 of the maximum
    ignored)."""
    from scipy.ndimage import gaussian_filter1d
    counts, edges = np.histogram(z, bins=_SILVERMAN_BINS, range=(lo, hi))
    dx = float(edges[1] - edges[0])
    f = gaussian_filter1d(counts.astype(np.float64), sigma=max(h / dx, 1e-6), mode="constant", truncate=5.0)
    tol = 1e-9 * float(f.max()) if f.size else 0.0
    d = np.diff(f)
    s = np.where(d > tol, 1, np.where(d < -tol, -1, 0))
    s = s[s != 0]
    if s.size == 0:
        return 1
    return int(np.count_nonzero((s[:-1] > 0) & (s[1:] < 0))) + int(s[-1] > 0)


def silverman_reading(z: NDArray[np.float64], *, n_boot: int = SILVERMAN_N_BOOT, max_n: int = SILVERMAN_MAX_N,
                      seed: int = 0) -> Tuple[float, float]:
    """
    Silverman's (1981) test of unimodality as a reference reading: the critical bandwidth h_crit (the smallest
    Gaussian bandwidth at which the kernel density of ``z`` has one mode, by bisection) and the smoothed-bootstrap p
    (variance-corrected resampling at h_crit; p = (b + 1) / (B + 1) with b the replicates that show more than one
    mode). The localizations are treated as independent, which they are not (one fluorophore gives many): the p is
    a reading, never a criterion. Deterministic given ``seed``; a sample above ``max_n`` is subsampled. (NaN, NaN)
    with fewer than 10 values or no spread.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < 10 or float(np.ptp(z)) <= 0.0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(int.from_bytes(b"silv", "big"),)))
    if z.size > max_n:
        z = np.sort(rng.choice(z, size=int(max_n), replace=False))
    sd = float(np.std(z))
    lo, hi = float(z.min()) - 4.0 * sd, float(z.max()) + 4.0 * sd
    a, b = (hi - lo) / _SILVERMAN_BINS, 4.0 * sd
    if _n_modes(z, b, lo, hi) > 1:
        return float("nan"), float("nan")
    if _n_modes(z, a, lo, hi) <= 1:
        h_crit = a
    else:
        for _ in range(40):
            m = math.sqrt(a * b)
            if _n_modes(z, m, lo, hi) > 1:
                a = m
            else:
                b = m
        h_crit = b
    mean, var = float(np.mean(z)), float(np.var(z))
    shrink = 1.0 / math.sqrt(1.0 + h_crit * h_crit / var) if var > 0 else 1.0
    exceed = 0
    for _ in range(int(n_boot)):
        x = rng.choice(z, size=z.size, replace=True)
        y = mean + (x - mean + h_crit * rng.standard_normal(z.size)) * shrink
        if _n_modes(y, h_crit, lo, hi) > 1:
            exceed += 1
    return float(h_crit), float((exceed + 1) / (int(n_boot) + 1))


def peak_location_sd_nm(sigma_nm: float, h_nm: float, n: float) -> float:
    """The sd of the position of a kernel-density mode (Parzen 1962): sqrt((sqrt 2 / 4) (s^2 + h^2)^(5/2) /
    (n h^3)) for a Gaussian ring of width s read at bandwidth h with n independent points."""
    s, h, n_ = float(sigma_nm), float(h_nm), float(n)
    if not (math.isfinite(s) and math.isfinite(h) and h > 0 and math.isfinite(n_) and n_ > 0):
        return float("nan")
    return float(math.sqrt((math.sqrt(2.0) / 4.0) * (s * s + h * h) ** 2.5 / (n_ * h ** 3)))


# ------------------------------------------------------------------- the count floor by thinning
def calibrate_count_floor(res: Any, *, ring_index: Optional[int] = None, tolerance: float = 0.90,
                          f_grid: Sequence[float] = THINNING_F_GRID, replicates: int = THINNING_REPLICATES,
                          seed: int = 0, eps_nm: Optional[float] = None,
                          min_samples: Optional[int] = None) -> FMinCalibration:
    """
    The count floor of criterion 3 calibrated on one axon: every localization of the reference ring's detected
    clusters is kept with probability f (Bernoulli thinning), and a cluster survives while some kept localization
    still has >= min_samples kept neighbours within eps in the axon-frame plane (the DBSCAN core condition, Ester et
    al. 1996; the run's eps and min_samples). The recall at f is the share of clusters that survive, averaged over
    ``replicates``; f_min is the fraction at which the mean recall falls to ``tolerance`` (linear interpolation
    between grid points). ``ring_index`` None -> the central ring. Deterministic given ``seed``.
    """
    from scipy.spatial import cKDTree

    rings = sorted(res.rings, key=lambda r: int(r.index))
    eps = float(getattr(res.params, "eps_nm", 25.0) if eps_nm is None else eps_nm)
    ms = int(res.params.min_samples if min_samples is None else min_samples)
    grid = tuple(sorted({float(f) for f in f_grid}, reverse=True))
    if not rings:
        return FMinCalibration(grid, (), (), float("nan"), float(tolerance), int(replicates), 0, -1, eps, ms,
                               int(seed), "no ring")
    if ring_index is None:
        pos = central_ring_position([int(np.asarray(r.loc_index).size) for r in rings])
        ring = rings[int(pos or 0)]
    else:
        found = [r for r in rings if int(r.index) == int(ring_index)]
        if not found:
            raise ValueError(f"calibrate_count_floor: no ring {ring_index}")
        ring = found[0]
    x = np.asarray(res.x_p, dtype=np.float64)
    y = np.asarray(res.y_p, dtype=np.float64)
    neighbours: List[List[NDArray[np.intp]]] = []
    for cl in ring.clusters:
        li = np.asarray(cl.loc_index, dtype=np.intp)
        pts = np.column_stack([x[li], y[li]])
        tree = cKDTree(pts)
        neighbours.append([np.asarray(v, dtype=np.intp) for v in tree.query_ball_point(pts, r=eps)])
    k = len(neighbours)
    if k == 0:
        return FMinCalibration(grid, (), (), float("nan"), float(tolerance), int(replicates), 0, int(ring.index), eps,
                               ms, int(seed), "the reference ring has no cluster")
    rng = np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(int.from_bytes(b"thin", "big"),)))
    means: List[float] = []
    sds: List[float] = []
    for f in grid:
        rec = np.empty(int(replicates))
        for r in range(int(replicates)):
            alive = 0
            for nb in neighbours:
                keep = rng.random(len(nb)) < f
                if not keep.any():
                    continue
                counts = np.fromiter((int(np.count_nonzero(keep[v])) for v in nb), dtype=np.int64, count=len(nb))
                if bool(np.any(keep & (counts >= ms))):
                    alive += 1
            rec[r] = alive / k
        means.append(float(rec.mean()))
        sds.append(float(rec.std(ddof=1)) if rec.size > 1 else 0.0)
    tol = float(tolerance)
    f_min = float("nan")
    note = ""
    if means[0] < tol:
        f_min, note = 1.0, "even the full ring keeps fewer clusters than the tolerance: no localization can be lost"
    else:
        f_min = float(grid[-1])
        note = f"the recall stays above the tolerance down to f = {grid[-1]:g}"
        for i in range(1, len(grid)):
            if means[i] < tol:
                f0, f1, m0, m1 = grid[i - 1], grid[i], means[i - 1], means[i]
                f_min = f1 + (tol - m1) * (f0 - f1) / (m0 - m1) if m0 != m1 else f0
                note = ""
                break
    return FMinCalibration(grid, tuple(means), tuple(sds), float(f_min), tol, int(replicates), k, int(ring.index),
                           eps, ms, int(seed), note)


# ------------------------------------------------------------------- the rule
def group_z_scale_of(params: ViabilityParams, precision_nm: Optional[Tuple[float, float]]) -> Tuple[float, str]:
    """The axial scale of the 3D variance groups and its source: the typed value, else sigma_xy / sigma_z of the
    file (median lateral over median axial precision), else 0.5 (the suggestion: astigmatic axial precision is 2-3x
    the lateral, Huang et al. 2008)."""
    from tools.mps_sizer import GROUP_Z_SCALE
    if params.group_z_scale is not None:
        return float(params.group_z_scale), "user"
    if precision_nm is not None:
        lp, lpz = (float(v) for v in precision_nm)
        if math.isfinite(lp) and math.isfinite(lpz) and lp > 0 and lpz > 0:
            return lp / lpz, f"from the file: lateral {lp:.0f} nm / axial {lpz:.0f} nm"
    return float(GROUP_Z_SCALE), "suggestion 0.5 (no precision in the file)"


def viability_v2(res: Any, zq: Optional[ZQuality] = None, *, xyz_lab_nm: Optional[Sequence[NDArray[np.float64]]] = None,
                 groups: Optional[NDArray[np.int64]] = None, p_ref_nm: Optional[float] = None,
                 params: Optional[ViabilityParams] = None, precision_nm: Optional[Tuple[float, float]] = None,
                 fmin_calibration: Optional[FMinCalibration] = None, sizer_map: bool = True,
                 reference_readings: bool = True) -> AxonViabilityV2:
    """
    The ring-pair viability rule v2 of a ``build_rings`` result.

    Parameters
    ----------
    res : a ``RingsResult`` (duck-typed): ``rings`` (index, centre_z_nm,
        loc_index, clusters, sigma_z_nm, lpz_median_nm), ``z_p`` (every
        input localization, input order), ``x_p`` / ``y_p`` and
        ``params.min_samples``.
    zq : its ``z_quality`` (the leak numbers and tolerances of criterion
        4); None -> run with ``params``.
    xyz_lab_nm : the localizations' LAB x, y, z (input order) for the 3D
        DBSCAN groups of the cluster-robust variance. Without it the
        axon-frame coordinates are used and a warning says so.
    groups : the groups themselves (overrides ``xyz_lab_nm``).
    p_ref_nm : the reference period; None -> ``zq.p_ref_nm`` or
        ``params.p_ref_nm``. h and the peak window scale with the axon's
        own period (``scale_period_nm``), which falls back to it.
    params : ``ViabilityParams`` (None: defaults; ``zq.params`` when zq
        carries them).
    precision_nm : (median lateral, median axial) precision of the file,
        for the groups' z scale; None -> the 0.5 suggestion.
    fmin_calibration : a ``calibrate_count_floor`` result for this axon;
        its f_min replaces the fitted line.
    sizer_map / reference_readings : compute the SiZer map over
        ``SIZER_MAP_H_OVER_P`` and Silverman's reading of every pair.

    Returns
    -------
    AxonViabilityV2
        Rings bottom to top, one ``PairViabilityV2`` per consecutive pair
        (none with fewer than two rings), the limiting factors.
    """
    from tools.mps_sizer import GROUP_EPS_NM, GROUP_MIN_SAMPLES, SIZER_ALPHA
    from tools.mps_sizer import groups_3d, sizer_profile, valley_between

    vp = params if params is not None else (getattr(zq, "params", None) or ViabilityParams())
    if p_ref_nm is None:
        p_ref_nm = getattr(zq, "p_ref_nm", None) if zq is not None else None
    p_ref = float(p_ref_nm) if p_ref_nm is not None else float(vp.p_ref_nm)
    p_ref_src = (str(getattr(zq, "p_ref_source", vp.p_ref_source)) if zq is not None and p_ref_nm is not None
                 else str(vp.p_ref_source))
    if zq is None:
        zq = z_quality(res, p_ref_nm=p_ref, params=vp)
    warnings: List[str] = []
    rings = sorted(res.rings, key=lambda r: int(r.index))
    p_scale, p_scale_src = scale_period_nm(rings, p_ref)
    ms = int(res.params.min_samples)
    n_locs = [int(np.asarray(r.loc_index).size) for r in rings]
    k_cl = [len(r.clusters) for r in rings]
    pos_c = reference_ring_position(n_locs, [int(r.index) for r in rings], vp.reference_ring, vp.reference_ring_index)
    ref_word = "central" if vp.reference_ring == "central" else "reference"
    zp = np.asarray(res.z_p, dtype=np.float64)

    # ---- groups of the cluster-robust variance
    zscale, zscale_src = group_z_scale_of(vp, precision_nm)
    if groups is not None:
        g = np.asarray(groups, dtype=np.int64)
        g_src = "given"
    elif xyz_lab_nm is not None:
        x, y, z = (np.asarray(v, dtype=np.float64) for v in xyz_lab_nm)
        g = groups_3d(x, y, z, GROUP_EPS_NM, GROUP_MIN_SAMPLES, zscale)
        g_src = "lab coordinates (3D DBSCAN)"
    else:
        g = groups_3d(np.asarray(res.x_p, dtype=np.float64), np.asarray(res.y_p, dtype=np.float64), zp,
                      GROUP_EPS_NM, GROUP_MIN_SAMPLES, zscale)
        g_src = "axon-frame coordinates (3D DBSCAN; lab coordinates not given)"
        warnings.append("SiZer groups from the axon-frame coordinates: the rule is defined on the lab coordinates")
    if g.size != zp.size:
        raise ValueError(f"viability_v2: {g.size} group labels for {zp.size} localizations")
    n_groups = int(np.unique(g).size) if g.size else 0

    # ---- SiZer on the whole z' profile, at the chosen bandwidth and over the map
    profile = None
    ok_profile = zp.size >= 2 and np.ptp(zp) > 0
    if ok_profile:
        profile = sizer_profile(zp, g, p_ref_nm=p_scale, c=float(vp.h_over_p), alpha=SIZER_ALPHA, robust=True,
                                small_sample=bool(vp.small_sample_correction))
    else:
        warnings.append("fewer than two distinct z values: no SiZer profile")
    peaks = list(profile.peaks) if profile is not None else []
    win = float(vp.peak_window_over_p) * p_scale

    # ---- reference ring, x and f_min
    if pos_c is not None and k_cl[pos_c] > 0 and ms > 0:
        if vp.x_statistic == "median":
            x_c = float(np.median([int(cl.n_locs) for cl in rings[pos_c].clusters])) / ms
        else:
            x_c = n_locs[pos_c] / k_cl[pos_c] / ms
    else:
        x_c = float("nan")
    fm = f_min_of_x(x_c)
    fmin_src = FMIN_SOURCE_FORMULA
    if fmin_calibration is not None and math.isfinite(fmin_calibration.f_min):
        fm = FMin(float(fmin_calibration.f_min), fm.x, True,
                  f"calibrated on ring {fmin_calibration.ring_index} by thinning (recall "
                  f"{fmin_calibration.tolerance:.2f}, {fmin_calibration.replicates} replicates)")
        fmin_src = FMIN_SOURCE_THINNING
    elif abs(float(vp.recall_tolerance) - 0.90) > 1e-9 and math.isfinite(fm.f_min):
        warnings.append(f"the fitted count-floor line belongs to a recall of 0.90, not {vp.recall_tolerance:.2f}: "
                        "calibrate the count floor on this axon")
    n_central = n_locs[pos_c] if pos_c is not None else 0

    # ---- best-precision ring and z edges
    lpz = [float(r.lpz_median_nm) for r in rings]
    fin = [i for i, v in enumerate(lpz) if math.isfinite(v)]
    pos_f = min(fin, key=lambda i: lpz[i]) if fin else None
    z_edge_src = "lab z" if xyz_lab_nm is not None else "axon-frame z'"
    z_for_edges = np.asarray(xyz_lab_nm[2], dtype=np.float64) if xyz_lab_nm is not None else zp
    edges = z_edge_truncation(z_for_edges)

    ring_v: List[RingViabilityV2] = []
    for i, r in enumerate(rings):
        cen = float(r.centre_z_nm)
        near = [p for p in peaks if abs(p - cen) <= win]
        trunc = ""
        if i == 0 and edges[0].truncated:
            trunc = "bottom"
        if i == len(rings) - 1 and edges[1].truncated:
            trunc = "top" if not trunc else "bottom+top"
        sig = float(r.sigma_z_nm)
        ring_v.append(RingViabilityV2(
            index=int(r.index), centre_z_nm=cen, n_locs=n_locs[i], k_clusters=k_cl[i],
            f_of_central=(n_locs[i] / n_central if n_central > 0 else float("nan")),
            locs_per_cluster=(n_locs[i] / k_cl[i] if k_cl[i] else float("nan")), sigma_z_nm=sig,
            sigma_over_p=sig / p_scale, lpz_median_nm=lpz[i], peak=bool(near),
            peak_z_nm=(min(near, key=lambda p: abs(p - cen)) if near else None), is_central=(i == pos_c),
            is_in_focus=(i == pos_f), truncated_edge=trunc))

    # ---- the SiZer map
    map_rows: List[SizerMapRow] = []
    if sizer_map and ok_profile:
        for c in SIZER_MAP_H_OVER_P:
            prof_c = (profile if (profile is not None and abs(c - float(vp.h_over_p)) < 1e-12) else
                      sizer_profile(zp, g, p_ref_nm=p_scale, c=c, alpha=SIZER_ALPHA, robust=True,
                                    small_sample=bool(vp.small_sample_correction)))
            rp = tuple(any(abs(p - rv.centre_z_nm) <= win for p in prof_c.peaks) for rv in ring_v)
            pv = tuple(valley_between(prof_c.grid, prof_c.sign, *sorted((ring_v[i].centre_z_nm,
                                                                         ring_v[i + 1].centre_z_nm)))
                       for i in range(len(ring_v) - 1))
            map_rows.append(SizerMapRow(float(c), float(c) * p_scale, tuple(prof_c.peaks), tuple(prof_c.valleys),
                                        rp, tuple(bool(v) for v in pv)))

    # ---- pairs
    by_pair = {(int(b.ring_a), int(b.ring_b)): b for b in getattr(zq, "boundaries", ())}
    pair_v: List[PairViabilityV2] = []
    for i in range(len(rings) - 1):
        ra, rb = ring_v[i], ring_v[i + 1]
        lo_c, hi_c = sorted((ra.centre_z_nm, rb.centre_z_nm))
        valley = bool(profile is not None and valley_between(profile.grid, profile.sign, lo_c, hi_c))
        b = by_pair.get((ra.index, rb.index))
        if b is None:
            warnings.append(f"rings {ra.index}-{rb.index}: no z_quality boundary (leak unknown)")
        spur = float(b.exp_spur_frac) if b is not None else float("nan")
        sv = float(getattr(b, "spur_viable", V2_SPUR_VIABLE)) if b is not None else V2_SPUR_VIABLE
        sm = float(getattr(b, "spur_marginal", V2_SPUR_MARGINAL)) if b is not None else V2_SPUR_MARGINAL
        verdict, reasons = verdict_v2(rings=(ra.index, rb.index), peak_a=ra.peak, peak_b=rb.peak, valley=valley,
                                      f_a=ra.f_of_central, f_b=rb.f_of_central, f_min=fm.f_min, exp_spur_frac=spur,
                                      x=fm.x, x_note=fm.note, p_ref_nm=p_scale, spur_viable=sv, spur_marginal=sm,
                                      window_over_p=float(vp.peak_window_over_p), reference_word=ref_word)
        count_ok = math.isfinite(fm.f_min) and all(math.isfinite(f) and f >= fm.f_min
                                                    for f in (ra.f_of_central, rb.f_of_central))
        h_crit = p_sil = float("nan")
        if reference_readings:
            li = np.concatenate([np.asarray(rings[i].loc_index, np.intp), np.asarray(rings[i + 1].loc_index, np.intp)])
            h_crit, p_sil = silverman_reading(zp[li], seed=int(getattr(res.params, "random_seed", 0)))
        pair_v.append(PairViabilityV2(
            ring_a=ra.index, ring_b=rb.index, peak_a=ra.peak, peak_b=rb.peak, valley=valley, f_a=ra.f_of_central,
            f_b=rb.f_of_central, f_min=fm.f_min, x=fm.x, x_in_range=fm.in_range, count_ok=count_ok,
            exp_spur_frac=spur, d_sep=(float(b.d_sep) if b is not None else float("nan")),
            d39_verdict=(str(b.verdict) if b is not None else ""), verdict=verdict, reasons=reasons,
            spur_viable=sv, spur_marginal=sm, tolerance_source=(str(getattr(b, "tolerance_source", "")) if b else ""),
            silverman_h_crit_nm=h_crit, silverman_p=p_sil,
            valley_by_h=tuple(row.pair_valley[i] for row in map_rows)))

    # the sd of a ring peak's position at the verdict's bandwidth, for the reference ring (independent points:
    # a lower bound with clustered localizations)
    peak_sd = float("nan")
    if pos_c is not None and profile is not None:
        peak_sd = peak_location_sd_nm(ring_v[pos_c].sigma_z_nm, profile.h_nm, max(k_cl[pos_c], 1))
    factors = _limiting_factors(ring_v, pair_v, edges, z_edge_src, fm, n_central,
                                (k_cl[pos_c] if pos_c is not None else 0), ms, p_scale, p_scale_src, fmin_src,
                                ref_word)
    return AxonViabilityV2(
        rule=VIABILITY_RULE_V2, p_ref_nm=p_ref, h_nm=(profile.h_nm if profile is not None else vp.h_over_p * p_scale),
        alpha=(profile.alpha if profile is not None else 0.05), min_samples=ms, rings=tuple(ring_v),
        pairs=tuple(pair_v), central_index=(ring_v[pos_c].index if pos_c is not None else None),
        in_focus_index=(ring_v[pos_f].index if pos_f is not None else None), x=fm.x, f_min=fm.f_min,
        x_in_range=fm.in_range, x_note=fm.note, groups_source=g_src, n_groups=n_groups, profile=profile,
        z_edges=edges, z_edges_source=z_edge_src, limiting_factors=factors,
        viable_pairs=tuple((p.ring_a, p.ring_b) for p in pair_v if p.verdict == VERDICT_VIABLE),
        marginal_pairs=tuple((p.ring_a, p.ring_b) for p in pair_v if p.verdict == VERDICT_MARGINAL),
        warnings=tuple(warnings), p_nm=p_scale, p_source=p_scale_src, p_ref_source=p_ref_src, params=vp,
        sizer_map=tuple(map_rows), group_z_scale=zscale, group_z_scale_source=zscale_src, peak_sd_nm=peak_sd,
        fmin_source=fmin_src, fmin_calibration=fmin_calibration, reference_ring=vp.reference_ring)


def _limiting_factors(rings: Sequence[RingViabilityV2], pairs: Sequence[PairViabilityV2],
                      edges: Tuple[EdgeTruncation, EdgeTruncation], z_src: str, fm: FMin, n_central: int,
                      k_central: int, ms: int, p_ref: float, p_src: str = "", fmin_src: str = FMIN_SOURCE_FORMULA,
                      ref_word: str = "central") -> Tuple[LimitingFactor, ...]:
    """The axon-level summary for the microscopy team: what limits the
    ring comparison, with the numbers the rule and the reports use."""
    out: List[LimitingFactor] = []
    # rings
    n = len(rings)
    out.append(LimitingFactor(
        "rings", "Rings detected", f"{n} ring(s), {max(n - 1, 0)} consecutive pair(s)",
        "at least 2 rings (one pair) to compare clusters between rings",
        LF_LIMITING if n < 2 else LF_OK, "rule v2"))
    # ring width / period: the RMS width of the best pair's two rings over P (= 1 / (sqrt 2 d) at a gap of P)
    p_words = f"P = {p_ref:.1f} nm, {p_src}" if p_src else f"P = {p_ref:.1f} nm"
    cut_in_best: List[int] = []
    if pairs:
        by_idx = {r.index: r for r in rings}
        rms = [math.sqrt(0.5 * (by_idx[p.ring_a].sigma_z_nm ** 2 + by_idx[p.ring_b].sigma_z_nm ** 2)) / p_ref
               for p in pairs]
        best = min(rms)
        bp = pairs[rms.index(best)] if best in rms else pairs[0]
        where = f"best pair {bp.ring_a}-{bp.ring_b} (RMS width of its two rings / P; {p_words})"
        # a ring whose data the z filter cuts is fitted narrower than it is: its pair's width is not evidence
        cut_in_best = [k for k in (bp.ring_a, bp.ring_b) if by_idx[k].truncated_edge]
        if cut_in_best:
            where += (f"; ring {', '.join(str(k) for k in cut_in_best)} sits at a cut z edge, so its width is "
                      f"underestimated (the other pairs: up to {max(rms):.3f})")
    elif rings:
        best = float(np.median([r.sigma_z_nm for r in rings])) / p_ref
        where = f"ring width / P ({p_words})"
    else:
        best = float("nan")
        where = "no ring"
    # judged on the value shown (three decimals): a reference mark, not a hard threshold
    shown = round(best, 3) if math.isfinite(best) else best
    status = LF_LIMITING if (math.isfinite(shown) and shown > LF_SIGMA_OVER_P_VALLEY) else (
        (LF_WARNING if cut_in_best else LF_OK) if math.isfinite(shown) else LF_INFO)
    d50 = 100 * valley_depth_equal(LF_SIGMA_OVER_P_VALLEY)
    d25 = 100 * valley_depth_equal(LF_SIGMA_OVER_P_NONE)
    d32 = 100 * valley_depth_equal(LF_SIGMA_OVER_P_D39)
    out.append(LimitingFactor(
        "axial_resolution", "Ring width / period (sigma_z / P)",
        f"{shown:.3f} ({where})" if math.isfinite(best) else where,
        f"smaller is better, no hard threshold: at {LF_SIGMA_OVER_P_VALLEY:.2f} the valley between two equal rings "
        f"is {d50:.0f} % deep, at {LF_SIGMA_OVER_P_D39:.2f} {d32:.0f} % (the separation d = 2.2), at "
        f"{LF_SIGMA_OVER_P_NONE:.2f} only {d25:.0f} %, and above 0.5 there is none; the instrument's axial resolution "
        "is only one ingredient of this width",
        status, "closed form (equal Gaussian rings); separation thresholds from simulation"))
    # localizations per cluster of the reference ring
    lo, hi = FMIN_X_RANGE
    if math.isfinite(fm.x):
        value = (f"x = {fm.x:.1f} ({n_central} localizations / {k_central} clusters in the {ref_word} ring / "
                 f"min_samples {ms})")
        if fmin_src == FMIN_SOURCE_THINNING:
            st = LF_OK
            need = (f"every ring needs >= {_pct(fm.f_min)} of the {ref_word} ring ({fm.note}); brighter clusters "
                    "lower f_min but raise the leak copies at equal separation")
        else:
            st = LF_LIMITING if fm.x < lo else (LF_OUTSIDE if fm.x > hi else LF_OK)
            need = (f"suggestion from a simulation line valid for x {lo:g}-{hi:g}: here every ring needs >= "
                    f"{_pct(fm.f_min)} of the {ref_word} ring" + (f" ({fm.note})" if fm.note else "")
                    + "; calibrate the count floor on this axon to replace it")
    else:
        value, st = f"not computable (no {ref_word} ring with clusters)", LF_INFO
        need = f"validated x {lo:g}-{hi:g}"
    out.append(LimitingFactor("locs_per_cluster", "Localizations per cluster (x)", value, need, st,
                              fmin_src))
    # z-filter truncation
    cut = [e for e in edges if e.truncated]
    if cut:
        value = "; ".join(f"{e.side} edge at {e.z_nm:.0f} nm still holds {_pct(e.edge_frac)} of the peak density"
                          for e in cut)
        st = LF_WARNING
    else:
        value = "no hard edge: the profile falls off before the data ends"
        st = LF_OK
    out.append(LimitingFactor(
        "z_truncation", f"z-filter truncation ({z_src})", value,
        "the profile should fall to ~0 before the z filter: a cut edge ring's % is underestimated and its 'peak' "
        "may be the cut", st, "descriptive"))
    # best axial precision
    foc = [r for r in rings if r.is_in_focus]
    cen = [r for r in rings if r.is_central]
    if foc:
        f0 = foc[0]
        posn = ("top" if f0 is rings[-1] else "bottom" if f0 is rings[0] else "middle") if n > 1 else "only"
        value = f"ring {f0.index} ({posn}; median lpz {f0.lpz_median_nm:.0f} nm)"
        if cen and cen[0] is not f0:
            value += f"; the {ref_word} ring {cen[0].index} has median lpz {cen[0].lpz_median_nm:.0f} nm"
        st = LF_INFO
    else:
        value, st = "unknown (no lpz)", LF_INFO
    out.append(LimitingFactor(
        "focus", "Ring with the best axial precision (smallest median lpz)", value,
        "information: astigmatic axial precision degrades away from focus; this ring need not be the reference one",
        st, "Huang et al. 2008"))
    # leak (criterion 4)
    fin = [p for p in pairs if math.isfinite(p.exp_spur_frac)]
    if fin:
        bp = min(fin, key=lambda p: p.exp_spur_frac - p.spur_viable)
        st = LF_OK if bp.exp_spur_frac <= bp.spur_viable else LF_LIMITING
        value = (f"best pair {bp.ring_a}-{bp.ring_b}: expected copies {100 * bp.exp_spur_frac:.1f} % vs tolerated "
                 f"{_tol_text(bp.spur_viable)}")
        need = (f"<= the pair's viable tolerance ({_tol_text(bp.spur_viable)}), <= its marginal one "
                f"({_tol_text(bp.spur_marginal)}); {bp.tolerance_source}" if bp.tolerance_source else
                f"<= {_tol_text(bp.spur_viable)} viable, <= {_tol_text(bp.spur_marginal)} marginal")
    else:
        st, value = LF_INFO, "no pair"
        need = f"<= {_tol_text(V2_SPUR_VIABLE)} viable, <= {_tol_text(V2_SPUR_MARGINAL)} marginal (suggestion)"
    out.append(LimitingFactor(
        "leak", "Expected leak copies (ring-level estimate)", value, need, st, "rule v2 criterion 4"))
    return tuple(out)


def viability_v2_key(v2: AxonViabilityV2, mode: str) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """(number of rings, selected pairs sorted) under rule v2: the
    identity a simulated replicate must share with the observed axon.
    ``mode`` "viable" or "viable+marginal"."""
    if mode not in Z_SELECTION_MODES:
        raise ValueError(f"mode must be one of {Z_SELECTION_MODES}, got {mode!r}")
    pairs = v2.viable_pairs if mode == "viable" else v2.viable_pairs + v2.marginal_pairs
    return len(v2.rings), tuple(sorted(pairs))
