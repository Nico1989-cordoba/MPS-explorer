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
through the astigmatism calibration and nothing else (02_investigacion
section 2.2 B2). It knows nothing of calibration error, of focal drift
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
``z_quality``                     D-39 first filter: per ring pair the
                                  separation d, valley, analytic leak,
                                  expected leak copies, events vs the
                                  Delta BIC requirement, and the verdict
                                  viable / marginal / not viable
``viability_v2``                  D-41 ring-pair viability rule v2: SiZer
                                  peak + valley (tools.mps_sizer), the
                                  localization minimum vs the central
                                  ring, the D-39 leak tiers, and the
                                  limiting factors of the axon
``calibrated_lpz_nm``             lpz x the NeNA ratio of the stratum of
                                  each localization's lab z (D-19, H5)

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
  instead of plugging in their mean: the tail is convex in the width,
  so the mean lpz underestimates the leak (Jensen) whenever the
  precisions are heterogeneous, which in an astigmatic z they always
  are.

Nothing here draws random numbers; two runs on the same input are
bit-identical.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

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
    evaluating the tail matters: the tail is convex in the width, so
    the value at the mean lpz is a lower bound (Jensen), and the gap
    grows with the spread of the precisions.

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
# D-39(b) thresholds, fixed on the simulated grid of the H5-E research
# (zquality/zq_thresholds.py, 2026-09-29) BEFORE any real axon was
# measured. The separation index d = |mu2 - mu1| / sqrt(s1^2 + s2^2)
# guards against unresolved boundaries (sigma/P <= 0.32 / 0.35 at equal
# widths); the expected leak-copy fraction is the analytic D-27c count;
# the information ratio asks for twice the events that give an expected
# Delta BIC of 10 (strong evidence of two components, ~95 % power).
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
# The default reference period P (D-15: the median axial period measured on
# the exploratory dataset) when the caller has no better reference for P.
P_REF_DEFAULT_NM = 170.8
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
# variable ``UNCALIBRATED_LABELS_ENV``; neither set (the default): no flag.
UNCALIBRATED_LABELS_ENV = "MPS_UNCALIBRATED_LABELS"
ROI2_PATTERN: Optional["re.Pattern[str]"] = None
ROI2_FLAG = "fuera del rango calibrado (adquisicion marcada)"
# The (d_sep, exp_spur_frac) range of the accepted S4 originals, filled
# in H5-E S5 as {"d_sep_min": ..., "exp_spur_frac_max": ...}; None until
# then (no envelope flag is raised).
CALIBRATED_ENVELOPE: Optional[Dict[str, float]] = None


def uncalibrated_label_pattern() -> Optional["re.Pattern[str]"]:
    """The pattern of ``calibration_range_flags``' acquisition flag: ``ROI2_PATTERN`` when set, else the regular
    expression in the ``MPS_UNCALIBRATED_LABELS`` environment variable (read at call time, so worker processes see
    it), else None (no input is flagged). An invalid expression raises ValueError naming the variable."""
    if ROI2_PATTERN is not None:
        return ROI2_PATTERN
    text = os.environ.get(UNCALIBRATED_LABELS_ENV, "").strip()
    if not text:
        return None
    try:
        return re.compile(text)
    except re.error as exc:
        raise ValueError(f"{UNCALIBRATED_LABELS_ENV} is not a valid regular expression: {exc}") from exc

_MIN_SIGMA_NM = 1e-6
_MIN_KEEP = 0.05


@dataclass(frozen=True)
class ZQualityThresholds:
    """The D-39(b) verdict thresholds; the defaults are the frozen ones."""

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
                     thresholds: ZQualityThresholds = ZQualityThresholds()) -> Tuple[str, Tuple[str, ...]]:
    """
    The D-39(b) verdict of one pair, with its reasons.

    VIABLE: an interior valley, no merge block, d >= d_viable, expected
    copies <= spur_viable and (without events, not applied) info ratio
    >= info_ratio_min. MARGINAL: valley, no block, d >= d_marginal and
    copies <= spur_marginal. Otherwise NOT VIABLE. A block is a spacing
    over merged_gap_over_p x P_ref (a lost ring) or a ring of the pair
    wider than merged_sigma_over_p x P_ref (``ring_flags``, one tuple per
    ring). Same rule and reason texts as the research prototype
    (zq_core.verdict), plus one reason for too few events.
    """
    t = thresholds
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
    if valley and not blocked and d_sep >= t.d_viable and exp_spur_frac <= t.spur_viable and info_ok:
        verdict = VERDICT_VIABLE
    elif valley and not blocked and d_sep >= t.d_marginal and exp_spur_frac <= t.spur_marginal:
        verdict = VERDICT_MARGINAL
    else:
        verdict = VERDICT_NOT_VIABLE
    if d_sep < t.d_marginal:
        reasons.append(f"d_sep {d_sep:.2f} < {t.d_marginal:g}")
    elif d_sep < t.d_viable:
        reasons.append(f"d_sep {d_sep:.2f} < {t.d_viable:g}")
    if exp_spur_frac > t.spur_marginal:
        reasons.append(f"expected leak copies {100 * exp_spur_frac:.1f} % > {100 * t.spur_marginal:g} %")
    elif exp_spur_frac > t.spur_viable:
        reasons.append(f"expected leak copies {100 * exp_spur_frac:.1f} % > {100 * t.spur_viable:g} %")
    if verdict != VERDICT_VIABLE and math.isfinite(info_ratio) and info_ratio < t.info_ratio_min:
        reasons.append(f"events {info_ratio:.2f} x n_req < {t.info_ratio_min:g}")
    return verdict, tuple(reasons)


def _expected_copies(ring: Any, f_to: float, mu_all: NDArray[np.float64], sd_all: NDArray[np.float64],
                     min_samples: int) -> float:
    """D-27c: sum over the ring's clusters of P[Bin(n0, F) >= min_samples],
    n0 = n_locs / kept fraction of the ring's component inside its own
    slab (floored at 5 %)."""
    c = int(ring.component_index)
    m, s = float(mu_all[c]), max(float(sd_all[c]), _MIN_SIGMA_NM)
    keep = float(norm.cdf((float(ring.z_hi_nm) - m) / s) - norm.cdf((float(ring.z_lo_nm) - m) / s))
    keep = min(max(keep, _MIN_KEEP), 1.0)
    return float(sum(float(binom.sf(min_samples - 1, int(round(int(cl.n_locs) / keep)), f_to)) for cl in ring.clusters))


def z_quality(res: Any, *, p_ref_nm: Optional[float] = None, min_samples: Optional[int] = None,
              posterior_min: float = ZQ_POSTERIOR_MIN,
              thresholds: ZQualityThresholds = ZQualityThresholds()) -> ZQuality:
    """
    How much axial information separates every consecutive ring pair of
    a ``build_rings`` result, and the D-39 verdict of each pair.

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
    p_ref_nm : the axial period the merge rules compare with; None ->
        ``P_REF_DEFAULT_NM`` (the source is recorded).
    min_samples : DBSCAN min_samples of the copy count; None ->
        ``res.params.min_samples``.
    posterior_min : a localization with a two-component posterior below
        this counts as ambiguous.
    thresholds : the D-39(b) thresholds.

    Returns
    -------
    ZQuality
        Rings bottom to top, one ``BoundaryQuality`` per consecutive pair
        and the axon verdict (the best pair's; ``AXON_FEWER_THAN_TWO_RINGS``
        with fewer than two rings).
    """
    if p_ref_nm is None:
        p_ref, p_src = P_REF_DEFAULT_NM, "default (exploratory-dataset median P, D-15)"
    else:
        p_ref, p_src = float(p_ref_nm), "argument"
    ms_min = int(res.params.min_samples) if min_samples is None else int(min_samples)
    t = thresholds
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
        e_ab = _expected_copies(ra, f_lo, mu_all, sd_all, ms_min)
        e_ba = _expected_copies(rb, f_hi, mu_all, sd_all, ms_min)
        spur = float((e_ab + e_ba) / (k_a + k_b)) if (k_a + k_b) else float("nan")
        kl = kl_mixture_vs_gauss(mu, sd, w)
        n_req = n_required(kl)
        n_ev: Optional[int] = None
        if ev is not None:
            e = ev[li]
            n_ev = int(np.unique(e[e >= 0]).size) + int(np.count_nonzero(e < 0))
        info = float(n_ev / n_req) if (n_ev and math.isfinite(n_req)) else float("nan")
        gap = float(mu[1] - mu[0])
        verdict, reasons = boundary_verdict(d_sep=d_sep, exp_spur_frac=spur, info_ratio=info, valley_depth=valley,
                                            gap_nm=gap, ring_flags=(flags_of[ia], flags_of[ib]), rings=(ia, ib),
                                            p_ref_nm=p_ref, thresholds=t)
        bounds.append(BoundaryQuality(
            ring_a=ia, ring_b=ib, d_sep=d_sep, ashman_d=d_sep * math.sqrt(2.0), valley_depth=valley, gap_nm=gap,
            cut_nm=cut, f_lo=f_lo, f_hi=f_hi, misassign=float(w[0] * f_lo + w[1] * f_hi), ambiguous_frac=amb,
            k_a=k_a, k_b=k_b, exp_spur_frac=spur, kl_nats=kl, n_req_events=n_req, n_events=n_ev, info_ratio=info,
            verdict=verdict, reasons=reasons))

    order = {VERDICT_VIABLE: 2, VERDICT_MARGINAL: 1, VERDICT_NOT_VIABLE: 0}
    if bounds:
        best = max(order[b.verdict] for b in bounds)
        axon = {2: VERDICT_VIABLE, 1: VERDICT_MARGINAL, 0: VERDICT_NOT_VIABLE}[best]
    else:
        axon = AXON_FEWER_THAN_TWO_RINGS
    return ZQuality(
        rings=tuple(ring_q), boundaries=tuple(bounds), axon_verdict=axon,
        viable_pairs=tuple((b.ring_a, b.ring_b) for b in bounds if b.verdict == VERDICT_VIABLE),
        marginal_pairs=tuple((b.ring_a, b.ring_b) for b in bounds if b.verdict == VERDICT_MARGINAL),
        p_ref_nm=p_ref, p_ref_source=p_src, min_samples=ms_min, thresholds=t, warnings=tuple(warnings))


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
    source path matches ``uncalibrated_label_pattern()`` (``ROI2_PATTERN``
    or the ``MPS_UNCALIBRATED_LABELS`` regular expression; none by
    default). After H5-E S5 fills
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
            out.append("pares " + ", ".join(bad) + ": fuera del rango calibrado (d_sep / copias esperadas fuera de lo "
                       "calibrado en S4)")
    return tuple(out)


# ===================================================================
#  Ring-pair viability rule v2 (D-41): SiZer peak + valley,
#  localization minimum vs the central ring, D-39 leak tiers
# ===================================================================
# The rule the user adopted on 2026-10-02 (Q-34, Q-35), from the zmin
# research (its report is with the private research notes):
#   (1) every ring of the pair has a significant SiZer PEAK within P/3 of
#       its centre and (2) there is a significant VALLEY between the two
#       ring centres (cluster-robust SiZer, h = 0.10 P, alpha 0.05,
#       simultaneous quantile; groups = 3D DBSCAN clusters);
#   (3) both rings hold f = n_locs(ring) / n_locs(central) >= f_min(x),
#       f_min(x) = 1.46 - 0.336 ln x, x = localizations per cluster of the
#       central ring / min_samples (fitted on x in [5.8, 13.3]; above the
#       range f_min(13.3), below it the formula capped at 100 %, both
#       flagged);
#   (4) the D-39 expected leak-copy fraction exp_spur_frac <= 2 % ->
#       VIABLE; <= 5 % with (1)-(3) -> MARGINAL (sensitivity only);
#       anything else NOT VIABLE.
VIABILITY_RULE_V2 = ("viability rule v2 (D-41): cluster-robust SiZer peak + valley (h 0.10 P, alpha 0.05), "
                     "localizations >= f_min(x) of the central ring, D-39 leak copies <= 2 % / 5 %")
V2_PEAK_WINDOW_OVER_P = 1.0 / 3.0
FMIN_INTERCEPT = 1.46
FMIN_SLOPE_PER_LN_X = -0.336
FMIN_X_RANGE = (5.8, 13.3)
FMIN_CAP = 1.0
V2_SPUR_VIABLE = ZQ_SPUR_VIABLE
V2_SPUR_MARGINAL = ZQ_SPUR_MARGINAL
# Reference numbers of the limiting-factor summary (descriptive; never a
# criterion). zmin report section 2: with sigma_z/P = 0.30 the SiZer
# valley is detected 80 % of the time when the weak ring has ~81 % of the
# central ring (4000 central localizations); at 0.36 not even with equal
# rings; D-39(b): d >= 2.2 is sigma_z/P <= 0.32 at equal widths.
LF_SIGMA_OVER_P_VALLEY = 0.30
LF_SIGMA_OVER_P_NONE = 0.36
LF_SIGMA_OVER_P_D39 = 0.32
# z-filter truncation flag (descriptive): an edge of the lab z range is
# "cut" when the TRUNC_BIN_NM next to it still hold >= TRUNC_EDGE_FRAC of
# the fullest bin and >= TRUNC_MIN_COUNT localizations (a natural tail
# ends at ~0). zmin report section 4: a z filter can cut the profile
# where the density is still a sizeable fraction of the maximum.
TRUNC_BIN_NM = 20.0
TRUNC_EDGE_FRAC = 0.05
TRUNC_MIN_COUNT = 10
LF_LIMITING = "limiting"
LF_OK = "ok"
LF_INFO = "info"
LF_WARNING = "warning"
LF_OUTSIDE = "outside calibrated range"


@dataclass(frozen=True)
class FMin:
    """f_min(x) of criterion 3: the minimum fraction of the central
    ring's localizations, the x it came from, whether x is inside the
    fitted range and, when not, what was done ("" inside)."""

    f_min: float
    x: float
    in_range: bool
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
    central ring, clusters, localizations per cluster, axial width and
    precision, the SiZer peak (and where), and its roles."""

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
    their numbers, the D-39 verdict next to it, the verdict and the plain
    reasons it is not better."""

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
    """``viability_v2``'s answer for one axon: the rule (with version),
    the rings and pairs, the central and in-focus rings (ring indices),
    criterion 3's x and f_min, the SiZer profile, the z-edge flags, the
    limiting factors and warnings."""

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


def f_min_of_x(x: float) -> FMin:
    """
    Criterion 3's threshold, f_min(x) = 1.46 - 0.336 ln x (zmin part B,
    max residual 0.016 on x in [5.8, 13.3]; 77 % [74.5; 79.2] at x ~ 8.1,
    the fraction that keeps >= 90 % of the central ring's cluster recall).

    Outside the fitted range it stays conservative and says so: x above
    13.3 takes f_min(13.3) (the formula keeps falling there), x below 5.8
    takes the extrapolated formula capped at 100 %. A non-finite or
    non-positive x gives NaN (the criterion then fails).
    """
    lo, hi = FMIN_X_RANGE
    if not (isinstance(x, (int, float, np.floating)) and math.isfinite(float(x)) and float(x) > 0):
        return FMin(float("nan"), float("nan"), False, "x not computable (central ring without clusters)")
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
    Q-35: the central ring of a stack (rings bottom to top, given by
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


def _pct(f: float) -> str:
    return f"{100.0 * f:.0f} %"


def verdict_v2(*, rings: Tuple[int, int], peak_a: bool, peak_b: bool, valley: bool, f_a: float, f_b: float,
               f_min: float, exp_spur_frac: float, x: float = float("nan"), x_note: str = "",
               p_ref_nm: float = P_REF_DEFAULT_NM) -> Tuple[str, Tuple[str, ...]]:
    """
    The rule-v2 verdict of one pair with its plain-language reasons.

    Criteria (1)-(3) must all hold; then the D-39 leak tier decides:
    exp_spur_frac <= 2 % VIABLE, <= 5 % MARGINAL (sensitivity only), else
    (or not finite) NOT VIABLE.
    """
    a, b = rings
    reasons: List[str] = []
    win = V2_PEAK_WINDOW_OVER_P * p_ref_nm
    for k, pk in ((a, peak_a), (b, peak_b)):
        if not pk:
            reasons.append(f"ring {k}: no significant peak within {win:.0f} nm of its centre (SiZer)")
    if not valley:
        reasons.append(f"no significant valley between rings {a} and {b} (the profile between them is flat or one-sided)")
    count_ok = math.isfinite(f_min) and all(math.isfinite(f) and f >= f_min for f in (f_a, f_b))
    if not math.isfinite(f_min):
        reasons.append("localization minimum not computable (central ring without clusters)")
    else:
        for k, f in ((a, f_a), (b, f_b)):
            if not (math.isfinite(f) and f >= f_min):
                reasons.append(f"ring {k} has {_pct(f) if math.isfinite(f) else 'an unknown fraction'} of the central "
                               f"ring's localizations; the minimum is {_pct(f_min)}"
                               + (f" (x = {x:.1f})" if math.isfinite(x) else "") + (f"; {x_note}" if x_note else ""))
    crit = peak_a and peak_b and valley and count_ok
    spur = float(exp_spur_frac)
    if not math.isfinite(spur):
        reasons.append("expected leak copies unknown (no axial mixture component)")
    elif spur > V2_SPUR_MARGINAL:
        reasons.append(f"expected leak copies {100 * spur:.1f} % > {100 * V2_SPUR_MARGINAL:g} %")
    elif spur > V2_SPUR_VIABLE:
        # "marginal" only when it is the verdict: a pair failing (1)-(3) is NOT VIABLE whatever its leak
        reasons.append(f"expected leak copies {100 * spur:.1f} % > {100 * V2_SPUR_VIABLE:g} % "
                       + ("(marginal: sensitivity only, D-39 d)" if crit else
                          f"(VIABLE needs <= {100 * V2_SPUR_VIABLE:g} %)"))
    if crit and math.isfinite(spur) and spur <= V2_SPUR_VIABLE:
        return VERDICT_VIABLE, tuple(reasons)
    if crit and math.isfinite(spur) and spur <= V2_SPUR_MARGINAL:
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


def viability_v2(res: Any, zq: Optional[ZQuality] = None, *, xyz_lab_nm: Optional[Sequence[NDArray[np.float64]]] = None,
                 groups: Optional[NDArray[np.int64]] = None, p_ref_nm: Optional[float] = None) -> AxonViabilityV2:
    """
    The ring-pair viability rule v2 (D-41) of a ``build_rings`` result.

    Parameters
    ----------
    res : a ``RingsResult`` (duck-typed): ``rings`` (index, centre_z_nm,
        loc_index, clusters, sigma_z_nm, lpz_median_nm), ``z_p`` (every
        input localization, input order), ``x_p`` / ``y_p`` and
        ``params.min_samples``.
    zq : its ``z_quality`` (the leak numbers of criterion 4); None -> run.
    xyz_lab_nm : the localizations' LAB x, y, z (input order) for the 3D
        DBSCAN groups of the cluster-robust variance -- the coordinates
        the zmin research used. Without it the axon-frame coordinates are
        used and a warning says so.
    groups : the groups themselves (overrides ``xyz_lab_nm``).
    p_ref_nm : the period P of h = 0.10 P and of the peak window P/3;
        None -> ``zq.p_ref_nm`` or ``P_REF_DEFAULT_NM``.

    Returns
    -------
    AxonViabilityV2
        Rings bottom to top, one ``PairViabilityV2`` per consecutive pair
        (none with fewer than two rings), the limiting factors.
    """
    from tools.mps_sizer import GROUP_EPS_NM, GROUP_MIN_SAMPLES, GROUP_Z_SCALE, SIZER_ALPHA, SIZER_H_OVER_P
    from tools.mps_sizer import groups_3d, sizer_profile, valley_between

    if p_ref_nm is None:
        p_ref_nm = getattr(zq, "p_ref_nm", None) if zq is not None else None
    p_ref = float(p_ref_nm) if p_ref_nm is not None else P_REF_DEFAULT_NM
    if zq is None:
        zq = z_quality(res, p_ref_nm=p_ref)
    warnings: List[str] = []
    rings = sorted(res.rings, key=lambda r: int(r.index))
    ms = int(res.params.min_samples)
    n_locs = [int(np.asarray(r.loc_index).size) for r in rings]
    k_cl = [len(r.clusters) for r in rings]
    pos_c = central_ring_position(n_locs)
    zp = np.asarray(res.z_p, dtype=np.float64)

    # ---- groups of the cluster-robust variance
    if groups is not None:
        g = np.asarray(groups, dtype=np.int64)
        g_src = "given"
    elif xyz_lab_nm is not None:
        x, y, z = (np.asarray(v, dtype=np.float64) for v in xyz_lab_nm)
        g = groups_3d(x, y, z, GROUP_EPS_NM, GROUP_MIN_SAMPLES, GROUP_Z_SCALE)
        g_src = "lab coordinates (3D DBSCAN)"
    else:
        g = groups_3d(np.asarray(res.x_p, dtype=np.float64), np.asarray(res.y_p, dtype=np.float64), zp,
                      GROUP_EPS_NM, GROUP_MIN_SAMPLES, GROUP_Z_SCALE)
        g_src = "axon-frame coordinates (3D DBSCAN; lab coordinates not given)"
        warnings.append("SiZer groups from the axon-frame coordinates: the rule is defined on the lab coordinates")
    if g.size != zp.size:
        raise ValueError(f"viability_v2: {g.size} group labels for {zp.size} localizations")
    n_groups = int(np.unique(g).size) if g.size else 0

    # ---- SiZer on the whole z' profile
    profile = None
    if zp.size >= 2 and np.ptp(zp) > 0:
        profile = sizer_profile(zp, g, p_ref_nm=p_ref, c=SIZER_H_OVER_P, alpha=SIZER_ALPHA, robust=True)
    else:
        warnings.append("fewer than two distinct z values: no SiZer profile")
    peaks = list(profile.peaks) if profile is not None else []
    win = V2_PEAK_WINDOW_OVER_P * p_ref

    # ---- central ring, x and f_min
    if pos_c is not None and k_cl[pos_c] > 0 and ms > 0:
        x_c = n_locs[pos_c] / k_cl[pos_c] / ms
    else:
        x_c = float("nan")
    fm = f_min_of_x(x_c)
    n_central = n_locs[pos_c] if pos_c is not None else 0

    # ---- in-focus ring and z edges
    lpz = [float(r.lpz_median_nm) for r in rings]
    fin = [i for i, v in enumerate(lpz) if math.isfinite(v)]
    pos_f = min(fin, key=lambda i: lpz[i]) if fin else None
    z_edge_src = "lab z" if xyz_lab_nm is not None else "axon-frame z'"
    z_for_edges = np.asarray(xyz_lab_nm[2], dtype=np.float64) if xyz_lab_nm is not None else zp
    edges = z_edge_truncation(z_for_edges)

    sig_over_p = {int(q.index): float(q.sigma_over_p) for q in getattr(zq, "rings", ())}
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
            sigma_over_p=sig_over_p.get(int(r.index), sig / p_ref), lpz_median_nm=lpz[i], peak=bool(near),
            peak_z_nm=(min(near, key=lambda p: abs(p - cen)) if near else None), is_central=(i == pos_c),
            is_in_focus=(i == pos_f), truncated_edge=trunc))

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
        verdict, reasons = verdict_v2(rings=(ra.index, rb.index), peak_a=ra.peak, peak_b=rb.peak, valley=valley,
                                      f_a=ra.f_of_central, f_b=rb.f_of_central, f_min=fm.f_min, exp_spur_frac=spur,
                                      x=fm.x, x_note=fm.note, p_ref_nm=p_ref)
        count_ok = math.isfinite(fm.f_min) and all(math.isfinite(f) and f >= fm.f_min
                                                    for f in (ra.f_of_central, rb.f_of_central))
        pair_v.append(PairViabilityV2(
            ring_a=ra.index, ring_b=rb.index, peak_a=ra.peak, peak_b=rb.peak, valley=valley, f_a=ra.f_of_central,
            f_b=rb.f_of_central, f_min=fm.f_min, x=fm.x, x_in_range=fm.in_range, count_ok=count_ok,
            exp_spur_frac=spur, d_sep=(float(b.d_sep) if b is not None else float("nan")),
            d39_verdict=(str(b.verdict) if b is not None else ""), verdict=verdict, reasons=reasons))

    factors = _limiting_factors(ring_v, pair_v, edges, z_edge_src, fm, n_central,
                                (k_cl[pos_c] if pos_c is not None else 0), ms, p_ref)
    return AxonViabilityV2(
        rule=VIABILITY_RULE_V2, p_ref_nm=p_ref, h_nm=(profile.h_nm if profile is not None else 0.1 * p_ref),
        alpha=(profile.alpha if profile is not None else 0.05), min_samples=ms, rings=tuple(ring_v),
        pairs=tuple(pair_v), central_index=(ring_v[pos_c].index if pos_c is not None else None),
        in_focus_index=(ring_v[pos_f].index if pos_f is not None else None), x=fm.x, f_min=fm.f_min,
        x_in_range=fm.in_range, x_note=fm.note, groups_source=g_src, n_groups=n_groups, profile=profile,
        z_edges=edges, z_edges_source=z_edge_src, limiting_factors=factors,
        viable_pairs=tuple((p.ring_a, p.ring_b) for p in pair_v if p.verdict == VERDICT_VIABLE),
        marginal_pairs=tuple((p.ring_a, p.ring_b) for p in pair_v if p.verdict == VERDICT_MARGINAL),
        warnings=tuple(warnings))


def _limiting_factors(rings: Sequence[RingViabilityV2], pairs: Sequence[PairViabilityV2],
                      edges: Tuple[EdgeTruncation, EdgeTruncation], z_src: str, fm: FMin, n_central: int,
                      k_central: int, ms: int, p_ref: float) -> Tuple[LimitingFactor, ...]:
    """The axon-level summary for the microscopy team: what limits the
    ring comparison, with the numbers the rule and the reports use."""
    out: List[LimitingFactor] = []
    # rings
    n = len(rings)
    out.append(LimitingFactor(
        "rings", "Rings detected", f"{n} ring(s), {max(n - 1, 0)} consecutive pair(s)",
        "at least 2 rings (one pair) to compare clusters between rings",
        LF_LIMITING if n < 2 else LF_OK, "D-41 (Q-35)"))
    # axial resolution: the RMS width of the best pair's two rings over P (= 1 / (sqrt 2 d) at a gap of P)
    cut_in_best: List[int] = []
    if pairs:
        by_idx = {r.index: r for r in rings}
        rms = [math.sqrt(0.5 * (by_idx[p.ring_a].sigma_z_nm ** 2 + by_idx[p.ring_b].sigma_z_nm ** 2)) / p_ref
               for p in pairs]
        best = min(rms)
        bp = pairs[rms.index(best)] if best in rms else pairs[0]
        where = f"best pair {bp.ring_a}-{bp.ring_b} (RMS width of its two rings / P = {p_ref:.1f} nm)"
        # a ring whose data the z filter cuts is fitted narrower than it is: its pair's width is not evidence
        cut_in_best = [k for k in (bp.ring_a, bp.ring_b) if by_idx[k].truncated_edge]
        if cut_in_best:
            where += (f"; ring {', '.join(str(k) for k in cut_in_best)} sits at a cut z edge, so its width is "
                      f"underestimated (the other pairs: up to {max(rms):.3f})")
    elif rings:
        best = float(np.median([r.sigma_z_nm for r in rings])) / p_ref
        where = f"ring width / P = {p_ref:.1f} nm"
    else:
        best = float("nan")
        where = "no ring"
    # judged on the value shown (three decimals): a reference from the simulations, not a hard threshold
    shown = round(best, 3) if math.isfinite(best) else best
    status = LF_LIMITING if (math.isfinite(shown) and shown > LF_SIGMA_OVER_P_VALLEY) else (
        (LF_WARNING if cut_in_best else LF_OK) if math.isfinite(shown) else LF_INFO)
    out.append(LimitingFactor(
        "axial_resolution", "Axial resolution (sigma_z / P)", f"{shown:.3f} ({where})" if math.isfinite(best) else where,
        f"smaller is better, no hard threshold: in the simulations (zmin report sec. 2) SiZer found the valley in ~80 % "
        f"of simulated profiles (4000 localizations in the central ring) at {LF_SIGMA_OVER_P_VALLEY:.2f} when the "
        f"weaker ring holds ~81 % of the central ring, and not in "
        f"80 % even with equal rings at {LF_SIGMA_OVER_P_NONE:.2f}; D-39 'viable' asks d >= 2.2 "
        f"(~{LF_SIGMA_OVER_P_D39:.2f})",
        status, "D-41 / zmin report sec. 2; D-39(b)"))
    # localizations per cluster of the central ring
    lo, hi = FMIN_X_RANGE
    if math.isfinite(fm.x):
        value = (f"x = {fm.x:.1f} ({n_central} localizations / {k_central} clusters in the central ring / "
                 f"min_samples {ms})")
        st = LF_LIMITING if fm.x < lo else (LF_OUTSIDE if fm.x > hi else LF_OK)
        need = (f"validated x {lo:g}-{hi:g} (f_min 87-60 %); here every ring needs >= {_pct(fm.f_min)} of the central "
                "ring" + (f" ({fm.note})" if fm.note else "") + "; brighter clusters lower f_min but raise the leak "
                "copies at equal separation (D-39 a)")
    else:
        value, st = "not computable (no central ring with clusters)", LF_INFO
        need = f"validated x {lo:g}-{hi:g}"
    out.append(LimitingFactor("locs_per_cluster", "Localizations per cluster (x)", value, need, st,
                              "D-41 (Q-34 criterion 3) / zmin report sec. 3"))
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
        "may be the cut", st, "zmin report sec. 4"))
    # focus
    foc = [r for r in rings if r.is_in_focus]
    cen = [r for r in rings if r.is_central]
    if foc:
        f0 = foc[0]
        posn = ("top" if f0 is rings[-1] else "bottom" if f0 is rings[0] else "middle") if n > 1 else "only"
        value = f"ring {f0.index} ({posn}; median lpz {f0.lpz_median_nm:.0f} nm)"
        if cen and cen[0] is not f0:
            value += f"; the central ring {cen[0].index} has median lpz {cen[0].lpz_median_nm:.0f} nm"
        st = LF_INFO
    else:
        value, st = "unknown (no lpz)", LF_INFO
    out.append(LimitingFactor(
        "focus", "Ring in focus (smallest median lpz)", value,
        "information: the in-focus ring need not be the central one", st,
        "zmin report sec. 1"))
    # leak (criterion 4)
    sp = [p.exp_spur_frac for p in pairs if math.isfinite(p.exp_spur_frac)]
    if sp:
        best_sp = min(sp)
        st = LF_OK if best_sp <= V2_SPUR_VIABLE else LF_LIMITING
        value = f"best pair {100 * best_sp:.1f} % expected leak copies"
    else:
        st, value = LF_INFO, "no pair"
    out.append(LimitingFactor(
        "leak", "Expected leak copies (D-39)", value,
        f"<= {100 * V2_SPUR_VIABLE:g} % viable, <= {100 * V2_SPUR_MARGINAL:g} % marginal", st,
        "D-39(a, b); D-41 criterion 4"))
    return tuple(out)


def viability_v2_key(v2: AxonViabilityV2, mode: str) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    """(number of rings, selected pairs sorted) under rule v2: the
    identity a simulated replicate must share with the observed axon
    (Q-29 + D-41). ``mode`` "viable" or "viable+marginal"."""
    if mode not in Z_SELECTION_MODES:
        raise ValueError(f"mode must be one of {Z_SELECTION_MODES}, got {mode!r}")
    pairs = v2.viable_pairs if mode == "viable" else v2.viable_pairs + v2.marginal_pairs
    return len(v2.rings), tuple(sorted(pairs))
