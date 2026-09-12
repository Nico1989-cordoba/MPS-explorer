# -*- coding: utf-8 -*-
"""
Gaps and patches around the axonal perimeter, and their correlation
between consecutive MPS segments.

The question this answers -- are the betaII-spectrin patches (and the gaps
between them) of one MPS segment at the same angular positions as those of
the next segment? -- has no published metric to reproduce. What follows is
therefore built from pieces that are already validated elsewhere in this
package, with every choice stated.

Where the profile comes from
----------------------------
Parameter 6 (``mps_occupancy``) already produces exactly the continuous
signal this needs: the perimeter discretized at sub-nanometre spacing, each
sample marked occupied or free by the union of the clusters' constrained
2D Gaussians at 3 sigma Mahalanobis. Patches are the occupied runs of that
mask and gaps are the free runs, so no sector count and no smoothing
bandwidth has to be invented here -- each cluster contributes the angular
footprint that was measured from its own localizations.

That matters for the comparison between segments. Smoothing cluster
CENTRES with a kernel (``mps_multisegment.circular_cross_correlation``)
cannot distinguish a patch that covers 20 degrees of perimeter from one
that covers 2, and its result depends on the bandwidth chosen -- as that
function's own docstring warns. Coverage has neither problem. Both are
computed and reported: agreement is evidence, disagreement is a warning
that the pattern is not what either statistic assumes.

Why significance comes from rotating the profile
------------------------------------------------
A patch spans hundreds of consecutive samples, so a coverage profile is
strongly autocorrelated and its effective degrees of freedom are far fewer
than its sample count. A textbook Pearson p-value on such a pair of
profiles is anti-conservative by orders of magnitude. The accepted remedy
is to build the null by rotating one profile against the other: that
preserves each profile's own autocorrelation exactly and destroys only the
alignment between them. It is the toroidal-shift / random-rotation test of
spatial ecology (Roxburgh & Chesson 1998; Harms et al. 2001) and the
"spin test" used to correlate cortical maps in neuroimaging
(Alexander-Bloch et al. 2018, NeuroImage) -- the same problem of two smooth
maps sharing one domain. Here the whole null distribution falls out of the
same FFT that computes the cross-correlation, so it costs nothing.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_occupancy import (
    ClusterEllipse,
    OccupancyResult,
    discretize_perimeter,
)

# Angular resolution of the shared profile. Patches are small: at ~30 %
# occupancy spread over ~70 clusters on a 25 um perimeter, a patch covers
# roughly 100 nm of arc, which is only ~1.4 degrees at that calibre. A
# 0.1 degree grid therefore puts ~14 samples across a patch, while one
# sample spans ~7 nm -- below the ~20 nm localization precision, so the
# grid cannot be what limits the measurement.
DEFAULT_N_THETA = 3600

# Perimeter samples required per angular bin before a bin's coverage is
# taken as measured rather than interpolated.
DEFAULT_MIN_SAMPLES_PER_BIN = 8


# ============================================================================
# Gap / patch decomposition of one segment
# ============================================================================

@dataclass
class GapPatchStats:
    """Occupied and free runs around one segment's perimeter."""

    n_patches: int
    n_gaps: int
    patch_lengths_nm: NDArray[np.float64]
    gap_lengths_nm: NDArray[np.float64]
    perimeter_nm: float
    occupied_fraction: float
    spacing_nm: float
    fully_occupied: bool
    fully_free: bool

    @property
    def median_patch_nm(self) -> Optional[float]:
        return (float(np.median(self.patch_lengths_nm))
                if self.patch_lengths_nm.size else None)

    @property
    def median_gap_nm(self) -> Optional[float]:
        return (float(np.median(self.gap_lengths_nm))
                if self.gap_lengths_nm.size else None)

    @property
    def max_gap_nm(self) -> Optional[float]:
        return (float(np.max(self.gap_lengths_nm))
                if self.gap_lengths_nm.size else None)

    def export_dict(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}n_patches": self.n_patches,
            f"{prefix}n_gaps": self.n_gaps,
            f"{prefix}median_patch_nm": self.median_patch_nm,
            f"{prefix}median_gap_nm": self.median_gap_nm,
            f"{prefix}max_gap_nm": self.max_gap_nm,
            f"{prefix}occupied_fraction": round(self.occupied_fraction, 4),
            f"{prefix}perimeter_nm": round(self.perimeter_nm, 1),
        }


def gap_patch_runs(
    occupied_mask: NDArray[np.bool_],
    spacing_nm: float,
) -> GapPatchStats:
    """
    Split a perimeter occupancy mask into patches (occupied runs) and gaps
    (free runs).

    The perimeter is a closed loop, so a run straddling the start of the
    array is one run, not two: sampling has to start somewhere, and letting
    that arbitrary point cut a patch in half would shorten one patch and
    invent another on every axon.

    Parameters
    ----------
    occupied_mask : per-sample occupancy along the perimeter, in contour
        order and equally spaced (``OccupancyResult.occupied_mask``).
    spacing_nm : arc length between consecutive samples
        (``OccupancyResult.point_spacing_nm``).

    Returns
    -------
    GapPatchStats
    """
    mask = np.asarray(occupied_mask, dtype=bool).ravel()
    n = mask.size
    if n == 0:
        raise ValueError("occupied_mask is empty.")
    if not np.isfinite(spacing_nm) or spacing_nm <= 0:
        raise ValueError(f"spacing_nm must be positive, got {spacing_nm!r}")

    perimeter = float(n) * float(spacing_nm)
    occupied_fraction = float(np.count_nonzero(mask)) / float(n)

    if mask.all() or (~mask).all():
        empty = np.array([], dtype=float)
        whole = np.array([perimeter], dtype=float)
        return GapPatchStats(
            n_patches=1 if mask.all() else 0,
            n_gaps=0 if mask.all() else 1,
            patch_lengths_nm=whole if mask.all() else empty,
            gap_lengths_nm=empty if mask.all() else whole,
            perimeter_nm=perimeter,
            occupied_fraction=occupied_fraction,
            spacing_nm=float(spacing_nm),
            fully_occupied=bool(mask.all()),
            fully_free=bool((~mask).all()),
        )

    # Rotate so the array starts at a state change: every run is then
    # contiguous and the wrap-around run is not split.
    changes = np.flatnonzero(mask != np.roll(mask, 1))
    rolled = np.roll(mask, -int(changes[0]))

    # Run boundaries of the rotated array.
    edges = np.flatnonzero(rolled != np.roll(rolled, 1))
    run_starts = edges                                  # edges[0] == 0
    run_lengths = np.diff(np.append(run_starts, n))
    run_values = rolled[run_starts]

    lengths_nm = run_lengths.astype(float) * float(spacing_nm)
    patches = lengths_nm[run_values]
    gaps = lengths_nm[~run_values]

    return GapPatchStats(
        n_patches=int(patches.size),
        n_gaps=int(gaps.size),
        patch_lengths_nm=patches,
        gap_lengths_nm=gaps,
        perimeter_nm=perimeter,
        occupied_fraction=occupied_fraction,
        spacing_nm=float(spacing_nm),
        fully_occupied=False,
        fully_free=False,
    )


# ============================================================================
# Angular coverage profile on a shared grid
# ============================================================================

@dataclass
class CoverageProfile:
    """
    Fraction of perimeter occupied at each angle about a shared centre.

    Continuous in [0, 1]: a bin containing perimeter that is half covered
    reads 0.5. Two segments of the same axon share the grid and the centre,
    so their profiles are directly comparable sample by sample.
    """

    theta_deg: NDArray[np.float64]          # (n_theta,) bin centres, 0..360
    coverage: NDArray[np.float64]           # (n_theta,) in [0, 1]
    n_samples: NDArray[np.int64]            # perimeter samples per bin
    center: NDArray[np.float64]             # (2,) shared angular origin
    n_interpolated_bins: int
    perimeter_nm: float
    spacing_nm: float
    occupied_mask: NDArray[np.bool_]        # the underlying arc-length mask
    warnings: List[str] = field(default_factory=list)

    @property
    def n_theta(self) -> int:
        return int(self.theta_deg.size)

    @property
    def mean_coverage(self) -> float:
        return float(np.mean(self.coverage))


def _occupied_by_ellipses(
    points: NDArray[np.float64],
    ellipses: List[ClusterEllipse],
    mahalanobis_threshold: float,
) -> NDArray[np.bool_]:
    """Union of the clusters' constrained Gaussians, evaluated at
    ``points`` -- the same test ``mps_occupancy.compute_occupancy`` applies,
    reusing the ellipses it already fitted."""
    occupied = np.zeros(len(points), dtype=bool)
    thresh_sq = float(mahalanobis_threshold) ** 2
    for e in ellipses:
        d = points - e.mean
        m2 = np.einsum("ij,jk,ik->i", d, e.inv_cov, d)
        occupied |= m2 <= thresh_sq
    return occupied


def coverage_profile(
    occupancy: OccupancyResult,
    contour: NDArray[np.float64],
    center: NDArray[np.float64],
    n_theta: int = DEFAULT_N_THETA,
    min_samples_per_bin: int = DEFAULT_MIN_SAMPLES_PER_BIN,
) -> CoverageProfile:
    """
    Re-express one segment's perimeter occupancy as a function of angle
    about a centre shared with the other segments of the same axon.

    The occupancy mask is indexed by arc length along that segment's own
    contour, and each segment has its own contour, so two segments' masks
    are not comparable sample by sample. Angle about one shared centre is
    what registers them.

    Parameters
    ----------
    occupancy : ``AxonAnalysis.occupancy`` for this segment. Its already
        fitted ellipses are reused; nothing is refitted.
    contour : the segment's perimeter vertices
        (``AxonAnalysis.perimeter.contour``).
    center : (2,) angular origin shared by every segment of the axon
        (``MultiSegmentAnalysis.axon_center``).
    n_theta : bins on the shared angular grid.
    min_samples_per_bin : if the stored occupancy was sampled too coarsely
        to put this many perimeter samples in every bin, the perimeter is
        re-sampled more finely before binning. Without this, bins fall
        empty purely from sampling: 10,000 samples over 3,600 bins average
        2.8 per bin, so ~6 % of bins would be empty by chance alone and the
        "gaps" they showed would be an artefact of the sampling rather than
        a property of the axon.

    Returns
    -------
    CoverageProfile
    """
    if n_theta < 8:
        raise ValueError(f"n_theta must be at least 8, got {n_theta}")

    ctr = np.asarray(center, dtype=float).ravel()
    if ctr.size != 2:
        raise ValueError(f"center must be (2,), got shape {ctr.shape}")

    warnings_: List[str] = []
    points = np.asarray(occupancy.perimeter_points, dtype=float)
    mask = np.asarray(occupancy.occupied_mask, dtype=bool)
    spacing = float(occupancy.point_spacing_nm)

    # --- re-sample if the stored mask is too coarse for this grid --------
    n_needed = int(n_theta) * int(max(min_samples_per_bin, 1))
    if points.shape[0] < n_needed:
        points, total_len, spacing = discretize_perimeter(contour, n_needed)
        mask = _occupied_by_ellipses(
            points, occupancy.ellipses, occupancy.mahalanobis_threshold)
        warnings_.append(
            f"Perimeter re-sampled from {occupancy.n_points:,} to "
            f"{len(points):,} points so every one of the {n_theta:,} angular "
            f"bins holds at least {min_samples_per_bin} samples."
        )
    else:
        total_len = float(occupancy.perimeter_length_nm)

    # --- bin by angle about the shared centre ---------------------------
    d = points - ctr
    theta = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    bin_width = 2.0 * np.pi / n_theta
    idx = np.minimum((theta / bin_width).astype(np.int64), n_theta - 1)

    counts = np.bincount(idx, minlength=n_theta).astype(np.int64)
    occupied_sum = np.bincount(
        idx, weights=mask.astype(float), minlength=n_theta)

    coverage = np.full(n_theta, np.nan, dtype=float)
    filled = counts > 0
    coverage[filled] = occupied_sum[filled] / counts[filled]

    # A contour that doubles back can still leave an angle unvisited. Fill
    # those linearly from the neighbouring angles, on the circle, and say
    # how many: a hole carried into the correlation as zero would read as a
    # gap that was never measured.
    n_interp = int(np.count_nonzero(~filled))
    if n_interp:
        if not np.any(filled):
            raise ValueError(
                "No perimeter sample fell in any angular bin; the contour and "
                "the centre are inconsistent.")
        pos = np.flatnonzero(filled).astype(float)
        val = coverage[filled]
        # Wrap one period either side so interpolation across 0/360 works.
        pos_ext = np.concatenate([pos - n_theta, pos, pos + n_theta])
        val_ext = np.concatenate([val, val, val])
        coverage[~filled] = np.interp(
            np.flatnonzero(~filled).astype(float), pos_ext, val_ext)
        warnings_.append(
            f"{n_interp} of {n_theta} angular bins held no perimeter sample "
            f"(the contour does not reach those angles) and were interpolated "
            f"from their neighbours."
        )

    theta_deg = np.degrees((np.arange(n_theta) + 0.5) * bin_width)

    return CoverageProfile(
        theta_deg=theta_deg,
        coverage=coverage,
        n_samples=counts,
        center=ctr,
        n_interpolated_bins=n_interp,
        perimeter_nm=float(total_len),
        spacing_nm=spacing,
        occupied_mask=mask,
        warnings=warnings_,
    )


# ============================================================================
# Correlation between two profiles
# ============================================================================

@dataclass
class ProfileCorrelation:
    """Correlation between the coverage profiles of two segments."""

    r_at_zero: float                        # aligned as imaged
    offsets_deg: NDArray[np.float64]        # rotation axis
    correlation: NDArray[np.float64]        # r at each rotation
    best_offset_deg: float                  # rotation maximising r
    max_correlation: float
    p_rotation: float                       # see ``profile_correlation``
    n_rotations: int
    warnings: List[str] = field(default_factory=list)

    @property
    def null_mean(self) -> float:
        return float(np.nanmean(self.correlation))

    @property
    def null_sd(self) -> float:
        return float(np.nanstd(self.correlation))

    @property
    def z_vs_null(self) -> Optional[float]:
        """r(0) in standard deviations of the rotation null. Reported
        alongside p because with a few thousand rotations p saturates at
        its floor once r(0) is the largest value in the null."""
        sd = self.null_sd
        if not np.isfinite(sd) or sd <= 0:
            return None
        return float((self.r_at_zero - self.null_mean) / sd)


def profile_correlation(
    profile_a: NDArray[np.float64],
    profile_b: NDArray[np.float64],
) -> ProfileCorrelation:
    """
    Correlate two coverage profiles at every rotation, and test the
    as-imaged alignment against that rotation null.

    Parameters
    ----------
    profile_a, profile_b : coverage on the SAME angular grid, about the
        same centre (``CoverageProfile.coverage``).

    Returns
    -------
    ProfileCorrelation, where

    * ``r_at_zero`` is the Pearson correlation with no rotation applied --
      the quantity the biological question asks about: are the patches of
      consecutive segments at the same angles?
    * ``p_rotation`` is the fraction of rotations whose correlation is at
      least ``r_at_zero``, i.e. a one-sided p-value against the null that
      the two profiles are unrelated in angle. One-sided because the
      hypothesis of aligned patches predicts positive correlation.
      Rotation 0 is part of the null, so p is never below 1/n_rotations.
    * ``max_correlation`` at ``best_offset_deg`` describes a pattern that
      repeats but rotated. It is a maximum over the whole null, so it is
      biased upward by construction and has no p-value of its own -- test
      instead whether the best offsets CONCENTRATE across many segment
      pairs (``mps_multisegment.fold_rotation``).

    Sign convention, verified numerically: ``correlation[k]`` equals
    ``sum_i a[i] * b[i + k]``, so ``best_offset_deg`` is how far B's
    pattern sits AHEAD of A's in angle. It is determined only modulo the
    pattern's own repeat period -- a ring of N evenly spaced patches
    correlates equally well at every multiple of 360/N, and which of those
    equal maxima ``argmax`` returns is arbitrary. Interpret the offset
    through ``fold_rotation``, never as a raw angle.
    """
    a = np.asarray(profile_a, dtype=float).ravel()
    b = np.asarray(profile_b, dtype=float).ravel()
    warnings_: List[str] = []

    if a.size != b.size:
        raise ValueError(
            f"Profiles must share a grid: got {a.size} and {b.size} samples.")
    if a.size < 8:
        raise ValueError(f"Profiles too short to correlate: {a.size} samples.")

    n = a.size
    offsets = np.degrees(np.linspace(0.0, 2.0 * np.pi, n, endpoint=False))

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Profiles contain non-finite values.")

    ac = a - a.mean()
    bc = b - b.mean()
    norm = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if norm <= 0:
        # A segment whose perimeter is entirely covered (or entirely free)
        # has no angular pattern at all, so no correlation is defined.
        return ProfileCorrelation(
            r_at_zero=float("nan"), offsets_deg=offsets,
            correlation=np.full(n, np.nan), best_offset_deg=float("nan"),
            max_correlation=float("nan"), p_rotation=float("nan"),
            n_rotations=n,
            warnings=["One profile is constant (perimeter uniformly occupied "
                      "or uniformly free): no angular correlation exists."],
        )

    # Circular cross-correlation by FFT. A circular shift changes neither
    # the mean nor the norm, so dividing by the two fixed norms gives the
    # exact Pearson correlation at every shift.
    corr = np.fft.irfft(
        np.conjugate(np.fft.rfft(ac)) * np.fft.rfft(bc), n=n) / norm

    r0 = float(corr[0])
    k = int(np.nanargmax(corr))
    p = float(np.count_nonzero(corr >= r0)) / float(n)

    if p <= 1.0 / n:
        warnings_.append(
            f"r(0) = {r0:.3f} exceeds every one of the {n:,} rotations, so p "
            f"is at its floor (< {1.0 / n:.1e}); read z_vs_null for the size "
            f"of the effect."
        )

    return ProfileCorrelation(
        r_at_zero=r0,
        offsets_deg=offsets,
        correlation=corr,
        best_offset_deg=float(offsets[k]),
        max_correlation=float(corr[k]),
        p_rotation=p,
        n_rotations=n,
        warnings=warnings_,
    )


def compare_segment_coverage(
    occupancy_a: OccupancyResult,
    contour_a: NDArray[np.float64],
    occupancy_b: OccupancyResult,
    contour_b: NDArray[np.float64],
    center: NDArray[np.float64],
    n_theta: int = DEFAULT_N_THETA,
    min_samples_per_bin: int = DEFAULT_MIN_SAMPLES_PER_BIN,
) -> Tuple[CoverageProfile, CoverageProfile, ProfileCorrelation]:
    """
    Build both segments' coverage profiles on the shared grid and correlate
    them. Convenience wrapper over ``coverage_profile`` and
    ``profile_correlation``.
    """
    prof_a = coverage_profile(
        occupancy_a, contour_a, center,
        n_theta=n_theta, min_samples_per_bin=min_samples_per_bin)
    prof_b = coverage_profile(
        occupancy_b, contour_b, center,
        n_theta=n_theta, min_samples_per_bin=min_samples_per_bin)
    corr = profile_correlation(prof_a.coverage, prof_b.coverage)
    return prof_a, prof_b, corr
