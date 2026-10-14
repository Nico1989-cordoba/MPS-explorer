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

Where along the membrane (SCI-6)
--------------------------------
Two segments' profiles must be registered on one coordinate before they
can be compared sample by sample. The default is ARC LENGTH along one
common membrane curve: the pooled centroid membrane of every segment's
clusters (the curve the column analysis projects on,
``tools.mps_unroll.fit_centroid_membrane``; cf. elliptic Fourier
descriptors, Kuhl & Giardina 1982), on which every segment's perimeter is
projected, s = 0 being the same point for all of them. It does not need a
star-shaped contour. The polar angle about a shared centre is kept as an
exploratory frame: it is a one-to-one coordinate along a closed curve only
if the curve is star-shaped about that centre, and on folded or notched
cross-sections one angular bin collects several membrane stretches; the
fraction of perimeter samples whose angle runs backwards is reported as a
diagnostic (``CoverageProfile.nonmonotonic_fraction``). When the common
curve cannot be fitted (too few clusters), the angle frame is used, with a
warning.

Why significance comes from rotating the profile
------------------------------------------------
A patch spans hundreds of consecutive samples, so a coverage profile is
strongly autocorrelated and its effective degrees of freedom are far fewer
than its sample count. A textbook Pearson p-value on such a pair of
profiles is anti-conservative by orders of magnitude. The accepted remedy
is to build the null by rotating one profile against the other: that
preserves each profile's own autocorrelation exactly and destroys only the
alignment between them. It is the toroidal-shift test of Lotwick &
Silverman (1982) on the circle (the torus translation of Harms et al. 2001;
the "spin test" of Alexander-Bloch et al. 2018) -- the same problem of two
smooth maps sharing one domain. The whole correlation curve falls out of
the same FFT that computes the cross-correlation; the p-value is taken
over ``DEFAULT_N_ROTATIONS`` rotations drawn without replacement (a number
decoupled from the bins), counting the unrotated one, p = (1 + b) / (1 +
n_rot) (Hope 1968; Phipson & Smyth 2010), one-sided. The null assumes
stationarity around the ring; shared geometry (interpolated bins, folds)
violates it, and spatially constrained nulls can exceed their nominal
level (Markello & Misic 2021): its false-positive rate on synthetic
independent profiles is measured by ``validate_ring_correlation_null.py``.
Neighbouring rotations are correlated over a patch width, so the number of
effectively independent rotations (perimeter / mean patch length) is
reported beside p.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.mps_occupancy import (
    ClusterEllipse,
    OccupancyResult,
    discretize_perimeter,
)

# Resolution of the shared profile (gaps-n-theta-3600): one bin is 1/3600 of
# the perimeter (0.1 degree in the angle frame), 1-11 nm of arc on
# perimeters of 5-40 um -- below the lateral localization precision of any
# SMLM dataset and far below a patch: with the published figures (4.08
# clusters/um, ~20 % occupancy, effective cluster radius ~25 nm; Gazal et
# al. 2026) a patch covers tens of nm of arc, i.e. 7 bins at 25 um. A
# software default, editable 360-7200.
DEFAULT_N_THETA = 3600

# Perimeter samples per bin (gaps-min-samples-per-bin-8). The samples are
# equally spaced along the contour, not random, so a bin is empty only
# where the contour does not reach it; what this sets is the quantisation of
# per-bin coverage, in steps of 1/m (12.5 % at m = 8). A software default.
DEFAULT_MIN_SAMPLES_PER_BIN = 8

# The rotation null (gaps-rotation-null-pvalue): rotations drawn without
# replacement from the non-zero shifts, decoupled from the number of bins
# (as in the arc null of the column analysis), with a fixed seed. Every
# shift is used when there are no more than this many.
DEFAULT_N_ROTATIONS = 1999
DEFAULT_ROTATION_SEED = 0

# The frames the profile can be registered in, and what to do with bins no
# perimeter sample reaches (gaps-interpolated-empty-bins).
FRAMES: Tuple[str, ...] = ("arc", "angle")
DEFAULT_FRAME = "arc"
EMPTY_BINS: Tuple[str, ...] = ("interpolate", "mask")
DEFAULT_EMPTY_BINS = "interpolate"
# The fewest bins two masked profiles must share at a rotation for its
# correlation to be computed.
MIN_SHARED_BINS = 8
# A stretch interpolated in BOTH profiles of a pair is shared structure the
# rotation null cannot remove (two linear ramps over the same bins correlate
# perfectly there): on synthetic independent profiles 10 % of such bins
# raised the false-positive rate at a nominal 5 % to about 45 %
# (validate_ring_correlation_null.py). A pair whose profiles share at least
# this fraction of interpolated bins is warned about.
SHARED_INTERPOLATED_WARN = 0.02


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

    theta_deg: NDArray[np.float64]          # (n_theta,) bin centres, 0..360 (of the common curve in the arc frame)
    coverage: NDArray[np.float64]           # (n_theta,) in [0, 1]
    n_samples: NDArray[np.int64]            # perimeter samples per bin
    center: NDArray[np.float64]             # (2,) shared angular origin
    n_interpolated_bins: int                # bins no sample reached (interpolated, or masked)
    perimeter_nm: float
    spacing_nm: float
    occupied_mask: NDArray[np.bool_]        # the underlying arc-length mask
    warnings: List[str] = field(default_factory=list)
    # SCI-6: the frame ("arc" or "angle"), how empty bins were handled
    # ("interpolate" or "mask"; ``measured`` is False where a bin was not
    # reached), and the fraction of perimeter samples whose polar angle
    # runs backwards along the contour (0 for a contour star-shaped about
    # the centre; the angle frame's diagnostic).
    frame: str = "angle"
    empty_bins: str = "interpolate"
    measured: NDArray[np.bool_] = field(default_factory=lambda: np.array([], dtype=bool))
    nonmonotonic_fraction: Optional[float] = None

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
    exclude_labels: Optional[Set[int]] = None,
    *,
    reference: Optional["ArcReference"] = None,
    empty_bins: str = DEFAULT_EMPTY_BINS,
) -> CoverageProfile:
    """
    Re-express one segment's perimeter occupancy as a function of position
    along a curve shared with the other segments of the same axon (arc
    length, with ``reference``) or of angle about a shared centre (without).

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
    min_samples_per_bin : the perimeter is re-sampled, when the stored
        occupancy is coarser, to this many equally spaced samples per bin on
        average, so that per-bin coverage is resolved in steps of
        1 / min_samples_per_bin. A bin is then empty only where the contour
        does not reach it.
    exclude_labels : cluster labels to leave out of the profile, on top of
        the curation the occupancy already applied. Forces the mask to be
        recomputed from the remaining ellipses. Used to ask what a
        segment's pattern looks like once the clusters it shares with its
        neighbour are removed.
    reference : the common curve of the arc-length frame (``arc_reference``);
        None registers by angle about ``center`` (exploratory).
    empty_bins : "interpolate" (default) fills a bin no sample reaches from
        its neighbours; "mask" leaves it out of the correlation
        (``CoverageProfile.measured``). Either way their number is reported.

    Returns
    -------
    CoverageProfile
    """
    if empty_bins not in EMPTY_BINS:
        raise ValueError(f"empty_bins must be one of {EMPTY_BINS}, got {empty_bins!r}")
    if n_theta < 8:
        raise ValueError(f"n_theta must be at least 8, got {n_theta}")

    ctr = np.asarray(center, dtype=float).ravel()
    if ctr.size != 2:
        raise ValueError(f"center must be (2,), got shape {ctr.shape}")

    warnings_: List[str] = []
    points = np.asarray(occupancy.perimeter_points, dtype=float)
    mask = np.asarray(occupancy.occupied_mask, dtype=bool)
    spacing = float(occupancy.point_spacing_nm)

    ellipses = occupancy.ellipses
    if exclude_labels:
        ellipses = [e for e in ellipses if e.label not in exclude_labels]
        if not ellipses:
            raise ValueError(
                "Every cluster was excluded; no coverage profile remains.")
        warnings_.append(
            f"{len(occupancy.ellipses) - len(ellipses)} of "
            f"{len(occupancy.ellipses)} clusters excluded from the profile."
        )

    # --- re-sample if the stored mask is too coarse for this grid --------
    n_needed = int(n_theta) * int(max(min_samples_per_bin, 1))
    if points.shape[0] < n_needed or exclude_labels:
        n_resample = max(points.shape[0], n_needed)
        points, total_len, spacing = discretize_perimeter(contour, n_resample)
        mask = _occupied_by_ellipses(
            points, ellipses, occupancy.mahalanobis_threshold)
        if not exclude_labels:
            warnings_.append(
                f"Perimeter re-sampled from {occupancy.n_points:,} to "
                f"{len(points):,} points ({min_samples_per_bin} equally spaced "
                f"samples per bin on average over {n_theta:,} bins)."
            )
    else:
        total_len = float(occupancy.perimeter_length_nm)

    # --- the angle diagnostic (always: it describes the contour) ---------
    d = points - ctr
    theta = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * np.pi)
    nonmono = _nonmonotonic_fraction(theta)

    # --- bin by position along the common curve, or by angle ------------
    bin_width = 2.0 * np.pi / n_theta
    if reference is not None:
        s_nm = reference.project(points)
        u = np.mod(s_nm / float(reference.length_nm), 1.0)
        idx = np.minimum((u * n_theta).astype(np.int64), n_theta - 1)
    else:
        idx = np.minimum((theta / bin_width).astype(np.int64), n_theta - 1)

    counts = np.bincount(idx, minlength=n_theta).astype(np.int64)
    occupied_sum = np.bincount(
        idx, weights=mask.astype(float), minlength=n_theta)

    coverage = np.full(n_theta, np.nan, dtype=float)
    filled = counts > 0
    coverage[filled] = occupied_sum[filled] / counts[filled]

    # A contour that doubles back can still leave a bin unvisited. Fill
    # those linearly from the neighbouring bins, on the circle (or mask
    # them), and say how many: a hole carried into the correlation as zero
    # would read as a gap that was never measured.
    n_interp = int(np.count_nonzero(~filled))
    where = "positions along the common curve" if reference is not None else "angles"
    if n_interp:
        if not np.any(filled):
            raise ValueError(
                "No perimeter sample fell in any bin; the contour and "
                "the frame are inconsistent.")
        pos = np.flatnonzero(filled).astype(float)
        val = coverage[filled]
        # Wrap one period either side so interpolation across 0/360 works.
        pos_ext = np.concatenate([pos - n_theta, pos, pos + n_theta])
        val_ext = np.concatenate([val, val, val])
        coverage[~filled] = np.interp(
            np.flatnonzero(~filled).astype(float), pos_ext, val_ext)
        warnings_.append(
            f"{n_interp} of {n_theta} bins held no perimeter sample "
            f"(the contour does not reach those {where}) and were "
            + ("interpolated from their neighbours." if empty_bins == "interpolate"
               else "masked out of the correlation.")
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
        frame="arc" if reference is not None else "angle",
        empty_bins=empty_bins,
        measured=np.asarray(filled, dtype=bool),
        nonmonotonic_fraction=nonmono,
    )


def _nonmonotonic_fraction(theta: NDArray[np.float64]) -> Optional[float]:
    """The share of consecutive perimeter samples (contour order) whose polar
    angle steps against the contour's overall sense of rotation: 0 for a
    contour star-shaped about the centre."""
    t = np.asarray(theta, dtype=float).ravel()
    if t.size < 3:
        return None
    step = np.angle(np.exp(1j * (np.roll(t, -1) - t)))
    total = float(np.sum(step))
    if total == 0.0:
        return None
    against = step * np.sign(total) < 0.0
    return float(np.count_nonzero(against)) / float(t.size)


@dataclass
class ArcReference:
    """The common membrane curve of the arc-length frame: every segment's
    perimeter is projected on it and binned by its arc s in [0, L)."""

    path: Any                   # tools.mps_matching.SmoothPath
    length_nm: float
    outward_sign: float
    source: str                 # what the curve is

    def project(self, points_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        from tools.mps_unroll import project_on_path
        s, _r = project_on_path(self.path, np.asarray(points_nm, dtype=float),
                                outward_sign=self.outward_sign)
        return np.asarray(s, dtype=float)


def arc_reference(ms: Any) -> ArcReference:
    """The pooled centroid membrane of the segments of ``ms`` (every analysed
    segment's kept cluster centres, each segment's own contour a start of the
    pooled tour): the curve of the column analysis
    (``tools.mps_unroll.fit_centroid_membrane``). ValueError when it cannot be
    fitted (too few clusters, a singular fit)."""
    from tools.mps_unroll import fit_centroid_membrane
    sets: List[NDArray[np.float64]] = []
    contours: List[Optional[NDArray[np.float64]]] = []
    for an in getattr(ms, "analyses", []):
        if an is None or np.asarray(an.centroids).size == 0:
            continue
        sets.append(np.asarray(an.centroids, dtype=float).reshape(-1, 2))
        c = None if an.perimeter is None else np.asarray(an.perimeter.contour, dtype=float)
        contours.append(c if c is not None and c.ndim == 2 and c.shape[0] >= 3 and np.isfinite(c).all()
                        else None)
    if not sets:
        raise ValueError("no segment has cluster centres")
    membrane = fit_centroid_membrane(sets, ring_contours=contours)
    pts = np.asarray(membrane.path.points_nm, dtype=float)
    area = 0.5 * float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    if area == 0.0:
        raise ValueError("the common curve encloses no area")
    return ArcReference(path=membrane.path, length_nm=float(membrane.length_nm),
                        outward_sign=-float(np.sign(area)),
                        source="pooled centroid membrane of every segment")


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
    p_rotation: float                       # one-sided; see ``profile_correlation``
    n_rotations: int                        # rotations the p-value was taken over
    warnings: List[str] = field(default_factory=list)
    n_bins: int = 0
    masked: bool = False

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
    *,
    n_rotations: int = DEFAULT_N_ROTATIONS,
    seed: int = DEFAULT_ROTATION_SEED,
    measured_a: Optional[NDArray[np.bool_]] = None,
    measured_b: Optional[NDArray[np.bool_]] = None,
) -> ProfileCorrelation:
    """
    Correlate two coverage profiles at every rotation, and test the
    as-imaged alignment against that rotation null.

    Parameters
    ----------
    profile_a, profile_b : coverage on the SAME grid, in the same frame
        (``CoverageProfile.coverage``).
    n_rotations : rotations of the null, drawn without replacement from the
        n - 1 non-zero shifts with ``seed``; every shift when n - 1 <= it.
    measured_a, measured_b : optional masks of the bins that were measured
        (``CoverageProfile.measured`` with ``empty_bins = "mask"``); the
        correlation at each rotation is then the Pearson correlation over
        the bins both profiles measured there.

    Returns
    -------
    ProfileCorrelation, where

    * ``r_at_zero`` is the Pearson correlation with no rotation applied --
      the quantity the biological question asks about: are the patches of
      consecutive segments at the same angles?
    * ``p_rotation`` is (1 + the number of null rotations whose correlation
      is at least ``r_at_zero``) / (1 + n_rotations): a one-sided p-value
      against the null that the two profiles are unrelated along the
      membrane, counting the unrotated configuration (Phipson & Smyth
      2010). One-sided because the hypothesis of aligned patches predicts
      positive correlation. With every shift used it is the fraction of the
      n rotations, rotation 0 included; either way p >= 1 / (1 + n_rot).
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

    masked = measured_a is not None or measured_b is not None
    if masked:
        ma = (np.ones(n, dtype=bool) if measured_a is None
              else np.asarray(measured_a, dtype=bool).ravel())
        mb = (np.ones(n, dtype=bool) if measured_b is None
              else np.asarray(measured_b, dtype=bool).ravel())
        if ma.size != n or mb.size != n:
            raise ValueError("The masks must share the profiles' grid.")
        if ma.all() and mb.all():
            masked = False
    if masked:
        corr = _masked_circular_correlation(a, b, ma, mb)
        if not np.isfinite(corr[0]):
            return ProfileCorrelation(
                r_at_zero=float("nan"), offsets_deg=offsets, correlation=corr,
                best_offset_deg=float("nan"), max_correlation=float("nan"),
                p_rotation=float("nan"), n_rotations=0,
                warnings=["Too few bins measured in both profiles, or one is constant there: "
                          "no correlation exists."], n_bins=n, masked=True)
        return _with_null(corr, offsets, n, n_rotations, seed, masked=True)

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
    return _with_null(corr, offsets, n, n_rotations, seed, masked=False)


def _with_null(corr: NDArray[np.float64], offsets: NDArray[np.float64], n: int,
               n_rotations: int, seed: int, *, masked: bool) -> ProfileCorrelation:
    """The p-value of r(0) over the rotation null, and the result."""
    warnings_: List[str] = []
    r0 = float(corr[0])
    k = int(np.nanargmax(corr))
    n_rot = int(n_rotations)
    if n_rot <= 0 or n_rot >= n - 1:
        b = int(np.count_nonzero(corr[1:] >= r0))
        n_used = n - 1
    else:
        rng = np.random.default_rng(int(seed))
        shifts = rng.choice(np.arange(1, n), size=n_rot, replace=False)
        b = int(np.count_nonzero(corr[shifts] >= r0))
        n_used = n_rot
    p = float(1 + b) / float(1 + n_used)

    if b == 0:
        warnings_.append(
            f"r(0) = {r0:.3f} exceeds every one of the {n_used:,} rotations of the null, so p "
            f"is at its floor ({1.0 / (1 + n_used):.1e}); read z_vs_null for the size "
            f"of the effect."
        )

    return ProfileCorrelation(
        r_at_zero=r0,
        offsets_deg=offsets,
        correlation=corr,
        best_offset_deg=float(offsets[k]),
        max_correlation=float(corr[k]),
        p_rotation=p,
        n_rotations=n_used,
        warnings=warnings_,
        n_bins=n,
        masked=masked,
    )


def _circ(x: NDArray[np.float64], y: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """sum_i x[i] y[i + k] for every shift k, by FFT."""
    return np.asarray(np.fft.irfft(np.conjugate(np.fft.rfft(x)) * np.fft.rfft(y), n=n), dtype=float)


def _masked_circular_correlation(a: NDArray[np.float64], b: NDArray[np.float64],
                                 ma: NDArray[np.bool_], mb: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Pearson correlation of a[i] and b[i + k] over the bins both measured,
    at every shift k (the same convention as the unmasked FFT)."""
    n = a.size
    fa, fb = ma.astype(float), mb.astype(float)
    A, B = a * fa, b * fb
    nk = np.rint(_circ(fa, fb, n))
    sa = _circ(A, fb, n)
    sb = _circ(fa, B, n)
    sab = _circ(A, B, n)
    saa = _circ(A * A, fb, n)
    sbb = _circ(fa, B * B, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sab - sa * sb / nk
        va = saa - sa * sa / nk
        vb = sbb - sb * sb / nk
        r = cov / np.sqrt(va * vb)
    bad = (nk < MIN_SHARED_BINS) | ~(va > 1e-12) | ~(vb > 1e-12)
    r = np.where(bad, np.nan, r)
    return np.asarray(np.clip(r, -1.0, 1.0), dtype=float)


@dataclass
class RingPairResult:
    """Relationship between the gap/patch patterns of two segments."""

    index_a: int
    index_b: int
    delta_z_nm: float
    boundary_relative_depth: Optional[float]
    boundary_is_true_valley: Optional[bool]
    correlation: ProfileCorrelation
    # SCI-6: the frame, how empty bins were handled, how many each profile
    # had, the effectively independent rotations (perimeter / mean patch
    # length) and the angle frame's non-monotonicity of each contour.
    frame: str = "angle"
    empty_bins: str = DEFAULT_EMPTY_BINS
    n_empty_bins_a: Optional[int] = None
    n_empty_bins_b: Optional[int] = None
    n_effective_rotations: Optional[float] = None
    angle_nonmonotonic_a: Optional[float] = None
    angle_nonmonotonic_b: Optional[float] = None

    def export_dict(self) -> Dict[str, Any]:
        c = self.correlation
        return {
            "segment_a": self.index_a,
            "segment_b": self.index_b,
            "delta_z_nm": round(self.delta_z_nm, 2),
            "boundary_relative_depth": self.boundary_relative_depth,
            "boundary_is_true_valley": self.boundary_is_true_valley,
            "coverage_r_at_zero": round(c.r_at_zero, 4),
            "coverage_p_rotation": round(c.p_rotation, 6),
            "coverage_z_vs_null": (None if c.z_vs_null is None
                                   else round(c.z_vs_null, 3)),
            "coverage_max_r": round(c.max_correlation, 4),
            "coverage_best_offset_deg": round(c.best_offset_deg, 2),
            "n_rotations": c.n_rotations,
            # SCI-6 (after every column of before).
            "correlation_frame": self.frame,
            "n_bins": c.n_bins,
            "empty_bins": self.empty_bins,
            "n_empty_bins_a": self.n_empty_bins_a,
            "n_empty_bins_b": self.n_empty_bins_b,
            "n_effective_rotations": (None if self.n_effective_rotations is None
                                      else round(self.n_effective_rotations, 1)),
            "angle_nonmonotonic_fraction_a": (None if self.angle_nonmonotonic_a is None
                                              else round(self.angle_nonmonotonic_a, 4)),
            "angle_nonmonotonic_fraction_b": (None if self.angle_nonmonotonic_b is None
                                              else round(self.angle_nonmonotonic_b, 4)),
        }


@dataclass
class RingAnalysis:
    """Gap/patch description of every segment of one axon, and the
    correlation between consecutive ones."""

    profiles: List[Optional[CoverageProfile]]
    runs: List[Optional[GapPatchStats]]
    pairs: List[RingPairResult]
    warnings: List[str] = field(default_factory=list)
    # SCI-6: the frame used (the arc frame falls back to the angle with a
    # warning when the common curve cannot be fitted), the frame asked for,
    # empty-bin handling and the null's rotations.
    frame: str = "angle"
    frame_requested: str = "angle"
    empty_bins: str = DEFAULT_EMPTY_BINS
    n_rotations: int = DEFAULT_N_ROTATIONS

    @property
    def n_described(self) -> int:
        return sum(1 for p in self.profiles if p is not None)


def analyze_rings(
    ms: Any,
    n_theta: int = DEFAULT_N_THETA,
    min_samples_per_bin: int = DEFAULT_MIN_SAMPLES_PER_BIN,
    *,
    frame: str = DEFAULT_FRAME,
    empty_bins: str = DEFAULT_EMPTY_BINS,
    n_rotations: int = DEFAULT_N_ROTATIONS,
    seed: int = DEFAULT_ROTATION_SEED,
) -> RingAnalysis:
    """
    Describe the gaps and patches of every segment of one axon and
    correlate consecutive segments.

    Parameters
    ----------
    ms : a ``tools.mps_multisegment.MultiSegmentAnalysis``. Typed loosely
        to keep this module importable without it.
    n_theta, min_samples_per_bin : passed to ``coverage_profile``.
    frame : "arc" (default: arc length along the common membrane curve) or
        "angle" (about the pooled centre; exploratory).
    empty_bins : "interpolate" (default) or "mask".
    n_rotations, seed : the rotation null of ``profile_correlation``.

    Returns
    -------
    RingAnalysis, with one entry per segment in ``ms.segments`` order
    (None where a segment has no perimeter occupancy to describe) and one
    pair entry per consecutive pair that could be compared.
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    warnings_: List[str] = []
    profiles: List[Optional[CoverageProfile]] = []
    runs: List[Optional[GapPatchStats]] = []

    if getattr(ms, "axon_center", None) is None:
        return RingAnalysis(
            [], [], [],
            ["No segment yielded clusters, so there is no shared frame "
             "and no gap/patch profile can be built."],
            frame=frame, frame_requested=frame, empty_bins=empty_bins, n_rotations=int(n_rotations))

    reference: Optional[ArcReference] = None
    used_frame = "angle"
    if frame == "arc":
        try:
            reference = arc_reference(ms)
            used_frame = "arc"
        except ValueError as exc:
            warnings_.append(
                f"The arc-length frame needs a common membrane curve, which could not be fitted "
                f"({exc}): the profiles are registered by angle about the pooled centre instead "
                f"(exploratory: one-to-one only on star-shaped contours).")

    for seg, an in zip(ms.segments, ms.analyses):
        if an is None or an.occupancy is None or an.perimeter is None:
            profiles.append(None)
            runs.append(None)
            warnings_.append(
                f"Segment {seg.index} (z = {seg.center_nm:.0f} nm) has no "
                f"perimeter occupancy, so it has no gap/patch profile.")
            continue
        try:
            prof = coverage_profile(
                an.occupancy, an.perimeter.contour, ms.axon_center,
                n_theta=n_theta, min_samples_per_bin=min_samples_per_bin,
                reference=reference, empty_bins=empty_bins)
            run = gap_patch_runs(
                an.occupancy.occupied_mask, an.occupancy.point_spacing_nm)
        except ValueError as exc:
            profiles.append(None)
            runs.append(None)
            warnings_.append(f"Segment {seg.index}: {exc}")
            continue
        profiles.append(prof)
        runs.append(run)
        warnings_.extend(
            f"Segment {seg.index}: {w}" for w in prof.warnings
            if "re-sampled" not in w)

    pos_by_index = {s.index: k for k, s in enumerate(ms.segments)}
    pairs: List[RingPairResult] = []
    for p in ms.pairs:
        ka, kb = pos_by_index.get(p.index_a), pos_by_index.get(p.index_b)
        if ka is None or kb is None:
            continue
        pa, pb = profiles[ka], profiles[kb]
        if pa is None or pb is None:
            warnings_.append(
                f"Segments {p.index_a}-{p.index_b} could not be compared: "
                f"one of them has no gap/patch profile.")
            continue
        mask_kw: Dict[str, Any] = {}
        if empty_bins == "mask":
            mask_kw = dict(measured_a=pa.measured, measured_b=pb.measured)
        elif pa.measured.size == pb.measured.size and pa.measured.size:
            shared = float(np.mean(~pa.measured & ~pb.measured))
            if shared >= SHARED_INTERPOLATED_WARN:
                warnings_.append(
                    f"Segments {p.index_a}-{p.index_b}: {shared:.0%} of the bins were measured in neither "
                    f"profile and are interpolated in both, a shared stretch the rotation null cannot remove "
                    f"(it raises the false positives): mask the unmeasured bins, and look for clusters off the "
                    f"membrane (inside the axon) that pull the contours and the common curve inward.")
        corr = profile_correlation(pa.coverage, pb.coverage, n_rotations=n_rotations,
                                   seed=seed, **mask_kw)
        warnings_.extend(
            f"Segments {p.index_a}-{p.index_b}: {w}" for w in corr.warnings)
        pairs.append(RingPairResult(
            index_a=p.index_a, index_b=p.index_b,
            delta_z_nm=p.delta_z_nm,
            boundary_relative_depth=p.boundary_relative_depth,
            boundary_is_true_valley=p.boundary_is_true_valley,
            correlation=corr,
            frame=used_frame,
            empty_bins=empty_bins,
            n_empty_bins_a=int(pa.n_interpolated_bins),
            n_empty_bins_b=int(pb.n_interpolated_bins),
            n_effective_rotations=_effective_rotations(runs[ka], runs[kb], n_theta),
            angle_nonmonotonic_a=pa.nonmonotonic_fraction,
            angle_nonmonotonic_b=pb.nonmonotonic_fraction,
        ))

    return RingAnalysis(
        profiles=profiles, runs=runs, pairs=pairs, warnings=warnings_,
        frame=used_frame, frame_requested=frame, empty_bins=empty_bins,
        n_rotations=int(n_rotations))


def _effective_rotations(run_a: Optional[GapPatchStats], run_b: Optional[GapPatchStats],
                         n_bins: int) -> Optional[float]:
    """The effectively independent rotations of the null: the mean perimeter
    over the mean patch length of the two segments (neighbouring rotations
    are correlated over a patch width), at most the number of bins."""
    lengths: List[float] = []
    perims: List[float] = []
    for run in (run_a, run_b):
        if run is None or run.fully_occupied or run.fully_free:
            continue
        lengths.extend(float(v) for v in np.asarray(run.patch_lengths_nm, dtype=float))
        perims.append(float(run.perimeter_nm))
    if not lengths or not perims:
        return None
    mean_patch = float(np.mean(lengths))
    if not (mean_patch > 0):
        return None
    return float(min(float(np.mean(perims)) / mean_patch, float(n_bins)))


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
    Build both segments' coverage profiles on the shared angular grid and
    correlate them (the exploratory angle frame). Convenience wrapper over
    ``coverage_profile`` and ``profile_correlation``.
    """
    prof_a = coverage_profile(
        occupancy_a, contour_a, center,
        n_theta=n_theta, min_samples_per_bin=min_samples_per_bin)
    prof_b = coverage_profile(
        occupancy_b, contour_b, center,
        n_theta=n_theta, min_samples_per_bin=min_samples_per_bin)
    corr = profile_correlation(prof_a.coverage, prof_b.coverage)
    return prof_a, prof_b, corr
