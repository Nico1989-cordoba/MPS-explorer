# -*- coding: utf-8 -*-
"""
Multi-segment analysis: every MPS segment of an axon, and the relationship
between consecutive ones.

Gazal et al. (2026) analyse ONE segment per axon -- the 180 nm axial slab
centred on the main peak of the z distribution. This module runs that same
per-segment analysis on EVERY dominant axial component the Gaussian mixture
finds, and then compares consecutive segments of the same axon.

That comparison is item (e) of the thesis plan's Figure 3:

    "Correlación longitudinal de la organización transversal, entre
     segmentos consecutivos: [...] podremos, por primera vez, determinar
     cuantitativamente si existe alguna 'regla' de organización entre
     segmentos consecutivos del MPS."

Because the plan itself states this has not been measured before, there is
no published metric to reproduce here. This module therefore provides
several well-defined comparison statistics and takes no position on which
one answers the biological question -- that is a decision for the
experimenter and their director, not for the software.

Two independent angular statistics are offered on purpose, because they
fail in different ways:

  * angular nearest-neighbour offset -- bandwidth-free, easy to state, but
    insensitive to a coherent global rotation shared by every cluster.
  * circular cross-correlation -- recovers a global rotation offset and its
    strength, but depends on a smoothing bandwidth.

Agreement between the two is evidence; disagreement is a warning that the
pattern is not what either statistic assumes.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from tools.cluster_quality import describe_roi
from tools.mps_analysis import (
    DEFAULT_EPS_NM,
    DEFAULT_MIN_SAMPLES,
    AxonAnalysis,
    analyze_axon,
)
from tools.mps_periodicity import (
    DEFAULT_SLAB_HALF_WIDTH_NM,
    ValleyResult,
    ZPeriodicityResult,
    density_valleys,
    fit_z_periodicity,
)

# A segment needs at least this many clusters before its angular pattern is
# worth comparing: with one or two points any rotation fits equally well.
MIN_CLUSTERS_FOR_ANGLE = 3


# ============================================================================
# Axial segmentation
# ============================================================================

@dataclass
class AxialSegment:
    """One MPS segment: an axial slab centred on a mixture component."""

    index: int
    center_nm: float
    zmin_nm: float
    zmax_nm: float
    weight: float               # GMM mixture weight of the component
    sigma_nm: float             # component standard deviation
    n_locs: int                 # localizations inside the slab
    overlap_with_previous_nm: float = 0.0
    # Which dominant mixture component this segment came from. Not the same
    # as ``index``, which counts only the segments that survived the
    # min_locs filter: once a component is dropped, the two segments either
    # side of it become "consecutive" without sharing a boundary, and only
    # this field can tell them apart.
    component_index: int = -1

    @property
    def width_nm(self) -> float:
        return self.zmax_nm - self.zmin_nm


def find_axial_segments(
    z_nm: NDArray[np.float64],
    half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
    mode: str = "paper",
    min_locs: int = DEFAULT_MIN_SAMPLES,
    z_result: Optional[ZPeriodicityResult] = None,
    guard_nm: float = 0.0,
) -> Tuple[List[AxialSegment], ZPeriodicityResult, ValleyResult, List[str]]:
    """
    Locate every MPS segment along the axon's axial coordinate.

    Each dominant component of the Gaussian mixture fitted to z becomes one
    segment, in ascending axial order.

    Parameters
    ----------
    z_nm : axial coordinates of the ROI, in nm.
    half_width_nm : half the slab thickness. 90 nm (a 180 nm slab) per the
        paper.
    mode : how to bound each slab.
        - "paper" (default): +/- half_width_nm around each component mean,
          exactly as the paper treats the single segment it analyses.
          Consecutive slabs OVERLAP whenever Delta-Z < 2 * half_width_nm,
          so a localization can contribute to two segments. Measured on 18
          real axons, 13 of 33 consecutive pairs overlap at +/-90 nm.
        - "valley": additionally clip each slab at the minimum of the
          fitted axial density between neighbouring components, so every
          localization belongs to at most one segment. Preferred over
          "partition" for comparing consecutive segments: sharing
          localizations between the two segments being compared inflates
          any similarity measured between them, and the valley is where the
          two rings actually separate.
        - "partition": as "valley" but cutting at the midpoint between
          neighbouring means. Kept for comparison; the midpoint assumes the
          density between two rings is symmetric, which it is not when the
          rings differ in weight or width.
        None is "correct": overlapping slabs share data between the
        segments being compared, while clipped slabs are not the same
        measurement the paper defines (they are narrower than 180 nm
        wherever the periodicity is short). The choice is the
        experimenter's.
    min_locs : segments with fewer localizations than this are dropped.
    z_result : a previously computed fit, to avoid refitting.
    guard_nm : width of a dead zone centred on each boundary, excluded from
        both neighbouring slabs. Ignored in "paper" mode, which has no
        boundary. 0 (default) leaves the slabs touching.

        This is a control, not a better segmentation. Axial localization
        precision in 3D dSTORM (~50-80 nm) is comparable to the slab
        thickness, so one physical ring deposits localizations on both
        sides of a boundary: disjoint slabs share no localization yet still
        share clusters, which alone produces similarity between
        consecutive segments. Re-running with a guard band and watching
        whether that similarity survives is what separates axial
        bleed-through from a real relationship between rings.

    Returns
    -------
    (segments, z_result, valleys, warnings) -- ``valleys`` describes every
    boundary between consecutive dominant components, whether or not this
    mode cuts there, so a caller can always ask how well two segments are
    separated axially.
    """
    z = np.asarray(z_nm, dtype=float).ravel()
    z = z[np.isfinite(z)]
    warnings_: List[str] = []

    if z_result is None:
        z_result = fit_z_periodicity(z)
    warnings_.extend(z_result.warnings)

    means = np.asarray(z_result.means_nm, dtype=float)
    weights = np.asarray(z_result.weights, dtype=float)
    sigmas = np.asarray(z_result.sigmas_nm, dtype=float)

    valleys = density_valleys(z_result)

    if means.size == 0:
        return ([], z_result, valleys,
                warnings_ + ["No dominant axial component found."])

    if mode not in ("paper", "valley", "partition"):
        raise ValueError(
            f"mode must be 'paper', 'valley' or 'partition', got {mode!r}")

    lows = means - half_width_nm
    highs = means + half_width_nm

    if mode != "paper" and means.size > 1:
        if mode == "valley":
            bounds = valleys.positions_nm
            warnings_.extend(valleys.warnings)
        else:
            bounds = valleys.midpoints_nm
        half_guard = max(float(guard_nm), 0.0) / 2.0
        lows[1:] = np.maximum(lows[1:], bounds + half_guard)
        highs[:-1] = np.minimum(highs[:-1], bounds - half_guard)

    segments: List[AxialSegment] = []
    dropped = 0
    for i, (m, lo, hi, w, s) in enumerate(zip(means, lows, highs, weights, sigmas)):
        n = int(np.count_nonzero((z >= lo) & (z <= hi)))
        if n < min_locs:
            dropped += 1
            continue
        overlap = 0.0
        if segments:
            overlap = max(0.0, segments[-1].zmax_nm - float(lo))
        segments.append(AxialSegment(
            index=len(segments), center_nm=float(m),
            zmin_nm=float(lo), zmax_nm=float(hi),
            weight=float(w), sigma_nm=float(s), n_locs=n,
            overlap_with_previous_nm=overlap,
            component_index=i,
        ))

    if dropped:
        blame = (f" The {guard_nm:.0f} nm guard band may be what emptied them."
                 if guard_nm > 0 and mode != "paper" else "")
        warnings_.append(
            f"{dropped} axial component(s) had fewer than {min_locs} "
            f"localizations in their slab and were dropped.{blame}"
        )

    overlapping = [s for s in segments if s.overlap_with_previous_nm > 0]
    if overlapping and mode == "paper":
        worst = max(s.overlap_with_previous_nm for s in overlapping)
        warnings_.append(
            f"{len(overlapping)} of {max(len(segments) - 1, 0)} consecutive "
            f"segment pair(s) overlap axially (up to {worst:.0f} nm), because "
            f"the periodicity is shorter than the {2 * half_width_nm:.0f} nm "
            f"slab. Those pairs share localizations, which inflates any "
            f"similarity measured between them. Use mode='valley' for "
            f"disjoint slabs."
        )

    return segments, z_result, valleys, warnings_


# ============================================================================
# Angular description of a segment
# ============================================================================

def cluster_angles(
    centroids: NDArray[np.float64], center: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Angular position of each cluster about ``center``, in [0, 2*pi)."""
    c = np.asarray(centroids, dtype=float)
    if c.size == 0:
        return np.array([])
    d = c - np.asarray(center, dtype=float).ravel()
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), 2 * np.pi)


def angular_nn_offsets_deg(
    angles_a: NDArray[np.float64], angles_b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Angular distance from each cluster in A to the nearest cluster in B.

    Bandwidth-free and directly interpretable: small values mean the
    clusters of one segment sit close to the angular positions of the next.
    Blind, by construction, to a rigid rotation shared by every cluster --
    that is what the cross-correlation below is for.

    Returns degrees in [0, 180].
    """
    a = np.asarray(angles_a, dtype=float).ravel()
    b = np.asarray(angles_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.array([])
    diff = np.abs(a[:, None] - b[None, :])
    diff = np.minimum(diff, 2 * np.pi - diff)      # wrap to [0, pi]
    return np.degrees(diff.min(axis=1))


def fold_rotation(
    rotation_deg: float, spacing_deg: float
) -> Optional[float]:
    """
    Express a rotation offset as a fraction of the mean angular spacing,
    folded into [0, 0.5].

    A ring of N regularly spaced clusters is invariant under rotation by
    360/N, so a measured rotation offset is only ever determined MODULO the
    mean angular spacing: verified on synthetic rings, true rotations of
    0, 22.5 and 90 degrees on an 8-cluster ring are recovered as 270, 202.5
    and 0 degrees -- all correct modulo 45, all with correlation 1.000.
    Reporting the raw angle would therefore be misleading.

    Folding also into [0, 0.5] removes the remaining mirror ambiguity
    (a rotation of 0.9 spacings is 0.1 spacings the other way), giving a
    single interpretable number:

        0.0  -> consecutive segments are angularly aligned
        0.5  -> consecutive segments are maximally staggered

    Returns None if the spacing is not positive.
    """
    if not np.isfinite(rotation_deg) or not np.isfinite(spacing_deg) or spacing_deg <= 0:
        return None
    frac = (rotation_deg % spacing_deg) / spacing_deg
    return float(min(frac, 1.0 - frac))


def _von_mises_density(
    angles: NDArray[np.float64], grid: NDArray[np.float64], kappa: float
) -> NDArray[np.float64]:
    """Circular kernel density of point angles, evaluated on ``grid``."""
    if angles.size == 0:
        return np.zeros_like(grid)
    # exp(kappa*cos(.)) overflows for the concentrations real data produces:
    # the bandwidth shrinks as clusters get denser, so a 97-cluster segment
    # gives kappa ~ 3.8e3 and exp() returns inf, after which every
    # correlation is NaN. Factor out exp(kappa): since cos <= 1 the shifted
    # exponent is <= 0, so exp() <= 1 and cannot overflow. The dropped
    # exp(kappa) is a positive constant common to the whole curve, and
    # Pearson correlation is invariant to it.
    d = np.exp(kappa * (np.cos(grid[:, None] - angles[None, :]) - 1.0))
    return d.sum(axis=1)


def circular_cross_correlation(
    angles_a: NDArray[np.float64],
    angles_b: NDArray[np.float64],
    bandwidth_deg: Optional[float] = None,
    n_grid: Optional[int] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Cross-correlate the angular patterns of two segments over all rotations.

    Both angular point sets are smoothed with a von Mises kernel and
    compared by Pearson correlation as one is rotated past the other.

    Parameters
    ----------
    bandwidth_deg : kernel width. None derives it from the data as a
        quarter of the mean angular spacing, 360 / N / 4, using the mean
        cluster count of the two segments, so the smoothing scales with
        how densely the perimeter is populated. The result DOES depend on
        this choice; report it alongside any number taken from here.
    n_grid : sampling of the rotation axis. None scales it with cluster
        density (200 samples per mean angular spacing). A fixed grid is a
        trap here: at 720 samples the step is 0.5 deg, and a segment with
        80 clusters has a 4.5 deg spacing whose half-step (2.25 deg) is not
        representable, so a perfectly staggered pair was reported as 0.444
        instead of 0.5 of a spacing.

    Returns
    -------
    (offsets_deg, correlation, bandwidth_deg_used) -- ``correlation[k]`` is
    the Pearson correlation when B is rotated by ``offsets_deg[k]``.
    """
    a = np.asarray(angles_a, dtype=float).ravel()
    b = np.asarray(angles_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([]), float("nan")

    if bandwidth_deg is None:
        n_mean = max((a.size + b.size) / 2.0, 1.0)
        bandwidth_deg = float(360.0 / n_mean / 4.0)
    sigma = np.radians(max(bandwidth_deg, 1e-6))
    kappa = 1.0 / (sigma ** 2)

    if n_grid is None:
        n_max = max(a.size, b.size, 1)
        n_grid = int(min(max(2048, 200 * n_max), 131072))

    grid = np.linspace(0.0, 2 * np.pi, n_grid, endpoint=False)
    da = _von_mises_density(a, grid, kappa)
    db = _von_mises_density(b, grid, kappa)

    # Circular cross-correlation by FFT. A circular shift preserves both the
    # mean and the norm, so centring once and dividing by the two fixed
    # norms gives the exact Pearson correlation at every shift -- and turns
    # an O(n^2) sweep into O(n log n), which is what makes the fine grid
    # above affordable.
    da = da - da.mean()
    db = db - db.mean()
    norm = np.linalg.norm(da) * np.linalg.norm(db)
    if norm <= 0 or not np.isfinite(norm):
        return np.degrees(grid), np.full(n_grid, np.nan), float(bandwidth_deg)

    corr = np.fft.irfft(
        np.conjugate(np.fft.rfft(da)) * np.fft.rfft(db), n=n_grid) / norm

    return np.degrees(grid), corr, float(bandwidth_deg)


# ============================================================================
# Consecutive-segment comparison
# ============================================================================

@dataclass
class SegmentPairComparison:
    """Relationship between two consecutive MPS segments of one axon."""

    index_a: int
    index_b: int
    delta_z_nm: float
    axial_overlap_nm: float

    # How well the axial density separates these two segments. None when a
    # dropped component sits between them, so they are consecutive segments
    # without sharing a single boundary. A depth near 0 means the pair is
    # two halves of one unresolved axial distribution rather than two
    # rings -- read this before reading any similarity below.
    boundary_relative_depth: Optional[float]
    boundary_is_true_valley: Optional[bool]

    n_clusters_a: int
    n_clusters_b: int
    perimeter_um_a: Optional[float]
    perimeter_um_b: Optional[float]
    occupancy_a: Optional[float]
    occupancy_b: Optional[float]
    median_1nn_a: Optional[float]
    median_1nn_b: Optional[float]

    # Angular relationship
    angular_nn_median_deg: Optional[float]
    angular_nn_mean_deg: Optional[float]
    best_rotation_deg: Optional[float]
    max_correlation: Optional[float]
    correlation_at_zero: Optional[float]
    bandwidth_deg: Optional[float]
    mean_angular_spacing_deg: Optional[float]
    # Rotation folded into [0, 0.5] as a fraction of the mean angular
    # spacing: 0 = consecutive rings angularly aligned, 0.5 = maximally
    # staggered. See ``fold_rotation`` for why the raw angle alone is not
    # interpretable.
    rotation_fraction_of_spacing: Optional[float]

    warnings: List[str] = field(default_factory=list)


def _pair_comparison(
    seg_a: AxialSegment, seg_b: AxialSegment,
    an_a: Optional[AxonAnalysis], an_b: Optional[AxonAnalysis],
    center: Optional[NDArray[np.float64]],
    bandwidth_deg: Optional[float],
    valleys: Optional[ValleyResult] = None,
) -> SegmentPairComparison:
    warnings_: List[str] = []

    depth: Optional[float] = None
    is_valley: Optional[bool] = None
    if (valleys is not None
            and seg_b.component_index == seg_a.component_index + 1
            and 0 <= seg_a.component_index < valleys.n_boundaries):
        depth = float(valleys.relative_depth[seg_a.component_index])
        is_valley = bool(valleys.is_true_valley[seg_a.component_index])
    elif seg_b.component_index != seg_a.component_index + 1:
        warnings_.append(
            f"Segments {seg_a.index} and {seg_b.index} are consecutive only "
            f"because component(s) between them were dropped, so they share "
            f"no single axial boundary."
        )

    def g(an, attr):
        return None if an is None else getattr(an, attr)

    n_a = 0 if an_a is None else an_a.n_clusters_kept
    n_b = 0 if an_b is None else an_b.n_clusters_kept

    ang_nn_med = ang_nn_mean = None
    best_rot = max_corr = corr0 = bw_used = spacing = rot_frac = None

    if (center is not None and an_a is not None and an_b is not None
            and n_a >= MIN_CLUSTERS_FOR_ANGLE and n_b >= MIN_CLUSTERS_FOR_ANGLE):
        ang_a = cluster_angles(an_a.centroids, center)
        ang_b = cluster_angles(an_b.centroids, center)

        nn = angular_nn_offsets_deg(ang_a, ang_b)
        if nn.size:
            ang_nn_med = float(np.median(nn))
            ang_nn_mean = float(np.mean(nn))

        offsets, corr, bw_used = circular_cross_correlation(
            ang_a, ang_b, bandwidth_deg=bandwidth_deg)
        if corr.size and np.any(np.isfinite(corr)):
            k = int(np.nanargmax(corr))
            best_rot = float(offsets[k])
            max_corr = float(corr[k])
            corr0 = float(corr[0])
        spacing = float(360.0 / ((n_a + n_b) / 2.0))
        if best_rot is not None:
            rot_frac = fold_rotation(best_rot, spacing)
    else:
        warnings_.append(
            f"Angular comparison skipped: segments have {n_a} and {n_b} "
            f"clusters, fewer than the {MIN_CLUSTERS_FOR_ANGLE} needed for a "
            f"rotation to be meaningful."
        )

    return SegmentPairComparison(
        index_a=seg_a.index, index_b=seg_b.index,
        delta_z_nm=seg_b.center_nm - seg_a.center_nm,
        axial_overlap_nm=seg_b.overlap_with_previous_nm,
        boundary_relative_depth=depth, boundary_is_true_valley=is_valley,
        n_clusters_a=n_a, n_clusters_b=n_b,
        perimeter_um_a=g(an_a, "perimeter_um"), perimeter_um_b=g(an_b, "perimeter_um"),
        occupancy_a=g(an_a, "occupancy_percent"), occupancy_b=g(an_b, "occupancy_percent"),
        median_1nn_a=g(an_a, "median_1nn_nm"), median_1nn_b=g(an_b, "median_1nn_nm"),
        angular_nn_median_deg=ang_nn_med, angular_nn_mean_deg=ang_nn_mean,
        best_rotation_deg=best_rot, max_correlation=max_corr,
        correlation_at_zero=corr0, bandwidth_deg=bw_used,
        mean_angular_spacing_deg=spacing,
        rotation_fraction_of_spacing=rot_frac,
        warnings=warnings_,
    )


@dataclass
class MultiSegmentAnalysis:
    """Every MPS segment of one axon, plus consecutive-segment comparisons."""

    source_name: str
    mode: str
    segments: List[AxialSegment]
    analyses: List[Optional[AxonAnalysis]]
    z_result: ZPeriodicityResult
    axon_center: Optional[NDArray[np.float64]]
    pairs: List[SegmentPairComparison]
    valleys: Optional[ValleyResult] = None
    warnings: List[str] = field(default_factory=list)
    # Which selection, and the guard band the segments were cut with.
    # Both tables carry them: a pairs table accumulated over axons had
    # neither, so two guard bands of one axon were indistinguishable.
    roi: str = ""
    guard_nm: float = 0.0

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_analyzed(self) -> int:
        return sum(1 for a in self.analyses if a is not None)

    def export_rows(self) -> List[Dict[str, Any]]:
        """One row per analysed segment, for CSV export."""
        rows: List[Dict[str, Any]] = []
        for seg, an in zip(self.segments, self.analyses):
            if an is None:
                continue
            row = an.export_dict()
            row.update({
                "guard_nm": self.guard_nm,
                "segment_index": seg.index,
                "segment_center_nm": round(seg.center_nm, 2),
                "segment_weight": round(seg.weight, 4),
                "segment_n_locs": seg.n_locs,
                "segment_overlap_prev_nm": round(seg.overlap_with_previous_nm, 2),
                "segment_mode": self.mode,
                "n_segments_in_axon": self.n_segments,
            })
            rows.append(row)
        return rows

    def pair_rows(self) -> List[Dict[str, Any]]:
        """One row per consecutive segment pair, for CSV export."""
        out: List[Dict[str, Any]] = []
        for p in self.pairs:
            out.append({
                "source": self.source_name,
                "roi": self.roi,
                "segment_mode": self.mode,
                "guard_nm": self.guard_nm,
                "segment_a": p.index_a, "segment_b": p.index_b,
                "delta_z_nm": round(p.delta_z_nm, 2),
                "axial_overlap_nm": round(p.axial_overlap_nm, 2),
                "boundary_relative_depth": p.boundary_relative_depth,
                "boundary_is_true_valley": p.boundary_is_true_valley,
                "n_clusters_a": p.n_clusters_a, "n_clusters_b": p.n_clusters_b,
                "perimeter_um_a": p.perimeter_um_a, "perimeter_um_b": p.perimeter_um_b,
                "occupancy_a": p.occupancy_a, "occupancy_b": p.occupancy_b,
                "median_1nn_a": p.median_1nn_a, "median_1nn_b": p.median_1nn_b,
                "angular_nn_median_deg": p.angular_nn_median_deg,
                "angular_nn_mean_deg": p.angular_nn_mean_deg,
                "best_rotation_deg": p.best_rotation_deg,
                "max_correlation": p.max_correlation,
                "correlation_at_zero": p.correlation_at_zero,
                "bandwidth_deg": p.bandwidth_deg,
                "mean_angular_spacing_deg": p.mean_angular_spacing_deg,
                "rotation_fraction_of_spacing": p.rotation_fraction_of_spacing,
            })
        return out


def analyze_all_segments(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    z_nm: NDArray[np.float64],
    *,
    source_name: str = "",
    half_width_nm: float = DEFAULT_SLAB_HALF_WIDTH_NM,
    mode: str = "paper",
    guard_nm: float = 0.0,
    bandwidth_deg: Optional[float] = None,
    run_randomization: bool = False,
    **analyze_kwargs: Any,
) -> MultiSegmentAnalysis:
    """
    Run the per-segment analysis on every MPS segment of one axon, then
    compare consecutive segments.

    Parameters
    ----------
    x_nm, y_nm, z_nm : the ROI's localizations, in nm, BEFORE any axial
        filtering (the mixture must see the full axial distribution).
    mode : "paper", "valley" or "partition" (see ``find_axial_segments``).
    guard_nm : dead zone at each boundary, as a bleed-through control (see
        ``find_axial_segments``).
    bandwidth_deg : angular smoothing for the cross-correlation; None
        derives it from cluster density.
    run_randomization : off by default here. The randomization control is
        by far the most expensive step and multiplies by the number of
        segments; enable it deliberately.
    **analyze_kwargs : forwarded to ``analyze_axon`` (eps_nm, min_samples,
        dbcv_threshold, mahalanobis_threshold, ...).

    Returns
    -------
    MultiSegmentAnalysis
    """
    x = np.asarray(x_nm, dtype=float).ravel()
    y = np.asarray(y_nm, dtype=float).ravel()
    z = np.asarray(z_nm, dtype=float).ravel()

    segments, z_result, valleys, warnings_ = find_axial_segments(
        z, half_width_nm=half_width_nm, mode=mode, guard_nm=guard_nm,
        min_locs=int(analyze_kwargs.get("min_samples", DEFAULT_MIN_SAMPLES)),
    )

    analyses: List[Optional[AxonAnalysis]] = []
    for seg in segments:
        try:
            an = analyze_axon(
                x, y, z,
                source_name=source_name,
                slab_override=(seg.zmin_nm, seg.zmax_nm),
                slab_half_width_nm=half_width_nm,
                run_randomization=run_randomization,
                **analyze_kwargs,
            )
        except Exception as exc:                          # noqa: BLE001
            warnings_.append(
                f"Segment {seg.index} (z = {seg.center_nm:.0f} nm) failed: {exc}")
            analyses.append(None)
            continue
        # analyze_axon flags a manual slab as a deviation from its automatic
        # choice. Here the slab IS the automatic choice for this segment, so
        # that particular note would be noise; everything else is kept.
        an.warnings = [w for w in an.warnings
                       if not w.startswith("Axial slab set manually")]
        analyses.append(an)

    # A single angular origin shared by every segment, so angles from
    # different segments are directly comparable. Built from the cluster
    # centres actually being compared rather than from all localizations,
    # which would be pulled by noise and by any interior signal.
    good = [a.centroids for a in analyses
            if a is not None and a.centroids.size]
    center = (np.concatenate(good, axis=0).mean(axis=0)
              if good else None)
    if center is None:
        warnings_.append(
            "No segment yielded clusters; angular comparisons unavailable.")

    pairs: List[SegmentPairComparison] = []
    for i in range(len(segments) - 1):
        pairs.append(_pair_comparison(
            segments[i], segments[i + 1],
            analyses[i], analyses[i + 1],
            center, bandwidth_deg, valleys,
        ))

    if len(segments) < 2:
        warnings_.append(
            f"Only {len(segments)} segment(s) found in this axon: no "
            f"consecutive pair to compare."
        )

    return MultiSegmentAnalysis(
        source_name=source_name, mode=mode,
        segments=segments, analyses=analyses, z_result=z_result,
        axon_center=center, pairs=pairs, valleys=valleys,
        warnings=warnings_, guard_nm=float(guard_nm),
        roi=describe_roi(analyze_kwargs.get("roi")),
    )
