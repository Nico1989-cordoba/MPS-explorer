# -*- coding: utf-8 -*-
"""
The unrolled cylinder of one axon and the circular cross pair-correlation
between its rings: milestone H4, module B, of
the research plan 03_plan.md (S3.2 ``unroll`` and
``cross_pcf_circular``; 02_investigacion.md B5 and P9; DECISIONES D-04,
D-11, D-24c, D-27) on the rings ``tools.mps_columns.build_rings``
returns and the matches ``tools.mps_matching.analyze_columns`` computed.

What is here
------------
``project_on_path``: every point of the lateral plane as (s, r) on the
smooth closed curve of ONE ring, the reference: s the curve arc of the
nearest point, r the signed offset, + outward. ``unroll``: every cluster
of every ring (and, optionally, every localization) in that common
coordinate, so that the axon reads as a cylinder cut open along a
generator, with s along the membrane and z' along the axis.
``circular_cross_pcf``: the cross pair-correlation g(delta s) of two
rings on the reference circle, a kernel estimate on a lag grid whose
expectation is 1 under independence (B5). ``cross_pcf_circular``: that
estimate against the arc-shift null with reflection on the same circle
(D-04, D-11), the pointwise envelope, the exchangeable global test of
D-24c through ``tools.mps_matching.leave_one_out_max_deviation``, the
peak of g and the Fourier phase of the dominant harmonic (exploratory).
``analyze_unroll``: the per-axon driver over the adjacent and k+2 pairs
of an ``AxonColumnsResult``.

H5-B (candidate A of H5B_SPEC.md): the 1D arc-matching test on the
reference circle. ``arc_eclipse_test`` matches two rings by their arcs
on the reference curve alone (``tools.mps_matching.match_arcs``: cyclic
arc distance, D-24b semantics) and judges the count against a
ROTATION of ring b on the circle with reflection, U ~ Uniform(0, L)
with NO exclusion, whose B + 1 configurations are exactly exchangeable
under a stationary process on the circle (the argument is in the
function's docstring); ``arc_joint_null`` is the per-axon joint null
(every ring but the reference rotated independently, T = the sum over
adjacent pairs); ``analyze_arc_columns`` the driver on the rings of
``build_rings`` and the settings of an ``AxonColumnsResult`` (tau_0,
seed, include_suspect), which also reports ``n_ambiguous`` per ring
(``ambiguous_projection``: clusters whose projection on the reference
curve is not unique). The pilot of 2026-09-25 motivates it: the 2D
joint test of H3 rejected far above its nominal level in leak-free M1
replicates on concave simulated contours, both null kinds alike,
because a shift along an interpolating curve through sparse, scattered
vertices is not a symmetry of the observed configuration (D-25); on
the 1D coordinate the rotation IS a symmetry of the null.

Why this next to the matching of H3: the matching answers "how many
clusters of ring b sit within tau of one of ring a" and nothing about
WHERE the others sit; the cross pcf is the whole function of the lag,
so an alternation (clusters of b in the gaps of a, M6), a torsion (b is
a shifted by delta) and a column pattern (a peak at 0) are read from
the same curve. The null is D-04's shift-with-reflection taken on the
1D unrolled coordinate: the ring's internal structure is kept, the
shift is on the common reference circle (D-27 records that it is the
unrolled counterpart of D-04's arc shift, not a re-projection of the
H3 null). No Qt, no matplotlib; every function is deterministic given
its seed; units are in the names (nm).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from tools.mps_columns import Ring, RingsResult
from tools.mps_membrane import (
    MEMBRANE_KNOT_SPACING_NM,
    MEMBRANE_LOCALIZATION_ITERATIONS,
    MEMBRANE_MIN_CLUSTERS_PER_KNOT,
    LocalizationMembrane,
    _periodic_bspline_design,
    _project,
    _tabulate_path,
    cluster_limited_knot_spacing_nm,
    localization_membrane_of_rings,
    pooled_centroid_tour,
    radial_scatter_on_membrane,
    self_crossings,
    smooth_polygon_membrane,
)
from tools.mps_matching import (
    ENVELOPE_PERCENTILES,
    JOINT_SPAWN_KEY,
    AxonColumnsResult,
    RingGeometry,
    RingPairMatch,
    SmoothPath,
    _null_summary,
    _phipson_smyth,
    assign_within_tau,
    cyclic_arc_distance_nm,
    leave_one_out_max_deviation,
    match_arcs,
    pooled_membrane_path,
    ring_geometry,
    smooth_closed_path,
)

# H5-D (D-34b, D-35c): ``fit_centroid_membrane`` / ``centroid_membrane_of_rings``
# are the repaired membrane of the pooled cluster centroids (recipe
# ``CENTROID_MEMBRANE_RECIPE``) and ``analyze_arc_columns(reference_curve=
# "centroid_membrane")`` the arc test on it -- the primary test since
# D-35(c), run on the rings the lumen cleaning left (``tools.mps_lumen``).
# The localization membrane is labelled a diagnostic (``ARC_CURVE_ROLES``);
# no existing curve changed.
#
# ``_null_summary`` and ``_phipson_smyth`` are the H3 module's own
# summaries of a count against its null (E*, sd*, zeta with the NaN rule
# on a constant null; the +1 Monte Carlo p-values): the arc test below
# reuses them verbatim rather than restating the formulas, so that a
# zeta or a p-value of the 1D test is the 2D test's arithmetic exactly.

__all__ = [
    "ARC_CURVE_ROLES",
    "ARC_LOCALIZATION_KNOT_SPACING_NM",
    "ARC_MEMBRANE_KNOT_SPACING_NM",
    "ARC_REFERENCE_CURVES",
    "ARC_SPAWN_KEY",
    "CENTROID_MEMBRANE_ITERATIONS",
    "CENTROID_MEMBRANE_KNOT_SPACING_NM",
    "CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT",
    "CENTROID_MEMBRANE_PRIMARY_NOTE",
    "CENTROID_MEMBRANE_RECIPE",
    "CURVATURE_WINDOW_ROWS",
    "KERNEL_IMAGES",
    "LOCALIZATION_MEMBRANE_DIAGNOSTIC_NOTE",
    "PCF_SPAWN_KEY",
    "PROJECTION_WINDOW_ROWS",
    "ArcColumnsResult",
    "ArcJointNull",
    "ArcPairMatch",
    "CentroidMembrane",
    "CrossPcf",
    "UnrollParams",
    "UnrollResult",
    "UnrolledAxon",
    "ambiguous_projection",
    "analyze_arc_columns",
    "analyze_unroll",
    "arc_eclipse_test",
    "arc_joint_null",
    "centroid_membrane_of_rings",
    "circular_cross_pcf",
    "cross_pcf_circular",
    "fit_centroid_membrane",
    "harmonic_phase",
    "lag_grid_nm",
    "project_on_path",
    "rotate_arcs",
    "unroll",
    "wrapped_gaussian_kernel",
]

# Spawn key of the pcf null's seed sequence, next to the two ring
# indices: ``SeedSequence(random_seed, spawn_key=(ring_a, ring_b,
# PCF_SPAWN_KEY))``. ``SeedSequence`` accepts non-negative integers only
# (the H3 module encodes "join" the same way), so the word "pcf" is the
# integer its three ASCII bytes spell, big-endian: distinct from every
# key of ``tools.mps_matching`` (small ring indices, and JOINT_SPAWN_KEY
# = "join"), so the pcf null of a pair never shares draws with its
# eclipse test.
PCF_SPAWN_KEY = int.from_bytes(b"pcf", "big")

# Images of the Gaussian kernel summed when it is wrapped on the circle:
# m = -2..2. The next image is exp(-(2 L / h)^2 / 2), which is 0 in
# double precision for any bandwidth below L / 8; the bandwidths of this
# module are 0.15 x p_bar = 0.15 L / K, i.e. L / 270 for a ring of 40
# clusters.
KERNEL_IMAGES: Tuple[int, ...] = (-2, -1, 0, 1, 2)

# Rows of the dense curve table on each side of the nearest sample
# whose sub-chords ``project_on_path`` projects on exactly. The table
# is sampled every 2 nm (``tools.mps_matching.PATH_STEP_NM``), so 8
# rows are 16 nm of curve on each side; the foot of the perpendicular
# from a point at distance d from a curve of curvature radius R lies
# on a sub-chord adjacent to the nearest sample whenever d < R (the
# distance to the curve grows away from the foot on both sides), and
# 16 nm of margin cover the residual asymmetry of that growth at the
# radii of a membrane (R >= 300 nm) for any d below 100 nm.
PROJECTION_WINDOW_ROWS = 8

# Null replicates held in memory at once when the null pcf is computed:
# 128 x K_a x K_b differences is 10 MB for two rings of 100 clusters.
_BLOCK = 128


# ============================================================================
# Parameters and results
# ============================================================================

@dataclass
class UnrollParams:
    """
    Parameters of the unrolling and of the cross pcf (H4 gets its own
    dataclass; the frozen ``columns_params.yaml`` is not touched -- the
    random seed is NOT here, it is ``ColumnsParams.random_seed``, D-22).

    ``bandwidth_fraction`` sets the kernel's sd as a fraction of the
    pair's mean spacing p_bar (B5: h = 0.15 d, so that neighbouring
    clusters, one spacing apart, are resolved and the coincident pairs
    of a column, a few nm apart, merge into one peak);
    ``bandwidth_nm`` overrides it when given. ``lag_step_nm`` is the
    grid step (the bins of the estimator); ``lag_max_fraction`` limits
    the global test and the peak search to |delta s| <= that fraction
    of p_bar (one spacing: beyond it the pcf of a ring pair repeats the
    intra-ring spacing structure, not the between-ring relation).
    ``n_null`` replicates of the arc-shift null; ``min_shift_fraction``
    excludes the shifts |U| < that fraction of p_bar_b, the D-04 / D-11
    rule of the H3 COUNT test. Its default here is 0 (no exclusion),
    amending the H4 specification's 0.5 (D-27): for the pcf CURVE test
    the exclusion breaks the exchangeability the global test rests on.
    A rigid shift U of ring b translates the whole curve, g_{a,b+U}(D)
    = g_{a,b}(D - U), so the replicates nearest the observed phase are
    the ones most alike to the observed curve; keeping every replicate
    at least m from phase 0 while two replicates may sit arbitrarily
    close to each other makes the leave-one-out studentisation score
    the observed curve against a set more alike among itself than to
    the observed, and p_global runs anti-conservative -- measured on
    independent rings: p <= 0.05 in 0.065 (2800 pairs pooled, Wilson
    [0.056, 0.075]) at K = 40, where m is 1.25 % of L, and 0.178 +/-
    0.031 with K_b = 4 (m = L / 8); without the exclusion 0.0475 (R 400)
    and 0.055 +/- 0.018, the levels the uniform-shift argument gives
    exactly (U ~ Haar on the circle makes the B + 1 configurations
    exchangeable). The parameter stays so that the pre-specified 0.5
    can be passed for comparison with the H3 null; ``CrossPcf.
    min_shift_nm`` records what was applied. ``include_locs``: also
    unroll every localization.
    """

    bandwidth_fraction: float = 0.15
    bandwidth_nm: Optional[float] = None
    lag_step_nm: float = 5.0
    lag_max_fraction: float = 1.0
    n_null: int = 1999
    min_shift_fraction: float = 0.0
    include_locs: bool = True

    def __post_init__(self) -> None:
        """Ranges: fractions and steps positive and finite, the
        bandwidth override positive when given, at least one null
        replicate, the minimum shift fraction non-negative."""
        for name in ("bandwidth_fraction", "lag_step_nm", "lag_max_fraction"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f"UnrollParams.{name} must be positive and finite, got {v}")
            setattr(self, name, v)
        if self.bandwidth_nm is not None:
            bw = float(self.bandwidth_nm)
            if not (math.isfinite(bw) and bw > 0.0):
                raise ValueError(f"UnrollParams.bandwidth_nm must be positive and finite, got {bw}")
            self.bandwidth_nm = bw
        if int(self.n_null) < 1 or int(self.n_null) != self.n_null:
            raise ValueError(f"UnrollParams.n_null must be an integer >= 1, got {self.n_null}")
        self.n_null = int(self.n_null)
        m = float(self.min_shift_fraction)
        if not (math.isfinite(m) and m >= 0.0):
            raise ValueError(f"UnrollParams.min_shift_fraction must be >= 0 and finite, got {m}")
        self.min_shift_fraction = m
        self.include_locs = bool(self.include_locs)


@dataclass
class UnrolledAxon:
    """
    One axon in the coordinate of its reference ring's curve: one row
    per cluster of every ring that has a contour (``RingsResult.rings``
    order, then ``Ring.clusters`` order), and optionally one row per
    localization of EVERY ring (``RingsResult.rings`` order, then
    ``Ring.loc_index`` order).

    ``s_nm`` is the curve arc of the nearest point of the reference
    curve to each cluster centroid, in [0, L); ``r_nm`` the signed
    offset from the curve, + outward (away from the polygon's
    interior, whichever way the tour runs); ``z_nm`` the median z' of
    the cluster's localizations. ``usable`` is the mask that enters the
    pcf (every cluster, or the unsuspected ones with
    ``include_suspect=False``). ``length_nm`` is the reference ring's
    CURVE length (``RingGeometry.length_nm``), the L of the circle.
    """

    source_name: str
    reference_ring: int
    length_nm: float
    ring_index: NDArray[np.int64]
    cluster_index: NDArray[np.int64]
    labels: NDArray[np.int64]
    usable: NDArray[np.bool_]
    s_nm: NDArray[np.float64]
    r_nm: NDArray[np.float64]
    z_nm: NDArray[np.float64]
    loc_ring: Optional[NDArray[np.int64]]
    loc_s_nm: Optional[NDArray[np.float64]]
    loc_r_nm: Optional[NDArray[np.float64]]
    loc_z_nm: Optional[NDArray[np.float64]]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CrossPcf:
    """
    The circular cross pcf of one ring pair on the reference circle,
    its arc-shift null and the summaries.

    ``lags_nm`` is the full grid (-L/2, L/2] with ``lag_step_nm`` (the
    step is L / T with T = round(L / step), so that the grid closes on
    the circle exactly; 0 is always a grid point), ``g_obs`` the
    estimate on it, ``g_null_mean`` / ``env_lo`` / ``env_hi`` the mean
    and the pointwise 2.5 / 97.5 % of the null replicates. The global
    test (``t_max_obs``, ``p_global``) is the leave-one-out maximum
    studentised deviation of D-24c over |lag| <= ``lag_max_nm``,
    p = (b + 1) / (B + 1); the peak (``peak_lag_nm``, ``g_peak``) is the
    argmax of ``g_obs`` on the same window; ``g_at_zero`` the estimate
    at lag 0. ``harmonic`` / ``phase_deg`` / ``phase_lag_nm`` are the
    exploratory phase descriptor (``harmonic_phase``). ``null_shift_nm``
    and ``null_reflect`` are the draws, so that any replicate can be
    rebuilt: s_b' = (sigma s_b + U) mod L with sigma = -1 where
    ``null_reflect`` is True.
    """

    ring_a: int
    ring_b: int
    reference_ring: int
    length_nm: float
    K_a: int
    K_b: int
    p_bar_nm: float
    lags_nm: NDArray[np.float64]
    g_obs: NDArray[np.float64]
    g_null_mean: NDArray[np.float64]
    env_lo: NDArray[np.float64]
    env_hi: NDArray[np.float64]
    lag_max_nm: float
    t_max_obs: float
    p_global: float
    g_at_zero: float
    peak_lag_nm: float
    g_peak: float
    harmonic: int
    phase_deg: float
    phase_lag_nm: float
    bandwidth_nm: float
    lag_step_nm: float
    n_null: int
    random_seed: int
    min_shift_nm: float
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    warnings: List[str] = field(default_factory=list)


@dataclass
class UnrollResult:
    """The unrolled axon and one ``CrossPcf`` per adjacent and per k+2
    pair of the ``AxonColumnsResult`` it was computed from, with the
    parameters and the seed (``cols.random_seed``) that produced them."""

    unrolled: UnrolledAxon
    pcf_adjacent: List[CrossPcf]
    pcf_k2: List[CrossPcf]
    params: UnrollParams
    random_seed: int
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Projection on the reference curve
# ============================================================================

def _signed_area_nm2(contour: NDArray[np.float64]) -> float:
    """Shoelace signed area of a closed polygon: positive when the tour
    runs counter-clockwise, negative when clockwise."""
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def project_on_path(
    path: SmoothPath,
    points_nm: NDArray[np.float64],
    *,
    outward_sign: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    The nearest point of the dense curve table to each of ``points_nm``
    (n, 2): its curve arc ``s_nm`` in [0, L) and the signed offset
    ``r_nm`` of the point from the curve, + outward.

    Parameters
    ----------
    path
        The smooth closed curve of the reference ring
        (``RingGeometry.path``), a dense polyline closed by the wrap.
    points_nm
        (n, 2) lateral coordinates in the frame of the curve.
    outward_sign
        +1 or -1: the sign that makes r positive on the side away from
        the polygon's interior. For a counter-clockwise tour the
        interior is on the LEFT of the tangent, and cross(tangent, p -
        proj) is positive on the left, so outward is -cross: pass
        ``-sign(signed area)`` of the tour (``unroll`` derives it from
        the reference contour; the harness checks it on a
        counter-clockwise and on a clockwise tour of the same ellipse).

    Returns
    -------
    (s_nm, r_nm)
        Two (n,) arrays. s is the cumulative table length at the
        projection (``SmoothPath.cum_nm`` of the sub-chord plus the
        fraction along it), reduced modulo L; r = cross(t, p - proj) x
        outward_sign with t the unit direction of the sub-chord.

    Notes
    -----
    Vectorised: a ``cKDTree`` on the table gives the nearest sample of
    every point, then the point is projected exactly (clipped to the
    segment) on the sub-chords within ``PROJECTION_WINDOW_ROWS`` rows
    of that sample on both sides and the smallest distance wins. The
    foot of the perpendicular lies on a sub-chord adjacent to the
    nearest sample for any point closer to the curve than its
    curvature radius (see the constant), which holds for every cluster
    and localization of a ring (|r| tens of nm, R hundreds). A point ON
    a table row projects with t = 1 on the sub-chord ending there (or
    t = 0 on the one starting there): both give the row's cumulative
    arc, s = ``cum_nm[row]`` to rounding, and r = 0 exactly (a zero
    offset has a zero cross product).
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"project_on_path: expected (n, 2) points, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("project_on_path: non-finite point")
    sign = float(outward_sign)
    if sign not in (1.0, -1.0):
        raise ValueError(f"project_on_path: outward_sign must be +1 or -1, got {outward_sign}")
    a = np.asarray(path.points_nm, dtype=np.float64)
    n_rows = a.shape[0]
    length = float(path.length_nm)
    if n_rows < 3 or not (length > 0.0):
        raise ValueError("project_on_path: the path needs at least three rows and a positive length")
    b = np.roll(a, -1, axis=0)
    ab = b - a                                                  # (N, 2) sub-chords
    edges = np.asarray(path.edges_nm, dtype=np.float64)
    edge2 = np.maximum(edges ** 2, 1e-300)
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    _dist, nearest = cKDTree(a).query(pts, k=1)
    nearest = np.asarray(nearest, dtype=np.int64).reshape(-1)
    w = min(PROJECTION_WINDOW_ROWS, n_rows // 2)
    offsets = np.arange(-w, w + 1, dtype=np.int64)              # sub-chords starting at rows v - w .. v + w
    rows = np.mod(nearest[:, None] + offsets[None, :], n_rows)  # (n, 2w + 1)
    a_c = a[rows]                                               # (n, C, 2)
    ab_c = ab[rows]
    ap = pts[:, None, :] - a_c
    t = np.clip(np.einsum("ncd,ncd->nc", ap, ab_c) / edge2[rows], 0.0, 1.0)
    proj = a_c + t[:, :, None] * ab_c
    d2 = ((pts[:, None, :] - proj) ** 2).sum(axis=2)
    best = np.argmin(d2, axis=1)
    idx = np.arange(pts.shape[0])
    row = rows[idx, best]
    t_best = t[idx, best]
    s = np.mod(cum[row] + t_best * edges[row], length)
    s = np.where(s >= length, s - length, s)
    s = np.where(s < 0.0, 0.0, s)
    tangent = ab[row] / np.sqrt(edge2[row])[:, None]
    offset = pts - proj[idx, best]
    r = (tangent[:, 0] * offset[:, 1] - tangent[:, 1] * offset[:, 0]) * sign
    return np.asarray(s, dtype=np.float64), np.asarray(r, dtype=np.float64)


# ============================================================================
# The circular cross pair-correlation
# ============================================================================

def lag_grid_nm(length_nm: float, lag_step_nm: float) -> Tuple[NDArray[np.float64], float]:
    """
    The lag grid of the estimator on a circle of length L: T =
    round(L / step) lags of L / T each, k (L / T) for k = -(T/2 - 1) ..
    T/2 (T even) or -(T - 1)/2 .. (T - 1)/2 (T odd), i.e. (-L/2, L/2]
    with 0 always a grid point.

    Why T bins of L / T and not bins of exactly ``lag_step_nm``: the
    estimator convolves a histogram of the pair differences with the
    kernel CIRCULARLY, which needs the bins to tile the circle exactly;
    when the step divides L (L = 8000 nm, step 5) the grid is 5 k
    exactly, otherwise the effective step differs from the nominal one
    by less than step / T (the ellipse of the harness: 4.9986 vs 5).
    Returns the lags and the effective step.
    """
    length = float(length_nm)
    step = float(lag_step_nm)
    if not (math.isfinite(length) and length > 0.0 and math.isfinite(step) and step > 0.0):
        raise ValueError(f"lag_grid_nm: length {length_nm} and step {lag_step_nm} must be positive")
    n_lags = max(3, int(round(length / step)))
    step_eff = length / n_lags
    zero_row = (n_lags // 2 - 1) if n_lags % 2 == 0 else (n_lags - 1) // 2
    lags = (np.arange(n_lags, dtype=np.float64) - zero_row) * step_eff
    return lags, step_eff


def wrapped_gaussian_kernel(
    d_nm: NDArray[np.float64], bandwidth_nm: float, length_nm: float,
) -> NDArray[np.float64]:
    """
    The Gaussian kernel of sd ``bandwidth_nm`` wrapped on the circle of
    length ``length_nm``, a density on the circle: k_h(d) = sum over
    the images m of ``KERNEL_IMAGES`` of phi((d + m L) / h) / h, with d
    first reduced to (-L/2, L/2]. It integrates to 1 over one turn (to
    the neglected images, 0 in double precision for h < L / 8), which
    is what makes E[g] = 1 under independence.
    """
    length = float(length_nm)
    h = float(bandwidth_nm)
    d = np.mod(np.asarray(d_nm, dtype=np.float64) + length / 2.0, length) - length / 2.0
    out = np.zeros_like(d)
    for m in KERNEL_IMAGES:
        out += np.exp(-0.5 * ((d + m * length) / h) ** 2)
    return np.asarray(out / (math.sqrt(2.0 * math.pi) * h), dtype=np.float64)


def _kernel_spectrum(
    lags: NDArray[np.float64], step_eff: float, bandwidth_nm: float, length_nm: float,
) -> NDArray[np.complex128]:
    """The FFT of the kernel sampled on the grid, normalised on the grid
    (sum x step = 1, so that the grid average of every estimate is
    exactly 1) and rolled so that its index 0 is lag 0: the multiplier
    of the histogram's spectrum in the circular convolution."""
    kern = wrapped_gaussian_kernel(lags, bandwidth_nm, length_nm)
    kern = kern / (float(kern.sum()) * step_eff)
    zero_row = int(np.flatnonzero(lags == 0.0)[0])
    return np.asarray(np.fft.fft(np.roll(kern, -zero_row)), dtype=np.complex128)


def _binned_pcf(
    d_nm: NDArray[np.float64],
    *,
    k_a: int,
    k_b: int,
    length_nm: float,
    step_eff: float,
    zero_row: int,
    kern_f: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """
    The estimator on the grid for each row of ``d_nm`` (B, K_a K_b), the
    pair differences s_j - s_i of B configurations: each difference is
    binned at its nearest lag (round(d / step) modulo T, which wraps it
    on the circle), the histogram is circularly convolved with the
    kernel by FFT and scaled by L / (K_a K_b). The binning moves every
    pair by at most step / 2 = 2.5 nm against a kernel of sd h >= 20
    nm: the harness measures the difference from the exact kernel sum
    at 0.03 in g units (max over replicates). Returns (B, T).
    """
    n_lags = kern_f.size
    n_conf = d_nm.shape[0]
    idx = np.mod(np.round(d_nm / step_eff).astype(np.int64) + zero_row, n_lags)
    flat = idx + (np.arange(n_conf, dtype=np.int64) * n_lags)[:, None]
    hist = np.bincount(flat.ravel(), minlength=n_conf * n_lags).reshape(n_conf, n_lags).astype(np.float64)
    conv = np.real(np.fft.ifft(np.fft.fft(hist, axis=1) * kern_f[None, :], axis=1))
    return np.asarray(float(length_nm) / (k_a * k_b) * conv, dtype=np.float64)


def _arcs(values: NDArray[np.float64], what: str) -> NDArray[np.float64]:
    """``values`` as a finite 1-D float64 array, or ValueError."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError(f"{what}: non-finite arc position")
    return arr


def circular_cross_pcf(
    s_a_nm: NDArray[np.float64],
    s_b_nm: NDArray[np.float64],
    length_nm: float,
    *,
    bandwidth_nm: float,
    lag_step_nm: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    The circular cross pair-correlation of two point sets on a circle
    of length L (B5): g(delta) = L / (K_a K_b) sum_ij k_h(wrap(s_j -
    s_i - delta)) with k_h the Gaussian kernel of sd ``bandwidth_nm``
    wrapped on the circle, evaluated on the lag grid of ``lag_grid_nm``
    (the delta = 0 bin centred on 0).

    Returns
    -------
    (lags_nm, g)
        The grid (T,) and the estimate (T,).

    Notes
    -----
    Normalisation: for two INDEPENDENT uniform sets E[k_h(wrap(s_j -
    s_i - delta))] = 1 / L for every pair, so E[g(delta)] = 1 at every
    lag; the kernel is normalised on the grid so that the grid average
    of g is exactly 1 for ANY two sets (the histogram has K_a K_b
    entries). g > 1 at a lag means an excess of pairs at that
    separation: a peak at 0 is the column pattern, a dip at 0 with a
    peak at p_bar / 2 the alternation, a peak at delta the torsion.
    Computed by ``_binned_pcf`` (histogram of the differences at the
    nearest lag, circular FFT convolution).
    """
    s_a = _arcs(s_a_nm, "s_a_nm")
    s_b = _arcs(s_b_nm, "s_b_nm")
    if s_a.size == 0 or s_b.size == 0:
        raise ValueError("circular_cross_pcf: both point sets must be non-empty")
    h = float(bandwidth_nm)
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError(f"circular_cross_pcf: bandwidth_nm must be positive, got {bandwidth_nm}")
    lags, step_eff = lag_grid_nm(length_nm, lag_step_nm)
    zero_row = int(np.flatnonzero(lags == 0.0)[0])
    kern_f = _kernel_spectrum(lags, step_eff, h, float(length_nm))
    d = (s_b[None, :] - s_a[:, None]).reshape(1, -1)
    g = _binned_pcf(d, k_a=s_a.size, k_b=s_b.size, length_nm=float(length_nm), step_eff=step_eff,
                    zero_row=zero_row, kern_f=kern_f)[0]
    return lags, g


def harmonic_phase(
    s_a_nm: NDArray[np.float64],
    s_b_nm: NDArray[np.float64],
    length_nm: float,
    harmonic: int,
) -> Tuple[float, float]:
    """
    The Fourier phase difference of harmonic m between two point sets
    on the circle: F_x(m) = sum_k exp(-2 pi i m s_k / L), dphi = arg(F_a
    conj(F_b)) in (-180, 180] degrees, and the lag it corresponds to,
    dphi / 360 x L / m. For s_b = s_a + delta, F_b = F_a exp(-2 pi i m
    delta / L), so F_a conj(F_b) has phase +2 pi m delta / L and the
    lag is exactly +delta (while |m delta| < L / 2): ring b = ring a
    shifted by +delta gives phase_lag = +delta. Exploratory (B5): only
    meaningful when the intra-ring spectrum has a peak at m, which a
    hard-core ring of K clusters has at m ~ K only weakly.
    """
    length = float(length_nm)
    m = int(harmonic)
    if m < 1:
        raise ValueError(f"harmonic_phase: the harmonic must be >= 1, got {harmonic}")
    s_a = _arcs(s_a_nm, "s_a_nm")
    s_b = _arcs(s_b_nm, "s_b_nm")
    fa = np.exp(-2j * np.pi * m * s_a / length).sum()
    fb = np.exp(-2j * np.pi * m * s_b / length).sum()
    dphi = float(np.angle(fa * np.conj(fb)))
    if dphi <= -math.pi:
        dphi += 2.0 * math.pi
    return math.degrees(dphi), dphi / (2.0 * math.pi) * length / m


def cross_pcf_circular(
    s_a_nm: NDArray[np.float64],
    s_b_nm: NDArray[np.float64],
    length_nm: float,
    *,
    p_bar_nm: float,
    params: UnrollParams,
    random_seed: int,
    ring_a: int,
    ring_b: int,
    reference_ring: int,
) -> CrossPcf:
    """
    The cross pcf of one ring pair on the reference circle against the
    arc-shift null with reflection (D-04, D-11 on the unrolled
    coordinate; D-27), with the pointwise envelope, the global test of
    D-24c, the peak and the phase descriptor.

    Parameters
    ----------
    s_a_nm, s_b_nm
        Arcs of the USABLE clusters of the two rings on the reference
        curve, in [0, L).
    length_nm
        L, the reference curve's length.
    p_bar_nm
        The pair's mean spacing on the reference curve, (L / K_a + L /
        K_b) / 2: it sets the bandwidth (h = ``bandwidth_fraction`` x
        p_bar unless ``bandwidth_nm`` is given), the window of the
        global test and of the peak (``lag_max_fraction`` x p_bar) and
        the dominant harmonic (m* = max(1, round(L / p_bar))).
    params, random_seed, ring_a, ring_b, reference_ring
        The parameters; the seed and the two ring indices seed the null
        (``SeedSequence(random_seed, spawn_key=(ring_a, ring_b,
        PCF_SPAWN_KEY))``); the reference ring is recorded.

    Notes
    -----
    Null replicate r: s_b' = (sigma_r s_b + U_r) mod L with sigma_r
    -1 (reflection) or +1 with probability 1/2 each and U_r ~ Uniform(m,
    L - m), m = ``min_shift_fraction`` x L / K_b (0 by default: U ~
    Uniform(0, L); U ~ Uniform(0, L) with a warning when 2 m >= L); the
    shifts are drawn first, then the reflections, as
    ``tools.mps_matching._draw_shifts`` does. Keeping ring b's internal
    configuration and moving it rigidly along the common circle is B5's
    "random rotation of rings" on the 1D coordinate: it preserves the
    intra-ring spacing structure (the hard core, the spectrum) and
    destroys only the between-ring relation, the same logic as D-04's
    arc shift, taken on the reference circle instead of on ring b's
    own curve (D-27). The reflection is a valid move because reflecting
    BOTH rings mirrors g (g_{-a,-b}(delta) = g_{a,b}(-delta)) and the
    null statistic (a maximum over a symmetric window) is invariant
    under that mirror; reflecting b alone does not leave g symmetric,
    and nothing here assumes it does.

    Global test (D-24c): the observed curve and the B null curves over
    |lag| <= lag_max are studentised leave-one-out by
    ``tools.mps_matching.leave_one_out_max_deviation`` (the public
    alias of the H3 helper, so the test is the eclipse curve's
    verbatim), t = max over the window, p = (#{t_null >= t_obs} + 1) /
    (B + 1). The B + 1 curves are exchangeable under independence of
    the two rings ONLY when the shifts are Haar on the circle, i.e.
    with m = 0: with the D-11 exclusion the observed phase is special
    (every replicate is >= m from it, no two replicates need be) and
    the level is inflated, the more so the larger m / L (measured
    levels in ``UnrollParams``). Needs B >= 2; otherwise p_global = 1
    with a warning. The envelope is ``np.percentile`` at
    ``ENVELOPE_PERCENTILES`` over the full grid.
    """
    s_a = _arcs(s_a_nm, "s_a_nm")
    s_b = _arcs(s_b_nm, "s_b_nm")
    length = float(length_nm)
    if s_a.size == 0 or s_b.size == 0:
        raise ValueError("cross_pcf_circular: both rings need at least one usable cluster")
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"cross_pcf_circular: length_nm must be positive, got {length_nm}")
    p_bar = float(p_bar_nm)
    if not (math.isfinite(p_bar) and p_bar > 0.0):
        raise ValueError(f"cross_pcf_circular: p_bar_nm must be positive, got {p_bar_nm}")
    if int(ring_a) < 0 or int(ring_b) < 0:
        raise ValueError(f"cross_pcf_circular: ring indices must be non-negative to seed the null, got "
                         f"({ring_a}, {ring_b})")
    warnings_: List[str] = []
    k_a, k_b = int(s_a.size), int(s_b.size)
    n_rep = int(params.n_null)
    h = float(params.bandwidth_nm) if params.bandwidth_nm is not None else float(params.bandwidth_fraction) * p_bar
    lag_max = float(params.lag_max_fraction) * p_bar
    lags, step_eff = lag_grid_nm(length, params.lag_step_nm)
    n_lags = lags.size
    zero_row = int(np.flatnonzero(lags == 0.0)[0])
    kern_f = _kernel_spectrum(lags, step_eff, h, length)
    g_obs = _binned_pcf((s_b[None, :] - s_a[:, None]).reshape(1, -1), k_a=k_a, k_b=k_b, length_nm=length,
                        step_eff=step_eff, zero_row=zero_row, kern_f=kern_f)[0]
    # The null's draws: shifts first, then reflections (the order of _draw_shifts).
    m = float(params.min_shift_fraction) * (length / k_b)
    lo, hi = m, length - m
    if 2.0 * m >= length:
        warnings_.append(f"rings {ring_a}-{ring_b}: the minimum shift {m:.1f} nm is at least half the "
                         f"reference length {length:.1f} nm (2 m >= L): U drawn from Uniform(0, L) "
                         "without the exclusion")
        lo, hi, m = 0.0, length, 0.0
    rng = np.random.default_rng(np.random.SeedSequence(int(random_seed),
                                                       spawn_key=(int(ring_a), int(ring_b), PCF_SPAWN_KEY)))
    shifts = np.asarray(rng.uniform(lo, hi, size=n_rep), dtype=np.float64)
    reflect = np.asarray(rng.random(n_rep) < 0.5, dtype=bool)
    null = np.empty((n_rep, n_lags), dtype=np.float64)
    for start in range(0, n_rep, _BLOCK):
        stop = min(n_rep, start + _BLOCK)
        sigma = np.where(reflect[start:stop], -1.0, 1.0)
        s_shift = np.mod(sigma[:, None] * s_b[None, :] + shifts[start:stop, None], length)   # (b, K_b)
        d = (s_shift[:, None, :] - s_a[None, :, None]).reshape(stop - start, -1)              # (b, K_a K_b)
        null[start:stop] = _binned_pcf(d, k_a=k_a, k_b=k_b, length_nm=length, step_eff=step_eff,
                                       zero_row=zero_row, kern_f=kern_f)
    g_null_mean = np.asarray(null.mean(axis=0), dtype=np.float64)
    env_lo, env_hi = np.percentile(null, ENVELOPE_PERCENTILES, axis=0)
    sel = np.abs(lags) <= lag_max
    if not np.any(sel):
        sel = lags == 0.0
        warnings_.append(f"rings {ring_a}-{ring_b}: lag_max {lag_max:.2f} nm is below the grid step "
                         f"{step_eff:.2f} nm: the window is the single lag 0")
    if n_rep >= 2:
        curves = np.concatenate([g_obs[sel][:, None], null[:, sel].T], axis=1)
        t_all = leave_one_out_max_deviation(curves)
        t_max = float(t_all[0])
        p_global = (int(np.count_nonzero(t_all[1:] >= t_max)) + 1) / (n_rep + 1)
    else:
        t_max, p_global = float("nan"), 1.0
        warnings_.append(f"rings {ring_a}-{ring_b}: no global test (t_max_obs NaN, p_global 1): "
                         f"n_null {n_rep} < 2, no sd to studentise with")
    g_win = g_obs[sel]
    i_peak = int(np.argmax(g_win))
    peak_lag = float(lags[sel][i_peak])
    g_peak = float(g_win.max())
    harmonic = max(1, int(round(length / p_bar)))
    phase_deg, phase_lag = harmonic_phase(s_a, s_b, length, harmonic)
    return CrossPcf(
        ring_a=int(ring_a), ring_b=int(ring_b), reference_ring=int(reference_ring), length_nm=length,
        K_a=k_a, K_b=k_b, p_bar_nm=p_bar,
        lags_nm=lags, g_obs=np.asarray(g_obs, dtype=np.float64), g_null_mean=g_null_mean,
        env_lo=np.asarray(env_lo, dtype=np.float64), env_hi=np.asarray(env_hi, dtype=np.float64),
        lag_max_nm=lag_max, t_max_obs=t_max, p_global=float(p_global),
        g_at_zero=float(g_obs[zero_row]), peak_lag_nm=peak_lag, g_peak=g_peak,
        harmonic=harmonic, phase_deg=float(phase_deg), phase_lag_nm=float(phase_lag),
        bandwidth_nm=h, lag_step_nm=float(params.lag_step_nm), n_null=n_rep, random_seed=int(random_seed),
        min_shift_nm=m, null_shift_nm=shifts, null_reflect=reflect, warnings=warnings_,
    )


# ============================================================================
# Unrolling an axon
# ============================================================================

def _empty_unrolled(source_name: str, warnings_: List[str]) -> UnrolledAxon:
    """An ``UnrolledAxon`` with no row (no ring could be unrolled)."""
    empty_f = np.zeros(0, dtype=np.float64)
    empty_i = np.zeros(0, dtype=np.int64)
    return UnrolledAxon(
        source_name=source_name, reference_ring=-1, length_nm=float("nan"),
        ring_index=empty_i, cluster_index=empty_i.copy(), labels=empty_i.copy(),
        usable=np.zeros(0, dtype=bool), s_nm=empty_f, r_nm=empty_f.copy(), z_nm=empty_f.copy(),
        loc_ring=None, loc_s_nm=None, loc_r_nm=None, loc_z_nm=None, warnings=warnings_)


def unroll(
    res: RingsResult,
    *,
    reference: Optional[int] = None,
    include_suspect: bool = True,
    params: UnrollParams = UnrollParams(),
) -> UnrolledAxon:
    """
    Every cluster of every ring of ``res`` (and every localization,
    with ``params.include_locs``) in the (s, r) coordinate of ONE
    ring's smooth curve, the reference (B5: the cylinder cut open).

    Parameters
    ----------
    res
        The rings of one axon (``build_rings``); the coordinates
        ``x_p``, ``y_p``, ``z_p`` are in the axon frame, as the ring
        contours and the cluster centroids are.
    reference
        The ring index whose curve is the circle. None: the ring with
        the most clusters (a tie: the lowest index), because the curve
        through the most vertices is the closest estimate of the
        membrane. A ring without a contour cannot be the reference
        (ValueError).
    include_suspect
        False leaves the clusters with an H2 suspect mark out of
        ``usable`` (they stay in the table).
    params
        ``include_locs`` only.

    Returns
    -------
    UnrolledAxon
        Cluster rows in ``res.rings`` order then ``Ring.clusters``
        order, for the rings that have a contour (a ring without one is
        left out with a warning); localization rows for EVERY ring, in
        ``res.rings`` order then ``Ring.loc_index`` order. ``r`` is +
        outward: the sign comes from the signed area of the reference
        tour (``project_on_path``).

    Notes
    -----
    Every ring is projected on the SAME curve, so the arcs of two rings
    are comparable directly (the H3 null needed a tour-origin offset per
    ring, D-24d; here there is none): the between-ring shift of a
    column is s_b - s_a on the reference. The offset r of a cluster of
    another ring measures how far that ring's membrane sits from the
    reference's at that arc (a taper, a tilt residual), which is why it
    is kept per row.
    """
    warnings_: List[str] = []
    rings: List[Ring] = list(res.rings)
    geoms: Dict[int, RingGeometry] = {}
    for ring in rings:
        try:
            geoms[int(ring.index)] = ring_geometry(ring, include_suspect=include_suspect)
        except ValueError as exc:
            warnings_.append(f"ring {ring.index} left out of the unrolled table: {exc}")
    if not geoms:
        raise ValueError("unroll: no ring has a contour consistent with its clusters; nothing to unroll")
    if reference is None:
        ref = max(geoms, key=lambda k: (geoms[k].n_clusters, -k))
    else:
        ref = int(reference)
        if ref not in geoms:
            raise ValueError(f"unroll: reference ring {reference} has no usable contour (rings with one: "
                             f"{sorted(geoms)})")
    geom_ref = geoms[ref]
    area = _signed_area_nm2(np.asarray(geom_ref.contour_nm, dtype=np.float64))
    if area == 0.0:
        raise ValueError(f"unroll: the reference ring {ref} has a degenerate contour (zero area)")
    outward_sign = -math.copysign(1.0, area)
    path = geom_ref.path
    length = float(geom_ref.length_nm)
    z_all = np.asarray(res.z_p, dtype=np.float64)
    ring_index: List[int] = []
    cluster_index: List[int] = []
    labels: List[int] = []
    usable: List[bool] = []
    centroids: List[NDArray[np.float64]] = []
    z_med: List[float] = []
    for ring in rings:
        geom = geoms.get(int(ring.index))
        if geom is None:
            continue
        for i, cl in enumerate(ring.clusters):
            ring_index.append(int(ring.index))
            cluster_index.append(i)
            labels.append(int(geom.labels[i]))
            usable.append(bool(geom.usable[i]))
            centroids.append(np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2))
            loc = np.asarray(cl.loc_index, dtype=np.intp)
            z_med.append(float(np.median(z_all[loc])) if loc.size else float("nan"))
    cent = np.vstack(centroids) if centroids else np.zeros((0, 2), dtype=np.float64)
    s_nm, r_nm = project_on_path(path, cent, outward_sign=outward_sign)
    loc_ring: Optional[NDArray[np.int64]] = None
    loc_s: Optional[NDArray[np.float64]] = None
    loc_r: Optional[NDArray[np.float64]] = None
    loc_z: Optional[NDArray[np.float64]] = None
    if params.include_locs:
        x_all = np.asarray(res.x_p, dtype=np.float64)
        y_all = np.asarray(res.y_p, dtype=np.float64)
        idx_all = (np.concatenate([np.asarray(r.loc_index, dtype=np.intp) for r in rings])
                   if rings else np.zeros(0, dtype=np.intp))
        loc_ring = (np.concatenate([np.full(np.asarray(r.loc_index).size, int(r.index), dtype=np.int64)
                                    for r in rings]) if rings else np.zeros(0, dtype=np.int64))
        pts = np.column_stack([x_all[idx_all], y_all[idx_all]])
        loc_s, loc_r = project_on_path(path, pts, outward_sign=outward_sign)
        loc_z = np.asarray(z_all[idx_all], dtype=np.float64)
    return UnrolledAxon(
        source_name=str(res.source_name), reference_ring=int(ref), length_nm=length,
        ring_index=np.asarray(ring_index, dtype=np.int64), cluster_index=np.asarray(cluster_index, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64), usable=np.asarray(usable, dtype=bool),
        s_nm=s_nm, r_nm=r_nm, z_nm=np.asarray(z_med, dtype=np.float64),
        loc_ring=loc_ring, loc_s_nm=loc_s, loc_r_nm=loc_r, loc_z_nm=loc_z, warnings=warnings_)


def _pcf_of_match(
    u: UnrolledAxon, match: RingPairMatch, params: UnrollParams, random_seed: int, warnings_: List[str],
) -> Optional[CrossPcf]:
    """The ``CrossPcf`` of one ``RingPairMatch`` from the unrolled table
    (usable clusters of its two rings), or None with a warning when a
    ring is not in the table or has no usable cluster."""
    ri = np.asarray(u.ring_index)
    sel_a = np.asarray(u.usable) & (ri == int(match.ring_a))
    sel_b = np.asarray(u.usable) & (ri == int(match.ring_b))
    k_a, k_b = int(np.count_nonzero(sel_a)), int(np.count_nonzero(sel_b))
    if k_a == 0 or k_b == 0:
        warnings_.append(f"rings {match.ring_a}-{match.ring_b}: no cross pcf ({k_a} and {k_b} usable "
                         "clusters in the unrolled table)")
        return None
    length = float(u.length_nm)
    p_bar = (length / k_a + length / k_b) / 2.0
    s_all = np.asarray(u.s_nm, dtype=np.float64)
    return cross_pcf_circular(s_all[sel_a], s_all[sel_b], length, p_bar_nm=p_bar, params=params,
                              random_seed=random_seed, ring_a=int(match.ring_a), ring_b=int(match.ring_b),
                              reference_ring=int(u.reference_ring))


def analyze_unroll(
    res: RingsResult,
    cols: AxonColumnsResult,
    *,
    params: UnrollParams = UnrollParams(),
    include_suspect: Optional[bool] = None,
    reference: Optional[int] = None,
    exclude_clusters: Optional[Sequence[Tuple[int, int]]] = None,
) -> UnrollResult:
    """
    The H4 module-B analysis of one axon: ``unroll`` on its rings, then
    ``cross_pcf_circular`` for every adjacent pair and every k+2 pair of
    ``cols`` (the persistence control of 02 P7 read on the pcf), all
    with ``params`` and the seed of the columns run (``cols.random_seed``,
    D-22: the seed travels with the pre-specified parameters).

    ``include_suspect`` defaults to ``cols.include_suspect`` so that the
    clusters entering the pcf are the ones that entered the matching;
    usable clusters only enter the pcf. ``exclude_clusters`` marks the
    given (ring index, cluster index in ``Ring.clusters`` order) rows
    unusable as well: the pcf has no leak awareness of its own -- a
    spurious child sits at its parent's arc, so under independent
    rings with leak the peak at lag 0 is significant without any real
    column (measured p_global <= 0.05 in 6 of 6 adjacent pairs of
    synthetic independent rings with ~45 children) -- and this is how
    the caller passes the children flagged by ``tools.mps_leak``
    (D-27; the H4 report computes both curves).

    What the exclusion removes and what it does not (re-review of
    2026-09-24, measured on twelve adjacent pairs of independent rings
    with leak): it removes the CHILDREN only, not the displacement of
    the real clusters. A real cluster whose nearest neighbouring-ring
    cluster sits 20-90 nm away absorbs that neighbour's leak (DBSCAN
    merges the leaked localizations into it: purity ~0.85) and its
    centroid moves toward the neighbour by +6.5 (se 1.8) / +8.8 (3.3) /
    +7.0 (1.6) nm at separations 20-40 / 40-60 / 60-90 nm, which at a
    bandwidth of ~30 nm inflates g(0): with every truth-spurious
    cluster removed g(0) stayed 1.14-1.48 (mean 1.29, 12 of 12 above
    1) and p_global <= 0.05 on 4 of 12 pairs, whereas the same rings'
    TRUE generator arcs give g(0) 0.995 (sd 0.06, 1 of 12 at 5 %) and
    the detected clusters of purity >= 0.99 (the ones far from any
    neighbour) 0.72. Excluding the flagged children (97 % of the
    unilateral ones) left 8 of 12. So the pair of curves with and
    without ``exclude_clusters`` is NOT a leak-free pcf, and p_global
    under leak must not be read as evidence of columns; real columns
    are unaffected in their verdict (g(0) 2.4-2.7, p 0.005-0.02 in
    every configuration). p_bar of a pair is (L / K_a + L / K_b) / 2
    on the reference curve, K the usable counts. With no ring to
    unroll or no pair the result is empty, with a warning. The inputs
    are not modified.
    """
    inc = bool(cols.include_suspect) if include_suspect is None else bool(include_suspect)
    seed = int(cols.random_seed)
    warnings_: List[str] = []
    try:
        unrolled = unroll(res, reference=reference, include_suspect=inc, params=params)
    except ValueError as exc:
        warnings_.append(f"no unrolled table: {exc}")
        return UnrollResult(unrolled=_empty_unrolled(str(res.source_name), [str(exc)]), pcf_adjacent=[],
                            pcf_k2=[], params=params, random_seed=seed, warnings=warnings_)
    warnings_.extend(unrolled.warnings)
    if exclude_clusters:
        wanted = {(int(k), int(i)) for k, i in exclude_clusters}
        rows = np.array([(int(k), int(i)) in wanted for k, i in zip(unrolled.ring_index, unrolled.cluster_index)],
                        dtype=bool)
        n_hit = int(np.count_nonzero(rows))
        if n_hit != len(wanted):
            warnings_.append(f"exclude_clusters: {len(wanted) - n_hit} of {len(wanted)} entries name no cluster of "
                             "the unrolled table")
        unrolled = dataclasses.replace(unrolled, usable=np.asarray(unrolled.usable, dtype=bool) & ~rows)
        warnings_.append(f"{n_hit} cluster(s) excluded from the pcf on request (exclude_clusters)")
    if not cols.adjacent and not cols.k2:
        warnings_.append("no ring pair in the columns result: no cross pcf")
    pcf_adjacent: List[CrossPcf] = []
    pcf_k2: List[CrossPcf] = []
    for match in cols.adjacent:
        p = _pcf_of_match(unrolled, match, params, seed, warnings_)
        if p is not None:
            pcf_adjacent.append(p)
            warnings_.extend(p.warnings)
    for match in cols.k2:
        p = _pcf_of_match(unrolled, match, params, seed, warnings_)
        if p is not None:
            pcf_k2.append(p)
            warnings_.extend(p.warnings)
    return UnrollResult(unrolled=unrolled, pcf_adjacent=pcf_adjacent, pcf_k2=pcf_k2, params=params,
                        random_seed=seed, warnings=warnings_)


# ============================================================================
# The 1D arc-matching test on the reference circle (H5-B, candidate A)
# ============================================================================

# Spawn key of the arc test's seed sequence, next to the two ring
# indices: ``SeedSequence(random_seed, spawn_key=(ring_a, ring_b,
# ARC_SPAWN_KEY))`` for a pair and ``(ARC_SPAWN_KEY, JOINT_SPAWN_KEY)``
# for the joint null. The word "arc" as the integer its three ASCII
# bytes spell, big-endian (``SeedSequence`` takes non-negative integers
# only): distinct from the pcf key ("pcf") and from every key of
# ``tools.mps_matching`` (small ring indices, "join"), so the rotation
# null of a pair never shares draws with its 2D eclipse test or its pcf
# null, although all three draw a uniform shift and a reflection.
ARC_SPAWN_KEY = int.from_bytes(b"arc", "big")

# Rows of the dense curve table on each side of a projection foot over
# which ``ambiguous_projection`` measures the local curvature: the
# tangent turn between the chord ending at the foot and the chord
# starting there, each CURVATURE_WINDOW_ROWS rows long (10 nm at
# ``PATH_STEP_NM``), over the arc between the chords' midpoints. The
# curvature of a membrane curve varies on the scale of its vertex
# spacing (>= the hard core, tens of nm), so 10 nm chords resolve it,
# and the angle of a 10 nm chord on 1000 nm coordinates is exact to
# ~1e-14 rad, far below the 1e-2 rad turn of a 1000 nm radius.
CURVATURE_WINDOW_ROWS = 5

# The curve ``analyze_arc_columns`` projects every ring on (its
# ``reference_curve`` keyword; the circle of the rotation null):
# "interpolating" = the reference ring's own interpolating spline
# (``RingGeometry.path``; H5B_SPEC S1's definition of candidate A, the
# default so that the existing output is unchanged) and
# "pooled_membrane" = the P-spline membrane fitted to the centroids of
# EVERY ring of the axon (``tools.mps_matching.pooled_membrane_path``,
# ``tools.mps_membrane.fit_membrane`` with its 400 nm knots), every ring
# projected on it, the reference ring only staying fixed in the joint
# null. Why the second exists (adversarial and statistics reviews of
# 2026-09-25, reproduced in a research probe):
# the interpolating curve through the reference ring's scattered
# centroids is a function of that ring's own scatter, so a ring
# projected on it is not a stationary process given the reference and
# the rotation null is not exact -- conservative for the reference
# against a projected ring (mean zeta -0.18 at ellipse / 30 nm / K 40,
# -0.25 at Fourier / 65 nm / K 37, R 300), liberal for two projected
# rings that share the curve's distortion (+0.65, t +11.6, p_excess <=
# 0.05 in 0.117 at Fourier / 65 nm / K 36 on a K 40 reference), and
# liberal on concave contours, where the curve bridges the concavities
# and is shorter than the membrane (research simulations without leak).
# On the pooled membrane the same cases are level or conservative. What remains conservative
# is the membrane's estimation error, which every ring pulls towards
# itself (the argument of ``tools.mps_matching.eclipse_test`` for the
# pooled 2D null); what it cannot remove is structure of the true
# membrane below the knot scale shared by the rings, which is the
# contour library's own roughness (``tools.mps_membrane.
# smooth_polygon_membrane``), not the test's.
#
# What the re-review of 2026-09-25 measured (fresh seeds, R 300, n_null
# 199, tau_s 60, every case on the interpolating curve, the pooled
# membrane and the TRUE contour) and what changed because of it. The
# true contour gives |mean zeta| <= 0.114 and rates 0.02-0.05 in all
# 16 cases: the rotation null itself is exact and every bias below is
# the curve estimated from the rings. Interpolating curve at 65 nm of
# scatter: reference against a projected ring -0.20 (K 37, t
# -3.4) / -0.12 (K 22), two projected rings +0.57 (K 33 on a K 37
# reference, t +9.6, p_excess <= 0.05 in 0.097), joint null with the
# reference at an end +0.32 (t +5.7), in the middle -0.28 (t -5.0): the
# level of ``analyze_arc_columns`` (reference = the densest ring, ties
# lowest index) is a MIXTURE over the reference's position -- liberal
# when the densest ring is at an end (one pair is then two projected
# rings), conservative when it is in the middle -- and a real axon's
# arc_p_A is one configuration, not the mixture (the pooled membrane
# does not depend on the choice). H5B_SPEC S2's
# "|mean zeta| <= 0.10 in every case" is therefore NOT met on the
# interpolating curve at 65 nm; ``power_columns`` keeps it as the
# specification's candidate A (``arc_*``), calibrated by the grid's
# null cells and by ``simnull``, not by the nominal level. Pooled
# membrane with the 2D null's 400 nm knots (the fix stage's first
# choice): level at 30 nm (-0.09 to +0.00) but CONSERVATIVE at 65 nm,
# -0.34 (K 37 pair, t -6.0), -0.34 (K 22 pair, t -5.8), -0.09 / -0.15
# (two projected), -0.28 / -0.35 / -0.24 (joint, t -4.0 to -6.3): a
# -0.3 sd bias of arcm_z_A under H0 is a ~30 % loss of power against a
# 1 sd effect, and it inflates the DEFICIT side, p_deficit <= 0.05 in
# 0.060-0.087 against exact rates of ~0.03 (the M6 decision through the
# deficit tail liberal by 2-3x). The mechanism is the self-pull of the
# P-spline: with ~2.3 centroids per 400 nm knot interval the curve
# follows each ring's own scatter (leverage h ~ dof / N ~ 0.3-0.45 at
# K 37 x 3 rings), the projected arcs inherit it, and the bias scales
# with the scatter's variance (x4.7 from 30 to 65 nm). Coarser knots
# remove most of it, and the 1D test can afford them where the 2D
# pooled null could not (``tools.mps_membrane.MEMBRANE_NULL_KNOT_SPACING
# _NM``'s trade-off): the 2D null places a shifted cluster at the
# curve's normal error delta off the membrane (first order), while the
# arc coordinate of a cluster at offset r moves only by ~(r - delta)
# delta' (second order), so structure the curve misses costs the arc
# test almost nothing -- the Fourier contour with EXACT clusters at K 20
# (7th harmonic wavelength 1207 nm) gives +0.054 / +0.062 / +0.066 at
# 400 / 600 / 800 nm knots against +0.059 on the true contour. Measured
# with fresh seeds (root 6007, R 300; a research probe), mean zeta at 400 / 600 / 800 nm / true: K 37
# pair -0.320 / -0.132 / -0.131 / -0.064; K 22 pair -0.360 / -0.126 /
# -0.050 / -0.023; two projected K 33/37 -0.224 / -0.078 / -0.063 /
# -0.064; K 18/22 -0.240 / -0.061 / -0.033 / -0.025; joint K 37
# reference at an end -0.323 / -0.116 / -0.013 / -0.001, in the middle
# -0.322 / -0.109 / -0.073 / +0.002; joint K 22 -0.301 / -0.061 /
# -0.042 / -0.020; sparse reference K 12 vs 37 -0.117 / +0.048 / +0.064
# / +0.124; every 30 nm and exact case within +/- 0.09 at all three
# spacings; p_deficit <= 0.05 at 600 nm 0.013-0.050 (exact ~0.03, z <=
# +1.8) against 0.047-0.077 at 400 nm. In the research grid without
# leak (the SAME simulated axons at both spacings) the 400 nm knots were
# conservative and the 600 nm knots level within the Monte Carlo
# resolution. 600 nm is chosen (``ARC_MEMBRANE
# _KNOT_SPACING_NM``): it removes 60-80 % of the conservative bias
# (the residual -0.06 to -0.13 at 65 nm, t <= 2.2, is within the
# specification's 0.10 + 2 se), it is the scale the radial scatter was
# measured on (``radial_scatter_nm``) and the scale grid v2 smooths the
# library contours to (so the simulated membrane has no structure the
# curve could miss), and it is H5B_SPEC's own number; 800 nm buys
# another ~0.05 at the pair cases but bridges structure at the scale of
# the smoothed contours' concavities, and a curve fitted to a third ring
# only (independent of both tested rings; level, +0.02 at K 37) needs
# three rings and is a near-interpolant of one ring at a typical
# spacing. The keyword
# ``membrane_knot_spacing_nm`` of ``analyze_arc_columns`` lets a
# harness or the verifier measure another spacing.
#
# "localization_membrane" (H5-C, H5C_SPEC S2; D-29c, the candidate for
# the primary test): the P-spline membrane fitted to the LOCALIZATIONS
# of every ring that entered (``tools.mps_membrane.
# localization_membrane_of_rings``: the pooled centroid P-spline as the
# initial curve, two robust weighted fits with knots every
# ``ARC_LOCALIZATION_KNOT_SPACING_NM``), every ring projected on it. Why a
# third curve: what stays conservative on the pooled centroid membrane
# is the curve's estimation error, which each ring's own scatter pulls
# towards itself (leverage ~dof / N with N = the centroids), and in the
# research simulations a residual bias remains (D-28f/m) while the
# TRUE membrane gives an exact test in every synthetic case. The
# localizations are ~100x more points than the centroids and the
# undetected clusters are in them too, so the curve is closer to the
# membrane and each cluster's leverage on it is its own few tens of
# localizations among the many of a knot interval. The leave-ring-out variant of the SAME curve (each pair's
# rings projected on the membrane of the OTHER rings' localizations) is
# reported per pair only (``ArcColumnsResult.loo_adjacent``): the joint
# null rotates every ring but the reference on ONE circle, so the joint
# statistic needs one shared curve, and a curve fitted without a pair's
# rings is a different circle for every pair.
#
# "centroid_membrane" (H5-D; D-33a, D-34b, D-35c: the PRIMARY curve since
# D-35) = the repaired membrane of the pooled cluster CENTROIDS of the
# cleaned rings, recipe ``CENTROID_MEMBRANE_RECIPE`` (``fit_centroid_membrane``;
# frozen in the research's pre-stated final candidate before
# its final runs, ported from r2lib.cmx_fit / weighted_fit / loco_arcs). Why
# the centroids and not the localizations (the user, D-33a): in (d)STORM one
# fluorophore is localized many times, so a localization-weighted curve
# weighs a cluster by its blinking. Why the old centroid membrane
# ("pooled_membrane") had to be repaired (D-34b): it parameterised each
# centroid by the chord of a ONE-start tour that zigzags between the rings
# (longer than the membrane) and can cross the lumen, which
# pulled close cross-ring pairs apart (+18.8 nm on a circle: conservative)
# and made its length guard fail at high density. The
# repair: the all-starts pooled tour, a 400 nm initial curve, each centroid's
# parameter = its projection on the current curve, two fits, the guard against
# the initial curve, and every centroid's arc read on the curve refitted
# WITHOUT it (leave-one-cluster-out, an exact rank-1 downdate), which removes
# the self-pull that made a 400 nm centroid fit conservative. Its level
# was measured on simulated axons without leak (D-34b: a small paired
# bias against the true curve, no fit failure); the localization
# membrane is conservative on the same axons, so it is
# a DIAGNOSTIC since D-35c (``ARC_CURVE_ROLES``). Not to be re-tuned on new
# data (D-34b: the number of iterations is a sensitive dial).
ARC_REFERENCE_CURVES: Tuple[str, ...] = ("interpolating", "pooled_membrane", "localization_membrane", "centroid_membrane")
# The frozen recipe of the "centroid_membrane" curve (D-34b, D-35c): centroid
# membrane X, knots 400 nm, 2 iterations, Leave-one-cluster-out arcs.
CENTROID_MEMBRANE_RECIPE = "cmX-k400-it2-L"
# Its settings (PRESTATED_final_candidate.json): the P-spline knots of the
# fits, floored at CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT detected clusters
# per knot interval of the initial curve (max(400, 3 L0 / K_pooled));
# the initial curve's knots (the chord-parameter P-spline of the pooled tour)
# and how much its length may differ from the tour's (the tour through
# scattered pooled centroids zigzags and is up to ~2x the membrane: the
# failure (c) of D-32e); the fits' length guard against the INITIAL curve;
# the fits' number, penalty and the minimum points per knot of the check.
CENTROID_MEMBRANE_KNOT_SPACING_NM = 400.0
CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT = 3.0
CENTROID_MEMBRANE_INITIAL_KNOT_SPACING_NM = 400.0
CENTROID_MEMBRANE_INITIAL_MAX_LENGTH_CHANGE = 0.9
CENTROID_MEMBRANE_MAX_LENGTH_CHANGE = 0.5
CENTROID_MEMBRANE_ITERATIONS = 2
CENTROID_MEMBRANE_PENALTY_REL = 0.01
CENTROID_MEMBRANE_MIN_POINTS_PER_KNOT = 1.0
# The leave-one-cluster-out arc: the nearest point of the refitted curve is
# searched within +/- this many knot intervals of the cluster's parameter, on
# a grid of this many parameters.
CENTROID_MEMBRANE_LOCO_WINDOW_KNOTS = 2.0
CENTROID_MEMBRANE_LOCO_GRID = 801
# Rows of the preview the length guard reads before the dense table is built
# (a runaway curve made a table of millions of rows in the research: one
# 4113 s evaluation).
CENTROID_MEMBRANE_PREVIEW_ROWS = 200
# What each curve's result is FOR since D-35(c) (``ArcColumnsResult.curve_role``,
# a label only: no number of any curve changes with it). The localization
# membrane's known bias, measured in D-34b, travels with every result on it.
LOCALIZATION_MEMBRANE_DIAGNOSTIC_NOTE = (
    "diagnostic only (D-35c), known conservative bias: on simulated axons without leak the arc test on the "
    "localization membrane reads below the true curve (D-34b)")
CENTROID_MEMBRANE_PRIMARY_NOTE = (
    f"primary (D-35c): the centroid membrane {CENTROID_MEMBRANE_RECIPE} of the cleaned rings (small paired bias "
    "against the true curve on simulated axons without leak, D-34b); not calibrated for axial leak (H5-E)")
ARC_CURVE_ROLES: Dict[str, str] = {
    "interpolating": "",
    "pooled_membrane": "",
    "localization_membrane": LOCALIZATION_MEMBRANE_DIAGNOSTIC_NOTE,
    "centroid_membrane": CENTROID_MEMBRANE_PRIMARY_NOTE,
}
# Knot spacing of the pooled P-spline membrane the arc test projects on
# (``analyze_arc_columns(reference_curve="pooled_membrane")`` and
# "localization_membrane" alike); the comment above has the
# measurements. Not the 2D null's 400 nm.
ARC_MEMBRANE_KNOT_SPACING_NM = 600.0
# Knot spacing of the LOCALIZATION membrane the arc test projects on
# (``analyze_arc_columns(reference_curve="localization_membrane")`` with
# ``membrane_knot_spacing_nm`` None; review of 2026-09-26, a deviation
# from H5C_SPEC S2's 600 nm, reported). At 600 nm the fitted curve cannot
# follow a narrow deep slot of the membrane (on a slot-shaped contour the
# fit from the TRUE curve shortens the contour and the arc test on it
# is liberal, because the slot's clusters
# are compressed onto the chord across it and their arcs co-vary in
# every ring), while at 400 nm the same fit is level there and on a
# second slot-shaped contour. The
# self-attraction that made the pooled CENTROID membrane conservative at
# 400 nm (-0.22 to -0.36, above) is small here because the curve is
# fitted to every localization of every cluster, detected or not: with
# the pooled-tour initial curve (``tools.mps_membrane.
# pooled_centroid_tour``) the 400 nm membrane stays within 0.07 sd of
# the true curve on every H5-B case (ellipse 30 / 65 nm, Fourier exact /
# 65 nm, two projected rings, joint with the reference at an end / in
# the middle; R 200) and is closer to the true curve than the 600 nm
# fit on slot-shaped contours (probes of 2026-09-26). What 400 nm costs:
# on rings whose RADII differ by 10 % (a steep taper, adjacent rings
# 200 nm apart) the curve zigzags between the rings and the test of
# two projected rings turns conservative (-1.2 sd; -0.12 at a 5 %
# taper, level); a conservative residual, reported.
#
# Re-review of 2026-09-27 (CORRECTS the "within 0.07 sd" above, which
# held only on the contours probed): over a library of contours (each
# contour as the grid-v2 polygon smoothing and as the estimator's own
# 600 nm curve; joint null with the reference at an end, paired against
# the true curve; a research probe) the 400 nm membrane reads within
# +/- 0.10 sd of the true curve on most cases but is liberal on
# contours with a narrow inward notch. Mechanism: the 400 nm curve
# bridges the notch, the clusters of
# its two walls project on the bridge at nearly the same arcs in every
# ring, and the arcs co-locate beyond what a rotation gives; the fit
# started from the TRUE curve does the same, so
# it is the basis, not the initial curve. No single spacing removes it
# without costing elsewhere (same cases, paired against the true curve):
# finer knots (250-300 nm) reduce it but turn conservative on many other
# cases (self-pull), and 600 nm makes it worse. A cluster-level cross-validated
# spacing and a cross-fitted curve (each cluster projected on the curve
# fitted without it) were probed and rejected: the first picks fine
# knots on the ellipse and the slot contours, the second swaps the
# self-pull for a pull towards the other rings' clusters. 400 nm
# stays: the smallest worst case on the harness cases and level on
# most library cases. The
# residual per-axon bias of notch-shaped membranes is a property of the
# estimator that the uncalibrated arc test keeps; the per-axon simulated
# null carries it only where the localizations resolve the notch
# (``power_columns.SIMNULL_MEMBRANE_CONTOUR_KNOT_NM``). Sparse
# two-ring axons: the knots are floored at
# ``tools.mps_membrane.MEMBRANE_MIN_CLUSTERS_PER_KNOT`` detected clusters
# per interval (``analyze_arc_columns`` with the default spacing).
ARC_LOCALIZATION_KNOT_SPACING_NM = 400.0


@dataclass
class ArcPairMatch:
    """
    One pair of rings matched by their ARCS on the reference circle
    within ``tau_nm`` and judged against the rotation null (H5-B,
    candidate A).

    ``i_a``/``j_b`` are cluster indices into the two arc arrays the
    test received (``Ring.clusters`` order when it comes from
    ``analyze_arc_columns``), ``d_nm`` their cyclic arc distances.
    ``K_a``/``K_b`` count the USABLE clusters; ``E_dir = n_matched /
    min(K_a, K_b)``, ``E_star`` and ``sd_star`` the mean and sample sd
    of the null count over min K, ``zeta`` the standardised observed
    count (NaN, with a warning, on a constant null) and the p-values
    Monte Carlo with the +1 of Phipson and Smyth (excess favours
    columns, deficit alternation, two-sided = min(1, 2 min)), all as
    ``RingPairMatch``. The draws are recorded: replicate r is s_b' =
    (sigma_r s_b + U_r) mod L with sigma_r = -1 where ``null_reflect``
    is True (the reflection s -> L - s, then the shift) and U_r =
    ``null_shift_nm[r]``, so ``rotate_arcs`` + ``match_arcs`` rebuild
    every ``null_n_matched[r]``. ``reference_ring`` is the ring whose
    curve is the circle and ``length_nm`` that curve's L; the rings'
    own indices are ``ring_a``/``ring_b``.
    """

    ring_a: int
    ring_b: int
    reference_ring: int
    length_nm: float
    tau_nm: float
    i_a: NDArray[np.intp]
    j_b: NDArray[np.intp]
    d_nm: NDArray[np.float64]
    n_matched: int
    K_a: int
    K_b: int
    E_dir: float
    E_star: float
    sd_star: float
    zeta: float
    p_excess: float
    p_deficit: float
    p_two_sided: float
    null_n_matched: NDArray[np.int64]
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    n_null: int
    random_seed: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class ArcJointNull:
    """
    The rotation null of one axon over its adjacent ring pairs at once,
    on the reference circle (the 1D counterpart of ``tools.mps_matching.
    JointNull``): T = the sum over adjacent pairs of the arc-matched
    count, observed against replicates in which every ring but the
    reference is rotated by its own independent (U, reflect).

    ``pairs`` are the (ring index, ring index) pairs used, consecutive
    by index (a gap breaks the pair, with a warning); ``n_matched_obs``
    their observed counts, ``T_obs`` the sum, ``T_null`` the (B,)
    replicate sums and ``null_n_matched`` the (B, n_pairs) counts behind
    them; ``T_A`` the mean over pairs of (E_dir - E*), E* from THIS
    null; ``z_A`` the standardised ``T_obs`` (NaN on a constant null)
    and the p-values as in ``ArcPairMatch``. ``null_shift_nm`` and
    ``null_reflect`` are (B, n_rings) in ``rings`` order with the
    reference ring's column identically 0 and False: the reference is
    never moved. ``E_dir_obs``, ``E_star`` and ``min_K`` are per pair.
    """

    tau_nm: float
    pairs: List[Tuple[int, int]]
    n_matched_obs: NDArray[np.int64]
    T_obs: int
    T_null: NDArray[np.int64]
    T_A: float
    z_A: float
    p_excess: float
    p_deficit: float
    p_two_sided: float
    n_null: int
    random_seed: int
    rings: List[int]
    reference_ring: int
    null_n_matched: NDArray[np.int64]
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    E_dir_obs: NDArray[np.float64]
    E_star: NDArray[np.float64]
    min_K: NDArray[np.int64]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CentroidMembrane:
    """
    The repaired centroid membrane of one axon (H5-D, ``fit_centroid_membrane``;
    recipe ``CENTROID_MEMBRANE_RECIPE``): ``path`` the fitted curve (a
    ``SmoothPath``, row 0 at parameter 0, ``length_nm`` its closed length),
    ``initial_length_nm`` the initial curve's length the guard compares
    with, ``knot_spacing_nm`` the fits' knots after the floor
    (``n_knots`` of them), ``iterations`` and ``penalty_rel`` the fit's
    settings, ``n_points`` the pooled centroids fitted. ``rings`` are the
    ring indices in the order their centroids were pooled; per ring,
    ``arcs_nm`` are the LEAVE-ONE-CLUSTER-OUT arcs of its clusters
    (``Ring.clusters`` order): each centroid's arc on this curve at the
    parameter of its nearest point on the curve refitted WITHOUT it -- the
    coordinate the arc test matches on -- and ``offsets_nm`` the signed
    offsets (+ outward) of the centroids from this curve. ``tour_order``
    is the pooled tour (indices into the pooled centroids) the initial
    curve followed; ``n_self_crossings`` counts the curve's crossings with
    itself (0 for any valid membrane; warned about otherwise).
    """

    path: SmoothPath
    length_nm: float
    initial_length_nm: float
    knot_spacing_nm: float
    n_knots: int
    iterations: int
    penalty_rel: float
    n_points: int
    rings: List[int]
    arcs_nm: Dict[int, NDArray[np.float64]]
    offsets_nm: Dict[int, NDArray[np.float64]]
    tour_order: NDArray[np.intp]
    n_self_crossings: int
    recipe: str = CENTROID_MEMBRANE_RECIPE
    warnings: List[str] = field(default_factory=list)


@dataclass
class ArcColumnsResult:
    """
    The arc test of one axon (``analyze_arc_columns``): the adjacent
    and k+2 ``ArcPairMatch`` at ``tau_nm`` on the reference ring's
    curve (``reference_ring``, ``length_nm``), the joint null and its
    summary ``T_A``/``z_A``/``p_A`` (``p_A`` two-sided), ``rings`` the
    ring indices that entered (a ring without a contour is left out
    with a warning). ``n_ambiguous`` maps each ring index to the number
    of its usable clusters whose projection on the reference curve is
    not unique (``ambiguous_projection``; 0 for the reference when the
    curve is its own, since its arcs are then exact): reported, never
    removed. It counts the curve's roughness at the ring's positions
    (an interpolating reference curve through centroids scattered 65 nm
    has curvature radii of ~80 nm between its vertices and flags ~11 %
    of clusters that sit ON the membrane; the true membrane, which
    gives an exact test, flags MORE on concave contours because its
    concavities are deeper than the shortcut curve's), so it is neither
    a measure of how far the ring sits from the membrane nor a validity
    gate of the test (reviews of 2026-09-25; ``ambiguous_projection``).
    ``excluded_clusters`` is the list of (ring index, cluster index) the
    caller's ``exclude_clusters`` actually took out of the matching (the
    entries that named a cluster, sorted). ``include_suspect``,
    ``n_null`` and ``random_seed`` are the settings applied;
    ``reference_curve`` is the curve every ring was projected on
    (``ARC_REFERENCE_CURVES``) and ``membrane_knot_spacing_nm`` the knot
    spacing of the pooled membrane under "pooled_membrane" and
    "localization_membrane" (NaN under "interpolating"), and of the
    fits of the centroid membrane (after its floor) under
    "centroid_membrane" (H5-D, whose fit is ``centroid_membrane``).

    H5-C fields (all at their defaults under the other two curves):
    ``membrane`` is the ``LocalizationMembrane`` every ring was projected
    on under "localization_membrane" (its ``path`` is the circle,
    ``length_nm`` its L; the fit's n_used, residual sd and warnings
    are there); ``radial_scatter_nm`` the ``radial_scatter_on_membrane``
    of ALL the clusters (usable or not: the membrane is geometry) of the
    rings that entered, on that curve -- the D-25 quantity the
    simulator's ``radial_offset_sd_nm`` must reproduce; ``loo_adjacent``
    the leave-ring-out diagnostic of each adjacent pair: the pair's two
    rings projected on the membrane fitted to the localizations of the
    OTHER rings when there are at least three (a curve independent of
    both tested rings), else on the pair's own pooled curve (then the
    same test as ``adjacent``, warned about), as an ``ArcPairMatch``
    whose ``length_nm`` is that curve's. It is a per-pair diagnostic
    and not a joint statistic (``ARC_REFERENCE_CURVES``); a pair whose
    leave-out membrane cannot be fitted is left out with a warning, so
    match its entries to ``adjacent`` by (``ring_a``, ``ring_b``).
    """

    source_name: str
    reference_ring: int
    length_nm: float
    tau_nm: float
    rings: List[int]
    adjacent: List[ArcPairMatch]
    k2: List[ArcPairMatch]
    joint: Optional[ArcJointNull]
    T_A: float
    z_A: float
    p_A: float
    n_ambiguous: Dict[int, int]
    excluded_clusters: List[Tuple[int, int]]
    include_suspect: bool
    n_null: int
    random_seed: int
    warnings: List[str] = field(default_factory=list)
    reference_curve: str = "interpolating"
    membrane_knot_spacing_nm: float = float("nan")
    membrane: Optional[LocalizationMembrane] = None
    radial_scatter_nm: float = float("nan")
    loo_adjacent: List[ArcPairMatch] = field(default_factory=list)
    # Review of 2026-09-26: the knots of the localization membrane ``radial_scatter_nm`` is measured about
    # (``tools.mps_membrane.MEMBRANE_KNOT_SPACING_NM`` = 600 nm, the simulator's D-25 quantity, whatever the arc
    # curve's knots) and the scatter of the same clusters about the arc curve itself (the 400 nm curve follows each
    # cluster a little, so it reads ~10-15 % lower; printed for the record). NaN under the other reference curves.
    radial_scatter_knot_spacing_nm: float = float("nan")
    radial_scatter_arc_curve_nm: float = float("nan")
    # H5-D (D-35c): the fitted centroid membrane under "centroid_membrane" (its path is the circle, its
    # leave-one-cluster-out arcs the ones matched; None under the other curves), and what the curve's result is for
    # (``ARC_CURVE_ROLES``: "primary" for the centroid membrane, "diagnostic" with its known conservative bias for the
    # localization membrane, "" for the two H5-B curves). Labels only: no number of any curve depends on them.
    centroid_membrane: Optional[CentroidMembrane] = None
    curve_role: str = ""


def _usable_mask(values: Optional[NDArray[np.bool_]], k: int, what: str) -> NDArray[np.bool_]:
    """A (k,) boolean mask from ``values`` (all True when None)."""
    if values is None:
        return np.ones(k, dtype=bool)
    mask = np.asarray(values, dtype=bool).ravel()
    if mask.shape != (k,):
        raise ValueError(f"{what}: expected {k} entries, got {mask.shape}")
    return mask


def _circle_and_tau(length_nm: float, tau_nm: float, n_null: int, what: str) -> Tuple[float, float, int]:
    """The circle's length, the tolerance and the replicate count, validated."""
    length = float(length_nm)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"{what}: length_nm must be a positive finite number, got {length_nm}")
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"{what}: tau_nm must be a positive finite number, got {tau_nm}")
    n_rep = int(n_null)
    if n_rep < 1 or n_rep != n_null:
        raise ValueError(f"{what}: n_null must be an integer >= 1, got {n_null}")
    return length, tau, n_rep


def rotate_arcs(
    s_nm: NDArray[np.float64],
    shift_nm: NDArray[np.float64],
    reflect: NDArray[np.bool_],
    length_nm: float,
) -> NDArray[np.float64]:
    """
    The (B, K) arcs of B rotations of one ring on the circle of length
    L: row r is (L - s if reflect[r] else s) + shift[r], modulo L -- the
    reflection s -> L - s about the circle's origin, then the shift.
    In this exact arithmetic (so that a replicate rebuilt by hand from
    the recorded draws is the module's bit for bit).
    """
    s = np.asarray(s_nm, dtype=np.float64).reshape(-1)
    shifts = np.asarray(shift_nm, dtype=np.float64).reshape(-1)
    flips = np.asarray(reflect, dtype=bool).reshape(-1)
    length = float(length_nm)
    base = np.where(flips[:, None], length - s[None, :], s[None, :])
    return np.asarray(np.mod(base + shifts[:, None], length), dtype=np.float64)


def _arc_null_counts(
    s_a: NDArray[np.float64],
    s_b: NDArray[np.float64],
    length_nm: float,
    tau_nm: float,
) -> NDArray[np.int64]:
    """
    Arc-matched pairs per replicate, (B,), for B configurations of ring
    b (``s_b`` (B, K_b), usable arcs only) against ring a, fixed
    (``s_a`` (K_a,)) or itself replicated ((B, K_a), the joint null).
    The (b, K_a, K_b) cyclic distance matrices of a block are formed at
    once (``cyclic_arc_distance_nm``, the arithmetic of ``match_arcs``);
    a replicate in which no cluster has two admissible partners is
    counted as its number of admissible pairs (each conflicts with
    nothing: the shortcut of ``_assign_within_tau``), the others go
    through ``assign_within_tau`` on their own matrix -- the very call
    ``match_arcs`` makes on the same numbers, so the counts agree with
    a per-replicate ``match_arcs`` exactly.
    """
    n_rep = int(s_b.shape[0])
    counts = np.zeros(n_rep, dtype=np.int64)
    if s_a.shape[-1] == 0 or s_b.shape[1] == 0:
        return counts
    for start in range(0, n_rep, _BLOCK):
        stop = min(start + _BLOCK, n_rep)
        a = s_a if s_a.ndim == 1 else s_a[start:stop]
        D = cyclic_arc_distance_nm(a[..., :, None] - s_b[start:stop][:, None, :], length_nm)   # (b, K_a, K_b)
        admissible = D <= tau_nm
        n_edges = admissible.sum(axis=(1, 2))
        simple = (admissible.sum(axis=2).max(axis=1) <= 1) & (admissible.sum(axis=1).max(axis=1) <= 1)
        counts[start:stop] = np.where(simple, n_edges, 0)
        for r in np.flatnonzero(~simple):
            counts[start + r] = assign_within_tau(D[r], tau_nm)[0].size
    return counts


def arc_eclipse_test(
    s_a_nm: NDArray[np.float64],
    s_b_nm: NDArray[np.float64],
    length_nm: float,
    tau_nm: float,
    *,
    n_null: int = 1999,
    random_seed: int = 0,
    ring_a: int = 0,
    ring_b: int = 1,
    reference_ring: int = 0,
    usable_a: Optional[NDArray[np.bool_]] = None,
    usable_b: Optional[NDArray[np.bool_]] = None,
) -> ArcPairMatch:
    """
    The eclipse test of one ring pair in ONE dimension: the clusters'
    arcs on the reference circle, matched within the arc tolerance
    ``tau_nm`` (``match_arcs``: cyclic distance, maximum cardinality
    then minimum total distance, D-24b), against ``n_null`` replicates
    in which ring b is ROTATED on the circle with a reflection drawn at
    random: s_b' = (sigma s_b + U) mod L, U ~ Uniform(0, L), sigma = -1
    with probability 1/2, NO exclusion (H5-B, candidate A).

    Why a rotation with no exclusion, when the 2D null of H3 excludes
    |U| < p_bar / 2 (D-11): the shift is here a symmetry of the null.
    Under H0 the two rings are independent and each is a stationary
    process on the circle (its law is invariant under the rotations
    and reflections of the circle: a hard-core process, a Poisson
    process, any process whose construction does not single out an
    origin), so for any fixed group element g, (s_a, g s_b) has the
    law of (s_a, s_b). Let h be a Haar (uniform rotation x fair
    reflection) element independent of everything and g_1 .. g_B iid
    Haar: the B + 1 configurations (s_a, h g_r s_b), r = 0 .. B with
    g_0 the identity, are iid Haar images of (s_a, s_b), hence
    exchangeable, and since (s_a, h s_b) has the law of the observed
    configuration the test that scores the observed count among the B
    null counts is exact: P(p_excess <= alpha) <= alpha, with equality
    up to the discreteness of the count (``validate_columns_h3.
    exact_exchangeable_rates``). The exclusion of D-11 would make the
    observed phase special (every replicate at least m from it, no two
    replicates need be), which breaks the exchangeability -- the same
    reason the pcf test of this module dropped it (``UnrollParams.
    min_shift_fraction``, D-27). What the 2D nulls lack is exactly this
    symmetry: a shift along an interpolating curve through sparse,
    scattered vertices changes the clusters' distances to the other
    ring by the curve's overshoot, not by a rigid move of the
    configuration (D-25; the pilot of 2026-09-25 measured rejection
    rates far above the nominal level in leak-free and leaky M1 cells).
    On the circle the
    rotation moves every cluster of b by the same arc, and nothing else
    changes. The argument needs the arcs of b to be a stationary
    process on the circle GIVEN the arcs of a, which holds when the
    circle is a curve independent of both rings' scatter (the true
    membrane: exact on the simulator's own contour, review of
    2026-09-25) and fails when the circle is the interpolating spline
    through ring a's own scattered centroids, as ``analyze_arc_columns``
    builds it by default: b's projected arcs then carry a's scatter
    (conservative, mean zeta -0.1 to -0.25), two projected rings share
    the curve's distortion (liberal, +0.65 at 65 nm scatter on the
    Fourier contour) and on concave contours the curve bridges the
    concavities (liberal); ``ARC_REFERENCE_CURVES`` and the pooled
    membrane of
    ``analyze_arc_columns(reference_curve="pooled_membrane")`` are the
    measured remedy. What the 1D test gives up: a match is a coincidence of
    arcs only, so two clusters at the same arc but different radial
    offsets (a cluster inside the membrane, a leak child at its
    parent's arc) are matched alike; the radial offset is reported by
    ``unroll`` and the leak by ``tools.mps_leak``, and the "clean"
    variant of ``analyze_arc_columns`` takes the flagged children out.

    The generator is ``default_rng(SeedSequence(random_seed,
    spawn_key=(ring_a, ring_b, ARC_SPAWN_KEY)))``; the shifts are drawn
    first, then the reflections (the order of ``tools.mps_matching.
    _draw_shifts``). The observed pairs never depend on the seed.

    Parameters
    ----------
    s_a_nm, s_b_nm : (K_a,), (K_b,)
        Arcs of the two rings' clusters on the reference curve, nm.
    length_nm
        L, the reference curve's closed length.
    tau_nm
        The arc tolerance tau_s (the run's tau_0, D-20: the same
        pre-specified tolerance, now measured along the membrane).
    n_null, random_seed
        Replicates and seed.
    ring_a, ring_b, reference_ring
        The ring indices (non-negative: they seed the null) and the
        reference ring, recorded.
    usable_a, usable_b : (K_a,), (K_b,) bool, optional
        Clusters allowed to enter; all when None. ``K_a``/``K_b`` of the
        result count the usable ones.

    Returns
    -------
    ArcPairMatch
    """
    length, tau, n_rep = _circle_and_tau(length_nm, tau_nm, n_null, "arc_eclipse_test")
    if int(ring_a) < 0 or int(ring_b) < 0 or int(reference_ring) < 0:
        raise ValueError(f"arc_eclipse_test: ring indices must be non-negative to seed the null, got "
                         f"({ring_a}, {ring_b}, reference {reference_ring})")
    s_a = _arcs(s_a_nm, "s_a_nm")
    s_b = _arcs(s_b_nm, "s_b_nm")
    ua = _usable_mask(usable_a, s_a.size, "usable_a")
    ub = _usable_mask(usable_b, s_b.size, "usable_b")
    warnings_: List[str] = []
    i_a, j_b, d = match_arcs(s_a, s_b, length, tau, usable_a=ua, usable_b=ub)
    n_obs = int(i_a.size)
    k_a, k_b = int(np.count_nonzero(ua)), int(np.count_nonzero(ub))
    min_k = min(k_a, k_b)
    rng = np.random.default_rng(np.random.SeedSequence(int(random_seed),
                                                       spawn_key=(int(ring_a), int(ring_b), ARC_SPAWN_KEY)))
    shifts = np.asarray(rng.uniform(0.0, length, size=n_rep), dtype=np.float64)
    reflect = np.asarray(rng.random(n_rep) < 0.5, dtype=bool)
    null = _arc_null_counts(s_a[ua], rotate_arcs(s_b[ub], shifts, reflect, length), length, tau)
    label = f"rings {ring_a}-{ring_b} on the curve of ring {reference_ring} at tau_s {tau:g} nm"
    e_star, sd_star, zeta = _null_summary(n_obs, null, min_k, warnings_, label)
    p_excess, p_deficit, p_two = _phipson_smyth(null, n_obs)
    return ArcPairMatch(
        ring_a=int(ring_a), ring_b=int(ring_b), reference_ring=int(reference_ring), length_nm=length, tau_nm=tau,
        i_a=i_a, j_b=j_b, d_nm=d, n_matched=n_obs, K_a=k_a, K_b=k_b,
        E_dir=n_obs / min_k if min_k else float("nan"),
        E_star=e_star, sd_star=sd_star, zeta=zeta,
        p_excess=p_excess, p_deficit=p_deficit, p_two_sided=p_two,
        null_n_matched=null, null_shift_nm=shifts, null_reflect=reflect,
        n_null=n_rep, random_seed=int(random_seed), warnings=warnings_,
    )


def arc_joint_null(
    ring_indices: Sequence[int],
    arcs_nm: Sequence[NDArray[np.float64]],
    length_nm: float,
    tau_nm: float,
    *,
    reference_ring: int,
    n_null: int = 1999,
    random_seed: int = 0,
    usable: Optional[Sequence[Optional[NDArray[np.bool_]]]] = None,
) -> ArcJointNull:
    """
    The joint rotation null of one axon on the reference circle: T =
    the sum over adjacent pairs (consecutive ring indices) of the
    arc-matched count, observed against replicates in which every ring
    but the reference is rotated by its own independent (U_k,
    reflect_k), U_k ~ Uniform(0, L), and every pair is re-matched with
    the rotated arcs of BOTH its rings. The pairs (k, k+1) and (k+1,
    k+2) share ring k+1, so their counts are dependent and the joint
    replicate carries that dependence into the null (02 B3 S6, as
    ``tools.mps_matching.axon_joint_null``).

    Exactness: the argument of ``arc_eclipse_test`` extends to several
    rings. Under H0 the rings are independent stationary processes on
    the circle, so for independent Haar elements h_k the tuple (s_ref,
    h_1 s_1, ..., h_n s_n) has the law of the observed one, and the B +
    1 configurations (s_ref, h_k g_k^r s_k) with g_k^0 the identity and
    g_k^r iid Haar are iid Haar images of the observed configuration,
    hence exchangeable, whatever the statistic of the whole
    configuration (T included). No relative-shift exclusion (D-24d) is
    needed: that rule repaired the contamination of the 2D joint null
    by its OWN exclusion window, which made the observed phase of a
    pair special; with no window there is nothing to repair, and a
    replicate that happens to re-create a pair's phase is exactly as
    likely as any other, which is what exchangeability means.

    Seeds: ``SeedSequence(random_seed, spawn_key=(ARC_SPAWN_KEY,
    JOINT_SPAWN_KEY))`` spawned once per ring in ``rings`` order (the
    ring indices sorted); ring k draws its shifts, then its
    reflections, from its own child; the reference's child is spawned
    and unused, so that adding a ring never changes the draws of the
    others.

    Parameters
    ----------
    ring_indices
        The ring indices (non-negative, distinct); sorted internally,
        the arcs and masks following.
    arcs_nm
        One (K_k,) array of arcs on the reference curve per ring, in
        ``ring_indices`` order (the reference's are its curve arcs).
    length_nm, tau_nm, n_null, random_seed
        As ``arc_eclipse_test``.
    reference_ring
        The ring whose curve is the circle; must be among the indices.
    usable
        Optional per-ring masks (None entries = all usable), in
        ``ring_indices`` order.

    Returns
    -------
    ArcJointNull
        Empty with a warning (T 0, NaN T_A and z_A, p-values 1) when
        there is no adjacent pair.
    """
    length, tau, n_rep = _circle_and_tau(length_nm, tau_nm, n_null, "arc_joint_null")
    idx = [int(k) for k in ring_indices]
    if len(idx) != len(arcs_nm):
        raise ValueError(f"arc_joint_null: {len(idx)} ring indices for {len(arcs_nm)} arc arrays")
    if usable is not None and len(usable) != len(idx):
        raise ValueError(f"arc_joint_null: {len(usable)} masks for {len(idx)} rings")
    if any(k < 0 for k in idx):
        raise ValueError(f"arc_joint_null: ring indices must be non-negative, got {idx}")
    if len(set(idx)) != len(idx):
        raise ValueError(f"arc_joint_null: repeated ring index in {idx}")
    ref = int(reference_ring)
    if ref not in idx:
        raise ValueError(f"arc_joint_null: reference ring {reference_ring} is not among the rings {idx}")
    order = sorted(range(len(idx)), key=lambda q: idx[q])
    rings = [idx[q] for q in order]
    arcs = [_arcs(arcs_nm[q], f"arcs of ring {idx[q]}") for q in order]
    masks = [_usable_mask(None if usable is None else usable[q], arcs[i].size, f"usable of ring {idx[q]}")
             for i, q in enumerate(order)]
    n_rings = len(rings)
    warnings_: List[str] = []
    pairs: List[Tuple[int, int]] = []
    for q in range(n_rings - 1):
        if rings[q + 1] == rings[q] + 1:
            pairs.append((q, q + 1))
        else:
            warnings_.append(f"rings {rings[q]} and {rings[q + 1]} are not consecutive (gap of "
                             f"{rings[q + 1] - rings[q] - 1} ring index): they form no adjacent pair")
    shifts = np.zeros((n_rep, n_rings), dtype=np.float64)
    flips = np.zeros((n_rep, n_rings), dtype=bool)
    children = np.random.SeedSequence(int(random_seed), spawn_key=(ARC_SPAWN_KEY, JOINT_SPAWN_KEY)).spawn(n_rings)
    positions: List[NDArray[np.float64]] = []
    for q in range(n_rings):
        rng = np.random.default_rng(children[q])
        if rings[q] != ref:
            shifts[:, q] = rng.uniform(0.0, length, size=n_rep)
            flips[:, q] = rng.random(n_rep) < 0.5
        positions.append(rotate_arcs(arcs[q][masks[q]], shifts[:, q], flips[:, q], length))
    n_pairs = len(pairs)
    n_obs = np.zeros(n_pairs, dtype=np.int64)
    min_k = np.zeros(n_pairs, dtype=np.int64)
    null = np.zeros((n_rep, n_pairs), dtype=np.int64)
    for p, (i, j) in enumerate(pairs):
        n_obs[p] = match_arcs(arcs[i], arcs[j], length, tau, usable_a=masks[i], usable_b=masks[j])[0].size
        min_k[p] = min(int(np.count_nonzero(masks[i])), int(np.count_nonzero(masks[j])))
        null[:, p] = _arc_null_counts(positions[i], positions[j], length, tau)
    t_null = null.sum(axis=1)
    t_obs = int(n_obs.sum())
    ring_pairs = [(rings[i], rings[j]) for i, j in pairs]
    if n_pairs == 0:
        warnings_.append(f"rings {rings}: no adjacent pair; the joint arc null is empty")
        return ArcJointNull(
            tau_nm=tau, pairs=[], n_matched_obs=n_obs, T_obs=0, T_null=t_null, T_A=float("nan"), z_A=float("nan"),
            p_excess=1.0, p_deficit=1.0, p_two_sided=1.0, n_null=n_rep, random_seed=int(random_seed), rings=rings,
            reference_ring=ref, null_n_matched=null, null_shift_nm=shifts, null_reflect=flips,
            E_dir_obs=np.zeros(0, dtype=np.float64), E_star=np.zeros(0, dtype=np.float64), min_K=min_k,
            warnings=warnings_)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_dir = np.where(min_k > 0, n_obs / np.maximum(min_k, 1), np.nan)
        e_star = np.where(min_k > 0, null.mean(axis=0) / np.maximum(min_k, 1), np.nan)
    if not np.all(min_k > 0):
        warnings_.append("a pair has no usable cluster on one side (min K = 0): its E_dir and E* are NaN and T_A is NaN")
    t_a = float(np.mean(e_dir - e_star))
    label = f"joint arc null of rings {rings} on the curve of ring {ref} at tau_s {tau:g} nm"
    _, _, z_a = _null_summary(t_obs, t_null, 1, warnings_, label)
    p_excess, p_deficit, p_two = _phipson_smyth(t_null, t_obs)
    return ArcJointNull(
        tau_nm=tau, pairs=ring_pairs, n_matched_obs=n_obs, T_obs=t_obs, T_null=t_null, T_A=t_a, z_A=z_a,
        p_excess=p_excess, p_deficit=p_deficit, p_two_sided=p_two, n_null=n_rep, random_seed=int(random_seed),
        rings=rings, reference_ring=ref, null_n_matched=null, null_shift_nm=shifts, null_reflect=flips,
        E_dir_obs=np.asarray(e_dir, dtype=np.float64), E_star=np.asarray(e_star, dtype=np.float64), min_K=min_k,
        warnings=warnings_)


def ambiguous_projection(
    path: SmoothPath,
    points_nm: NDArray[np.float64],
    r_nm: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Which of ``points_nm`` (n, 2) have NO unique projection on the
    closed curve ``path`` -- the diagnostic behind ``n_ambiguous`` of
    ``analyze_arc_columns`` (``r_nm`` (n,) are their signed offsets from
    ``project_on_path``). A point is flagged on either of two grounds:

    * the fold: |r| >= the local curvature radius of the curve at the
      foot of the projection. Along the curve near the foot the
      distance from the point behaves as sqrt(r^2 + (1 - kappa |r|)^2
      (s - s*)^2), so once |r| reaches 1 / kappa the foot is no longer
      a minimum of the distance and the arc coordinate the projection
      reports is meaningless (the point sits at or beyond the centre
      of curvature, where the normals of the curve cross). The
      curvature is measured on the dense table over
      ``CURVATURE_WINDOW_ROWS`` rows on each side of the foot;
    * the tie: the distance along the curve has a second local minimum,
      outside the basin of the nearest foot (the rows reachable from it
      while the distance does not decrease), closer than twice the
      nearest distance. Such a point lies within half its own offset of
      the curve's medial axis (the locus of points with two nearest
      feet: the centre of a circle, the major-axis segment of an
      ellipse), so a displacement smaller than the offset that put it
      there would move its arc to another part of the ring. The fold is
      the local statement of the same fact; the tie catches the points
      that are far from every part of the curve (a cluster at the axon
      centre projects on the nearest of two opposite walls).

    What the count means (reviews of 2026-09-25, which corrected the
    first version of this docstring): the fold compares |r| with the
    curvature radius of the CURVE, and when the curve is the
    interpolating spline through a ring's scattered centroids that
    radius between two vertices is ~p^2 / (8 sigma) for vertex spacing
    p and scatter sigma -- ~170 nm at 30 nm and ~80 nm at 65 nm of
    scatter with p = 200 nm -- so clusters that sit ON the membrane are
    flagged whenever their own 65 nm offset from the wobbling curve
    exceeds it: 11 % of the clusters of a ring on the membrane at 65 nm
    of scatter (R 800), 24 % for a ring 170 nm inside the
    reference ellipse, and the
    TRUE membrane, on which the rotation null is exact, flags more than
    the interpolating curve on concave contours because its
    concavities are deeper. The count is therefore a measure of the
    curve's roughness at the projected positions, not of the ring's
    distance from the membrane and not a validity gate of the test; on
    a smooth curve (the pooled P-spline membrane, an ellipse) both
    grounds do hold only for a cluster far from the membrane. The tie
    rule is beyond H5B_SPEC S1's definition (the fold alone): it is
    kept because the fold misses a cluster at the centre of an ellipse
    (|r| ~ 1000 nm against a curvature radius of 2250 nm at the nearest
    wall, yet two walls are equally near), and it adds a share of the
    flags on concave contours (compliance review of 2026-09-25).
    Reported and never removed. Returns an (n,) boolean array.
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    r = np.asarray(r_nm, dtype=np.float64).reshape(-1)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    if pts.ndim != 2 or pts.shape[1] != 2 or r.shape != (pts.shape[0],):
        raise ValueError(f"ambiguous_projection: expected (n, 2) points and (n,) offsets, got {pts.shape} and {r.shape}")
    a = np.asarray(path.points_nm, dtype=np.float64)
    n_rows = a.shape[0]
    edges = np.asarray(path.edges_nm, dtype=np.float64)
    w = max(1, min(CURVATURE_WINDOW_ROWS, n_rows // 4))
    d = np.hypot(pts[:, None, 0] - a[None, :, 0], pts[:, None, 1] - a[None, :, 1])      # (n, N)
    foot = np.argmin(d, axis=1)
    # The fold: curvature at the foot from the turn between the chords (foot - w -> foot) and (foot -> foot + w).
    before = a[np.mod(foot - w, n_rows)]
    after = a[np.mod(foot + w, n_rows)]
    t1 = a[foot] - before
    t2 = after - a[foot]
    turn = np.abs(np.arctan2(t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0], (t1 * t2).sum(axis=1)))
    arc = np.zeros(pts.shape[0], dtype=np.float64)
    for j in range(2 * w):
        arc += edges[np.mod(foot - w + j, n_rows)]
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(turn > 0.0, 0.5 * arc / np.maximum(turn, 1e-300), np.inf)
    fold = np.abs(r) >= radius
    # The tie: the second basin of the distance along the curve, closer than twice the nearest distance.
    rolled = np.take_along_axis(d, np.mod(np.arange(n_rows)[None, :] + foot[:, None], n_rows), axis=1)   # min at column 0
    diff = np.diff(rolled, axis=1)                                          # (n, N - 1)
    neg = diff < 0.0
    pos = diff > 0.0
    has_fwd, has_bwd = neg.any(axis=1), pos.any(axis=1)
    k_f = np.where(has_fwd, np.argmax(neg, axis=1), n_rows - 1)            # last row of the forward basin
    m = np.where(has_bwd, n_rows - 2 - np.argmax(pos[:, ::-1], axis=1), -1)   # last row outside the backward basin
    tie = np.zeros(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        lo, hi = int(k_f[i]) + 1, int(m[i])
        if lo <= hi:
            tie[i] = bool(rolled[i, lo:hi + 1].min() < 2.0 * rolled[i, 0])
    return np.asarray(fold | tie, dtype=bool)


def fit_centroid_membrane(
    centroids_by_ring: Sequence[NDArray[np.float64]],
    *,
    ring_indices: Optional[Sequence[int]] = None,
    ring_contours: Optional[Sequence[Optional[NDArray[np.float64]]]] = None,
    knot_spacing_nm: float = CENTROID_MEMBRANE_KNOT_SPACING_NM,
    min_clusters_per_knot: Optional[float] = CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT,
    iterations: int = CENTROID_MEMBRANE_ITERATIONS,
) -> CentroidMembrane:
    """
    The repaired centroid membrane of one axon (H5-D, recipe
    ``CENTROID_MEMBRANE_RECIPE``; D-34b, D-35c), with the leave-one-cluster-out
    arc of every centroid -- the port, operation for operation, of the
    research's r2lib.cmx_fit (tour_curve + cluster_limited_knot_spacing_nm +
    weighted_fit with unit weights and no robust drop) and loco_arcs, so that
    it reproduces the frozen candidate bit for bit (a private research
    validator compares them).

    1. The pooled centroids of every ring given (``centroids_by_ring``, each
       (K_k, 2) in ``Ring.clusters`` order; every cluster, usable or not:
       the membrane is geometry) in the order of
       ``tools.mps_membrane.pooled_centroid_tour`` (every ring's own tour
       ``ring_contours[k]`` as a start, 2-opt + Or-opt + 2-opt, the
       shortest kept), and the chord-parameter P-spline of that polygon with
       knots every ``CENTROID_MEMBRANE_INITIAL_KNOT_SPACING_NM``
       (``smooth_polygon_membrane``, its length within
       ``CENTROID_MEMBRANE_INITIAL_MAX_LENGTH_CHANGE`` of the tour's),
       tabulated by ``smooth_closed_path``: the initial curve, of length L0.
    2. The knots: ``knot_spacing_nm``, raised with ``min_clusters_per_knot``
       (None: not raised) to at least that many pooled centroids per knot
       interval of L0 (``cluster_limited_knot_spacing_nm``).
    3. ``iterations`` fits of the periodic cubic P-spline (penalty
       ``CENTROID_MEMBRANE_PENALTY_REL`` x mean diag(B'B), second
       differences) to the pooled centroids, each centroid parameterised by
       its projection on the previous curve (the initial one first); every
       fit's length is checked against L0 (``CENTROID_MEMBRANE_MAX_LENGTH_
       CHANGE``) on a preview of ``CENTROID_MEMBRANE_PREVIEW_ROWS`` rows
       before the dense table is built.
    4. The arcs: every centroid's arc on the final curve at the parameter of
       its nearest point on the curve refitted WITHOUT it (the exact
       Sherman-Morrison downdate of the last fit), searched within
       +/- ``CENTROID_MEMBRANE_LOCO_WINDOW_KNOTS`` knot intervals of its
       parameter on ``CENTROID_MEMBRANE_LOCO_GRID`` points.

    ``ring_indices`` label the rings (default 0, 1, ...). Raises ValueError
    with fewer than four pooled centroids, no ring with three, fewer
    centroids than knots, a singular fit, or a curve whose length leaves
    the guard.
    """
    sets = [np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in centroids_by_ring]
    ids = list(range(len(sets))) if ring_indices is None else [int(k) for k in ring_indices]
    if len(ids) != len(sets):
        raise ValueError(f"fit_centroid_membrane: {len(ids)} ring indices for {len(sets)} centroid sets")
    if len(set(ids)) != len(ids):
        raise ValueError(f"fit_centroid_membrane: repeated ring index in {ids}")
    n_iter = int(iterations)
    if n_iter < 1 or n_iter != iterations:
        raise ValueError(f"fit_centroid_membrane: iterations must be an integer >= 1, got {iterations}")
    if not sets:
        raise ValueError("fit_centroid_membrane: no ring given")
    pts = np.concatenate(sets, axis=0)
    if not np.isfinite(pts).all():
        raise ValueError("fit_centroid_membrane: non-finite centroid")
    warnings_: List[str] = []
    # 1. the all-starts pooled tour and the initial curve (r2lib.tour_curve)
    order = pooled_centroid_tour(sets, ring_contours=None if ring_contours is None else list(ring_contours))
    curve0 = smooth_polygon_membrane(pts[order], knot_spacing_nm=float(CENTROID_MEMBRANE_INITIAL_KNOT_SPACING_NM),
                                     max_length_change=float(CENTROID_MEMBRANE_INITIAL_MAX_LENGTH_CHANGE))
    init = smooth_closed_path(curve0)
    # 2. the knots (r2lib.cmx_fit)
    knots = float(knot_spacing_nm)
    if not (math.isfinite(knots) and knots > 0.0):
        raise ValueError(f"fit_centroid_membrane: knot_spacing_nm must be positive and finite, got {knot_spacing_nm}")
    if min_clusters_per_knot is not None and float(min_clusters_per_knot) > 0:
        knots = cluster_limited_knot_spacing_nm(knots, int(pts.shape[0]), float(init.length_nm),
                                                min_clusters_per_knot=float(min_clusters_per_knot))
    # 3. the fits (r2lib.weighted_fit with unit base weights, robust_c None, param_knot_spacing_nm None)
    L0 = float(init.length_nm)
    n_knots = max(4, int(round(L0 / float(knots))))
    if pts.shape[0] < CENTROID_MEMBRANE_MIN_POINTS_PER_KNOT * n_knots:
        raise ValueError(f"fit_centroid_membrane: {pts.shape[0]} centroids for {n_knots} knots")
    s0, _r0 = _project(init, pts)
    u = s0 / L0
    w = np.ones(pts.shape[0]) * np.ones(pts.shape[0])
    path: Optional[SmoothPath] = None
    coef: Optional[NDArray[np.float64]] = None
    inv: Optional[NDArray[np.float64]] = None
    guard = float(CENTROID_MEMBRANE_MAX_LENGTH_CHANGE)
    for it in range(n_iter):
        B = _periodic_bspline_design(u, n_knots)
        WB = B * w[:, None]
        gram = B.T @ WB
        eye = np.eye(n_knots)
        D = np.roll(eye, -1, axis=1) - 2.0 * eye + np.roll(eye, 1, axis=1)
        pen = gram + float(CENTROID_MEMBRANE_PENALTY_REL) * float(np.mean(np.diag(gram))) * (D.T @ D)
        try:
            inv = np.linalg.inv(pen)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"fit_centroid_membrane: singular penalised normal matrix ({pts.shape[0]} centroids, "
                             f"{n_knots} knots)") from exc
        rhs = WB.T @ pts
        coef = inv @ rhs
        u_0 = np.arange(int(CENTROID_MEMBRANE_PREVIEW_ROWS)) / float(CENTROID_MEMBRANE_PREVIEW_ROWS)
        prev = _periodic_bspline_design(u_0, n_knots) @ coef
        L_prev = float(np.hypot(*(np.roll(prev, -1, axis=0) - prev).T).sum())
        if not (np.isfinite(L_prev) and abs(L_prev - L0) <= guard * L0):
            raise ValueError(f"fit_centroid_membrane: fit {it + 1}: length {L_prev:.0f} nm against the initial curve's "
                             f"{L0:.0f} nm (preview; guard {guard:g})")
        path = _tabulate_path(coef, n_knots, "fit_centroid_membrane")
        if abs(path.length_nm - L0) > guard * L0:
            raise ValueError(f"fit_centroid_membrane: fit {it + 1}: length {path.length_nm:.0f} nm against the initial "
                             f"curve's {L0:.0f} nm (guard {guard:g})")
        s, _off = _project(path, pts)
        if it < n_iter - 1:
            u = s / float(path.length_nm)
    assert path is not None and coef is not None and inv is not None
    # 4. the leave-one-cluster-out arcs (r2lib.loco_arcs)
    n_dense = int(np.asarray(path.points_nm).shape[0])
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    Bi = _periodic_bspline_design(u, n_knots)
    fitted = Bi @ coef
    arcs_all = np.empty(pts.shape[0])
    du = float(CENTROID_MEMBRANE_LOCO_WINDOW_KNOTS) / n_knots
    grid_off = np.linspace(-du, du, int(CENTROID_MEMBRANE_LOCO_GRID))
    for i in range(pts.shape[0]):
        b = Bi[i]
        wi = float(w[i])
        if wi > 0:
            h = wi * float(b @ inv @ b)
            ri = pts[i] - fitted[i]
            coef_i = coef - np.outer(inv @ b, ri) * (wi / (1.0 - h))
        else:
            coef_i = coef
        ug = np.mod(u[i] + grid_off, 1.0)
        cg = _periodic_bspline_design(ug, n_knots) @ coef_i
        j = int(np.argmin(np.hypot(cg[:, 0] - pts[i, 0], cg[:, 1] - pts[i, 1])))
        us = float(ug[j]) * n_dense
        j0 = int(math.floor(us)) % n_dense
        frac = us - math.floor(us)
        arcs_all[i] = (cum[j0] + frac * (cum[j0 + 1] - cum[j0])) % float(path.length_nm)
    area = _signed_area_nm2(np.asarray(path.points_nm, dtype=np.float64))
    if area == 0.0:
        raise ValueError("fit_centroid_membrane: the fitted curve is degenerate (zero area)")
    _s_full, offsets_all = project_on_path(path, pts, outward_sign=-math.copysign(1.0, area))
    n_cross = int(self_crossings(path))
    if n_cross:
        warnings_.append(f"the centroid membrane crosses itself {n_cross} time(s): a membrane cannot, so the fit is "
                         "suspect (inspect the axon's clusters for lumen clusters the cleaning left)")
    arcs: Dict[int, NDArray[np.float64]] = {}
    offsets: Dict[int, NDArray[np.float64]] = {}
    pos = 0
    for k, sset in zip(ids, sets):
        arcs[k] = np.asarray(arcs_all[pos:pos + sset.shape[0]], dtype=np.float64)
        offsets[k] = np.asarray(offsets_all[pos:pos + sset.shape[0]], dtype=np.float64)
        pos += sset.shape[0]
    return CentroidMembrane(
        path=path, length_nm=float(path.length_nm), initial_length_nm=L0, knot_spacing_nm=float(knots), n_knots=int(n_knots),
        iterations=n_iter, penalty_rel=float(CENTROID_MEMBRANE_PENALTY_REL), n_points=int(pts.shape[0]), rings=ids,
        arcs_nm=arcs, offsets_nm=offsets, tour_order=np.asarray(order, dtype=np.intp), n_self_crossings=n_cross,
        warnings=warnings_)


def _ring_contour_or_none(ring: Ring) -> Optional[NDArray[np.float64]]:
    """``Ring.contour_nm`` when it is a usable tour start ((>= 3, 2), finite), else None (r2lib.tour_curve's rule)."""
    c = ring.contour_nm
    c = None if c is None else np.asarray(c, dtype=np.float64)
    return c if c is not None and c.ndim == 2 and c.shape[0] >= 3 and np.isfinite(c).all() else None


def centroid_membrane_of_rings(
    res: RingsResult,
    ring_indices: Optional[Sequence[int]] = None,
    *,
    knot_spacing_nm: float = CENTROID_MEMBRANE_KNOT_SPACING_NM,
    min_clusters_per_knot: Optional[float] = CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT,
    iterations: int = CENTROID_MEMBRANE_ITERATIONS,
) -> CentroidMembrane:
    """
    ``fit_centroid_membrane`` of the rings of ``res`` (every ring with a
    contour consistent with its clusters, ``tools.mps_matching.
    ring_geometry``, as ``analyze_arc_columns`` takes them; or the rings
    ``ring_indices``), each ring's validated tour as a start of the pooled
    tour -- the curve ``analyze_arc_columns(reference_curve=
    "centroid_membrane")`` projects on, for a caller that wants to draw it
    (the review window draws it before and after a cleaning). ValueError
    when a ring index names no ring with a geometry, or the fit fails.
    """
    by_index = {int(r.index): r for r in res.rings}
    geoms: Dict[int, RingGeometry] = {}
    for k in sorted(by_index):
        try:
            geoms[k] = ring_geometry(by_index[k], include_suspect=True)
        except ValueError:
            continue
    ids = sorted(geoms) if ring_indices is None else [int(k) for k in ring_indices]
    missing = [k for k in ids if k not in geoms]
    if missing:
        raise ValueError(f"centroid_membrane_of_rings: ring(s) {missing} have no contour consistent with their clusters "
                         f"(rings with one: {sorted(geoms)})")
    return fit_centroid_membrane([np.asarray(geoms[k].centroids_nm, dtype=np.float64) for k in ids], ring_indices=ids,
                                 ring_contours=[_ring_contour_or_none(by_index[k]) for k in ids],
                                 knot_spacing_nm=knot_spacing_nm, min_clusters_per_knot=min_clusters_per_knot,
                                 iterations=iterations)


def _empty_arc_result(
    source_name: str, tau: float, rings: List[int], inc: bool, n_rep: int, seed: int, warnings_: List[str],
) -> ArcColumnsResult:
    """An ``ArcColumnsResult`` with no pair (no reference, or one ring)."""
    return ArcColumnsResult(
        source_name=source_name, reference_ring=-1, length_nm=float("nan"), tau_nm=tau, rings=rings, adjacent=[],
        k2=[], joint=None, T_A=float("nan"), z_A=float("nan"), p_A=float("nan"), n_ambiguous={k: 0 for k in rings},
        excluded_clusters=[], include_suspect=inc, n_null=n_rep, random_seed=seed, warnings=warnings_)


def analyze_arc_columns(
    res: RingsResult,
    cols: AxonColumnsResult,
    *,
    params: UnrollParams = UnrollParams(),
    tau_nm: Optional[float] = None,
    include_suspect: Optional[bool] = None,
    reference: Optional[int] = None,
    exclude_clusters: Optional[Sequence[Tuple[int, int]]] = None,
    reference_curve: str = "interpolating",
    membrane_knot_spacing_nm: Optional[float] = None,
    membrane_iterations: int = MEMBRANE_LOCALIZATION_ITERATIONS,
    membrane_max_points: Optional[int] = None,
) -> ArcColumnsResult:
    """
    The arc test of one axon (H5-B, candidate A): every ring's clusters
    as arcs on ONE closed curve of the axon, ``arc_eclipse_test`` for
    every adjacent and every k+2 pair (the persistence control of 02
    P7), ``arc_joint_null`` over the adjacent pairs, all at tau_s =
    ``tau_nm`` (None: ``cols.tau0_nm``, the pre-specified tolerance,
    now along the membrane), with ``n_null = params.n_null`` and the
    seed of the columns run (``cols.random_seed``, D-22).

    The reference is the ring with the most clusters (a tie: the lowest
    index), as ``unroll``, or ``reference``: the ring that stays fixed
    in the joint null. ``reference_curve`` (``ARC_REFERENCE_CURVES``)
    picks the circle: with "interpolating" (the default; H5B_SPEC S1)
    it is the reference ring's own interpolating spline, its clusters'
    arcs are its curve arcs exactly (``RingGeometry.arc_nm``, r = 0)
    and every other ring's clusters are projected on it by
    ``project_on_path`` (s the arc of the nearest point, r the signed
    offset, + outward); with "pooled_membrane" it is the P-spline
    membrane of ALL the rings' centroids (``tools.mps_matching.
    pooled_membrane_path`` with knots every ``membrane_knot_spacing_nm``,
    None = ``ARC_MEMBRANE_KNOT_SPACING_NM`` = 600 nm, recorded in the
    result) and every ring, the reference included, is projected on
    it. The second is the remedy of the reviews of 2026-09-25 (the
    comment at ``ARC_REFERENCE_CURVES``: the interpolating reference
    curve is a function of the reference ring's own scatter, which
    makes the rotation null conservative for the reference against a
    projected ring, liberal for two projected rings -- so the level of
    the joint test is a mixture over where the densest ring sits,
    liberal with the reference at an end, conservative in the middle
    -- and liberal on concave contours, where the pooled membrane is
    level or conservative); ``power_columns``
    reports both. What the pooled membrane does NOT reach (re-review of
    2026-09-25; the numbers at ``ARC_REFERENCE_CURVES``): H5B_SPEC S2's
    |mean zeta| <= 0.10 exactly, at 65 nm of scatter it stays
    conservative by -0.06 to -0.13 sd at 600 nm knots (-0.22 to -0.36
    at the 2D null's 400 nm, which is why the spacing is its own), the
    true membrane alone gives the exact test. With
    "localization_membrane" (H5-C, D-29c: the candidate for the primary
    test) the circle is the P-spline membrane fitted to the
    LOCALIZATIONS of every ring that entered (``tools.mps_membrane.
    localization_membrane_of_rings``: the pooled centroid P-spline as
    the initial curve, ``membrane_iterations`` robust weighted fits with
    knots every ``membrane_knot_spacing_nm``, None =
    ``ARC_LOCALIZATION_KNOT_SPACING_NM`` = 400 nm since the review of
    2026-09-26 (600 nm before; the argument at the constant) and,
    with that default only, floored at ``tools.mps_membrane.
    MEMBRANE_MIN_CLUSTERS_PER_KNOT`` detected clusters per knot
    interval (re-review of 2026-09-27; ``membrane_knot_spacing_nm``
    records the spacing used, and a warning says when it moved),
    ``membrane_max_points`` a seeded subsample of the localizations or
    None = all; the comment at ``ARC_REFERENCE_CURVES`` says why), every
    ring projected on it; the result then carries the fit
    (``membrane``), the radial scatter of every cluster about it
    (``radial_scatter_nm``) and the leave-ring-out per-pair diagnostic
    (``loo_adjacent``: each adjacent pair re-tested on the membrane of
    the OTHER rings' localizations when there are at least three rings,
    else on the pooled curve itself). With "centroid_membrane" (H5-D,
    D-35c: the PRIMARY curve, on the rings the lumen cleaning left) the
    circle is the repaired membrane of the pooled centroids of every ring
    that entered (``fit_centroid_membrane``, recipe
    ``CENTROID_MEMBRANE_RECIPE``: knots every ``membrane_knot_spacing_nm``,
    None = ``CENTROID_MEMBRANE_KNOT_SPACING_NM`` floored at
    ``CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT`` clusters per interval,
    ``membrane_iterations`` fits; any other setting is warned about as
    not the frozen recipe) and every cluster's arc is its
    leave-one-cluster-out arc on it; the result carries the fit
    (``centroid_membrane``), ``length_nm`` its length and
    ``membrane_knot_spacing_nm`` the knots used, and a curve that cannot
    be fitted raises ValueError. ``curve_role`` labels what each curve's
    result is for since D-35(c) (``ARC_CURVE_ROLES``: the localization
    membrane is a diagnostic with a known conservative bias).
    A ``reference`` given explicitly with fewer clusters
    than another ring is warned about under "interpolating": its curve
    is then shorter than the membrane (review of 2026-09-25) and the
    null liberal; the rule that picks the densest ring is the guard,
    and the pooled membrane does not depend on the choice.
    ``include_suspect`` defaults to
    ``cols.include_suspect`` so that the clusters entering the arc
    matching are the ones that entered the 2D one; ``exclude_clusters``
    (ring index, cluster index in ``Ring.clusters`` order) marks more
    clusters unusable -- this is how the caller builds the "clean"
    variants of the test, passing the leak-explained children of
    ``cols.leak`` (the single-parent rule, or the ``_any`` rule of
    D-27b) -- and the entries applied are recorded in
    ``excluded_clusters``; an entry that names no cluster is warned
    about, not fatal. ``n_ambiguous`` counts, per ring, the usable
    clusters whose projection on the curve is not unique
    (``ambiguous_projection``; 0 for the reference under
    "interpolating"): they stay in the matching, since removing them
    would make the count depend on a diagnostic, and the number says
    how rough the curve is at the ring's positions, not how far the
    ring sits from the membrane and not whether the test is exact
    (``ArcColumnsResult``). A ring without a contour consistent with its
    clusters is left out with a warning; with no ring, or fewer than
    two, the result is empty with a warning. The inputs are not
    modified.
    """
    inc = bool(cols.include_suspect) if include_suspect is None else bool(include_suspect)
    seed = int(cols.random_seed)
    n_rep = int(params.n_null)
    tau = float(cols.tau0_nm if tau_nm is None else tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"analyze_arc_columns: tau_nm must be positive and finite, got {tau}")
    curve_kind = str(reference_curve)
    if curve_kind not in ARC_REFERENCE_CURVES:
        raise ValueError(f"analyze_arc_columns: reference_curve must be one of {ARC_REFERENCE_CURVES}, got {reference_curve!r}")
    name = str(res.source_name)
    warnings_: List[str] = []
    geoms: Dict[int, RingGeometry] = {}
    for ring in sorted(res.rings, key=lambda r: int(r.index)):
        try:
            geoms[int(ring.index)] = ring_geometry(ring, include_suspect=inc)
        except ValueError as exc:
            warnings_.append(f"ring {ring.index} left out of the arc test: {exc}")
    ids = sorted(geoms)
    if not geoms:
        warnings_.append("no ring has a contour consistent with its clusters; no arc test")
        return _empty_arc_result(name, tau, ids, inc, n_rep, seed, warnings_)
    if reference is None:
        ref = max(geoms, key=lambda k: (geoms[k].n_clusters, -k))
    else:
        ref = int(reference)
        if ref not in geoms:
            raise ValueError(f"analyze_arc_columns: reference ring {reference} has no usable contour (rings with "
                             f"one: {ids})")
    geom_ref = geoms[ref]
    if reference is not None and curve_kind == "interpolating":
        densest = max(geoms, key=lambda k: (geoms[k].n_clusters, -k))
        if geoms[densest].n_clusters > geom_ref.n_clusters:
            warnings_.append(f"reference ring {ref} has {geom_ref.n_clusters} clusters against {geoms[densest].n_clusters} of "
                             f"ring {densest}: the interpolating curve of a sparser reference is shorter than the membrane and "
                             "the rotation null liberal (review of 2026-09-25); prefer the default reference (the densest "
                             "ring) or reference_curve='pooled_membrane'")
    knots = float("nan")
    membrane: Optional[LocalizationMembrane] = None
    centroid_fit: Optional[CentroidMembrane] = None
    radial_scatter = float("nan")
    radial_scatter_arc = float("nan")
    scatter_knots = float("nan")
    if curve_kind in ("pooled_membrane", "localization_membrane", "centroid_membrane"):
        if curve_kind == "centroid_membrane":
            default_knots = CENTROID_MEMBRANE_KNOT_SPACING_NM
        else:
            default_knots = ARC_LOCALIZATION_KNOT_SPACING_NM if curve_kind == "localization_membrane" else ARC_MEMBRANE_KNOT_SPACING_NM
        knots = float(default_knots if membrane_knot_spacing_nm is None else membrane_knot_spacing_nm)
        if not (math.isfinite(knots) and knots > 0.0):
            raise ValueError(f"analyze_arc_columns: membrane_knot_spacing_nm must be positive and finite, got {membrane_knot_spacing_nm}")
    n_iter = int(membrane_iterations)
    cap = None if membrane_max_points is None else int(membrane_max_points)
    # Re-review of 2026-09-27: with the default spacing, the localization membrane's knots are floored at
    # MEMBRANE_MIN_CLUSTERS_PER_KNOT detected clusters per interval (tools.mps_membrane.cluster_limited_knot_spacing_nm;
    # untouched at typical cluster densities, ~600 nm on the sparse two-ring synthetics); an explicit spacing is used as given.
    knots_requested = knots
    min_per_knot: Optional[float] = (MEMBRANE_MIN_CLUSTERS_PER_KNOT
                                     if curve_kind == "localization_membrane" and membrane_knot_spacing_nm is None else None)
    if curve_kind == "pooled_membrane":
        try:
            path = pooled_membrane_path([geoms[k] for k in ids], knot_spacing_nm=knots)
        except ValueError as exc:
            raise ValueError(f"analyze_arc_columns: the pooled membrane of rings {ids} cannot be fitted ({exc})") from exc
        area = _signed_area_nm2(np.asarray(path.points_nm, dtype=np.float64))
    elif curve_kind == "localization_membrane":
        try:
            membrane = localization_membrane_of_rings(res, ids, knot_spacing_nm=knots, iterations=n_iter, max_points=cap,
                                                      min_clusters_per_knot=min_per_knot)
        except ValueError as exc:
            raise ValueError(f"analyze_arc_columns: the localization membrane of rings {ids} cannot be fitted ({exc})") from exc
        knots_requested = knots
        knots = float(membrane.knot_spacing_nm)
        if knots != knots_requested:
            warnings_.append(f"localization membrane: knots every {knots:.0f} nm instead of {knots_requested:g} (at least "
                             f"{MEMBRANE_MIN_CLUSTERS_PER_KNOT:g} detected clusters per knot interval)")
        warnings_.extend(f"localization membrane: {w}" for w in membrane.warnings)
        path = membrane.path
        area = _signed_area_nm2(np.asarray(path.points_nm, dtype=np.float64))
        all_centroids = np.concatenate([np.asarray(geoms[k].centroids_nm, dtype=np.float64).reshape(-1, 2) for k in ids], axis=0)
        _offsets, radial_scatter_arc = radial_scatter_on_membrane(all_centroids, path)
        # The radial scatter is the simulator's D-25 quantity: about the localization membrane at the scatter
        # estimate's own knots (MEMBRANE_KNOT_SPACING_NM, what sim_config_from_axon measures and simnull closes), so
        # that the fidelity of a simulated null compares like with like whatever the arc curve's knots (review of
        # 2026-09-26: with the arc curve at 400 nm the M1 leak axon, whose clusters are ON the membrane, read 1.9 nm
        # against 2.5 in its simulations closed at 600 nm -- a 30 % "misfit" of pure centroid noise).
        scatter_knots = float(MEMBRANE_KNOT_SPACING_NM)
        if abs(knots - scatter_knots) <= 1e-9:
            radial_scatter = radial_scatter_arc
        else:
            try:
                m_sc = localization_membrane_of_rings(res, ids, knot_spacing_nm=scatter_knots, iterations=n_iter, max_points=cap)
                _o_sc, radial_scatter = radial_scatter_on_membrane(all_centroids, m_sc.path)
            except ValueError as exc:
                warnings_.append(f"the {scatter_knots:g} nm localization membrane of the radial scatter cannot be fitted ({exc}); "
                                 "the scatter about the arc curve is reported instead")
                radial_scatter, scatter_knots = radial_scatter_arc, knots
    elif curve_kind == "centroid_membrane":
        # H5-D (D-34b, D-35c): the repaired centroid membrane of every ring that entered, the recipe frozen with the
        # default keywords (the knots floored at CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT clusters per interval only then).
        by_ring = {int(r.index): r for r in res.rings}
        try:
            centroid_fit = fit_centroid_membrane(
                [np.asarray(geoms[k].centroids_nm, dtype=np.float64) for k in ids], ring_indices=ids,
                ring_contours=[_ring_contour_or_none(by_ring[k]) for k in ids], knot_spacing_nm=knots,
                min_clusters_per_knot=CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT if membrane_knot_spacing_nm is None else None,
                iterations=n_iter)
        except ValueError as exc:
            raise ValueError(f"analyze_arc_columns: the centroid membrane of rings {ids} cannot be fitted ({exc})") from exc
        knots_requested = knots
        knots = float(centroid_fit.knot_spacing_nm)
        if membrane_knot_spacing_nm is not None or n_iter != CENTROID_MEMBRANE_ITERATIONS:
            warnings_.append(f"centroid membrane fitted with knots every {knots:g} nm and {n_iter} iteration(s): not the frozen "
                             f"recipe {CENTROID_MEMBRANE_RECIPE} (D-34b), which the level was measured for")
        elif knots != knots_requested:
            warnings_.append(f"centroid membrane: knots every {knots:.0f} nm instead of {knots_requested:g} (at least "
                             f"{CENTROID_MEMBRANE_MIN_CLUSTERS_PER_KNOT:g} clusters per knot interval, as the recipe floors them)")
        if cap is not None:
            warnings_.append("membrane_max_points is ignored by the centroid membrane (it is fitted to the centroids)")
        warnings_.extend(f"centroid membrane: {w}" for w in centroid_fit.warnings)
        path = centroid_fit.path
        area = _signed_area_nm2(np.asarray(path.points_nm, dtype=np.float64))
    else:
        path = geom_ref.path
        area = _signed_area_nm2(np.asarray(geom_ref.contour_nm, dtype=np.float64))
    if area == 0.0:
        raise ValueError(f"analyze_arc_columns: the {curve_kind} curve of reference ring {ref} is degenerate (zero area)")
    outward_sign = -math.copysign(1.0, area)
    length = float(path.length_nm)
    arcs: Dict[int, NDArray[np.float64]] = {}
    offsets: Dict[int, NDArray[np.float64]] = {}
    usable: Dict[int, NDArray[np.bool_]] = {}
    for k, g in geoms.items():
        if k == ref and curve_kind == "interpolating":
            arcs[k] = np.asarray(g.arc_nm, dtype=np.float64).copy()
            offsets[k] = np.zeros(g.n_clusters, dtype=np.float64)
        elif curve_kind == "centroid_membrane" and centroid_fit is not None:
            # The leave-one-cluster-out arcs of the recipe (the full curve's arc at the parameter of the nearest point
            # of the curve refitted without the cluster); the offsets from the full curve, for ambiguous_projection.
            arcs[k] = np.asarray(centroid_fit.arcs_nm[k], dtype=np.float64).copy()
            offsets[k] = np.asarray(centroid_fit.offsets_nm[k], dtype=np.float64).copy()
        else:
            arcs[k], offsets[k] = project_on_path(path, np.asarray(g.centroids_nm, dtype=np.float64),
                                                  outward_sign=outward_sign)
        usable[k] = np.asarray(g.usable, dtype=bool).copy()
    applied: List[Tuple[int, int]] = []
    if exclude_clusters:
        wanted = sorted({(int(k), int(i)) for k, i in exclude_clusters})
        for k, i in wanted:
            if k in geoms and 0 <= i < geoms[k].n_clusters:
                usable[k][i] = False
                applied.append((k, i))
        if len(applied) != len(wanted):
            warnings_.append(f"exclude_clusters: {len(wanted) - len(applied)} of {len(wanted)} entries name no "
                             "cluster of the rings that entered")
        warnings_.append(f"{len(applied)} cluster(s) excluded from the arc matching on request (exclude_clusters)")
    n_ambiguous: Dict[int, int] = {}
    # The H5-B wording is kept verbatim for the two H5-B curves (review of 2026-09-26: the warning text of
    # reference_curve="pooled_membrane" had lost its "of ring {ref}"; an existing output must not change); only the
    # H5-C curve gets its own label.
    if curve_kind == "localization_membrane":
        curve_label = "the localization membrane"
    elif curve_kind == "centroid_membrane":
        curve_label = "the centroid membrane"
    else:
        curve_label = f"the {'pooled membrane' if curve_kind == 'pooled_membrane' else 'curve'} of ring {ref}"
    for k, g in geoms.items():
        if k == ref and curve_kind == "interpolating":
            n_ambiguous[k] = 0
            continue
        sel = usable[k]
        flagged = ambiguous_projection(path, np.asarray(g.centroids_nm, dtype=np.float64)[sel], offsets[k][sel])
        n_ambiguous[k] = int(np.count_nonzero(flagged))
        if n_ambiguous[k]:
            warnings_.append(f"ring {k}: {n_ambiguous[k]} of {int(np.count_nonzero(sel))} usable cluster(s) have no "
                             f"unique projection on {curve_label} (reported, kept in the matching)")
    if len(ids) < 2:
        warnings_.append(f"{len(ids)} ring(s) with a contour: fewer than 2 rings, nothing to match across rings")
        out = _empty_arc_result(name, tau, ids, inc, n_rep, seed, warnings_)
        out.reference_ring, out.length_nm, out.n_ambiguous, out.excluded_clusters = ref, length, n_ambiguous, applied
        out.reference_curve = curve_kind
        out.membrane_knot_spacing_nm = knots
        out.membrane, out.radial_scatter_nm = membrane, radial_scatter
        out.centroid_membrane, out.curve_role = centroid_fit, ARC_CURVE_ROLES.get(curve_kind, "")
        return out

    def test(a: int, b: int, label: str, s_a: Optional[NDArray[np.float64]] = None, s_b: Optional[NDArray[np.float64]] = None,
             circle_nm: Optional[float] = None) -> ArcPairMatch:
        m = arc_eclipse_test(arcs[a] if s_a is None else s_a, arcs[b] if s_b is None else s_b,
                             length if circle_nm is None else circle_nm, tau, n_null=n_rep, random_seed=seed, ring_a=a, ring_b=b,
                             reference_ring=ref, usable_a=usable[a], usable_b=usable[b])
        warnings_.extend(f"{label} {a}-{b}: {w}" for w in m.warnings)
        return m

    adjacent = [test(k, k + 1, "adjacent") for k in ids if k + 1 in geoms]
    k2 = [test(k, k + 2, "k+2") for k in ids if k + 2 in geoms]
    loo_adjacent: List[ArcPairMatch] = []
    if curve_kind == "localization_membrane":
        # The leave-ring-out diagnostic: the pair on the membrane of the OTHER rings' localizations (a curve independent
        # of both tested rings), else -- with two rings -- on the pooled curve itself. Per pair, never joint.
        for m_adj in adjacent:
            a, b = int(m_adj.ring_a), int(m_adj.ring_b)
            others = [k for k in ids if k not in (a, b)]
            if not others:
                warnings_.append(f"leave-ring-out {a}-{b}: no other ring; the pair is tested on the pooled localization membrane")
                loo_adjacent.append(test(a, b, "leave-ring-out"))
                continue
            try:
                m_loo = localization_membrane_of_rings(res, others, knot_spacing_nm=(knots if min_per_knot is None else knots_requested),
                                                       iterations=n_iter, max_points=cap, min_clusters_per_knot=min_per_knot)
            except ValueError as exc:
                warnings_.append(f"leave-ring-out {a}-{b}: the membrane of rings {others} cannot be fitted ({exc}); pair left out")
                continue
            loo_path = m_loo.path
            loo_sign = -math.copysign(1.0, _signed_area_nm2(np.asarray(loo_path.points_nm, dtype=np.float64)))
            s_a, _r_a = project_on_path(loo_path, np.asarray(geoms[a].centroids_nm, dtype=np.float64), outward_sign=loo_sign)
            s_b, _r_b = project_on_path(loo_path, np.asarray(geoms[b].centroids_nm, dtype=np.float64), outward_sign=loo_sign)
            loo_adjacent.append(test(a, b, "leave-ring-out", s_a, s_b, float(loo_path.length_nm)))
    if len(adjacent) < len(ids) - 1:
        warnings_.append(f"ring indices {ids} are not all consecutive: {len(adjacent)} adjacent pair(s) of "
                         f"{len(ids) - 1} possible")
    joint = arc_joint_null(ids, [arcs[k] for k in ids], length, tau, reference_ring=ref, n_null=n_rep,
                           random_seed=seed, usable=[usable[k] for k in ids])
    warnings_.extend(f"joint null: {w}" for w in joint.warnings)
    return ArcColumnsResult(
        source_name=name, reference_ring=ref, length_nm=length, tau_nm=tau, rings=ids, adjacent=adjacent, k2=k2,
        joint=joint, T_A=joint.T_A, z_A=joint.z_A, p_A=joint.p_two_sided, n_ambiguous=n_ambiguous,
        excluded_clusters=applied, include_suspect=inc, n_null=n_rep, random_seed=seed, warnings=warnings_,
        reference_curve=curve_kind, membrane_knot_spacing_nm=knots, membrane=membrane, radial_scatter_nm=radial_scatter,
        radial_scatter_knot_spacing_nm=scatter_knots, radial_scatter_arc_curve_nm=radial_scatter_arc,
        loo_adjacent=loo_adjacent, centroid_membrane=centroid_fit, curve_role=ARC_CURVE_ROLES.get(curve_kind, ""))
