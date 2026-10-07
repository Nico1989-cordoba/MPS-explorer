# -*- coding: utf-8 -*-
"""
The membrane of a ring -- or of several rings of one axon pooled -- as a
periodic P-spline, separated from the radial scatter of the cluster
centroids about it.

``radial_scatter_nm`` is H5's membrane / scatter split of one ring's
contour (the simulator's ``radial_offset_sd``; ``sim_config_from_axon``),
moved here from ``tools.mps_simulate_axon`` UNCHANGED in behaviour (that
module re-exports it, its constants and ``MembraneFit``, so every caller
and ``validate_simulate_axon.py`` see the same objects). ``fit_membrane``
is H5-B's addition (H5B_SPEC S2, candidate B): the same P-spline fitted
to points in ANY order -- the centroids of every ring of the axon except
the one whose null is drawn -- and tabulated as a ``SmoothPath`` so that
``tools.mps_matching`` can walk it (``null_kind="pooled_offset"``).

Why a periodic P-spline with knots every ``MEMBRANE_KNOT_SPACING_NM``
(the argument of H5, kept next to the constants below): a curve fitted
THROUGH the centroids (D-25's s = K smoothing spline, rms residual 1 nm
by construction) is the scatter, not the membrane, and a curve smoothed
to the scatter's own scale is an unstable estimate at K = 20-40. Three
mean cluster spacings per knot leave each knot interval ~3 clusters and
the fit K - n_knots degrees of freedom, a cubic spline with 600 nm knots
follows a membrane of curvature radius >= 600 nm to < 1 nm, and the
second-difference penalty only bites where a knot interval holds no
vertex.

Why the pooled membrane must not be fitted to the moved ring's own
points (H5B_SPEC S2, the pilot of 2026-09-25): a curve estimated from
ring b's centroids is a function of b's configuration, so moving b
along it does not produce an exchangeable configuration -- the
interpolating and the conserved-offset nulls both rejected far above
the nominal level in leak-free M1 replicates on concave simulated
contours. Fitted to the OTHER rings only, the curve is independent of
b under the null and b's shifted configuration is exchangeable with its
observed one up to the estimation error of the membrane, argued in
``tools.mps_matching.eclipse_test``.

H5-C (H5C_SPEC S1; D-29c): the membrane from the LOCALIZATIONS.
``fit_membrane_from_localizations`` fits the same periodic P-spline to
every localization of the rings (~100x more points than the centroids,
and not a function of any single cluster's configuration), parameterised
by the arc of each localization's projection on an initial curve, with
a robust step that drops the background and refits;
``localization_membrane_of_rings`` builds it from a ``RingsResult`` with
the pooled centroid P-spline as the initial curve, optionally leaving
one ring out; ``radial_scatter_on_membrane`` is the radial scatter of a
set of centroids about that curve, the D-25 quantity, now identifiable
(``radial_scatter_nm`` can return NaN or 0, D-28b).
Why the localizations: on realistic simulated geometry the arc test on the pooled
CENTROID membrane keeps the level in rate but with a residual bias
(D-28f) that the TRUE membrane does not have (exact
in the 16 synthetic cases of the H5-B adversary); the localizations are
the closest thing to the true membrane the data offer.

No Qt, no matplotlib; deterministic (``fit_membrane_from_localizations``
draws its optional subsample from a seeded generator).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from tools.mps_columns import Ring, RingsResult
from tools.mps_geometry import reconstruct_perimeter
from tools.mps_matching import PATH_MIN_SAMPLES, PATH_STEP_NM, SmoothPath, ring_geometry, smooth_closed_path

__all__ = [
    "MEMBRANE_CURVE_SAMPLES",
    "MEMBRANE_KNOT_SPACING_NM",
    "MEMBRANE_LOCALIZATION_ITERATIONS",
    "MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT",
    "MEMBRANE_MAX_LENGTH_CHANGE",
    "MEMBRANE_MIN_CLUSTERS_PER_KNOT",
    "MEMBRANE_NULL_KNOT_SPACING_NM",
    "MEMBRANE_PARAMETER_KNOT_SPACING_NM",
    "MEMBRANE_PENALTY_REL",
    "MEMBRANE_ROBUST_C",
    "MEMBRANE_RING_TOUR_STARTS",
    "MEMBRANE_TOUR_MAX_PASSES",
    "INITIAL_ORDERS",
    "LocalizationMembrane",
    "MembraneFit",
    "cluster_limited_knot_spacing_nm",
    "fit_membrane",
    "fit_membrane_from_localizations",
    "localization_membrane_of_rings",
    "pooled_centroid_curve",
    "pooled_centroid_tour",
    "radial_scatter_nm",
    "radial_scatter_on_membrane",
    "self_crossings",
    "smooth_polygon_membrane",
]

# Knot spacing (chord length) of the periodic least-squares B-spline that
# separates a measured ring's MEMBRANE from the radial SCATTER of its
# clusters (``radial_scatter_nm``, ``sim_config_from_axon``; review of
# 2026-09-25). A curve fitted THROUGH the centroids (D-25's s = K spline,
# rms residual 1 nm by construction) cannot carry a scatter, and a curve
# smoothed to the scatter's own scale (s = K sigma^2) is an unstable
# estimate of the membrane at K = 20-40 (leave-one-out probes of
# 2026-09-25: variance factors 2-3 with a spread of the same size). Three
# mean cluster spacings (~200 nm on typical rings) per knot leave each knot
# interval ~3 clusters and the fit K - n_knots degrees of freedom, and a
# cubic spline with 600 nm knots follows a membrane of curvature radius
# >= 600 nm to < 1 nm (h^4 / (384 R^3) = 0.34 nm at R = 1000); structure
# of the membrane BELOW that scale is counted as scatter (12 nm on the
# Fourier-perturbed ellipse of the probes, 1.4-1.8 nm on the exact one).
MEMBRANE_KNOT_SPACING_NM = 600.0
# Relative weight of the second-difference penalty on the periodic
# B-spline coefficients (a P-spline, Eilers & Marx): lambda = this x
# mean(diag(B'B)). It only bites where the vertices leave a coefficient
# unconstrained (a knot interval without a vertex: hard-core spacings
# are irregular, and a plain least-squares spline then oscillates
# between the vertices, a 154 um curve for a 20 um ring in the review
# probe of 2026-09-25) and costs ~1.5 of the fitted degrees of freedom
# elsewhere (dof_fit 10.6 of 13 knots on the harness ellipse; the
# variance factor on a noisy circle 0.96; 15 nm recovered as 14.2-15.4
# at K = 20-67). Ten times more shrinks the ellipse tips by 8 nm and
# inflates the scatter by 40 %.
MEMBRANE_PENALTY_REL = 0.01
# Knot spacing of the membrane the POOLED NULL walks (``fit_membrane``'s
# default; ``null_kind="pooled_offset"``), finer than the 600 nm of the
# scatter estimate above because the two fits answer different
# questions. The scatter estimate wants the membrane's structure below
# the knot scale counted AS scatter (it is, for the simulator); the null
# wants the opposite: every real bend of the membrane that both rings
# share must be IN the curve, because a shifted cluster keeps its offset
# from the curve, not from the membrane, and structure the curve misses
# is a spread the null adds to every shifted pair (the observed pair is
# aligned on it, the shifted one is not) -- fewer null matches, zeta
# biased in favour of columns. Measured on the review's Fourier-perturbed
# ellipse (harmonics 2/3/5/7 up to 0.10 relative, wavelength of the 7th
# 1207 nm, curvature radius down to 308 nm; the arc-null validator's
# contour, probes of 2026-09-25): at 600 nm knots (two knot intervals
# per wavelength of the 7th harmonic) the curve misses the contour by
# 16 nm (median, exact rings K = 40) and the pooled null's mean zeta on
# 30 nm-scattered rings is +0.47 to +0.79 (R = 80-100, se 0.12), against
# +0.15 the specification allows; at 400-450 nm it is 6 nm and the
# mean zeta -0.09 to +0.24 (the interpolating kind on the same pairs:
# -0.09 to +0.40), and the ellipse cases stay where 600 nm put them (K =
# 40, 30 nm: +0.01 vs -0.01; two non-reference rings: +0.02 vs +0.02).
# The other way the knot spacing can fail is too FINE: with fewer than
# ~2 centroids per knot interval the P-spline (penalty 1 %) is nearly
# an interpolant of the other ring and D-25's overshoot between its
# vertices comes back (ellipse K = 40, 30 nm: +0.11 at 300 nm knots; K =
# 20, 30 nm, spacing 400 nm: +0.03 at 600, +0.12-0.16 at 400, and on the
# Fourier contour at K = 20 the coarse side is worse still, +0.95 at
# 800 nm). Two centroids per knot interval at a realistic spacing (~200
# nm) is 400 nm, and a cubic
# spline with 400 nm knots follows a curvature radius R to h^4 / (384
# R^3) = 1.8 nm at R = 308 (11.6 nm at 600 nm knots). A stronger
# penalty with finer knots (the P-spline's own way out of the trade-off)
# was probed too -- 300 nm / 5 %, 250 nm / 20 % -- and is equivalent
# within the noise (+0.22 / +0.26 on the Fourier 30 nm case at K = 40),
# so the penalty stays H5's and one number moves. H5B_SPEC wrote the
# signature with knot_spacing_nm=600: the deviation and these numbers
# are reported for D-28, and ``radial_scatter_nm`` keeps 600. The 1D
# ARC test on the pooled membrane (``tools.mps_unroll.analyze_arc_columns
# (reference_curve="pooled_membrane")``) does NOT use this constant: it
# reads the curve only for the arc coordinate, which structure the
# curve misses shifts to second order, so it can afford the coarser
# spacing that removes the fit's self-pull (``tools.mps_unroll.
# ARC_MEMBRANE_KNOT_SPACING_NM`` = 600, the argument and the numbers
# there).
MEMBRANE_NULL_KNOT_SPACING_NM = 400.0
# Vertices of ``MembraneFit.curve_nm``, the fitted membrane resampled as
# a closed polygon: the same 200 as ``tools.mps_simulate_axon.
# MEASURED_CONTOUR_VERTICES`` (the measured contour the simulator
# carries), so ``radial_scatter_nm``'s default is unchanged by the move.
MEMBRANE_CURVE_SAMPLES = 200
# ``fit_membrane``'s guard against a fitted curve that loops or collapses:
# the closed length of the tabulated curve may differ from the length of
# the tour through the points by at most this fraction. It is looser
# than ``radial_scatter_nm``'s 0.1 on purpose: the tour through the
# POOLED centroids of two rings that scatter 30 nm about the membrane
# zigzags between the rings' clusters and is ~11 % longer than the
# membrane (8821 vs 7933 nm on the harness ellipse at K = 40, probe of
# 2026-09-25), which is not a looping curve.
MEMBRANE_MAX_LENGTH_CHANGE = 0.5
# Finite-difference step in the unit parameter for the tangent of the
# fitted curve at the vertices (``_signed_offsets``): 1e-4 of the tour
# is ~1 nm on a ring, far below any knot interval.
_TANGENT_DU = 1e-4
# ``fit_membrane_from_localizations`` (H5-C). Fits of the P-spline to
# the localizations: the first on the parameterisation the initial
# curve gives, each later one on the projection of the localizations on
# the curve just fitted, after the robust step. Two are enough: the
# initial curve (the pooled centroid P-spline) is within tens of nm of
# the membrane, the parameter of a localization moves by ~(r - delta)
# delta' (second order) when the curve moves by delta, and the harness
# measures the arcs of one and two iterations within 20 nm of each
# other and the curve of one iteration already within its tolerance.
MEMBRANE_LOCALIZATION_ITERATIONS = 2
# Robust cut, in units of the MAD-sd of the signed offsets: a
# localization farther than this from the current curve (the initial
# curve before the first fit, then the curve just fitted) gets weight
# ZERO in the next fit (a drop, not a Huber down-weighting). Why a drop:
# the localizations this rule removes are the uniform background of
# the pick (hundreds of nm from the membrane; 5 % of the localizations
# in the harness, more on a realistic axon) and the tail of the
# leak and of the cluster scatter, none of which carries membrane
# information; a Huber weight c x sd / |r| keeps a pull of c x sd x
# sign per point, which is ~0.7 nm net on the harness ellipse (5 %
# background, 58 / 42 inside / outside) but tens of nm INWARD on an
# axon whose lumen is full of background (on a slot-shaped library
# contour most of the background lies inside, far from the membrane:
# a weighted pull of tens of nm). At 3 MAD-sd a Gaussian
# scatter loses 0.27 % of its points, nothing; the count kept is
# reported (``LocalizationMembrane.n_used``). The MAD-sd is that of ALL
# the offsets (background included), so it is inflated where the
# background is heavy (above the cluster scatter on a realistic axon)
# and the cut correspondingly looser, which is the safe
# direction: a cut tighter than the cluster scatter would drop
# membrane clusters.
MEMBRANE_ROBUST_C = 3.0
# Fewer localizations than this many per knot make the fit degenerate
# (ValueError): with 3 per knot the weighted least squares still has
# every coefficient constrained by data rather than by the penalty.
MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT = 3
# Knot spacing of the intermediate fits whose curve RE-PARAMETERISES the
# localizations (``fit_membrane_from_localizations``, step 4), finer
# than the membrane's own. Measured on a slot-shaped library contour
# (the library contour smoothed at 600 nm of its raw chord length, which
# leaves structure below 600 nm; probes of 2026-09-26): re-projecting on
# the curve just fitted at the membrane's spacing is a CONTRACTING
# fixed point wherever the basis bridges a bay -- the bay's
# localizations project onto the bridge, their parameter span shrinks,
# the next fit bridges more: from the TRUE contour as initial curve the
# exact contour refits farther from the truth and shorter than the
# polygon's own-order P-spline at the same spacings -- while on a
# convex membrane the same re-projection is what removes the initial
# curve's distortion (ellipse of validate_columns_h3, 40 centroids
# scattered 65 nm as the initial curve: 5.9 nm median after one fit,
# 1.3 after the refit). A parameterisation curve with knots every 300
# nm (770 localizations per knot interval on 20 000 points, ~1 nm of
# noise) follows the bays the 600 nm membrane cannot, so the membrane is
# then fitted in a parameter close to the true arc and reaches its
# basis's own limit instead of the contracted one. None restores the
# literal step 4 (re-projection on the membrane's own curve).
MEMBRANE_PARAMETER_KNOT_SPACING_NM = 300.0
# 1.4826 x MAD is the sd of a Gaussian: the robust scale of the offsets.
_MAD_TO_SD = 1.4826
# Improvement passes of each local search of ``pooled_centroid_tour`` (2-opt,
# then Or-opt, then 2-opt again): a pass that improves nothing ends the
# search, and on the pooled centroids of a realistic axon the
# searches end after a few passes (probes of 2026-09-26); the bound only
# guarantees termination.
MEMBRANE_TOUR_MAX_PASSES = 50
# Starts of the tour of ONE ring built by ``pooled_centroid_tour`` when no
# ring contour is given: the ring's centroids in polar order about their
# mean, rolled by 0, 1/4, 1/2 and 3/4 of the ring, each improved by the
# local search, the shortest kept (the rolled starts are what
# ``tools.mps_geometry.reconstruct_perimeter(all_starts=True)`` does for
# ``build_rings``, at every start). With ONE start the tour of a
# slot-shaped ring jumps the slot often enough to leave the arc test on
# the resulting membrane liberal on a slot-shaped library contour (grid v2
# smoothing); with four starts it is close to level and eight starts add
# nothing (probes of 2026-09-26), at 60 ms per ring.
MEMBRANE_RING_TOUR_STARTS = 4
# How ``localization_membrane_of_rings`` orders the pooled centroids of its
# initial curve: "tour" (the default since the review of 2026-09-26: the
# shortest closed tour ``pooled_centroid_tour`` finds from every ring's
# order) or "reference_arc" (the first H5-C recipe: the order of their arc
# on the densest ring's interpolating curve, kept to reproduce the numbers
# of that review; ``_pooled_centroid_path_reference_arc``).
INITIAL_ORDERS: Tuple[str, ...] = ("tour", "reference_arc")
# Re-review of 2026-09-27 (minor finding: the membrane validator's S3
# ellipse, 65 nm scatter, K 20 per ring, two rings, read -0.09 to -0.12
# sd CONSERVATIVE against the true curve at 400 nm knots, seed after
# seed). The fit's self-pull -- each detected cluster drags the curve
# towards its own centroid in proportion to its share of the
# localizations under a basis function -- grows as the clusters per knot
# interval fall, and on that case 40 detected clusters over 7.9 um leave
# two per 400 nm interval (denser cases hold several more). A floor on the
# knot spacing of this many DETECTED clusters (every ring's clusters,
# pooled) per knot interval, h >= c x L / K_pooled
# (``cluster_limited_knot_spacing_nm``), leaves every axon with >= 3 per
# interval at the requested spacing untouched (typical densities: well
# under 100 nm per pooled cluster)
# and moves the sparse synthetic cases to ~600 nm, where the same probe
# read -0.004 / +0.017 (65 nm) and +0.010 / +0.018 (30 nm) at K 20
# (a research probe, R 200, paired against the true
# curve). ``tools.mps_unroll.analyze_arc_columns`` applies it when its
# knot spacing is not given explicitly.
MEMBRANE_MIN_CLUSTERS_PER_KNOT = 3.0
# Vertices of the curve ``self_crossings`` tests (the tabulated curve is
# subsampled to at most this many; a crossing loop of the membrane fits
# of the review spans hundreds of nm, far above the ~40 nm subsample).
_CROSSING_MAX_VERTICES = 400


def _signed_area_nm2(contour: NDArray[np.float64]) -> float:
    """Shoelace signed area of a closed polygon: positive when the tour
    runs counter-clockwise (the same convention as ``tools.mps_matching``
    and ``tools.mps_simulate_axon``, each of which keeps its own copy)."""
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _periodic_bspline_design(u: NDArray[np.float64], n_knots: int, degree: int = 3) -> NDArray[np.float64]:
    """(n, n_knots) design matrix of the periodic uniform B-spline basis of
    ``degree`` with ``n_knots`` knots at j / n_knots on the unit circle,
    at parameters ``u`` (taken modulo 1): the ordinary basis on the
    knot grid extended by ``degree`` knots on each side, its columns
    wrapped modulo n_knots."""
    uu = np.mod(np.asarray(u, dtype=np.float64).reshape(-1), 1.0)
    t_ext = np.arange(-degree, n_knots + degree + 1, dtype=np.float64) / n_knots
    dense = BSpline.design_matrix(uu, t_ext, degree).toarray()
    out = np.zeros((uu.size, n_knots), dtype=np.float64)
    for i in range(dense.shape[1]):
        out[:, i % n_knots] += dense[:, i]
    return out


@dataclass
class MembraneFit:
    """What ``radial_scatter_nm`` and ``fit_membrane`` return for one
    membrane: ``sigma_nm`` the radial scatter of the points about the
    fitted membrane (NaN when the fit is not identifiable),
    ``offsets_nm`` (K,) their signed residuals along the outward normal
    (+ outward; in the INPUT order of the points), ``n_knots`` and
    ``dof`` = floor(K - trace of the hat matrix) of the fit, ``curve_nm``
    the fitted membrane resampled at ``n_samples`` equally spaced
    parameters (closed polygon, vertex 0's parameter first),
    ``length_nm`` its closed length.

    ``path`` and ``order`` are ``fit_membrane``'s only (None from
    ``radial_scatter_nm``, whose result is unchanged by H5-B): the
    fitted curve tabulated densely as a ``SmoothPath`` (the curve the
    pooled null of ``tools.mps_matching`` walks; ``length_nm`` is then
    that table's closed length) and the tour through the input points
    that parameterised the fit (``order[v]`` = the input point at tour
    position v)."""

    sigma_nm: float
    offsets_nm: NDArray[np.float64]
    n_knots: int
    dof: int
    curve_nm: NDArray[np.float64]
    length_nm: float
    path: Optional[SmoothPath] = None
    order: Optional[NDArray[np.intp]] = None


@dataclass
class _PSpline:
    """The solved periodic P-spline of a tour-ordered polygon (private):
    the chord-length parameter ``u`` of its vertices, the tour's closed
    ``length``, the knot count, the design matrix, the inverse of the
    penalised normal matrix (None when singular), the fitted degrees of
    freedom (trace of the hat matrix) and the (n_knots, 2) coefficients
    (None when singular)."""

    u: NDArray[np.float64]
    length: float
    n_knots: int
    design: NDArray[np.float64]
    inv: Optional[NDArray[np.float64]]
    dof_fit: float
    coef: Optional[NDArray[np.float64]]


def _solve_pspline(c: NDArray[np.float64], knot_spacing_nm: float, penalty_rel: float, what: str) -> _PSpline:
    """The P-spline of the closed polygon ``c`` (K, 2) in tour order:
    n_knots = max(4, round(L / ``knot_spacing_nm``)) uniform knots in the
    chord-length parameter, least squares plus lambda = ``penalty_rel``
    x mean(diag(B'B)) times the squared second differences of the
    periodic coefficients. The arithmetic is exactly the one
    ``radial_scatter_nm`` had before the move (H5-B): the same
    expressions in the same order, so its numbers are bit for bit the
    same. A zero-length polygon raises ValueError."""
    edges = np.hypot(*(np.roll(c, -1, axis=0) - c).T)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    length = float(cum[-1])
    if not (length > 0.0):
        raise ValueError(f"{what}: the polygon has zero length")
    u = cum[:-1] / length
    n_knots = max(4, int(round(length / float(knot_spacing_nm))))
    design = _periodic_bspline_design(u, n_knots)
    gram = design.T @ design
    eye = np.eye(n_knots)
    second_diff = np.roll(eye, -1, axis=1) - 2.0 * eye + np.roll(eye, 1, axis=1)
    penalised = gram + float(penalty_rel) * float(np.mean(np.diag(gram))) * (second_diff.T @ second_diff)
    try:
        inv = np.linalg.inv(penalised)
    except np.linalg.LinAlgError:
        return _PSpline(u=u, length=length, n_knots=n_knots, design=design, inv=None, dof_fit=float("nan"), coef=None)
    dof_fit = float(np.trace(design @ inv @ design.T))
    coef = inv @ design.T @ c
    return _PSpline(u=u, length=length, n_knots=n_knots, design=design, inv=inv, dof_fit=dof_fit, coef=coef)


def _signed_offsets(c: NDArray[np.float64], ps: _PSpline) -> NDArray[np.float64]:
    """The signed residual of every vertex of ``c`` along the outward
    normal of the fitted curve at its parameter (+ outward by the tour's
    signed area), exact to first order in r / R: the residual vector
    projected on the normal of the finite-difference tangent."""
    assert ps.coef is not None
    fit = ps.design @ ps.coef
    du = _TANGENT_DU
    tang = (_periodic_bspline_design(ps.u + du, ps.n_knots) - _periodic_bspline_design(ps.u - du, ps.n_knots)) @ ps.coef / (2.0 * du)
    norm_t = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.where(norm_t > 0.0, norm_t, 1.0)
    sign = -math.copysign(1.0, _signed_area_nm2(c))
    normal_out = sign * np.column_stack([-tang[:, 1], tang[:, 0]])
    return np.asarray(np.einsum("ij,ij->i", c - fit, normal_out), dtype=np.float64)


def _tabulate(ps: _PSpline, n_samples: int) -> Tuple[NDArray[np.float64], float]:
    """The fitted curve at ``n_samples`` equally spaced parameters (a
    closed polygon, vertex 0's parameter first) and its closed length."""
    assert ps.coef is not None
    u_s = np.arange(int(n_samples), dtype=np.float64) / int(n_samples)
    curve = np.asarray(_periodic_bspline_design(u_s, ps.n_knots) @ ps.coef, dtype=np.float64)
    ce = np.hypot(*(np.roll(curve, -1, axis=0) - curve).T)
    return curve, float(ce.sum())


def radial_scatter_nm(
    contour_nm: NDArray[np.float64],
    *,
    knot_spacing_nm: float = MEMBRANE_KNOT_SPACING_NM,
    n_samples: int = MEMBRANE_CURVE_SAMPLES,
    min_dof: int = 3,
    penalty_rel: float = MEMBRANE_PENALTY_REL,
    max_length_change: float = 0.1,
) -> MembraneFit:
    """
    The radial scatter of a ring's cluster centroids about its membrane
    and the membrane itself, from the closed polygon ``contour_nm``
    (K, 2) through the centroids in tour order (``Ring.contour_nm``).

    The membrane is the periodic cubic B-spline with n_knots = max(4,
    round(L / ``knot_spacing_nm``)) uniform knots in the chord-length
    parameter u of the polygon, fitted to the vertices as a P-spline:
    least squares plus a second-difference penalty on the periodic
    coefficients, lambda = ``penalty_rel`` x mean(diag(B'B))
    (``_periodic_bspline_design``; a LINEAR smoother, unlike FITPACK's
    s-driven knot placement; the penalty only bites where a knot
    interval holds no vertex, see ``MEMBRANE_PENALTY_REL``), and the
    scatter is the residual sd along the outward normal corrected for
    the fitted degrees of freedom, the trace of the hat matrix B (B'B +
    lambda D'D)^-1 B': sigma^2 = sum r_v^2 / (K - dof_fit), exact for
    iid noise on a membrane the spline follows (0.96 of the truth on a
    noisy circle at K = 40; 3.6 nm on average on the exact ellipse of
    validate_columns_h3 at K = 40, 4.6 at K = 20; a 15 nm scatter
    recovered as 14.2-15.4 at K = 20-67; probes of 2026-09-25). The
    signed residual is the residual vector projected on the fitted
    curve's outward normal at u_v (+ outward by the tour's signed
    area), exact to first order in r / R. Not identifiable (sigma NaN,
    the offsets NaN, the curve the polygon itself) when K - dof_fit <
    ``min_dof`` or when the fitted curve's length differs from the
    polygon's by more than ``max_length_change`` (a curve that loops).
    Why not D-25's s = K smoothing spline: it passes within 1 nm rms of
    the vertices by construction, so a scatter about it is ~1 nm
    whatever the truth (``MEMBRANE_KNOT_SPACING_NM`` for the rest of
    the argument).

    H5-B moved this function here from ``tools.mps_simulate_axon``
    (which re-exports it) without changing a number: the fit is
    ``_solve_pspline``, the offsets ``_signed_offsets`` and the
    resampled curve ``_tabulate``, the same expressions in the same
    order (the arc-null validator, section 3, holds its numbers on two
    fixed inputs to 1e-9). ``path`` and ``order`` of the result stay
    None here.
    """
    c = np.asarray(contour_nm, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 3:
        raise ValueError(f"radial_scatter_nm: expected a (K >= 3, 2) polygon, got shape {c.shape}")
    k = int(c.shape[0])
    ps = _solve_pspline(c, knot_spacing_nm, penalty_rel, "radial_scatter_nm")
    length = ps.length
    n_knots = ps.n_knots
    if ps.inv is None or ps.coef is None:
        return MembraneFit(sigma_nm=float("nan"), offsets_nm=np.full(k, np.nan), n_knots=n_knots, dof=0,
                           curve_nm=c.copy(), length_nm=length)
    dof_fit = ps.dof_fit
    dof = int(math.floor(k - dof_fit))
    not_identifiable = MembraneFit(sigma_nm=float("nan"), offsets_nm=np.full(k, np.nan), n_knots=n_knots, dof=dof,
                                   curve_nm=c.copy(), length_nm=length)
    if k - dof_fit < float(min_dof):
        return not_identifiable
    offsets = _signed_offsets(c, ps)
    sigma = math.sqrt(max(float(np.sum(offsets ** 2)) / (k - dof_fit), 0.0))
    curve, curve_length = _tabulate(ps, int(n_samples))
    if not np.isfinite(curve).all() or abs(curve_length - length) > float(max_length_change) * length:
        return not_identifiable
    return MembraneFit(sigma_nm=sigma, offsets_nm=np.asarray(offsets, dtype=np.float64), n_knots=n_knots, dof=dof,
                       curve_nm=curve, length_nm=curve_length)


def fit_membrane(
    points_nm: NDArray[np.float64],
    *,
    knot_spacing_nm: float = MEMBRANE_NULL_KNOT_SPACING_NM,
    penalty_rel: float = MEMBRANE_PENALTY_REL,
    n_samples: int = MEMBRANE_CURVE_SAMPLES,
    min_dof: int = 3,
    max_length_change: float = MEMBRANE_MAX_LENGTH_CHANGE,
) -> MembraneFit:
    """
    The P-spline membrane of ``points_nm`` (n >= 3, 2) given in ANY
    order -- one ring's centroids, or the centroids of several rings of
    one axon pooled -- tabulated as a ``SmoothPath`` (``MembraneFit.
    path``) that ``tools.mps_matching`` walks for ``null_kind=
    "pooled_offset"`` (H5B_SPEC S2: the leave-ring-out membrane of ring
    b is this fit on every OTHER ring's centroids; the rationale is in
    the module docstring and in ``eclipse_test``).

    The fit is ``radial_scatter_nm``'s (``_solve_pspline``: knots every
    ``knot_spacing_nm`` of the tour, penalty ``penalty_rel``) with a
    FINER default knot spacing, ``MEMBRANE_NULL_KNOT_SPACING_NM`` = 400
    nm against the scatter estimate's 600 (the argument and the
    measured trade-off are next to that constant: the null needs every
    bend of the membrane the rings share to be in the curve, and 600 nm
    knots miss a curvature radius of 300 nm by 16 nm). What also
    differs is the parameterisation: the points carry no
    tour, so one is built with ``tools.mps_geometry.reconstruct_perimeter``
    (polar order about the centroid, then 2-opt from ONE start:
    ``all_starts`` costs 4-5x more -- 220 vs 60 ms on 80 pooled points,
    probe of 2026-09-25 -- and buys a shorter tour, which matters for a
    perimeter length but not for a parameter the 600 nm knots smooth
    over). Pooled centroids of two rings that scatter about the
    membrane interleave along it and the tour zigzags between them
    (~11 % longer than the membrane at 30 nm scatter, K = 40); the
    chord-length parameter is then slightly non-uniform along the
    membrane and the knot count follows the tour's length (15 instead
    of 13 knots there), neither of which moves the fitted curve by more
    than the scatter of a knot interval's ~3-6 points, the estimation
    error the pooled null lives with anyway. The dense table is the one
    ``tools.mps_matching.smooth_closed_path`` makes: max(``PATH_MIN_SAMPLES``,
    ceil(L / ``PATH_STEP_NM``)) equally spaced parameters, row 0 the
    tour's first point (the origin of the arc coordinate and of the
    null's reflection), ``vertex_row`` the row nearest each INPUT point
    (bookkeeping: the points are not on the curve; their arc is their
    projection).

    Returns
    -------
    MembraneFit
        ``path`` the dense table (``length_nm`` its closed length),
        ``order`` the tour, ``curve_nm`` the curve at ``n_samples``
        parameters, ``offsets_nm`` the signed residuals in INPUT order
        and ``sigma_nm`` the dof-corrected scatter (NaN with fewer than
        ``min_dof`` residual degrees of freedom, as ``radial_scatter_nm``).

    Raises
    ------
    ValueError
        Fewer than three points, a non-finite one, a singular fit, a
        non-finite curve or one whose length differs from the tour's by
        more than ``max_length_change`` (a curve that loops or
        collapses): a membrane that cannot be fitted is no curve for a
        null to walk, so the caller hears about it instead of getting
        the polygon back.
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError(f"fit_membrane: expected (n >= 3, 2) points, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("fit_membrane: non-finite coordinate")
    k = int(pts.shape[0])
    order = np.asarray(reconstruct_perimeter(pts, refine=True, all_starts=False).order, dtype=np.intp).ravel()
    if order.shape != (k,) or sorted(order.tolist()) != list(range(k)):
        raise ValueError("fit_membrane: the tour is not a permutation of the points")
    c = pts[order]
    ps = _solve_pspline(c, knot_spacing_nm, penalty_rel, "fit_membrane")
    if ps.inv is None or ps.coef is None:
        raise ValueError(f"fit_membrane: singular penalised normal matrix ({k} points, {ps.n_knots} knots)")
    dof = int(math.floor(k - ps.dof_fit))
    offsets_tour = _signed_offsets(c, ps)
    offsets = np.empty(k, dtype=np.float64)
    offsets[order] = offsets_tour
    if k - ps.dof_fit < float(min_dof):
        sigma = float("nan")
        offsets = np.full(k, np.nan, dtype=np.float64)
    else:
        sigma = math.sqrt(max(float(np.sum(offsets_tour ** 2)) / (k - ps.dof_fit), 0.0))
    n_dense = max(int(PATH_MIN_SAMPLES), int(math.ceil(ps.length / float(PATH_STEP_NM))))
    dense, dense_length = _tabulate(ps, n_dense)
    if not np.isfinite(dense).all() or not (dense_length > 0.0):
        raise ValueError("fit_membrane: the fitted curve is not finite")
    if abs(dense_length - ps.length) > float(max_length_change) * ps.length:
        raise ValueError(f"fit_membrane: the fitted curve's length {dense_length:.0f} nm differs from the tour's "
                         f"{ps.length:.0f} nm by more than {max_length_change:g} of it (a looping or collapsed curve)")
    edges = np.hypot(*(np.roll(dense, -1, axis=0) - dense).T)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    if int(np.count_nonzero(edges > 0.0)) < 3:
        raise ValueError("fit_membrane: the fitted curve is degenerate")
    _, nearest = cKDTree(dense).query(pts, k=1)
    path = SmoothPath(points_nm=dense, vertex_row=np.asarray(nearest, dtype=np.intp).reshape(-1),
                      edges_nm=edges, cum_nm=cum, length_nm=float(cum[-1]))
    curve, _curve_length = _tabulate(ps, int(n_samples))
    return MembraneFit(sigma_nm=sigma, offsets_nm=offsets, n_knots=ps.n_knots, dof=dof, curve_nm=curve,
                       length_nm=float(cum[-1]), path=path, order=order)


def smooth_polygon_membrane(
    contour_nm: NDArray[np.float64],
    *,
    knot_spacing_nm: float = MEMBRANE_KNOT_SPACING_NM,
    penalty_rel: float = MEMBRANE_PENALTY_REL,
    n_samples: Optional[int] = None,
    max_length_change: float = MEMBRANE_MAX_LENGTH_CHANGE,
) -> NDArray[np.float64]:
    """
    The P-spline membrane of an ORDERED closed polygon ``contour_nm``
    (K0 >= 4, 2), resampled at ``n_samples`` equally spaced parameters
    (default max(``MEMBRANE_CURVE_SAMPLES``, K0)): the fit of
    ``_solve_pspline`` (knots every ``knot_spacing_nm`` of the polygon's
    chord length, penalty ``penalty_rel``) in the polygon's OWN vertex
    order -- no tour is rebuilt, unlike ``fit_membrane``, because a
    contour is already a tour and ``reconstruct_perimeter`` on a
    concave one may re-order it.

    Why it exists (statistics review of 2026-09-25, finding on a
    contour library of real rings):
    the library entries were written by the H5 ``measure`` step as the
    s = K smoothing spline of each reference ring's centroid polygon,
    which passes within ~1 nm of the centroids, so they carry the real
    rings' radial centroid scatter as if it were membrane structure --
    rough below the knot scale and longer than their 600 nm P-spline,
    unlike the harness's Fourier contour and ellipse (research probes).
    ``simulate_axon`` then adds ANOTHER independent scatter
    (``radial_offset_sd_nm``, measured about a 600 nm P-spline
    membrane), so every simulated ring shares the library's roughness
    and every membrane null and the arc null read it as columns
    (the noleak cells of the review were liberal, with the true smooth
    contour of the simulator exact). The
    consistent membrane is the one the scatter was measured against:
    ``SimConfig.contour_smoothing_knot_nm`` makes ``simulate_axon``
    replace the resolved contour by this curve at that knot spacing
    (600 nm = ``MEMBRANE_KNOT_SPACING_NM`` in ``power_grid_v2.yaml`` and
    ``simnull``'s default).

    Raises
    ------
    ValueError
        Fewer than four vertices, a non-finite one, a singular fit, a
        non-finite or degenerate curve, or a curve whose length differs
        from the polygon's by more than ``max_length_change`` of it (the
        wiggliest entry of the research library shrinks by well under
        half at 600 nm, so the default 0.5 passes every entry and still
        catches a collapse).
    """
    c = np.asarray(contour_nm, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 4:
        raise ValueError(f"smooth_polygon_membrane: expected a (K0 >= 4, 2) polygon, got shape {c.shape}")
    if not np.isfinite(c).all():
        raise ValueError("smooth_polygon_membrane: non-finite coordinate")
    if not (float(knot_spacing_nm) > 0.0):
        raise ValueError(f"smooth_polygon_membrane: knot_spacing_nm must be positive, got {knot_spacing_nm}")
    ps = _solve_pspline(c, float(knot_spacing_nm), float(penalty_rel), "smooth_polygon_membrane")
    if ps.inv is None or ps.coef is None:
        raise ValueError(f"smooth_polygon_membrane: singular penalised normal matrix ({c.shape[0]} vertices, {ps.n_knots} knots)")
    n_out = int(n_samples) if n_samples is not None else max(int(MEMBRANE_CURVE_SAMPLES), int(c.shape[0]))
    if n_out < 4:
        raise ValueError(f"smooth_polygon_membrane: n_samples must be >= 4, got {n_samples}")
    curve, curve_length = _tabulate(ps, n_out)
    if not np.isfinite(curve).all() or not (curve_length > 0.0):
        raise ValueError("smooth_polygon_membrane: the fitted curve is not finite")
    if abs(curve_length - ps.length) > float(max_length_change) * ps.length:
        raise ValueError(f"smooth_polygon_membrane: the fitted curve's length {curve_length:.0f} nm differs from the polygon's "
                         f"{ps.length:.0f} nm by more than {max_length_change:g} of it (a looping or collapsed curve)")
    edges = np.hypot(*(np.roll(curve, -1, axis=0) - curve).T)
    if int(np.count_nonzero(edges > 0.0)) < 3:
        raise ValueError("smooth_polygon_membrane: the fitted curve is degenerate")
    return np.asarray(curve, dtype=np.float64)


# ============================================================================
# H5-C: the membrane from the localizations
# ============================================================================

@dataclass
class LocalizationMembrane:
    """
    The membrane of one axon (or of a subset of its rings) fitted to the
    LOCALIZATIONS (``fit_membrane_from_localizations``): ``path`` the
    tabulated curve (a ``SmoothPath``: row 0 is the point at parameter
    0, the initial curve's origin, so that arcs are comparable between
    fits on the same initial curve; ``vertex_row`` is empty because the
    localizations are not vertices of the curve) and ``length_nm`` its
    closed length. ``n_points`` localizations were given, ``n_used``
    kept a non-zero weight after the robust step (the subsample when
    ``max_points`` cut the input). ``knot_spacing_nm``, ``n_knots``,
    ``penalty_rel`` and ``iterations`` are the fit's settings,
    ``dof_fit`` the trace of its hat matrix (the fitted degrees of
    freedom, <= n_knots), ``residual_sd_nm`` the robust sd (1.4826 x
    MAD) of the kept localizations' signed offsets from the final curve
    -- the membrane's thickness as the localizations see it: the
    cluster scatter, the cluster width and the linkage error together
    -- and ``initial_curve`` says where the parameterisation came from
    ("given": the caller's path; "pooled_centroids": the pooled P-spline
    of the rings' centroids, ``localization_membrane_of_rings``).
    ``n_self_crossings`` counts the crossings of the final curve with
    itself (``self_crossings``: 0 for any valid membrane) and
    ``initial_order`` records how the pooled centroids of the initial
    curve were ordered (``INITIAL_ORDERS``; "" when the path was given).
    """

    path: SmoothPath
    length_nm: float
    n_points: int
    n_used: int
    knot_spacing_nm: float
    n_knots: int
    penalty_rel: float
    iterations: int
    dof_fit: float
    residual_sd_nm: float
    initial_curve: str
    warnings: List[str] = field(default_factory=list)
    # Review of 2026-09-26: the number of self-crossings of the tabulated
    # curve (``self_crossings``; a membrane cannot cross itself, so > 0
    # marks a failed fit -- the lumen-crossing loops of the first initial
    # curve -- and is warned about) and how the initial curve's centroids
    # were ordered (``INITIAL_ORDERS``; "" for a given initial path).
    n_self_crossings: int = 0
    initial_order: str = ""


def _outward_sign(points_nm: NDArray[np.float64]) -> float:
    """The ``outward_sign`` of ``tools.mps_unroll.project_on_path`` for a
    closed table: -sign of its signed area (counter-clockwise -> -1)."""
    area = _signed_area_nm2(np.asarray(points_nm, dtype=np.float64))
    if area == 0.0:
        raise ValueError("the closed curve has zero area (degenerate)")
    return -math.copysign(1.0, area)


def _project(path: SmoothPath, xy: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(arc, signed offset + outward) of ``xy`` (n, 2) on ``path`` by
    ``tools.mps_unroll.project_on_path`` (vectorised; validated in H4).
    Imported at the call: ``tools.mps_unroll`` imports this module for
    the localization membrane, and the reverse top-level import would
    be circular (the same arrangement as ``tools.mps_matching``'s
    deferred import of ``fit_membrane``)."""
    from tools.mps_unroll import project_on_path  # deferred: mps_unroll imports this module
    return project_on_path(path, xy, outward_sign=_outward_sign(path.points_nm))


def _solve_weighted_pspline(
    u: NDArray[np.float64],
    xy: NDArray[np.float64],
    w: NDArray[np.float64],
    n_knots: int,
    penalty_rel: float,
    what: str,
) -> Tuple[NDArray[np.float64], float]:
    """
    The periodic P-spline x(u), y(u) with ``n_knots`` uniform knots
    fitted to the points ``xy`` (n, 2) at parameters ``u`` (n,) with
    weights ``w`` (n,): weighted least squares plus the second-difference
    penalty of ``_solve_pspline`` generalised to n points -- lambda =
    ``penalty_rel`` x mean(diag(B'WB)), the same relative weight, so
    that it bites only where the weighted data leave a coefficient
    unconstrained. Returns the (n_knots, 2) coefficients and the trace
    of the hat matrix B (B'WB + lambda D'D)^-1 B'W (the fitted degrees
    of freedom, computed as trace((B'WB + lambda D'D)^-1 B'WB) on the
    n_knots x n_knots matrices, never on the n x n hat matrix).
    ValueError when the penalised normal matrix is singular or the
    solution is not finite.
    """
    design = _periodic_bspline_design(u, n_knots)
    weighted = design * w[:, None]
    gram = design.T @ weighted
    eye = np.eye(n_knots)
    second_diff = np.roll(eye, -1, axis=1) - 2.0 * eye + np.roll(eye, 1, axis=1)
    penalised = gram + float(penalty_rel) * float(np.mean(np.diag(gram))) * (second_diff.T @ second_diff)
    try:
        inv = np.linalg.inv(penalised)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{what}: singular penalised normal matrix ({u.size} points, {n_knots} knots)") from exc
    coef = np.asarray(inv @ (weighted.T @ xy), dtype=np.float64)
    if not np.isfinite(coef).all():
        raise ValueError(f"{what}: the fitted coefficients are not finite ({u.size} points, {n_knots} knots)")
    return coef, float(np.trace(inv @ gram))


def _tabulate_path(coef: NDArray[np.float64], n_knots: int, what: str) -> SmoothPath:
    """The fitted curve as a ``SmoothPath``: the dense table of
    ``tools.mps_matching.smooth_closed_path`` (max(``PATH_MIN_SAMPLES``,
    ceil(L / ``PATH_STEP_NM``)) equally spaced parameters, row 0 at
    parameter 0), an empty ``vertex_row``. The row count is set from
    the length of a 200-sample preview of the curve so that the table's
    chord is ~``PATH_STEP_NM`` whatever the length. ValueError on a
    non-finite or degenerate curve."""
    u_0 = np.arange(int(MEMBRANE_CURVE_SAMPLES), dtype=np.float64) / int(MEMBRANE_CURVE_SAMPLES)
    preview = np.asarray(_periodic_bspline_design(u_0, n_knots) @ coef, dtype=np.float64)
    length_0 = float(np.hypot(*(np.roll(preview, -1, axis=0) - preview).T).sum()) if np.isfinite(preview).all() else float("nan")
    if not (math.isfinite(length_0) and length_0 > 0.0):
        raise ValueError(f"{what}: the fitted curve is not finite or has zero length")
    n_dense = max(int(PATH_MIN_SAMPLES), int(math.ceil(length_0 / float(PATH_STEP_NM))))
    u_s = np.arange(n_dense, dtype=np.float64) / n_dense
    dense = np.asarray(_periodic_bspline_design(u_s, n_knots) @ coef, dtype=np.float64)
    if not np.isfinite(dense).all():
        raise ValueError(f"{what}: the fitted curve is not finite")
    edges = np.hypot(*(np.roll(dense, -1, axis=0) - dense).T)
    if int(np.count_nonzero(edges > 0.0)) < 3:
        raise ValueError(f"{what}: the fitted curve is degenerate (fewer than three distinct rows)")
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=dense, vertex_row=np.zeros(0, dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def fit_membrane_from_localizations(
    x_nm: NDArray[np.float64],
    y_nm: NDArray[np.float64],
    *,
    initial_path: SmoothPath,
    knot_spacing_nm: float = MEMBRANE_KNOT_SPACING_NM,
    penalty_rel: float = MEMBRANE_PENALTY_REL,
    iterations: int = MEMBRANE_LOCALIZATION_ITERATIONS,
    robust_c: float = MEMBRANE_ROBUST_C,
    max_points: Optional[int] = None,
    random_seed: int = 0,
    max_length_change: float = MEMBRANE_MAX_LENGTH_CHANGE,
    parameter_knot_spacing_nm: Optional[float] = MEMBRANE_PARAMETER_KNOT_SPACING_NM,
) -> LocalizationMembrane:
    """
    The membrane of an axon as the periodic P-spline fitted to its
    LOCALIZATIONS (H5C_SPEC S1; D-29c), given an initial closed curve
    within tens of nm of it (the pooled centroid P-spline of
    ``localization_membrane_of_rings``, or any ``SmoothPath``).

    Algorithm and why each step
    ---------------------------
    1. Parameter: every localization gets u = (arc of its projection on
       ``initial_path``) / L_0 (``tools.mps_unroll.project_on_path``,
       vectorised). A curve through a point cloud needs a parameter per
       point, and the projection on a nearby curve is the natural one:
       it is exact when the curve is the membrane and moves only to
       second order, ~(r - delta) delta', when the curve is delta off
       it (the argument at ``tools.mps_unroll.ARC_REFERENCE_CURVES``).
    2. Fit: the periodic cubic P-spline x(u), y(u) with n_knots = max(4,
       round(L_0 / ``knot_spacing_nm``)) uniform knots (the count is
       fixed by the INITIAL length so that repeated fits on one axon
       share a basis), weighted least squares plus the second-difference
       penalty of ``_solve_pspline`` at the same relative weight
       (``_solve_weighted_pspline``); the knots are the scatter
       estimate's 600 nm (``MEMBRANE_KNOT_SPACING_NM``: a cubic spline
       with 600 nm knots follows a curvature radius >= 600 nm to < 1
       nm, and structure below that scale counts as scatter). With
       many hundreds of localizations per knot interval the curve's
       own noise is ~sd / sqrt(n), a few nm, and the penalty is
       idle: the fit is the data's.
    3. Robust step: the localizations whose signed offset from the
       curve lies farther than ``robust_c`` x (1.4826 x MAD) from the
       median offset get weight ZERO (a drop; ``MEMBRANE_ROBUST_C`` says
       why not a Huber weight): the uniform background of the pick and
       the tails of the leak and of the cluster scatter must not pull
       the curve. The rule is applied FIRST to the offsets from the
       initial curve, so that even the first fit is weighted: an
       unweighted first fit is captured by a dense background blob
       where the membrane is sparse (the simulator's background fills
       the contour's bounding box, so outside a concavity there is
       more background than membrane), and a robust step computed
       from the captured curve then drops the membrane's own
       localizations there and keeps the blob -- large excursions
       on a slot-shaped library contour, none once the initial curve's
       residuals select the points (probe of 2026-09-26).
    4. Re-parameterise on the fitted curve and refit with the weights
       recomputed from its offsets; ``iterations`` fits in all. The
       fits BEFORE the last one, whose only purpose is the
       parameterisation, use finer knots (``parameter_knot_spacing_nm``,
       ``MEMBRANE_PARAMETER_KNOT_SPACING_NM`` = 300 nm; None = the
       membrane's own, the literal re-projection): re-projecting on a
       curve at the membrane's spacing is a contracting fixed point
       wherever that basis bridges a bay (the argument and the
       numbers at the constant). After the last fit the localizations
       are projected on the final curve once more, the robust rule is
       applied to those offsets for the record (``n_used``) and the
       robust sd of the kept offsets is ``residual_sd_nm``.
    5. Tabulate as a ``SmoothPath`` (``_tabulate_path``: the dense table
       of ``smooth_closed_path``, row 0 at parameter 0). Because u = 0
       is the initial curve's origin and each refit keeps the previous
       curve's parameter, the origin of the arc coordinate stays the
       initial curve's origin (within the tangential drift of the fit,
       a few nm; the harness checks 20 nm) and its orientation, so arcs
       of different fits on one initial curve are comparable (the
       harness: arcs of one and two iterations within 3 nm).

    ``max_points`` (None = all): when the axon has more localizations,
    a random subsample of that size, drawn without replacement from
    ``default_rng(SeedSequence(random_seed))``, is fitted (cost
    control; ``n_points`` still counts the input, ``n_used`` the kept
    subsample). The full fit of 21 000 points takes ~0.3 s, so the
    default is all.

    Raises
    ------
    ValueError
        x and y of different lengths, a non-finite coordinate, a
        negative or non-finite ``penalty_rel``, fewer
        than ``MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT`` x n_knots
        points (after the subsample, or kept by the robust step), an
        initial path without a positive length, localizations that lie
        systematically off the initial curve (median offset beyond
        ``robust_c`` robust sd: the curve does not describe them), a
        singular fit, a non-finite or degenerate curve, or a fitted
        curve whose closed length differs from the initial curve's by
        more than ``max_length_change`` of it
        (``MEMBRANE_MAX_LENGTH_CHANGE``: a curve that looped, collapsed
        or ran off to points that are not the membrane the initial
        curve described).
    """
    x = np.asarray(x_nm, dtype=np.float64).reshape(-1)
    y = np.asarray(y_nm, dtype=np.float64).reshape(-1)
    what = "fit_membrane_from_localizations"
    if x.shape != y.shape:
        raise ValueError(f"{what}: x and y have different lengths ({x.size} and {y.size})")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"{what}: non-finite coordinate")
    length_0 = float(initial_path.length_nm)
    if not (math.isfinite(length_0) and length_0 > 0.0) or np.asarray(initial_path.points_nm).shape[0] < 3:
        raise ValueError(f"{what}: the initial path needs at least three rows and a positive length")
    if not (float(knot_spacing_nm) > 0.0):
        raise ValueError(f"{what}: knot_spacing_nm must be positive, got {knot_spacing_nm}")
    n_iter = int(iterations)
    if n_iter < 1:
        raise ValueError(f"{what}: iterations must be >= 1, got {iterations}")
    if not (float(robust_c) > 0.0):
        raise ValueError(f"{what}: robust_c must be positive, got {robust_c}")
    # Review of 2026-09-26: a negative penalty makes the penalised normal matrix indefinite and the fit returned a
    # curve of the wrong length with dof_fit ~1 and no error (penalty -0.5 on 20 000 ellipse localizations: L 8004
    # against 7933, no warning); NaN / inf were caught only as "coefficients not finite". Zero is allowed (plain
    # weighted least squares: every knot interval of an axon holds hundreds of localizations).
    if not (math.isfinite(float(penalty_rel)) and float(penalty_rel) >= 0.0):
        raise ValueError(f"{what}: penalty_rel must be finite and >= 0, got {penalty_rel}")
    n_points = int(x.size)
    warnings_: List[str] = []
    xy = np.column_stack([x, y])
    if max_points is not None:
        cap = int(max_points)
        if cap < 1:
            raise ValueError(f"{what}: max_points must be >= 1 or None, got {max_points}")
        if n_points > cap:
            rng = np.random.default_rng(np.random.SeedSequence(int(random_seed)))
            keep = np.sort(rng.choice(n_points, size=cap, replace=False))
            xy = xy[keep]
            warnings_.append(f"{cap} of {n_points} localizations fitted (max_points; seed {int(random_seed)})")
    n_fit = int(xy.shape[0])
    n_knots = max(4, int(round(length_0 / float(knot_spacing_nm))))
    if parameter_knot_spacing_nm is None:
        n_knots_param = n_knots
    else:
        if not (float(parameter_knot_spacing_nm) > 0.0):
            raise ValueError(f"{what}: parameter_knot_spacing_nm must be positive or None, got {parameter_knot_spacing_nm}")
        n_knots_param = max(n_knots, int(round(length_0 / float(parameter_knot_spacing_nm))))
    if n_fit < MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT * n_knots_param:
        raise ValueError(f"{what}: {n_fit} localizations for {n_knots_param} knots (fewer than "
                         f"{MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT} per knot)")
    def robust_weights(offsets_nm: NDArray[np.float64], final: bool) -> NDArray[np.float64]:
        med = float(np.median(offsets_nm))
        scale = _MAD_TO_SD * float(np.median(np.abs(offsets_nm - med)))
        if scale > 0.0:
            return np.where(np.abs(offsets_nm - med) <= float(robust_c) * scale, 1.0, 0.0)
        if final:
            warnings_.append("the offsets' MAD is zero: no localization dropped by the robust step")
        return np.ones(offsets_nm.size, dtype=np.float64)

    s0, r0 = _project(initial_path, xy)
    u = s0 / length_0
    # The localizations must lie ON the initial curve up to the robust scale: a cloud that sits systematically off it
    # (its median offset beyond robust_c MAD-sd of the offsets) is not the membrane that curve describes -- the
    # robust rule below would keep the cloud as mutually consistent points and the fit would return a curve of
    # plausible length through the wrong points (localizations on a 250 nm circle inside an ellipse, harness).
    med_0 = float(np.median(r0))
    scale_0 = _MAD_TO_SD * float(np.median(np.abs(r0 - med_0)))
    if abs(med_0) > float(robust_c) * max(scale_0, 1e-9):
        raise ValueError(f"{what}: the localizations lie systematically off the initial curve (median offset {med_0:.0f} nm, "
                         f"robust sd {scale_0:.0f} nm): that curve does not describe them")
    # The robust step starts from the INITIAL curve's residuals: the pooled centroid curve is within tens of nm of the
    # membrane, so the far background is out before any fit (MEMBRANE_ROBUST_C: an unweighted first fit is captured
    # by a dense background blob where the membrane is sparse, and the robust step then locks the capture in).
    w = robust_weights(r0, False)
    path: Optional[SmoothPath] = None
    dof_fit = float("nan")
    offsets = np.zeros(n_fit, dtype=np.float64)
    for it in range(n_iter):
        k_it = n_knots if it == n_iter - 1 else n_knots_param   # the parameterisation fits are finer, the last is the membrane
        if int(np.count_nonzero(w)) < MEMBRANE_LOCALIZATION_MIN_POINTS_PER_KNOT * k_it:
            raise ValueError(f"{what}: the robust step kept {int(np.count_nonzero(w))} localizations for {k_it} knots")
        coef, dof_fit = _solve_weighted_pspline(u, xy, w, k_it, float(penalty_rel), what)
        path = _tabulate_path(coef, k_it, what)
        if abs(path.length_nm - length_0) > float(max_length_change) * length_0:
            raise ValueError(f"{what}: the fitted curve's length {path.length_nm:.0f} nm differs from the initial curve's "
                             f"{length_0:.0f} nm by more than {max_length_change:g} of it (a looping, collapsed or runaway curve)")
        s, offsets = _project(path, xy)
        u = s / float(path.length_nm)
        w = robust_weights(offsets, it == n_iter - 1)
    assert path is not None
    if n_fit >= 3 and int(np.count_nonzero(w)) < 3:
        raise ValueError(f"{what}: the robust step kept fewer than three localizations")
    kept = w > 0.0
    n_used = int(np.count_nonzero(kept))
    r_kept = offsets[kept]
    med_kept = float(np.median(r_kept))
    residual_sd = _MAD_TO_SD * float(np.median(np.abs(r_kept - med_kept)))
    dropped = n_fit - n_used
    if dropped > 0.2 * n_fit:
        warnings_.append(f"the robust step dropped {dropped} of {n_fit} localizations ({dropped / n_fit:.0%}): the initial "
                         "curve may be far from the membrane, or the pick holds much background")
    if abs(path.length_nm - length_0) > 0.1 * length_0:
        warnings_.append(f"the fitted curve's length {path.length_nm:.0f} nm differs from the initial curve's {length_0:.0f} nm "
                         f"by {abs(path.length_nm - length_0) / length_0:.0%}")
    n_cross = self_crossings(path)
    if n_cross:
        warnings_.append(f"the fitted curve crosses itself {n_cross} time(s): not a membrane (a loop, typically across the lumen "
                         "from a badly ordered initial curve); arcs on it are not a stationary coordinate")
    return LocalizationMembrane(
        path=path, length_nm=float(path.length_nm), n_points=n_points, n_used=n_used, knot_spacing_nm=float(knot_spacing_nm),
        n_knots=n_knots, penalty_rel=float(penalty_rel), iterations=n_iter, dof_fit=dof_fit, residual_sd_nm=residual_sd,
        initial_curve="given", warnings=warnings_, n_self_crossings=int(n_cross))


def self_crossings(path: SmoothPath, *, max_vertices: int = _CROSSING_MAX_VERTICES) -> int:
    """
    The number of proper crossings of the closed curve ``path`` with
    itself: the table subsampled to at most ``max_vertices`` vertices
    (every ``ceil(N / max_vertices)``-th row), every pair of
    non-adjacent edges tested by the orientation predicate (strict
    signs: touching or collinear edges do not count). A membrane is a
    simple closed curve, so any crossing marks a failed fit; the
    loops the review of 2026-09-26 found span hundreds of nm and
    survive the subsample. O(max_vertices^2) vectorised (~10 ms).
    """
    p = np.asarray(path.points_nm, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 4:
        return 0
    step = max(1, int(math.ceil(p.shape[0] / float(max_vertices))))
    a = p[::step]
    n = int(a.shape[0])
    if n < 4:
        return 0
    b = np.roll(a, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]

    def orient(p1: NDArray[np.float64], p2: NDArray[np.float64], p3: NDArray[np.float64]) -> NDArray[np.float64]:
        return (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])

    d1 = orient(a[i], b[i], a[j])
    d2 = orient(a[i], b[i], b[j])
    d3 = orient(a[j], b[j], a[i])
    d4 = orient(a[j], b[j], b[i])
    return int(np.count_nonzero((d1 * d2 < 0.0) & (d3 * d4 < 0.0)))


def _tour_length(order: NDArray[np.intp], dist: NDArray[np.float64]) -> float:
    """Closed length of the tour ``order`` under the distance matrix."""
    return float(np.sum(dist[order, np.roll(order, -1)]))


def _two_opt(order: NDArray[np.intp], dist: NDArray[np.float64], max_passes: int) -> NDArray[np.intp]:
    """
    2-opt on a closed tour: for every edge (a, b) = (o[i], o[i+1]) the
    best exchange with a later edge (c, d) = (o[j], o[j+1]) -- reverse
    o[i+1 .. j] when d(a, c) + d(b, d) < d(a, b) + d(c, d) -- applied at
    once (first-improvement per i, best j), until a pass improves
    nothing or ``max_passes``. Removes every crossing of a Euclidean
    tour. Deterministic.
    """
    o = np.asarray(order, dtype=np.intp).copy()
    n = int(o.size)
    if n < 4:
        return np.asarray(o, dtype=np.intp)
    for _ in range(int(max_passes)):
        improved = False
        for ii in range(n - 2):
            a, b = int(o[ii]), int(o[ii + 1])
            jj = np.arange(ii + 2, n)
            c = o[jj]
            d = o[(jj + 1) % n]
            delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
            if ii == 0:
                delta[-1] = 0.0   # (c, d) = (o[n-1], o[0]) shares the vertex a: not an exchange
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                j = int(jj[k])
                o[ii + 1:j + 1] = o[ii + 1:j + 1][::-1].copy()
                improved = True
        if not improved:
            break
    return np.asarray(o, dtype=np.intp)


def _or_opt(order: NDArray[np.intp], dist: NDArray[np.float64], max_passes: int, max_segment: int = 3) -> NDArray[np.intp]:
    """
    Or-opt on a closed tour: every run of 1, 2 or 3 consecutive vertices
    is moved, forwards or reversed, to the edge where inserting it costs
    least, when that saves length; until a pass improves nothing or
    ``max_passes``. What it adds to 2-opt: a centroid of one ring left
    on the wrong side of a narrow slot (the tour zigzags across the
    slot) is re-inserted on its own side, a move 2-opt cannot make in
    one exchange. Deterministic.
    """
    o = np.asarray(order, dtype=np.intp).copy()
    n = int(o.size)
    if n < 5:
        return np.asarray(o, dtype=np.intp)
    for _ in range(int(max_passes)):
        improved = False
        for seg_len in range(1, int(max_segment) + 1):
            if seg_len > n - 3:
                break
            i = 0
            while i < n:
                idx = [(i + t) % n for t in range(seg_len)]
                prev = int(o[(i - 1) % n])
                nxt = int(o[(i + seg_len) % n])
                seg = o[idx]
                gain = dist[prev, seg[0]] + dist[seg[-1], nxt] - dist[prev, nxt]
                rest = np.delete(o, idx)
                nb = np.roll(rest, -1)
                ins_f = dist[rest, seg[0]] + dist[seg[-1], nb] - dist[rest, nb]
                ins_r = dist[rest, seg[-1]] + dist[seg[0], nb] - dist[rest, nb]
                kf, kr = int(np.argmin(ins_f)), int(np.argmin(ins_r))
                if ins_f[kf] <= ins_r[kr]:
                    cost, k, moved = float(ins_f[kf]), kf, seg
                else:
                    cost, k, moved = float(ins_r[kr]), kr, seg[::-1]
                if cost < float(gain) - 1e-9:
                    o = np.concatenate([rest[:k + 1], moved, rest[k + 1:]]).astype(np.intp)
                    improved = True
                i += 1
        if not improved:
            break
    return np.asarray(o, dtype=np.intp)


def _improved_tour(order: NDArray[np.intp], dist: NDArray[np.float64], max_passes: int) -> NDArray[np.intp]:
    """2-opt, Or-opt, 2-opt from ``order`` (the local search of
    ``pooled_centroid_tour``)."""
    o = _two_opt(order, dist, max_passes)
    o = _or_opt(o, dist, max_passes)
    return _two_opt(o, dist, max_passes)


def _linear_closed_path(polygon_nm: NDArray[np.float64]) -> SmoothPath:
    """The closed POLYGON (no spline) tabulated as a ``SmoothPath`` every
    ~``PATH_STEP_NM`` along each edge (zero-length edges dropped), so
    that ``project_on_path`` projects on the polygon itself: a
    projection on a ring's tour that cannot overshoot between vertices
    the way an interpolating spline through scattered centroids does."""
    poly = np.asarray(polygon_nm, dtype=np.float64)
    nxt = np.roll(poly, -1, axis=0)
    seg = np.hypot(*(nxt - poly).T)
    parts = []
    for p0, p1, length in zip(poly, nxt, seg):
        if not (length > 0.0):
            continue
        m = max(1, int(math.ceil(float(length) / float(PATH_STEP_NM))))
        t = np.arange(m, dtype=np.float64) / m
        parts.append(p0 + t[:, None] * (p1 - p0))
    if len(parts) < 3:
        raise ValueError("the polygon has fewer than three distinct vertices")
    dense = np.vstack(parts)
    edges = np.hypot(*(np.roll(dense, -1, axis=0) - dense).T)
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return SmoothPath(points_nm=dense, vertex_row=np.zeros(0, dtype=np.intp), edges_nm=edges, cum_nm=cum,
                      length_nm=float(cum[-1]))


def _points_or_raise(points_nm: NDArray[np.float64], what: str) -> NDArray[np.float64]:
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{what}: expected (K, 2) points, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError(f"{what}: non-finite coordinate")
    return pts


def pooled_centroid_tour(
    centroids_by_ring: Sequence[NDArray[np.float64]],
    *,
    ring_contours: Optional[Sequence[Optional[NDArray[np.float64]]]] = None,
    max_passes: int = MEMBRANE_TOUR_MAX_PASSES,
    ring_tour_starts: int = MEMBRANE_RING_TOUR_STARTS,
) -> NDArray[np.intp]:
    """
    A short closed tour through the pooled cluster centroids of several
    rings of one axon: the order (indices into the concatenation of
    ``centroids_by_ring`` in the given ring order) of the initial curve
    of ``localization_membrane_of_rings`` (review of 2026-09-26).

    Starts: for every ring with at least three centroids, the pooled
    centroids in the order of their arc on that ring's closed tour
    POLYGON (``ring_contours[k]`` when given -- ``Ring.contour_nm``, the
    tour ``build_rings`` validated -- else a tour of that ring's own
    centroids built here: polar order about their mean rolled to
    ``ring_tour_starts`` evenly spaced starts, each improved by the same
    local search, the shortest kept; ``MEMBRANE_RING_TOUR_STARTS``),
    projected by ``tools.mps_unroll.project_on_path`` on
    the polygon itself (``_linear_closed_path``). Each start is improved
    by 2-opt, Or-opt (runs of 1-3 vertices re-inserted, reversal
    allowed) and 2-opt again (``_improved_tour``, at most
    ``max_passes`` passes each), and the shortest tour is kept (ties:
    the lowest ring). Deterministic.

    Why (the finding of the review): pooled centroids of three rings
    scattered 65 nm about a contour with a narrow deep slot (two
    slot-shaped contours of the library) are ordered wrongly by any single ring's
    curve wherever that ring's tour jumps the slot, and the P-spline of
    the wrong order loops across the lumen; the localization fit
    started from such a curve cannot recover (its parameter comes from
    the projection on it), and the arc test on the result was clearly
    liberal where the fit started from the TRUE curve is level.
    A shortest tour through ALL the pooled points follows the band of
    localizations down one side of a slot and up the other (a zigzag
    across the slot is longer), so it orders the pooled centroids as
    the membrane does; several starts guard the local search against a
    start that jumps the slot. Measured (probes of 2026-09-26, joint
    null with the reference at an end, 400 nm membrane): close to level
    on the slot-shaped library contours where the first recipe was
    clearly liberal, and within 0.07 of the true curve on every H5-B case.

    Raises ValueError with fewer than four pooled centroids, a
    non-finite one, or no ring with three centroids.
    """
    sets = [_points_or_raise(c, "pooled_centroid_tour").reshape(-1, 2) for c in centroids_by_ring]
    if ring_contours is not None and len(ring_contours) != len(sets):
        raise ValueError(f"pooled_centroid_tour: {len(ring_contours)} ring contours for {len(sets)} rings")
    pooled = np.concatenate(sets, axis=0) if sets else np.zeros((0, 2))
    n = int(pooled.shape[0])
    if n < 4:
        raise ValueError(f"pooled_centroid_tour: fewer than four pooled centroids ({n})")
    n_starts = int(ring_tour_starts)
    if n_starts < 1:
        raise ValueError(f"pooled_centroid_tour: ring_tour_starts must be >= 1, got {ring_tour_starts}")
    dist = np.hypot(pooled[:, None, 0] - pooled[None, :, 0], pooled[:, None, 1] - pooled[None, :, 1])
    from tools.mps_unroll import project_on_path  # deferred: mps_unroll imports this module
    best: Optional[Tuple[float, NDArray[np.intp]]] = None
    for k, pts in enumerate(sets):
        if pts.shape[0] < 3:
            continue
        contour = None if ring_contours is None else ring_contours[k]
        if contour is None:
            d_k = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
            polar = np.argsort(np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean()), kind="stable").astype(np.intp)
            ring_best: Optional[Tuple[float, NDArray[np.intp]]] = None
            for r_start in range(min(n_starts, int(polar.size))):
                o_k = _improved_tour(np.roll(polar, -int(round(r_start * polar.size / n_starts))), d_k, max_passes)
                l_k = _tour_length(o_k, d_k)
                if ring_best is None or l_k < ring_best[0] - 1e-9:
                    ring_best = (l_k, o_k)
            assert ring_best is not None
            contour = pts[ring_best[1]]
        poly = _points_or_raise(contour, f"pooled_centroid_tour: ring {k} contour")
        try:
            ring_path = _linear_closed_path(poly)
            sign = _outward_sign(ring_path.points_nm)
        except ValueError:
            continue
        s, _r = project_on_path(ring_path, pooled, outward_sign=sign)
        start = np.argsort(np.asarray(s, dtype=np.float64), kind="stable").astype(np.intp)
        tour = _improved_tour(start, dist, max_passes)
        length = _tour_length(tour, dist)
        if best is None or length < best[0] - 1e-9:
            best = (length, tour)
    if best is None:
        raise ValueError("pooled_centroid_tour: no ring with three centroids and a non-degenerate tour")
    return best[1]


def pooled_centroid_curve(
    centroids_by_ring: Sequence[NDArray[np.float64]],
    *,
    knot_spacing_nm: float = MEMBRANE_NULL_KNOT_SPACING_NM,
    ring_contours: Optional[Sequence[Optional[NDArray[np.float64]]]] = None,
) -> SmoothPath:
    """
    The initial curve of ``localization_membrane_of_rings`` from pooled
    centroids (review of 2026-09-26): ``smooth_polygon_membrane`` (knots
    every ``knot_spacing_nm``, 400 nm = the 2D null's spacing, finer than
    the membrane's so that the parameterisation follows every bend the
    centroids show) of the pooled centroids in the order of
    ``pooled_centroid_tour``, tabulated by ``smooth_closed_path``. The
    arrays form (``ring_contours`` None: each ring's tour built here) is
    what the harness exercises; ``_pooled_centroid_path`` passes the
    rings' own ``contour_nm``. ValueError from the tour or the fit.
    """
    if not (math.isfinite(float(knot_spacing_nm)) and float(knot_spacing_nm) > 0.0):
        raise ValueError(f"pooled_centroid_curve: knot_spacing_nm must be positive and finite, got {knot_spacing_nm}")
    order = pooled_centroid_tour(centroids_by_ring, ring_contours=ring_contours)
    pooled = np.concatenate([np.asarray(c, dtype=np.float64).reshape(-1, 2) for c in centroids_by_ring], axis=0)
    curve = smooth_polygon_membrane(pooled[order], knot_spacing_nm=float(knot_spacing_nm))
    return smooth_closed_path(curve)


def _pooled_centroid_path(rings: Sequence[Ring], knot_spacing_nm: float) -> SmoothPath:
    """
    The initial curve of ``localization_membrane_of_rings`` (default
    ``initial_order="tour"``): ``pooled_centroid_curve`` of the rings'
    centroids (every cluster, usable or not: the membrane is geometry)
    with each ring's own validated tour (``Ring.contour_nm``) as the
    start of the local search (``pooled_centroid_tour``: why a tour
    through ALL the pooled centroids, and the numbers).

    Raises ValueError when fewer than four centroids are pooled or no
    ring has a tour.
    """
    sets = [np.asarray([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2) for cl in r.clusters],
                       dtype=np.float64).reshape(-1, 2) for r in rings]
    if sum(int(s.shape[0]) for s in sets) < 4:
        raise ValueError(f"fewer than four cluster centroids in rings {[int(r.index) for r in rings]}")
    contours: List[Optional[NDArray[np.float64]]] = []
    for r, s in zip(rings, sets):
        c = None if r.contour_nm is None else np.asarray(r.contour_nm, dtype=np.float64)
        contours.append(c if c is not None and c.ndim == 2 and c.shape[0] >= 3 and np.isfinite(c).all() else None)
    return pooled_centroid_curve(sets, knot_spacing_nm=float(knot_spacing_nm), ring_contours=contours)


def _pooled_centroid_path_reference_arc(rings: Sequence[Ring], knot_spacing_nm: float) -> SmoothPath:
    """
    The FIRST initial curve of ``localization_membrane_of_rings``
    (``initial_order="reference_arc"``; the default until the review of
    2026-09-26, kept to reproduce its numbers): the P-spline
    (``smooth_polygon_membrane``, knots every ``knot_spacing_nm``) of the
    pooled cluster centroids of ``rings`` (every cluster, usable or not)
    ordered by the arc of their projection on the interpolating curve
    of the ring with the most clusters (ties: the lowest index;
    ``tools.mps_matching.ring_geometry``, the ring's own validated
    perimeter of ``build_rings``), tabulated by ``smooth_closed_path``.

    Why not a tour through the pooled centroids (``tools.mps_matching.
    pooled_membrane_path`` = ``fit_membrane``, 2-opt from one start; or
    from every start): pooled centroids of two or three rings scattered
    65 nm about a concave contour interleave, and a 2-opt tour through
    them cuts across the lumen where the interleaving is worst -- on
    a slot-shaped library contour the single-start tour keeps an edge
    across the lumen and its P-spline runs far off the contour, and the
    all-starts tour of the two rings left when ring 0 is out does too
    (probes of 2026-09-26);
    the robust step of the localization fit, started from such a
    curve, drops the very localizations of the cut region and the fit
    never recovers (the scatter of the left-out ring measured about
    twice the one about the exact contour's membrane). A ring's own
    perimeter never crosses the lumen (H1), the other rings' centroids
    projected on its curve fall in order along it, and the P-spline of
    the pooled centroids in that order goes in and out of every bay
    that holds a centroid, much closer to the contour in the same
    cases, on par with the tour where the tour works. ``fit_membrane``
    itself is not
    changed: the 2D pooled null of H5-B holds its numbers.

    What the review of 2026-09-26 found wrong with it: on slot-shaped
    library contours (a narrow, deep slot) the densest
    ring's tour jumps the slot in many replicates, the
    pooled centroids projected on it are then interleaved from the two
    sides of the slot, and the P-spline of that order runs loops across
    the lumen (many replicates far off the fit from the true curve; the
    arc test on the resulting membrane clearly liberal).
    ``pooled_centroid_curve`` replaces it.

    Raises ValueError when a ring has no contour (``ring_geometry``) or
    fewer than four centroids are pooled.
    """
    points = [np.asarray(cl.centroid_nm, dtype=np.float64).reshape(1, 2) for r in rings for cl in r.clusters]
    if len(points) < 4:
        raise ValueError(f"fewer than four cluster centroids in rings {[int(r.index) for r in rings]}")
    cents = np.concatenate(points, axis=0)
    if not np.isfinite(cents).all():
        raise ValueError("non-finite cluster centroid")
    ref = max(rings, key=lambda r: (len(r.clusters), -int(r.index)))
    ref_path = ring_geometry(ref).path
    s, _r = _project(ref_path, cents)
    order = np.argsort(s, kind="stable")
    curve = smooth_polygon_membrane(cents[order], knot_spacing_nm=float(knot_spacing_nm))
    return smooth_closed_path(curve)


def cluster_limited_knot_spacing_nm(
    knot_spacing_nm: float,
    n_clusters: int,
    length_nm: float,
    *,
    min_clusters_per_knot: float = MEMBRANE_MIN_CLUSTERS_PER_KNOT,
) -> float:
    """
    ``knot_spacing_nm`` raised, when needed, so that every knot interval
    of a curve of ``length_nm`` holds on average at least
    ``min_clusters_per_knot`` of the ``n_clusters`` detected clusters:
    max(h, c x L / K). The floor limits the fit's self-pull, which grows
    as the clusters per knot interval fall (the argument and the
    numbers at ``MEMBRANE_MIN_CLUSTERS_PER_KNOT``). With no cluster, a
    non-finite length or a non-positive ``min_clusters_per_knot`` the
    spacing is returned unchanged. ValueError on a non-positive or
    non-finite ``knot_spacing_nm``.
    """
    h = float(knot_spacing_nm)
    if not (math.isfinite(h) and h > 0.0):
        raise ValueError(f"cluster_limited_knot_spacing_nm: knot_spacing_nm must be positive and finite, got {knot_spacing_nm}")
    c = float(min_clusters_per_knot)
    k = int(n_clusters)
    length = float(length_nm)
    if k <= 0 or not (math.isfinite(length) and length > 0.0) or not (math.isfinite(c) and c > 0.0):
        return h
    return max(h, c * length / k)


def localization_membrane_of_rings(
    res: RingsResult,
    ring_indices: Sequence[int],
    *,
    exclude_ring: Optional[int] = None,
    initial_path: Optional[SmoothPath] = None,
    initial_knot_spacing_nm: float = MEMBRANE_NULL_KNOT_SPACING_NM,
    knot_spacing_nm: float = MEMBRANE_KNOT_SPACING_NM,
    penalty_rel: float = MEMBRANE_PENALTY_REL,
    iterations: int = MEMBRANE_LOCALIZATION_ITERATIONS,
    robust_c: float = MEMBRANE_ROBUST_C,
    max_points: Optional[int] = None,
    random_seed: int = 0,
    initial_order: str = "tour",
    min_clusters_per_knot: Optional[float] = None,
) -> LocalizationMembrane:
    """
    The localization membrane of the rings ``ring_indices`` of ``res``
    (``fit_membrane_from_localizations`` on the ``x_p``/``y_p`` of every
    localization of those rings, ``Ring.loc_index``), leaving
    ``exclude_ring`` out when given (leave-ring-out: the curve is then
    independent of that ring's localizations, and the radial scatter of
    its centroids about it needs no degrees-of-freedom correction,
    ``radial_scatter_on_membrane``). The initial curve is
    ``initial_path`` when given (``initial_curve`` "given"), else the
    pooled P-spline membrane of the SAME rings' centroids (``initial_curve``
    "pooled_centroids"), knots every ``initial_knot_spacing_nm`` (400 nm
    = the 2D null's spacing, finer than the localization fit's so that
    the initial parameterisation follows every bend the centroids
    show), with the centroids ordered by ``initial_order``
    (``INITIAL_ORDERS``): "tour" (the default since the review of
    2026-09-26; ``pooled_centroid_tour``, the shortest closed tour
    through all the pooled centroids from every ring's own tour) or
    "reference_arc" (the first recipe, the order of their arc on the
    densest ring's interpolating curve, which loops across the lumen
    of slot-shaped axons; ``_pooled_centroid_path_reference_arc``). The
    remaining keywords are ``fit_membrane_from_localizations``'s, except
    ``min_clusters_per_knot`` (None, the default: ``knot_spacing_nm`` as
    given; re-review of 2026-09-27): the knot spacing is then raised to
    ``cluster_limited_knot_spacing_nm`` of the rings' clusters (every
    cluster of the rings fitted) on the INITIAL curve's length, and the
    fitted ``knot_spacing_nm`` records the spacing used. The
    localizations are taken
    in ring order (sorted index), each ring's in ``loc_index`` order;
    ``n_points`` is their count.

    Raises
    ------
    ValueError
        A ring index that names no ring of ``res``, a repeated index,
        an ``exclude_ring`` not among ``ring_indices``, no ring left,
        a ring without a contour when the initial curve must be built
        from the centroids, or a fit that fails.
    """
    by_index = {int(r.index): r for r in res.rings}
    ids = [int(k) for k in ring_indices]
    if len(set(ids)) != len(ids):
        raise ValueError(f"localization_membrane_of_rings: repeated ring index in {ids}")
    unknown = [k for k in ids if k not in by_index]
    if unknown:
        raise ValueError(f"localization_membrane_of_rings: ring index {unknown} names no ring of {res.source_name!r} "
                         f"(rings {sorted(by_index)})")
    if exclude_ring is not None:
        if int(exclude_ring) not in ids:
            raise ValueError(f"localization_membrane_of_rings: exclude_ring {exclude_ring} is not among the rings {ids}")
        ids = [k for k in ids if k != int(exclude_ring)]
    if not ids:
        raise ValueError("localization_membrane_of_rings: no ring left to fit the membrane to")
    rings: List[Ring] = [by_index[k] for k in sorted(ids)]
    index = np.concatenate([np.asarray(r.loc_index, dtype=np.intp).reshape(-1) for r in rings])
    x = np.asarray(res.x_p, dtype=np.float64)[index]
    y = np.asarray(res.y_p, dtype=np.float64)[index]
    if str(initial_order) not in INITIAL_ORDERS:
        raise ValueError(f"localization_membrane_of_rings: initial_order must be one of {list(INITIAL_ORDERS)}, got {initial_order!r}")
    if not (math.isfinite(float(penalty_rel)) and float(penalty_rel) >= 0.0):
        raise ValueError(f"localization_membrane_of_rings: penalty_rel must be finite and >= 0, got {penalty_rel}")
    if initial_path is None:
        try:
            if str(initial_order) == "tour":
                init = _pooled_centroid_path(rings, float(initial_knot_spacing_nm))
            else:
                init = _pooled_centroid_path_reference_arc(rings, float(initial_knot_spacing_nm))
        except ValueError as exc:
            raise ValueError(f"localization_membrane_of_rings: the pooled centroid membrane of rings {sorted(ids)} cannot be "
                             f"fitted ({exc})") from exc
        kind = "pooled_centroids"
    else:
        init = initial_path
        kind = "given"
    if min_clusters_per_knot is not None:
        knot_spacing_nm = cluster_limited_knot_spacing_nm(knot_spacing_nm, sum(len(r.clusters) for r in rings), float(init.length_nm),
                                                          min_clusters_per_knot=float(min_clusters_per_knot))
    out = fit_membrane_from_localizations(x, y, initial_path=init, knot_spacing_nm=knot_spacing_nm, penalty_rel=penalty_rel,
                                          iterations=iterations, robust_c=robust_c, max_points=max_points,
                                          random_seed=random_seed)
    out.initial_curve = kind
    out.initial_order = str(initial_order) if kind == "pooled_centroids" else ""
    return out


def radial_scatter_on_membrane(
    centroids_nm: NDArray[np.float64],
    membrane: SmoothPath,
) -> Tuple[NDArray[np.float64], float]:
    """
    The radial scatter of cluster centroids about a membrane curve that
    was NOT fitted to them: the signed offsets (K,) of ``centroids_nm``
    (K, 2) from ``membrane`` (``tools.mps_unroll.project_on_path``, +
    outward by the curve's signed area) and sigma = sqrt(mean of the
    squared offsets) -- the rms about ZERO, with no degrees-of-freedom
    correction and no mean removed. The D-25 quantity, identifiable
    (H5C_SPEC S1): ``radial_scatter_nm`` measured the scatter about a
    P-spline fitted to the same K centroids and had to correct for the
    hat matrix's trace, and that fit was not always
    identifiable (D-28b). What remains (corrected by the review
    of 2026-09-26, which measured it on the ellipse with a realistic
    measurement): the localizations come in CLUSTERS that share their
    cluster's radial offset, so the curve's effective sample is the
    number of clusters, not of localizations, and a cluster's leverage
    on the pooled curve is ~dof / K_clusters (about a tenth at 600 nm
    knots on a dense axon), not the 1-3 % the localization count
    suggests: the pooled estimate reads ~7 % LOW at 65 nm (63.0 +/- 2.4
    for a true 68; 15.5 for 15.7 at 15 nm; 5.5 at a true 0, the
    centroid noise), and more at finer knots (the arc test's 400 nm
    curve, ``tools.mps_unroll.ARC_LOCALIZATION_KNOT_SPACING_NM``). The
    leave-ring-out estimate (``exclude_ring``) is not the dof-free one
    either: the curve of the OTHER rings carries their clusters'
    scatter, which adds in quadrature, and it reads 17-21 % HIGH (80.1
    for 68 at 65 nm). Neither bias reaches a calibrated p: ``simnull``
    closes the scatter against this same pooled estimator
    (``tools.mps_simulate_axon._calibrate_by_simulation``), so the
    simulated axons carry it too. Why about zero and not about the mean: the fit puts the
    curve at the localizations' centre, so a non-zero mean offset of a
    ring's centroids is a real radial displacement of that ring (the
    only radial parameter the simulator has, ``SimConfig.
    radial_offset_sd_nm``, is a zero-mean scatter, and the rms is what
    it must reproduce). Empty input gives (empty, NaN).
    """
    pts = np.asarray(centroids_nm, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64), float("nan")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"radial_scatter_on_membrane: expected (K, 2) centroids, got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("radial_scatter_on_membrane: non-finite centroid")
    _s, r = _project(membrane, pts)
    offsets = np.asarray(r, dtype=np.float64)
    return offsets, float(math.sqrt(float(np.mean(offsets ** 2))))
