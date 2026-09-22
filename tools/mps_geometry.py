# -*- coding: utf-8 -*-
"""
Axonal perimeter reconstruction and per-cluster area / effective radius.

Implements parameters 2, 3 and 4 of Gazal et al. (2026):

  (2) "The axonal perimeter was reconstructed by connecting the centers of
       mass of the betaII-spectrin clusters."
  (3) "The area of each cluster was estimated as the area of the smallest
       convex polygon containing all localizations of the cluster."
  (4) effective radius = radius of a circle of equivalent area,
       r_eff = sqrt(A / pi). Median ~25 nm, compatible with a single
       spectrin tetramer given localization precision, antibody linkage
       error and two labelling sites per tetramer.

Reference values (34 axons, sciatic nerve):
  * cluster area  : median 1965 nm^2 (unimodal, right tail = oligomers)
  * r_eff         : median ~25 nm (also the resolution floor)
  * N vs perimeter: y = 4.08x - 12.62 (x in um), Pearson r = 0.81

Perimeter ordering
------------------
The paper does not specify HOW the centres of mass are connected, and for
the markedly irregular, non-convex axon cross-sections of the sciatic
nerve the choice matters a great deal (a convex hull would be wrong).

This module orders the centroids by polar angle about their own centroid
and then removes any self-intersections with a 2-opt pass. Polar-angle
order alone is exact for star-shaped contours but fails on deep
concavities; 2-opt repairs precisely those cases, since an untangled
closed tour is always shorter than a self-crossing one.

The centroid of the spectrin cluster centres is used as the angular
origin -- NOT a second channel. betaII-spectrin is always channel 1 here,
while channel 2 varies by experiment (this repo's own example data pairs
spectrin with adducin, not tubulin), so depending on it would break
whenever the second marker changes or is absent.

Contour health
--------------
The tour above always returns a number, and until now the only thing
that could contradict it was the self-intersection count -- which stays
at zero when a cluster from inside the axon becomes a vertex, when a gap
in the ring is closed by a chord, and when both happen at once. The
number then looks exactly as healthy as a good one.

``contour_health`` adds two checks that do fire, both calibrated on
simulated rings whose perimeter is known, so that the limits sit above
what a correct contour produces rather than where they happen to catch
this dataset:

  vertex depth inside the hull of all the vertices, as a fraction of the
  hull's effective radius. The rings simulated have a 1.6 um radius and
  their centres scattered radially by a Gaussian. With a standard
  deviation of up to 80 nm (5 % of the radius) the deepest vertex of a
  circle reaches 0.29; with 40 nm, 0.21 with gaps of up to 160 degrees,
  0.21 with a dent half the radius deep, 0.305 on an oval squeezed to a
  waist a third of its width and 0.386 on an hourglass with a waist a
  quarter of its width. A single cluster planted inside the ring sits at
  a median of 0.63. The limit is 0.40: above those healthy shapes, below
  the typical planted cluster.

  It holds only for that scatter. The depth grows with the scatter
  relative to the radius: with a standard deviation of 150 nm (9 % of
  the radius) healthy circles reach 0.48, and a smaller axon with the
  same scatter in nm is the same case. Where the centres scatter off the
  outline more than the simulated rings did, a vertex over the limit can
  be that scatter rather than a cluster from inside the axon.

  That margin is bought with sensitivity. At 0.40 the check only sees a
  centre more than ~0.6 um inside a 1.6 um axon, which is far blunter
  than a second channel would be -- the tubulin mask flags an interior
  at 250 nm. It is meant to catch the contours that are plainly wrong,
  not to measure the interior.

  Measured on the 18 April axons (2026-09-22), the centres scatter off a
  smooth outline by a median of 9.6 % of the hull radius, 6 to 16 % --
  twice what the 0.40 was calibrated for -- and the check fired on 14 of
  the 18. So the count stays, because it is right for clean data, and
  what it CLAIMS is now read against the axon's own scatter
  (``radial_scatter``): the depth a healthy ring with that scatter and
  that many centres stays under 99 times in 100 is 0.237 + 1.052 s
  sqrt(2 ln K) of the hull radius -- the upper envelope over a circle, an
  oval, a peanut and the waist-a-third cross-section, s from 4 to 20 %
  and K from 30 to 140, no case above it. Only centres deeper than that
  as well are called probably off the membrane; the rest are said to be
  within what the scatter explains. Tried first and ruled out, each on
  measurement: measuring depth from a fitted smooth outline instead of
  the hull (less sensitive, 19 % against 93 % for a centre at half the
  radius), and a per-axon limit from a parametric bootstrap (11 to 32 %
  false alarms on healthy circles, because the fit absorbs scatter). On
  the 18, 10 axons have centres deeper than their own scatter reaches,
  and the four whose interior clusters the widefield images confirm are
  among them.

  longest step over the median step, for the chord a gap forces. This
  one separates poorly and its limit is correspondingly loose: random
  angular spacing on simulated rings with no gap reached 18.8, so the
  limit is 20. With 70 centres it sees a 90 degree gap but not a 40
  degree one. The ratio is taken against the median step, which grows as
  the centres get fewer, so with fewer centres a gap of the same size is
  missed more often.

Two other numbers are reported and deliberately NOT flagged. The ratio
of the tour to the convex hull of the same centres is what first made
this dataset look wrong, but it depends on the number of clusters and on
how far they scatter off a smooth outline: a perfectly correct contour
of a ring with 140 centres scattered by 80 nm already reads 1.9, which
is what axon 7 of April reads. And the share of the length carried by
the few steps above 3x the median runs near 30 % on a healthy ring.
Neither can carry a threshold; both are useful next to one that can.

Centre of the contour
---------------------
Not a parameter of the paper. ``contour_centre`` gives the area centroid
of the contour: the centre of mass of the region it encloses. Three
centres were compared on the anchored contours of the 18 April axons by
leaving each cluster out in turn and measuring how far the centre moved
(largest move per axon, median / worst): area centroid 39 / 150 nm; the
maximum of a laminar flow with no slip on the contour 147 / 2168 nm; the
centre of the largest inscribed circle 276 / 2404 nm. On 2/axon6_roi2,
a contour with two lobes, the last two moved by more than 2 um where the
area centroid moved 62 nm (on 2/axon9_roi2, also two-lobed, 703 and
1080 nm against 64 nm); that is why it is the one computed here. It is a geometric centre
of the contour and nothing more: like the perimeter, it is only as good
as the contour, and a cluster from inside the axon that becomes a vertex
moves it.

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError

# Paper reference values, for comparison in the results panel.
PAPER_MEDIAN_CLUSTER_AREA_NM2 = 1965.0
PAPER_MEDIAN_R_EFF_NM = 25.0
PAPER_SLOPE_CLUSTERS_PER_UM = 4.08
PAPER_INTERCEPT_CLUSTERS = -12.62

# --- contour health ---------------------------------------------------------
# An edge this many times the median is a jump, not a step along the ring.
LONG_EDGE_FACTOR = 3.0
# A vertex this far inside the hull of all the vertices, as a fraction of
# the hull's own effective radius (2 * area / perimeter, which is R for a
# circle), is deeper than any healthy ring reaches. Against the radius and
# not against the median step, because the median step shrinks as 1/K and
# would make the same axon more or less suspicious with the DBSCAN
# settings alone. Calibrated in the module docstring.
DEEP_VERTEX_FRACTION = 0.40
# The same question asked of THIS axon's scatter. How far the centres
# scatter off a smooth outline is estimated from the centres themselves: a
# robust fit of the radius against the angle (harmonics up to this order,
# Tukey's biweight, so a few centres from inside do not drag it in), and
# 1.4826 x MAD of what is left. Measured on simulated rings, that estimate
# comes out low -- a median of 0.85 of the true scatter over 36 cases of
# scatter, number of centres and shape -- so it is divided by that. It
# stays rough: one ring's estimate ranges from 0.43 to 1.06 of the truth at
# 35 centres and 0.64 to 1.22 at 95, which is why it only qualifies the
# warning and never sets the count.
SCATTER_HARMONICS = 4
SCATTER_ESTIMATE_BIAS = 0.85
# Below this many centres the fit above has too few to spare.
SCATTER_MIN_CENTRES = 15
# How deep the deepest vertex of a HEALTHY ring goes, as a fraction of the
# hull radius, for radial scatter s (a fraction of the radius) and K
# centres: 0.237 + 1.052 * s * sqrt(2 ln K). The 99th percentile over
# 400 simulated rings per case, the most permissive of four shapes (circle,
# oval 0.7, peanut, and the waist-a-third cross-section the concave check
# uses), fitted over s = 4-20 % and K = 30-140 and moved up so that no case
# lies above it. A vertex deeper than that is deeper than a healthy axon of
# any of those shapes, with this scatter, reaches 99 times in 100.
HEALTHY_DEPTH_INTERCEPT = 0.237
HEALTHY_DEPTH_SLOPE = 1.052
# One step this many times the median is a chord across the axon. Wide,
# because random angular spacing reached 18.8 on simulated gapless rings
# (the largest of them, not a bound: a rarer ring can pass it).
GAPLESS_MAX_OVER_MEDIAN = 18.8
MAX_OVER_MEDIAN_LIMIT = 20.0


# ============================================================================
# Perimeter reconstruction (parameter 2)
# ============================================================================

def _polar_angle_order(points: NDArray[np.float64]) -> NDArray[np.intp]:
    """Indices ordering points counter-clockwise by angle about their centroid."""
    centre = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    return np.argsort(ang)


def _segments_properly_intersect(
    p1: NDArray[np.float64], p2: NDArray[np.float64],
    p3: NDArray[np.float64], p4: NDArray[np.float64],
) -> bool:
    """True if segment p1-p2 properly crosses segment p3-p4."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _tour_length(points: NDArray[np.float64], order: NDArray[np.intp]) -> float:
    """Total length of the closed polygon visiting points in `order`."""
    pts = points[order]
    closed = np.vstack([pts, pts[:1]])
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _two_opt(
    points: NDArray[np.float64],
    order: NDArray[np.intp],
    max_passes: int = 50,
) -> Tuple[NDArray[np.intp], int]:
    """
    2-opt refinement of a closed tour: repeatedly reverse a sub-path when
    doing so shortens the tour. Removes self-intersections (a crossing
    tour is never locally optimal under 2-opt) and tightens the contour
    around concavities that polar-angle ordering gets wrong.

    Returns (improved_order, n_improvements).
    """
    order = order.copy()
    n = len(order)
    if n < 4:
        return order, 0

    improvements = 0
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Skip the pair that would sever the closing edge.
                if i == 0 and j == n - 1:
                    continue
                a, b = points[order[i]], points[order[i + 1]]
                c, d = points[order[j]], points[order[(j + 1) % n]]
                before = (np.linalg.norm(a - b) + np.linalg.norm(c - d))
                after = (np.linalg.norm(a - c) + np.linalg.norm(b - d))
                if after < before - 1e-9:
                    order[i + 1:j + 1] = order[i + 1:j + 1][::-1]
                    improved = True
                    improvements += 1
        if not improved:
            break
    return order, improvements


# Slack of the vectorised screen in _two_opt_fast, in the coordinates'
# units (nm). Far above any rounding difference between the screen and the
# legacy arithmetic, and far below any real improvement.
_SCREEN_SLACK = 1e-6


def _two_opt_fast(
    points: NDArray[np.float64],
    order: NDArray[np.intp],
    max_passes: int = 50,
) -> Tuple[NDArray[np.intp], int]:
    """
    The same 2-opt as ``_two_opt``, taking exactly the same decisions,
    about ten times faster.

    For each i, every candidate j is screened in one numpy operation with
    a slightly relaxed test; the candidates are then walked in order with
    the legacy arithmetic, so the first j that passes is the one the
    legacy loop would take. After a reversal the scan resumes at j + 1 on
    the modified tour, as the legacy loop does. Checked tour for tour
    against ``_two_opt`` in validate_geometry.py.
    """
    order = np.array(order, dtype=np.intp, copy=True)
    n = len(order)
    if n < 4:
        return order, 0

    improvements = 0
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            j = i + 2
            while j < n:
                js = np.arange(j, n)
                if i == 0:
                    # The pair that would sever the closing edge.
                    js = js[js != n - 1]
                if js.size == 0:
                    break
                pts = points[order]
                a, b = pts[i], pts[i + 1]
                c, d = pts[js], pts[(js + 1) % n]
                before = (np.sqrt(((a - b) ** 2).sum())
                          + np.sqrt(((c - d) ** 2).sum(axis=1)))
                after = (np.sqrt(((a - c) ** 2).sum(axis=1))
                         + np.sqrt(((b - d) ** 2).sum(axis=1)))
                taken = None
                for jj in js[after < before - 1e-9 + _SCREEN_SLACK]:
                    a0, b0 = points[order[i]], points[order[i + 1]]
                    c0, d0 = points[order[jj]], points[order[(jj + 1) % n]]
                    if (np.linalg.norm(a0 - c0) + np.linalg.norm(b0 - d0)
                            < np.linalg.norm(a0 - b0)
                            + np.linalg.norm(c0 - d0) - 1e-9):
                        taken = int(jj)
                        break
                if taken is None:
                    break
                order[i + 1:taken + 1] = order[i + 1:taken + 1][::-1]
                improved = True
                improvements += 1
                j = taken + 1
        if not improved:
            break
    return order, improvements


def _count_self_intersections(
    points: NDArray[np.float64], order: NDArray[np.intp]
) -> int:
    """Number of properly crossing edge pairs in the closed polygon."""
    pts = points[order]
    n = len(pts)
    count = 0
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c, d = pts[j], pts[(j + 1) % n]
            if _segments_properly_intersect(a, b, c, d):
                count += 1
    return count


@dataclass
class ContourHealth:
    """
    Does the reconstructed tour behave like the perimeter of a ring?

    Every number here is computed from the contour's own vertices. No
    image, no second channel and no assumption about the axon: these are
    statements about the polygon, not about biology.

    Two of them carry a warning. ``n_deep_vertices`` counts the centres
    that sit further inside the convex hull of all the centres than any
    healthy ring reaches, which is what a cluster from inside the axon
    looks like. ``max_over_median`` catches the other failure: one step
    long enough to be a chord across the axon, which is how a contour
    closes a wide gap in the ring.

    ``tour_over_hull`` and ``length_in_long_edges`` are reported but never
    flagged, because neither separates a healthy contour from a broken one
    (module docstring, "Contour health"). They are here as context: the
    first says how much longer the quoted perimeter is than the shortest
    outline enclosing the same centres, the second how much of it rests on
    a few long steps.
    """

    hull_perimeter_nm: float
    hull_radius_nm: float
    tour_over_hull: float
    edge_median_nm: float
    edge_max_nm: float
    max_over_median: float
    n_long_edges: int
    length_in_long_edges: float
    n_deep_vertices: int
    max_depth_nm: float
    warnings: List[str] = field(default_factory=list)
    # How far the centres scatter off a smooth outline, estimated from
    # them (radial_scatter); None with too few centres to fit.
    scatter_nm: Optional[float] = None
    # The depth a healthy ring with that scatter and this many centres
    # stays under 99 times in 100, as a fraction of the hull radius.
    healthy_depth_fraction: Optional[float] = None
    # Deep centres that are deeper than that as well: the ones this
    # axon's own scatter does not explain.
    n_deep_beyond_scatter: Optional[int] = None

    @property
    def hull_perimeter_um(self) -> float:
        return self.hull_perimeter_nm / 1000.0

    @property
    def depth_limit_nm(self) -> float:
        """How deep a vertex must be before it is counted as interior."""
        return DEEP_VERTEX_FRACTION * self.hull_radius_nm

    @property
    def scatter_percent(self) -> Optional[float]:
        """The scatter as a percentage of the hull radius."""
        if self.scatter_nm is None or self.hull_radius_nm <= 0:
            return None
        return 100.0 * self.scatter_nm / self.hull_radius_nm

    @property
    def scatter_depth_limit_nm(self) -> Optional[float]:
        """How deep a healthy ring with this axon's scatter reaches."""
        if self.healthy_depth_fraction is None:
            return None
        return self.healthy_depth_fraction * self.hull_radius_nm

    @property
    def has_interior_vertices(self) -> bool:
        """Some centres are deeper inside than a ring's ever are."""
        return self.n_deep_vertices > 0

    @property
    def bridges_a_gap(self) -> bool:
        """One step is long enough to be a chord across the axon."""
        return self.max_over_median > MAX_OVER_MEDIAN_LIMIT


@dataclass
class RadialProfile:
    """Centres against the smooth outline fitted to them, for plotting."""

    # The origin of the polar coordinates: the centres' mean.
    centre: NDArray[np.float64]
    theta: NDArray[np.float64]       # each centre's angle, radians
    r: NDArray[np.float64]           # its distance from the origin, nm
    coef: NDArray[np.float64]        # the harmonic outline's coefficients
    scatter_nm: float

    def outline_at(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """The fitted outline's radius at these angles, nm."""
        t = np.asarray(theta, dtype=float)
        cols = [np.ones_like(t)]
        for j in range(1, SCATTER_HARMONICS + 1):
            cols += [np.cos(j * t), np.sin(j * t)]
        return np.column_stack(cols) @ self.coef


def radial_scatter(points: NDArray[np.float64]) -> Optional[float]:
    """How far centres scatter off a smooth closed outline, in nm.

    ``radial_profile`` does the work; this is its one number.
    """
    profile = radial_profile(points)
    return None if profile is None else profile.scatter_nm


def radial_profile(points: NDArray[np.float64]) -> Optional[RadialProfile]:
    """
    How far centres scatter off a smooth closed outline, and the outline.

    The radius about the centres' mean is fitted against the angle with
    harmonics up to SCATTER_HARMONICS, reweighted with Tukey's biweight so
    that centres far off the outline -- a cluster from inside the axon --
    stop pulling it; the scatter is the upper half-spread of what remains,
    (Q75 - Q50) / 0.6745, which a centre from inside cannot reach, divided
    by SCATTER_ESTIMATE_BIAS. None with fewer than SCATTER_MIN_CENTRES
    centres.

    Rough by nature: from one ring of 35 centres it can be off by a
    factor of two either way. Use it to qualify a statement, never to
    make one.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < SCATTER_MIN_CENTRES or not np.isfinite(pts).all():
        return None
    d = pts - pts.mean(axis=0)
    theta = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])
    cols = [np.ones_like(theta)]
    for j in range(1, SCATTER_HARMONICS + 1):
        cols += [np.cos(j * theta), np.sin(j * theta)]
    X = np.column_stack(cols)
    w = np.ones_like(r)
    res = r - r.mean()
    for _ in range(25):
        root = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(X * root[:, None], r * root, rcond=None)
        res = r - X @ coef
        mad = float(np.median(np.abs(res - np.median(res))))
        if mad <= 0.0:
            break
        u = res / (4.685 * 1.4826 * mad)
        w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    # The scale from the OUTER half of the residuals only. A centre from
    # inside the axon deviates inward, into the lower tail, so a
    # two-sided scale would let the very centres this check looks for
    # raise the bar they are judged against. Measured with 20 % of the
    # centres planted inside: the two-sided MAD came out 1.29 to 1.37
    # times the true scatter at 95 centres, the outer half 1.09 to 1.17.
    q50, q75 = np.percentile(res, [50.0, 75.0])
    outer = float(q75 - q50) / 0.6745
    if not np.isfinite(outer) or outer <= 0.0:
        return None
    return RadialProfile(centre=pts.mean(axis=0), theta=theta, r=r,
                         coef=np.asarray(coef, dtype=float),
                         scatter_nm=outer / SCATTER_ESTIMATE_BIAS)


def vertex_depths(contour: NDArray[np.float64]
                  ) -> Tuple[NDArray[np.float64], float]:
    """How far inside the convex hull each vertex sits (nm), and the
    hull's effective radius (2 x area / perimeter)."""
    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    hull = ConvexHull(pts)
    vertices = pts[hull.vertices]
    closed = np.vstack([vertices, vertices[:1]])
    length = float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())
    depth = -(pts @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1)
    return depth, 2.0 * float(hull.volume) / length


def healthy_depth_fraction(scatter_fraction: float, n_centres: int) -> float:
    """How deep a healthy ring's deepest vertex goes, / hull radius."""
    k = max(int(n_centres), 3)
    return (HEALTHY_DEPTH_INTERCEPT
            + HEALTHY_DEPTH_SLOPE * scatter_fraction
            * float(np.sqrt(2.0 * np.log(k))))


def contour_health(contour: NDArray[np.float64]) -> Optional[ContourHealth]:
    """
    Measure a closed contour against the shape it is supposed to be.

    Parameters
    ----------
    contour : (K, 2) vertices in contour order, in nm (open: the closing
        edge back to the first vertex is added here).

    Returns
    -------
    ContourHealth, or None when the vertices are degenerate (fewer than
    three, or exactly on one line) and no hull exists to compare against.
    """
    contour = np.asarray(contour, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        return None
    # One non-finite vertex makes every length NaN, and NaN compares false
    # against any limit, so the checks below would report a clean bill of
    # health on a contour that has none. reconstruct_perimeter warns.
    if not np.isfinite(contour).all():
        return None

    closed = np.vstack([contour, contour[:1]])
    edges = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(edges.sum())
    median = float(np.median(edges))
    if total <= 0.0 or median <= 0.0:
        return None

    try:
        hull = ConvexHull(contour)
    except QhullError:
        return None
    vertices = contour[hull.vertices]
    hull_closed = np.vstack([vertices, vertices[:1]])
    hull_length = float(np.linalg.norm(np.diff(hull_closed, axis=0),
                                       axis=1).sum())
    if hull_length <= 0.0:
        return None

    # How far inside the hull each vertex sits. The hull's facet equations
    # are normal . x + offset <= 0 inside, with unit normals, so the
    # distance to the nearest facet is minus the largest of them -- exact,
    # and free of the edge cases a point-in-polygon test has.
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    depth = -(contour @ normals.T + offsets).max(axis=1)

    # 2 * area / perimeter is the radius for a circle, and stays a sensible
    # calibre for the irregular cross-sections of the sciatic nerve. Only a
    # hull with no area is treated as degenerate here: a thin sliver still
    # gets a (tiny) radius, and on it every vertex looks deep.
    hull_radius = 2.0 * float(hull.volume) / hull_length
    if not np.isfinite(hull_radius) or hull_radius <= 0.0:
        return None
    long_edge = edges > LONG_EDGE_FACTOR * median
    deep = depth > DEEP_VERTEX_FRACTION * hull_radius
    health = ContourHealth(
        hull_perimeter_nm=hull_length,
        hull_radius_nm=hull_radius,
        tour_over_hull=total / hull_length,
        edge_median_nm=median,
        edge_max_nm=float(edges.max()),
        max_over_median=float(edges.max()) / median,
        n_long_edges=int(long_edge.sum()),
        length_in_long_edges=float(edges[long_edge].sum() / total),
        n_deep_vertices=int(deep.sum()),
        max_depth_nm=float(depth.max()),
    )
    scatter = radial_scatter(contour)
    if scatter is not None:
        health.scatter_nm = scatter
        health.healthy_depth_fraction = healthy_depth_fraction(
            scatter / hull_radius, len(contour))
        beyond = depth > max(DEEP_VERTEX_FRACTION,
                             health.healthy_depth_fraction) * hull_radius
        health.n_deep_beyond_scatter = int(beyond.sum())

    if health.has_interior_vertices:
        base = (
            f"{health.n_deep_vertices} of the {len(contour)} centres on "
            f"this contour sit more than {health.depth_limit_nm:,.0f} nm "
            f"inside their own convex hull, the deepest by "
            f"{health.max_depth_nm:,.0f} nm: deeper than any simulated ring "
            f"whose centres scatter off the outline by up to 5 % of its "
            f"radius reaches, with gaps, dents or a waist a third of its "
            f"width. ")
        tail = (
            f"Look at the contour before using the perimeter: it is "
            f"{health.tour_over_hull:.2f} times the hull of the same "
            f"centres ({health.hull_perimeter_um:.2f} um), and clusters per "
            f"um, occupancy and the randomization all follow it.")
        limit = health.scatter_depth_limit_nm
        if health.scatter_nm is None or limit is None:
            middle = (
                f"Too few centres to measure how far this axon's scatter "
                f"off its outline goes, so whether that is scatter or "
                f"centres from inside the axon cannot be told apart here. ")
        elif health.n_deep_beyond_scatter:
            middle = (
                f"This axon's centres scatter off a smooth outline by about "
                f"{health.scatter_nm:,.0f} nm "
                f"({health.scatter_nm / health.hull_radius_nm:.0%} of its "
                f"radius, estimated from its own centres, so rough), and a "
                f"healthy ring that scatters that much stays under "
                f"{limit:,.0f} nm 99 times in 100. "
                + (f"1 of them is deeper than that too: it is probably not "
                   f"on the membrane, or the axon is more concave than any "
                   f"simulated shape. "
                   if health.n_deep_beyond_scatter == 1 else
                   f"{health.n_deep_beyond_scatter} of them are deeper than "
                   f"that too: those are probably not on the membrane, or "
                   f"the axon is more concave than any simulated shape. "))
        else:
            middle = (
                f"But this axon's centres scatter off a smooth outline by "
                f"about {health.scatter_nm:,.0f} nm "
                f"({health.scatter_nm / health.hull_radius_nm:.0%} of its "
                f"radius, estimated from its own centres, so rough), and a "
                f"healthy ring that scatters that much reaches "
                f"{limit:,.0f} nm: these depths are within what its own "
                f"scatter explains, so they may be scatter rather than "
                f"centres from inside the axon. ")
        health.warnings.append(base + middle + tail)
    if health.bridges_a_gap:
        health.warnings.append(
            f"One step of the contour is {health.max_over_median:.1f} times "
            f"the median ({health.edge_max_nm:,.0f} nm against "
            f"{health.edge_median_nm:,.0f} nm). Random angular spacing "
            f"reached {GAPLESS_MAX_OVER_MEDIAN:g} on simulated rings with no "
            f"gap; past the limit of {MAX_OVER_MEDIAN_LIMIT:.0f} the contour "
            f"is probably closing a gap with a chord, which cuts across the "
            f"axon and puts every localization it passes on the wrong side "
            f"of the contour."
        )
    return health


# ============================================================================
# Centre of the contour
# ============================================================================

@dataclass
class ContourCentre:
    """
    The area centroid of a closed contour: the centre of mass of the
    region it encloses, taken as uniform. It weighs the enclosed area, not
    the vertices, so a stretch of the ring crowded with clusters does not
    pull it the way it pulls the mean of the cluster centres.

    ``max_shift_nm`` is how far it moves, at most, when any one vertex is
    left out of the contour with the order of the rest kept (no new tour
    is built). It is a sensitivity to single clusters, not a confidence
    interval. A vertex whose removal makes the contour cross itself -- the
    chord that replaces it cuts another edge, as it can past a deep notch
    -- is skipped, since that contour has no centre. None with fewer than
    four vertices, where leaving one out leaves no area, or when every
    removal is skipped.
    """

    x_nm: float
    y_nm: float
    area_nm2: float
    max_shift_nm: Optional[float] = None

    @property
    def area_um2(self) -> float:
        return self.area_nm2 / 1e6


def _chord_crosses(p: NDArray[np.float64]) -> NDArray[np.bool_]:
    """
    For each vertex i of the closed polygon ``p``, whether the chord from
    vertex i-1 to vertex i+1 properly crosses an edge of the polygon other
    than the four that touch i-1, i or i+1: whether leaving vertex i out
    makes a polygon without crossings cross itself.
    """
    k = len(p)
    a, b = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)   # chord ends
    e0, e1 = p, np.roll(p, -1, axis=0)                     # edge j: j, j+1

    def orient(o, d, q):
        """Sign-carrying cross product (d - o) x (q - o), broadcast."""
        return ((d[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1])
                - (d[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0]))

    A, B = a[:, None, :], b[:, None, :]            # chord i, along rows
    E0, E1 = e0[None, :, :], e1[None, :, :]        # edge j, along columns
    crosses = ((orient(E0, E1, A) * orient(E0, E1, B) < 0)
               & (orient(A, B, E0) * orient(A, B, E1) < 0))
    # The edges i-2 .. i+1 end at i-1 or i+1, or are the two replaced.
    offset = (np.arange(k)[None, :] - np.arange(k)[:, None]) % k
    crosses &= ~np.isin(offset, (k - 2, k - 1, 0, 1))
    return np.asarray(crosses.any(axis=1))


def contour_centre(contour: NDArray[np.float64]) -> Optional[ContourCentre]:
    """
    Area centroid of a closed polygon (shoelace formula), and how far it
    moves when each vertex is left out in turn.

    Parameters
    ----------
    contour : (K, 2) vertices in contour order, in nm, open (the closing
        edge is implied). The polygon must not cross itself: the formula
        then weighs each lobe by its signed area, and the result is not the
        centre of anything. ``reconstruct_perimeter`` only calls this on a
        contour without crossings.

    Returns
    -------
    ContourCentre, or None when the contour has fewer than three vertices,
    a non-finite vertex, or no area (all vertices on one line).
    """
    contour = np.asarray(contour, dtype=float)
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 3:
        return None
    if not np.isfinite(contour).all():
        return None
    # About the vertices' own mean: coordinates of tens of micrometres
    # would otherwise lose digits in the cross products.
    origin = contour.mean(axis=0)
    p = contour - origin
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y                      # twice each triangle's area
    twice_area = float(cross.sum())
    # The most each cross product could be, |p_k| |p_k+1|: its rounding
    # error is a few 1e-16 of this whatever the shape, so an area below
    # 1e-12 of it is rounding, not area -- vertices on one line.
    scale = float((np.hypot(x, y) * np.hypot(x1, y1)).sum())
    if scale <= 0.0 or abs(twice_area) <= 1e-12 * scale:
        return None
    mx = float(((x + x1) * cross).sum())
    my = float(((y + y1) * cross).sum())
    cx, cy = mx / (3.0 * twice_area), my / (3.0 * twice_area)

    max_shift: Optional[float] = None
    k = len(p)
    if k >= 4:
        # Leaving vertex i out replaces the edges (i-1, i) and (i, i+1) by
        # the chord (i-1, i+1): subtract their terms, add the chord's.
        xp, yp = np.roll(x, 1), np.roll(y, 1)
        cross_prev = np.roll(cross, 1)            # edge (i-1, i)
        chord = xp * y1 - x1 * yp                 # edge (i-1, i+1)
        area_i = twice_area - cross_prev - cross + chord
        mx_i = (mx - (xp + x) * cross_prev - (x + x1) * cross
                + (xp + x1) * chord)
        my_i = (my - (yp + y) * cross_prev - (y + y1) * cross
                + (yp + y1) * chord)
        ok = (np.abs(area_i) > 1e-12 * scale) & ~_chord_crosses(p)
        if ok.any():
            sx = mx_i[ok] / (3.0 * area_i[ok]) - cx
            sy = my_i[ok] / (3.0 * area_i[ok]) - cy
            max_shift = float(np.hypot(sx, sy).max())
    return ContourCentre(x_nm=cx + float(origin[0]),
                         y_nm=cy + float(origin[1]),
                         area_nm2=abs(twice_area) / 2.0,
                         max_shift_nm=max_shift)


@dataclass
class PerimeterResult:
    """Reconstructed axonal contour from cluster centres of mass."""

    order: NDArray[np.intp]              # index order into the centroid array
    contour: NDArray[np.float64]         # (K, 2) centroids in contour order
    perimeter_nm: float
    perimeter_um: float
    n_clusters: int
    n_2opt_improvements: int
    self_intersections_before: int
    self_intersections_after: int
    warnings: List[str] = field(default_factory=list)
    health: Optional[ContourHealth] = None
    # 2-opt starting points tried, and the length each one reached (nm).
    # One start is the legacy behaviour; with all of them the shortest tour
    # is kept and the spread says how much the start would have mattered.
    n_starts: int = 1
    start_lengths_nm: Optional[NDArray[np.float64]] = None
    # Where the order of the tour came from. "automatic" is the polar
    # angle plus 2-opt below; anything else was set by a person, and the
    # difference has to survive every later step -- it is what stops a
    # hand-set contour being exported as a measured one, and what stops
    # with_every_start rebuilding it.
    order_source: str = "automatic"
    # The area centroid of the contour; None when it crosses itself or
    # encloses no area (contour_centre).
    centre: Optional[ContourCentre] = None

    @property
    def start_spread_um(self) -> Optional[float]:
        """Longest minus shortest tour over the starts tried, in um."""
        if self.start_lengths_nm is None or len(self.start_lengths_nm) < 2:
            return None
        return float(np.ptp(self.start_lengths_nm)) / 1000.0

    @property
    def clusters_per_um(self) -> Optional[float]:
        if self.perimeter_um <= 0:
            return None
        return self.n_clusters / self.perimeter_um

    def expected_n_clusters_from_paper(self) -> Optional[float]:
        """N predicted by the paper's regression for this perimeter."""
        if self.perimeter_um <= 0:
            return None
        return (PAPER_SLOPE_CLUSTERS_PER_UM * self.perimeter_um
                + PAPER_INTERCEPT_CLUSTERS)


def reconstruct_perimeter(
    centroids: NDArray[np.float64],
    refine: bool = True,
    custom_order: Optional[NDArray[np.intp]] = None,
    all_starts: bool = False,
) -> PerimeterResult:
    """
    Reconstruct the axonal perimeter by connecting cluster centres of mass.

    Parameters
    ----------
    centroids : (K, 2) array of cluster centres of mass, in nm.
    refine : run the 2-opt untangling pass (default True). Disable to get
        the raw polar-angle contour.
    custom_order : explicit ordering to use instead of the automatic one.
        This is the hook for expert manual override from the GUI: the user
        can reorder/repair the contour by hand when the automatic result
        is wrong for an unusual axon shape.
    all_starts : run 2-opt from every starting point of the polar cycle and
        keep the shortest tour. 2-opt only accepts improvements, so where it
        ends depends on where it starts: on the 18 April axons, rolling the
        start changed the perimeter in 144 of 252 rolled starts (14 per
        axon besides the original), by up to 12.9 %.
        The shortest over every start does not depend on it, and the MPS
        analysis builds its contour this way (tools.mps_analysis). Off by
        default here: one start is what this program did before
        2026-09-19, and validate_full_18axons.py still uses it.

    Returns
    -------
    PerimeterResult

    Raises
    ------
    ValueError if fewer than 3 centroids are supplied (no closed contour).
    """
    centroids = np.asarray(centroids, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError(f"centroids must be (K, 2), got {centroids.shape}")

    k = len(centroids)
    if k < 3:
        raise ValueError(
            f"Need at least 3 cluster centres to close a contour, got {k}. "
            f"This axon has too few surviving clusters to reconstruct a perimeter."
        )

    warnings_: List[str] = []
    finite = np.isfinite(centroids).all(axis=1)
    n_bad = int(np.count_nonzero(~finite))
    if n_bad:
        warnings_.append(
            f"{n_bad} of the {k} cluster centres is not a finite coordinate. "
            f"The perimeter below is NaN and so is everything computed from "
            f"it, and no check can contradict it, because NaN compares false "
            f"against every limit. Find those clusters before reading any "
            f"number from this axon."
        )

    if custom_order is not None:
        order = np.asarray(custom_order, dtype=np.intp)
        if sorted(order.tolist()) != list(range(k)):
            raise ValueError(
                "custom_order must be a permutation of all centroid indices."
            )
        before = _count_self_intersections(centroids, order)
        health = contour_health(centroids[order])
        crossing = ([] if before == 0 else [
            f"The manual contour crosses itself at {before} point(s): the "
            f"area it encloses, and so its centre, are not defined."])
        return PerimeterResult(
            order=order,
            contour=centroids[order],
            perimeter_nm=_tour_length(centroids, order),
            perimeter_um=_tour_length(centroids, order) / 1000.0,
            n_clusters=k,
            # No 2-opt ran, so neither number describes this tour: zero
            # starts rather than one, which is what a reader of n_starts
            # would otherwise take for a 2-opt result.
            n_starts=0,
            order_source="set by hand",
            n_2opt_improvements=0,
            self_intersections_before=before,
            self_intersections_after=before,
            warnings=(warnings_
                      + ["Manual contour ordering supplied by the user."]
                      + crossing + (health.warnings if health else [])),
            health=health,
            centre=contour_centre(centroids[order]) if before == 0 else None,
        )

    order = _polar_angle_order(centroids)
    xi_before = _count_self_intersections(centroids, order)

    n_improvements = 0
    n_starts = 1
    start_lengths: Optional[NDArray[np.float64]] = None
    if refine and all_starts:
        # With three centres every start gives the same triangle; they are
        # still tried, so the result says it came from every start.
        lengths = np.empty(k)
        best: Optional[Tuple[float, NDArray[np.intp], int]] = None
        for start in range(k):
            tour, n_imp = _two_opt_fast(centroids, np.roll(order, -start))
            lengths[start] = _tour_length(centroids, tour)
            # Strictly shorter, so ties go to the earliest start and the
            # result is the same every time.
            if best is None or lengths[start] < best[0] - 1e-9:
                best = (float(lengths[start]), tour, n_imp)
        assert best is not None
        _, order, n_improvements = best
        n_starts = k
        start_lengths = lengths
    elif refine:
        order, n_improvements = _two_opt(centroids, order)
    xi_after = _count_self_intersections(centroids, order)

    if xi_before > 0:
        warnings_.append(
            f"Polar-angle ordering produced {xi_before} self-intersection(s); "
            f"2-opt refinement resolved them down to {xi_after}. This axon has a "
            f"strongly concave contour -- inspect the reconstructed perimeter "
            f"before trusting the value."
        )
    if xi_after > 0:
        warnings_.append(
            f"The reconstructed contour still self-intersects at {xi_after} "
            f"point(s). The perimeter is unreliable, and the area it "
            f"encloses and its centre are not defined; consider correcting "
            f"the contour manually."
        )

    length = _tour_length(centroids, order)
    health = contour_health(centroids[order])
    if health is not None:
        warnings_.extend(health.warnings)
    return PerimeterResult(
        order=order,
        contour=centroids[order],
        perimeter_nm=length,
        perimeter_um=length / 1000.0,
        n_clusters=k,
        n_2opt_improvements=n_improvements,
        self_intersections_before=xi_before,
        self_intersections_after=xi_after,
        warnings=warnings_,
        health=health,
        n_starts=n_starts,
        start_lengths_nm=start_lengths,
        centre=contour_centre(centroids[order]) if xi_after == 0 else None,
    )


# ============================================================================
# Cluster area and effective radius (parameters 3 and 4)
# ============================================================================

@dataclass
class ClusterAreaResult:
    """Convex-hull areas and effective radii for all clusters of one axon."""

    labels: NDArray[np.intp]             # cluster label per entry
    areas_nm2: NDArray[np.float64]
    r_eff_nm: NDArray[np.float64]
    n_locs: NDArray[np.intp]             # localizations per cluster
    degenerate_labels: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def median_area_nm2(self) -> Optional[float]:
        return float(np.median(self.areas_nm2)) if self.areas_nm2.size else None

    @property
    def median_r_eff_nm(self) -> Optional[float]:
        return float(np.median(self.r_eff_nm)) if self.r_eff_nm.size else None


def effective_radius_nm(area_nm2: NDArray[np.float64]) -> NDArray[np.float64]:
    """r_eff = sqrt(A / pi) -- radius of the circle of equivalent area."""
    return np.sqrt(np.asarray(area_nm2, dtype=float) / np.pi)


def compute_cluster_areas(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    exclude_labels: Optional[set] = None,
) -> ClusterAreaResult:
    """
    Convex-hull area and effective radius for every retained cluster.

    Parameters
    ----------
    x, y : localization coordinates in nm (the axial slab that was clustered).
    labels : DBSCAN labels (-1 = noise, excluded).
    exclude_labels : cluster labels to skip (the automatically detected bad
        clusters from ``tools.cluster_quality``).

    Returns
    -------
    ClusterAreaResult

    Notes
    -----
    A convex hull needs at least 3 non-collinear points. With min_samples=10
    that is normally satisfied, but a perfectly collinear cluster raises
    QhullError; such clusters are recorded in ``degenerate_labels`` and given
    area 0 rather than silently dropped, so the cluster count stays
    consistent with the perimeter reconstruction.
    """
    exclude = exclude_labels or set()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    out_labels: List[int] = []
    areas: List[float] = []
    n_locs: List[int] = []
    degenerate: List[int] = []

    for label in np.unique(labels):
        if label == -1 or int(label) in exclude:
            continue
        mask = labels == label
        pts = np.column_stack([x[mask], y[mask]])
        out_labels.append(int(label))
        n_locs.append(int(mask.sum()))

        if len(pts) < 3:
            areas.append(0.0)
            degenerate.append(int(label))
            continue
        try:
            hull = ConvexHull(pts)
            # For 2D inputs scipy's ConvexHull.volume IS the enclosed area
            # (and .area is the perimeter) -- a well-known footgun.
            areas.append(float(hull.volume))
        except QhullError:
            areas.append(0.0)
            degenerate.append(int(label))

    areas_arr = np.array(areas, dtype=float)
    warnings_: List[str] = []
    if degenerate:
        warnings_.append(
            f"{len(degenerate)} cluster(s) were geometrically degenerate "
            f"(collinear or <3 localizations) and were assigned area 0: "
            f"labels {degenerate}."
        )

    return ClusterAreaResult(
        labels=np.array(out_labels, dtype=np.intp),
        areas_nm2=areas_arr,
        r_eff_nm=effective_radius_nm(areas_arr),
        n_locs=np.array(n_locs, dtype=np.intp),
        degenerate_labels=degenerate,
        warnings=warnings_,
    )
