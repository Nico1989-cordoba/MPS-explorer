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
  hull's effective radius. Over 13,000 simulated healthy contours the
  deepest vertex reaches 0.29 (circle, centres scattered by up to 80 nm
  on a 1.6 um radius), 0.21 (gaps of up to 160 degrees), 0.21 (a dent
  half the radius deep), 0.305 (an oval squeezed to a waist a third of
  its width) and 0.386 (an hourglass with a waist a quarter of its
  width). A cluster planted inside the ring sits at 0.63. The limit is
  0.40: above every healthy shape simulated, below the planted ones.

  That margin is bought with sensitivity. At 0.40 the check only sees a
  centre more than ~0.6 um inside a 1.6 um axon, which is far blunter
  than a second channel would be -- the tubulin mask flags an interior
  at 250 nm. It is meant to catch the contours that are plainly wrong
  without ever calling a healthy one broken, not to measure the
  interior.

  longest step over the median step, for the chord a gap forces. This
  one separates poorly and its limit is correspondingly loose: random
  angular spacing on a ring with no gap at all already reaches 18.8, so
  the limit is 20, which sees a 90 degree gap but not a 40 degree one.

Two other numbers are reported and deliberately NOT flagged. The ratio
of the tour to the convex hull of the same centres is what first made
this dataset look wrong, but it depends on the number of clusters and on
how far they scatter off a smooth outline: a perfectly correct contour
of a ring with 140 centres scattered by 80 nm already reads 1.9, which
is what axon 7 of April reads. And the share of the length carried by
the few steps above 3x the median runs near 30 % on a healthy ring.
Neither can carry a threshold; both are useful next to one that can.

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
# One step this many times the median is a chord across the axon. Wide,
# because random angular spacing alone reaches 18.8 on a gapless ring.
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

    @property
    def hull_perimeter_um(self) -> float:
        return self.hull_perimeter_nm / 1000.0

    @property
    def depth_limit_nm(self) -> float:
        """How deep a vertex must be before it is counted as interior."""
        return DEEP_VERTEX_FRACTION * self.hull_radius_nm

    @property
    def has_interior_vertices(self) -> bool:
        """Some centres are deeper inside than a ring's ever are."""
        return self.n_deep_vertices > 0

    @property
    def bridges_a_gap(self) -> bool:
        """One step is long enough to be a chord across the axon."""
        return self.max_over_median > MAX_OVER_MEDIAN_LIMIT


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
    three, or all on one line) and no hull exists to compare against.
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
    # calibre for the irregular cross-sections of the sciatic nerve. A
    # sliver of a hull would give a radius near zero and make every vertex
    # look deep, so it is treated as degenerate instead.
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

    if health.has_interior_vertices:
        health.warnings.append(
            f"{health.n_deep_vertices} of the {len(contour)} centres on "
            f"this contour sit more than {health.depth_limit_nm:,.0f} nm "
            f"inside their own convex hull, the deepest by "
            f"{health.max_depth_nm:,.0f} nm. No simulated ring reaches that "
            f"depth -- not with a gap of 160 degrees, not with a dent half "
            f"the radius deep, not on a cross-section as flat as a peanut "
            f"whose waist is a third of its width. So either those centres "
            f"are not on the membrane, or this axon is more concave than "
            f"any of those. Look at the contour before using the "
            f"perimeter: it is {health.tour_over_hull:.2f} times the hull "
            f"of the same centres ({health.hull_perimeter_um:.2f} um), and "
            f"clusters per um, occupancy and the randomization all follow "
            f"it."
        )
    if health.bridges_a_gap:
        health.warnings.append(
            f"One step of the contour is {health.max_over_median:.0f} times "
            f"the median ({health.edge_max_nm:,.0f} nm against "
            f"{health.edge_median_nm:,.0f} nm). Random angular spacing "
            f"alone reaches {MAX_OVER_MEDIAN_LIMIT:.0f} on a ring with no "
            f"gap at all; past that the contour is closing a gap with a "
            f"chord, which cuts across the axon and puts every "
            f"localization it passes on the wrong side of the contour."
        )
    return health


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
        return PerimeterResult(
            order=order,
            contour=centroids[order],
            perimeter_nm=_tour_length(centroids, order),
            perimeter_um=_tour_length(centroids, order) / 1000.0,
            n_clusters=k,
            n_2opt_improvements=0,
            self_intersections_before=before,
            self_intersections_after=before,
            warnings=(["Manual contour ordering supplied by the user."]
                      + (health.warnings if health else [])),
            health=health,
        )

    order = _polar_angle_order(centroids)
    xi_before = _count_self_intersections(centroids, order)

    n_improvements = 0
    if refine:
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
            f"point(s). The perimeter is unreliable; consider correcting the "
            f"contour manually."
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
