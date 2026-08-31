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

    if custom_order is not None:
        order = np.asarray(custom_order, dtype=np.intp)
        if sorted(order.tolist()) != list(range(k)):
            raise ValueError(
                "custom_order must be a permutation of all centroid indices."
            )
        before = _count_self_intersections(centroids, order)
        return PerimeterResult(
            order=order,
            contour=centroids[order],
            perimeter_nm=_tour_length(centroids, order),
            perimeter_um=_tour_length(centroids, order) / 1000.0,
            n_clusters=k,
            n_2opt_improvements=0,
            self_intersections_before=before,
            self_intersections_after=before,
            warnings=["Manual contour ordering supplied by the user."],
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
