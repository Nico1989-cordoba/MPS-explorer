# -*- coding: utf-8 -*-
"""
Automatic bad-cluster detection for DBSCAN clusters of βII-spectrin
localizations.

Replaces the manual "click on a cluster center to mark it as bad" workflow
(``MPS_explorer.rx`` / ``update_display_after_cluster_removal``) with two
objective, automatic criteria that can be combined:

1. Edge-touching: a cluster with at least one localization within
   ``edge_margin_nm`` of the ROI boundary is considered incomplete — its
   centroid and convex-hull area are biased by the arbitrary ROI cut, which
   would directly bias the reconstructed perimeter (parameter 2) and the
   cluster-area distribution (parameter 3).
2. DBCV (Density-Based Clustering Validation) per-cluster validity index —
   a data-driven quality score for each cluster's shape/density, computed
   with ``hdbscan.validity.validity_index``. Low-scoring clusters look more
   like noise fragments than a real spectrin tetramer footprint.

The area-based / size-based filtering some SMLM pipelines use is
deliberately NOT implemented here: Gazal et al. (2026) explicitly keep the
larger clusters in their analysis, interpreting them as possible spectrin
oligomers rather than as artifacts (see cluster-area discussion, median
1965 nm^2 with a right tail). Filtering by size would silently contradict
the paper's own interpretation.

@author: Nicolás (ngomez) + Claude
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from tools.mps_settings import DEFAULT_DBCV_THRESHOLD

try:
    from hdbscan.validity import validity_index
    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - hdbscan is a hard requirement.txt dep,
    _HDBSCAN_AVAILABLE = False


# ============================================================================
# ROI boundary description
# ============================================================================
#
# The live GUI has three ROI shapes (MPS_explorer.update_ROI): circular,
# square, and polygon (drawn interactively or loaded). Edge-touching needs a
# distance-to-boundary function for whichever shape produced the currently
# loaded ROI. Represent it with a tiny tagged tuple so this module has zero
# dependency on the PyQt ROI widget objects themselves (those live in
# MPS_explorer.py) — this module only needs plain numbers.

@dataclass(frozen=True)
class CircularROI:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class SquareROI:
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class PolygonROI:
    # (N, 2) array of vertices, in order, closed or open (both handled).
    vertices: NDArray[np.float64]


ROIShape = Union[CircularROI, SquareROI, PolygonROI]


def _distance_to_boundary_circle(
    x: NDArray[np.float64], y: NDArray[np.float64], roi: CircularROI
) -> NDArray[np.float64]:
    """Distance from each point to the circle boundary (positive = inside)."""
    r = np.hypot(x - roi.center_x, y - roi.center_y)
    return roi.radius - r


def _distance_to_boundary_square(
    x: NDArray[np.float64], y: NDArray[np.float64], roi: SquareROI
) -> NDArray[np.float64]:
    """Distance from each point to the nearest square edge (positive = inside)."""
    d_left = x - roi.xmin
    d_right = roi.xmax - x
    d_bottom = y - roi.ymin
    d_top = roi.ymax - y
    return np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))


def _distance_to_boundary_polygon(
    x: NDArray[np.float64], y: NDArray[np.float64], roi: PolygonROI
) -> NDArray[np.float64]:
    """
    Signed distance from each point to the nearest polygon edge.

    Positive well inside the polygon, close to zero near an edge. Sign is
    not corrected for points outside the polygon (points passed in here are
    already known to be inside the ROI, since they came out of
    ``update_ROI``'s point-in-polygon filter) — only the *magnitude* near
    zero (i.e. "close to some edge") matters for edge-touching detection.
    """
    verts = roi.vertices
    n = len(verts)
    points = np.column_stack([x, y])
    min_dist = np.full(len(points), np.inf)

    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq == 0:
            dist = np.hypot(points[:, 0] - p1[0], points[:, 1] - p1[1])
        else:
            t = np.clip(((points - p1) @ seg) / seg_len_sq, 0.0, 1.0)
            proj = p1 + t[:, None] * seg
            dist = np.hypot(points[:, 0] - proj[:, 0], points[:, 1] - proj[:, 1])
        min_dist = np.minimum(min_dist, dist)

    return min_dist


def distance_to_roi_boundary(
    x: NDArray[np.float64], y: NDArray[np.float64], roi: ROIShape
) -> NDArray[np.float64]:
    """
    Distance from each (x, y) point to the ROI boundary.

    For circular and square ROIs the sign is meaningful (positive =
    strictly inside). For polygon ROIs only the magnitude is meaningful
    (see ``_distance_to_boundary_polygon``) since points here are assumed
    already ROI-filtered.

    Parameters
    ----------
    x, y : arrays of the same length
    roi : CircularROI | SquareROI | PolygonROI

    Returns
    -------
    Array of distances, same length as x.
    """
    if isinstance(roi, CircularROI):
        return _distance_to_boundary_circle(x, y, roi)
    if isinstance(roi, SquareROI):
        return _distance_to_boundary_square(x, y, roi)
    if isinstance(roi, PolygonROI):
        return _distance_to_boundary_polygon(x, y, roi)
    raise TypeError(f"Unknown ROI shape type: {type(roi)!r}")


def points_in_polygon(
    points: NDArray[np.float64], polygon: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """
    Ray casting: True for each (x, y) row of ``points`` inside ``polygon``.

    ``polygon`` is an (M, 2) array of vertices in order; the closing edge
    is implied.
    """
    points = np.asarray(points, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    if len(polygon) < 3:
        return inside       # no area: a polygon being rebuilt, say
    p1 = polygon[0]
    for i in range(len(polygon)):
        p2 = polygon[(i + 1) % len(polygon)]
        ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
        in_band = (y >= ymin) & (y < ymax)
        dy = p2[1] - p1[1]
        if dy != 0 and np.any(in_band):
            t = (y[in_band] - p1[1]) / dy
            x_cross = p1[0] + t * (p2[0] - p1[0])
            inside[in_band] ^= x[in_band] <= x_cross
        p1 = p2
    return inside


def points_in_roi(
    x: NDArray[np.float64], y: NDArray[np.float64], roi: ROIShape
) -> NDArray[np.bool_]:
    """
    Which points an ROI selects, by the same rules as the main window: a
    circle includes its boundary, a square excludes its edges.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if isinstance(roi, CircularROI):
        return np.hypot(x - roi.center_x, y - roi.center_y) <= roi.radius
    if isinstance(roi, SquareROI):
        return ((x > roi.xmin) & (x < roi.xmax)
                & (y > roi.ymin) & (y < roi.ymax))
    if isinstance(roi, PolygonROI):
        return points_in_polygon(np.column_stack([x, y]), roi.vertices)
    raise TypeError(f"Unknown ROI shape type: {type(roi)!r}")


# ============================================================================
# Criterion 1: edge-touching clusters
# ============================================================================

def find_edge_touching_clusters(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    roi: Optional[ROIShape],
    edge_margin_nm: float = 25.0,
) -> Set[int]:
    """
    Identify DBSCAN cluster labels with at least one localization within
    ``edge_margin_nm`` of the ROI boundary.

    Parameters
    ----------
    x, y : localization coordinates in nm (the ROI-filtered set that was
        clustered, i.e. ``self.xroi``/``self.yroi`` in MPS_explorer).
    labels : DBSCAN cluster labels for each (x, y) point (-1 = noise).
    roi : ROI boundary description, or None to skip this criterion
        entirely (returns an empty set).
    edge_margin_nm : distance threshold in nm. Default 25 nm matches the
        DBSCAN epsilon used for spectrin clusters (Gazal et al. 2026) — a
        cluster core point closer than one epsilon-radius to the boundary
        could plausibly have had neighbours cut off by the ROI edge.

    Returns
    -------
    Set of cluster labels (excluding -1/noise) flagged as edge-touching.
    """
    if roi is None:
        return set()

    dist = distance_to_roi_boundary(x, y, roi)
    near_edge = np.abs(dist) <= edge_margin_nm

    bad: Set[int] = set()
    for label in np.unique(labels):
        if label == -1:
            continue
        if np.any(near_edge[labels == label]):
            bad.add(int(label))
    return bad


# ============================================================================
# Criterion 2: DBCV per-cluster validity index
# ============================================================================

def compute_dbcv_per_cluster(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> Dict[int, float]:
    """
    Compute the DBCV (Density-Based Clustering Validation) per-cluster
    validity index for an existing DBSCAN labelling.

    Uses ``hdbscan.validity.validity_index(..., per_cluster_scores=True)``,
    the same function already prototyped in this repo for parameter-search
    (``tools/test_DBCV_data_axons.py``), but applied here to a *fixed*
    (eps, min_samples) labelling rather than to select the labelling
    itself.

    Parameters
    ----------
    x, y : localization coordinates (nm).
    labels : DBSCAN cluster labels (-1 = noise).

    Returns
    -------
    Dict mapping cluster label -> per-cluster DBCV score (roughly in
    [-1, 1]; higher is better). Noise (-1) is not included. Returns an
    empty dict if hdbscan is unavailable, if there are fewer than 2
    clusters (DBCV is undefined for a single cluster), or if
    validity_index raises internally (e.g. degenerate cluster geometry).
    """
    if not _HDBSCAN_AVAILABLE:
        return {}

    unique_labels = [lab for lab in np.unique(labels) if lab != -1]
    if len(unique_labels) < 2:
        # DBCV needs at least 2 clusters to be meaningful.
        return {}

    X = np.column_stack([x, y]).astype(np.double)
    try:
        _overall_index, per_cluster_scores = validity_index(
            X, labels.astype(np.intp), per_cluster_scores=True
        )
    except Exception:
        return {}

    # per_cluster_scores is indexed 0..max(label); noise is excluded from
    # the labels DBCV considers, but the returned array is aligned to
    # cluster label values 0..n_clusters-1 (DBSCAN labels clusters
    # contiguously from 0, so this alignment holds as long as labels are
    # the direct output of sklearn's DBSCAN).
    scores: Dict[int, float] = {}
    for lab in unique_labels:
        idx = int(lab)
        if 0 <= idx < len(per_cluster_scores):
            scores[idx] = float(per_cluster_scores[idx])
    return scores


def find_low_dbcv_clusters(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    dbcv_threshold: float = DEFAULT_DBCV_THRESHOLD,
) -> Tuple[Set[int], Dict[int, float]]:
    """
    Identify cluster labels whose DBCV per-cluster validity score is below
    ``dbcv_threshold``.

    Parameters
    ----------
    dbcv_threshold : default -1.0, i.e. OFF. DBCV scores are roughly
        bounded in [-1, 1], and were measured (on real axon ROIs, see
        validate_cluster_quality.py) to correlate with log10(cluster area)
        at -0.78 to -0.79: large clusters score low almost regardless of
        shape, because DBCV measures density cohesion and a large cluster
        is, by construction, less dense than a compact one with the same
        point count. At the previous default of 0.0 this removed the
        single largest cluster in every test axon checked -- exactly the
        clusters Gazal et al. (2026) interpret as spectrin oligomers and
        explicitly keep. Raise this only deliberately, aware it doubles as
        a size filter.

    Returns
    -------
    (bad_labels, all_scores) — the flagged set, and the full per-cluster
    score dict (for logging/inspection regardless of the threshold).
    """
    scores = compute_dbcv_per_cluster(x, y, labels)
    bad = {lab for lab, score in scores.items() if score < dbcv_threshold}
    return bad, scores


# ============================================================================
# Combined criterion
# ============================================================================

@dataclass
class BadClusterReport:
    """Full accounting of why each bad cluster was flagged, for logging
    and for the results panel (so the user can see *why* a cluster was
    dropped, not just that it was)."""
    bad_labels: Set[int]
    edge_touching: Set[int]
    low_dbcv: Set[int]
    dbcv_scores: Dict[int, float]
    # Set when the edge-touching criterion was disabled by the runaway
    # guard (see identify_bad_clusters). Carries a human-readable reason
    # that the GUI must surface to the user — silently keeping every
    # cluster would be just as misleading as silently dropping them all.
    edge_criterion_disabled: Optional[str] = None


def identify_bad_clusters(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    roi: Optional[ROIShape],
    edge_margin_nm: float = 25.0,
    dbcv_threshold: float = DEFAULT_DBCV_THRESHOLD,
    max_edge_removal_fraction: float = 0.5,
) -> BadClusterReport:
    """
    Combined automatic bad-cluster criterion: a cluster is "bad" if it is
    edge-touching OR has a low DBCV score (union of both criteria).

    This replaces the manual click-to-remove workflow
    (``MPS_explorer.rx``). Downstream parameters (perimeter, area,
    occupancy, 1NN, randomization) should be computed on the surviving
    ("good") clusters only.

    Runaway guard
    -------------
    In an axon cross-section the βII-spectrin clusters form a *ring* at the
    axon periphery. If the user draws the ROI snugly around that ring
    (which the interactive polygon tool makes easy, and which is the
    natural way to isolate one axon), then essentially every cluster sits
    within ``edge_margin_nm`` of the ROI boundary and the edge criterion
    removes the entire axon. Measured on this repo's own example data
    (ROI6 βII-spectrin): a polygon traced along the cluster ring flags
    43/43 clusters (100%); dilating it +50 nm still flags 79%.

    Silently deleting every cluster would produce an empty analysis that
    looks like a data problem rather than a criterion problem. So when the
    edge criterion would remove more than ``max_edge_removal_fraction`` of
    all clusters, it is disabled for this ROI, the DBCV criterion is used
    alone, and the reason is recorded in
    ``BadClusterReport.edge_criterion_disabled`` for the GUI to surface.

    Parameters
    ----------
    x, y : ROI-filtered localizations that were clustered.
    labels : DBSCAN labels for those localizations.
    roi : ROI boundary, or None to skip the edge-touching criterion.
    edge_margin_nm : see ``find_edge_touching_clusters``.
    dbcv_threshold : see ``find_low_dbcv_clusters``.
    max_edge_removal_fraction : guard threshold, default 0.5 (50%).

    Returns
    -------
    BadClusterReport with the union of bad labels plus a breakdown of
    which criterion(s) flagged each one.
    """
    edge_bad = find_edge_touching_clusters(x, y, labels, roi, edge_margin_nm)
    dbcv_bad, dbcv_scores = find_low_dbcv_clusters(x, y, labels, dbcv_threshold)

    n_clusters = len([lab for lab in np.unique(labels) if lab != -1])
    disabled_reason: Optional[str] = None

    if n_clusters > 0 and len(edge_bad) > max_edge_removal_fraction * n_clusters:
        disabled_reason = (
            f"Edge-touching criterion disabled: it flagged "
            f"{len(edge_bad)}/{n_clusters} clusters "
            f"({100 * len(edge_bad) / n_clusters:.0f}%), above the "
            f"{100 * max_edge_removal_fraction:.0f}% guard. This usually means "
            f"the ROI was drawn tightly around the spectrin ring, so the ring "
            f"itself lies on the ROI boundary. Only the DBCV criterion was "
            f"applied. Redraw the ROI with more margin around the axon if you "
            f"want edge-based removal to apply."
        )
        edge_bad = set()

    return BadClusterReport(
        bad_labels=edge_bad | dbcv_bad,
        edge_touching=edge_bad,
        low_dbcv=dbcv_bad,
        dbcv_scores=dbcv_scores,
        edge_criterion_disabled=disabled_reason,
    )


def good_cluster_centroids(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    labels: NDArray[np.int64],
    bad_labels: Set[int],
) -> NDArray[np.float64]:
    """
    Compute (K, 2) centroids for every cluster label NOT in ``bad_labels``
    and not noise (-1).

    Mirrors the centroid computation already in ``MPS_explorer.cluster()``
    (``np.mean(cluster_points, axis=0)``, rounded to 2 decimals) so that
    swapping in automatic bad-cluster removal produces numerically
    identical centroids to today's manual flow for whatever clusters
    survive.
    """
    centroids: List[NDArray[np.float64]] = []
    for label in np.unique(labels):
        if label == -1 or int(label) in bad_labels:
            continue
        mask = labels == label
        centroids.append(np.array([np.mean(x[mask]), np.mean(y[mask])]))
    if not centroids:
        return np.empty((0, 2))
    return np.around(np.array(centroids), decimals=2)
