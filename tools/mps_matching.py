# -*- coding: utf-8 -*-
"""
Ring-to-ring matching within a tolerance tau, its arc-shift null, the
eclipse curve over tau, the joint null per axon and the columns of one
axon: milestone H3 of the research plan 03_plan.md
(S3.1-S3.2, S3.6) on the rings that ``tools.mps_columns.build_rings``
returns (DECISIONES D-02, D-03, D-04, D-11, D-22).

What is here
------------
``RingGeometry``, ``ring_geometry``, ``ring_geometry_from_centroids``: a
ring as its clusters, the closed polygon through them (the tour) and the
smooth closed curve through the polygon's vertices, with the curve arc
of every cluster. ``SmoothPath``, ``smooth_closed_path``,
``point_on_curve`` (alias ``point_on_polygon``), ``point_on_path``,
``arc_shift``: the interpolating periodic cubic spline the null walks
(D-04, D-24a; ``NULL_KINDS``/``null_kind=``: D-25, H5; ``NULL_KINDS_ALL``: H5-B). ``BIG_COST``, ``match_rings``:
one-to-one assignment within tau, maximum cardinality then minimum
total distance (D-02, D-11, D-24b). ``RingPairMatch``, ``eclipse_test``:
one pair of rings against the arc-shift null with reflection (D-04,
D-11). ``EclipseCurve``, ``eclipse_curve``: the same over a grid of tau,
ONE set of shifts for every tau (D-03), exchangeable global test
(D-24c). ``JointNull``, ``axon_joint_null``, ``joint_shift_statistic``:
one null per axon (02 S2.2 B3, D-24d). ``Column``, ``build_columns``:
chains through adjacent matches. ``AxonColumnsResult``,
``analyze_columns``: the per-axon driver; tau_0 and the grids come from
the parameters file, nothing here (D-22).

Why a separate module: ``tools.mps_columns`` is the data model and the
rings (H1, H2); this is the statistics on top of it. No Qt, no
matplotlib. Every function is deterministic given its seed. The design
reasoning is in the docstring of the function it concerns; the
amendments D-24a-e (review of 2026-09-23, DECISIONES.md) are named
where they apply, as are the two points of the re-review of the same
day: the D-24d window measured against the tour origins
(``_redraw_relative``, fixed here) and the spline null on clusters
that scatter about the membrane (``arc_shift``, a pending design
point of D-24a).

@author: Nicolas (ngomez) + Claude
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from typing import Mapping  # H5-B, on its own line so that the diff against HEAD f2e5cf6 stays textually additive

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from tools.mps_columns import SUSPECT_MARKS, Cluster, ColumnsParams, Ring, RingsResult
from tools.mps_geometry import reconstruct_perimeter
from tools.mps_randomization import smooth_contour_bspline

__all__ = [
    "BIG_COST",
    "CONTOUR_TOL_NM",
    "ENVELOPE_PERCENTILES",
    "JOINT_SPAWN_KEY",
    "NULL_KINDS",
    "NULL_KINDS_ALL",
    "PATH_MIN_SAMPLES",
    "PATH_MIN_SAMPLES_PER_EDGE",
    "PATH_STEP_NM",
    "PATH_VERTEX_TOL_NM",
    "AxonColumnsResult",
    "Column",
    "EclipseCurve",
    "JointNull",
    "RingGeometry",
    "RingPairMatch",
    "SmoothPath",
    "analyze_columns",
    "arc_shift",
    "assign_within_tau",
    "axon_joint_null",
    "build_columns",
    "cyclic_arc_distance_nm",
    "eclipse_curve",
    "eclipse_test",
    "joint_shift_statistic",
    "leave_one_out_membranes",
    "match_arcs",
    "match_rings",
    "point_on_curve",
    "point_on_path",
    "point_on_polygon",
    "pooled_membrane_path",
    "ring_geometry",
    "ring_geometry_from_centroids",
    "smooth_closed_path",
]

# Cost of a pair farther apart than tau in the assignment matrix. It is
# never chosen: leaving both clusters on their dummies costs 2 M = 2 tau
# (K_a + K_b + 1) (``_assign_within_tau``, D-24b), which is below 1e9 nm
# for any ring this program meets (tau 1000 nm and 500 000 clusters
# would be needed), and the pairs the solver returns are filtered on
# D <= tau anyway. Without this truncation the dummies would pair two
# clusters up to 2 M apart (02_investigacion B3).
BIG_COST = 1e9

# A ring's contour holds its centroids rounded to 0.01 nm
# (tools.cluster_quality.good_cluster_centroids), so a vertex and the
# exact ``Cluster.centroid_nm`` agree to 0.005 nm per axis. A larger
# difference means the contour was not built from these clusters.
CONTOUR_TOL_NM = 0.01

# Spawn key of the joint null's seed sequence. The H3 specification wrote
# ``spawn_key=("joint",)``; ``numpy.random.SeedSequence`` accepts
# non-negative integers only (numpy 2.4.6: ValueError "unrecognized seed
# string"), so the word is encoded as the integer its four ASCII bytes
# spell, big-endian. Distinct from every (ring_a, ring_b) key of
# ``eclipse_test``, which are small ring indices.
JOINT_SPAWN_KEY = int.from_bytes(b"join", "big")

# Pointwise envelope of the null eclipse curve, in percent.
ENVELOPE_PERCENTILES: Tuple[float, float] = (2.5, 97.5)

# Sampling of the smooth closed path (``smooth_closed_path``, D-24a): the
# spline is tabulated as a dense polyline with, per polygon edge, at
# least PATH_MIN_SAMPLES_PER_EDGE samples and one sample per PATH_STEP_NM
# of the edge, and at least PATH_MIN_SAMPLES rows in total. A 2 nm
# sub-chord on a membrane of radius 670 nm has a sagitta of 2^2 / (8 x
# 670) = 0.0007 nm, so walking the polyline is the curve to < 1e-3 nm on
# rings whose curvature radius is >= 670 nm (the harness measures the
# table against a 1 nm first-principles spline: 7e-4 nm; the re-review
# of 2026-09-23 against a 1e5-sample splprep evaluation: 8e-4 nm on
# membrane-smooth rings, up to ~4e-3 nm between vertices that scatter
# about the membrane, where the spline bends more tightly). All of it is
# negligible against tau (60-80 nm). The spline passes through every
# vertex; the largest distance between a vertex and the spline evaluated
# at its parameter is checked against PATH_VERTEX_TOL_NM before the
# vertex is written into its row exactly (measured 1e-6 nm on a measured
# ring, 1e-10 on a synthetic one; the tolerance is a guard against a
# degenerate fit, not a rounding budget).
PATH_STEP_NM = 2.0
PATH_MIN_SAMPLES_PER_EDGE = 50
PATH_MIN_SAMPLES = 2000
PATH_VERTEX_TOL_NM = 1e-6

# The two curves a null can move the clusters along (D-25, H5).
# "interpolating": the periodic cubic spline THROUGH the clusters
# (``RingGeometry.path``, D-24a; the H3 null). "conserved_offset": the
# SMOOTHING periodic B-spline of the contour (``RingGeometry.smooth_path``,
# ``smooth_contour_bspline`` with s = K, D-25's candidate remedy), along
# which every cluster keeps its own signed radial offset from that
# curve. Every function that draws or applies a null takes ``null_kind``
# (default "interpolating", which reproduces today's outputs bit for
# bit) and records it in its result.
NULL_KINDS: Tuple[str, ...] = ("interpolating", "conserved_offset")
# The third kind, "pooled_offset" (H5-B, candidate B of H5B_SPEC; D-25's
# remedy done right): the membrane the moved ring walks is the periodic
# P-spline of ``tools.mps_membrane.fit_membrane`` fitted to the centroids
# of EVERY OTHER ring of the axon -- leave-ring-out, so that the curve is
# independent of the moved ring's configuration under the null --
# attached to the geometry as ``RingGeometry.membrane``; every cluster
# keeps its signed offset from that curve (``membrane_offset_nm``) and
# the draws are made on its length and mean spacing (``eclipse_test``
# for the argument and the pilot numbers that motivate it). It is
# registered in ``NULL_KINDS_ALL``, the tuple ``_check_kind`` validates
# against, and NOT appended to ``NULL_KINDS``: ``validate_simulate_axon.py``
# (H5, frozen at 56 checks) asserts that ``NULL_KINDS`` is exactly the
# two H5 kinds, and H5B_SPEC's "in NULL_KINDS" cannot be honoured
# without breaking it (the deviation is recorded in
# the arc-null validator, kept with the research notes). Callers that enumerate the kinds a null can
# take (a grid cell's ``null_kind``) should read ``NULL_KINDS_ALL``.
NULL_KINDS_ALL: Tuple[str, ...] = NULL_KINDS + ("pooled_offset",)

# Rows of the dense smoothed-curve table on each side of the nearest
# sample whose sub-chords ``_project_on_path`` projects on exactly: the
# same window as ``tools.mps_unroll.PROJECTION_WINDOW_ROWS`` (8 rows =
# 16 nm of curve per side at ``PATH_STEP_NM``), for the same reason: the
# foot of the perpendicular from a point closer to the curve than its
# curvature radius lies on a sub-chord adjacent to the nearest sample.
# The helper is a copy of ``mps_unroll.project_on_path`` (validated in
# H4) because ``mps_unroll`` imports this module and the reverse import
# would be circular; both give the same (s, r) bit for bit.
_PROJECTION_WINDOW_ROWS = 8

# Null replicates whose distance matrices are held in memory at once:
# 128 x K_a x K_b x T booleans is 25 MB for two rings of 100 clusters
# over a 19-point tau grid, and the assignment loop below is per
# replicate anyway.
_BLOCK = 128


# ============================================================================
# Geometry of a ring for the matching
# ============================================================================

@dataclass
class SmoothPath:
    """
    The smooth closed curve through a ring's polygon vertices, tabulated
    as a dense polyline: the membrane along which the null moves the
    clusters (D-04, ``arc_shift``).

    ``points_nm`` are the samples in curve order, closed by the wrap to
    the first; ``vertex_row[v]`` is the row of polygon vertex v, which
    the table holds exactly. ``edges_nm`` are the sub-chord lengths
    (the closing one last), ``cum_nm`` the cumulative length at each
    row plus the closing length, ``length_nm`` the closed length. The
    arc coordinate of the null is the cumulative length along this
    table, and a point at arc s is found by ``_walk_polygon`` on it.
    """

    points_nm: NDArray[np.float64]      # (N, 2) dense samples of the curve, curve order
    vertex_row: NDArray[np.intp]        # (K,) row of polygon vertex v in points_nm
    edges_nm: NDArray[np.float64]       # (N,) sub-chord lengths, closing one last
    cum_nm: NDArray[np.float64]         # (N + 1,) cumulative length per row, then the closed length
    length_nm: float                    # closed length of the curve


@dataclass
class RingGeometry:
    """A ring as the matching sees it: its clusters, in ``Ring.clusters``
    order, the closed polygon through them (the tour) and the smooth
    closed curve through the polygon's vertices, along which the null
    moves the clusters (D-04, D-24a).

    ``contour_nm[v]`` is the vertex at tour position v and equals
    ``centroids_nm[order[v]]`` (to ``CONTOUR_TOL_NM`` when the polygon
    was built by ``build_rings``, exactly when built here from the
    centroids); it is kept for reporting and it is the input of the
    curve. ``path`` is the interpolating periodic cubic spline through
    the vertices, tabulated densely (``smooth_closed_path``); it is
    built here from ``contour_nm`` and never passed in, and so are
    ``arc_nm`` and ``length_nm``: ``arc_nm[i]`` is the cumulative CURVE
    length at the vertex of cluster i (``arc_nm[order[0]] == 0``, the
    arcs increase along the tour) and ``length_nm`` the closed curve
    length, the L of the null (``arc_shift``, ``_draw_shifts``). The
    polygon's own arcs and closed length (``Ring.length_nm``) are
    ``polygon_arc_nm`` and ``polygon_length_nm``. Both are derived from
    the contour by the dataclass itself so that a geometry can never
    carry arcs inconsistent with the curve its null walks.

    Why the curve and not the polygon (D-24a): a cluster shifted onto a
    chord sits inside a convex membrane by the chord's sagitta, so a
    polygon-path null matches fewer pairs than the vertices do and its
    zeta is biased in favour of columns (measured +0.19 to +0.24 sd at
    200 nm spacing on a 670 nm radius, +0.85 at 400 nm; see
    ``arc_shift``). ``usable`` says which clusters enter the matching
    (False = a cluster left out); ``suspect`` carries the H2 suspect
    mark of each cluster (any mark of ``SUSPECT_MARKS``; all False for a
    synthetic ring), so that ``eclipse_test`` can honour
    ``include_suspect`` on a geometry too. The polygon and the curve
    keep EVERY cluster, usable or not, because the membrane does not
    change when a cluster is doubted. ``labels`` are the DBSCAN labels,
    for reporting.

    The smoothed curve (D-25, H5): ``smooth_path`` is the SMOOTHING
    periodic B-spline of the contour -- ``smooth_contour_bspline`` with
    the pre-registered scale s = K and unit weights, fitted on the
    CLOSED contour (first vertex appended: handed the open contour,
    ``splprep(per=1)`` overwrites the last vertex with the first and
    leaves that cluster out of the fit, a 9 nm offset at K = 40 and 87
    nm at K = 20 on membrane-exact rings, measured 2026-09-25) -- and
    tabulated densely like ``path`` (``_smoothing_closed_path``); it is
    None with fewer than four distinct vertices or when the fit fails.
    ``smooth_arc_nm[i]`` and ``radial_offset_nm[i]`` are the arc of the
    projection of cluster i on that curve and its signed distance from
    it (+ outward: ``_project_on_path``, the projection of
    ``mps_unroll``), ``smooth_length_nm`` its closed length, the L of
    the conserved-offset null (``arc_shift(kind="conserved_offset")``);
    NaN when ``smooth_path`` is None. With s = K the curve nearly
    interpolates (rms residual <= 1 nm by construction: FITPACK's s is
    a summed squared residual in nm^2 with unit weights, so K nm^2 over
    K vertices is 1 nm rms, NOT one scatter sd per vertex), so on
    membrane-exact rings the offsets are a few nm at most and on rings
    whose clusters scatter about the membrane the curve follows the
    scatter and the offsets stay ~1 nm; what that buys the null is
    measured in ``arc_shift``. ``smoothing_nm2`` overrides s (the knob
    D-28 needs to decide the scale: s = K sigma^2 with sigma the
    clusters' lateral precision is scipy's own rule of thumb once the
    weights are 1 / sigma); every driver of this module leaves it None.

    The pooled membrane (H5-B): ``membrane`` is a curve estimated WITHOUT
    this ring -- the P-spline of ``tools.mps_membrane.fit_membrane``
    through the centroids of the axon's other rings
    (``pooled_membrane_path``, ``leave_one_out_membranes``), attached by
    ``eclipse_test(null_kind="pooled_offset", ...)``, ``axon_joint_null``
    and ``analyze_columns`` (``_attach_membrane``: a shallow copy of the
    geometry carrying it; the interpolating and smoothed curves are not
    rebuilt) or given at construction; None (the default) when no pooled
    null is asked, and nothing else changes. ``membrane_arc_nm[i]`` and
    ``membrane_offset_nm[i]`` are the arc of cluster i's projection on
    that curve and its signed offset from it (+ outward by the curve's
    own orientation, ``_membrane_sign``), ``membrane_length_nm`` its
    closed length, the L of the pooled-offset null; NaN without a
    membrane.
    """

    index: int
    centroids_nm: NDArray[np.float64]   # (K, 2) all clusters, ring.clusters order (axon frame)
    contour_nm: NDArray[np.float64]     # (K, 2) closed polygon through the same points, tour order
    order: NDArray[np.intp]             # (K,) order[v] = cluster index at vertex v
    usable: NDArray[np.bool_]           # (K,) clusters that enter the matching
    labels: NDArray[np.int64]           # (K,) DBSCAN labels, for reporting
    suspect: Optional[NDArray[np.bool_]] = None   # (K,) H2 suspect mark; all False when omitted
    smoothing_nm2: Optional[float] = None         # s of the smoothed curve (nm^2); None = the pre-registered K (D-25)
    membrane: Optional[SmoothPath] = field(default=None, repr=False)   # the leave-ring-out pooled membrane (H5-B); None = none attached
    arc_nm: NDArray[np.float64] = field(init=False)          # (K,) CURVE arc of each CLUSTER; arc_nm[order[0]] = 0
    length_nm: float = field(init=False)                      # closed CURVE length (the null's L)
    polygon_arc_nm: NDArray[np.float64] = field(init=False)  # (K,) polygon arc of each CLUSTER (cumulative chords)
    polygon_length_nm: float = field(init=False)              # closed polygon length (Ring.length_nm)
    path: SmoothPath = field(init=False, repr=False)          # the smooth curve, from contour_nm
    smooth_path: Optional[SmoothPath] = field(init=False, repr=False)   # the s = K smoothing curve (D-25); None below 4 vertices
    smooth_arc_nm: NDArray[np.float64] = field(init=False)   # (K,) arc of each CLUSTER's projection on smooth_path
    radial_offset_nm: NDArray[np.float64] = field(init=False)   # (K,) signed offset of each CLUSTER from smooth_path, + outward
    smooth_length_nm: float = field(init=False)               # closed length of smooth_path (the conserved-offset null's L)
    membrane_arc_nm: NDArray[np.float64] = field(init=False)     # (K,) arc of each CLUSTER's projection on membrane (NaN without one)
    membrane_offset_nm: NDArray[np.float64] = field(init=False)  # (K,) signed offset of each CLUSTER from membrane, + outward
    membrane_length_nm: float = field(init=False)                # closed length of membrane (the pooled-offset null's L)

    def __post_init__(self) -> None:
        """Validate the shapes, default ``suspect`` to no mark, build the
        smooth path from the contour and derive the curve and polygon
        arcs of every cluster from it; then the smoothed curve of D-25
        and every cluster's arc and radial offset on it."""
        self.centroids_nm = _as_points(self.centroids_nm, f"ring {self.index} centroids")
        self.contour_nm = _as_points(self.contour_nm, f"ring {self.index} contour")
        k = int(self.centroids_nm.shape[0])
        order = np.asarray(self.order, dtype=np.intp).ravel()
        if self.contour_nm.shape != (k, 2) or order.shape != (k,) or sorted(order.tolist()) != list(range(k)):
            raise ValueError(f"ring {self.index}: contour {self.contour_nm.shape} and order {order.shape} "
                             f"do not describe {k} clusters")
        self.order = order
        self.usable = _mask(self.usable, k, f"ring {self.index} usable")
        lab = np.asarray(self.labels, dtype=np.int64).ravel()
        if lab.shape != (k,):
            raise ValueError(f"ring {self.index}: {lab.size} labels for {k} clusters")
        self.labels = lab
        if self.suspect is None:
            self.suspect = np.zeros(k, dtype=bool)
        else:
            self.suspect = _mask(self.suspect, k, f"ring {self.index} suspect")
        self.path = smooth_closed_path(self.contour_nm)
        arc = np.empty(k, dtype=np.float64)
        arc[order] = self.path.cum_nm[self.path.vertex_row]
        self.arc_nm = arc
        self.length_nm = float(self.path.length_nm)
        _, cum = _polygon_arcs(self.contour_nm)
        poly = np.empty(k, dtype=np.float64)
        poly[order] = cum[:-1]
        self.polygon_arc_nm = poly
        self.polygon_length_nm = float(cum[-1])
        # D-25: the smoothing curve and the clusters' arcs / offsets on it (cluster order).
        s_smooth = float(k) if self.smoothing_nm2 is None else float(self.smoothing_nm2)
        if not (math.isfinite(s_smooth) and s_smooth >= 0.0):
            raise ValueError(f"ring {self.index}: smoothing_nm2 must be a finite value >= 0, got {self.smoothing_nm2!r}")
        self.smooth_path = _smoothing_closed_path(self.contour_nm, s_smooth) if k >= 4 else None
        if self.smooth_path is None:
            self.smooth_arc_nm = np.full(k, np.nan, dtype=np.float64)
            self.radial_offset_nm = np.full(k, np.nan, dtype=np.float64)
            self.smooth_length_nm = float("nan")
        else:
            s_arc, r_off = _project_on_path(self.smooth_path, self.centroids_nm, _outward_sign(self.contour_nm))
            self.smooth_arc_nm = s_arc
            self.radial_offset_nm = r_off
            self.smooth_length_nm = float(self.smooth_path.length_nm)
        self._derive_membrane()

    def _derive_membrane(self) -> None:
        """Every cluster's arc and signed offset on ``membrane`` (H5-B),
        NaN without one; the offset's sign is the membrane curve's own
        orientation (``_membrane_sign``), the one ``_shifted_positions``
        walks with, so that a cluster projects back with the same r."""
        k = int(self.centroids_nm.shape[0])
        if self.membrane is None:
            self.membrane_arc_nm = np.full(k, np.nan, dtype=np.float64)
            self.membrane_offset_nm = np.full(k, np.nan, dtype=np.float64)
            self.membrane_length_nm = float("nan")
            return
        if not isinstance(self.membrane, SmoothPath):
            raise TypeError(f"ring {self.index}: membrane must be a SmoothPath, got {type(self.membrane).__name__}")
        s_arc, r_off = _project_on_path(self.membrane, self.centroids_nm, _membrane_sign(self.membrane))
        self.membrane_arc_nm = s_arc
        self.membrane_offset_nm = r_off
        self.membrane_length_nm = float(self.membrane.length_nm)

    @property
    def n_clusters(self) -> int:
        return int(self.centroids_nm.shape[0])

    @property
    def n_usable(self) -> int:
        return int(np.count_nonzero(self.usable))

    @property
    def path_arc_nm(self) -> NDArray[np.float64]:
        """Alias of ``arc_nm`` (the curve arc of each cluster)."""
        return self.arc_nm

    @property
    def path_length_nm(self) -> float:
        """Alias of ``length_nm`` (the closed curve length, the null's L)."""
        return self.length_nm

    @property
    def mean_spacing_nm(self) -> float:
        """p_bar: the curve length per cluster, over ALL clusters (the
        polygon's vertices), which sets the null's minimum shift."""
        return self.length_nm / self.n_clusters if self.n_clusters else float("nan")


def _as_points(values: Any, what: str) -> NDArray[np.float64]:
    """``values`` as a finite (K, 2) float64 array (a copy), or ValueError."""
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{what}: expected (K, 2) coordinates, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what}: non-finite coordinate")
    return arr


def _polygon_arcs(contour: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Edge lengths of the closed polygon and the cumulative arc at each
    vertex, with the closing length as the (K + 1)-th entry."""
    nxt = np.roll(contour, -1, axis=0)
    edges = np.hypot(nxt[:, 0] - contour[:, 0], nxt[:, 1] - contour[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(edges)])
    return edges, cum


def _geometry_from_polygon(
    index: int,
    centroids: NDArray[np.float64],
    contour: NDArray[np.float64],
    order: NDArray[np.intp],
    usable: NDArray[np.bool_],
    labels: NDArray[np.int64],
    suspect: Optional[NDArray[np.bool_]] = None,
) -> RingGeometry:
    """Assemble a ``RingGeometry`` once the polygon is known (the curve
    and every arc are built by the dataclass itself)."""
    return RingGeometry(index=int(index), centroids_nm=centroids, contour_nm=contour,
                        order=order, usable=usable, labels=labels, suspect=suspect)


def _mask(values: Optional[Any], k: int, what: str) -> NDArray[np.bool_]:
    """A (K,) boolean mask from ``values`` (all True when None)."""
    if values is None:
        return np.ones(k, dtype=bool)
    mask = np.asarray(values, dtype=bool).ravel()
    if mask.shape != (k,):
        raise ValueError(f"{what}: expected {k} entries, got {mask.shape}")
    return mask


def ring_geometry_from_centroids(
    index: int,
    centroids_nm: NDArray[np.float64],
    *,
    labels: Optional[NDArray[np.int64]] = None,
    usable: Optional[NDArray[np.bool_]] = None,
) -> RingGeometry:
    """
    The geometry of a ring given only its cluster centroids: the polygon
    is ``tools.mps_geometry.reconstruct_perimeter(all_starts=True)``, the
    tour the MPS analysis builds for a ring (polar order, 2-opt from
    every start, shortest kept), so a synthetic ring is matched on
    exactly the polygon a measured ring would get.

    Parameters
    ----------
    index : int
        Ring index (it seeds the null of every pair the ring enters).
    centroids_nm : (K, 2)
        Cluster centroids, nm; K >= 3 (a closed contour needs three).
    labels : (K,) int, optional
        DBSCAN labels; 0..K-1 when omitted.
    usable : (K,) bool, optional
        Clusters that enter the matching; all when omitted.

    Raises
    ------
    ValueError
        Fewer than three centroids, a non-finite one, or a label/usable
        vector of the wrong length (``reconstruct_perimeter`` raises for
        the first).
    """
    centroids = _as_points(centroids_nm, f"ring {index}")
    k = centroids.shape[0]
    per = reconstruct_perimeter(centroids, all_starts=True)
    order = np.asarray(per.order, dtype=np.intp)
    contour = centroids[order]
    lab = (np.arange(k, dtype=np.int64) if labels is None
           else np.asarray(labels, dtype=np.int64).ravel())
    if lab.shape != (k,):
        raise ValueError(f"ring {index}: {lab.size} labels for {k} clusters")
    return _geometry_from_polygon(index, centroids, contour, order,
                                  _mask(usable, k, f"ring {index} usable"), lab)


def ring_geometry(ring: Ring, *, include_suspect: bool = True) -> RingGeometry:
    """
    The geometry of a measured ring from ``Ring.clusters`` (the exact
    centroids) and ``ring.analysis.perimeter`` (the tour), verified
    against each other.

    The contour is the one the ring analysis built (``Ring.contour_nm``,
    the rounded centroids in tour order) and ``order`` is that
    perimeter's ``order``; each vertex must lie within ``CONTOUR_TOL_NM``
    of the centroid it claims to be, or the contour does not belong to
    these clusters and a ValueError says so. The curve, its arcs and
    its length are built from the contour (``RingGeometry``);
    ``polygon_length_nm`` is ``Ring.length_nm`` to rounding (the same
    sum in a different order).

    ``usable`` is every cluster with ``include_suspect``; without it, the
    clusters carrying any mark of ``SUSPECT_MARKS`` (``Cluster.suspect``,
    H2) are left out of the matching -- but not of the polygon, which is
    the membrane and does not depend on which clusters are doubted.

    Raises
    ------
    ValueError
        No contour (``Ring.contour_nm`` is None: fewer than three kept
        clusters, or the perimeter failed), no perimeter in the
        analysis, a contour or order that does not describe
        ``ring.clusters``, or a vertex off its centroid by more than
        ``CONTOUR_TOL_NM``.
    """
    if ring.contour_nm is None:
        raise ValueError(f"ring {ring.index}: no contour (fewer than 3 kept clusters or "
                         "no perimeter); it cannot enter the matching")
    per = getattr(ring.analysis, "perimeter", None)
    if per is None:
        raise ValueError(f"ring {ring.index}: the ring analysis has no perimeter")
    clusters: List[Cluster] = list(ring.clusters)
    k = len(clusters)
    centroids = (np.array([np.asarray(cl.centroid_nm, dtype=np.float64).reshape(2)
                           for cl in clusters], dtype=np.float64)
                 if k else np.zeros((0, 2), dtype=np.float64))
    contour = np.array(ring.contour_nm, dtype=np.float64)
    order = np.asarray(per.order, dtype=np.intp).ravel()
    if k < 3 or contour.shape != (k, 2) or order.shape != (k,):
        raise ValueError(f"ring {ring.index}: contour {contour.shape} and order {order.shape} "
                         f"for {k} clusters")
    if sorted(order.tolist()) != list(range(k)):
        raise ValueError(f"ring {ring.index}: the perimeter order is not a permutation of the clusters")
    if not (np.isfinite(centroids).all() and np.isfinite(contour).all()):
        raise ValueError(f"ring {ring.index}: non-finite centroid or contour vertex")
    deviation = float(np.abs(contour - centroids[order]).max())
    if deviation > CONTOUR_TOL_NM:
        raise ValueError(f"ring {ring.index}: a contour vertex is {deviation:.3g} nm from the "
                         f"centroid it should be (limit {CONTOUR_TOL_NM} nm): the contour was "
                         "not built from these clusters")
    suspect = np.array([bool(cl.suspect) for cl in clusters], dtype=bool)
    usable = np.ones(k, dtype=bool) if include_suspect else ~suspect
    labels = np.array([int(cl.label) for cl in clusters], dtype=np.int64)
    return _geometry_from_polygon(ring.index, centroids, contour, order, usable, labels, suspect)


def _walk_polygon(
    contour: NDArray[np.float64],
    edges: NDArray[np.float64],
    cum: NDArray[np.float64],
    s_nm: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Points of the closed polygon at arc positions ``s_nm`` (any shape,
    already reduced modulo the length): the edge is found by bisection
    in the cumulative arcs and the point interpolated on it. A position
    at a vertex gives that vertex exactly (t = 0); a position that
    rounding pushed to the closing length gives the first vertex."""
    k = contour.shape[0]
    v = np.searchsorted(cum, s_nm, side="right") - 1
    v = np.clip(v, 0, k - 1)
    span = np.where(edges[v] > 0.0, edges[v], 1.0)
    t = np.clip((s_nm - cum[v]) / span, 0.0, 1.0)
    start = contour[v]
    return np.asarray(start + t[..., None] * (contour[(v + 1) % k] - start), dtype=np.float64)


def smooth_closed_path(
    contour_nm: NDArray[np.float64],
    *,
    step_nm: float = PATH_STEP_NM,
    min_samples_per_edge: int = PATH_MIN_SAMPLES_PER_EDGE,
    min_samples: int = PATH_MIN_SAMPLES,
) -> SmoothPath:
    """
    The smooth closed curve through the vertices of a closed polygon, as
    a dense polyline (D-24a): the interpolating periodic cubic spline
    ``scipy.interpolate.splprep([x, y], s=0, per=True)`` on a COPY of the
    closed contour (first vertex repeated at the end; ``splprep``
    modifies its input), with the parameter u FITPACK derives from the
    cumulative chord lengths (normalised to [0, 1]), evaluated with
    ``splev`` at ``max(min_samples_per_edge, ceil(edge / step_nm),
    ceil(min_samples / K))`` equally spaced u per edge, the vertex
    itself first. The spline interpolates, so ``splev`` at a vertex's u
    returns the vertex to rounding (checked against
    ``PATH_VERTEX_TOL_NM``, ValueError beyond it) and the vertex is
    then written into its row exactly: every vertex is a row of the
    table. The arc coordinate of the null is the cumulative length of
    this polyline (``SmoothPath.cum_nm``), i.e. the curve's arc length
    to the sagitta of a sub-chord: < 1e-3 nm at ``PATH_STEP_NM`` on
    rings whose curvature radius is >= 670 nm (the harness measures
    7e-4 nm against a 1 nm first-principles spline), up to ~4e-3 nm
    between vertices that scatter about the membrane (15 nm jitter,
    re-review of 2026-09-23). The curve interpolates the CONTOUR: on a
    measured ring the contour holds the centroids rounded to 0.01 nm
    (``CONTOUR_TOL_NM``), so the exact ``Cluster.centroid_nm`` lie on
    the curve to 0.005 nm, not exactly; on a synthetic ring built from
    the centroids they lie on it exactly.

    Why this curve: it is the C2 closed curve through the clusters, the
    natural estimate of a smooth membrane from the points on it, and
    the one D-24a pre-registers. On the synthetic 1500 x 1000 nm
    ellipse of ``validate_columns_h3`` it stays within 4 nm of the true
    membrane at 200 nm spacing (K = 40; the chords' midpoints are up to
    112 nm inside), and its null's mean zeta agrees with a shift along
    the exact ellipse within 0.06 sd at K = 40 and K = 20 (the polygon
    path: +0.19 to +0.24 and +0.85; harness section 4). Through
    vertices that scatter radially about the membrane (a jittered M5
    ring, real rings) the spline overshoots between the vertices by
    tens of nm, and the null it carries is then biased in the
    hypothesis direction on sparse rings (+0.13 sd at K = 40 and +0.22
    at K = 20 with 15 nm jitter; -0.09 on a denser ring, K = 60 on a
    1200 nm circle): the pending design point recorded in ``arc_shift``.

    Consecutive coincident vertices (a zero-length edge) share one knot
    and one row. Raises ValueError with fewer than three distinct
    vertices or a non-finite one.
    """
    contour = _as_points(contour_nm, "contour")
    k = contour.shape[0]
    edges, cum = _polygon_arcs(contour)
    if k < 3 or int(np.count_nonzero(edges > 0.0)) < 3:
        raise ValueError("smooth_closed_path: a closed curve needs at least three distinct vertices")
    step = float(step_nm)
    if not (math.isfinite(step) and step > 0.0):
        raise ValueError(f"smooth_closed_path: step_nm must be positive, got {step_nm}")
    n_min = max(max(1, int(min_samples_per_edge)), int(math.ceil(max(1, int(min_samples)) / k)))
    knots = np.flatnonzero(edges > 0.0)                  # vertices that start a real edge
    xy_knots = np.vstack([contour[knots], contour[knots[:1]]])
    # splprep on copies: it modifies its input; u = FITPACK's normalised
    # cumulative chord length (u_knots[-1] = 1).
    tck, u_knots = splprep([xy_knots[:, 0].copy(), xy_knots[:, 1].copy()], s=0, per=True)
    u_knots = np.asarray(u_knots, dtype=np.float64)
    worst = 0.0
    rows: List[NDArray[np.float64]] = []
    vertex_row = np.empty(k, dtype=np.intp)
    n_rows = 0
    for q, v in enumerate(knots.tolist()):
        n = max(n_min, int(math.ceil(edges[v] / step)))
        u = u_knots[q] + (u_knots[q + 1] - u_knots[q]) * np.arange(n, dtype=np.float64) / n
        block = np.column_stack(splev(u, tck)).astype(np.float64)
        worst = max(worst, float(np.abs(block[0] - contour[v]).max()))
        block[0] = contour[v]
        vertex_row[v] = n_rows
        rows.append(block)
        n_rows += n
    if worst > PATH_VERTEX_TOL_NM:
        raise ValueError(f"smooth_closed_path: the spline misses a vertex by {worst:.3g} nm "
                         f"(limit {PATH_VERTEX_TOL_NM} nm): degenerate contour")
    points = np.vstack(rows)
    for v in reversed(range(2 * k)):          # two passes: a run of coincident vertices may wrap
        if edges[v % k] <= 0.0:
            vertex_row[v % k] = vertex_row[(v + 1) % k]
    path_edges, path_cum = _polygon_arcs(points)
    return SmoothPath(points_nm=points, vertex_row=vertex_row, edges_nm=path_edges,
                      cum_nm=path_cum, length_nm=float(path_cum[-1]))


# ============================================================================
# The smoothed curve and the conserved radial offsets (D-25, H5)
# ============================================================================

def _outward_sign(contour: NDArray[np.float64]) -> float:
    """The sign that makes a radial offset positive on the side away from
    the polygon's interior: -1 for a counter-clockwise tour (positive
    shoelace area; the interior is on the left of the tangent and
    cross(t, p - proj) is positive on the left), +1 for a clockwise
    one -- ``-sign(signed area)``, the convention of
    ``mps_unroll.project_on_path``."""
    return -math.copysign(1.0, _signed_area_nm2(contour))


def _smoothing_closed_path(
    contour: NDArray[np.float64],
    smoothing: float,
    *,
    step_nm: float = PATH_STEP_NM,
    min_samples: int = PATH_MIN_SAMPLES,
) -> Optional[SmoothPath]:
    """
    The SMOOTHING periodic B-spline of a closed polygon as a dense
    polyline (D-25): ``tools.mps_randomization.smooth_contour_bspline``
    (``splprep(s=smoothing, per=1, k=3)``, unit weights) on the CLOSED
    contour -- the distinct consecutive vertices with the first one
    appended, so that FITPACK's overwrite of the last point with the
    first costs no vertex -- evaluated at ``max(min_samples, ceil(polygon
    length / step_nm))`` equally spaced values of the chord-length
    parameter u (row 0 is the smoothed image of vertex 0, the origin of
    the arc coordinate and of the reflection, as vertex 0 is for
    ``path``). ``smoothing`` is s: the pre-registered scale is K, the
    number of clusters (D-25), i.e. a summed squared residual of at
    most K nm^2 over the vertices, an rms residual of 1 nm.

    Unlike ``smooth_closed_path`` the vertices are NOT on the curve:
    ``vertex_row[v]`` is the row nearest to vertex v (for bookkeeping;
    the arc of a cluster is its projection, ``_project_on_path``).
    Returns None with fewer than four distinct vertices, when the fit
    fails (``smooth_contour_bspline`` then hands back the raw contour,
    recognised by its row count) or when it returns a non-finite or
    degenerate table: a ring without a smoothed curve has no
    conserved-offset null (``arc_shift`` raises ValueError for it).
    """
    edges, cum = _polygon_arcs(contour)
    knots = np.flatnonzero(edges > 0.0)
    if knots.size < 4:
        return None
    closed = np.vstack([contour[knots], contour[knots[:1]]])
    n = max(int(min_samples), int(math.ceil(float(cum[-1]) / float(step_nm))))
    try:
        points = np.asarray(smooth_contour_bspline(closed, n_samples=n, smoothing=float(smoothing)), dtype=np.float64)
    except Exception:  # noqa: BLE001 - a failed fit is "no smoothed curve", not a crash of the matching
        return None
    if points.shape != (n, 2) or not np.isfinite(points).all():
        return None
    path_edges, path_cum = _polygon_arcs(points)
    if not (path_cum[-1] > 0.0) or int(np.count_nonzero(path_edges > 0.0)) < 3:
        return None
    _, nearest = cKDTree(points).query(contour, k=1)
    return SmoothPath(points_nm=points, vertex_row=np.asarray(nearest, dtype=np.intp).reshape(-1),
                      edges_nm=path_edges, cum_nm=path_cum, length_nm=float(path_cum[-1]))


def _project_on_path(
    path: SmoothPath,
    points_nm: NDArray[np.float64],
    outward_sign: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    The nearest point of the dense curve table to each of ``points_nm``
    (n, 2): its curve arc ``s_nm`` in [0, L) and the signed offset
    ``r_nm`` of the point from the curve, + outward (``outward_sign``
    +1 or -1, see ``_outward_sign``). A copy of
    ``tools.mps_unroll.project_on_path`` (H4; the reverse import would
    be circular), same arithmetic, same (s, r) bit for bit: a
    ``cKDTree`` on the table gives the nearest sample, the point is
    projected exactly (clipped) on the sub-chords within
    ``_PROJECTION_WINDOW_ROWS`` rows of it on both sides and the
    smallest distance wins; s = ``cum_nm`` of the winning sub-chord plus
    the fraction along it, r = cross(t, p - proj) x sign with t the
    unit sub-chord. A point ON a table row has r = 0 exactly.
    """
    pts = np.asarray(points_nm, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"_project_on_path: expected (n, 2) points, got shape {pts.shape}")
    sign = float(outward_sign)
    a = np.asarray(path.points_nm, dtype=np.float64)
    n_rows = a.shape[0]
    length = float(path.length_nm)
    if n_rows < 3 or not (length > 0.0):
        raise ValueError("_project_on_path: the path needs at least three rows and a positive length")
    b = np.roll(a, -1, axis=0)
    ab = b - a                                                  # (N, 2) sub-chords
    edges = np.asarray(path.edges_nm, dtype=np.float64)
    edge2 = np.maximum(edges ** 2, 1e-300)
    cum = np.asarray(path.cum_nm, dtype=np.float64)
    _dist, nearest = cKDTree(a).query(pts, k=1)
    nearest = np.asarray(nearest, dtype=np.int64).reshape(-1)
    w = min(_PROJECTION_WINDOW_ROWS, n_rows // 2)
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


def _check_kind(null_kind: Any) -> str:
    """``null_kind`` as one of ``NULL_KINDS_ALL`` (the H5 kinds of
    ``NULL_KINDS`` and H5-B's "pooled_offset"), or ValueError."""
    kind = str(null_kind)
    if kind not in NULL_KINDS_ALL:
        raise ValueError(f"null_kind must be one of {NULL_KINDS_ALL}, got {null_kind!r}")
    return kind


def _null_curve(geom: RingGeometry, kind: str) -> Tuple[SmoothPath, NDArray[np.float64], float]:
    """(table, arc of every cluster, closed length) of the curve the null
    of ``kind`` walks: ``path`` / ``arc_nm`` / ``length_nm`` for
    "interpolating" (the very objects, so that kind's arithmetic is
    unchanged), ``smooth_path`` / ``smooth_arc_nm`` / ``smooth_length_nm``
    for "conserved_offset"; ValueError when the ring has no smoothed
    curve."""
    if kind == "interpolating":
        return geom.path, geom.arc_nm, geom.length_nm
    if kind == "pooled_offset":
        if geom.membrane is None:
            raise ValueError(f"ring {geom.index}: no pooled membrane attached: null_kind 'pooled_offset' needs "
                             "membrane=<SmoothPath> or other_rings=[every other ring of the axon] "
                             "(eclipse_test / eclipse_curve), or membranes= (axon_joint_null)")
        return geom.membrane, geom.membrane_arc_nm, geom.membrane_length_nm
    if geom.smooth_path is None:
        raise ValueError(f"ring {geom.index}: no smoothed curve (fewer than 4 distinct vertices or the "
                         "smoothing fit failed): null_kind 'conserved_offset' is unavailable for it")
    return geom.smooth_path, geom.smooth_arc_nm, geom.smooth_length_nm


def _walk_with_offset(
    path: SmoothPath,
    s_nm: NDArray[np.float64],
    r_nm: NDArray[np.float64],
    outward_sign: float,
) -> NDArray[np.float64]:
    """Points C(s) + r n(s) of the dense table: the foot at arc ``s_nm``
    (any shape (..., K), already reduced modulo the length; the edge by
    bisection as ``_walk_polygon``) displaced by ``r_nm`` (K,) along the
    outward unit normal of the sub-chord holding the foot, n = sign x
    (-t_y, t_x) with t the unit sub-chord -- the normal against which
    ``_project_on_path`` measures r, so the displaced point projects
    back with the same r (exactly, as long as its foot stays on that
    sub-chord; within r sin(angle between adjacent sub-chords) of arc
    otherwise, ~0.05 nm at 2 nm sub-chords on a membrane)."""
    k = path.points_nm.shape[0]
    v = np.searchsorted(path.cum_nm, s_nm, side="right") - 1
    v = np.clip(v, 0, k - 1)
    span = np.where(path.edges_nm[v] > 0.0, path.edges_nm[v], 1.0)
    t = np.clip((s_nm - path.cum_nm[v]) / span, 0.0, 1.0)
    start = path.points_nm[v]
    chord = path.points_nm[(v + 1) % k] - start
    foot = start + t[..., None] * chord
    tangent = chord / span[..., None]
    normal = float(outward_sign) * np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    return np.asarray(foot + np.asarray(r_nm, dtype=np.float64)[..., None] * normal, dtype=np.float64)


def point_on_curve(
    contour_nm: NDArray[np.float64],
    length_nm: float,
    s_nm: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    The point of the smooth closed curve through the vertices of
    ``contour_nm`` (``smooth_closed_path``, D-24a) at curve arc ``s_nm``
    (taken modulo ``length_nm``), walking the curve from its first
    vertex: (2,) for a scalar ``s_nm``, (..., 2) for an array. At the
    curve arc of a vertex it returns that vertex exactly.

    ``length_nm`` must be the curve's own closed length
    (``RingGeometry.length_nm``); it is checked against the curve built
    here to a relative 1e-9, because a wrong length (the polygon's, for
    instance) would silently wrap the arc at the wrong place. The curve
    is rebuilt on every call: for repeated use hold a ``RingGeometry``
    and call ``point_on_path``. ``point_on_polygon`` is the name the H3
    specification gave this function (when the null walked the
    polygon); it is kept as an alias.
    """
    path = smooth_closed_path(contour_nm)
    length = float(length_nm)
    if not (length > 0.0 and abs(length - path.length_nm) <= 1e-9 * max(1.0, length)):
        raise ValueError(f"point_on_curve: length_nm {length} is not this curve's closed length "
                         f"{path.length_nm} (the polygon's length is {float(_polygon_arcs(_as_points(contour_nm, 'contour'))[1][-1])})")
    s = np.mod(np.asarray(s_nm, dtype=np.float64), length)
    return _walk_polygon(path.points_nm, path.edges_nm, path.cum_nm, s)


point_on_polygon = point_on_curve


def point_on_path(
    geom: RingGeometry,
    s_nm: Union[float, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    The point of the ring's smooth closed curve at curve arc ``s_nm``
    (taken modulo ``length_nm``): (2,) for a scalar, (..., 2) for an
    array. At ``arc_nm[i]`` it is the vertex of cluster i, exactly;
    elsewhere the dense table is interpolated linearly.
    """
    path = geom.path
    s = np.mod(np.asarray(s_nm, dtype=np.float64), path.length_nm)
    return _walk_polygon(path.points_nm, path.edges_nm, path.cum_nm, s)


def _shifted_arcs(
    geom: RingGeometry,
    shift_nm: NDArray[np.float64],
    reflect: NDArray[np.bool_],
    kind: str = "interpolating",
) -> NDArray[np.float64]:
    """Curve arcs of every cluster after an optional reflection and a
    shift along the null's curve, reflection FIRST (D-24e): s' = (L - s)
    mod L when reflected, then (s' + U) mod L, with s and L the arcs
    and the closed length of the curve of ``kind`` (``_null_curve``:
    ``arc_nm`` / ``length_nm`` for "interpolating", ``smooth_arc_nm`` /
    ``smooth_length_nm`` for "conserved_offset"). ``shift_nm`` and
    ``reflect`` broadcast against the (K,) arcs on a new leading axis
    (scalars give (K,), (B,) give (B, K)), with the same elementwise
    arithmetic on both paths."""
    _, s, length = _null_curve(geom, kind)
    reflected = np.mod(length - s, length)
    sr = np.where(reflect[..., None], reflected, s)
    return np.mod(sr + shift_nm[..., None], length)


def _shifted_positions(
    geom: RingGeometry,
    shift_nm: NDArray[np.float64],
    reflect: NDArray[np.bool_],
    kind: str = "interpolating",
) -> NDArray[np.float64]:
    """The positions at ``_shifted_arcs``: (K, 2) for scalar draws, (B,
    K, 2) for a vector of B draws. "interpolating": the points of the
    smooth path at the arcs; "conserved_offset": the points of the
    smoothed curve at the arcs displaced by each cluster's own radial
    offset along the outward normal there (``_walk_with_offset``);
    "pooled_offset": the same on the attached pooled membrane with the
    offsets measured from it (``membrane_offset_nm``, the membrane's
    own orientation)."""
    if kind == "interpolating":
        path = geom.path
        return _walk_polygon(path.points_nm, path.edges_nm, path.cum_nm,
                             _shifted_arcs(geom, shift_nm, reflect))
    path, _, _ = _null_curve(geom, kind)
    if kind == "pooled_offset":
        offsets, sign = geom.membrane_offset_nm, _membrane_sign(path)
    else:
        offsets, sign = geom.radial_offset_nm, _outward_sign(geom.contour_nm)
    return _walk_with_offset(path, _shifted_arcs(geom, shift_nm, reflect, kind), offsets, sign)


def arc_shift(
    geom: RingGeometry,
    shift_nm: float,
    reflect: bool,
    *,
    kind: str = "interpolating",
) -> NDArray[np.float64]:
    """
    The new position of EVERY cluster of the ring (usable or not, in
    cluster order) after moving it along the ring's own membrane, the
    interpolating periodic cubic spline through its clusters
    (``RingGeometry.path``, D-24a): the curve arc s of each cluster
    becomes s' = (L - s) mod L if ``reflect``, THEN (s' + shift) mod L
    (reflection before the shift, D-24e: 03_plan S3.2 wrote the two
    steps in the other order, which gives the same null distribution
    because U ~ Uniform(m, L - m) is symmetric about L / 2), and the
    cluster is placed at that arc of the curve. The clusters stay on
    the membrane by construction (D-04); reflection reverses their
    cyclic order and is an equally valid null (02_investigacion B3):
    both preserve the sequence of consecutive spacings, so the null
    keeps the ring's internal structure and its density and only loses
    its phase.

    ``kind`` (D-25, H5; one of ``NULL_KINDS``): "interpolating" is the
    above, today's null. "conserved_offset" moves the clusters along
    the SMOOTHING B-spline of the contour instead (``RingGeometry.
    smooth_path``, s = K): the arc s_i of cluster i's projection on
    that curve (``smooth_arc_nm``) is reflected and shifted the same
    way on its closed length L_s (``smooth_length_nm``), and the
    cluster is placed at C(s') + r_i n(s'), the curve point displaced
    by the cluster's OWN signed radial offset r_i (``radial_offset_nm``)
    along the outward normal there -- so every cluster keeps its
    distance from the estimated membrane and only its phase along it
    is lost, and the sequence of spacings ON THE SMOOTHED CURVE is
    conserved (a cyclic permutation, reversed under reflection). Why:
    the move is exchangeable under iid isotropic scatter of the
    clusters about a smooth membrane, up to the smoothing bias of the
    estimated curve, which the interpolating spline lacks (it takes
    the scatter as membrane and overshoots between the vertices; the
    table below). A shift of 0 without reflection returns the exact
    centroids for either kind; ``kind="conserved_offset"`` raises
    ValueError on a ring without a smoothed curve (fewer than four
    distinct vertices).

    Why the smooth curve and not the polygon (review of 2026-09-23,
    D-24a): a cluster shifted onto a chord sits inside a convex
    membrane by the chord's sagitta g^2 / (8 R), so it is farther from
    the other ring's clusters than a vertex is, the polygon-path null
    matches fewer pairs and zeta is biased in favour of columns.
    Measured under M1 (independent rings on the 1500 x 1000 nm ellipse,
    B = 199; the reference shifts the same draws along the exact
    ellipse; ``validate_columns_h3`` section 4, se 0.05-0.06):

    ==========  =========  =========  ==========
    K, tau      polygon    spline     reference
    ==========  =========  =========  ==========
    40, 60      +0.19 sd   +0.04 sd   +0.03 sd
    20, 60      +0.85 sd   +0.03 sd   -0.02 sd
    ==========  =========  =========  ==========

    (the polygon's numbers are those of the first H3 version and of the
    reproduction probe, +0.24 at B = 99; at K = 60 on a 1200 nm circle
    with tau = 80 the chord sagitta is 1.6 nm and the harness prints the spline's
    figure as a diagnostic).

    Design point of D-24a (re-review of 2026-09-23), answered by D-25's
    ``kind="conserved_offset"`` (H5): the table above holds for
    clusters that sit EXACTLY on the membrane. The interpolating spline
    through vertices that scatter about it (a cluster's lateral
    precision, ~15 nm) is not the membrane: it overshoots between the
    vertices (median 50-60 nm, up to 160-520 nm off the true ellipse on
    jittered rings), so a shifted cluster is farther from the other
    ring's clusters than the vertices are and the null again matches
    fewer pairs on sparse rings. Measured against a reference that
    shifts each cluster along the exact membrane carrying its own
    jitter vector (exchangeable under iid isotropic jitter), same rings
    and the SAME draws, paired difference of the mean zeta (module
    minus reference): K = 40, 15 nm jitter: +0.125 sd (se 0.012); K =
    20, 15 nm: +0.219 (the module's mean zeta +0.229, outside the
    +/-0.15 band the harness asks on membrane-exact rings); K = 20, 8
    nm: +0.097; on a denser ring (circle R = 1200 nm,
    K = 60, tau 80, 15 nm): -0.088 (conservative there). The
    conserved-offset kind is the smoothing-curve remedy named there,
    with the pre-registered scale s = K; ``validate_simulate_axon.py``
    section 5 measures both kinds on the same scattered M1 rings and
    the same draws (15 nm isotropic scatter, B = 199, tau 60, R = 200
    pairs per K, se of a mean zeta 0.07-0.08; implementation run of
    2026-09-25, with the scatter-scaled s = K sigma^2 probed alongside):

    ==================  =============  ===============  ====================
    rings               interpolating  conserved s = K  conserved s = K s^2
    ==================  =============  ===============  ====================
    K 40, 15 nm         +0.127         +0.109           +0.079
    K 20, 15 nm         +0.209         +0.190           +0.193
    K 40, exact (150)   +0.097         +0.098           +0.327
    K 20, exact (150)   +0.031         +0.035           +0.243
    joint K 20, 15 nm   +0.487 (z_A)   +0.462 (z_A)     +0.393 (z_A)
    ==================  =============  ===============  ====================

    What it says: at s = K the smoothed curve nearly interpolates (rms
    residual 0.9 nm about 15 nm-scattered vertices; FITPACK's s is a
    summed squared residual in nm^2 with unit weights), so the
    conserved-offset move is the interpolating one to within 0.02 sd
    and does NOT remove the sparse-ring bias; a scatter-scaled s does
    not remove it either at K = 20 (the curve is still a median 10 nm
    off the true membrane: K centroids 400 nm apart cannot fix the
    membrane better than that) and it breaks the membrane-exact case
    (the curve is smoothed 7-9 nm off the true membrane and the carried
    offsets land the clusters off it). The rejection rates of every
    variant stay inside [0.01, 0.09] at the 5 % level; the bias is in
    the mean zeta, i.e. in the effect size T_A the between-axon
    inference works on. The interpolating kind stays the default
    because it is the pre-registered H3 null; D-28 decides, and a
    membrane estimate from more than the K centroids (the ring's
    localization cloud, or the contours of the axon's rings pooled) is
    the direction that the reference of the re-review -- the exact
    membrane carrying each cluster's own scatter, unbiased -- points
    to.

    A shift of 0 without reflection is the identity and returns the
    exact centroids, not their rounded vertices (the two differ by up
    to 0.005 nm on a measured ring, by nothing on a synthetic one; at
    any other shift the clusters are placed on the curve through the
    rounded vertices, i.e. ``arc_shift(L, False)`` is the centroids to
    0.005 nm on a measured ring).
    """
    kind_ = _check_kind(kind)
    shift = float(shift_nm)
    if not math.isfinite(shift):
        raise ValueError(f"arc_shift: shift_nm must be finite, got {shift}")
    if shift == 0.0 and not reflect:
        return np.array(geom.centroids_nm, dtype=np.float64, copy=True)
    return _shifted_positions(geom, np.asarray(shift, dtype=np.float64),
                              np.asarray(bool(reflect)), kind_)


def _geometry_of(
    obj: Union[Ring, RingGeometry], include_suspect: bool, warnings_: List[str],
) -> RingGeometry:
    """
    A ``RingGeometry`` for the matching, honouring ``include_suspect``:
    built from a ``Ring`` with that flag, or the given geometry -- with
    its suspect clusters (``RingGeometry.suspect``) also masked out when
    ``include_suspect`` is False and the geometry still had them usable
    (a copy; the input is not modified). The other direction cannot be
    honoured: a geometry whose mask already excludes suspect clusters
    keeps excluding them under ``include_suspect=True`` (the mask may
    exclude for other reasons) and a warning says so, so that the
    ``include_suspect`` stamp of a result never claims more than the
    masks recorded next to it.
    """
    if isinstance(obj, Ring):
        return ring_geometry(obj, include_suspect=include_suspect)
    if not isinstance(obj, RingGeometry):
        raise TypeError(f"expected Ring or RingGeometry, got {type(obj).__name__}")
    suspect = np.asarray(obj.suspect, dtype=bool)
    usable = np.asarray(obj.usable, dtype=bool)
    if not include_suspect:
        n_in = int(np.count_nonzero(usable & suspect))
        if n_in:
            return dataclasses.replace(obj, usable=usable & ~suspect)
    else:
        n_out = int(np.count_nonzero(~usable & suspect))
        if n_out:
            warnings_.append(f"ring {obj.index}: include_suspect=True requested but the geometry's mask "
                             f"already leaves {n_out} suspect cluster(s) out; the mask decides "
                             "(usable_a/usable_b record what entered)")
    return obj


# ============================================================================
# One-to-one matching within tau
# ============================================================================

def _distance_matrix(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Lateral Euclidean distances (Ka, Kb) between two point sets; with a
    leading replicate axis on either side, (..., Ka, Kb). The same
    arithmetic (hypot of the two differences) wherever a distance is
    formed, so that a count reproduced by hand equals the vectorised one
    bit for bit."""
    return np.hypot(a[..., :, None, 0] - b[..., None, :, 0],
                    a[..., :, None, 1] - b[..., None, :, 1])


def _dummy_cost(tau_nm: float, k_a: int, k_b: int) -> float:
    """M = tau (K_a + K_b + 1), the cost of leaving one cluster on a
    dummy (D-24b; see ``match_rings`` for why it forces the maximum
    cardinality)."""
    return float(tau_nm) * (int(k_a) + int(k_b) + 1)


def _assign_within_tau(D: NDArray[np.float64], tau_nm: float) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    The pairs (rows, cols) of the truncated assignment on a (K_a, K_b)
    distance matrix: cost d for a pair within tau, ``BIG_COST`` beyond
    it, M = tau (K_a + K_b + 1) for leaving a row or a column on a dummy
    (``_dummy_cost``, D-24b), 0 for a dummy on a dummy. Rows and columns
    with no admissible partner are left out before the solver (they end
    on a dummy whatever happens; M is still computed from the full K_a
    and K_b), and when every remaining row and column has exactly one
    admissible partner the answer is those pairs (each is worth 2 M - d
    > 0 and conflicts with nothing), so the solver only runs where
    there is a choice to make. Both shortcuts give the solver's own
    answer.
    """
    admissible = D <= tau_nm
    rows = np.flatnonzero(admissible.any(axis=1))
    cols = np.flatnonzero(admissible.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    sub = admissible[np.ix_(rows, cols)]
    if int(sub.sum(axis=1).max()) <= 1 and int(sub.sum(axis=0).max()) <= 1:
        r, c = np.nonzero(sub)
        return rows[r], cols[c]
    ka, kb = rows.size, cols.size
    dummy = _dummy_cost(tau_nm, D.shape[0], D.shape[1])
    cost = np.zeros((ka + kb, ka + kb), dtype=np.float64)
    cost[:ka, :kb] = np.where(sub, D[np.ix_(rows, cols)], BIG_COST)
    cost[:ka, kb:] = dummy
    cost[ka:, :kb] = dummy
    r, c = linear_sum_assignment(cost)
    r = np.asarray(r, dtype=np.intp)
    c = np.asarray(c, dtype=np.intp)
    real = (r < ka) & (c < kb)
    r, c = r[real], c[real]
    keep = sub[r, c]
    return rows[r[keep]], cols[c[keep]]


def match_rings(
    cent_a_nm: NDArray[np.float64],
    cent_b_nm: NDArray[np.float64],
    tau_nm: float,
    *,
    usable_a: Optional[NDArray[np.bool_]] = None,
    usable_b: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """
    One-to-one matching of the clusters of ring a to those of ring b
    within the lateral tolerance tau (D-02: Euclidean distance between
    centroids in the axon frame, no centre and no angle).

    The matching is the maximum-cardinality matching among the
    admissible pairs (d <= tau) and, among those, the one of minimum
    total distance (D-24b, 01_formalizacion S1.2). It is computed as the
    minimum-cost assignment of a (Ka + Kb) square matrix (D-11, 03_plan
    S3.2): ``C[:Ka, :Kb] = where(D <= tau, D, BIG_COST)``, ``C[:Ka, Kb:]
    = M`` (an a-cluster left unmatched costs M), ``C[Ka:, :Kb] = M``
    (idem for b), ``C[Ka:, Kb:] = 0``, with ``M = tau (Ka + Kb + 1)``;
    the pairs are the assignments of a real row to a real column with
    ``D <= tau``. Why M guarantees the maximum cardinality: with n
    pairs the cost is (sum of the n distances) + M (Ka + Kb - 2 n), so
    one more admissible pair saves 2 M = 2 tau (Ka + Kb + 1), while the
    distance sum of any matching is at most min(Ka, Kb) tau < M; an
    extra pair therefore always pays, whatever it does to the
    distances, and among the matchings of maximum cardinality the
    cheapest is the one of minimum distance sum. Consequences: a pair
    that conflicts with nothing is always taken; on the trap of B3 (a1
    near b1 and b2, a2 near b2 only) the assignment takes the two pairs
    the greedy nearest neighbour misses; on two clusters 1.5 tau apart
    it takes none, where an untruncated matrix would pair them (1.5 tau
    < 2 M); and the count is monotone non-decreasing in tau (the
    admissible set only grows with tau, so the maximum matching cannot
    shrink), replicate by replicate of a null. The first H3 version
    put the dummies at tau, which the specification described as
    maximum cardinality but is not (an extra pair was then worth
    between tau and 2 tau, and fewer, closer pairs could win: a =
    (0, 0), (55, 0), (110, 0) against b = (-55, 0), (0, 0), (55, 0) at
    tau = 60 gave 2 pairs at 0 nm instead of 3 at 55 nm); D-24b makes
    the pre-registered semantics the rule.

    Parameters
    ----------
    cent_a_nm, cent_b_nm : (Ka, 2), (Kb, 2)
        Centroids in the axon frame, nm.
    tau_nm : float
        Tolerance, nm (> 0).
    usable_a, usable_b : (Ka,), (Kb,) bool, optional
        Clusters allowed to enter; all when None.

    Returns
    -------
    (i_a, j_b, d_nm)
        Indices into the two inputs and their distances, one entry per
        pair, sorted by ``i_a``; three empty arrays without a pair.
    """
    a = _as_points(cent_a_nm, "cent_a_nm")
    b = _as_points(cent_b_nm, "cent_b_nm")
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"match_rings: tau_nm must be a positive finite number, got {tau}")
    ia = np.flatnonzero(_mask(usable_a, a.shape[0], "usable_a"))
    jb = np.flatnonzero(_mask(usable_b, b.shape[0], "usable_b"))
    empty = (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.float64))
    if ia.size == 0 or jb.size == 0:
        return empty
    D = _distance_matrix(a[ia], b[jb])
    r, c = _assign_within_tau(D, tau)
    if r.size == 0:
        return empty
    i_a = np.asarray(ia[r], dtype=np.intp)
    j_b = np.asarray(jb[c], dtype=np.intp)
    d = np.asarray(D[r, c], dtype=np.float64)
    order = np.argsort(i_a, kind="stable")
    return i_a[order], j_b[order], d[order]


def assign_within_tau(D_nm: NDArray[np.float64], tau_nm: float) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """
    The pairs (rows, cols) of the truncated assignment on ANY finite
    (K_a, K_b) distance matrix: the maximum-cardinality matching among
    the entries <= tau and, among those, the one of minimum total
    distance (D-24b). This is the public entry of ``_assign_within_tau``
    (H5-B), so that a matching in another metric -- the cyclic arc
    distance of ``match_arcs`` on the reference circle, the 1D test of
    ``tools.mps_unroll`` -- runs through the same solver, the same dummy
    cost and the same shortcuts as ``match_rings`` and never
    re-implements the trap logic. Entries beyond tau may hold any finite
    value: they are never chosen. Two empty arrays when either side is
    empty or no entry is within tau; ValueError on a non-2-D or
    non-finite matrix or a tau that is not positive and finite.
    """
    D = np.asarray(D_nm, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"assign_within_tau: expected a 2-D distance matrix, got shape {D.shape}")
    if D.size and not np.isfinite(D).all():
        raise ValueError("assign_within_tau: non-finite distance")
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"assign_within_tau: tau_nm must be a positive finite number, got {tau}")
    if D.shape[0] == 0 or D.shape[1] == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    return _assign_within_tau(D, tau)


def cyclic_arc_distance_nm(d_nm: NDArray[np.float64], length_nm: float) -> NDArray[np.float64]:
    """
    The distance on a circle of length L between two arc positions
    whose difference is ``d_nm``: d reduced modulo L, then the shorter
    way round, min(d, L - d), in [0, L / 2]. Any shape; a full turn is
    no distance (s and s + L are the same point). The same arithmetic
    wherever a cyclic distance is formed (``match_arcs`` and the null
    counts of ``tools.mps_unroll``), so that a count reproduced by hand
    from the recorded draws equals the vectorised one bit for bit.
    """
    length = float(length_nm)
    d = np.mod(np.asarray(d_nm, dtype=np.float64), length)
    return np.asarray(np.minimum(d, length - d), dtype=np.float64)


def match_arcs(
    s_a_nm: NDArray[np.float64],
    s_b_nm: NDArray[np.float64],
    length_nm: float,
    tau_nm: float,
    *,
    usable_a: Optional[NDArray[np.bool_]] = None,
    usable_b: Optional[NDArray[np.bool_]] = None,
) -> Tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """
    One-to-one matching of two point sets on a circle of length L
    within the ARC tolerance tau (H5-B, candidate A): the 1D analogue of
    ``match_rings`` for two rings projected on the same reference curve
    (``tools.mps_unroll.analyze_arc_columns``), with the cyclic arc
    distance |s_j - s_i|_L in place of the lateral Euclidean one.

    The semantics are D-24b's exactly: the maximum-cardinality matching
    among the admissible pairs (cyclic distance <= tau) and, among
    those, the one of minimum total distance, solved by
    ``assign_within_tau`` on the (K_a, K_b) matrix of cyclic distances
    (``cyclic_arc_distance_nm``); the 3-vs-2 trap of ``match_rings``
    (a = 0, 55, 110 against b = -55, 0, 55 at tau 60: three pairs at 55
    nm, not two at 0) holds on the circle as on the plane, and so does
    the monotonicity of the count in tau. A pair may straddle s = 0 / L:
    s = 2 and s = L - 3 are 5 nm apart.

    Parameters
    ----------
    s_a_nm, s_b_nm : (K_a,), (K_b,)
        Arc positions on the same circle, nm (any real value; reduced
        modulo L by the distance).
    length_nm : float
        L, the circle's length (> 0).
    tau_nm : float
        The arc tolerance, nm (> 0).
    usable_a, usable_b : (K_a,), (K_b,) bool, optional
        Clusters allowed to enter; all when None.

    Returns
    -------
    (i_a, j_b, d_nm)
        Indices into the two inputs and their cyclic distances, one
        entry per pair, sorted by ``i_a``; three empty arrays without a
        pair.
    """
    s_a = np.asarray(s_a_nm, dtype=np.float64).reshape(-1)
    s_b = np.asarray(s_b_nm, dtype=np.float64).reshape(-1)
    if not (np.isfinite(s_a).all() and np.isfinite(s_b).all()):
        raise ValueError("match_arcs: non-finite arc position")
    length = float(length_nm)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"match_arcs: length_nm must be a positive finite number, got {length_nm}")
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"match_arcs: tau_nm must be a positive finite number, got {tau}")
    ia = np.flatnonzero(_mask(usable_a, s_a.size, "usable_a"))
    jb = np.flatnonzero(_mask(usable_b, s_b.size, "usable_b"))
    empty = (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.float64))
    if ia.size == 0 or jb.size == 0:
        return empty
    D = cyclic_arc_distance_nm(s_a[ia][:, None] - s_b[jb][None, :], length)
    r, c = _assign_within_tau(D, tau)
    if r.size == 0:
        return empty
    i_a = np.asarray(ia[r], dtype=np.intp)
    j_b = np.asarray(jb[c], dtype=np.intp)
    d = np.asarray(D[r, c], dtype=np.float64)
    order = np.argsort(i_a, kind="stable")
    return i_a[order], j_b[order], d[order]


def _null_counts(
    pos_a: NDArray[np.float64],
    usable_a: NDArray[np.bool_],
    pos_b: NDArray[np.float64],
    usable_b: NDArray[np.bool_],
    taus_nm: NDArray[np.float64],
) -> NDArray[np.int64]:
    """
    Matched pairs per tau and per replicate, (T, B), for B configurations
    of ring b (``pos_b`` (B, Kb, 2)) against ring a, fixed (``pos_a``
    (Ka, 2)) or itself replicated ((B, Ka, 2), the joint null). Only the
    usable clusters enter. The distance matrix of a replicate is formed
    once and reused for every tau: the same shifted configurations serve
    the whole grid, so the curve is internally consistent and costs one
    shift per replicate (D-03). Replicates in which no cluster has two
    admissible partners are counted without the solver (see
    ``_assign_within_tau``).
    """
    taus = np.asarray(taus_nm, dtype=np.float64).ravel()
    n_rep = int(pos_b.shape[0])
    counts = np.zeros((taus.size, n_rep), dtype=np.int64)
    a = pos_a[..., usable_a, :]
    b = pos_b[:, usable_b, :]
    if a.shape[-2] == 0 or b.shape[1] == 0 or taus.size == 0:
        return counts
    for start in range(0, n_rep, _BLOCK):
        stop = min(start + _BLOCK, n_rep)
        ab = a if a.ndim == 2 else a[start:stop]
        D = _distance_matrix(ab, b[start:stop])                     # (b, Ka, Kb)
        admissible = D[None, ...] <= taus[:, None, None, None]      # (T, b, Ka, Kb)
        n_edges = admissible.sum(axis=(2, 3))
        simple = ((admissible.sum(axis=3).max(axis=2) <= 1)
                  & (admissible.sum(axis=2).max(axis=2) <= 1))
        counts[:, start:stop] = np.where(simple, n_edges, 0)
        for t, r in zip(*np.nonzero(~simple)):
            counts[t, start + r] = _assign_within_tau(D[r], float(taus[t]))[0].size
    return counts


# ============================================================================
# Null draws and their summaries
# ============================================================================

def _phipson_smyth(null: NDArray[np.int64], observed: int) -> Tuple[float, float, float]:
    """Monte Carlo p-values with the +1 of Phipson and Smyth (2010):
    excess (null >= observed), deficit (null <= observed) and the
    two-sided min(1, 2 min(excess, deficit)) of D-11. Excess favours
    columns (M5/M7), deficit alternation (M6)."""
    n_null = int(null.size)
    p_excess = (int(np.count_nonzero(null >= observed)) + 1) / (n_null + 1)
    p_deficit = (int(np.count_nonzero(null <= observed)) + 1) / (n_null + 1)
    return p_excess, p_deficit, min(1.0, 2.0 * min(p_excess, p_deficit))


def _null_summary(
    observed: int,
    null: NDArray[np.int64],
    min_k: int,
    warnings_: List[str],
    label: str,
) -> Tuple[float, float, float]:
    """(E_star, sd_star, zeta) of one null: the mean and the sample sd
    (ddof = 1) of the null counts over min(K_a, K_b), and the
    standardised observed count. zeta is NaN, with a warning, when the
    null is constant (sd 0): a null that never varies cannot standardise
    anything, and the p-values carry the verdict there."""
    mean = float(null.mean()) if null.size else float("nan")
    sd = float(null.std(ddof=1)) if null.size > 1 else 0.0
    if min_k > 0:
        e_star, sd_star = mean / min_k, sd / min_k
    else:
        e_star = sd_star = float("nan")
        warnings_.append(f"{label}: no usable cluster on one side (min K = 0): E_dir, E* undefined")
    if sd > 0.0:
        zeta = (observed - mean) / sd
    else:
        zeta = float("nan")
        warnings_.append(f"{label}: the null n_matched is constant ({mean:g} in every replicate), "
                         "its sd is 0 and zeta is undefined (NaN); read the p-values")
    return e_star, sd_star, zeta


def _draw_shifts(
    rng: np.random.Generator,
    n_null: int,
    geom: RingGeometry,
    min_shift_fraction: float,
    warnings_: List[str],
    kind: str = "interpolating",
) -> Tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    """
    The null's draws for one ring: U ~ Uniform(m, L - m) with m =
    ``min_shift_fraction`` x p_bar (p_bar = L / K, L the closed length
    of the ring's curve and K ALL its clusters, the polygon's vertices),
    then reflect ~ Bernoulli(1/2), in that order from ``rng`` -- the
    order every caller uses, so that ``eclipse_test`` and
    ``eclipse_curve`` draw the same configurations from the same seed.
    Excluding |U| < p_bar / 2 (D-11) keeps the null away from the
    observed phase, where a shift smaller than half the mean spacing
    would leave most clusters within tau of themselves. When 2 m >= L
    there is no room left: U ~ Uniform(0, L) with a warning, and the
    returned minimum shift is 0. L and p_bar are those of the curve of
    ``kind`` (``_null_curve``): the interpolating spline's, or the
    smoothed curve's for "conserved_offset" (a few nm shorter, since
    the smoothed curve cuts the scatter's zigzag), or the attached
    pooled membrane's for "pooled_offset" (H5-B).
    """
    if kind == "interpolating":
        length = float(geom.length_nm)
        m = float(min_shift_fraction) * geom.mean_spacing_nm
    else:
        _, _, length = _null_curve(geom, kind)
        length = float(length)
        m = float(min_shift_fraction) * (length / geom.n_clusters if geom.n_clusters else float("nan"))
    if not (length > 0.0 and math.isfinite(m) and m >= 0.0):
        raise ValueError(f"ring {geom.index}: perimeter {length} nm, minimum shift {m} nm")
    lo, hi = m, length - m
    if 2.0 * m >= length:
        warnings_.append(f"ring {geom.index}: the minimum shift {m:.1f} nm is at least half the "
                         f"perimeter {length:.1f} nm (2 m >= L): U drawn from Uniform(0, L) "
                         "without the exclusion")
        lo, hi, m = 0.0, length, 0.0
    shifts = np.asarray(rng.uniform(lo, hi, size=int(n_null)), dtype=np.float64)
    reflect = np.asarray(rng.random(int(n_null)) < 0.5, dtype=bool)
    return shifts, reflect, m


def _pair_seed(random_seed: int, ga: RingGeometry, gb: RingGeometry) -> np.random.Generator:
    """The generator of one ring pair: ``SeedSequence(random_seed,
    spawn_key=(a.index, b.index))``, so a pair's null depends on nothing
    but the seed and the two ring indices."""
    if ga.index < 0 or gb.index < 0:
        raise ValueError(f"ring indices must be non-negative to seed the null, got "
                         f"({ga.index}, {gb.index})")
    return np.random.default_rng(np.random.SeedSequence(int(random_seed),
                                                        spawn_key=(int(ga.index), int(gb.index))))


def _include_suspect_stamp(
    inputs: Sequence[Union[Ring, RingGeometry]],
    geoms: Sequence[RingGeometry],
    include_suspect: bool,
    warnings_: List[str],
) -> bool:
    """
    The ``include_suspect`` flag a result records (review of 2026-09-23,
    item 5 of the fix decisions). With ``Ring`` inputs only it is the
    argument: their geometries were built with it and the masks follow
    it. When a ``RingGeometry`` is among the inputs the argument is not
    what decides (the geometry's mask does, see ``_geometry_of``), so
    the stamp is the flag the masks imply: True when every cluster of
    every geometry that entered is usable, False when any is left out;
    a warning names the disagreement when the argument said otherwise.
    The stamp then says what was applied, never what was asked.
    """
    if not any(isinstance(obj, RingGeometry) for obj in inputs):
        return bool(include_suspect)
    left_out = [(g.index, int(np.count_nonzero(~g.usable))) for g in geoms if not np.all(g.usable)]
    implied = not left_out
    if implied != bool(include_suspect):
        if implied:
            warnings_.append(f"include_suspect=False requested but no cluster of ring(s) "
                             f"{[g.index for g in geoms]} is left out (no suspect mark, mask all True): "
                             "every cluster entered and the result records include_suspect=True")
        else:
            detail = ", ".join(f"ring {k}: {n}" for k, n in left_out)
            warnings_.append(f"include_suspect=True requested but the geometry masks leave cluster(s) out "
                             f"({detail}): the result records include_suspect=False (what was applied)")
    return implied


def _check_null_args(tau_nm: Any, n_null: Any, min_shift_fraction: Any) -> Tuple[float, int, float]:
    """The null's scalar arguments, validated."""
    tau = float(tau_nm)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau_nm must be a positive finite number, got {tau}")
    n_rep = int(n_null)
    if n_rep < 1:
        raise ValueError(f"n_null must be >= 1, got {n_null}")
    frac = float(min_shift_fraction)
    if not (math.isfinite(frac) and frac >= 0.0):
        raise ValueError(f"min_shift_fraction must be >= 0, got {min_shift_fraction}")
    return tau, n_rep, frac


# ============================================================================
# The pooled leave-ring-out membrane (H5-B, null_kind "pooled_offset")
# ============================================================================

def _membrane_sign(path: SmoothPath) -> float:
    """The outward sign of a membrane table: ``_outward_sign`` of the
    dense curve itself (its orientation is the tour ``fit_membrane``
    built, not the moved ring's), used both to project the clusters on
    it and to walk them back, so the offsets are conserved."""
    return _outward_sign(np.asarray(path.points_nm, dtype=np.float64))


def _attach_membrane(geom: RingGeometry, path: SmoothPath) -> RingGeometry:
    """A shallow copy of ``geom`` carrying ``path`` as its pooled
    membrane with the derived arcs and offsets (``_derive_membrane``);
    the geometry itself is not modified and its interpolating and
    smoothed curves are not rebuilt (``dataclasses.replace`` would
    re-run ``__post_init__``, two spline fits and their tables, for
    nothing). The same object when it already carries that path."""
    if geom.membrane is path:
        return geom
    out = copy.copy(geom)
    out.membrane = path
    out._derive_membrane()
    return out


def pooled_membrane_path(
    rings: Sequence[Union[Ring, RingGeometry]],
    *,
    knot_spacing_nm: Optional[float] = None,
) -> SmoothPath:
    """
    The pooled membrane of ``rings``: ``tools.mps_membrane.fit_membrane``
    on ALL their cluster centroids concatenated (usable or not: the
    membrane is geometry and does not change when a cluster is
    doubted), as a ``SmoothPath``. Called with every ring of the axon
    EXCEPT the one whose null is drawn, it is the leave-ring-out
    membrane of ``null_kind="pooled_offset"``; with one ring it is that
    ring's own P-spline membrane (``radial_scatter_nm``'s curve, the
    two-ring case of the spec). ``knot_spacing_nm`` None keeps
    ``fit_membrane``'s default (``MEMBRANE_NULL_KNOT_SPACING_NM``, 400
    nm: the 2D null's spacing, unchanged); the arc test on the pooled
    membrane passes its own, coarser spacing (``tools.mps_unroll.
    ARC_MEMBRANE_KNOT_SPACING_NM``, the argument there). Deterministic.
    ``tools.mps_membrane`` imports this module for ``SmoothPath``, so
    the import here is deferred to the call (the reverse top-level
    import would be circular).

    Raises
    ------
    ValueError
        No ring, fewer than three centroids in total, or a fit that
        fails (``fit_membrane``).
    """
    from tools.mps_membrane import fit_membrane  # deferred: mps_membrane imports SmoothPath from here
    if not rings:
        raise ValueError("pooled_membrane_path: no ring to fit the membrane to")
    geoms = [_geometry_of(r, True, []) for r in rings]
    points = np.concatenate([np.asarray(g.centroids_nm, dtype=np.float64) for g in geoms], axis=0)
    fit = fit_membrane(points) if knot_spacing_nm is None else fit_membrane(points, knot_spacing_nm=float(knot_spacing_nm))
    if fit.path is None:
        raise ValueError("pooled_membrane_path: fit_membrane returned no path")
    return fit.path


def leave_one_out_membranes(
    rings: Sequence[Union[Ring, RingGeometry]],
    *,
    include_suspect: bool = True,
) -> Dict[int, SmoothPath]:
    """
    The leave-ring-out membrane of every ring of an axon: ring k ->
    ``pooled_membrane_path`` of every OTHER ring (H5-B: the curve ring
    k's null walks must not depend on ring k). Keyed by ring index;
    what ``axon_joint_null(membranes=...)`` and ``analyze_columns``
    use. A repeated ring index or a single ring raises ValueError.
    """
    geoms = _sorted_geometries(rings, include_suspect, [])
    if len(geoms) < 2:
        raise ValueError(f"leave_one_out_membranes: {len(geoms)} ring(s); a leave-ring-out membrane needs at least two")
    return {g.index: pooled_membrane_path([o for o in geoms if o.index != g.index]) for g in geoms}


def _resolve_membrane(
    gb: RingGeometry,
    kind: str,
    membrane: Optional[SmoothPath],
    other_rings: Optional[Sequence[Union[Ring, RingGeometry]]],
    warnings_: List[str],
) -> RingGeometry:
    """Ring b with the membrane its pooled-offset null walks attached
    (``eclipse_test`` / ``eclipse_curve``): ``membrane`` when given, else
    the pooled membrane of ``other_rings`` (which must not contain ring
    b: the leave-ring-out property is the point), else the membrane the
    geometry already carries; none of the three -> ValueError. Both
    ``membrane`` and ``other_rings`` -> ValueError (ambiguous). For the
    other kinds the two keywords are ignored with a warning."""
    if kind != "pooled_offset":
        if membrane is not None or other_rings is not None:
            warnings_.append(f"ring {gb.index}: membrane / other_rings given but null_kind is {kind!r}: ignored "
                             "(they belong to null_kind 'pooled_offset')")
        return gb
    if membrane is not None and other_rings is not None:
        raise ValueError("null_kind 'pooled_offset': give membrane=<SmoothPath> OR other_rings=[...], not both")
    if membrane is None and other_rings is not None:
        others = [_geometry_of(o, True, []) for o in other_rings]
        if any(o.index == gb.index for o in others):
            raise ValueError(f"null_kind 'pooled_offset': other_rings contains ring {gb.index}, the ring being moved; "
                             "the membrane must be fitted to the OTHER rings only (leave-ring-out)")
        membrane = pooled_membrane_path(others)
    if membrane is None:
        if gb.membrane is not None:
            return gb
        raise ValueError(f"ring {gb.index}: null_kind 'pooled_offset' needs the membrane of ring b's null: "
                         "membrane=<SmoothPath> (tools.mps_membrane.fit_membrane(...).path) or "
                         "other_rings=[every other ring of the axon]")
    if not isinstance(membrane, SmoothPath):
        raise TypeError(f"membrane must be a SmoothPath, got {type(membrane).__name__}")
    return _attach_membrane(gb, membrane)


def _with_membranes(
    geoms: Sequence[RingGeometry],
    kind: str,
    membranes: Optional[Mapping[int, SmoothPath]],
    warnings_: List[str],
) -> List[RingGeometry]:
    """The geometries of an axon with their pooled membranes attached
    (``axon_joint_null``, ``joint_shift_statistic``): for
    "pooled_offset", ``membranes[k]`` when given for ring k, else the
    membrane the geometry already carries, else the leave-ring-out
    membranes are built for every ring that lacks one
    (``leave_one_out_membranes``); for the other kinds ``membranes`` is
    ignored with a warning."""
    if kind != "pooled_offset":
        if membranes is not None:
            warnings_.append(f"membranes given but null_kind is {kind!r}: ignored (they belong to 'pooled_offset')")
        return list(geoms)
    given: Dict[int, SmoothPath] = dict(membranes) if membranes is not None else {}
    if any(g.index not in given and g.membrane is None for g in geoms):
        built = leave_one_out_membranes(geoms)
        for g in geoms:
            if g.index not in given and g.membrane is None:
                given[g.index] = built[g.index]
    return [_attach_membrane(g, given[g.index]) if g.index in given else g for g in geoms]


# ============================================================================
# Eclipse test for one pair of rings
# ============================================================================

@dataclass
class RingPairMatch:
    """
    One pair of rings matched within ``tau_nm`` and judged against the
    arc-shift null (D-04, D-11).

    ``i_a``/``j_b`` are cluster indices in each ring's ``clusters``
    order, ``d_nm`` their distances. ``K_a``/``K_b`` count the USABLE
    clusters; ``E_dir = n_matched / min(K_a, K_b)`` is the directed
    eclipse (the fraction of the sparser ring that has a partner) and
    ``E_sym = 2 n_matched / (K_a + K_b)`` the symmetric one. ``E_star``
    and ``sd_star`` are the mean and sample sd of the null's
    ``n_matched`` over min K, ``zeta`` the standardised observed count
    (NaN, with a warning, on a constant null), the p-values Monte Carlo
    with the +1 of Phipson and Smyth: excess favours columns, deficit
    alternation, two-sided = min(1, 2 min). ``delta_bar_nm`` is the mean
    displacement b - a over the pairs (a rigid lateral offset between
    the rings shows here; a torsion does not, because tangent
    displacements cancel around a closed contour) and ``d_median_nm``
    the median matched distance; both NaN without a pair. The null's
    draws are recorded (``null_shift_nm``, ``null_reflect``) so any
    replicate can be rebuilt with ``arc_shift`` and ``match_rings``.
    ``min_shift_nm`` is the exclusion actually applied (0 when it had to
    be dropped). ``usable_a``/``usable_b`` and ``labels_a``/``labels_b``
    carry each ring's mask and DBSCAN labels for ``build_columns``.
    ``null_kind`` is the curve the null walked (``NULL_KINDS``, D-25;
    "interpolating" = the H3 null).
    """

    ring_a: int
    ring_b: int
    tau_nm: float
    i_a: NDArray[np.intp]
    j_b: NDArray[np.intp]
    d_nm: NDArray[np.float64]
    n_matched: int
    K_a: int
    K_b: int
    E_dir: float
    E_sym: float
    E_star: float
    sd_star: float
    zeta: float
    p_excess: float
    p_deficit: float
    p_two_sided: float
    delta_bar_nm: NDArray[np.float64]
    d_median_nm: float
    null_n_matched: NDArray[np.int64]
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    n_null: int
    random_seed: int
    min_shift_nm: float
    include_suspect: bool
    warnings: List[str] = field(default_factory=list)
    usable_a: Optional[NDArray[np.bool_]] = None
    usable_b: Optional[NDArray[np.bool_]] = None
    labels_a: Optional[NDArray[np.int64]] = None
    labels_b: Optional[NDArray[np.int64]] = None
    null_kind: str = "interpolating"


def _observed_match(
    ga: RingGeometry, gb: RingGeometry, tau_nm: float,
) -> Tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """The observed pairs of two geometries (usable clusters only)."""
    return match_rings(ga.centroids_nm, gb.centroids_nm, tau_nm,
                       usable_a=ga.usable, usable_b=gb.usable)


def _pair_label(ga: RingGeometry, gb: RingGeometry, tau_nm: float) -> str:
    """The prefix of a pair's warnings."""
    return f"rings {ga.index}-{gb.index} at tau {tau_nm:g} nm"


def eclipse_test(
    a: Union[Ring, RingGeometry],
    b: Union[Ring, RingGeometry],
    tau_nm: float,
    *,
    n_null: int = 1999,
    random_seed: int = 0,
    min_shift_fraction: float = 0.5,
    include_suspect: bool = True,
    null_kind: str = "interpolating",
    membrane: Optional[SmoothPath] = None,
    other_rings: Optional[Sequence[Union[Ring, RingGeometry]]] = None,
) -> RingPairMatch:
    """
    The eclipse test of one pair of rings at one tolerance: the observed
    matching (``match_rings``) against ``n_null`` replicates in which
    ring b is moved along its own polygon by U ~ Uniform(m, L_b - m)
    with a reflection drawn at random (``arc_shift``) and re-matched to
    the unmoved ring a with the same tau and the same usable masks
    (D-04, D-11; ``_draw_shifts`` for m). ``null_kind`` names the curve
    the move follows (``NULL_KINDS``, D-25: "interpolating" is the H3
    null and the default; "conserved_offset" moves ring b along its
    smoothed curve with every cluster keeping its radial offset, the
    draws then made on that curve's length and mean spacing); it is
    recorded in the result.

    ``null_kind="pooled_offset"`` (H5-B, candidate B; D-25's remedy done
    right) moves ring b along a membrane estimated WITHOUT ring b: the
    periodic P-spline of ``tools.mps_membrane.fit_membrane`` (knots
    every ``tools.mps_membrane.MEMBRANE_NULL_KNOT_SPACING_NM`` = 400 nm
    of the tour, not the 600 nm of the scatter estimate: the argument
    and the measured trade-off are next to that constant; H5B_SPEC S2
    wrote 600, the deviation is recorded for D-28) through the
    centroids of every OTHER ring of the axon
    -- ``other_rings`` (which must not contain b), or the curve itself
    as ``membrane`` (a ``SmoothPath``; ``analyze_columns`` attaches the
    leave-ring-out membranes it builds for the whole axon), one of the
    two required (neither -> ValueError; with two rings the curve is
    ring a's own membrane). Each cluster of b keeps its signed offset
    from that curve (``RingGeometry.membrane_offset_nm``) while its arc
    on it is reflected about the curve's row 0 and shifted, U ~
    Uniform(m, L_M - m) with L_M the membrane's length and m = L_M /
    (2 K_b) (D-11 on that curve's mean spacing). Why not b's own curve:
    a curve fitted to b's centroids -- the interpolating spline, or the
    s = K smoothing spline of "conserved_offset", both within ~1 nm of
    the centroids -- is a function of b's configuration, so the shifted
    configurations are not exchangeable with the observed one: between
    the vertices the curve overshoots the membrane, a shifted cluster
    lands on the overshoot, farther from ring a than the vertices are,
    the null matches fewer pairs and zeta is biased in favour of
    columns. In the research pilot of 2026-09-25 (simulated axons of
    realistic, concave geometry) both kinds rejected far above the 0.05
    level in the LEAK-FREE M1 replicates, with a positive mean z_A, and
    more so with a clean measurement; the leak doubled it. Fitted to the
    other rings only, the curve
    is independent of b under the null (the rings' scatters are
    independent), so rotating b's (s, r) on it is an exchangeable move
    up to the curve's ESTIMATION ERROR. What remains, and its expected
    direction: write the true membrane M, the estimate M_hat = M -
    delta (delta the error along the normal, a smooth field of
    variance h sigma^2 at a's vertices when fitted to a's K centroids
    scattered sigma about M, h = dof_fit / K ~ 0.3 the hat-matrix
    leverage) and a's and b's clusters at M + eps. The normal distance
    between a's cluster at s' and b's cluster shifted from s to s' is
    eps_b(s) + delta(s) - delta(s') - eps_a(s'), where delta(s') = -(H
    eps_a)(s') is a's own smoothed scatter: its variance is sigma^2 [1
    + h + (1 - 2h + sum_j H_ij^2)] = 2 sigma^2 for a projection
    smoother (sum_j H_ij^2 = H_ii = h), exactly the observed spread of
    b - a, and BELOW it for the P-spline (the ridge penalty makes sum_j
    H_ij^2 < h) and whenever delta(s) and delta(s') are positively
    correlated (shifts shorter than a knot interval): the null's
    clusters sit no farther from a's than the observed ones, the null
    matches at least as many pairs, and the residual bias is
    conservative (zeta <= 0 in expectation), of the order of the
    penalty's share of the leverage. What is NOT removed: structure of
    the true membrane below the knot scale, which the P-spline smooths
    away and both rings share (the observed pair is aligned on it, the
    shifted one is not): a liberal bias of the order of that structure's
    variance over the scatter's -- 12 nm rms on the Fourier-perturbed
    ellipse of the arc-null validator against 30 nm of scatter (1.16
    of the observed spread), nothing on a membrane of curvature radius
    >= 600 nm -- and, with exact clusters on such a membrane (no
    scatter), the whole of it. The harness measures both directions
    paired with the interpolating kind on the same rings and draws.
    What the pairwise argument does NOT cover is the joint null of
    ``axon_joint_null``: there every non-reference ring is moved on
    its own leave-ring-out membrane at once, so the membrane of ring k
    (fitted to its partners' OBSERVED centroids) no longer holds its
    partners' current positions once they are moved too, the
    partner-pull that makes the pairwise null conservative is lost and
    the estimation error alone remains, which is liberal: measured
    (statistics review of 2026-09-25, fresh seeds, R 300 axons of 3
    rings, n_null 199, tau 60) p_excess <= 0.05 in 0.100 on the ellipse
    at K 40 / 30 nm scatter (exact rate 0.037; p_two_sided 0.047, mean
    z_A +0.44), 0.097 at K 37 / 65 nm (interpolating 0.063: WORSE), and
    on a contour library with the true centres it is clearly liberal,
    as is the interpolating kind
    -- the library's shared sub-knot roughness (``tools.mps_membrane.
    smooth_polygon_membrane``) dominates both. The joint pooled null is
    therefore reported as NOT level one-sided; the arc-null validator
    (section 3) asserts its two-sided rate against the specification's
    ceiling and prints the one-sided one, and D-28 records it.
    Re-review of 2026-09-25 (fresh seeds, root 9151, R 300 pairs /
    axons, n_null 199, tau 60, paired with the interpolating kind; the
    numbers that D-28 must carry): H5B_SPEC S2's "expected direction:
    conservative" is contradicted in EVERY case measured -- pairwise on
    the Fourier contour at K 20 the leave-ring-out curve does NOT halve
    the bias S3 asks it to halve (+0.358 with 30 nm scatter, p_excess
    <= 0.05 in 0.060, interpolating +0.447; +0.488 with exact clusters,
    0.083, interpolating +0.453: the curve fitted to 20 centroids with
    14 knots has an estimation error of the scatter's own order), only
    at K 40 does it (+0.054 / -0.038 on the Fourier / ellipse 30 nm
    cases, interpolating +0.229 / +0.288); the joint null of 3-ring
    axons gives mean z_A +0.247 (ellipse 30 nm K 40, t +4.4), +0.399
    (Fourier 30 nm K 40, t +7.1), +0.413 (Fourier 65 nm K 37, t +7.8,
    WORSE than the interpolating +0.325) and +0.381 (Fourier 65 nm K
    22), p_excess <= 0.05 in 0.047-0.073 against exact 0.037-0.039;
    and in the research grid, after the contour fix of grid v2 (library
    contours smoothed to their 600 nm P-spline), ``pooled_z_A`` is
    biased upward under H0 about as much as the pre-registered
    interpolating null. The ``pooled_*`` columns of the power grid are therefore
    NOT a level test at the realistic geometry, as biased as the
    pre-registered null and in the same direction; a real axon's
    pooled_z_A is read against the calibration table of the matching
    null cell (``power_columns.py summarize``) or its own ``simnull``,
    never against the nominal level.
    With ``other_rings`` the membrane is fitted to ALL their clusters
    (usable or not); the keywords are ignored with a warning under the
    other kinds.

    Why this null: a shift along the membrane keeps everything of ring b
    except its phase (the clusters' spacings, their number, the
    perimeter, the density per um), which is the one thing a column
    would fix; a rotation about a centre would take the clusters off
    the membrane, and a Poisson ring would lose the hard core of real
    clusters. Reflection adds the mirror phase (02_investigacion B3).

    The generator is ``default_rng(SeedSequence(random_seed,
    spawn_key=(a.index, b.index)))``: the same seed gives the same
    replicates, another seed or other ring indices give others, and the
    observed pairs never depend on the seed. With a ``Ring`` the
    geometry is built with ``ring_geometry(ring,
    include_suspect=include_suspect)`` and the result is stamped with
    the argument; with a ``RingGeometry``, ``include_suspect=False``
    also masks out the clusters its ``suspect`` marks
    (``_geometry_of``) and the result is stamped with the flag the
    masks imply (True only when every cluster entered; a warning when
    the argument disagreed: ``_include_suspect_stamp``), and
    ``usable_a``/``usable_b`` record the masks that actually entered.

    Returns
    -------
    RingPairMatch
        With ``E_dir``, ``E_star``, ``zeta`` and the three p-values; NaN
        ``delta_bar_nm`` and ``d_median_nm`` without a pair.
    """
    tau, n_rep, frac = _check_null_args(tau_nm, n_null, min_shift_fraction)
    kind = _check_kind(null_kind)
    warnings_: List[str] = []
    ga = _geometry_of(a, include_suspect, warnings_)
    gb = _geometry_of(b, include_suspect, warnings_)
    gb = _resolve_membrane(gb, kind, membrane, other_rings, warnings_)
    stamp = _include_suspect_stamp((a, b), (ga, gb), include_suspect, warnings_)
    label = _pair_label(ga, gb, tau)
    i_a, j_b, d = _observed_match(ga, gb, tau)
    n_obs = int(i_a.size)
    k_a, k_b = ga.n_usable, gb.n_usable
    min_k = min(k_a, k_b)
    rng = _pair_seed(random_seed, ga, gb)
    shifts, reflect, min_shift = _draw_shifts(rng, n_rep, gb, frac, warnings_, kind)
    positions = _shifted_positions(gb, shifts, reflect, kind)
    null = _null_counts(ga.centroids_nm, ga.usable, positions, gb.usable, np.array([tau]))[0]
    e_star, sd_star, zeta = _null_summary(n_obs, null, min_k, warnings_, label)
    p_excess, p_deficit, p_two = _phipson_smyth(null, n_obs)
    if n_obs:
        delta_bar = np.asarray((gb.centroids_nm[j_b] - ga.centroids_nm[i_a]).mean(axis=0),
                               dtype=np.float64)
        d_median = float(np.median(d))
    else:
        delta_bar = np.full(2, np.nan, dtype=np.float64)
        d_median = float("nan")
    return RingPairMatch(
        ring_a=ga.index, ring_b=gb.index, tau_nm=tau,
        i_a=i_a, j_b=j_b, d_nm=d,
        n_matched=n_obs, K_a=k_a, K_b=k_b,
        E_dir=n_obs / min_k if min_k else float("nan"),
        E_sym=2.0 * n_obs / (k_a + k_b) if (k_a + k_b) else float("nan"),
        E_star=e_star, sd_star=sd_star, zeta=zeta,
        p_excess=p_excess, p_deficit=p_deficit, p_two_sided=p_two,
        delta_bar_nm=delta_bar, d_median_nm=d_median,
        null_n_matched=null, null_shift_nm=shifts, null_reflect=reflect,
        n_null=n_rep, random_seed=int(random_seed), min_shift_nm=min_shift,
        include_suspect=stamp, warnings=warnings_,
        usable_a=ga.usable.copy(), usable_b=gb.usable.copy(),
        labels_a=ga.labels.copy(), labels_b=gb.labels.copy(),
        null_kind=kind,
    )


# ============================================================================
# Eclipse curve over tau
# ============================================================================

def _leave_one_out_max_deviation(curves: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    For each of the n curves in the columns of ``curves`` (T, n): the
    maximum over the T rows of |x - mean| / sd, mean and sample sd (ddof
    1) taken over the OTHER n - 1 curves, in closed form from the row
    sums and sums of squares. A deviation over a zero sd is 0 when the
    deviation is 0 (the curve equals the constant the others share) and
    +inf otherwise. Needs n >= 3 (a sample sd of n - 1 >= 2 values).
    """
    n = curves.shape[1]
    if n < 3:
        raise ValueError(f"leave-one-out studentisation needs at least 3 curves, got {n}")
    total = curves.sum(axis=1, keepdims=True)
    total_sq = (curves ** 2).sum(axis=1, keepdims=True)
    mean_out = (total - curves) / (n - 1)
    var_out = (total_sq - curves ** 2 - (n - 1) * mean_out ** 2) / (n - 2)
    sd_out = np.sqrt(np.maximum(var_out, 0.0))
    dev = np.abs(curves - mean_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd_out > 0.0, dev / sd_out, np.where(dev > 0.0, np.inf, 0.0))
    return np.asarray(t.max(axis=0), dtype=np.float64)


leave_one_out_max_deviation = _leave_one_out_max_deviation
"""Public alias of ``_leave_one_out_max_deviation`` (H4: ``tools.mps_unroll`` reuses the D-24c global test verbatim)."""


@dataclass
class EclipseCurve:
    """
    The eclipse of one pair of rings over a grid of tolerances, every
    tau judged against the SAME null replicates (D-03: the curve is
    always reported, whatever tau_0 is).

    Per tau: the observed count and ``E_dir_obs``, the null's ``E_star``
    and ``sd_star`` (as ``eclipse_test`` would give at that tau with the
    same seed), ``zeta`` (NaN where the null is constant) and the
    pointwise envelope, the ``ENVELOPE_PERCENTILES`` of the null E_dir.
    ``t_max_obs`` is the largest |E_dir_obs - E_star| / sd_star over the
    grid (+inf where the null is constant and the observed count is
    not) and ``p_global`` its Monte Carlo p-value against the same
    statistic of every replicate, each scored against the other B
    curves of the set so that the B + 1 statistics are exchangeable
    (``eclipse_curve``): a max-studentised-deviation global test in the
    spirit of Myllymaki et al. (2017), which reads the whole curve at
    once instead of the tau where it happens to leave the envelope.
    ``null_n_matched`` is the (T, B) matrix of null counts behind all of
    it; every replicate's count is non-decreasing in tau, because the
    matching is of maximum cardinality within tau and the admissible
    set only grows with tau (D-24b, ``match_rings``). ``include_suspect``
    is stamped as in ``eclipse_test``; ``null_kind`` is the curve the
    null walked (``NULL_KINDS``, D-25).
    """

    ring_a: int
    ring_b: int
    tau_grid_nm: NDArray[np.float64]
    n_matched_obs: NDArray[np.int64]
    E_dir_obs: NDArray[np.float64]
    E_star: NDArray[np.float64]
    sd_star: NDArray[np.float64]
    zeta: NDArray[np.float64]
    envelope_lo: NDArray[np.float64]
    envelope_hi: NDArray[np.float64]
    t_max_obs: float
    p_global: float
    n_null: int
    random_seed: int
    include_suspect: bool
    null_n_matched: NDArray[np.int64]
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    K_a: int
    K_b: int
    min_shift_nm: float
    warnings: List[str] = field(default_factory=list)
    null_kind: str = "interpolating"


def eclipse_curve(
    a: Union[Ring, RingGeometry],
    b: Union[Ring, RingGeometry],
    tau_grid_nm: Sequence[float],
    *,
    n_null: int = 1999,
    random_seed: int = 0,
    min_shift_fraction: float = 0.5,
    include_suspect: bool = True,
    null_kind: str = "interpolating",
    membrane: Optional[SmoothPath] = None,
    other_rings: Optional[Sequence[Union[Ring, RingGeometry]]] = None,
) -> EclipseCurve:
    """
    ``eclipse_test`` over a grid of tau with one set of null replicates:
    the B shifted configurations of ring b are drawn once (the same
    draws ``eclipse_test`` makes from the same seed), each replicate's
    distance matrix is formed once, and only the assignment is run per
    tau. The curve is therefore internally consistent (a replicate that
    matches many pairs at 60 nm is the same one at 80 nm) and at any tau
    of the grid reproduces ``eclipse_test`` exactly. ``null_kind`` as in
    ``eclipse_test`` (D-25), recorded in the result; ``membrane`` /
    ``other_rings`` as there for "pooled_offset" (H5-B).

    The global test: ``t_max_obs = max_tau |E_dir_obs - E*| / sd*`` with
    E* and sd* the mean and sample sd of the B replicates, and every
    replicate r is scored the same way against the other B curves of
    the set (the observed one and the B - 1 other replicates; closed
    form from the per-tau sums and sums of squares, no extra
    assignment), ``p_global = (#{t_r >= t_obs} + 1) / (B + 1)``. Each
    of the B + 1 curves is thus judged against the same number of
    others by the same rule, so under the null the B + 1 statistics are
    exchangeable and the p-value is exact at every B. The H3
    specification described scoring each replicate against the mean and
    sd of all B replicates, itself included, and called the O(1/B)
    difference conservative; it is liberal (a replicate is pulled
    towards a mean and sd that contain it: measured p_global <= 0.05 in
    11.0 % of exchangeable Gaussian curves at B = 19, 5.8 % at B = 99,
    against 4.9 % for this scoring; review of 2026-09-23), which is why
    the symmetric scoring is used. At a tau where the other B curves are
    constant the studentised deviation is 0 when the curve equals that
    constant and +inf when it does not (a warning names the taus where
    the observed count differs from a constant null: the observed curve
    is then more extreme than every replicate there). A
    max-studentised-deviation global test in the spirit of Myllymaki
    et al. (2017).
    """
    grid = np.asarray(tau_grid_nm, dtype=np.float64).ravel()
    if grid.size == 0 or not (np.isfinite(grid).all() and (grid > 0.0).all()):
        raise ValueError("tau_grid_nm must hold positive finite tolerances")
    _, n_rep, frac = _check_null_args(float(grid[0]), n_null, min_shift_fraction)
    kind = _check_kind(null_kind)
    warnings_: List[str] = []
    ga = _geometry_of(a, include_suspect, warnings_)
    gb = _geometry_of(b, include_suspect, warnings_)
    gb = _resolve_membrane(gb, kind, membrane, other_rings, warnings_)
    stamp = _include_suspect_stamp((a, b), (ga, gb), include_suspect, warnings_)
    k_a, k_b = ga.n_usable, gb.n_usable
    min_k = min(k_a, k_b)
    n_obs = np.array([_observed_match(ga, gb, float(t))[0].size for t in grid], dtype=np.int64)
    rng = _pair_seed(random_seed, ga, gb)
    shifts, reflect, min_shift = _draw_shifts(rng, n_rep, gb, frac, warnings_, kind)
    positions = _shifted_positions(gb, shifts, reflect, kind)
    null = _null_counts(ga.centroids_nm, ga.usable, positions, gb.usable, grid)     # (T, B)
    n_tau = grid.size
    e_star = np.full(n_tau, np.nan)
    sd_star = np.full(n_tau, np.nan)
    zeta = np.full(n_tau, np.nan)
    for t in range(n_tau):
        e_star[t], sd_star[t], zeta[t] = _null_summary(
            int(n_obs[t]), null[t], min_k, warnings_, _pair_label(ga, gb, float(grid[t])))
    if min_k:
        e_obs = n_obs / min_k
        e_null = null / min_k
        lo, hi = np.percentile(e_null, ENVELOPE_PERCENTILES, axis=1)
    else:
        e_obs = np.full(n_tau, np.nan)
        e_null = np.full(null.shape, np.nan)
        lo = hi = np.full(n_tau, np.nan)
    if min_k and n_rep >= 2:
        t_all = _leave_one_out_max_deviation(np.concatenate([e_obs[:, None], e_null], axis=1))
        t_max = float(t_all[0])
        t_null = t_all[1:]                                                       # (B,)
        p_global = (int(np.count_nonzero(t_null >= t_max)) + 1) / (n_rep + 1)
    else:
        t_max, p_global = float("nan"), 1.0
        warnings_.append(f"rings {ga.index}-{gb.index}: no global test (t_max_obs NaN, p_global 1): "
                         + ("no usable cluster on one side" if not min_k
                            else f"n_null {n_rep} < 2, no sd to studentise with"))
    # A constant null equals its first replicate at every tau where sd is 0.
    frozen = ~(np.isfinite(sd_star) & (sd_star > 0.0)) & (n_obs != null[:, 0])
    if np.any(frozen):
        taus = ", ".join(f"{v:g}" for v in grid[frozen])
        warnings_.append(f"rings {ga.index}-{gb.index}: at tau {taus} nm the null is constant and "
                         "the observed count differs from it: the observed curve is more extreme "
                         "than every replicate there (t_max_obs inf)")
    return EclipseCurve(
        ring_a=ga.index, ring_b=gb.index, tau_grid_nm=grid,
        n_matched_obs=n_obs, E_dir_obs=np.asarray(e_obs, dtype=np.float64),
        E_star=e_star, sd_star=sd_star, zeta=zeta,
        envelope_lo=np.asarray(lo, dtype=np.float64), envelope_hi=np.asarray(hi, dtype=np.float64),
        t_max_obs=t_max, p_global=p_global,
        n_null=n_rep, random_seed=int(random_seed), include_suspect=stamp,
        null_n_matched=null, null_shift_nm=shifts, null_reflect=reflect,
        K_a=k_a, K_b=k_b, min_shift_nm=min_shift, warnings=warnings_,
        null_kind=kind,
    )


# ============================================================================
# Joint null per axon
# ============================================================================

@dataclass
class JointNull:
    """
    The null of one axon over its adjacent ring pairs at once (02 B3
    S6, 01 S1.6): pairs that share a ring are not independent, so their
    counts are summed and the sum is judged against replicates in which
    every ring but the first is shifted by its own independent draw.

    ``pairs`` are the (ring index, ring index) pairs used, consecutive by
    ``Ring.index`` (a gap breaks the pair, with a warning);
    ``n_matched_obs`` their observed counts, ``T_obs`` the sum,
    ``T_null`` the (B,) replicate sums and ``null_n_matched`` the (B,
    n_pairs) counts behind them. ``T_A`` is the mean over pairs of
    (E_dir - E*), with E* from THIS null (the per-axon effect size, the
    statistic the between-axon inference of H5+ works on), ``z_A`` the
    standardised ``T_obs`` (NaN on a constant null) and the p-values as
    in ``RingPairMatch``. ``null_shift_nm``/``null_reflect`` are (B,
    n_rings) with column 0 identically 0 and False: the first ring is
    never moved; from the second shifted ring on, a draw with the
    previous ring's reflection whose shift lies within that ring's
    minimum shift of the offset that re-creates the previous pair's
    observed phase is redrawn (D-24d, see ``axon_joint_null`` and
    ``_redraw_relative``); ``origin_offset_nm`` (n_rings; the arc on
    ring k's curve of ring k-1's tour origin, 0 for column 0 and 1) and
    ``tour_orientation`` (n_rings; +1 when ring k's tour runs the same
    way round as ring k-1's, -1 otherwise) are the quantities that
    rule used, recorded so that the harness can verify it. ``rings``
    lists the ring indices in the column order; ``include_suspect`` is
    stamped as in ``eclipse_test``; ``null_kind`` is the curve every
    shifted ring walked (``NULL_KINDS``, D-25; the origin offsets and
    the lengths of the D-24d rule are then measured on that curve).
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
    null_n_matched: NDArray[np.int64]
    null_shift_nm: NDArray[np.float64]
    null_reflect: NDArray[np.bool_]
    E_dir_obs: NDArray[np.float64]
    E_star: NDArray[np.float64]
    min_K: NDArray[np.int64]
    min_shift_nm: NDArray[np.float64]
    min_shift_fraction: float
    include_suspect: bool
    warnings: List[str] = field(default_factory=list)
    origin_offset_nm: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    tour_orientation: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    null_kind: str = "interpolating"


def _sorted_geometries(
    rings: Sequence[Union[Ring, RingGeometry]], include_suspect: bool, warnings_: List[str],
) -> List[RingGeometry]:
    """The geometries in ascending ring index; a repeated index is an
    error (two rings cannot both be 'the next one')."""
    geoms = sorted((_geometry_of(r, include_suspect, warnings_) for r in rings), key=lambda g: g.index)
    indices = [g.index for g in geoms]
    if len(set(indices)) != len(indices):
        raise ValueError(f"repeated ring index in {indices}")
    return geoms


def _adjacent_positions(geoms: Sequence[RingGeometry], warnings_: List[str]) -> List[Tuple[int, int]]:
    """Positions (q, q + 1) in ``geoms`` whose ring indices are
    consecutive; a gap in the indices breaks the pair, with a warning."""
    pairs: List[Tuple[int, int]] = []
    for q in range(len(geoms) - 1):
        lo, hi = geoms[q].index, geoms[q + 1].index
        if hi == lo + 1:
            pairs.append((q, q + 1))
        else:
            warnings_.append(f"rings {lo} and {hi} are not consecutive (gap of {hi - lo - 1} "
                             "ring index): they form no adjacent pair")
    return pairs


def _signed_area_nm2(contour: NDArray[np.float64]) -> float:
    """Shoelace signed area of a closed polygon: positive when the tour
    runs counter-clockwise (the polar order of ``reconstruct_perimeter``
    before 2-opt), negative when clockwise."""
    x, y = contour[:, 0], contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _nearest_arc_on_path(path: SmoothPath, p_nm: NDArray[np.float64]) -> float:
    """Curve arc of the point of the dense table nearest to ``p_nm``:
    the projection of p on every sub-chord, clipped to the sub-chord,
    the smallest distance wins (the harness's ``nearest_on_polygon`` on
    the table gives the same arc)."""
    a = path.points_nm
    b = np.roll(a, -1, axis=0)
    ab = b - a
    p = np.asarray(p_nm, dtype=np.float64).reshape(2)
    edge2 = np.maximum(path.edges_nm ** 2, 1e-300)
    t = np.clip(((p - a) * ab).sum(axis=1) / edge2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d2 = ((p - proj) ** 2).sum(axis=1)
    v = int(np.argmin(d2))
    return float(np.mod(path.cum_nm[v] + t[v] * path.edges_nm[v], path.length_nm))


nearest_arc_on_path = _nearest_arc_on_path
"""Public alias of ``_nearest_arc_on_path`` (H4: ``tools.mps_unroll`` projects cluster centroids on the reference curve with it)."""


def _tour_origin_offset(prev: RingGeometry, geom: RingGeometry, kind: str = "interpolating") -> Tuple[float, int]:
    """
    How ring ``geom``'s arc coordinate relates to the previous ring's
    (D-24d): ``(o, sigma)`` with ``o`` the arc on ``geom``'s curve of the
    point nearest the previous ring's tour origin (``prev.contour_nm[0]``,
    the vertex at arc 0 of ``prev``) and ``sigma`` +1 when the two tours
    run the same way round (equal signs of the shoelace area), -1 when
    opposite. A point at arc s of the previous ring sits, physically, at
    about arc sigma s + o of this ring; ``o`` is 0 and ``sigma`` 1 for a
    ring against itself. (``reconstruct_perimeter`` builds its tours
    from the counter-clockwise polar order, so on two real rings
    ``sigma`` is +1 in practice, and ``o`` is the arc between the two
    rings' first vertices: a median 120 nm on independent rings of
    200 nm spacing, one spacing when one ring lacks the other's first
    cluster; it is not 0. Neither is assumed: both are measured.)

    With ``kind`` "conserved_offset" the two curves are the smoothed
    ones (``_null_curve``): the previous ring's origin is row 0 of its
    smoothed table (the point its reflection is taken about) and ``o``
    is that point's arc on this ring's smoothed curve.
    """
    if kind == "interpolating":
        o = _nearest_arc_on_path(geom.path, prev.contour_nm[0])
    else:
        prev_path, _, _ = _null_curve(prev, kind)
        path, _, _ = _null_curve(geom, kind)
        o = _nearest_arc_on_path(path, prev_path.points_nm[0])
    sigma = 1 if _signed_area_nm2(prev.contour_nm) * _signed_area_nm2(geom.contour_nm) >= 0.0 else -1
    return o, sigma


def _redraw_relative(
    rng: np.random.Generator,
    geom: RingGeometry,
    shifts: NDArray[np.float64],
    flips: NDArray[np.bool_],
    prev_shifts: NDArray[np.float64],
    prev_flips: NDArray[np.bool_],
    min_shift_nm: float,
    origin_offset_nm: float = 0.0,
    orientation: int = 1,
    prev_length_nm: Optional[float] = None,
    length_nm: Optional[float] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    The relative-shift exclusion of the joint null (D-24d, amended after
    the re-review of 2026-09-23): the draws (U, reflect) of a ring whose
    reflection equals the previous ring's and whose shift lies within
    ``min_shift_nm`` (cyclic, modulo this ring's curve length L) of the
    offset at which the previous ring's observed phase is re-created
    are redrawn from ``rng`` -- U ~ Uniform(m, L - m), then reflect ~
    Bernoulli(1/2), the order of ``_draw_shifts`` -- until none is.
    L is ``geom.length_nm`` unless ``length_nm`` is given (the smoothed
    curve's length under ``null_kind`` "conserved_offset", D-25).

    Where the re-creation sits. Let the previous ring's arc s map to
    this ring's arc sigma s + o (``_tour_origin_offset``: o the arc of
    the previous ring's tour origin on this ring's curve, sigma +1 for
    tours running the same way round, -1 otherwise), L' the previous
    ring's length, and let both rings be moved by (U', r') and (U, r).
    The pair's observed phase (the same clusters at the same relative
    positions, mirrored when both are reflected) is re-created, for
    every cluster at once, only when r == r' and

    * r == r' == False:  U - sigma U'                      == 0 (mod L)
    * r == r' == True:   U - sigma U' - (2 o + sigma L')   == 0 (mod L)

    (with one reflected and the other not, the relative displacement
    depends on the cluster's arc and no window re-creates the phase).
    Reflection is s -> L - s ABOUT THE TOUR ORIGIN, which is why the
    origins enter in the reflected case only: the first H3 version
    tested |U - U'| < m for both cases, which is right when the two
    rings' tour origins coincide (the harness's copies rings, which
    share their first vertex) and otherwise excludes a harmless window
    and leaves the harmful one: on a copies axon whose third ring lacks
    the second ring's origin cluster (o one spacing off), 8-11 of 1000
    reflect/reflect replicates carried the observed count of the pair
    into T_null. The window is measured against the exact re-creation
    offset of the copies case; on two real rings, whose curves are not
    copies, it is the same window to the accuracy with which one ring's
    arc maps onto the other's (a few nm on adjacent rings of one axon).

    Only the offending replicates are redrawn, so the others keep their
    first draw and every draw stays recorded (``joint_shift_statistic``
    reproduces every replicate). The allowed set is never empty (the
    window has length 2 m < L - 2 m for K >= 3, and the other reflection
    is always allowed); a guard raises RuntimeError after 10 000 rounds.
    """
    length = geom.length_nm if length_nm is None else float(length_nm)
    m = float(min_shift_nm)
    sigma = 1 if int(orientation) >= 0 else -1
    prev_length = length if prev_length_nm is None else float(prev_length_nm)
    mirror_nm = 2.0 * float(origin_offset_nm) + sigma * prev_length
    out_shift = np.array(shifts, dtype=np.float64, copy=True)
    out_flip = np.array(flips, dtype=bool, copy=True)

    def offending() -> NDArray[np.bool_]:
        target = np.where(out_flip, mirror_nm, 0.0)
        d = np.mod(out_shift - sigma * prev_shifts - target, length)
        return np.asarray((out_flip == prev_flips) & (np.minimum(d, length - d) < m), dtype=bool)

    bad = offending()
    for _round in range(10000):
        if not bad.any():
            return out_shift, out_flip
        n_bad = int(bad.sum())
        out_shift[bad] = rng.uniform(m, length - m, size=n_bad)
        out_flip[bad] = rng.random(n_bad) < 0.5
        bad = offending()
    raise RuntimeError(f"ring {geom.index}: the relative-shift exclusion could not be satisfied "
                       "in 10 000 rounds")


def joint_shift_statistic(
    rings: Sequence[Union[Ring, RingGeometry]],
    tau_nm: float,
    shift_nm: Sequence[float],
    reflect: Sequence[bool],
    *,
    include_suspect: bool = True,
    null_kind: str = "interpolating",
    membranes: Optional[Mapping[int, SmoothPath]] = None,
) -> Tuple[int, NDArray[np.int64]]:
    """
    One replicate of the joint null done by hand: ring k is moved by
    ``arc_shift(ring_k, shift_nm[k], reflect[k], kind=null_kind)`` (one
    entry per ring, in ascending ring index), every adjacent pair is
    re-matched with the shifted positions of BOTH its rings, and the
    counts are summed. Returns (T, the count per adjacent pair). The
    first ring is the reference and is never moved: a non-zero shift or
    a reflection for it raises ValueError. ``axon_joint_null`` records
    its draws and its ``null_kind`` so that this function reproduces
    every ``T_null[r]`` from them; with "pooled_offset" the membranes
    are ``membranes`` (ring index -> ``SmoothPath``) or, when None, the
    leave-ring-out membranes rebuilt from the rings (deterministic, the
    same curves ``axon_joint_null`` built).
    """
    kind = _check_kind(null_kind)
    geoms = _sorted_geometries(rings, include_suspect, [])
    geoms = _with_membranes(geoms, kind, membranes, [])
    tau = float(tau_nm)
    shifts = np.asarray(shift_nm, dtype=np.float64).ravel()
    flips = np.asarray(reflect, dtype=bool).ravel()
    if shifts.shape != (len(geoms),) or flips.shape != (len(geoms),):
        raise ValueError(f"one shift and one reflect per ring: {len(geoms)} rings, "
                         f"{shifts.size} shifts, {flips.size} reflects")
    if geoms and (shifts[0] != 0.0 or flips[0]):
        raise ValueError(f"the first ring ({geoms[0].index}) is the reference of the joint null "
                         f"and is never shifted: got shift {shifts[0]} nm, reflect {bool(flips[0])}")
    positions = [g.centroids_nm for g in geoms]
    for k in range(1, len(geoms)):
        positions[k] = arc_shift(geoms[k], float(shifts[k]), bool(flips[k]), kind=kind)
    pairs = _adjacent_positions(geoms, [])
    per_pair = np.array([
        match_rings(positions[i], positions[j], tau,
                    usable_a=geoms[i].usable, usable_b=geoms[j].usable)[0].size
        for i, j in pairs], dtype=np.int64)
    return int(per_pair.sum()), per_pair


def axon_joint_null(
    rings: Sequence[Union[Ring, RingGeometry]],
    tau_nm: float,
    *,
    n_null: int = 1999,
    random_seed: int = 0,
    min_shift_fraction: float = 0.5,
    include_suspect: bool = True,
    null_kind: str = "interpolating",
    membranes: Optional[Mapping[int, SmoothPath]] = None,
) -> JointNull:
    """
    The joint null of one axon: T = the sum over adjacent pairs of the
    matched count, observed against replicates in which every ring but
    the first (lowest index) is shifted along its own polygon by an
    independent (U_k, reflect_k) and every pair is re-matched with the
    shifted positions of BOTH its rings. The pairs (k, k+1) and (k+1,
    k+2) share ring k+1, so their counts are dependent and a product of
    per-pair p-values would be anticonservative; the joint replicate
    carries that dependence into the null (02 B3 S6, B7). ``null_kind``
    as in ``eclipse_test`` (D-25): with "conserved_offset" every
    shifted ring walks its smoothed curve keeping its offsets, and the
    lengths, mean spacings and origin offsets of the D-24d rule below
    are those of the smoothed curves; recorded in the result. With
    "pooled_offset" (H5-B) every shifted ring k walks the leave-ring-out
    membrane of the axon fitted to every ring but k -- ``membranes[k]``
    when given (ring index -> ``SmoothPath``, what ``analyze_columns``
    passes), the membrane the geometry carries, or else built here by
    ``leave_one_out_membranes`` -- keeping its offsets from it; lengths,
    mean spacings and origin offsets of the D-24d rule are then those of
    the membranes (each ring's own, as each ring's curve differs).

    Seeds: ``SeedSequence(random_seed, spawn_key=(JOINT_SPAWN_KEY,))``
    spawned once per ring, in ring order; ring k >= 1 draws its U and
    reflect from its own child as ``_draw_shifts`` does. Ring 0's child
    is spawned and unused, so that adding a ring never changes the
    draws of the others.

    The exclusion |U_k| >= m_k = p_bar_k / 2 (D-11) applies to each
    ring's own shift, i.e. relative to the unshifted ring 0, and that
    is the rule of the first shifted ring (a reflected ring against the
    fixed one never re-creates the pair's phase for every cluster at
    once, so no window is needed there). The count of a pair (k, k+1)
    with k >= 1 depends on the RELATIVE displacement of its two rings:
    when both draw the same reflection and their shifts nearly agree,
    the replicate re-creates that pair's observed phase, which happened
    in about 1 / (2 K) of the replicates (measured 1.1 % at K = 40,
    where those replicates counted 29.8 pairs against 20.8 for the
    rest; review of 2026-09-23). Hence D-24d, the relative-shift
    exclusion: for every shifted ring after the first, (U_k, reflect_k)
    is redrawn from the ring's own generator until reflect_k !=
    reflect_{k-1} or the cyclic distance -- modulo L_k, the length of
    ring k's curve -- between U_k and the offset that re-creates the
    phase is at least m_k: that offset is sigma U_{k-1} when neither
    ring is reflected and sigma U_{k-1} + 2 o_k + sigma L_{k-1} when
    both are, o_k being the arc on ring k's curve of ring k-1's tour
    origin and sigma the relative sense of the two tours
    (``_tour_origin_offset``, ``_redraw_relative``, which derives the
    two cases; re-review of 2026-09-23: reflection is about each ring's
    own tour origin, so the origins cancel only in the unreflected
    case, and the first rule, |U_k - U_{k-1}| < m_k for both cases,
    left the reflect/reflect quarter of the contamination in place
    whenever the origins differed). Nothing is redrawn when the ring's
    own exclusion had to be dropped (m_k = 0). The rule keeps every
    draw recorded, so ``joint_shift_statistic`` reproduces every
    replicate from ``null_shift_nm``/``null_reflect``, and it changes
    nothing for an axon of two rings.

    ``T_A`` is the mean over pairs of (E_dir_obs - E*), E* being the
    mean null count of that pair in THIS null over min(K_a, K_b);
    ``z_A`` and the p-values are those of ``T_obs`` against ``T_null``.
    With no adjacent pair (one ring, or gaps everywhere) the result is
    empty with a warning: T 0, NaN T_A and z_A, p-values 1.
    """
    tau, n_rep, frac = _check_null_args(tau_nm, n_null, min_shift_fraction)
    kind = _check_kind(null_kind)
    warnings_: List[str] = []
    geoms = _sorted_geometries(rings, include_suspect, warnings_)
    geoms = _with_membranes(geoms, kind, membranes, warnings_)
    stamp = _include_suspect_stamp(rings, geoms, include_suspect, warnings_)
    n_rings = len(geoms)
    pairs = _adjacent_positions(geoms, warnings_)
    shifts = np.zeros((n_rep, n_rings), dtype=np.float64)
    flips = np.zeros((n_rep, n_rings), dtype=bool)
    min_shift = np.zeros(n_rings, dtype=np.float64)
    origin_offset = np.zeros(n_rings, dtype=np.float64)
    orientation = np.ones(n_rings, dtype=np.int64)
    positions: List[NDArray[np.float64]] = [g.centroids_nm for g in geoms]
    children = np.random.SeedSequence(int(random_seed), spawn_key=(JOINT_SPAWN_KEY,)).spawn(n_rings)
    for k in range(1, n_rings):
        rng = np.random.default_rng(children[k])
        shifts[:, k], flips[:, k], min_shift[k] = _draw_shifts(rng, n_rep, geoms[k], frac, warnings_, kind)
        if k >= 2:
            origin_offset[k], orientation[k] = _tour_origin_offset(geoms[k - 1], geoms[k], kind)
            if min_shift[k] > 0.0:
                if kind == "interpolating":
                    shifts[:, k], flips[:, k] = _redraw_relative(
                        rng, geoms[k], shifts[:, k], flips[:, k], shifts[:, k - 1], flips[:, k - 1],
                        float(min_shift[k]), origin_offset_nm=float(origin_offset[k]),
                        orientation=int(orientation[k]), prev_length_nm=float(geoms[k - 1].length_nm))
                else:
                    shifts[:, k], flips[:, k] = _redraw_relative(
                        rng, geoms[k], shifts[:, k], flips[:, k], shifts[:, k - 1], flips[:, k - 1],
                        float(min_shift[k]), origin_offset_nm=float(origin_offset[k]),
                        orientation=int(orientation[k]), prev_length_nm=float(_null_curve(geoms[k - 1], kind)[2]),
                        length_nm=float(_null_curve(geoms[k], kind)[2]))
        positions[k] = _shifted_positions(geoms[k], shifts[:, k], flips[:, k], kind)
    n_pairs = len(pairs)
    n_obs = np.zeros(n_pairs, dtype=np.int64)
    min_k = np.zeros(n_pairs, dtype=np.int64)
    null = np.zeros((n_rep, n_pairs), dtype=np.int64)
    for q, (i, j) in enumerate(pairs):
        n_obs[q] = _observed_match(geoms[i], geoms[j], tau)[0].size
        min_k[q] = min(geoms[i].n_usable, geoms[j].n_usable)
        null[:, q] = _null_counts(positions[i], geoms[i].usable, positions[j], geoms[j].usable,
                                  np.array([tau]))[0]
    t_null = null.sum(axis=1)
    t_obs = int(n_obs.sum())
    ring_ids = [g.index for g in geoms]
    if n_pairs == 0:
        warnings_.append(f"rings {ring_ids}: no adjacent pair; the joint null is empty")
        return JointNull(
            tau_nm=tau, pairs=[], n_matched_obs=n_obs, T_obs=0, T_null=t_null,
            T_A=float("nan"), z_A=float("nan"), p_excess=1.0, p_deficit=1.0, p_two_sided=1.0,
            n_null=n_rep, random_seed=int(random_seed), rings=ring_ids,
            null_n_matched=null, null_shift_nm=shifts, null_reflect=flips,
            E_dir_obs=np.zeros(0), E_star=np.zeros(0), min_K=min_k, min_shift_nm=min_shift,
            min_shift_fraction=frac, include_suspect=stamp, warnings=warnings_,
            origin_offset_nm=origin_offset, tour_orientation=orientation, null_kind=kind)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_dir = np.where(min_k > 0, n_obs / np.maximum(min_k, 1), np.nan)
        e_star = np.where(min_k > 0, null.mean(axis=0) / np.maximum(min_k, 1), np.nan)
    if not np.all(min_k > 0):
        warnings_.append("a pair has no usable cluster on one side (min K = 0): its E_dir and E* "
                         "are NaN and T_A is NaN")
    t_a = float(np.mean(e_dir - e_star))
    label = f"joint null of rings {ring_ids} at tau {tau:g} nm"
    _, _, z_a = _null_summary(t_obs, t_null, 1, warnings_, label)
    p_excess, p_deficit, p_two = _phipson_smyth(t_null, t_obs)
    return JointNull(
        tau_nm=tau, pairs=[(geoms[i].index, geoms[j].index) for i, j in pairs],
        n_matched_obs=n_obs, T_obs=t_obs, T_null=t_null,
        T_A=t_a, z_A=z_a, p_excess=p_excess, p_deficit=p_deficit, p_two_sided=p_two,
        n_null=n_rep, random_seed=int(random_seed), rings=ring_ids,
        null_n_matched=null, null_shift_nm=shifts, null_reflect=flips,
        E_dir_obs=np.asarray(e_dir, dtype=np.float64), E_star=np.asarray(e_star, dtype=np.float64),
        min_K=min_k, min_shift_nm=min_shift, min_shift_fraction=frac,
        include_suspect=stamp, warnings=warnings_,
        origin_offset_nm=origin_offset, tour_orientation=orientation, null_kind=kind,
    )


# ============================================================================
# Columns
# ============================================================================

@dataclass
class Column:
    """
    A chain of clusters linked by adjacent matches: ``members`` are
    (ring index, cluster index in that ring's order) in ascending ring,
    ``labels`` their DBSCAN labels in the same order, ``length`` the
    rings spanned, and ``touches_first``/``touches_last`` whether the
    chain reaches the lowest / highest ring index of the axon (a chain
    that does is censored by the axial range: its true length is at
    least this, 02 B6). An unmatched cluster is a column of length 1.
    """

    members: List[Tuple[int, int]]
    labels: List[int]
    length: int
    touches_first: bool
    touches_last: bool


def build_columns(
    matches: Sequence[RingPairMatch],
    ring_indices: Sequence[int],
    *,
    ring_geometries: Optional[Sequence[RingGeometry]] = None,
) -> List[Column]:
    """
    Chains through the adjacent matches (``ring_b == ring_a + 1``; any
    other match is ignored). A cluster has at most one partner on each
    side, so the chains never branch: every cluster is in exactly one
    column, and a cluster with no partner is a column of length 1,
    included because the length distribution needs it (02 B6). Sorted
    by (start ring, cluster index).

    Which clusters exist, and which are usable, comes from the matches
    themselves (``RingPairMatch.usable_a``/``usable_b``, or 0..K-1 when
    a match carries no mask) or, for a ring no match mentions, from
    ``ring_geometries``. ``Column.labels`` are the DBSCAN labels the
    matches or geometries carry, the cluster index where none is known.
    Nothing is dropped silently: every ring of ``ring_indices`` must be
    described by a match or a geometry, every adjacent match must name
    rings of ``ring_indices`` and usable clusters of them, and the
    columns returned hold every registered usable cluster exactly once
    (checked; review of 2026-09-23).

    Parameters
    ----------
    matches : sequence of RingPairMatch
        The adjacent matches at one tau (``AxonColumnsResult.adjacent``).
    ring_indices : sequence of int
        Every ring index of the axon; the lowest and highest set the
        censoring flags.
    ring_geometries : sequence of RingGeometry, optional
        The rings, for their masks and labels.

    Raises
    ------
    ValueError
        A ring described twice with different clusters or labels; an
        adjacent match naming a ring outside ``ring_indices`` or a
        cluster index that is not a usable cluster of that ring; a ring
        of ``ring_indices`` that no match and no geometry describes; a
        cluster with two partners on one side.
    """
    indices = sorted({int(k) for k in ring_indices})
    if not indices:
        return []
    first, last = indices[0], indices[-1]
    clusters: Dict[int, NDArray[np.intp]] = {}
    labels: Dict[int, NDArray[np.int64]] = {}

    def register(ring: int, usable: Optional[NDArray[np.bool_]], k: int,
                 lab: Optional[NDArray[np.int64]]) -> None:
        """Record the usable clusters and labels of a ring, once."""
        members = (np.arange(k, dtype=np.intp) if usable is None
                   else np.flatnonzero(np.asarray(usable, dtype=bool)).astype(np.intp))
        if ring in clusters and not np.array_equal(clusters[ring], members):
            raise ValueError(f"ring {ring}: the matches disagree on its usable clusters")
        clusters[ring] = members
        if lab is not None:
            lab_arr = np.asarray(lab, dtype=np.int64).ravel()
            if ring in labels and not np.array_equal(labels[ring], lab_arr):
                raise ValueError(f"ring {ring}: the matches disagree on its labels")
            labels[ring] = lab_arr

    known = set(indices)
    for g in ring_geometries or ():
        if g.index in known:
            register(g.index, g.usable, g.n_clusters, g.labels)
    adjacent = [m for m in matches if m.ring_b == m.ring_a + 1]
    for m in adjacent:
        if m.ring_a not in known or m.ring_b not in known:
            raise ValueError(f"match {m.ring_a}-{m.ring_b} names a ring outside ring_indices {indices}")
        register(m.ring_a, m.usable_a, m.K_a, m.labels_a)
        register(m.ring_b, m.usable_b, m.K_b, m.labels_b)
    missing = [k for k in indices if k not in clusters]
    if missing:
        raise ValueError(f"ring(s) {missing} of ring_indices are described by no adjacent match and no "
                         "geometry: their clusters are unknown and the column lengths would be wrong")
    registered: Set[Tuple[int, int]] = {(ring, int(i)) for ring, members in clusters.items()
                                        for i in members.tolist()}
    successor: Dict[Tuple[int, int], Tuple[int, int]] = {}
    has_predecessor: Set[Tuple[int, int]] = set()
    for m in adjacent:
        for i, j in zip(np.asarray(m.i_a).tolist(), np.asarray(m.j_b).tolist()):
            head, tail = (m.ring_a, int(i)), (m.ring_b, int(j))
            for node in (head, tail):
                if node not in registered:
                    raise ValueError(f"match {m.ring_a}-{m.ring_b} pairs cluster {node}, which is not a "
                                     f"usable cluster of ring {node[0]} ({clusters[node[0]].size} usable)")
            if head in successor or tail in has_predecessor:
                raise ValueError(f"cluster {head} or {tail} has two partners on one side")
            successor[head] = tail
            has_predecessor.add(tail)
    columns: List[Column] = []
    for ring in indices:
        for i in clusters[ring].tolist():
            node = (ring, int(i))
            if node in has_predecessor:
                continue
            members = [node]
            while node in successor:
                node = successor[node]
                members.append(node)
            labs = [int(labels[r][c]) if r in labels and c < labels[r].size else c
                    for r, c in members]
            columns.append(Column(members=members, labels=labs, length=len(members),
                                  touches_first=members[0][0] == first,
                                  touches_last=members[-1][0] == last))
    placed = [node for col in columns for node in col.members]
    if len(placed) != len(registered) or set(placed) != registered:
        raise ValueError(f"build_columns placed {len(placed)} cluster(s) of {len(registered)} registered "
                         "(internal inconsistency)")
    columns.sort(key=lambda col: (col.members[0][0], col.members[0][1]))
    return columns


# ============================================================================
# Per-axon driver
# ============================================================================

@dataclass
class AxonColumnsResult:
    """
    Everything H3 says about one axon, at the pre-registered tau_0 and
    over the pre-registered grids (D-22: the values travel in
    ``params``; nothing here is a constant).

    ``adjacent`` are the (k, k+1) matches at tau_0, ``k2`` the (k, k+2)
    ones (the persistence control: a column that spans two rings should
    also match across the middle one, a leak should not), ``curves``
    the E(tau) of each adjacent pair over ``params.tau_grid_nm``,
    ``sensitivity`` the adjacent matches at each tau of
    ``params.tau_sensitivity_nm`` (the entry at tau_0 is ``adjacent``
    itself), ``joint`` the joint null and ``T_A``/``z_A``/``p_A`` its
    summary (``p_A`` two-sided), ``columns`` the chains through
    ``adjacent``. ``rings`` lists the ring indices that entered (a ring
    without a contour is left out with a warning). ``leak`` is H4's.
    ``null_kind`` is the curve every null of the run walked
    (``NULL_KINDS``, D-25; "interpolating" = H3).
    """

    source_name: str
    tau0_nm: float
    rings: List[int]
    adjacent: List[RingPairMatch]
    k2: List[RingPairMatch]
    curves: List[EclipseCurve]
    sensitivity: Dict[float, List[RingPairMatch]]
    joint: Optional[JointNull]
    columns: List[Column]
    T_A: float
    z_A: float
    p_A: float
    params: ColumnsParams
    include_suspect: bool
    n_null: int
    random_seed: int
    leak: Any = None
    warnings: List[str] = field(default_factory=list)
    null_kind: str = "interpolating"


def analyze_columns(
    res: RingsResult,
    params: ColumnsParams,
    *,
    include_suspect: bool = True,
    n_null: Optional[int] = None,
    source_name: str = "",
    null_kind: str = "interpolating",
) -> AxonColumnsResult:
    """
    The H3 analysis of one axon from its rings: the adjacent and k+2
    eclipse tests at ``params.tau0_nm``, the curves over
    ``params.tau_grid_nm``, the sensitivity at ``params.tau_sensitivity_nm``,
    the joint null and the columns, all with ``params.random_seed``,
    ``params.min_shift_fraction`` and ``n_null`` (``params.n_null`` when
    None). Every pair's null is seeded by the two ring indices, so the
    adjacent match at tau_0, the curve and the sensitivity of one pair
    share their shifted configurations. ``null_kind`` (``NULL_KINDS``,
    D-25) is passed to every eclipse test, curve and the joint null and
    recorded on the result; the default is the H3 null.

    With ``null_kind="pooled_offset"`` (H5-B) the leave-ring-out
    membranes of the axon are built once (``leave_one_out_membranes``:
    ring k's from every ring but k) and attached to the geometries, so
    that every test, curve and the joint null of the run walk them.

    ``include_suspect=False`` leaves the clusters with a suspect mark
    (H2) out of every matching -- the sensitivity D-20 asks for -- while
    the polygons keep them. With fewer than two rings that have a
    contour the result is empty, with a warning; a ring whose contour
    is missing or inconsistent with its clusters is dropped with a
    warning naming it. Warnings of every step are collected. The input
    is not modified.
    """
    n_rep = int(params.n_null if n_null is None else n_null)
    kind = _check_kind(null_kind)
    seed = int(params.random_seed)
    frac = float(params.min_shift_fraction)
    tau0 = float(params.tau0_nm)
    name = source_name or str(res.source_name)
    warnings_: List[str] = []
    geoms: List[RingGeometry] = []
    for ring in sorted(res.rings, key=lambda r: r.index):
        try:
            geoms.append(ring_geometry(ring, include_suspect=include_suspect))
        except ValueError as exc:
            warnings_.append(f"ring {ring.index} left out: {exc}")
    ring_ids = [g.index for g in geoms]
    if not include_suspect:
        by_id = {r.index: r for r in res.rings}
        marked = [cl for k in ring_ids for cl in by_id[k].clusters if cl.suspect]
        if marked:
            per_mark = ", ".join(
                f"{mark} {sum(1 for cl in marked if mark in cl.suspect)}" for mark in SUSPECT_MARKS)
            warnings_.append(f"include_suspect False: {len(marked)} cluster(s) with a suspect mark "
                             f"left out of every matching ({per_mark}); the polygons keep them")
    empty = AxonColumnsResult(
        source_name=name, tau0_nm=tau0, rings=ring_ids, adjacent=[], k2=[], curves=[],
        sensitivity={float(t): [] for t in params.tau_sensitivity_nm}, joint=None, columns=[],
        T_A=float("nan"), z_A=float("nan"), p_A=float("nan"), params=params,
        include_suspect=bool(include_suspect), n_null=n_rep, random_seed=seed, warnings=warnings_, null_kind=kind)
    if len(geoms) < 2:
        warnings_.append(f"{len(geoms)} ring(s) with a contour: fewer than 2 rings, nothing to "
                         "match across rings")
        return empty
    if len({g.index for g in geoms}) != len(geoms):
        raise ValueError(f"repeated ring index in {ring_ids}")
    membranes: Optional[Dict[int, SmoothPath]] = None
    if kind == "pooled_offset":
        membranes = leave_one_out_membranes(geoms, include_suspect=include_suspect)
        geoms = _with_membranes(geoms, kind, membranes, warnings_)
    by_index = {g.index: g for g in geoms}
    adjacent_pairs = [(g, by_index[g.index + 1]) for g in geoms if g.index + 1 in by_index]
    k2_pairs = [(g, by_index[g.index + 2]) for g in geoms if g.index + 2 in by_index]
    if len(adjacent_pairs) < len(geoms) - 1:
        warnings_.append(f"ring indices {ring_ids} are not all consecutive: "
                         f"{len(adjacent_pairs)} adjacent pair(s) of {len(geoms) - 1} possible")

    def test(ga: RingGeometry, gb: RingGeometry, tau: float, label: str) -> RingPairMatch:
        """One eclipse test with the axon's settings; its warnings are
        collected under ``label`` ("adjacent", "k+2" or "sensitivity")."""
        m = eclipse_test(ga, gb, tau, n_null=n_rep, random_seed=seed, min_shift_fraction=frac,
                         include_suspect=include_suspect, null_kind=kind)
        warnings_.extend(f"{label} {ga.index}-{gb.index}: {w}" for w in m.warnings)
        return m

    adjacent = [test(ga, gb, tau0, "adjacent") for ga, gb in adjacent_pairs]
    k2 = [test(ga, gb, tau0, "k+2") for ga, gb in k2_pairs]
    curves: List[EclipseCurve] = []
    for ga, gb in adjacent_pairs:
        c = eclipse_curve(ga, gb, params.tau_grid_nm, n_null=n_rep, random_seed=seed,
                          min_shift_fraction=frac, include_suspect=include_suspect, null_kind=kind)
        warnings_.extend(f"curve {ga.index}-{gb.index}: {w}" for w in c.warnings)
        curves.append(c)
    sensitivity: Dict[float, List[RingPairMatch]] = {}
    for t in params.tau_sensitivity_nm:
        tau = float(t)
        sensitivity[tau] = (adjacent if tau == tau0
                            else [test(ga, gb, tau, "sensitivity") for ga, gb in adjacent_pairs])
    joint = axon_joint_null(geoms, tau0, n_null=n_rep, random_seed=seed, min_shift_fraction=frac,
                            include_suspect=include_suspect, null_kind=kind, membranes=membranes)
    warnings_.extend(f"joint null: {w}" for w in joint.warnings)
    columns = build_columns(adjacent, ring_ids, ring_geometries=geoms)
    return AxonColumnsResult(
        source_name=name, tau0_nm=tau0, rings=ring_ids, adjacent=adjacent, k2=k2, curves=curves,
        sensitivity=sensitivity, joint=joint, columns=columns,
        T_A=joint.T_A, z_A=joint.z_A, p_A=joint.p_two_sided, params=params,
        include_suspect=bool(include_suspect), n_null=n_rep, random_seed=seed, warnings=warnings_, null_kind=kind)
