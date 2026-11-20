#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tessellation family of clustering strategies: Voronoi density segmentation.

Each localisation owns the Voronoi cell of the plane that is closer to it than
to any other localisation; the inverse of the cell area is a search-radius-free
estimate of the local density (Andronov et al., 2016). A cluster is a connected
set of cells (cells sharing a Voronoi edge) whose density is at or above a
threshold, with two conventions for that threshold:

``threshold_mode='factor'``
    SR-Tesseler (Levet et al., 2015, Nat Methods 12, 1065): keep cells whose
    *first-rank* density (the cell plus its edge neighbours, ``density_rank=1``)
    is at least ``density_factor`` times the mean density. Levet et al.
    segmented all their biological data "by thresholding the first-rank
    density by twice the average localization density in the image", the
    average being "the number of localizations divided by the image area".
``threshold_mode='csr'``
    ClusterViSu (Andronov et al., 2016, Sci Rep 6, 24084): simulate complete
    spatial randomness (CSR) with the same ``n`` "throughout the ROI"
    (``csr_iterations`` sets, 50 in the paper), and place the threshold at the
    *first crossing* of the observed and the mean random density
    distributions, once the observed excess leaves the Monte Carlo envelope
    <n> +- 1.96 sigma (the paper uses cell areas; the crossing is the same on
    1/area). When no bin leaves the envelope or the distributions do not cross (no excess of dense
    cells), the ``csr_quantile`` quantile of the pooled random densities is
    used instead and the diagnostics say so (``threshold_rule``).

The mean density and the random sets (REVIEW_D0 A3)
---------------------------------------------------
"The image" of Levet et al. and "the ROI" of Andronov et al. are, on an axon
ring, regions that include the empty axoplasm: with L = 20 um the convex hull
of the ring is ~5x the band the localizations occupy, so n / hull area made
2 x mean density select almost the whole band (merge 0.50-0.75 in the D0
benchmark). Here the reference region (``reference_region='occupied'``, the
default) is the region the localizations occupy
(``tools.clustering_strategies.OccupiedSupport``: a dilated occupancy grid,
clipped to the window) [derivation: the mean density of the null is taken
over the support of the data; grid and dilation ad hoc]. Both the mean density
and the CSR sets use it. ``reference_region='window'`` restores the literal
rule (n / window area, CSR over the whole window).

Border handling
---------------
Cells at the edge of the window are unbounded in the plain Voronoi diagram.
Here the point set is mirrored across the four sides of the window's bounding
box before the diagram is built, which bounds every original cell by that box
(the perpendicular bisector of a point and its mirror image is the box side),
and each cell is then clipped to the clip window with Sutherland-Hodgman
(convex clip polygon). The clip window is the convex hull of the ROI polygon
when one is given (a concave ROI is approximated by its hull, see
``TessellationDiagnostics.window``), or else the convex hull of the data
pushed outwards by one mean inter-point spacing and made convex again (ad hoc,
see ``hull_margin_spacings``; pushing vertices radially can leave reflex
vertices on elongated hulls, REVIEW_D0 M1). Densities are therefore finite
everywhere; a point outside the window has a cell of zero area, density 0 and
the label -1.

Coincident points (REVIEW_D0 A2)
--------------------------------
Qhull keeps one copy of each repeated point, so the other copy got no edge and
ended as an isolated noise point. The table is now deduplicated before the
tessellation: every copy shares its cell, the density of a cell counts its
multiplicity (rank 0: m / area; rank 1: summed multiplicities over summed
areas), component sizes count copies, and every copy receives the label of
its cell.

Scope
-----
Two-dimensional only: a third column (z) is ignored. The 3D extension with
Voronoi polyhedra (3DClusterViSu, Andronov et al., 2018, Bioinformatics 34,
3004; fast construction in Voro3D, Levet, 2023) is left for a later phase.

Interface: ``VoronoiDensityStrategy`` follows ``ClusteringStrategy`` from
``tools.clustering_strategies`` (``fit(data) -> labels``, ``get_strategy_name``,
``get_params``).

References (APA short)
----------------------
Levet, F., Hosy, E., Kechkar, A., Butler, C., Beghin, A., Choquet, D., &
Sibarita, J.-B. (2015). SR-Tesseler. Nature Methods, 12(11), 1065-1071.
Andronov, L., Orlov, I., Lutz, Y., Vonesch, J.-L., & Klaholz, B. P. (2016).
ClusterViSu. Scientific Reports, 6, 24084.
Andronov, L., Michalon, J., Ouararhni, K., Orlov, I., Hamiche, A., Vonesch,
J.-L., & Klaholz, B. P. (2018). 3DClusterViSu. Bioinformatics, 34(17), 3004-3012.
Levet, F. (2023). Voro3D. Frontiers in Bioinformatics, 3, 1249291.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, Voronoi

from tools.clustering_strategies import OCCUPIED_DILATION, OCCUPIED_GRID_NM, ClusteringStrategy, OccupiedSupport

# --------------------------------------------------------------------------- defaults and their origin
DENSITY_FACTOR_DEFAULT = 2.0
"""Literature: Levet et al. (2015) segmented all biological data with first-rank density > 2 x the mean
density (``density_rank=1``, the default of ``voronoi_factor``). With ``density_rank=0`` (plain 1/area) the
same factor is a looser cut on noisier densities; the factor itself "may vary ... depending on the
difference in molecular densities inside and outside the structure of interest" (Levet et al., 2015)."""

DENSITY_RANK_FACTOR_DEFAULT = 1
"""Literature: SR-Tesseler thresholds the first-rank density (Levet et al., 2015). Simulation (project,
2026-10-16, ``scratchpad/phaseD/fix_algo/t_tess2.py``): on the synthetic benchmark rank 0 recovers more
molecules whole (sparse_lab1 replicate 0, occupied region: exact 0.47 vs 0.12, missed 0.19 vs 0.45),
because the first-rank average lowers the density of the cells at a cluster's rim; neither reaches DBSCAN
25/10 (exact 0.88). The default follows the paper; rank 0 is one spinbox away.

Why neither works on this benchmark [simulation (project), ``fix_algo/t_tess3.py`` / ``t_tess4.py``, one
replicate, 2026-10-16]: (1) the first-rank average mixes each rim cell of a 28-56-localization cluster with the
large background cells around it, so 30 % of a molecule's localizations fall below 2 x mean (14 % with rank 0)
and the molecule is no longer whole; (2) with rank 0 no factor separates both failure modes: factor 2/3/4/6
gives exact 0.47/0.44/0.41/0.38 with missed 0.19/0.20/0.23/0.27 and merge 0.19/0.17/0.15/0.10 (sparse_lab1),
because edge-connected dense cells form chains between neighbouring molecules (no distance limit) while
raising the cut erodes them; (3) an undilated occupied region (mean closer to the band density) is worse
(rank 0: exact 0.38, missed 0.27). Levet et al. (2015) add a second, object-relative threshold and a minimum
distance (50 nm for GlyR, Fig 3h) that are not implemented here. Verdict: NOT recommended for molecule-scale
clusters (see ``docs/clustering_methods.md``)."""

DENSITY_RANK_CSR_DEFAULT = 0
"""Literature: ClusterViSu works on the cell areas, i.e. rank 0 (Andronov et al., 2016)."""

CSR_QUANTILE_DEFAULT = 0.99
"""Ad hoc (one-sided 1 %): FALLBACK only, used when the observed and random density distributions do not
cross (no excess of dense cells). The rule of Andronov et al. (2016) is the first crossing."""

CSR_ITERATIONS_DEFAULT = 50
"""Literature: Andronov et al. (2016) averaged 50 random sets (3DClusterViSu, 2018, also 50). Halving or
doubling it changes the Monte Carlo variance of the mean random distribution by 2x, its error by ~1.4x."""

CSR_HIST_BINS = 64
"""Ad hoc: bins of the observed and random log-density histograms on which the first crossing is read
(each smoothed with a 3-bin moving average, ad hoc, so one empty tail bin is not a 'crossing')."""

CSR_LOCS_PER_SEED_DEFAULT = 1
"""Ad hoc: 1 = the random sets are single, independent localizations (plain CSR). Andronov et al. (2016,
Fig S5) show that a fluorophore localized several times makes plain CSR call random emitters clustered
unless the null repeats each random point; ``csr_locs_per_seed`` > 1 draws n / k random emitters and k
localizations around each (Gaussian, ``csr_seed_sigma_nm``). The project keeps 1 by default because its
question is whether a cluster is one molecule (each molecule's localizations SHOULD form a cluster); set it
to the localizations per label when the question is whether labels aggregate."""

CSR_SEED_SIGMA_NM_DEFAULT = 12.0
"""Literature: ~ NeNA localization precision of the project's dSTORM data (Gazal et al., 2026); only used
when ``csr_locs_per_seed`` > 1."""

MIN_CLUSTER_POINTS_DEFAULT = 5
"""Ad hoc: half the min_samples (10) of the project's DBSCAN; Levet et al. (2015) report no minimum size and
Andronov et al. (2016) use a minimum number of events without giving the value."""

HULL_MARGIN_SPACINGS_DEFAULT = 1.0
"""Ad hoc: without a ROI the clip window is the convex hull of the data pushed outwards by this
many mean inter-point spacings, sqrt(hull area / n), so border cells are about the size of a
CSR cell rather than cut at the hull."""

_EPS = 1e-9


# --------------------------------------------------------------------------- geometry helpers
def polygon_area(poly: np.ndarray) -> float:
    """Unsigned shoelace area of a simple polygon ``(k, 2)``; 0 for fewer than 3 vertices."""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_window(points: np.ndarray) -> np.ndarray:
    """Convex hull of ``points`` as a counter-clockwise polygon ``(k, 2)``."""
    hull = ConvexHull(points)
    return np.asarray(points[hull.vertices], dtype=np.float64)


def expand_convex_polygon(poly: np.ndarray, margin: float) -> np.ndarray:
    """Push every vertex of a convex polygon away from its vertex centroid by ``margin``
    (an approximation of the true offset polygon, exact for a regular polygon's vertices)."""
    c = poly.mean(axis=0)
    d = poly - c
    norm = np.linalg.norm(d, axis=1)
    norm = np.where(norm < _EPS, 1.0, norm)
    return np.asarray(poly + margin * d / norm[:, None], dtype=np.float64)


def clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of ``subject`` ``(k, 2)`` by the convex, counter-clockwise
    polygon ``clip`` ``(m, 2)``. Returns the clipped polygon (possibly empty, ``(0, 2)``)."""
    out = np.asarray(subject, dtype=np.float64)
    m = clip.shape[0]
    for i in range(m):
        if out.shape[0] == 0:
            break
        a = clip[i]
        b = clip[(i + 1) % m]
        d = (b[0] - a[0]) * (out[:, 1] - a[1]) - (b[1] - a[1]) * (out[:, 0] - a[0])
        inside = d >= -_EPS
        if bool(inside.all()):
            continue
        prev = np.roll(out, 1, axis=0)
        d_prev = np.roll(d, 1)
        inside_prev = np.roll(inside, 1)
        crossing = inside != inside_prev
        denom = d_prev - d
        denom = np.where(np.abs(denom) < _EPS, _EPS, denom)
        t = (d_prev / denom)[:, None]
        inter = prev + t * (out - prev)
        k = out.shape[0]
        order = np.concatenate([2 * np.arange(k)[crossing], 2 * np.arange(k)[inside] + 1])
        pts = np.concatenate([inter[crossing], out[inside]], axis=0)
        out = pts[np.argsort(order, kind="stable")]
    return out


def points_in_convex_polygon(points: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Boolean mask of ``points`` ``(k, 2)`` inside (or on) the convex CCW polygon ``clip``."""
    a = clip
    b = np.roll(clip, -1, axis=0)
    ex = (b[:, 0] - a[:, 0])[None, :]
    ey = (b[:, 1] - a[:, 1])[None, :]
    px = points[:, 0][:, None] - a[:, 0][None, :]
    py = points[:, 1][:, None] - a[:, 1][None, :]
    cross = ex * py - ey * px
    return np.asarray((cross >= -_EPS).all(axis=1))


# --------------------------------------------------------------------------- tessellation
@dataclass
class VoronoiCells:
    """Clipped Voronoi tessellation of a 2D point set."""
    area: np.ndarray                      # (n,) cell area in nm^2 after clipping; 0 for degenerate cells
    ridge_points: np.ndarray              # (r, 2) pairs of input indices whose cells share an edge
    window: np.ndarray                    # (k, 2) CCW clip polygon used
    polygons: Optional[List[np.ndarray]] = None   # clipped cell polygons when requested


def voronoi_cells(points: np.ndarray, window: np.ndarray, keep_polygons: bool = False) -> VoronoiCells:
    """Voronoi cells of ``points`` ``(n, 2)`` clipped to the convex CCW ``window``.

    The points are mirrored across the four sides of the window's bounding box so that every
    original cell is bounded; cells with a vertex outside the window are clipped with
    Sutherland-Hodgman, the others are measured directly. Cost: one qhull run on ``5 n``
    points plus a Python loop over the ``n`` cells (about 20 us per interior cell)."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n < 4:
        return VoronoiCells(np.zeros(n), np.zeros((0, 2), dtype=np.int64), window, [] if keep_polygons else None)
    xmin, ymin = np.minimum(window.min(axis=0), pts.min(axis=0))
    xmax, ymax = np.maximum(window.max(axis=0), pts.max(axis=0))
    pad = _EPS + 1e-6 * max(xmax - xmin, ymax - ymin, 1.0)
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad
    mirrored = np.concatenate([
        pts,
        np.column_stack([2.0 * xmin - pts[:, 0], pts[:, 1]]),
        np.column_stack([2.0 * xmax - pts[:, 0], pts[:, 1]]),
        np.column_stack([pts[:, 0], 2.0 * ymin - pts[:, 1]]),
        np.column_stack([pts[:, 0], 2.0 * ymax - pts[:, 1]]),
    ])
    vor = Voronoi(mirrored)
    vertex_inside = points_in_convex_polygon(vor.vertices, window)
    area = np.zeros(n, dtype=np.float64)
    polygons: List[np.ndarray] = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if len(region) < 3 or -1 in region:
            if keep_polygons:
                polygons.append(np.zeros((0, 2)))
            continue
        idx = np.asarray(region, dtype=np.int64)
        poly = vor.vertices[idx]
        if not bool(vertex_inside[idx].all()):
            poly = clip_convex_polygon(poly, window)
        area[i] = polygon_area(poly)
        if keep_polygons:
            polygons.append(poly)
    rp = np.asarray(vor.ridge_points, dtype=np.int64)
    keep = (rp[:, 0] < n) & (rp[:, 1] < n)
    return VoronoiCells(area, rp[keep], window, polygons if keep_polygons else None)


def cell_density(area: np.ndarray, ridge_points: np.ndarray, rank: int = 0,
                 weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Local density per cell: ``m / area`` (rank 0, ClusterViSu) or the first-rank density of
    SR-Tesseler (rank 1): localizations in the cell plus its edge neighbours, divided by their
    summed area. ``weights`` (default 1) is the multiplicity m of each point (coincident copies
    share one cell). Degenerate cells (area 0) get density 0 and do not contribute to neighbours."""
    valid = area > 0
    m = np.ones(area.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if rank == 0:
        return np.where(valid, m / np.where(valid, area, 1.0), 0.0)
    n = area.shape[0]
    ok = valid[ridge_points[:, 0]] & valid[ridge_points[:, 1]]
    r0 = ridge_points[ok, 0]
    r1 = ridge_points[ok, 1]
    sum_area = area.copy()
    sum_area += np.bincount(r0, weights=area[r1], minlength=n)
    sum_area += np.bincount(r1, weights=area[r0], minlength=n)
    count = m + np.bincount(r0, weights=m[r1], minlength=n) + np.bincount(r1, weights=m[r0], minlength=n)
    return np.where(valid, count / np.where(valid, sum_area, 1.0), 0.0)


def connected_cell_clusters(selected: np.ndarray, ridge_points: np.ndarray, min_points: int,
                            weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Labels of the connected components of selected cells joined by shared Voronoi edges;
    -1 for unselected cells and for components with fewer than ``min_points`` localizations
    (``weights`` = multiplicity of each cell's point, default 1)."""
    n = selected.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if not bool(selected.any()):
        return labels
    ok = selected[ridge_points[:, 0]] & selected[ridge_points[:, 1]]
    g = coo_matrix((np.ones(int(ok.sum())), (ridge_points[ok, 0], ridge_points[ok, 1])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    comp = np.asarray(comp, dtype=np.int64)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    size = np.bincount(comp[selected], weights=w[selected], minlength=n)
    keep = selected & (size[comp] >= max(1, int(min_points)))
    if bool(keep.any()):
        _, inv = np.unique(comp[keep], return_inverse=True)
        labels[keep] = inv
    return labels


def first_crossing_threshold(observed: np.ndarray, random_sets: List[np.ndarray],
                             bins: int = CSR_HIST_BINS) -> Optional[float]:
    """Density at which the observed distribution stops exceeding the mean random one (Andronov et al.,
    2016: Monte Carlo envelope <n> +- 1.96 sigma of the random sets and threshold at the "first
    intersection" of the observed and mean random distributions, on areas; the same point on 1/area).

    Distributions are histograms of log10 density on ``bins`` common bins, each normalised to 1 and
    smoothed with a 3-bin moving average (ad hoc). The observed excess must leave the envelope
    (obs > mean + 1.96 sd of the random sets) in at least one bin above the random median; the scan
    starts at the bin of largest such excess and returns the interpolated density where obs first
    falls to the random mean. ``None`` when no bin leaves the envelope (e.g. CSR data) or there is no
    crossing; the caller then falls back to a quantile of the random pool."""
    obs = observed[observed > 0]
    sets = [r[r > 0] for r in random_sets if np.count_nonzero(r > 0) >= 2]
    if obs.size < 2 or len(sets) < 2:
        return None
    lo = float(np.log10(min(obs.min(), min(r.min() for r in sets))))
    hi = float(np.log10(max(obs.max(), max(r.max() for r in sets))))
    if not hi > lo:
        return None
    edges = np.linspace(lo, hi, int(bins) + 1)
    kern = np.ones(3) / 3.0

    def pdf(v: np.ndarray) -> np.ndarray:
        return np.asarray(np.convolve(np.histogram(np.log10(v), edges)[0] / v.size, kern, mode="same"))

    po = pdf(obs)
    pr_all = np.stack([pdf(r) for r in sets])
    pr = pr_all.mean(axis=0)
    sd = pr_all.std(axis=0, ddof=1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    upper = centres >= float(np.log10(np.median(np.concatenate(sets))))
    excess = po - (pr + 1.96 * sd)
    cand = np.flatnonzero(upper & (excess > 0))
    if cand.size == 0:
        return None
    j = int(cand[np.argmax(excess[cand])])
    diff = po - pr
    below = np.flatnonzero(diff[:j] <= 0)
    if below.size == 0:
        return None
    k = int(below[-1])                     # last bin (going down from j) where the excess is gone
    d0, d1 = diff[k], diff[k + 1]          # d0 <= 0 < d1
    t = float(-d0 / (d1 - d0)) if d1 != d0 else 0.0
    return float(10.0 ** (centres[k] + t * (centres[k + 1] - centres[k])))


# --------------------------------------------------------------------------- strategy
@dataclass
class TessellationDiagnostics:
    """What the interface may show after a fit (per-localization arrays: coincident copies share the
    value of their common cell)."""
    threshold_mode: str
    threshold_density: float                  # nm^-2, the cut actually applied
    mean_density: float                       # n / reference area
    density: np.ndarray                       # (n,) per-localisation density
    cell_area: np.ndarray                     # (n,) clipped cell areas
    window: np.ndarray                        # (k, 2) clip polygon
    n_selected: int
    n_clusters: int
    csr_probabilities: np.ndarray = field(default_factory=lambda: np.zeros(0))   # grid of quantile levels
    csr_density_quantiles: np.ndarray = field(default_factory=lambda: np.zeros(0))   # CSR pool at those levels
    observed_density_quantiles: np.ndarray = field(default_factory=lambda: np.zeros(0))   # data at those levels
    cell_polygons: Optional[List[np.ndarray]] = None
    reference_area: float = 0.0               # nm^2 of the region the mean density / CSR sets refer to
    reference_region: str = "occupied"
    threshold_rule: str = ""                  # 'factor', 'first crossing' or 'quantile fallback'
    n_duplicates: int = 0                     # localizations that repeat an earlier one exactly


class VoronoiDensityStrategy(ClusteringStrategy):
    """Voronoi density segmentation (SR-Tesseler / ClusterViSu), 2D.

    Parameters
    ----------
    threshold_mode : 'factor' or 'csr'
        'factor': density >= density_factor x mean density (Levet et al., 2015).
        'csr': density above the first crossing of the observed and random distributions
        (Andronov et al., 2016), ``csr_quantile`` of the random pool as fallback.
    density_factor : float
        Multiplier of the mean density (literature default 2, Levet et al., 2015).
    csr_quantile, csr_iterations : float, int
        Fallback quantile (ad hoc) and number of CSR Monte Carlo sets (50, Andronov et al., 2016).
    csr_locs_per_seed, csr_seed_sigma_nm : int, float
        Localizations per random emitter and their spread (Andronov et al., 2016, Fig S5); 1 = plain
        CSR (default, see the module constants).
    density_rank : 0 or 1
        0: 1/area (ClusterViSu); 1: first-rank density over the cell and its neighbours
        (SR-Tesseler, the default here).
    min_cluster_points : int
        Smallest connected component reported as a cluster, in localizations (ad hoc).
    roi_polygon : (m, 2) array or None
        Clip window; its convex hull is used. None: convex hull of the data pushed out by
        ``hull_margin_spacings`` mean spacings.
    reference_region : 'occupied' or 'window'
        Region of the mean density and of the CSR sets: the occupied region of the data (default,
        derivation, REVIEW_D0 A3) or the whole clip window (the literal rule of both papers).
    support_grid_nm, support_dilation : float, int
        Grid and dilation of the occupied region (ad hoc, ``tools.clustering_strategies``).
    rng_seed : int or None
        Seed of the CSR simulation; the fit is deterministic for a given seed.
    keep_polygons : bool
        Keep the clipped cell polygons in the diagnostics (memory: one small array per point).
    """

    def __init__(
        self,
        threshold_mode: str = "factor",
        density_factor: float = DENSITY_FACTOR_DEFAULT,
        csr_quantile: float = CSR_QUANTILE_DEFAULT,
        csr_iterations: int = CSR_ITERATIONS_DEFAULT,
        density_rank: int = DENSITY_RANK_FACTOR_DEFAULT,
        min_cluster_points: int = MIN_CLUSTER_POINTS_DEFAULT,
        roi_polygon: Optional[np.ndarray] = None,
        hull_margin_spacings: float = HULL_MARGIN_SPACINGS_DEFAULT,
        rng_seed: Optional[int] = 0,
        keep_polygons: bool = False,
        reference_region: str = "occupied",
        support_grid_nm: float = OCCUPIED_GRID_NM,
        support_dilation: int = OCCUPIED_DILATION,
        csr_locs_per_seed: int = CSR_LOCS_PER_SEED_DEFAULT,
        csr_seed_sigma_nm: float = CSR_SEED_SIGMA_NM_DEFAULT,
    ) -> None:
        if threshold_mode not in ("factor", "csr"):
            raise ValueError("threshold_mode must be 'factor' or 'csr'")
        if int(density_rank) not in (0, 1):
            raise ValueError("density_rank must be 0 or 1")
        if not 0.0 < csr_quantile < 1.0:
            raise ValueError("csr_quantile must lie in (0, 1)")
        if reference_region not in ("occupied", "window"):
            raise ValueError("reference_region must be 'occupied' or 'window'")
        if int(csr_locs_per_seed) < 1:
            raise ValueError("csr_locs_per_seed must be >= 1")
        self.threshold_mode = threshold_mode
        self.density_factor = float(density_factor)
        self.csr_quantile = float(csr_quantile)
        self.csr_iterations = int(csr_iterations)
        self.density_rank = int(density_rank)
        self.min_cluster_points = int(min_cluster_points)
        self.roi_polygon = None if roi_polygon is None else np.asarray(roi_polygon, dtype=np.float64)
        self.hull_margin_spacings = float(hull_margin_spacings)
        self.rng_seed = rng_seed
        self.keep_polygons = bool(keep_polygons)
        self.reference_region = reference_region
        self.support_grid_nm = float(support_grid_nm)
        self.support_dilation = int(support_dilation)
        self.csr_locs_per_seed = int(csr_locs_per_seed)
        self.csr_seed_sigma_nm = float(csr_seed_sigma_nm)
        self.diagnostics: Optional[TessellationDiagnostics] = None
        self._scale: Optional[float] = None

    # -- window ------------------------------------------------------------
    def _window(self, xy: np.ndarray) -> np.ndarray:
        if self.roi_polygon is not None:
            return convex_window(self.roi_polygon)
        hull = convex_window(xy)
        spacing = float(np.sqrt(polygon_area(hull) / max(xy.shape[0], 1)))
        # Re-hull: radial pushing can leave reflex vertices on elongated hulls (REVIEW_D0 M1).
        return convex_window(expand_convex_polygon(hull, self.hull_margin_spacings * spacing))

    def _support(self, xy: np.ndarray, window: np.ndarray) -> OccupiedSupport:
        """Occupied region of the data, restricted to cells whose centre lies in the window."""
        sup = OccupiedSupport(xy, self.support_grid_nm, self.support_dilation)
        cells = np.argwhere(sup.mask)
        centres = sup.origin + (cells + 0.5) * sup.cell_nm
        inside = points_in_convex_polygon(centres, window)
        if bool(inside.any()):
            sup.mask[cells[~inside, 0], cells[~inside, 1]] = False
            sup._cells = cells[inside]
        return sup

    # -- CSR reference -------------------------------------------------------
    def _random_set(self, n: int, window: np.ndarray, support: Optional[OccupiedSupport],
                    rng: np.random.Generator) -> np.ndarray:
        """n random localizations inside the window: uniform over the reference region, or, with
        ``csr_locs_per_seed`` = k > 1, n / k uniform emitters with k Gaussian localizations each."""
        lo = window.min(axis=0)
        hi = window.max(axis=0)
        k = self.csr_locs_per_seed
        chunks: List[np.ndarray] = []
        got = 0
        while got < n:
            m = max(2 * (n - got), 16)
            n_seed = max(int(np.ceil(m / k)), 1)
            seeds = support.sample(n_seed, rng) if support is not None else rng.uniform(lo, hi, size=(n_seed, 2))
            cand = (np.repeat(seeds, k, axis=0) + rng.normal(0.0, self.csr_seed_sigma_nm, size=(n_seed * k, 2))
                    if k > 1 else seeds)
            cand = cand[points_in_convex_polygon(cand, window)]
            chunks.append(cand)
            got += cand.shape[0]
        return np.asarray(np.concatenate(chunks, axis=0)[:n])

    def _csr_densities(self, n: int, window: np.ndarray, rng: np.random.Generator,
                       support: Optional[OccupiedSupport] = None) -> List[np.ndarray]:
        """Cell densities of each of the ``csr_iterations`` random sets of ``n`` points (see
        ``_random_set``), same clipping and density rank as the data."""
        sets: List[np.ndarray] = []
        for _ in range(self.csr_iterations):
            sim = self._random_set(n, window, support, rng)
            cells = voronoi_cells(sim, window)
            dens = cell_density(cells.area, cells.ridge_points, self.density_rank)
            sets.append(dens[dens > 0])
        return sets

    # -- interface -------------------------------------------------------------
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Cluster labels per row of ``data`` ``(n, >=2)``, x and y in nm; -1 outside clusters."""
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("data must be (n, >=2) with x, y in the first two columns")
        xy = arr[:, :2]
        n = xy.shape[0]
        labels: np.ndarray = np.full(n, -1, dtype=np.int64)
        self._scale = None
        if n < max(4, self.min_cluster_points):
            self.diagnostics = None
            return labels
        # Coincident points share one cell (REVIEW_D0 A2): tessellate the unique points.
        uxy, inv, mult = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).reshape(-1)
        mult_f = mult.astype(np.float64)
        if uxy.shape[0] < 4:
            self.diagnostics = None
            return labels
        window = self._window(xy)
        cells = voronoi_cells(uxy, window, keep_polygons=self.keep_polygons)
        density_u = cell_density(cells.area, cells.ridge_points, self.density_rank, weights=mult_f)
        support: Optional[OccupiedSupport] = None
        if self.reference_region == "occupied":
            support = self._support(xy, window)
            ref_area = support.area_nm2
            n_ref = int(support.contains(xy).sum())
        else:
            ref_area = polygon_area(window)
            n_ref = int(points_in_convex_polygon(xy, window).sum())
        mean_density = float(n_ref / ref_area) if ref_area > 0 else 0.0
        probs = np.linspace(0.5, 0.999, 100)
        csr_q: np.ndarray = np.zeros(0)
        density = density_u[inv]
        if self.threshold_mode == "factor":
            threshold = self.density_factor * mean_density
            rule = "factor"
            selected_u = (density_u >= threshold) & (density_u > 0)
        else:
            rng = np.random.default_rng(self.rng_seed)
            sets = self._csr_densities(n, window, rng, support)
            pool = np.concatenate(sets) if sets else np.zeros(0)
            crossing = first_crossing_threshold(density, sets)
            if crossing is not None:
                threshold, rule = crossing, "first crossing"
            else:
                threshold = float(np.quantile(pool, self.csr_quantile)) if pool.size else float("inf")
                rule = "quantile fallback"
            csr_q = np.quantile(pool, probs) if pool.size else np.zeros(0)
            selected_u = (density_u > threshold) & (density_u > 0)
        labels_u = connected_cell_clusters(selected_u, cells.ridge_points, self.min_cluster_points, weights=mult_f)
        labels = labels_u[inv]
        positive = density[density > 0]
        sel_area = cells.area[selected_u & np.isfinite(cells.area) & (cells.area > 0)]
        self._scale = float(np.sqrt(np.mean(sel_area))) if sel_area.size else None
        polygons = None
        if cells.polygons is not None:
            polygons = [cells.polygons[int(i)] for i in inv]
        self.diagnostics = TessellationDiagnostics(
            threshold_mode=self.threshold_mode,
            threshold_density=float(threshold),
            mean_density=mean_density,
            density=density,
            cell_area=cells.area[inv],
            window=window,
            n_selected=int(selected_u[inv].sum()),
            n_clusters=int(labels.max() + 1) if labels.size else 0,
            csr_probabilities=probs if self.threshold_mode == "csr" else np.zeros(0),
            csr_density_quantiles=csr_q,
            observed_density_quantiles=np.quantile(positive, probs) if positive.size else np.zeros(0),
            cell_polygons=polygons,
            reference_area=float(ref_area),
            reference_region=self.reference_region,
            threshold_rule=rule,
            n_duplicates=int(n - uxy.shape[0]),
        )
        return labels

    def scale_nm(self) -> Optional[float]:
        """Square root of the mean area of the selected (dense) cells [derivation]: the
        equivalent spacing of the localizations inside clusters, the one length the
        tessellation defines. Informative only: it is an intra-cluster spacing (a few nm),
        not a separation scale, so the edge-curation margin of the analysis is not this
        number (see ``ClusteringStrategy.edge_margin_nm``). ``None`` before ``fit`` or when
        nothing was selected."""
        return self._scale

    def get_strategy_name(self) -> str:
        return "Voronoi density (SR-Tesseler)" if self.threshold_mode == "factor" else "Voronoi density (CSR Monte Carlo)"

    def get_params(self) -> Dict[str, Any]:
        """JSON-serialisable parameters (the ROI polygon as a list of [x, y])."""
        return {
            "strategy": "voronoi_density",
            "threshold_mode": self.threshold_mode,
            "density_factor": self.density_factor,
            "csr_quantile": self.csr_quantile,
            "csr_iterations": self.csr_iterations,
            "density_rank": self.density_rank,
            "min_cluster_points": self.min_cluster_points,
            "roi_polygon": None if self.roi_polygon is None else self.roi_polygon.tolist(),
            "hull_margin_spacings": self.hull_margin_spacings,
            "rng_seed": self.rng_seed,
            "reference_region": self.reference_region,
            "support_grid_nm": self.support_grid_nm,
            "support_dilation": self.support_dilation,
            "csr_locs_per_seed": self.csr_locs_per_seed,
            "csr_seed_sigma_nm": self.csr_seed_sigma_nm,
        }


# --------------------------------------------------------------------------- family
def tessellation_family() -> Dict[str, Dict[str, Any]]:
    """The strategies of this family: name -> {'cls', 'description', 'defaults', 'origins'}.

    ``defaults`` are keyword arguments for ``cls``; ``origins`` labels each one as literature
    (with citation), derivation, simulation or ad hoc."""
    common_defaults: Dict[str, Any] = {
        "min_cluster_points": MIN_CLUSTER_POINTS_DEFAULT,
        "hull_margin_spacings": HULL_MARGIN_SPACINGS_DEFAULT,
        "reference_region": "occupied",
        "support_grid_nm": OCCUPIED_GRID_NM,
        "support_dilation": OCCUPIED_DILATION,
    }
    common_origins: Dict[str, str] = {
        "min_cluster_points": "ad hoc: half this program's DBSCAN min_samples (10); not given in Levet 2015 / Andronov 2016",
        "hull_margin_spacings": "ad hoc: one mean inter-point spacing beyond the convex hull when no ROI is given",
        "reference_region": "derivation: mean density and random sets over the region the data occupy, not the "
                            "hull of the ring (REVIEW_D0 A3); 'window' = literal rule of Levet 2015 / Andronov 2016",
        "support_grid_nm": "ad hoc: 4 x the compatibility eps (25 nm), see OCCUPIED_GRID_NM",
        "support_dilation": "ad hoc: one grid cell around every occupied cell",
    }
    return {
        "voronoi_factor": {
            "cls": VoronoiDensityStrategy,
            "description": "SR-Tesseler: first-rank density above density_factor x the mean density of the occupied "
                           "region, cells merged by shared edges",
            "defaults": {"threshold_mode": "factor", "density_factor": DENSITY_FACTOR_DEFAULT,
                         "density_rank": DENSITY_RANK_FACTOR_DEFAULT, **common_defaults},
            "origins": {
                "density_factor": "literature: Levet et al. (2015), first-rank density thresholded at twice the "
                                  "average localization density",
                "density_rank": "literature: first-rank density (Levet et al., 2015)",
                **common_origins,
            },
        },
        "voronoi_csr": {
            "cls": VoronoiDensityStrategy,
            "description": "ClusterViSu: cells denser than the first crossing of the observed and the mean random "
                           "(CSR, same n, occupied region) density distributions",
            "defaults": {
                "threshold_mode": "csr",
                "density_rank": DENSITY_RANK_CSR_DEFAULT,
                "csr_quantile": CSR_QUANTILE_DEFAULT,
                "csr_iterations": CSR_ITERATIONS_DEFAULT,
                "csr_locs_per_seed": CSR_LOCS_PER_SEED_DEFAULT,
                "csr_seed_sigma_nm": CSR_SEED_SIGMA_NM_DEFAULT,
                "rng_seed": 0,
                **common_defaults,
            },
            "origins": {
                "density_rank": "literature: cell areas, rank 0 (Andronov et al., 2016)",
                "csr_quantile": "ad hoc: fallback only, when the observed and random distributions do not cross; "
                                "the rule is the first crossing (Andronov et al., 2016)",
                "csr_iterations": "literature: 50 random sets (Andronov et al., 2016)",
                "csr_locs_per_seed": "ad hoc: 1 = plain CSR; > 1 repeats each random emitter (Andronov et al., 2016, "
                                     "Fig S5)",
                "csr_seed_sigma_nm": "literature: ~NeNA precision of the project's data (Gazal et al., 2026)",
                "rng_seed": "ad hoc: fixed seed for reproducible thresholds",
                **common_origins,
            },
        },
    }


__all__ = [
    "VoronoiDensityStrategy", "TessellationDiagnostics", "VoronoiCells", "voronoi_cells", "cell_density",
    "connected_cell_clusters", "clip_convex_polygon", "points_in_convex_polygon", "convex_window",
    "expand_convex_polygon", "polygon_area", "tessellation_family", "first_crossing_threshold",
]
