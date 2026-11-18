#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph / topological family of clustering strategies for localisation tables.

Two strategies with the ``ClusteringStrategy`` interface of ``tools.clustering_strategies`` (``fit(data) -> labels``
with -1 for unclustered points, ``get_strategy_name``, ``get_params``), built only on numpy and scipy:

- ``ToMAToStrategy``: Topological Mode Analysis Tool (Chazal, Guibas, Oudot & Skraba, 2013, J. ACM 60(6), 41),
  applied to SMLM by Pike et al. (2020, Bioinformatics 36(5), 1614-1621; R package RSMLM). A density is estimated at
  every localisation, a neighbourhood graph is built, each point is attached to its densest neighbour (discrete
  gradient ascent, union-find) and two density peaks are merged when the lower one's prominence -- its persistence,
  birth density minus the density at which it meets a higher peak -- is below a threshold tau. The persistence
  diagram (birth, death) of every peak is kept as a diagnostic so tau can be read off its largest gap.
- ``GraphCommunityStrategy``: a multi-scale neighbourhood-graph strategy INSPIRED BY StormGraph (Scurll et al., 2019,
  bioRxiv 515627; preprint, the research note located no peer-reviewed version). It is NOT the published StormGraph:
  StormGraph weights edges by Monte Carlo over per-localisation uncertainties and finds a multi-level hierarchy with
  Infomap community detection. This substitute keeps the parts that need no new dependency (radius graph, Gaussian
  edge weights, node-degree threshold calibrated on a random cloud at level alpha, scale chosen on the plateau of the
  number of connected components) and replaces community detection by connected components of the pruned graph.
  It stands in until the project decides to add a community-detection dependency or to import external labels.

Every parameter states its origin: literature (short APA citation), derivation, simulation (project), or ad hoc.
Defaults verified on the synthetic benchmark on 2026-10-16 (``scratchpad/phaseD/VERIF_DEFAULTS_D0.md`` sections 5-6,
runs B-E; official run in ``docs/sims/clustering_benchmark/``):

- ToMATo follows Pike et al. (2020): a RADIUS graph (``k_graph=None``, ``radius_nm=20``) and a density counted around
  each point within that scale (``density='kde'``, ``bandwidth_nm=10`` = radius / 2). With the earlier k-NN graph
  every point had neighbours, so nothing was ever noise (missed 0, split 0.22-0.44); with the radius graph a point
  that cannot be linked stays alone and ``min_cluster_size`` sends it to -1, which is how Pike et al. obtain noise.
- GraphCommunity uses a FIXED radius of 25 nm (the scan over radii chose large radii and merged clusters, merge
  0.24-0.74) and draws its random clouds in the region the data occupy (``OccupiedSupport``), not in the bounding box
  of a ring, which includes the empty axoplasm (REVIEW_D0 M3).

References (APA 7)
------------------
Chazal, F., Guibas, L. J., Oudot, S. Y., & Skraba, P. (2013). Persistence-based clustering in Riemannian manifolds.
    Journal of the ACM, 60(6), Article 41. https://doi.org/10.1145/2535927
Pike, J. A., Khan, A. O., Pallini, C., Thomas, S. G., Mund, M., Ries, J., Poulter, N. S., & Styles, I. B. (2020).
    Topological data analysis quantifies biological nano-structure from single molecule localization microscopy.
    Bioinformatics, 36(5), 1614-1621. https://doi.org/10.1093/bioinformatics/btz788
Scurll, J. M., Abraham, L., Zheng, D. W., Tafteh, R., Chou, K. C., Gold, M. R., & Coombs, D. (2019). StormGraph: A
    graph-based algorithm for quantitative clustering analysis of diverse single-molecule localization microscopy
    data [Preprint]. bioRxiv. https://doi.org/10.1101/515627

Units: the strategies are unit-agnostic, but the defaults are written for coordinates in nanometres (as the rest of
MPS Explorer uses after the Picasso pixel conversion), in 2D or 3D.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tools.clustering_strategies import ClusteringStrategy, OccupiedSupport, core_distance_scale

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Volume of the unit ball in d dimensions, used to turn a k-th neighbour distance into a density (derivation:
# f = k / (n V_d r_k^d) is the classical k-NN density estimator; the 1/n factor is dropped so that f is in points
# per nm^d and does not change when the field of view is cropped).
_UNIT_BALL_VOLUME: Dict[int, float] = {1: 2.0, 2: float(np.pi), 3: 4.0 * float(np.pi) / 3.0}


# --------------------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------------------

def _as_points(data: npt.ArrayLike) -> FloatArray:
    """Return ``data`` as a C-contiguous float64 array of shape (n, d), d in {1, 2, 3}."""
    pts = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.ndim != 2 or pts.shape[1] not in _UNIT_BALL_VOLUME:
        raise ValueError("data must have shape (n, d) with d in {1, 2, 3}; got %r" % (pts.shape,))
    return pts


def knn_density(points: FloatArray, k: int) -> FloatArray:
    """
    k-th nearest neighbour density, in points per nm^d: f_i = k / (V_d * r_k(i)^d).

    Origin: derivation (classical k-NN density estimate, Loftsgaarden & Quesenberry, 1965, Ann. Math. Stat. 36,
    1049-1051, without the 1/n factor so the value is a physical density). The k-th neighbour excludes the point
    itself. A zero distance (duplicated localisations) is clamped to the smallest positive k-th distance of the
    table so the density stays finite.
    """
    n, d = points.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    kk = int(min(max(k, 1), n - 1)) if n > 1 else 0
    if kk == 0:
        return np.ones(n, dtype=np.float64)
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=kk + 1)
    rk = np.asarray(dist[:, -1], dtype=np.float64)
    positive = rk[rk > 0]
    floor = float(positive.min()) if positive.size else 1.0
    rk = np.where(rk > 0, rk, floor)
    dens = np.asarray(float(kk) / (_UNIT_BALL_VOLUME[d] * rk ** d), dtype=np.float64)
    return dens


def kde_density(points: FloatArray, bandwidth: float, cutoff_sigmas: float = 3.0) -> FloatArray:
    """
    Gaussian kernel density, in points per nm^d, truncated at ``cutoff_sigmas`` bandwidths (derivation: the
    Gaussian kernel beyond 3 sigma carries < 1.2 % of its mass in 2D, so truncating keeps the neighbour search
    local). The point itself contributes its own kernel value at zero distance, as in a plain KDE.
    """
    n, d = points.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=cutoff_sigmas * bandwidth, output_type="ndarray")
    norm = (2.0 * np.pi) ** (-0.5 * d) * bandwidth ** (-d)
    dens = np.full(n, norm, dtype=np.float64)
    if pairs.shape[0]:
        i = pairs[:, 0]
        j = pairs[:, 1]
        diff = points[i] - points[j]
        w = norm * np.exp(-0.5 * np.einsum("ij,ij->i", diff, diff) / bandwidth ** 2)
        np.add.at(dens, i, w)
        np.add.at(dens, j, w)
    return dens


def neighbourhood_graph(points: FloatArray, k: Optional[int], radius: Optional[float]) -> Tuple[IntArray, IntArray]:
    """
    Undirected neighbourhood graph as two arrays of endpoints (i < j, no duplicates, no self loops).

    Either a symmetrised k-NN graph (``k``) or a radius graph (``radius``); when both are given the k-NN graph is
    used and ``radius`` is ignored. Returns empty arrays when there is nothing to connect.
    """
    n = points.shape[0]
    empty = np.zeros(0, dtype=np.int64)
    if n < 2:
        return empty, empty
    tree = cKDTree(points)
    if k is not None:
        kk = int(min(max(k, 1), n - 1))
        _, idx = tree.query(points, k=kk + 1)
        src = np.repeat(np.arange(n, dtype=np.int64), kk)
        dst = np.asarray(idx[:, 1:], dtype=np.int64).ravel()
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        keep = lo != hi
        edges = np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)
        return edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    if radius is None or radius <= 0:
        raise ValueError("either k or a positive radius is required")
    pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    if pairs.shape[0] == 0:
        return empty, empty
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)


def _adjacency_lists(n: int, a: IntArray, b: IntArray) -> Tuple[IntArray, IntArray]:
    """CSR-like neighbour lists (indptr, indices) of an undirected edge list."""
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    counts = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, dst.astype(np.int64)


# --------------------------------------------------------------------------------------------------------------
# ToMATo
# --------------------------------------------------------------------------------------------------------------

TOMATO_RADIUS_NM = 20.0
"""Literature: Pike et al. (2020) build ToMATo's graph with a search radius and used 20 nm on integrin dSTORM data
(their Fig 4); ~2 x the NeNA precision of this project (10-12 nm, Gazal et al., 2026). Simulation (project): verification
run D (2026-10-16), radius 20 + KDE sigma 10 is "preferable" in both regimes of the synthetic benchmark; radius 15 already
loses molecules whole (exact 0.63); radius 25 is preferable in the sparse regime and fails the medium one only on time."""

TOMATO_BANDWIDTH_NM = 10.0
"""Derivation: Pike et al. (2020) count the detections within the search radius; a Gaussian kernel of sigma = radius / 2
puts ~86 % of its 2D mass inside the radius (1 - exp(-2)), the same kernel the GraphCommunity edge weights use."""

TOMATO_K_DENSITY = 10
"""Ad hoc (taken from the k = 10-20 range of StormGraph's k-NN histograms, Scurll et al., 2019, preprint; not a ToMATo
value): only used with ``density='knn'``."""

TOMATO_MIN_CLUSTER_SIZE = 3
"""Ad hoc (taken from StormGraph's MinCluSize >= 3, Scurll et al., 2019): with the radius graph it is what turns
unlinked points into noise. Run E (2026-10-16): 10 gives more one-to-one molecules but loses more (missed 0.11 vs 0.04)."""

def _compact_labels(labels: IntArray) -> IntArray:
    """Renumber the non-negative labels 0..K-1 in order of first appearance; -1 stays -1."""
    out: IntArray = labels.copy()
    valid = out >= 0
    if valid.any():
        _, first_idx, inv = np.unique(out[valid], return_index=True, return_inverse=True)
        order = np.argsort(first_idx)
        rank = np.empty(order.size, dtype=np.int64)
        rank[order] = np.arange(order.size)
        out[valid] = rank[inv.ravel()]
    return out


def tomato_core(
    density: FloatArray, indptr: IntArray, indices: IntArray, tau: float
) -> Tuple[IntArray, FloatArray, FloatArray, IntArray]:
    """
    ToMATo on a graph with a given density (Chazal et al., 2013, Algorithm 1).

    Points are visited by decreasing density. A point with no denser neighbour starts a peak; otherwise it joins
    the peak of its densest neighbour. For every other denser neighbour whose peak differs, the lower of the two
    peaks is merged into the higher one when its prominence ``birth - f_i`` is below ``tau`` (its death is recorded
    at ``f_i``); with ``tau = inf`` every meeting is a merge and the output is the full persistence diagram.

    Returns ``(root, birth, death, peak_index)``: the peak index of every point (index of the peak's own point),
    and for each peak created its birth density, death density (nan while alive) and point index.
    """
    n = density.shape[0]
    order = np.argsort(-density, kind="stable")
    parent = np.arange(n, dtype=np.int64)
    birth_of = np.full(n, np.nan, dtype=np.float64)
    death_of = np.full(n, np.nan, dtype=np.float64)
    visited = np.zeros(n, dtype=bool)
    peaks: List[int] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            nxt = int(parent[x])
            parent[x] = root
            x = nxt
        return root

    dens_l = density.tolist()
    indptr_l = indptr.tolist()
    indices_l = indices.tolist()
    for i in order.tolist():
        f_i = dens_l[i]
        best = -1
        best_f = -np.inf
        higher: List[int] = []
        for j in indices_l[indptr_l[i]:indptr_l[i + 1]]:
            if visited[j]:
                higher.append(j)
                if dens_l[j] > best_f:
                    best_f = dens_l[j]
                    best = j
        visited[i] = True
        if best < 0:
            birth_of[i] = f_i
            peaks.append(i)
            continue
        parent[i] = find(best)
        r_i = int(parent[i])
        for j in higher:
            r_j = find(j)
            if r_j == r_i:
                continue
            # The lower peak is the candidate for merging into the higher one.
            if birth_of[r_j] < birth_of[r_i]:
                lo, hi = r_j, r_i
            else:
                lo, hi = r_i, r_j
            if birth_of[lo] - f_i < tau:
                parent[lo] = hi
                death_of[lo] = f_i
                r_i = hi
    root = np.array([find(i) for i in range(n)], dtype=np.int64)
    peak_index = np.array(peaks, dtype=np.int64)
    return root, birth_of[peak_index], death_of[peak_index], peak_index


def suggest_tau(persistences: npt.ArrayLike, min_peaks: int = 1, min_ratio: float = 2.0) -> float:
    """
    Persistence threshold at the largest RELATIVE gap of the sorted persistences (ad hoc).

    Sorted in decreasing order, the diagram of a clustered table shows a few large prominences (the clusters) and
    a tail of small ones (noise and sampling wiggles). The split is placed where consecutive values differ by the
    largest ratio p[i] / p[i+1], and tau is their geometric mean; a ratio is read in log-space because the cluster
    prominences themselves spread over a factor of a few (clouds of 15-30 points) while the noise tail is 10-100x
    below. If no ratio reaches ``min_ratio`` the diagram is read as "no tail" and every peak is kept
    (tau = half the smallest persistence). ``min_peaks`` guards the search so at least that many peaks stay above
    tau. With no positive finite value the function returns 0.0.
    Origin: ad hoc (largest gap, min_ratio = 2); Pike et al. (2020) choose tau by eye on the persistence diagram,
    and this rule is only a starting point that the user must check on the diagram.
    """
    p = np.asarray(persistences, dtype=np.float64)
    p = np.sort(p[np.isfinite(p) & (p > 0)])[::-1]
    if p.size == 0:
        return 0.0
    if p.size == 1:
        return float(0.5 * p[0])
    ratios = p[:-1] / p[1:]
    start = max(int(min_peaks) - 1, 0)
    if start >= ratios.size:
        return float(0.5 * p[-1])
    g = int(np.argmax(ratios[start:])) + start
    if ratios[g] < min_ratio:
        return float(0.5 * p[-1])
    return float(np.sqrt(p[g] * p[g + 1]))


class ToMAToStrategy(ClusteringStrategy):
    """
    ToMATo clustering (Chazal et al., 2013; SMLM use: Pike et al., 2020).

    Parameters
    ----------
    k_density : int
        Neighbour rank of the k-NN density estimate, only with ``density='knn'``. Default 10 [ad hoc: taken from
        StormGraph's k-NN range, Scurll et al., 2019; ToMATo's own scale is a radius, Pike et al., 2020].
    k_graph : int or None
        Neighbours of a symmetrised k-NN graph. Default None: radius graph [literature: Pike et al., 2020]. A k-NN
        graph gives every point neighbours, so no point is ever noise (verification run B: missed 0, split 0.22-0.44).
    radius_nm : float or None
        Radius of the neighbourhood graph when ``k_graph`` is None. Default 20 nm [literature: Pike et al., 2020, 20 nm
        on dSTORM; simulation (project): run D]. Also the KDE bandwidth source when ``bandwidth_nm`` is None.
    tau : float or None
        Persistence threshold in density units (points per nm^d). None: chosen by ``suggest_tau`` from the full
        persistence diagram of the table [ad hoc; Pike et al., 2020 choose it by eye on the diagram].
    density : {'knn', 'kde'}
        Density estimator. 'kde' (default): truncated Gaussian kernel of ``bandwidth_nm`` [derivation: Pike et al.
        count detections within the radius]; 'knn': k-th neighbour.
    bandwidth_nm : float or None
        Kernel bandwidth for ``density='kde'``. Default 10 nm = radius / 2 [derivation, see TOMATO_BANDWIDTH_NM].
        None -> radius / 2, or 10 nm without a radius.
    density_floor : float or None
        Points with density below this value (points per nm^d) are labelled -1 before the ascent. Default None and
        it should stay so: the floor is NOT part of ToMATo (Pike et al., 2020 obtain noise from points that cannot
        be linked) and in verification run D a floor of min_samples / (pi eps^2) lost 17 % of the molecules [ad hoc,
        optional, kept for exploration only].
    min_cluster_size : int
        Clusters with fewer points are relabelled -1. Default 3 [ad hoc: taken from StormGraph's MinCluSize, Scurll
        et al., 2019].
    """

    def __init__(
        self,
        k_density: int = TOMATO_K_DENSITY,
        k_graph: Optional[int] = None,
        radius_nm: Optional[float] = TOMATO_RADIUS_NM,
        tau: Optional[float] = None,
        density: str = "kde",
        bandwidth_nm: Optional[float] = TOMATO_BANDWIDTH_NM,
        density_floor: Optional[float] = None,
        min_cluster_size: int = TOMATO_MIN_CLUSTER_SIZE,
    ) -> None:
        if density not in ("knn", "kde"):
            raise ValueError("density must be 'knn' or 'kde'")
        if k_graph is None and (radius_nm is None or radius_nm <= 0):
            raise ValueError("k_graph=None requires a positive radius_nm")
        self.k_density = int(k_density)
        self.k_graph = None if k_graph is None else int(k_graph)
        self.radius_nm = None if radius_nm is None else float(radius_nm)
        self.tau = None if tau is None else float(tau)
        self.density = density
        self.bandwidth_nm = None if bandwidth_nm is None else float(bandwidth_nm)
        self.density_floor = None if density_floor is None else float(density_floor)
        self.min_cluster_size = int(min_cluster_size)
        # Diagnostics of the last fit.
        self.density_values_: Optional[FloatArray] = None
        self.persistence_pairs_: Optional[FloatArray] = None   # (n_peaks, 2): birth, death
        self.peak_index_: Optional[IntArray] = None
        self.immortal_: Optional[npt.NDArray[np.bool_]] = None
        self.tau_used_: Optional[float] = None
        self.n_clusters_: int = 0
        self.fit_seconds_: float = 0.0
        self._scale: Optional[float] = None

    # -- interface --------------------------------------------------------------------------------------------

    def scale_nm(self) -> Optional[float]:
        """The graph radius of a radius graph [literature: Pike et al. 2020] -- the distance below
        which two points can be linked, i.e. the separation scale; the kernel bandwidth for a
        k-NN graph with a KDE density; otherwise the median k_density-th neighbour distance of
        the clustered points [derivation], the scale of the k-NN density."""
        if self.radius_nm is not None and self.k_graph is None:
            return float(self.radius_nm)
        if self.density == "kde" and self.bandwidth_nm is not None:
            return float(self.bandwidth_nm)
        return self._scale

    def get_strategy_name(self) -> str:
        return "ToMATo"

    def get_params(self) -> Dict[str, Any]:
        return {
            "k_density": self.k_density,
            "k_graph": self.k_graph,
            "radius_nm": self.radius_nm,
            "tau": self.tau,
            "density": self.density,
            "bandwidth_nm": self.bandwidth_nm,
            "density_floor": self.density_floor,
            "min_cluster_size": self.min_cluster_size,
        }

    def estimate_density(self, points: FloatArray) -> FloatArray:
        if self.density == "knn":
            return knn_density(points, self.k_density)
        if self.bandwidth_nm is not None:
            bw = self.bandwidth_nm
        else:
            bw = 0.5 * self.radius_nm if self.radius_nm is not None else TOMATO_BANDWIDTH_NM
        return kde_density(points, bw)

    def fit(self, data: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        pts = _as_points(data)
        n = pts.shape[0]
        labels = np.full(n, -1, dtype=np.int64)
        if n == 0:
            self.density_values_ = np.zeros(0)
            self.persistence_pairs_ = np.zeros((0, 2))
            self.peak_index_ = np.zeros(0, dtype=np.int64)
            self.tau_used_ = self.tau if self.tau is not None else 0.0
            self.n_clusters_ = 0
            self.fit_seconds_ = time.perf_counter() - t0
            return labels
        dens = self.estimate_density(pts)
        self.density_values_ = dens
        keep = np.ones(n, dtype=bool) if self.density_floor is None else dens >= self.density_floor
        sub = np.flatnonzero(keep)
        if sub.size == 0:
            self.persistence_pairs_ = np.zeros((0, 2))
            self.peak_index_ = np.zeros(0, dtype=np.int64)
            self.tau_used_ = self.tau if self.tau is not None else 0.0
            self.n_clusters_ = 0
            self.fit_seconds_ = time.perf_counter() - t0
            return labels
        sub_pts = pts[sub]
        sub_dens = dens[sub]
        a, b = neighbourhood_graph(sub_pts, self.k_graph, self.radius_nm)
        indptr, indices = _adjacency_lists(sub.size, a, b)
        # Full diagram (tau = inf) for the diagnostic and, if needed, for the choice of tau.
        _, birth, death, peak_index = tomato_core(sub_dens, indptr, indices, np.inf)
        # A peak that never meets a higher one (one per connected component of the graph) is "immortal": its
        # death is set at the lowest density of the table (the density floor when given) so its prominence is
        # finite and takes part in the choice of tau (derivation: that is the level at which the whole table is
        # one component of the sublevel filtration). ``immortal_`` keeps which ones they were.
        immortal = np.isnan(death)
        base = float(self.density_floor) if self.density_floor is not None else float(sub_dens.min())
        death = np.where(immortal, base, death)
        self.persistence_pairs_ = np.stack([birth, death], axis=1)
        self.immortal_ = immortal
        self.peak_index_ = sub[peak_index]
        if self.tau is None:
            tau = suggest_tau(birth - death)
        else:
            tau = self.tau
        self.tau_used_ = float(tau)
        root, _, _, _ = tomato_core(sub_dens, indptr, indices, tau)
        _, inv, counts = np.unique(root, return_inverse=True, return_counts=True)
        sub_labels = inv.astype(np.int64)
        small = counts[inv] < self.min_cluster_size
        sub_labels[small] = -1
        # Renumber 0..K-1 in order of appearance so labels are compact and deterministic.
        sub_labels = _compact_labels(sub_labels)
        labels[sub] = sub_labels
        self._scale = core_distance_scale(pts, labels, self.k_density)
        self.n_clusters_ = int(labels.max() + 1) if (labels >= 0).any() else 0
        self.fit_seconds_ = time.perf_counter() - t0
        return labels

    def persistences(self) -> FloatArray:
        """Prominences (birth - death) of the last fit's full diagram, for the user's choice of tau."""
        if self.persistence_pairs_ is None:
            return np.zeros(0)
        p = self.persistence_pairs_[:, 0] - self.persistence_pairs_[:, 1]
        return np.asarray(p[np.isfinite(p)], dtype=np.float64)


# --------------------------------------------------------------------------------------------------------------
# Graph community (StormGraph-inspired substitute)
# --------------------------------------------------------------------------------------------------------------

GRAPH_RADIUS_NM = 25.0
"""Ad hoc: the compatibility eps of this program's DBSCAN (25 nm, Gazal et al., 2026); Pike et al. (2020) used 20 nm
for ToMATo and Nieves et al. (2023) swept 5-100 nm. Simulation (project): verification run B (2026-10-16), the fixed
radius is "preferable" in the sparse regime of the synthetic benchmark while the plateau scan merged clusters
(merge 0.24-0.74). The scan stays available through ``radius_nm=None`` / ``scan_scales``."""


def weighted_degree(points: FloatArray, a: IntArray, b: IntArray, radius: float) -> FloatArray:
    """
    Sum of Gaussian edge weights w = exp(-d^2 / (2 (radius/2)^2)) at every node (derivation: a kernel of sigma =
    radius/2 makes an edge at the graph radius weigh e^-2 ~ 0.14, so the weight is near zero where the graph is
    cut; the node degree is then a kernel density up to a constant). Isolated nodes get 0.
    """
    n = points.shape[0]
    deg = np.zeros(n, dtype=np.float64)
    if a.size == 0:
        return deg
    diff = points[a] - points[b]
    w = np.exp(-0.5 * np.einsum("ij,ij->i", diff, diff) / (0.5 * radius) ** 2)
    np.add.at(deg, a, w)
    np.add.at(deg, b, w)
    return deg


def _components(n: int, a: IntArray, b: IntArray, keep: npt.NDArray[np.bool_], min_size: int) -> IntArray:
    """Connected components of the graph restricted to kept nodes; components under ``min_size`` -> -1."""
    labels = np.full(n, -1, dtype=np.int64)
    sub = np.flatnonzero(keep)
    if sub.size == 0:
        return labels
    m = sub.size
    pos = np.full(n, -1, dtype=np.int64)
    pos[sub] = np.arange(m)
    if a.size:
        ok = keep[a] & keep[b]
        row = pos[a[ok]]
        col = pos[b[ok]]
    else:
        row = np.zeros(0, dtype=np.int64)
        col = np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(row.size, dtype=np.float64), (row, col)), shape=(m, m)).tocsr()
    _, comp = connected_components(graph, directed=False)
    comp = np.asarray(comp, dtype=np.int64)
    counts = np.bincount(comp)
    comp_labels = _compact_labels(np.where(counts[comp] >= min_size, comp, -1).astype(np.int64))
    labels[sub] = comp_labels
    return labels


class GraphCommunityStrategy(ClusteringStrategy):
    """
    Multi-scale neighbourhood-graph clustering INSPIRED BY StormGraph (Scurll et al., 2019, bioRxiv 515627).

    THIS IS NOT THE PUBLISHED STORMGRAPH. StormGraph (a) weights edges by Monte Carlo over each localisation's
    uncertainty and prunes edges present in < 75 % of simulations, (b) finds a multi-level cluster hierarchy with
    Infomap (Louvain only in a preliminary pass), and (c) has a "molecule counted several times" filter. None of
    those are implemented here: there is no community-detection library in the environment (no networkx, no
    igraph) and the project has not decided to add one. What is kept, with no new dependency:

    - a radius graph (edges between localisations closer than ``radius_nm``; StormGraph's r0 graph), with
      Gaussian edge weights whose sum at a node is its weighted degree (a kernel density up to a constant);
    - pruning of low-density nodes: StormGraph thresholds the node degree at the (1 - alpha) percentile of random
      clouds, alpha being "the maximum probability that a completely random localisation is classified as
      clustered" (typical 0.05). Here the random cloud has the same number of points, uniform over the region the
      data occupy (``null_region='occupied'``, ``OccupiedSupport``: a dilated 100-nm occupancy grid in x, y;
      z, if any, uniform over its range) [derivation: the null lives where the data could be; REVIEW_D0 M3 -- the
      bounding box of a ring includes the empty axoplasm and made the threshold too low to prune], ``n_random``
      draws averaged (seeded). ``null_region='bbox'`` restores the bounding box. ``prune_quantile`` instead removes
      the lowest-degree fraction of the data itself (ad hoc alternative, no simulation);
    - connected components of the pruned graph (scipy.sparse.csgraph) in place of community detection, components
      under ``min_cluster_size`` discarded;
    - a scale scan: ``scan_scales`` evaluates several radii and proposes the one on the longest plateau of the
      number of clusters (StormGraph estimates r0 at the elbow of connected components versus epsilon; the plateau
      criterion here is ad hoc and, on the synthetic benchmark, chose radii that merged clusters). ``fit`` uses
      ``radius_nm`` (default 25 nm) and runs the scan only when it is None.

    Replace this class by the real algorithm, or import StormGraph labels, before quoting StormGraph results.

    Parameters
    ----------
    radius_nm : float or None
        Graph radius. Default 25 nm [ad hoc, see GRAPH_RADIUS_NM]. None: chosen by ``scan_scales`` over ``radii_nm``.
    radii_nm : sequence of float
        Candidate radii for the scan. Default 10-60 nm in steps of 5 (ad hoc: from the localisation precision up
        to the ~50-nm gap that separates ToMATo/DBSCAN clusters in the benchmark sweeps of Nieves et al., 2023, who
        swept 5-100 nm; to verify on the data).
    alpha : float
        Random-cloud level of the degree threshold. Default 0.05 [literature, preprint: Scurll et al., 2019, typical
        value as quoted by the project's research note; not verified in the full text].
    n_random : int
        Random clouds averaged for the threshold. Default 5 [ad hoc].
    prune_quantile : float or None
        If given, replaces the random-cloud threshold by the quantile of the data's own degrees [ad hoc].
    min_cluster_size : int
        Default 3 [ad hoc: taken from StormGraph's MinCluSize, Scurll et al., 2019].
    null_region : {'occupied', 'bbox'}
        Where the random clouds are drawn: the occupied region of the data (default, derivation) or the bounding box.
    seed : int
        Seed of the random clouds, so a fit is reproducible.
    """

    def __init__(
        self,
        radius_nm: Optional[float] = GRAPH_RADIUS_NM,
        radii_nm: Sequence[float] = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0),
        alpha: float = 0.05,
        n_random: int = 5,
        prune_quantile: Optional[float] = None,
        min_cluster_size: int = 3,
        seed: int = 0,
        null_region: str = "occupied",
    ) -> None:
        if null_region not in ("occupied", "bbox"):
            raise ValueError("null_region must be 'occupied' or 'bbox'")
        self.null_region = null_region
        self.radius_nm = None if radius_nm is None else float(radius_nm)
        self.radii_nm: Tuple[float, ...] = tuple(float(r) for r in radii_nm)
        self.alpha = float(alpha)
        self.n_random = int(n_random)
        self.prune_quantile = None if prune_quantile is None else float(prune_quantile)
        self.min_cluster_size = int(min_cluster_size)
        self.seed = int(seed)
        # Diagnostics of the last fit / scan.
        self.degree_: Optional[FloatArray] = None
        self.degree_threshold_: Optional[float] = None
        self.radius_used_: Optional[float] = None
        self.scan_: Optional[Dict[str, Any]] = None
        self.n_clusters_: int = 0
        self.fit_seconds_: float = 0.0

    def scale_nm(self) -> Optional[float]:
        """The radius actually used by the last fit [ad hoc: given (default 25 nm), or chosen on the
        plateau of the scan]; ``None`` before ``fit``."""
        return None if self.radius_used_ is None else float(self.radius_used_)

    def get_strategy_name(self) -> str:
        return "GraphCommunity (StormGraph-inspired, connected components)"

    def get_params(self) -> Dict[str, Any]:
        return {
            "radius_nm": self.radius_nm,
            "radii_nm": list(self.radii_nm),
            "alpha": self.alpha,
            "n_random": self.n_random,
            "prune_quantile": self.prune_quantile,
            "min_cluster_size": self.min_cluster_size,
            "seed": self.seed,
            "null_region": self.null_region,
        }

    # -- pieces -------------------------------------------------------------------------------------------------

    def degree_threshold(self, points: FloatArray, radius: float, degree: FloatArray) -> float:
        """Node-degree threshold: (1 - alpha) quantile of random clouds, or the data's own ``prune_quantile``."""
        if self.prune_quantile is not None:
            return float(np.quantile(degree, self.prune_quantile)) if degree.size else 0.0
        n, d = points.shape
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        rng = np.random.default_rng(self.seed)
        support = OccupiedSupport(points[:, :2]) if (self.null_region == "occupied" and d >= 2) else None
        qs: List[float] = []
        for _ in range(max(self.n_random, 1)):
            if support is not None:
                xy = support.sample(n, rng)
                cloud = xy if d == 2 else np.column_stack([xy, lo[2:] + (hi[2:] - lo[2:]) * rng.random((n, d - 2))])
            else:
                cloud = lo + (hi - lo) * rng.random((n, d))
            ra, rb = neighbourhood_graph(cloud, None, radius)
            deg = weighted_degree(cloud, ra, rb, radius)
            qs.append(float(np.quantile(deg, 1.0 - self.alpha)))
        return float(np.mean(qs))

    def labels_at(self, points: FloatArray, radius: float) -> Tuple[IntArray, FloatArray, float]:
        """Labels, degrees and the degree threshold at one radius."""
        a, b = neighbourhood_graph(points, None, radius)
        deg = weighted_degree(points, a, b, radius)
        thr = self.degree_threshold(points, radius, deg)
        keep = deg > thr
        labels = _components(points.shape[0], a, b, keep, self.min_cluster_size)
        return labels, deg, thr

    def scan_scales(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Number of clusters at every candidate radius and the proposed radius.

        Stability (ad hoc): for each radius the relative change of the cluster count to its neighbours in the
        scan, s_i = |N_i - N_{i-1}| / max(N_i, 1) + |N_{i+1} - N_i| / max(N_i, 1); the proposed radius has the
        smallest s_i (ties: the smallest radius), among radii with at least one cluster. Returns a dict with
        ``radii``, ``n_clusters``, ``n_clustered`` (points not -1), ``stability`` and ``radius``.
        """
        pts = _as_points(data)
        radii = list(self.radii_nm)
        counts: List[int] = []
        clustered: List[int] = []
        for r in radii:
            labels, _, _ = self.labels_at(pts, r)
            counts.append(int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0)
            clustered.append(int(np.count_nonzero(labels >= 0)))
        n_arr = np.asarray(counts, dtype=np.float64)
        stab = np.zeros(len(radii), dtype=np.float64)
        for i in range(len(radii)):
            denom = max(n_arr[i], 1.0)
            if i > 0:
                stab[i] += abs(n_arr[i] - n_arr[i - 1]) / denom
            if i + 1 < len(radii):
                stab[i] += abs(n_arr[i + 1] - n_arr[i]) / denom
        candidates = [i for i in range(len(radii)) if counts[i] > 0]
        if candidates:
            best = min(candidates, key=lambda i: (stab[i], radii[i]))
            chosen = radii[best]
        else:
            chosen = radii[0] if radii else 0.0
        result: Dict[str, Any] = {
            "radii": radii,
            "n_clusters": counts,
            "n_clustered": clustered,
            "stability": stab.tolist(),
            "radius": float(chosen),
        }
        self.scan_ = result
        return result

    def fit(self, data: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        pts = _as_points(data)
        if pts.shape[0] == 0:
            self.degree_ = np.zeros(0)
            self.degree_threshold_ = 0.0
            self.radius_used_ = self.radius_nm
            self.n_clusters_ = 0
            self.fit_seconds_ = time.perf_counter() - t0
            return np.zeros(0, dtype=np.int64)
        radius = self.radius_nm if self.radius_nm is not None else float(self.scan_scales(pts)["radius"])
        labels, deg, thr = self.labels_at(pts, radius)
        self.degree_ = deg
        self.degree_threshold_ = thr
        self.radius_used_ = radius
        self.n_clusters_ = int(labels.max() + 1) if labels.max() >= 0 else 0
        self.fit_seconds_ = time.perf_counter() - t0
        return labels


# --------------------------------------------------------------------------------------------------------------
# Family registry
# --------------------------------------------------------------------------------------------------------------

def topological_family() -> Dict[str, Dict[str, Any]]:
    """
    The strategies of this family: name -> {"cls", "description", "defaults", "origins"}.

    ``defaults`` are the constructor defaults; ``origins`` states where each one comes from (literature with a
    short citation, derivation, simulation or ad hoc) so a settings panel can show it next to the value.
    """
    tomato_cls: Type[ClusteringStrategy] = ToMAToStrategy
    graph_cls: Type[ClusteringStrategy] = GraphCommunityStrategy
    return {
        "ToMATo": {
            "cls": tomato_cls,
            "description": (
                "Density peaks merged by persistence (Chazal et al., 2013; SMLM: Pike et al., 2020) on a 20-nm "
                "radius graph with a kernel density. Two free parameters: the radius and the persistence "
                "threshold tau, read on the persistence diagram."
            ),
            "defaults": ToMAToStrategy().get_params(),
            "origins": {
                "k_density": "ad hoc: taken from StormGraph's k-NN range (Scurll et al., 2019); only used with "
                             "density='knn'",
                "k_graph": "literature: None = radius graph (Pike et al., 2020); a k-NN graph never yields noise",
                "radius_nm": "literature: 20 nm search radius on dSTORM (Pike et al., 2020); simulation (project): "
                             "verification run D, 2026-10-16",
                "tau": "ad hoc when None (largest gap of the diagram); Pike et al. (2020) choose it by eye",
                "density": "derivation: Pike et al. (2020) count detections within the radius; Gaussian kernel",
                "bandwidth_nm": "derivation: radius / 2 (kernel with 86 % of its mass inside the radius)",
                "density_floor": "ad hoc, optional and not part of ToMATo (run D: a floor lost 17 % of molecules)",
                "min_cluster_size": "ad hoc: taken from StormGraph's MinCluSize >= 3 (Scurll et al., 2019)",
            },
        },
        "GraphCommunity": {
            "cls": graph_cls,
            "description": (
                "StormGraph-INSPIRED substitute (Scurll et al., 2019, preprint): radius graph with Gaussian weights, "
                "node-degree threshold from random clouds at level alpha, connected components instead of Infomap "
                "communities, fixed 25-nm radius (the plateau scan is optional). Not the published algorithm."
            ),
            "defaults": GraphCommunityStrategy().get_params(),
            "origins": {
                "radius_nm": "ad hoc: 25 nm = this program's DBSCAN eps; simulation (project): verification run B "
                             "(2026-10-16), the plateau scan (None) merged clusters",
                "radii_nm": "ad hoc (10-60 nm; Nieves et al., 2023 swept 5-100 nm); only used when radius_nm is None",
                "alpha": "literature, preprint: 0.05 typical (Scurll et al., 2019), not verified in the full text",
                "n_random": "ad hoc",
                "prune_quantile": "ad hoc (alternative to alpha)",
                "min_cluster_size": "ad hoc: taken from StormGraph's MinCluSize >= 3 (Scurll et al., 2019)",
                "seed": "ad hoc (reproducibility)",
                "null_region": "derivation: random clouds over the region the data occupy, not the bounding box "
                               "(REVIEW_D0 M3)",
            },
        },
    }


__all__ = [
    "ToMAToStrategy",
    "GraphCommunityStrategy",
    "topological_family",
    "suggest_tau",
    "tomato_core",
    "knn_density",
    "kde_density",
    "neighbourhood_graph",
    "weighted_degree",
]
