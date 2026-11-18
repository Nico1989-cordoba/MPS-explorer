#!/usr/bin/env python3
"""
Density family of clustering strategies: OPTICS, FOCAL and scikit-learn's
HDBSCAN.

Every class here follows the interface of ``tools.clustering_strategies``
(``ClusteringStrategy.fit(data) -> labels``, ``get_strategy_name()``,
``get_params()``): ``fit`` takes an ``(n, 2)`` or ``(n, 3)`` array of
localization coordinates in nanometres and returns one integer label per
row, ``-1`` for noise. ``tools/clustering_strategies.py`` itself is not
touched; ``density_family()`` is the registry a later integration reads to
offer these methods next to DBSCAN and HDBSCAN in the interface.

Why these three
---------------
The research "Qué es un clúster en SMLM" (2026-10-15, §5.2) groups the
SMLM clustering literature into five families; this module completes the
*density* family, the one DBSCAN (the project's current method) belongs
to. OPTICS (Ankerst, Breunig, Kriegel & Sander, 1999) and HDBSCAN
(Campello, Moulavi & Sander, 2013; McInnes, Healy & Astels, 2017)
relax DBSCAN's single global density with a hierarchy; FOCAL (Mazouchi &
Milstein, 2016) replaces the neighbourhood search with a grid, which
makes it O(n) and ties its one length scale to the localization
precision (Nino, Djayakarsana & Milstein, 2020). None of the four models
multiple blinking: they all operate on localizations (Nieves et al.,
2023), so a fluorophore that blinks many times is indistinguishable from a
biological cluster unless events are linked beforehand.

Parameter origins are labelled in every docstring with one of:
``literature`` (with a short APA citation), ``derivation``, ``simulation``
(the project's own simulations, with the script named) or ``ad hoc``.

Dependencies: numpy, scipy and scikit-learn only (all already required by
the project). The ``hdbscan`` package is *not* needed here.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from sklearn.cluster import OPTICS as _SkOPTICS

from tools.clustering_strategies import ClusteringStrategy, OccupiedSupport, core_distance_max, core_distance_scale

try:
    from sklearn.cluster import HDBSCAN as _SkHDBSCAN
    SKLEARN_HDBSCAN_AVAILABLE = True
except ImportError:  # scikit-learn < 1.3
    _SkHDBSCAN = None
    SKLEARN_HDBSCAN_AVAILABLE = False


def _as_xy(data: np.ndarray) -> np.ndarray:
    """Validate the input and return it as a C-contiguous float64 array.

    Accepts ``(n, 2)`` or ``(n, 3)`` arrays (x, y[, z] in nanometres). An
    empty input is allowed and yields an empty label array downstream.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(
            f"expected an (n, 2) or (n, 3) array of coordinates, got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("coordinates must be finite (no NaN/inf)")
    return np.ascontiguousarray(arr)


_EDGE_STRUCTURE = ndimage.generate_binary_structure(2, 1)
"""4-connectivity (cells sharing an edge) [literature: FOCAL3D border cells 'share an edge or face' with a
core cell, Nino et al. 2020, Fig 1C; also scipy.ndimage.label's default]."""


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Return labels renumbered 0..k-1 in order of first appearance; -1 kept."""
    out = np.full(labels.shape[0], -1, dtype=np.int64)
    keep = labels >= 0
    if keep.any():
        _, inv = np.unique(labels[keep], return_inverse=True)
        out[keep] = inv
    return out


def _count_clusters(labels: np.ndarray) -> Tuple[int, int]:
    """(number of clusters, number of noise points) of a label array."""
    n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))
    return n_clusters, n_noise


# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------

class OPTICSStrategy(ClusteringStrategy):
    """
    OPTICS clustering strategy (Ankerst et al., 1999) on
    ``sklearn.cluster.OPTICS``.

    OPTICS orders the localizations so that spatially close points are
    consecutive and records, for each, the *reachability distance*: the
    smallest neighbourhood radius at which it would be reached from a core
    point. The ordering plus the reachability (the "reachability plot") is a
    DBSCAN result for every eps at once; clusters are then extracted either
    with the ξ steepness criterion of the paper (``cluster_method='xi'``) or
    as a DBSCAN cut at one eps (``cluster_method='dbscan'``).

    Parameters
    ----------
    min_samples : int
        Points in a neighbourhood for a point to be core (same meaning as in
        DBSCAN; Ankerst et al., 1999, §3.1). Default 10 — ad hoc: mirrors the
        ``min_samples = 10`` of the project's DBSCAN (Gazal et al., 2026
        pipeline), so that the two methods agree on what a core point is.
    xi : float
        Minimum relative steepness of the reachability plot that opens or
        closes a cluster (Ankerst et al., 1999, §4.2, "ξ-steep areas").
        Default 0.1 — simulation (project): verification run F (2026-10-16),
        on the synthetic benchmark xi 0.1 gives exact 0.60 / 0.51 (sparse /
        medium) against 0.37 / 0.29 with scikit-learn's default 0.05; the
        value is otherwise ad hoc (Ankerst et al., 1999, define xi but give
        no value).
    max_eps : float
        Largest neighbourhood radius considered, in nm. Default 100 —
        simulation (project): the research simulation ``sim1_strategies.py``
        (2026-10-15) used 100 nm (4 x the compatibility eps of 25 nm); a
        finite value is editable in the panel (``inf`` was not, REVIEW_D0
        M5) and makes the reachability plot comparable across axons.
        Reachabilities above it are ``inf``. It does NOT bound the running
        time: simulation (project), verification run F (2026-10-16): with
        max_eps 100 or 50 nm OPTICS still takes 175-204 x DBSCAN's time (the
        cost is scikit-learn's per-point loop). ``math.inf`` restores the
        complete ordering.

    No default of this class reached "recommendable" in the synthetic
    benchmark (verification run F: xi 0.1 raises exact to 0.60 / 0.51 but
    leaves 30-39 % of the molecules missed; min_samples 5 destroys it):
    OPTICS is offered as a diagnostic (the reachability plot), not as a
    replacement for DBSCAN.
    cluster_method : {'xi', 'dbscan'}
        How clusters are read off the ordering (see above). Default 'xi'
        (the method of the paper).
    eps : float, optional
        Cut radius for ``cluster_method='dbscan'``, in nm. ``None`` (default)
        uses ``max_eps``, as scikit-learn does. Ignored by 'xi'.
    metric : str
        Distance metric (default 'euclidean').
    logger : logging.Logger, optional

    Attributes (after ``fit``)
    --------------------------
    reachability_, ordering_, core_distances_, predecessor_ : np.ndarray
        Copies of scikit-learn's arrays, for the reachability plot and
        diagnostics. ``reachability_plot()`` returns them in plot order.

    References
    ----------
    Ankerst, M., Breunig, M. M., Kriegel, H.-P., & Sander, J. (1999). OPTICS:
    Ordering points to identify the clustering structure. SIGMOD Record,
    28(2), 49-60. https://doi.org/10.1145/304181.304187
    """

    def __init__(
        self,
        min_samples: int = 10,
        xi: float = 0.1,
        max_eps: float = 100.0,
        cluster_method: str = "xi",
        eps: Optional[float] = None,
        metric: str = "euclidean",
        logger: Optional[logging.Logger] = None,
    ):
        if cluster_method not in ("xi", "dbscan"):
            raise ValueError("cluster_method must be 'xi' or 'dbscan'")
        if int(min_samples) < 2:
            raise ValueError("min_samples must be >= 2")
        if not (0.0 <= float(xi) < 1.0):
            raise ValueError("xi must be in [0, 1)")
        if float(max_eps) <= 0.0:
            raise ValueError("max_eps must be > 0")
        if eps is not None and float(eps) <= 0.0:
            raise ValueError("eps must be > 0 when given")
        self.min_samples = int(min_samples)
        self.xi = float(xi)
        self.max_eps = float(max_eps)
        self.cluster_method = cluster_method
        self.eps = None if eps is None else float(eps)
        self.metric = metric
        self.logger = logger
        self._scale: Optional[float] = None
        self.model: Any = None
        self.reachability_: Optional[np.ndarray] = None
        self.ordering_: Optional[np.ndarray] = None
        self.core_distances_: Optional[np.ndarray] = None
        self.predecessor_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """Run OPTICS and return one label per row (-1 = noise)."""
        xy = _as_xy(data)
        n = xy.shape[0]
        if n < self.min_samples:
            # scikit-learn raises for n < min_samples; too few points to
            # hold one core point is simply "all noise".
            self._clear_diagnostics(n)
            return np.full(n, -1, dtype=np.int64)
        self.model = _SkOPTICS(
            min_samples=self.min_samples,
            xi=self.xi,
            max_eps=self.max_eps,
            cluster_method=self.cluster_method,
            eps=self.eps,
            metric=self.metric,
        )
        with warnings.catch_warnings():
            # Pure noise within max_eps: scikit-learn warns that every
            # reachability is inf; that is the honest answer ("no cluster"),
            # not a misconfiguration, so the notice is not repeated per axon.
            warnings.filterwarnings("ignore", message="All reachability values are inf", category=UserWarning)
            self.model.fit(xy)
        # Explicit annotations: sklearn attributes are Any under
        # --ignore-missing-imports, and the module must not return Any.
        labels: np.ndarray = np.asarray(self.model.labels_, dtype=np.int64)
        self.reachability_ = np.asarray(self.model.reachability_, dtype=np.float64)
        self.ordering_ = np.asarray(self.model.ordering_, dtype=np.int64)
        self.core_distances_ = np.asarray(self.model.core_distances_, dtype=np.float64)
        self.predecessor_ = np.asarray(self.model.predecessor_, dtype=np.int64)
        labels = _relabel_consecutive(labels)
        inside = labels >= 0
        self._scale = (float(np.median(self.core_distances_[inside]))
                       if int(inside.sum()) else None)
        finite = self.core_distances_[inside]
        finite = finite[np.isfinite(finite)]
        self._max_core = float(finite.max()) if finite.size else None
        if self.logger:
            k, noise = _count_clusters(labels)
            self.logger.debug("OPTICS(%s): %d clusters, %d noise points", self.cluster_method, k, noise)
        return labels

    def _clear_diagnostics(self, n: int) -> None:
        self.model = None
        self.reachability_ = np.full(n, np.inf)
        self.ordering_ = np.arange(n, dtype=np.int64)
        self.core_distances_ = np.full(n, np.inf)
        self.predecessor_ = np.full(n, -1, dtype=np.int64)

    def reachability_plot(self) -> Tuple[np.ndarray, np.ndarray]:
        """(ordering, reachability in that order) for plotting; call after ``fit``.

        Infinite reachabilities (points never reached within ``max_eps``)
        are left as ``inf``; a plot should clip them to its y-range.
        """
        if self.reachability_ is None or self.ordering_ is None:
            raise RuntimeError("call fit() before reachability_plot()")
        return self.ordering_.copy(), self.reachability_[self.ordering_].copy()

    def get_strategy_name(self) -> str:
        return "OPTICS"

    def scale_nm(self) -> Optional[float]:
        """eps of the DBSCAN cut when that is the extraction; otherwise the
        median core distance (``min_samples``-th neighbour) of the clustered
        points [derivation], the scale the xi extraction works at."""
        if self.cluster_method == "dbscan" and self.eps is not None:
            return float(self.eps)
        return self._scale

    def edge_margin_nm(self, compat_eps: float) -> float:
        """max(compat_eps, eps of the DBSCAN cut or the largest core distance of the clustered
        points) [derivation, see ``ClusteringStrategy.edge_margin_nm``]."""
        if self.cluster_method == "dbscan" and self.eps is not None:
            return max(float(compat_eps), float(self.eps))
        return max(float(compat_eps), float(getattr(self, "_max_core", None) or 0.0))

    def get_params(self) -> dict:
        """JSON-serialisable parameters (``inf`` is written as ``None``)."""
        return {
            "min_samples": self.min_samples,
            "xi": self.xi,
            "max_eps": None if math.isinf(self.max_eps) else self.max_eps,
            "cluster_method": self.cluster_method,
            "eps": self.eps,
            "metric": self.metric,
        }


# ---------------------------------------------------------------------------
# FOCAL
# ---------------------------------------------------------------------------

class FOCALStrategy(ClusteringStrategy):
    """
    FOCAL clustering strategy (Mazouchi & Milstein, 2016) as FOCAL3D
    describes it (Nino, Djayakarsana & Milstein, 2020, Fig 1B-D), own 2-D
    implementation.

    Steps (Nino et al., 2020, Fig 1; the 2016 paper is their ref. [14]):

    1. Bin the localizations on a square grid of side ``grid_nm`` and count
       them per cell.
    2. *Enhanced density map* (Fig 1B): every non-empty cell is replaced by
       the sum of the counts of its 3x3 neighbourhood; empty cells stay 0.
    3. *Core cells* (Fig 1C): enhanced value >= ``min_l``. *Border cells*:
       non-empty cells below the threshold that share an edge with a core
       cell.
    4. *Candidates* (Fig 1D): connected components of core + border cells,
       4-connected (cells sharing an edge; ``scipy.ndimage.label``'s default
       structure, the connectivity the "share an edge" rule implies), kept
       when they have at least ``min_c`` cells.

    Every localization inside a kept cell (core or border) receives the
    cluster's label; all others are noise (-1). The cost is O(n) plus one
    pass over the grid image, with no neighbourhood search.

    Remaining differences from the paper (labelled):

    * ``out-of-focus filter`` not implemented: the project filters by ROI
      and by membrane anchoring elsewhere.
    * ``2-D only``: only x, y are binned; FOCAL3D bins voxels.
    * ``grid origin`` at the minimum x and y of the input; no half-cell
      offset scan.
    * The grid image costs (extent / grid_nm)^2 cells of int32: ~44 MB for a
      100-um field at 15 nm. Fine for one axon ROI; a whole field of view
      should be cropped first.

    Parameters
    ----------
    grid_nm : float
        Side of a grid cell, in nm. Default 15 — literature: Delta is chosen
        "based on the localization precision" (Nino et al., 2020), here
        ~1.25-1.5 x the 10-12 nm precision of the project's data (NeNA,
        Gazal et al., 2026); simulation (project): verification run E
        (2026-10-16, ``scratchpad/phaseD/verif/runE_refine``), the only grid
        "preferable" to DBSCAN 25/10 in both regimes of the benchmark.
    min_l : int
        Threshold on the *enhanced* (3x3-summed) map, the analogue of
        DBSCAN's minPts (Nino et al., 2020, Table 1). Default 6 — simulation
        (project): verification run E. FOCAL3D fixes it instead as the lowest
        value at which a uniform random copy of the data gives no cluster
        (Nino et al., 2020); ``suggest_min_l`` implements that recipe.
    min_c : int
        Minimum cells (core + border) for a cluster to be kept, the analogue
        of eps (Nino et al., 2020, Table 1). Default 3 — ad hoc; FOCAL3D
        chooses it "in the flat region" of the curve number of clusters vs
        minC (Nino et al., 2020).
    logger : logging.Logger, optional

    Attributes (after ``fit``)
    --------------------------
    cell_index_ : np.ndarray, shape (n, 2), int
        Grid cell (ix, iy) of every localization.
    n_core_cells_, n_border_cells_ : int
        Core and border cells before the ``min_c`` filter.

    References
    ----------
    Mazouchi, A., & Milstein, J. N. (2016). Fast Optimized Cluster Algorithm
    for Localizations (FOCAL): A spatial cluster analysis for super-resolved
    microscopy. Bioinformatics, 32(5), 747-754.
    https://doi.org/10.1093/bioinformatics/btv630
    Nino, D. F., Djayakarsana, D., & Milstein, J. N. (2020). FOCAL3D: A
    3-dimensional clustering package for single-molecule localization
    microscopy. PLOS Computational Biology, 16(12), e1008479.
    https://doi.org/10.1371/journal.pcbi.1008479
    Known limits: small clusters, fine grids, dense or overlapping clusters
    (Khater, Nabi & Hamarneh, 2020); sparse data (Nieves et al., 2023).
    """

    def __init__(
        self,
        grid_nm: float = 15.0,
        min_l: int = 6,
        min_c: int = 3,
        logger: Optional[logging.Logger] = None,
    ):
        if float(grid_nm) <= 0.0:
            raise ValueError("grid_nm must be > 0")
        if int(min_l) < 1:
            raise ValueError("min_l must be >= 1")
        if int(min_c) < 1:
            raise ValueError("min_c must be >= 1")
        self.grid_nm = float(grid_nm)
        self.min_l = int(min_l)
        self.min_c = int(min_c)
        self.logger = logger
        self.cell_index_: Optional[np.ndarray] = None
        self.n_core_cells_: int = 0
        self.n_border_cells_: int = 0

    def _grid(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cell index per point (n, 2), count image, enhanced 3x3 image); one empty cell of padding per side."""
        origin = xy.min(axis=0)
        cell = np.floor((xy - origin) / self.grid_nm).astype(np.int64) + 1
        shape = (int(cell[:, 0].max()) + 2, int(cell[:, 1].max()) + 2)
        img = np.zeros(shape, dtype=np.int32)
        np.add.at(img, (cell[:, 0], cell[:, 1]), 1)
        enhanced = ndimage.convolve(img, np.ones((3, 3), dtype=np.int32), mode="constant") * (img > 0)
        return cell, img, enhanced

    def _cell_labels(self, img: np.ndarray, enhanced: np.ndarray, min_l: int) -> Tuple[np.ndarray, int, int, int]:
        """(label image, -1 outside kept clusters; n clusters; n core cells; n border cells)."""
        core = enhanced >= min_l
        border = ndimage.binary_dilation(core, structure=_EDGE_STRUCTURE) & ~core & (img > 0)
        cand = core | border
        lab_img, n = ndimage.label(cand, structure=_EDGE_STRUCTURE)
        out = np.full(img.shape, -1, dtype=np.int64)
        if n == 0:
            return out, 0, int(core.sum()), int(border.sum())
        sizes = np.bincount(lab_img.ravel(), minlength=n + 1)
        sizes[0] = 0
        keep = np.flatnonzero(sizes >= self.min_c)
        remap = np.full(n + 1, -1, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        out = remap[lab_img]
        return out, int(keep.size), int(core.sum()), int(border.sum())

    def fit(self, data: np.ndarray) -> np.ndarray:
        """Run FOCAL and return one label per row (-1 = noise)."""
        xy = _as_xy(data)[:, :2]
        n = xy.shape[0]
        self.n_core_cells_ = 0
        self.n_border_cells_ = 0
        if n == 0:
            self.cell_index_ = np.zeros((0, 2), dtype=np.int64)
            return np.full(0, -1, dtype=np.int64)
        cell, img, enhanced = self._grid(xy)
        self.cell_index_ = cell - 1
        lab_img, _, n_core, n_border = self._cell_labels(img, enhanced, self.min_l)
        self.n_core_cells_ = n_core
        self.n_border_cells_ = n_border
        labels = _relabel_consecutive(lab_img[cell[:, 0], cell[:, 1]])
        if self.logger:
            k, noise = _count_clusters(labels)
            self.logger.debug(
                "FOCAL(grid=%.1f nm, min_l=%d, min_c=%d): %d core + %d border cells, %d clusters, %d noise points",
                self.grid_nm, self.min_l, self.min_c, n_core, n_border, k, noise,
            )
        return labels

    def suggest_min_l(
        self,
        data: np.ndarray,
        max_min_l: int = 100,
        n_random: int = 5,
        seed: int = 0,
        support: Optional[OccupiedSupport] = None,
    ) -> int:
        """Smallest ``min_l`` at which a randomised copy of ``data`` yields no
        cluster, with the current ``grid_nm`` and ``min_c``.

        FOCAL3D's recipe (Nino et al., 2020): minL* is the "lowest threshold
        value where FOCAL3D returns zero clusters for the randomly scattered
        data", with the same number of localizations in "the same volume as
        the experimental data". The random copies (``n_random`` draws; the
        value returned works for all of them) are uniform over the region the
        data occupy (:class:`OccupiedSupport`, or the ``support`` given, e.g.
        built from the drawn ROI), never over the bounding box, which on an
        axon ring includes the empty axoplasm and lowers minL* (REVIEW_D0
        M3). Returns ``max_min_l`` if no value up to it is clean.
        """
        xy = _as_xy(data)[:, :2]
        n = xy.shape[0]
        if n == 0:
            return 1
        rng = np.random.default_rng(seed)
        region = support if support is not None else OccupiedSupport(xy)
        maps = [self._grid(region.sample(n, rng))[1:] for _ in range(int(n_random))]
        for min_l in range(1, int(max_min_l) + 1):
            if all(self._cell_labels(img, enh, min_l)[1] == 0 for img, enh in maps):
                return min_l
        return int(max_min_l)

    def get_strategy_name(self) -> str:
        return "FOCAL"

    def scale_nm(self) -> Optional[float]:
        """The grid cell side [literature: FOCAL's only length scale, Mazouchi & Milstein 2016].

        Informative: the 3x3 enhanced map makes the effective search window
        3 x grid_nm wide; the edge-curation margin of the analysis
        is not this number (see ``ClusteringStrategy.edge_margin_nm``)."""
        return float(self.grid_nm)

    def get_params(self) -> dict:
        return {"grid_nm": self.grid_nm, "min_l": self.min_l, "min_c": self.min_c}


# ---------------------------------------------------------------------------
# HDBSCAN (scikit-learn)
# ---------------------------------------------------------------------------

class SklearnHDBSCANStrategy(ClusteringStrategy):
    """
    HDBSCAN clustering strategy on ``sklearn.cluster.HDBSCAN``
    (scikit-learn >= 1.3), an alternative to ``HDBSCANStrategy`` of
    ``tools.clustering_strategies`` when the ``hdbscan`` package is absent.

    HDBSCAN (Campello et al., 2013) builds the hierarchy of DBSCAN
    solutions over all eps and keeps the clusters that are most stable
    across it ("excess of mass"); the ``hdbscan`` package (McInnes et al.,
    2017) and scikit-learn's port implement the same algorithm, so results
    agree up to tie-breaking and the handling of infinite distances.

    Parameters
    ----------
    min_cluster_size : int
        Smallest group kept as a cluster (Campello et al., 2013, m_clSize).
        Default 10 — derivation: about a third of the localizations of one
        labelled molecule in the benchmark (28 / 3).
    min_samples : int, optional
        Neighbours that define the core distance (density smoothing).
        ``None`` sets it equal to ``min_cluster_size`` as both libraries do;
        the registry uses 5 — simulation (project): verification run A
        (2026-10-16): with 10, molecules of 28 localizations lose their
        periphery (missed 0.09-0.12); 5 improves exact, merge and missed in
        both regimes.
    metric : str
        Distance metric (default 'euclidean').
    cluster_selection_method : {'eom', 'leaf'}
        'eom' (default, the paper's stability criterion) or 'leaf' (finest
        clusters of the hierarchy; McInnes et al., 2017).
    allow_single_cluster : bool
        Whether the root may be returned as one cluster (default False, as
        in both implementations).
    logger : logging.Logger, optional

    References
    ----------
    Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based
    clustering based on hierarchical density estimates. In J. Pei, V. S.
    Tseng, L. Cao, H. Motoda, & G. Xu (Eds.), Advances in knowledge
    discovery and data mining: PAKDD 2013, Proceedings, Part II (Lecture
    Notes in Computer Science, Vol. 7819, pp. 160-172). Springer.
    https://doi.org/10.1007/978-3-642-37456-2_14
    McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical
    density based clustering. The Journal of Open Source Software, 2(11), 205.
    https://doi.org/10.21105/joss.00205
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        allow_single_cluster: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        if not SKLEARN_HDBSCAN_AVAILABLE:
            raise ImportError("sklearn.cluster.HDBSCAN needs scikit-learn >= 1.3")
        if int(min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be >= 2")
        if min_samples is not None and int(min_samples) < 1:
            raise ValueError("min_samples must be >= 1 when given")
        if cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = None if min_samples is None else int(min_samples)
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = bool(allow_single_cluster)
        self.logger = logger
        self._scale: Optional[float] = None
        self.model: Any = None
        self.probabilities_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """Run HDBSCAN and return one label per row (-1 = noise)."""
        xy = _as_xy(data)
        n = xy.shape[0]
        needed = max(self.min_cluster_size, self.min_samples or 0)
        if n <= needed:
            self.model = None
            self.probabilities_ = np.zeros(n)
            return np.full(n, -1, dtype=np.int64)
        self.model = _SkHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
        )
        with warnings.catch_warnings():
            # scikit-learn 1.8+ warns that the default of its `copy`
            # argument changes in 1.10; the argument does not exist before
            # 1.8, so it is not passed and the notice is silenced instead.
            warnings.filterwarnings("ignore", message=".*`copy`.*", category=FutureWarning)
            self.model.fit(xy)
        labels: np.ndarray = np.asarray(self.model.labels_, dtype=np.int64)
        # scikit-learn marks infinite-distance points -2 and missing data -3;
        # the project's convention is a single noise label.
        labels[labels < 0] = -1
        self.probabilities_ = np.asarray(self.model.probabilities_, dtype=np.float64)
        labels = _relabel_consecutive(labels)
        self._scale = core_distance_scale(xy, labels, int(self.min_samples or self.min_cluster_size))
        self._max_core = core_distance_max(xy, labels, int(self.min_samples or self.min_cluster_size))
        if self.logger:
            k, noise = _count_clusters(labels)
            self.logger.debug("HDBSCAN(sklearn): %d clusters, %d noise points", k, noise)
        return labels

    def get_strategy_name(self) -> str:
        return "HDBSCAN (scikit-learn)"

    def scale_nm(self) -> Optional[float]:
        """Median core distance of the clustered points [derivation]; see HDBSCANStrategy.scale_nm."""
        return self._scale

    def edge_margin_nm(self, compat_eps: float) -> float:
        """max(compat_eps, largest core distance of the clustered points) [derivation]; see
        HDBSCANStrategy.edge_margin_nm."""
        return max(float(compat_eps), float(getattr(self, "_max_core", None) or 0.0))

    def get_params(self) -> dict:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def density_family() -> Dict[str, Dict[str, Any]]:
    """Registry of the density-family strategies of this module.

    Returns ``{name: {"cls", "description", "citation", "defaults"}}`` where
    ``defaults`` maps every constructor parameter to ``{"value", "origin"}``
    (``origin`` is one of literature / derivation / simulation / ad hoc,
    with the source). The integration PR reads this to populate the
    strategy chooser of the interface; nothing here instantiates a model.
    """
    return {
        "OPTICS": {
            "cls": OPTICSStrategy,
            "description": (
                "Density ordering (reachability plot) with xi-steepness or a DBSCAN cut; "
                "no single eps, clusters of different density."
            ),
            "citation": "Ankerst, Breunig, Kriegel & Sander (1999), SIGMOD Record 28(2), 49-60",
            "available": True,
            "defaults": {
                "min_samples": {"value": 10, "origin": "ad hoc: project DBSCAN min_samples (Gazal et al., 2026)"},
                "xi": {"value": 0.1, "origin": "simulation: verification run F of the project (2026-10-16), exact 0.60 vs 0.37 with 0.05; diagnostic method, otherwise ad hoc (Ankerst et al. 1999 give no value)"},
                "max_eps": {"value": 100.0, "origin": "simulation: sim1_strategies.py of the project (2026-10-15) used 100 nm = 4 x eps; editable (inf was not, REVIEW_D0 M5); does NOT reduce the time (verification run F: 175-204 x DBSCAN)"},
                "cluster_method": {"value": "xi", "origin": "literature: Ankerst et al. (1999), xi method"},
                "eps": {"value": None, "origin": "ad hoc: only for cluster_method='dbscan'"},
            },
        },
        "FOCAL": {
            "cls": FOCALStrategy,
            "description": (
                "Grid density (Nino et al. 2020, Fig 1): 3x3-enhanced cell counts >= min_l are core, edge-adjacent "
                "non-empty cells are border, 4-connected components with >= min_c cells are clusters; O(n)."
            ),
            "citation": "Mazouchi & Milstein (2016), Bioinformatics 32(5), 747-754; Nino et al. (2020), PLoS Comput Biol 16(12)",
            "available": True,
            "defaults": {
                "grid_nm": {"value": 15.0, "origin": "literature: grid by the localization precision (Nino et al., 2020), ~1.25-1.5 x NeNA 10-12 nm; simulation: verification run E of the project, 2026-10-16"},
                "min_l": {"value": 6, "origin": "simulation: verification run E of the project, threshold on the 3x3-enhanced map (minL ~ minPts, Nino et al., 2020, Table 1); FOCAL3D's own recipe: suggest_min_l"},
                "min_c": {"value": 3, "origin": "ad hoc: FOCAL3D takes the flat region of clusters-vs-minC (Nino et al., 2020)"},
            },
        },
        "HDBSCAN (scikit-learn)": {
            "cls": SklearnHDBSCANStrategy,
            "description": (
                "Hierarchical DBSCAN over all eps, most stable clusters kept; scikit-learn port, "
                "used when the hdbscan package is missing."
            ),
            "citation": "Campello, Moulavi & Sander (2013), PAKDD; McInnes, Healy & Astels (2017), JOSS 2(11), 205",
            "available": SKLEARN_HDBSCAN_AVAILABLE,
            "defaults": {
                "min_cluster_size": {"value": 10, "origin": "derivation: about a third of the localizations of one labelled molecule in the benchmark (28 / 3, c_label 14, Dempsey et al. 2011)"},
                "min_samples": {"value": 5, "origin": "simulation: verification run A of the project (2026-10-16), = min_cluster_size / 2; better than 10 in both regimes"},
                "cluster_selection_method": {"value": "eom", "origin": "literature: stability criterion of Campello et al. (2013)"},
                "allow_single_cluster": {"value": False, "origin": "ad hoc: library default"},
            },
        },
    }
