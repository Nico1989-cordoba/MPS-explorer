#!/usr/bin/env python3
"""
Clustering Strategy Pattern for DBSCAN and HDBSCAN

Provides an abstract strategy interface for clustering algorithms,
allowing selection of DBSCAN or HDBSCAN based on dataset size.

Strategy Pattern Benefits:
- Easy to switch between algorithms
- Encapsulates algorithm complexity
- Enables automatic strategy selection
- Maintains clean interface
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from sklearn.cluster import DBSCAN
import logging

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit clustering model and return cluster labels.

        Parameters
        ----------
        data : np.ndarray
            Input data points (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Cluster labels where -1 indicates noise/unclustered points
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the clustering strategy."""
        pass

    def scale_nm(self) -> Optional[float]:
        """The method's own length scale in nm, read after ``fit`` (informative).

        Exported as ``cluster_scale_nm``. Each subclass documents where its
        value comes from; ``None`` means the method has no length scale of
        its own. It is NOT the edge-curation margin: for HDBSCAN / OPTICS /
        ToMATo-kNN it is a median core distance and for Voronoi a cell
        spacing, intra-cluster scales of 10-15 nm (VERIF_DEFAULTS_D0 section
        9) that would under-protect the edge compared with eps 25 nm. The
        margin is :meth:`edge_margin_nm`.
        """
        return None

    def edge_margin_nm(self, compat_eps: float) -> float:
        """Edge-curation margin in nm (``identify_bad_clusters(edge_margin_nm=...)``).

        Default: ``max(compat_eps, scale_nm())`` [derivation: a cluster
        within the separation scale of the ROI edge may be cut by it; the
        method's own separation scale is used when it is larger, and the
        reference pipeline's eps (25 nm, Gazal et al. 2026) is the floor so
        no experimental method curates less than DBSCAN would; the floor is
        ad hoc]. DBSCAN overrides it with eps itself (the compatibility
        path, unchanged); HDBSCAN / OPTICS use the largest core distance of
        the clustered points, the eps that would reproduce them.
        """
        s = self.scale_nm()
        floor = float(compat_eps)
        if s is None or not np.isfinite(s) or s <= 0:
            return floor
        return max(floor, float(s))


class DBSCANStrategy(ClusteringStrategy):
    """
    DBSCAN clustering strategy.

    Best for:
    - Small to medium datasets (<100k points)
    - When density-based clustering is needed
    - When epsilon parameter is known/estimated

    Characteristics:
    - O(n²) in worst case, O(n log n) typical
    - Parameters: eps, min_samples
    - Density-based approach
    """

    def __init__(
        self,
        eps: float = 1.0,
        min_samples: int = 5,
        metric: str = "euclidean",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize DBSCAN strategy.

        Parameters
        ----------
        eps : float
            Epsilon parameter for DBSCAN
        min_samples : int
            Minimum samples for core point
        metric : str
            Distance metric (default: euclidean)
        logger : logging.Logger, optional
            Logger instance for reporting
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.logger = logger
        # sklearn is not stub-typed (installed without type hints, or absent
        # entirely in the type-check CI job), so DBSCAN itself resolves to
        # Any under --ignore-missing-imports; the explicit annotation still
        # documents the intent and lets mypy narrow after assignment.
        self.model: Optional[DBSCAN] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN model.

        Parameters
        ----------
        data : np.ndarray
            Input data points (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        # Explicit annotation: fit_predict's return type is Any under
        # --ignore-missing-imports (no sklearn stubs available), so without
        # it mypy flags "Returning Any from function declared to return
        # ndarray" even though the runtime value genuinely is one.
        labels: np.ndarray = self.model.fit_predict(data)

        if self.logger:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            self.logger.debug(
                f"DBSCAN: {n_clusters} clusters, {n_noise} noise points"
            )

        return labels

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "DBSCAN"

    def scale_nm(self) -> Optional[float]:
        """eps itself [literature: Gazal et al. 2026, eps = 25 nm]."""
        return float(self.eps)

    def edge_margin_nm(self, compat_eps: float) -> float:
        """eps itself, whatever ``compat_eps`` says: the edge margin of the
        reference pipeline (compatibility path, bit for bit as before Phase D)."""
        return float(self.eps)

    def get_params(self) -> dict:
        """Get strategy parameters."""
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric
        }


class HDBSCANStrategy(ClusteringStrategy):
    """
    HDBSCAN (Hierarchical DBSCAN) clustering strategy.

    Best for:
    - Large datasets (>100k points)
    - Variable-density clusters
    - When DBSCAN eps is hard to tune
    - Need faster clustering

    Characteristics:
    - O(n log n) typical
    - Hierarchical approach
    - Automatic cluster selection
    - Handles variable density well
    - Less sensitive to parameters

    Performance:
    - 3-10x faster than DBSCAN on large datasets
    - Better memory efficiency
    - More robust to parameter selection
    """

    def __init__(
        self,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        metric: str = "euclidean",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HDBSCAN strategy.

        Parameters
        ----------
        min_samples : int
            Number of samples in a neighborhood for a point to be core
        min_cluster_size : int
            Minimum number of samples in a cluster
        metric : str
            Distance metric (default: euclidean)
        logger : logging.Logger, optional
            Logger instance for reporting
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN not installed. Install with: pip install hdbscan"
            )

        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self._scale: Optional[float] = None
        self._max_core: Optional[float] = None
        self.metric = metric
        self.logger = logger
        # hdbscan is imported conditionally (may be absent) and has no
        # stubs, so Any is the honest declared type rather than a specific
        # class reference that may not resolve consistently.
        self.model: Optional[Any] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN model.

        Parameters
        ----------
        data : np.ndarray
            Input data points (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        self.model = hdbscan.HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            metric=self.metric
        )
        self.model.fit(data)
        labels: np.ndarray = self.model.labels_
        self._scale = core_distance_scale(np.asarray(data, dtype=float), labels,
                                          int(self.min_samples or self.min_cluster_size))
        self._max_core = core_distance_max(np.asarray(data, dtype=float), labels,
                                           int(self.min_samples or self.min_cluster_size))

        if self.logger:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            self.logger.debug(
                f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points"
            )

        return labels

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "HDBSCAN"

    def scale_nm(self) -> Optional[float]:
        """Median core distance of the clustered points [derivation].

        HDBSCAN has no eps: its density scale at each point is the distance
        to its ``min_samples``-th neighbour (the core distance, Campello et
        al. 2013). The median over the points inside clusters is the one
        number that plays eps's role; ``None`` before ``fit``. Informative:
        the edge margin is :meth:`edge_margin_nm`.
        """
        return self._scale

    def edge_margin_nm(self, compat_eps: float) -> float:
        """max(compat_eps, largest core distance of the clustered points)
        [derivation: the eps at which DBSCAN would make every clustered
        point a core point, i.e. HDBSCAN's effective separation scale]."""
        return max(float(compat_eps), float(self._max_core or 0.0))



    def get_params(self) -> dict:
        """Get strategy parameters."""
        return {
            "min_samples": self.min_samples,
            "min_cluster_size": self.min_cluster_size,
            "metric": self.metric
        }


class AutoClusteringStrategy(ClusteringStrategy):
    """
    Automatic strategy selector.

    Chooses between DBSCAN and HDBSCAN based on dataset size:
    - < 100k points: DBSCAN (fast, well-tuned)
    - >= 100k points: HDBSCAN (faster, more robust)

    This strategy bridges both implementations seamlessly.
    """

    # Threshold for switching from DBSCAN to HDBSCAN
    HDBSCAN_THRESHOLD = 100000

    def __init__(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5,
        metric: str = "euclidean",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize automatic strategy selector.

        Parameters
        ----------
        eps : float, optional
            Epsilon for DBSCAN (used if DBSCAN is selected)
        min_samples : int
            Minimum samples parameter
        metric : str
            Distance metric
        logger : logging.Logger, optional
            Logger instance
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.logger = logger
        self.strategy: Optional[ClusteringStrategy] = None
        self.selected_strategy_name: Optional[str] = None

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Fit using automatically selected strategy.

        Parameters
        ----------
        data : np.ndarray
            Input data points (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Cluster labels
        """
        n_points = len(data)

        # Select strategy based on dataset size
        if n_points >= self.HDBSCAN_THRESHOLD:
            # Use HDBSCAN for large datasets
            if not HDBSCAN_AVAILABLE:
                if self.logger:
                    self.logger.warning(
                        "HDBSCAN not available, falling back to DBSCAN. "
                        "Install with: pip install hdbscan"
                    )
                self.strategy = self._create_dbscan_strategy()
            else:
                if self.logger:
                    self.logger.info(
                        f"Dataset has {n_points:,} points (>= {self.HDBSCAN_THRESHOLD:,}). "
                        "Using HDBSCAN for better performance."
                    )
                self.strategy = self._create_hdbscan_strategy()
        else:
            # Use DBSCAN for small/medium datasets
            if self.logger:
                self.logger.info(
                    f"Dataset has {n_points:,} points "
                    f"(< {self.HDBSCAN_THRESHOLD:,}). Using DBSCAN."
                )
            self.strategy = self._create_dbscan_strategy()

        self.selected_strategy_name = self.strategy.get_strategy_name()

        # Delegate to selected strategy
        return self.strategy.fit(data)

    def _create_dbscan_strategy(self) -> DBSCANStrategy:
        """Create DBSCAN strategy instance."""
        if self.eps is None:
            # Use a reasonable default if eps not provided
            eps_value = 1.0
        else:
            eps_value = self.eps

        return DBSCANStrategy(
            eps=eps_value,
            min_samples=self.min_samples,
            metric=self.metric,
            logger=self.logger
        )

    def _create_hdbscan_strategy(self) -> HDBSCANStrategy:
        """Create HDBSCAN strategy instance."""
        # For HDBSCAN, min_samples and min_cluster_size should be similar
        min_cluster_size = max(5, self.min_samples)

        return HDBSCANStrategy(
            min_samples=self.min_samples,
            min_cluster_size=min_cluster_size,
            metric=self.metric,
            logger=self.logger
        )

    def get_strategy_name(self) -> str:
        """Return selected strategy name."""
        if self.strategy is None:
            return "AutoStrategy (not yet fitted)"
        return f"AutoStrategy ({self.selected_strategy_name})"

    def get_selected_strategy(self) -> Optional[ClusteringStrategy]:
        """Get the selected strategy instance."""
        return self.strategy

    def get_selected_strategy_name(self) -> Optional[str]:
        """Get just the selected strategy name."""
        return self.selected_strategy_name


def create_clustering_strategy(
    strategy_type: str = "auto",
    eps: Optional[float] = None,
    min_samples: int = 5,
    min_cluster_size: Optional[int] = None,
    metric: str = "euclidean",
    logger: Optional[logging.Logger] = None
) -> ClusteringStrategy:
    """
    Factory function to create clustering strategy.

    Parameters
    ----------
    strategy_type : str
        Strategy type: "dbscan", "hdbscan", or "auto" (default)
    eps : float, optional
        Epsilon for DBSCAN
    min_samples : int
        Minimum samples parameter
    min_cluster_size : int, optional
        Minimum cluster size for HDBSCAN (default: min_samples if not specified)
    metric : str
        Distance metric
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    ClusteringStrategy
        Strategy instance

    Raises
    ------
    ValueError
        If strategy_type is invalid or HDBSCAN not available for selected strategy
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "auto":
        return AutoClusteringStrategy(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            logger=logger
        )
    elif strategy_type == "dbscan":
        if eps is None:
            raise ValueError("eps parameter required for DBSCAN strategy")
        return DBSCANStrategy(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            logger=logger
        )
    elif strategy_type == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN not installed. Install with: pip install hdbscan"
            )
        # Use provided min_cluster_size or default to min_samples if not specified
        mcs = min_cluster_size if min_cluster_size is not None else min_samples
        return HDBSCANStrategy(
            min_samples=min_samples,
            min_cluster_size=mcs,
            metric=metric,
            logger=logger
        )
    else:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            "Use 'dbscan', 'hdbscan', or 'auto'."
        )


# ---------------------------------------------------------------------------
# The algorithm selector's labels (decision P-R58)
# ---------------------------------------------------------------------------
# The entry that runs AutoClusteringStrategy used to read "Auto", which said
# nothing of what it does: DBSCAN below HDBSCAN_THRESHOLD localizations and
# HDBSCAN above, a switch made for speed and not justified by the data, and
# used by the main window's own clustering preview only. Its honest name says
# that it adapts, and that it is exploratory. "Auto" stays readable: a stored
# setting, a script or a test that names it still selects the same strategy.
ADAPTIVE_LABEL = "Adaptive (exploratory)"
LEGACY_ADAPTIVE_LABELS: Tuple[str, ...] = ("Auto",)
ALGORITHM_LABELS: Tuple[str, ...] = (ADAPTIVE_LABEL, "DBSCAN", "HDBSCAN")


def strategy_for_label(label: str) -> str:
    """The ``create_clustering_strategy`` type an algorithm label stands for.

    "DBSCAN" -> "dbscan", "HDBSCAN" -> "hdbscan", the adaptive entry (and its
    old name "Auto") -> "auto". Anything else is DBSCAN, the analysis' own
    algorithm, rather than a switch nobody chose.
    """
    text = str(label).strip()
    if text == ADAPTIVE_LABEL or text in LEGACY_ADAPTIVE_LABELS or text.lower() == "auto":
        return "auto"
    if text.upper() == "HDBSCAN":
        return "hdbscan"
    return "dbscan"


def label_for(label: str) -> str:
    """The label shown today for a stored one ("Auto" -> the adaptive entry's name)."""
    kind = strategy_for_label(label)
    return {"auto": ADAPTIVE_LABEL, "hdbscan": "HDBSCAN"}.get(kind, "DBSCAN")


# ===========================================================================
# Phase D registry (D0, 2026-10-16)
# ===========================================================================

def core_distance_scale(xy: np.ndarray, labels: np.ndarray, k: int) -> Optional[float]:
    """Median distance from each clustered point to its k-th neighbour [derivation].

    The scale that methods without an eps (HDBSCAN, OPTICS-xi, ToMATo with a
    k-NN density) actually work at: the core distance of Campello et al.
    (2013) / the k-NN density scale. ``None`` when fewer than k+1 points are
    inside clusters.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(xy, dtype=float)[:, :2]
    inside = np.asarray(labels) >= 0
    if int(inside.sum()) <= k:
        return None
    sub_pts = pts[inside]
    d, _ = cKDTree(sub_pts).query(sub_pts, k=k + 1)
    return float(np.median(d[:, -1]))


def core_distance_max(xy: np.ndarray, labels: np.ndarray, k: int) -> Optional[float]:
    """Largest distance from a clustered point to its k-th clustered neighbour [derivation].

    The eps at which DBSCAN(eps, k + 1) would make every clustered point a
    core point: the separation scale of a hierarchical density method, used
    as its edge-curation margin (``ClusteringStrategy.edge_margin_nm``).
    ``None`` when fewer than k+1 points are inside clusters.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(xy, dtype=float)[:, :2]
    inside = np.asarray(labels) >= 0
    if int(inside.sum()) <= k:
        return None
    sub_pts = pts[inside]
    d, _ = cKDTree(sub_pts).query(sub_pts, k=k + 1)
    return float(np.max(d[:, -1]))


OCCUPIED_GRID_NM = 100.0
"""Side of the grid that maps the region the localizations occupy [ad hoc: 4 x
the compatibility eps of 25 nm, wider than one cluster (~60 nm across in the
benchmark) and half the cluster spacing of the sparse regime (200 nm), so the
occupied cells of consecutive clusters touch after one dilation]."""

OCCUPIED_DILATION = 1
"""Grid cells added around every occupied cell (8-connected) [ad hoc: one cell
closes the gaps between consecutive clusters of a ring without filling the
empty axoplasm of an axon wider than ~3 cells]."""


class OccupiedSupport:
    """The region the localizations occupy, as a dilated occupancy grid (2-D).

    Every null model of Phase D that draws "random localizations" must draw
    them where the data could be, not in the bounding box or the convex hull
    of a ring, which include the empty axoplasm (REVIEW_D0_MODULES A3 / M3:
    on a 20-um ring the hull is ~5x the band the localizations occupy). Levet
    et al. (2015) take the mean density as n / image area and Andronov et al.
    (2016) draw their random sets "throughout the ROI"; on an axon ring both
    regions include the empty interior, so here the region is the set of
    ``cell_nm`` grid cells holding at least one localization, dilated by
    ``dilation`` cells [derivation: the reference region of the null is the
    region of support of the data; grid side and dilation are ad hoc, see
    OCCUPIED_GRID_NM / OCCUPIED_DILATION]. Only x and y are used.
    """

    def __init__(self, xy: np.ndarray, cell_nm: float = OCCUPIED_GRID_NM,
                 dilation: int = OCCUPIED_DILATION) -> None:
        from scipy import ndimage
        pts = np.asarray(xy, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            raise ValueError("OccupiedSupport needs at least one (x, y) point")
        pts = pts[:, :2]
        if float(cell_nm) <= 0:
            raise ValueError("cell_nm must be > 0")
        self.cell_nm = float(cell_nm)
        self.dilation = max(int(dilation), 0)
        self.origin = pts.min(axis=0) - (self.dilation + 1) * self.cell_nm
        idx = np.floor((pts - self.origin) / self.cell_nm).astype(np.int64)
        shape = (int(idx[:, 0].max()) + self.dilation + 2, int(idx[:, 1].max()) + self.dilation + 2)
        mask = np.zeros(shape, dtype=bool)
        mask[idx[:, 0], idx[:, 1]] = True
        if self.dilation:
            mask = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool),
                                           iterations=self.dilation)
        self.mask = mask
        self._cells = np.argwhere(mask)

    @property
    def area_nm2(self) -> float:
        """Area of the occupied region in nm^2."""
        return float(self._cells.shape[0]) * self.cell_nm ** 2

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of the points (x, y in the first two columns) inside the region."""
        p = np.asarray(points, dtype=float)[:, :2]
        idx = np.floor((p - self.origin) / self.cell_nm).astype(np.int64)
        ok = ((idx[:, 0] >= 0) & (idx[:, 1] >= 0)
              & (idx[:, 0] < self.mask.shape[0]) & (idx[:, 1] < self.mask.shape[1]))
        out = np.zeros(p.shape[0], dtype=bool)
        out[ok] = self.mask[idx[ok, 0], idx[ok, 1]]
        return out

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` points uniform over the region (every cell has the same area)."""
        pick = rng.integers(0, self._cells.shape[0], size=int(n))
        return np.asarray(self.origin + (self._cells[pick] + rng.random((int(n), 2))) * self.cell_nm,
                          dtype=np.float64)


class StrategyInfo:
    """One entry of :data:`STRATEGY_REGISTRY`.

    ``defaults`` maps a constructor parameter to ``{"value", "origin"}`` so
    the interface shows each number with where it comes from (literature /
    derivation / simulation / ad hoc). ``experimental`` is False only for
    DBSCAN (D10): every other method is hidden until the user ticks
    'Experimental clustering methods'.
    """

    def __init__(self, name: str, cls: Type[ClusteringStrategy], family: str, description: str,
                 defaults: Dict[str, Dict[str, Any]], experimental: bool = True,
                 available: bool = True, citation: str = "") -> None:
        self.name = name
        self.cls = cls
        self.family = family
        self.description = description
        self.defaults = defaults
        self.experimental = experimental
        self.available = available
        self.citation = citation

    def default_params(self) -> Dict[str, Any]:
        return {k: v["value"] for k, v in self.defaults.items()}

    def numeric_params(self) -> List[str]:
        """Parameters the interface edits with a spinbox (ints and floats; None/str/bool are left as they are)."""
        return [k for k, v in self.defaults.items()
                if isinstance(v["value"], (int, float)) and not isinstance(v["value"], bool)]


COMPAT_METHOD = "DBSCAN"
STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {}


def _merge_defaults(defaults: Dict[str, Any], origins: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in defaults.items():
        if isinstance(v, dict) and "value" in v:
            out[k] = {"value": v["value"], "origin": str(v.get("origin", "ad hoc"))}
        else:
            out[k] = {"value": v, "origin": origins.get(k, "ad hoc")}
    return out


def strategy_registry() -> Dict[str, StrategyInfo]:
    """Build (once) and return the registry: DBSCAN (compatibility, not
    experimental), HDBSCAN (the ``hdbscan`` package, optional D9), the density,
    tessellation and topological families, and external labels.

    The families are imported here, not at module level, because they import
    :class:`ClusteringStrategy` from this module.
    """
    if STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY
    reg = STRATEGY_REGISTRY
    reg[COMPAT_METHOD] = StrategyInfo(
        COMPAT_METHOD, DBSCANStrategy, "density",
        "DBSCAN(eps, min samples): the reference pipeline (compatibility mode 25/10).",
        {"eps": {"value": 25.0, "origin": "literature: Gazal et al. (2026), eps = 25 nm"},
         "min_samples": {"value": 10, "origin": "the reference pipeline's value (Gazal et al., 2026, preprint; only eps = 25 nm is among the verified sentences of the paper)"}},
        experimental=False, citation="Ester, Kriegel, Sander & Xu (1996), KDD")
    reg["HDBSCAN"] = StrategyInfo(
        "HDBSCAN", HDBSCANStrategy, "density",
        "Hierarchical DBSCAN (hdbscan package; optional dependency, D9).",
        {"min_samples": {"value": 5, "origin": "simulation: verification run A of the project (2026-10-16), = min_cluster_size / 2"},
         "min_cluster_size": {"value": 10, "origin": "derivation: about a third of the localizations of one labelled molecule in the benchmark (28 / 3)"}},
        available=HDBSCAN_AVAILABLE, citation="Campello, Moulavi & Sander (2013), PAKDD")
    try:
        from tools.cluster_density import density_family
        for name, e in density_family().items():
            reg[name] = StrategyInfo(name, e["cls"], "density", e["description"],
                                     _merge_defaults(e["defaults"], {}),
                                     available=bool(e.get("available", True)), citation=e.get("citation", ""))
    except ImportError:  # pragma: no cover
        pass
    try:
        from tools.cluster_tessellation import tessellation_family
        for name, e in tessellation_family().items():
            reg[name] = StrategyInfo(name, e["cls"], "tessellation", e["description"],
                                     _merge_defaults(e["defaults"], e.get("origins", {})),
                                     citation="Levet et al. (2015); Andronov et al. (2016)")
    except ImportError:  # pragma: no cover
        pass
    try:
        from tools.cluster_topological import topological_family
        for name, e in topological_family().items():
            reg[name] = StrategyInfo(name, e["cls"], "topological", e["description"],
                                     _merge_defaults(e["defaults"], e.get("origins", {})),
                                     citation="Chazal et al. (2013); Scurll et al. (2019)")
    except ImportError:  # pragma: no cover
        pass
    try:
        from tools.cluster_benchmark import ExternalLabelsStrategy
        reg["External labels"] = StrategyInfo(
            "External labels", ExternalLabelsStrategy, "external",
            "Labels computed outside (CSV with x, y, label); the scale is the one the user declares.",
            {"csv_path": {"value": "", "origin": "user: file chosen in the interface"},
             "scale_nm": {"value": 25.0, "origin": "user: declared scale; default = DBSCAN eps [ad hoc]"},
             "tol_nm": {"value": 1e-3, "origin": "ad hoc: exact round trips"}})
    except ImportError:  # pragma: no cover
        pass
    return reg


def registry_names(include_experimental: bool = True, available_only: bool = True) -> List[str]:
    """Names in interface order: DBSCAN first, then the families; hides the experimental ones on request (D10)."""
    return [n for n, e in strategy_registry().items()
            if (include_experimental or not e.experimental) and (e.available or not available_only)]


def make_strategy(name: str, params: Optional[Dict[str, Any]] = None) -> ClusteringStrategy:
    """Instantiate a registered strategy with ``params`` over its defaults.

    Unknown parameter names are ignored (a settings file of a newer version
    must not break an older program). Raises ``KeyError`` for an unknown
    method and ``ImportError`` for one whose package is missing.
    """
    info = strategy_registry()[name]
    if not info.available:
        raise ImportError(f"{name}: its package is not installed (optional dependency)")
    kw = info.default_params()
    for k, v in (params or {}).items():
        if k in kw:
            kw[k] = v
    if info.cls is HDBSCANStrategy or info.cls is DBSCANStrategy:
        kw = {k: v for k, v in kw.items() if k in ("eps", "min_samples", "min_cluster_size")}
    # A None default means "the constructor's own default" (OPTICS max_eps is
    # declared None = unbounded but the constructor takes a number).
    kw = {k: v for k, v in kw.items() if v is not None}
    return info.cls(**kw)


def strategy_params_json(strategy: ClusteringStrategy) -> Dict[str, Any]:
    """The parameters actually used, JSON-safe (numpy scalars and tuples become plain values)."""
    getter = getattr(strategy, "get_params", None)
    raw = dict(getter()) if callable(getter) else {}
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (tuple, list)):
            out[k] = [float(x) for x in v]
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif k in ("logger", "roi_polygon"):
            continue
        else:
            out[k] = v
    return out
