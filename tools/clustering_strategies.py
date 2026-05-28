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
from typing import Optional, Tuple
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
        self.model = None

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
        labels = self.model.fit_predict(data)

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
        self.metric = metric
        self.logger = logger
        self.model = None

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
        labels = self.model.labels_

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
        self.strategy = None
        self.selected_strategy_name = None

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
        return HDBSCANStrategy(
            min_samples=min_samples,
            min_cluster_size=min_samples,
            metric=metric,
            logger=logger
        )
    else:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            "Use 'dbscan', 'hdbscan', or 'auto'."
        )
