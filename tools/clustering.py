#!/usr/bin/env python3
"""
DBSCAN Parameter Optimization Module

Provides automated parameter estimation for DBSCAN clustering to reduce manual
tuning and improve clustering success rates.

Features:
  - Adaptive epsilon estimation using KNN distance plot
  - Automatic min_samples scaling based on dataset size
  - K-distance graph visualization support
  - Logging integration
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.neighbors import NearestNeighbors
import logging


def estimate_optimal_eps(
    data: np.ndarray,
    k: int = 5,
    percentile: float = 90
) -> float:
    """
    Estimate optimal epsilon for DBSCAN using KNN distance method.

    This method analyzes the K-distance graph (distances to k-th nearest
    neighbor) and uses the specified percentile as the epsilon estimate.
    This automatically adapts to data density and distribution.

    Parameters
    ----------
    data : np.ndarray
        Input data points with shape (n_samples, n_features).
        Typically (n_points, 2) for x,y coordinates.
    k : int, optional
        Number of nearest neighbors to consider (default: 5).
        Typical range: 3-10. Higher k = more conservative estimate.
    percentile : float, optional
        Percentile of distances to use as eps estimate (default: 90).
        Range: 50-95. Higher = more permissive clustering.

    Returns
    -------
    float
        Estimated epsilon value for DBSCAN.

    Notes
    -----
    The KNN distance method works by:
    1. Computing distances to k-th nearest neighbor for all points
    2. Sorting these distances
    3. Using the specified percentile as the eps threshold

    Intuition: The "elbow" in the sorted distance graph indicates the
    natural density threshold. The percentile provides a robust estimate
    without manual inspection.

    References
    ----------
    Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
    A density-based algorithm for discovering clusters in large spatial
    databases with noise. In KDD (Vol. 96, pp. 226-231).

    Examples
    --------
    >>> data = np.random.normal(5000, 1000, (1000, 2))
    >>> eps = estimate_optimal_eps(data, k=5, percentile=90)
    >>> print(f"Estimated epsilon: {eps:.2f}")
    """
    if len(data) < k:
        # Fallback for very small datasets
        return 1.0

    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)

    # Extract distances to k-th neighbor (last column)
    k_distances = np.sort(distances[:, k - 1])

    # Estimate eps as percentile of sorted k-distances
    eps_estimate = np.percentile(k_distances, percentile)

    return float(eps_estimate)


def estimate_min_samples(n_points: int, dimensionality: int = 2) -> int:
    """
    Estimate optimal min_samples parameter for DBSCAN.

    Provides automatic scaling based on dataset size and dimensionality.
    Uses the heuristic: min_samples >= 2 * dimensionality for small datasets,
    and sqrt(n_points) for larger datasets.

    Parameters
    ----------
    n_points : int
        Number of points in the dataset.
    dimensionality : int, optional
        Number of dimensions (features) in the data (default: 2).
        Typically 2 for x,y coordinates, 3 for x,y,z.

    Returns
    -------
    int
        Recommended min_samples value for DBSCAN.

    Notes
    -----
    The heuristic is based on:
    - Small datasets: min_samples = 2 * d (where d = dimensionality)
    - Large datasets: min_samples = sqrt(n) for better balance
    - This prevents overfitting on large datasets while maintaining
      sensitivity on small datasets

    Examples
    --------
    >>> min_samples = estimate_min_samples(10000, dimensionality=2)
    >>> print(f"Recommended min_samples: {min_samples}")
    Recommended min_samples: 100

    >>> min_samples = estimate_min_samples(1000, dimensionality=2)
    >>> print(f"Recommended min_samples: {min_samples}")
    Recommended min_samples: 4
    """
    # Base rule: 2x dimensionality (minimum)
    base_min_samples = 2 * dimensionality

    # Scale for larger datasets using sqrt rule
    # This prevents treating noise as individual clusters in large datasets
    if n_points > 10000:
        sqrt_min_samples = int(np.sqrt(n_points))
        return max(base_min_samples, sqrt_min_samples)
    else:
        return base_min_samples


def analyze_clustering_quality(
    labels: np.ndarray,
    n_points: int
) -> dict:
    """
    Analyze clustering results to assess quality.

    Computes statistics about the clustering to help determine if
    parameters are appropriate.

    Parameters
    ----------
    labels : np.ndarray
        Cluster labels from DBSCAN (shape: n_points).
        -1 indicates noise points.
    n_points : int
        Total number of points in dataset.

    Returns
    -------
    dict
        Dictionary with clustering statistics:
        - n_clusters: Number of clusters found
        - n_noise: Number of noise points
        - noise_percentage: Percentage of noise points
        - avg_cluster_size: Average points per cluster
        - quality_assessment: String assessment (Good/Moderate/Poor)

    Examples
    --------
    >>> labels = np.array([-1, 0, 0, 1, 1, 1, -1])
    >>> stats = analyze_clustering_quality(labels, 7)
    >>> print(stats['quality_assessment'])
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    noise_pct = 100 * n_noise / n_points if n_points > 0 else 0

    if n_clusters > 0:
        avg_cluster_size = (n_points - n_noise) / n_clusters
    else:
        avg_cluster_size = 0

    # Assess quality based on noise percentage and cluster count
    if n_clusters == 0:
        quality = "Poor (no clusters found - eps too small)"
    elif noise_pct > 80:
        quality = "Poor (too much noise - eps too small)"
    elif noise_pct > 50:
        quality = "Moderate (high noise - consider adjusting eps)"
    elif noise_pct > 20:
        quality = "Good (reasonable noise level)"
    else:
        quality = "Excellent (low noise, well-clustered)"

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_percentage": noise_pct,
        "avg_cluster_size": avg_cluster_size,
        "quality_assessment": quality,
    }


def suggest_parameter_adjustment(
    current_eps: float,
    current_min_samples: int,
    labels: np.ndarray,
    n_points: int,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[float], Optional[int]]:
    """
    Suggest parameter adjustments based on clustering results.

    Analyzes current clustering quality and recommends changes to eps
    and min_samples to improve results.

    Parameters
    ----------
    current_eps : float
        Current epsilon parameter value.
    current_min_samples : int
        Current min_samples parameter value.
    labels : np.ndarray
        Cluster labels from DBSCAN.
    n_points : int
        Total number of points.
    logger : logging.Logger, optional
        Logger for reporting suggestions.

    Returns
    -------
    Tuple[Optional[float], Optional[int]]
        Suggested (eps, min_samples) adjustments.
        Returns (None, None) if current parameters are adequate.

    Examples
    --------
    >>> eps_adjust, ms_adjust = suggest_parameter_adjustment(
    ...     current_eps=0.5,
    ...     current_min_samples=5,
    ...     labels=labels,
    ...     n_points=1000
    ... )
    """
    stats = analyze_clustering_quality(labels, n_points)
    n_clusters = stats["n_clusters"]
    noise_pct = stats["noise_percentage"]

    suggested_eps = None
    suggested_min_samples = None

    if n_clusters == 0:
        # No clusters found - eps is too small
        suggested_eps = current_eps * 1.5
        msg = f"No clusters found. Suggest increasing eps from {current_eps:.2f} to {suggested_eps:.2f}"
    elif noise_pct > 80:
        # Too much noise - eps is too small
        suggested_eps = current_eps * 1.3
        msg = f"High noise ({noise_pct:.1f}%). Suggest increasing eps from {current_eps:.2f} to {suggested_eps:.2f}"
    elif noise_pct < 5 and n_clusters < 3:
        # Very few clusters, almost no noise - eps might be too large
        suggested_eps = current_eps * 0.8
        msg = f"Few clusters with low noise. Suggest decreasing eps from {current_eps:.2f} to {suggested_eps:.2f}"
    else:
        msg = f"Current parameters adequate. {stats['quality_assessment']}"

    if logger:
        logger.info(f"Parameter Assessment: {msg}")
        logger.info(
            f"Clustering: {n_clusters} clusters, {stats['n_noise']} noise "
            f"({noise_pct:.1f}%), avg size: {stats['avg_cluster_size']:.1f}"
        )

    return suggested_eps, suggested_min_samples


# Utility function for integration with UI
def get_auto_parameters(
    roi_data: np.ndarray,
    use_adaptive_eps: bool = True,
    use_adaptive_min_samples: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[float, int]:
    """
    Get automatic DBSCAN parameters for given ROI data.

    Convenience function that computes both eps and min_samples
    automatically based on the data.

    Parameters
    ----------
    roi_data : np.ndarray
        ROI-selected data points (n_points, 2).
    use_adaptive_eps : bool, optional
        If True, estimate eps automatically (default: True).
    use_adaptive_min_samples : bool, optional
        If True, estimate min_samples automatically (default: True).
    logger : logging.Logger, optional
        Logger instance for reporting.

    Returns
    -------
    Tuple[float, int]
        (estimated_eps, estimated_min_samples)

    Examples
    --------
    >>> roi_data = roi_points[:, :2]  # x, y columns
    >>> eps, min_samples = get_auto_parameters(roi_data)
    >>> print(f"Auto parameters: eps={eps:.2f}, min_samples={min_samples}")
    """
    eps = (
        estimate_optimal_eps(roi_data)
        if use_adaptive_eps
        else 1.0
    )

    min_samples = (
        estimate_min_samples(len(roi_data), dimensionality=2)
        if use_adaptive_min_samples
        else 5
    )

    if logger:
        logger.info(
            f"Auto-Parameters: eps={eps:.3f}, min_samples={min_samples} "
            f"(data: {len(roi_data):,} points)"
        )

    return eps, min_samples
