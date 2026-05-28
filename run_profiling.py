#!/usr/bin/env python3
"""
Profiling Runner for MPS Explorer Performance Analysis

This script profiles key DBSCAN clustering and data processing operations
to identify performance bottlenecks and scalability issues.

Usage:
    python run_profiling.py              # Default test sizes
    python run_profiling.py --verbose    # Verbose output
    python run_profiling.py --export     # Export JSON report

Features:
    - Profile file import operations
    - Profile ROI filtering (circular and square)
    - Profile DBSCAN clustering with various parameters
    - Analyze computational complexity
    - Generate detailed performance reports
"""

import sys
import argparse
import numpy as np
from typing import Tuple
import logging

from profiler import PerformanceProfiler, get_profiler
from logging_config import setup_logging, get_logger


def generate_synthetic_data(n_points: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic localization data for testing.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays of x, y, z coordinates in nanometers.
    """
    np.random.seed(seed)

    # Generate data with realistic distribution
    x = np.random.normal(loc=5000, scale=1000, size=n_points)
    y = np.random.normal(loc=5000, scale=1000, size=n_points)
    z = np.random.normal(loc=100, scale=50, size=n_points)

    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def profile_roi_filtering(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         profiler: PerformanceProfiler, logger: logging.Logger) -> None:
    """
    Profile ROI filtering operations.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays.
    profiler : PerformanceProfiler
        Profiler instance.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("="*80)
    logger.info("PROFILING: ROI FILTERING OPERATIONS")
    logger.info("="*80)

    data_points = np.column_stack((x, y))
    center = np.array([5000.0, 5000.0])
    radius = 1000.0

    # Profile circular ROI filtering (vectorized)
    logger.info(f"Testing circular ROI filtering on {len(x):,} points...")
    profiler.start_timer("roi_circular_filter_vectorized")
    distances = np.linalg.norm(data_points - center, axis=1)
    mask = distances <= radius
    roi_points = data_points[mask]
    profiler.stop_timer("roi_circular_filter_vectorized", len(x))
    logger.info(f"  Result: {len(roi_points):,} points inside ROI")

    # Profile z-filtering
    logger.info(f"Testing z-filtering on {len(roi_points):,} points...")
    profiler.start_timer("roi_z_filter")
    z_filtered = z[mask]
    z_mask = (z_filtered > 50) & (z_filtered < 150)
    z_final = z_filtered[z_mask]
    profiler.stop_timer("roi_z_filter", len(roi_points))
    logger.info(f"  Result: {len(z_final):,} points after z-filter")

    # Profile square ROI filtering
    logger.info(f"Testing square ROI filtering on {len(x):,} points...")
    profiler.start_timer("roi_square_filter")
    x_min, x_max = 4000, 6000
    y_min, y_max = 4000, 6000
    square_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    square_points = data_points[square_mask]
    profiler.stop_timer("roi_square_filter", len(x))
    logger.info(f"  Result: {len(square_points):,} points inside square")


def profile_clustering(x: np.ndarray, y: np.ndarray, profiler: PerformanceProfiler,
                      logger: logging.Logger) -> None:
    """
    Profile DBSCAN clustering with various parameters.

    Parameters
    ----------
    x, y : np.ndarray
        X and Y coordinates for clustering.
    profiler : PerformanceProfiler
        Profiler instance.
    logger : logging.Logger
        Logger instance.
    """
    from sklearn.cluster import DBSCAN

    logger.info("="*80)
    logger.info("PROFILING: DBSCAN CLUSTERING")
    logger.info("="*80)

    data_points = np.column_stack((x, y))

    # Test different parameter combinations
    parameters = [
        (20, 0.5, "Tight clustering (eps=0.5, min_samples=20)"),
        (10, 1.0, "Medium clustering (eps=1.0, min_samples=10)"),
        (5, 2.0, "Loose clustering (eps=2.0, min_samples=5)"),
    ]

    for min_samples, eps, description in parameters:
        logger.info(f"\nTesting {description} on {len(x):,} points...")
        operation_name = f"dbscan_eps{eps}_ms{min_samples}"

        profiler.start_timer(operation_name)
        result = DBSCAN(eps=eps, min_samples=min_samples).fit(data_points)
        profiler.stop_timer(operation_name, len(x))

        n_clusters = len(set(result.labels_)) - (1 if -1 in result.labels_ else 0)
        n_noise = list(result.labels_).count(-1)
        logger.info(f"  Result: {n_clusters} clusters, {n_noise:,} noise points")

    # Profile centroid calculation
    logger.info(f"\nTesting centroid calculation...")
    result = DBSCAN(eps=1.0, min_samples=10).fit(data_points)
    labels = result.labels_

    profiler.start_timer("centroid_calculation")
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_points = data_points[labels == label]
        centroids.append(np.mean(cluster_points, axis=0))
    centroids = np.array(centroids)
    profiler.stop_timer("centroid_calculation", len(unique_labels))
    logger.info(f"  Computed {len(centroids)} centroids")


def profile_distance_calculations(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                 profiler: PerformanceProfiler, logger: logging.Logger) -> None:
    """
    Profile distance-based calculations.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays.
    profiler : PerformanceProfiler
        Profiler instance.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("="*80)
    logger.info("PROFILING: DISTANCE CALCULATIONS")
    logger.info("="*80)

    data_3d = np.column_stack((x, y, z))
    data_2d = np.column_stack((x, y))

    # Profile pairwise distance to a single point (common operation)
    reference_point = np.mean(data_2d, axis=0)
    logger.info(f"Testing pairwise distances to reference point (n={len(x):,})...")
    profiler.start_timer("distances_to_point")
    distances_to_ref = np.linalg.norm(data_2d - reference_point, axis=1)
    profiler.stop_timer("distances_to_point", len(x))
    logger.info(f"  Computed {len(distances_to_ref)} distances to reference")

    # Profile pairwise distance to multiple reference points (e.g., cluster centroids)
    n_reference_points = min(100, len(x) // 100)
    reference_indices = np.random.choice(len(x), n_reference_points, replace=False)
    reference_points = data_2d[reference_indices]

    logger.info(f"Testing distances to {n_reference_points} reference points...")
    profiler.start_timer("distances_to_multiple_points")
    distances_to_refs = np.zeros((len(x), n_reference_points))
    for i, ref_point in enumerate(reference_points):
        distances_to_refs[:, i] = np.linalg.norm(data_2d - ref_point, axis=1)
    profiler.stop_timer("distances_to_multiple_points", len(x))
    logger.info(f"  Computed {distances_to_refs.shape[0]}x{distances_to_refs.shape[1]} distance matrix")


def profile_data_sorting_filtering(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                  profiler: PerformanceProfiler, logger: logging.Logger) -> None:
    """
    Profile data sorting and filtering operations.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays.
    profiler : PerformanceProfiler
        Profiler instance.
    logger : logging.Logger
        Logger instance.
    """
    logger.info("="*80)
    logger.info("PROFILING: SORTING AND FILTERING OPERATIONS")
    logger.info("="*80)

    # Profile sorting by z coordinate
    logger.info(f"Testing sorting {len(z):,} points by z-coordinate...")
    profiler.start_timer("sort_by_z")
    sorted_indices = np.argsort(z)
    profiler.stop_timer("sort_by_z", len(z))

    # Profile filtering by threshold
    logger.info(f"Testing filtering points by multiple thresholds...")
    profiler.start_timer("filter_by_threshold")
    mask = (z > 50) & (z < 150) & (x > 4000) & (x < 6000)
    filtered_points = np.column_stack((x, y, z))[mask]
    profiler.stop_timer("filter_by_threshold", len(z))
    logger.info(f"  Result: {len(filtered_points):,} points after filtering")

    # Profile unique value extraction
    logger.info(f"Testing unique value extraction...")
    profiler.start_timer("unique_values")
    unique_x = np.unique(np.round(x, decimals=1))
    unique_y = np.unique(np.round(y, decimals=1))
    unique_z = np.unique(np.round(z, decimals=1))
    profiler.stop_timer("unique_values", len(z))
    logger.info(f"  Found {len(unique_x)}x{len(unique_y)}x{len(unique_z)} unique coordinate values")


def profile_scalability(profiler: PerformanceProfiler, logger: logging.Logger) -> None:
    """
    Profile operations with increasing dataset sizes to analyze scalability.

    Parameters
    ----------
    profiler : PerformanceProfiler
        Profiler instance.
    logger : logging.Logger
        Logger instance.
    """
    from sklearn.cluster import DBSCAN

    logger.info("="*80)
    logger.info("PROFILING: SCALABILITY ANALYSIS (Increasing Dataset Sizes)")
    logger.info("="*80)

    dataset_sizes = [1000, 5000, 10000, 50000, 100000]
    eps_values = [0.5, 1.0, 2.0]

    for eps in eps_values:
        logger.info(f"\nScalability test: DBSCAN with eps={eps}")
        logger.info(f"{'Size':<10} {'Time (ms)':<12} {'Time/Size (µs)':<18}")
        logger.info("-" * 40)

        for size in dataset_sizes:
            x, y, _ = generate_synthetic_data(size)
            data_points = np.column_stack((x, y))

            profiler.start_timer(f"dbscan_scalability_eps{eps}")
            DBSCAN(eps=eps, min_samples=10).fit(data_points)
            elapsed = profiler.stop_timer(f"dbscan_scalability_eps{eps}", size)

            time_per_point = (elapsed * 1_000_000) / size
            logger.info(f"{size:<10} {elapsed*1000:<12.2f} {time_per_point:<18.2f}")


def main() -> int:
    """Main entry point for profiling."""
    parser = argparse.ArgumentParser(
        description="Profile MPS Explorer DBSCAN clustering and data processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_profiling.py              # Standard profiling
  python run_profiling.py --verbose    # Verbose output
  python run_profiling.py --export     # Export JSON report
  python run_profiling.py --scalability # Test scalability with varying sizes
        """
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--export", "-e",
        action="store_true",
        help="Export profiling data as JSON"
    )
    parser.add_argument(
        "--scalability", "-s",
        action="store_true",
        help="Run scalability analysis with increasing dataset sizes"
    )
    parser.add_argument(
        "--size", "-n",
        type=int,
        default=100000,
        help="Number of test data points (default: 100000)"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)

    # Get profiler instance
    profiler = get_profiler()
    profiler.set_logger(logger)

    logger.info("MPS Explorer Performance Profiling")
    logger.info(f"Test dataset size: {args.size:,} points")

    try:
        # Generate synthetic test data
        logger.info("Generating synthetic test data...")
        x, y, z = generate_synthetic_data(args.size)
        logger.info(f"Generated {len(x):,} test points")

        # Run profiling suites
        profile_roi_filtering(x, y, z, profiler, logger)
        profile_clustering(x, y, profiler, logger)
        profile_distance_calculations(x, y, z, profiler, logger)
        profile_data_sorting_filtering(x, y, z, profiler, logger)

        if args.scalability:
            profile_scalability(profiler, logger)

        # Generate and display reports
        logger.info("\n" + profiler.generate_report())

        if args.export:
            json_report = profiler.generate_json_report()
            with open("profiling_report.json", "w") as f:
                f.write(json_report)
            logger.info("JSON report exported to profiling_report.json")

        return 0

    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
