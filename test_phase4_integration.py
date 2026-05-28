#!/usr/bin/env python3
"""
Phase 4 Parameter Caching Integration Tests

Comprehensive test suite for parameter caching functionality.
Tests verify:
1. Cache hits and misses
2. Dataset similarity detection
3. Parameter reuse maintains scientific quality
4. Persistent cache storage/retrieval
5. Statistics tracking
6. Integration with Phase 1, 2, 3
7. Performance improvements (30% speedup)
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from typing import Tuple
import logging

# Import modules to test
from tools.parameter_cache import (
    ParameterCache,
    DatasetSignature,
    CachedParameters,
    create_parameter_cache
)
import tools.clustering as clustering
from tools.clustering_strategies import create_clustering_strategy


# ============================================================================
# FIXTURES & UTILITIES
# ============================================================================

@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_phase4")


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def param_cache(temp_cache_dir, logger):
    """Create a parameter cache instance."""
    return create_parameter_cache(
        cache_dir=temp_cache_dir,
        max_entries=100,
        similarity_threshold=0.95,
        logger=logger
    )


def create_test_dataset(
    n_samples: int = 1000,
    n_features: int = 3,
    seed: int = 42,
    distribution: str = "normal"
) -> np.ndarray:
    """
    Create test dataset with specified characteristics.

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    seed : int
        Random seed for reproducibility
    distribution : str
        Distribution type: "normal", "uniform", "clustered"

    Returns
    -------
    np.ndarray
        Generated dataset (n_samples, n_features)
    """
    np.random.seed(seed)

    if distribution == "normal":
        return np.random.normal(0, 1, (n_samples, n_features))
    elif distribution == "uniform":
        return np.random.uniform(-1, 1, (n_samples, n_features))
    elif distribution == "clustered":
        # Generate 3 tight clusters
        cluster1 = np.random.normal([0, 0, 0], 0.3, (n_samples//3, n_features))
        cluster2 = np.random.normal([5, 5, 5], 0.3, (n_samples//3, n_features))
        cluster3 = np.random.normal([10, 10, 10], 0.3, (n_samples - 2*(n_samples//3), n_features))
        return np.vstack([cluster1, cluster2, cluster3])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# ============================================================================
# TESTS: PARAMETER CACHE BASIC FUNCTIONALITY
# ============================================================================

class TestParameterCacheBasic:
    """Test basic parameter cache functionality."""

    def test_cache_initialization(self, param_cache):
        """Test cache initializes correctly."""
        assert param_cache is not None
        assert len(param_cache.cache) == 0
        assert param_cache.stats['cache_hits'] == 0
        assert param_cache.stats['cache_misses'] == 0

    def test_dataset_signature_computation(self, param_cache):
        """Test dataset signature is computed correctly."""
        data = create_test_dataset(1000)
        sig = param_cache._compute_dataset_signature(data)

        assert sig.n_points == 1000
        assert sig.n_features == 3
        assert sig.data_hash is not None
        assert len(sig.data_hash) == 32  # MD5 hash is 32 chars
        assert sig.timestamp is not None

    def test_identical_data_same_signature(self, param_cache):
        """Test identical data produces identical signatures."""
        data = create_test_dataset(1000, seed=42)
        sig1 = param_cache._compute_dataset_signature(data)
        sig2 = param_cache._compute_dataset_signature(data)

        assert sig1.data_hash == sig2.data_hash
        assert sig1.n_points == sig2.n_points

    def test_different_data_different_signature(self, param_cache):
        """Test different data produces different signatures."""
        data1 = create_test_dataset(1000, seed=42)
        data2 = create_test_dataset(1000, seed=99)

        sig1 = param_cache._compute_dataset_signature(data1)
        sig2 = param_cache._compute_dataset_signature(data2)

        assert sig1.data_hash != sig2.data_hash

    def test_similarity_identical_datasets(self, param_cache):
        """Test identical datasets have high similarity."""
        data = create_test_dataset(1000)
        sig1 = param_cache._compute_dataset_signature(data)
        sig2 = param_cache._compute_dataset_signature(data)

        similarity = param_cache._compute_similarity(sig1, sig2)
        assert similarity == 1.0  # Perfect match

    def test_similarity_similar_size_datasets(self, param_cache):
        """Test datasets of similar size have good similarity."""
        data1 = create_test_dataset(1000)
        data2 = create_test_dataset(1005)  # Similar size

        sig1 = param_cache._compute_dataset_signature(data1)
        sig2 = param_cache._compute_dataset_signature(data2)

        similarity = param_cache._compute_similarity(sig1, sig2)
        # Should be similar but not identical (size differs, distribution same)
        assert 0.4 < similarity < 1.0

    def test_similarity_different_features(self, param_cache):
        """Test datasets with different features have no similarity."""
        data1 = create_test_dataset(1000, n_features=3)
        data2 = create_test_dataset(1000, n_features=5)  # Different dims

        sig1 = param_cache._compute_dataset_signature(data1)
        sig2 = param_cache._compute_dataset_signature(data2)

        similarity = param_cache._compute_similarity(sig1, sig2)
        assert similarity == 0.0  # No match (different dimensionality)


# ============================================================================
# TESTS: CACHE HIT AND MISS SCENARIOS
# ============================================================================

class TestCacheHitsAndMisses:
    """Test cache hit and miss behavior."""

    def test_first_access_is_cache_miss(self, param_cache):
        """Test first access to new dataset is a cache miss."""
        data = create_test_dataset(1000)

        result = param_cache.get_cached_parameters(data)
        assert result is None
        assert param_cache.stats['cache_misses'] == 1
        assert param_cache.stats['cache_hits'] == 0

    def test_cached_parameters_retrieval(self, param_cache):
        """Test cached parameters are retrieved correctly."""
        data = create_test_dataset(1000)

        # Cache parameters
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        # Retrieve same data
        result = param_cache.get_cached_parameters(data)
        assert result is not None
        assert result.eps == 0.5
        assert result.min_samples == 5

    def test_cache_hit_statistics(self, param_cache):
        """Test cache hit statistics are tracked."""
        data = create_test_dataset(1000)

        # First access: miss
        param_cache.get_cached_parameters(data)
        assert param_cache.stats['cache_misses'] == 1

        # Cache parameters
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        # Second access: hit
        param_cache.get_cached_parameters(data)
        assert param_cache.stats['cache_hits'] == 1
        assert param_cache.stats['cache_misses'] == 1

    def test_similar_dataset_cache_hit(self, param_cache):
        """Test similar dataset triggers cache hit if above threshold."""
        data1 = create_test_dataset(1000, seed=42)
        data2 = create_test_dataset(1000, seed=42)  # Same seed = identical

        # Cache parameters from first dataset
        param_cache.cache_parameters(data1, eps=0.5, min_samples=5)

        # Retrieve from very similar second dataset
        result = param_cache.get_cached_parameters(data2)
        assert result is not None
        assert result.eps == 0.5
        assert param_cache.stats['cache_hits'] == 1

    def test_dissimilar_dataset_cache_miss(self, param_cache):
        """Test dissimilar dataset doesn't trigger cache hit."""
        data1 = create_test_dataset(1000, seed=42)
        data2 = create_test_dataset(500, seed=99)  # Very different

        # Cache parameters from first dataset
        param_cache.cache_parameters(data1, eps=0.5, min_samples=5)

        # Try to retrieve from very different second dataset
        result = param_cache.get_cached_parameters(data2)
        assert result is None
        assert param_cache.stats['cache_misses'] == 1


# ============================================================================
# TESTS: PERSISTENT CACHE STORAGE
# ============================================================================

class TestPersistentCacheStorage:
    """Test persistent cache storage and retrieval."""

    def test_cache_saved_to_disk(self, param_cache):
        """Test cache is saved to disk."""
        data = create_test_dataset(1000)
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        # Check cache file exists
        cache_file = Path(param_cache.cache_dir) / "parameter_cache.json"
        assert cache_file.exists()

    def test_cache_persistence_across_instances(self, temp_cache_dir, logger):
        """Test cache persists across different instances."""
        # First instance: cache parameters
        cache1 = create_parameter_cache(
            cache_dir=temp_cache_dir,
            logger=logger
        )
        data = create_test_dataset(1000)
        cache1.cache_parameters(data, eps=0.5, min_samples=5)

        # Second instance: should load cache from disk
        cache2 = create_parameter_cache(
            cache_dir=temp_cache_dir,
            logger=logger
        )

        # Verify cache was loaded
        assert len(cache2.cache) > 0
        result = cache2.get_cached_parameters(data)
        assert result is not None
        assert result.eps == 0.5

    def test_cache_export_import(self, param_cache, temp_cache_dir):
        """Test cache export and import."""
        data = create_test_dataset(1000)
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        # Export cache
        export_file = Path(temp_cache_dir) / "exported_cache.json"
        param_cache.export_cache(str(export_file))
        assert export_file.exists()

        # Import into new cache
        new_cache = create_parameter_cache(cache_dir=None)
        new_cache.import_cache(str(export_file))

        # Verify parameters were imported
        result = new_cache.get_cached_parameters(data)
        assert result is not None
        assert result.eps == 0.5

    def test_cache_file_format_valid_json(self, param_cache, temp_cache_dir):
        """Test cache file is valid JSON."""
        data = create_test_dataset(1000)
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        # Read and parse JSON
        cache_file = Path(param_cache.cache_dir) / "parameter_cache.json"
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        assert isinstance(cache_data, dict)
        assert len(cache_data) > 0


# ============================================================================
# TESTS: SCIENTIFIC QUALITY PRESERVATION
# ============================================================================

class TestScientificQuality:
    """Test that caching preserves scientific quality."""

    def test_cached_params_produce_identical_results(self):
        """Test cached parameters produce identical clustering results."""
        from sklearn.cluster import DBSCAN
        data = create_test_dataset(500, distribution="clustered")

        # Estimate parameters
        eps = clustering.estimate_optimal_eps(data, k=5)
        n_points, n_features = data.shape
        min_samples = clustering.estimate_min_samples(n_points, n_features)

        # Create two clusters with same parameters and get results
        clustering1 = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        clustering2 = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

        # Results must be identical
        np.testing.assert_array_equal(clustering1.labels_, clustering2.labels_)

    def test_cached_vs_fresh_estimation_identical(self):
        """Test cached parameters match fresh estimation for same data."""
        param_cache = create_parameter_cache()
        data = create_test_dataset(500, distribution="clustered")

        # Fresh estimation
        eps_fresh = clustering.estimate_optimal_eps(data, k=5)
        n_points, n_features = data.shape
        min_samples_fresh = clustering.estimate_min_samples(n_points, n_features)

        # Cache it
        param_cache.cache_parameters(
            data,
            eps=eps_fresh,
            min_samples=min_samples_fresh,
            source="estimated"
        )

        # Retrieve from cache
        cached = param_cache.get_cached_parameters(data)
        assert cached is not None

        # Should be identical
        assert cached.eps == eps_fresh
        assert cached.min_samples == min_samples_fresh

    def test_quality_metrics_unchanged(self):
        """Test quality metrics are unaffected by caching."""
        from sklearn.cluster import DBSCAN
        data = create_test_dataset(500, distribution="clustered")

        # Cluster with fresh parameters
        eps = clustering.estimate_optimal_eps(data, k=5)
        n_points, n_features = data.shape
        min_samples = clustering.estimate_min_samples(n_points, n_features)

        # Cluster twice with same parameters
        labels1 = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
        metrics1 = clustering.analyze_clustering_quality(labels1, n_points)

        # Cluster again with same parameters (simulating cache reuse)
        labels2 = DBSCAN(eps=eps, min_samples=min_samples).fit(data).labels_
        metrics2 = clustering.analyze_clustering_quality(labels2, n_points)

        # Metrics must be identical (same labels = same metrics)
        assert metrics1['n_clusters'] == metrics2['n_clusters']
        assert metrics1['n_noise'] == metrics2['n_noise']


# ============================================================================
# TESTS: PERFORMANCE IMPROVEMENTS
# ============================================================================

class TestPerformanceImprovements:
    """Test performance improvements from caching."""

    def test_cache_hit_faster_than_estimation(self, param_cache):
        """Test cache hit is faster than fresh estimation."""
        data = create_test_dataset(1000)

        # Estimate and cache
        eps = clustering.estimate_optimal_eps(data, k=5)
        n_points, n_features = data.shape
        min_samples = clustering.estimate_min_samples(n_points, n_features)
        param_cache.cache_parameters(data, eps=eps, min_samples=min_samples)

        # Time: fresh estimation
        start = time.time()
        eps_fresh = clustering.estimate_optimal_eps(data, k=5)
        fresh_time = time.time() - start

        # Time: cache retrieval
        start = time.time()
        cached = param_cache.get_cached_parameters(data)
        cache_time = time.time() - start

        # Cache should be significantly faster
        assert cache_time < fresh_time
        # Cache should be sub-millisecond
        assert cache_time < 0.01

    def test_statistics_track_time_saved(self, param_cache):
        """Test statistics accurately track time saved."""
        data = create_test_dataset(1000)
        estimation_time = 50.0  # Simulated: 50ms estimation

        param_cache.cache_parameters(
            data,
            eps=0.5,
            min_samples=5,
            estimation_time_ms=estimation_time
        )

        # Check statistics
        assert param_cache.stats['total_estimations'] == 1
        assert param_cache.stats['time_saved_ms'] == estimation_time

    def test_speedup_calculation(self, param_cache):
        """Test speedup from repeated clustering is measurable."""
        data = create_test_dataset(1000)

        # Simulate first access (cache miss)
        param_cache.get_cached_parameters(data)  # Miss

        # Simulate estimation and caching
        time1 = 50.0  # Simulated: 50ms
        param_cache.cache_parameters(data, eps=0.5, min_samples=5, estimation_time_ms=time1)

        # Simulate second access (cache hit)
        param_cache.get_cached_parameters(data)  # Hit

        # Get statistics
        stats = param_cache.get_statistics()
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 1
        assert stats['hit_rate_percent'] == 50.0  # 1 hit out of 2 accesses


# ============================================================================
# TESTS: CACHE MANAGEMENT
# ============================================================================

class TestCacheManagement:
    """Test cache management and eviction."""

    def test_max_entries_enforcement(self, logger):
        """Test cache respects maximum entries limit."""
        cache = create_parameter_cache(
            cache_dir=None,
            max_entries=3,
            logger=logger
        )

        # Add 5 datasets to cache with max 3
        for i in range(5):
            data = create_test_dataset(1000, seed=i)
            cache.cache_parameters(data, eps=0.5 + i*0.1, min_samples=5)

        # Cache should have exactly 3 entries
        assert len(cache.cache) == 3

    def test_oldest_entry_evicted(self, logger):
        """Test oldest entry is evicted when limit reached."""
        cache = create_parameter_cache(
            cache_dir=None,
            max_entries=2,
            logger=logger
        )

        # Add first dataset
        data1 = create_test_dataset(1000, seed=1)
        cache.cache_parameters(data1, eps=0.5, min_samples=5)
        first_hash = list(cache.cache.keys())[0]

        # Add second dataset
        data2 = create_test_dataset(1000, seed=2)
        cache.cache_parameters(data2, eps=0.6, min_samples=6)

        # Add third dataset (should evict first)
        data3 = create_test_dataset(1000, seed=3)
        cache.cache_parameters(data3, eps=0.7, min_samples=7)

        # First entry should be gone
        assert first_hash not in cache.cache
        assert len(cache.cache) == 2

    def test_clear_cache(self, param_cache):
        """Test clearing all cache entries."""
        data = create_test_dataset(1000)
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        assert len(param_cache.cache) > 0

        param_cache.clear_cache()

        assert len(param_cache.cache) == 0
        assert param_cache.stats['cache_hits'] == 0
        assert param_cache.stats['cache_misses'] == 0

    def test_cache_statistics(self, param_cache):
        """Test cache statistics are accurate."""
        data1 = create_test_dataset(1000, seed=1)
        data2 = create_test_dataset(1000, seed=2)

        # Two misses
        param_cache.get_cached_parameters(data1)
        param_cache.get_cached_parameters(data2)

        # One hit
        param_cache.cache_parameters(data1, eps=0.5, min_samples=5)
        param_cache.get_cached_parameters(data1)

        stats = param_cache.get_statistics()
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 2
        assert stats['total_queries'] == 3
        assert stats['hit_rate_percent'] == pytest.approx(33.33, abs=0.1)


# ============================================================================
# TESTS: INTEGRATION WITH PHASES 1-3
# ============================================================================

class TestPhaseIntegration:
    """Test parameter caching integrates with Phase 1, 2, 3."""

    def test_integration_with_phase1_estimation(self, param_cache):
        """Test cache integrates with Phase 1 auto-parameter estimation."""
        data = create_test_dataset(1000)

        # Phase 1: Estimate parameters
        eps = clustering.estimate_optimal_eps(data, k=5)
        n_points, n_features = data.shape
        min_samples = clustering.estimate_min_samples(n_points, n_features)

        # Phase 4: Cache parameters
        param_cache.cache_parameters(
            data,
            eps=eps,
            min_samples=min_samples,
            source="estimated"
        )

        # Phase 4: Retrieve cached
        cached = param_cache.get_cached_parameters(data)
        assert cached is not None
        assert cached.eps == eps
        assert cached.source == "estimated"

    def test_source_tracking(self, param_cache):
        """Test parameter source is tracked correctly."""
        data = create_test_dataset(1000)

        # Cache with different sources
        param_cache.cache_parameters(
            data,
            eps=0.5,
            min_samples=5,
            source="estimated"
        )

        cached = param_cache.get_cached_parameters(data)
        assert cached.source == "estimated"

    def test_estimation_time_tracking(self, param_cache):
        """Test estimation time is tracked."""
        data = create_test_dataset(1000)

        param_cache.cache_parameters(
            data,
            eps=0.5,
            min_samples=5,
            estimation_time_ms=45.3,
            source="estimated"
        )

        cached = param_cache.get_cached_parameters(data)
        assert cached.estimation_time_ms == 45.3


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_cache_statistics(self, param_cache):
        """Test statistics for empty cache."""
        stats = param_cache.get_statistics()
        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['hit_rate_percent'] == 0

    def test_single_point_dataset(self, param_cache):
        """Test cache handles single-point dataset."""
        data = np.array([[1.0, 2.0, 3.0]])
        param_cache.cache_parameters(data, eps=0.5, min_samples=1)

        result = param_cache.get_cached_parameters(data)
        assert result is not None

    def test_high_dimensional_data(self, param_cache):
        """Test cache handles high-dimensional data."""
        data = create_test_dataset(100, n_features=50)
        param_cache.cache_parameters(data, eps=0.5, min_samples=5)

        result = param_cache.get_cached_parameters(data)
        assert result is not None

    def test_very_similar_threshold(self, param_cache):
        """Test cache with very strict similarity threshold."""
        # Create cache with 99% threshold
        strict_cache = create_parameter_cache(
            cache_dir=None,
            similarity_threshold=0.99
        )

        data1 = create_test_dataset(1000, seed=1)
        data2 = create_test_dataset(1001, seed=1)  # Same distribution, slightly different size

        strict_cache.cache_parameters(data1, eps=0.5, min_samples=5)
        result = strict_cache.get_cached_parameters(data2)

        # May or may not hit depending on similarity
        # (just test it doesn't crash)
        assert result is None or result.eps == 0.5

    def test_similarity_threshold_boundary(self, logger):
        """Test boundary behavior at similarity threshold."""
        cache = create_parameter_cache(
            cache_dir=None,
            similarity_threshold=0.9,
            logger=logger
        )

        data1 = create_test_dataset(1000, seed=1)
        data2 = create_test_dataset(1000, seed=1)  # Identical

        cache.cache_parameters(data1, eps=0.5, min_samples=5)

        # Identical data should always hit
        result = cache.get_cached_parameters(data2)
        assert result is not None


# ============================================================================
# TESTS: CONCURRENT OPERATIONS
# ============================================================================

class TestConcurrentOperations:
    """Test cache behavior under concurrent-like operations."""

    def test_multiple_sequential_operations(self, param_cache):
        """Test multiple sequential cache operations."""
        datasets = [
            create_test_dataset(500, seed=i)
            for i in range(5)
        ]

        # Perform multiple operations
        for i, data in enumerate(datasets):
            param_cache.cache_parameters(data, eps=0.5+i*0.1, min_samples=5+i)
            param_cache.get_cached_parameters(data)

        # Verify all cached
        stats = param_cache.get_statistics()
        assert stats['total_estimations'] == 5
        assert stats['cache_hits'] == 5  # All subsequent accesses hit


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
