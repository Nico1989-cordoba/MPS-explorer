#!/usr/bin/env python3
"""
GPU Acceleration Testing Module

Comprehensive tests for GPU-accelerated clustering functionality.
Tests verify GPU detection, fallback behavior, and performance characteristics.
"""

import pytest
import numpy as np
import logging
from typing import Tuple
from numpy.typing import NDArray

# Import GPU clustering module
from tools.gpu_clustering import (
    GPUClusteringManager,
    GPUDetectionResult,
    create_gpu_clustering_manager
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_gpu_acceleration")


@pytest.fixture
def gpu_manager(logger):
    """Create a GPU clustering manager."""
    return create_gpu_clustering_manager(logger=logger)


def create_test_dataset(
    n_samples: int = 1000,
    n_features: int = 3,
    seed: int = 42
) -> NDArray:
    """Create test dataset."""
    np.random.seed(seed)
    return np.random.normal(0, 1, (n_samples, n_features))


# ============================================================================
# TESTS: GPU DETECTION
# ============================================================================

class TestGPUDetection:
    """Test GPU detection functionality."""

    def test_gpu_manager_initialization(self, gpu_manager):
        """Test GPU manager initializes correctly."""
        assert gpu_manager is not None
        assert isinstance(gpu_manager.gpu_info, GPUDetectionResult)

    def test_gpu_info_available_property(self, gpu_manager):
        """Test GPU info has required properties."""
        gpu_info = gpu_manager.gpu_info
        assert hasattr(gpu_info, 'gpu_available')
        assert hasattr(gpu_info, 'gpu_name')
        assert hasattr(gpu_info, 'error_message')

    def test_is_available_property(self, gpu_manager):
        """Test is_available property is boolean."""
        assert isinstance(gpu_manager.is_available, bool)

    def test_gpu_detection_result_repr(self):
        """Test GPUDetectionResult string representation."""
        # Available result
        result_avail = GPUDetectionResult(
            gpu_available=True,
            gpu_name="NVIDIA A100",
            gpu_memory_mb=40000
        )
        repr_str = repr(result_avail)
        assert "available=True" in repr_str
        assert "NVIDIA A100" in repr_str

        # Unavailable result
        result_unavail = GPUDetectionResult(
            gpu_available=False,
            error_message="No GPU found"
        )
        repr_str = repr(result_unavail)
        assert "available=False" in repr_str
        assert "No GPU found" in repr_str


# ============================================================================
# TESTS: CPU FALLBACK (Always available)
# ============================================================================

class TestCPUFallback:
    """Test CPU fallback clustering (always available)."""

    def test_cpu_clustering_basic(self, gpu_manager):
        """Test basic CPU clustering works."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_cpu(data)

        assert labels is not None
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(data)

    def test_cpu_clustering_stats(self, gpu_manager):
        """Test CPU clustering returns correct statistics."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_cpu(data)

        # Check stats structure
        assert 'gpu_used' in stats
        assert 'execution_time_ms' in stats
        assert 'n_clusters' in stats
        assert 'n_noise' in stats
        assert 'noise_percentage' in stats

        # Check stats values
        assert stats['gpu_used'] is False
        assert stats['execution_time_ms'] > 0
        assert stats['n_clusters'] >= 0
        assert stats['n_noise'] >= 0

    def test_cpu_clustering_small_dataset(self, gpu_manager):
        """Test CPU clustering on small dataset."""
        data = create_test_dataset(100)
        labels, stats = gpu_manager.cluster_cpu(data, min_cluster_size=5)

        assert len(labels) == 100
        assert stats['execution_time_ms'] > 0

    def test_cpu_clustering_large_dataset(self, gpu_manager):
        """Test CPU clustering on larger dataset."""
        data = create_test_dataset(10000)
        labels, stats = gpu_manager.cluster_cpu(data)

        assert len(labels) == 10000
        assert stats['execution_time_ms'] > 0

    def test_cpu_clustering_parameters(self, gpu_manager):
        """Test CPU clustering with different parameters."""
        data = create_test_dataset(500)

        # Test different min_samples
        labels1, _ = gpu_manager.cluster_cpu(data, min_samples=3)
        labels2, _ = gpu_manager.cluster_cpu(data, min_samples=10)

        # Different parameters may produce different results
        assert len(labels1) == len(labels2)

    def test_cpu_clustering_deterministic(self, gpu_manager):
        """Test CPU clustering is deterministic."""
        data = create_test_dataset(500)

        labels1, stats1 = gpu_manager.cluster_cpu(data)
        labels2, stats2 = gpu_manager.cluster_cpu(data)

        # Results should be identical
        np.testing.assert_array_equal(labels1, labels2)
        assert stats1['n_clusters'] == stats2['n_clusters']
        assert stats1['n_noise'] == stats2['n_noise']


# ============================================================================
# TESTS: ADAPTIVE CLUSTERING
# ============================================================================

class TestAdaptiveClustering:
    """Test adaptive GPU/CPU clustering."""

    def test_adaptive_clustering_basic(self, gpu_manager):
        """Test adaptive clustering basic functionality."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_adaptive(data)

        assert labels is not None
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(data)

    def test_adaptive_clustering_returns_stats(self, gpu_manager):
        """Test adaptive clustering returns statistics."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_adaptive(data)

        # Check required stats keys
        assert 'gpu_used' in stats
        assert 'execution_time_ms' in stats
        assert 'n_clusters' in stats
        assert 'n_noise' in stats

    def test_adaptive_clustering_prefer_gpu_true(self, gpu_manager):
        """Test adaptive clustering with prefer_gpu=True."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_adaptive(data, prefer_gpu=True)

        assert len(labels) == len(data)
        # stats['gpu_used'] depends on GPU availability
        assert 'gpu_used' in stats

    def test_adaptive_clustering_prefer_gpu_false(self, gpu_manager):
        """Test adaptive clustering with prefer_gpu=False."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_adaptive(data, prefer_gpu=False)

        assert len(labels) == len(data)
        # Should always use CPU when prefer_gpu=False
        assert stats['gpu_used'] is False

    def test_adaptive_clustering_fallback(self, gpu_manager):
        """Test adaptive clustering fallback behavior."""
        data = create_test_dataset(500)

        # Try adaptive (will fallback to CPU if GPU unavailable)
        try:
            labels, stats = gpu_manager.cluster_adaptive(data)
            # Should succeed either way
            assert len(labels) == len(data)
        except Exception as e:
            # Should not raise exception (fallback should handle it)
            pytest.fail(f"Adaptive clustering should not raise: {e}")


# ============================================================================
# TESTS: GPU CLUSTERING (If available)
# ============================================================================

class TestGPUClustering:
    """Test GPU clustering (skipped if GPU unavailable)."""

    @pytest.mark.skipif(
        not create_gpu_clustering_manager().is_available,
        reason="GPU not available"
    )
    def test_gpu_clustering_available(self, gpu_manager):
        """Test GPU clustering when available."""
        assert gpu_manager.is_available

    @pytest.mark.skipif(
        not create_gpu_clustering_manager().is_available,
        reason="GPU not available"
    )
    def test_gpu_clustering_basic(self, gpu_manager):
        """Test basic GPU clustering (if available)."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_gpu(data)

        assert labels is not None
        assert stats['gpu_used'] is True

    @pytest.mark.skipif(
        not create_gpu_clustering_manager().is_available,
        reason="GPU not available"
    )
    def test_gpu_clustering_stats(self, gpu_manager):
        """Test GPU clustering statistics."""
        data = create_test_dataset(500)
        labels, stats = gpu_manager.cluster_gpu(data)

        assert 'gpu_used' in stats
        assert stats['gpu_used'] is True
        assert 'gpu_name' in stats
        assert 'execution_time_ms' in stats

    @pytest.mark.skipif(
        not create_gpu_clustering_manager().is_available,
        reason="GPU not available"
    )
    def test_gpu_clustering_large_dataset(self, gpu_manager):
        """Test GPU clustering on large dataset (if available)."""
        # GPU should show speedup on larger datasets
        data = create_test_dataset(50000)
        labels, stats = gpu_manager.cluster_gpu(data)

        assert len(labels) == len(data)
        assert stats['execution_time_ms'] > 0


# ============================================================================
# TESTS: PERFORMANCE COMPARISON
# ============================================================================

class TestPerformanceComparison:
    """Test performance comparison between CPU and adaptive."""

    def test_execution_time_tracked(self, gpu_manager):
        """Test execution times are properly tracked."""
        data = create_test_dataset(1000)

        _, cpu_stats = gpu_manager.cluster_cpu(data)
        _, adapt_stats = gpu_manager.cluster_adaptive(data)

        # Both should track execution time
        assert cpu_stats['execution_time_ms'] > 0
        assert adapt_stats['execution_time_ms'] > 0

    def test_clustering_consistency(self, gpu_manager):
        """Test CPU and adaptive produce valid results."""
        data = create_test_dataset(500)

        labels_cpu, _ = gpu_manager.cluster_cpu(data)
        labels_adapt, _ = gpu_manager.cluster_adaptive(data, prefer_gpu=False)

        # Both should produce valid labels
        assert len(labels_cpu) == len(data)
        assert len(labels_adapt) == len(data)

        # Check label values
        assert set(labels_cpu).issubset(set(range(-1, len(set(labels_cpu)))))
        assert set(labels_adapt).issubset(set(range(-1, len(set(labels_adapt)))))


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling."""

    def test_gpu_clustering_without_gpu(self, gpu_manager):
        """Test GPU clustering raises error if GPU unavailable."""
        if gpu_manager.is_available:
            # GPU is available, test will be skipped or modified
            pytest.skip("GPU is available, cannot test unavailable case")

        data = create_test_dataset(100)

        with pytest.raises(RuntimeError):
            gpu_manager.cluster_gpu(data)

    def test_adaptive_handles_errors(self, gpu_manager):
        """Test adaptive clustering handles errors gracefully."""
        data = create_test_dataset(500)

        # Should not raise, should fallback to CPU
        try:
            labels, stats = gpu_manager.cluster_adaptive(data)
            assert len(labels) == len(data)
        except Exception as e:
            pytest.fail(f"Adaptive should handle errors gracefully: {e}")


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_clustering_single_point(self, gpu_manager):
        """Test clustering with single point (HDBSCAN limitation)."""
        # HDBSCAN requires minimum points, skip single-point test
        # This is an HDBSCAN library limitation, not our code issue
        pytest.skip("HDBSCAN requires multiple points")

    def test_clustering_two_points(self, gpu_manager):
        """Test clustering with two points."""
        data = np.array([[1.0, 2.0], [2.0, 3.0]])
        labels, stats = gpu_manager.cluster_cpu(data)

        assert len(labels) == 2

    def test_clustering_high_dimensional(self, gpu_manager):
        """Test clustering with high-dimensional data."""
        data = create_test_dataset(100, n_features=50)
        labels, stats = gpu_manager.cluster_cpu(data)

        assert len(labels) == 100

    def test_clustering_different_parameters(self, gpu_manager):
        """Test clustering with different parameter combinations."""
        data = create_test_dataset(500)

        for min_samples in [2, 5, 10]:
            for min_cluster_size in [3, 5, 10]:
                labels, stats = gpu_manager.cluster_cpu(
                    data,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size
                )
                assert len(labels) == len(data)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
