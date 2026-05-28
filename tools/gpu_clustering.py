#!/usr/bin/env python3
"""
GPU-Accelerated Clustering Module

Provides GPU acceleration for HDBSCAN clustering using RAPIDS cuML.
Automatically detects GPU availability and falls back to CPU if needed.

Key Features:
- Automatic GPU detection (NVIDIA CUDA)
- RAPIDS HDBSCAN for 10-100x speedup
- Graceful fallback to CPU HDBSCAN
- Performance metrics tracking
- Transparent integration with Phase 2
"""

import logging
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray
import numpy as np


class GPUDetectionResult:
    """Result of GPU detection check."""

    def __init__(
        self,
        gpu_available: bool,
        gpu_name: Optional[str] = None,
        cuda_version: Optional[str] = None,
        gpu_memory_mb: Optional[float] = None,
        error_message: Optional[str] = None
    ):
        self.gpu_available = gpu_available
        self.gpu_name = gpu_name
        self.cuda_version = cuda_version
        self.gpu_memory_mb = gpu_memory_mb
        self.error_message = error_message

    def __repr__(self) -> str:
        if self.gpu_available:
            return (
                f"GPUDetectionResult(available=True, "
                f"name={self.gpu_name}, memory={self.gpu_memory_mb}MB)"
            )
        else:
            return f"GPUDetectionResult(available=False, error={self.error_message})"


class GPUClusteringManager:
    """
    Manager for GPU-accelerated clustering.

    Handles detection, initialization, and execution of GPU-based clustering
    with automatic fallback to CPU when GPU is unavailable.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize GPU clustering manager.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for reporting GPU status
        """
        self.logger = logger or logging.getLogger(__name__)
        self._gpu_available = False
        self._rapids_available = False
        self._cuml = None
        self._gpu_info: Optional[GPUDetectionResult] = None

        # Detect GPU on initialization
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect GPU availability and load RAPIDS if available."""
        try:
            # Try importing RAPIDS cuML
            import cuml  # type: ignore
            self._cuml = cuml
            self._rapids_available = True

            # Try to detect CUDA GPU
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                if device_count > 0:
                    # Get info on first GPU
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 ** 2)

                    self._gpu_available = True
                    self._gpu_info = GPUDetectionResult(
                        gpu_available=True,
                        gpu_name=gpu_name,
                        gpu_memory_mb=gpu_memory
                    )

                    self.logger.info(
                        f"GPU detected: {gpu_name} ({gpu_memory:.0f}MB). "
                        f"RAPIDS HDBSCAN available for 10-100x speedup."
                    )
                else:
                    self._gpu_info = GPUDetectionResult(
                        gpu_available=False,
                        error_message="No NVIDIA GPUs detected"
                    )
                    self.logger.debug("No NVIDIA GPUs detected on system")

            except ImportError:
                # pynvml not available, try direct GPU test
                try:
                    import cupy  # type: ignore
                    # If cupy imports successfully, GPU likely available
                    self._gpu_available = True
                    self._gpu_info = GPUDetectionResult(
                        gpu_available=True,
                        gpu_name="NVIDIA GPU (detected via cupy)",
                        gpu_memory_mb=None
                    )
                    self.logger.info(
                        "GPU detected via cupy. RAPIDS HDBSCAN available."
                    )
                except ImportError:
                    self._gpu_info = GPUDetectionResult(
                        gpu_available=False,
                        error_message="Could not initialize GPU libraries"
                    )
                    self.logger.debug("GPU detection libraries not available")

        except ImportError:
            self._rapids_available = False
            self._gpu_info = GPUDetectionResult(
                gpu_available=False,
                error_message="RAPIDS cuML not installed"
            )
            self.logger.debug(
                "RAPIDS cuML not installed. Install with: "
                "pip install cuml (requires GPU)"
            )

    @property
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_available and self._rapids_available

    @property
    def gpu_info(self) -> GPUDetectionResult:
        """Get GPU detection information."""
        return self._gpu_info or GPUDetectionResult(
            gpu_available=False,
            error_message="GPU detection not performed"
        )

    def cluster_gpu(
        self,
        data: NDArray,
        min_samples: int = 5,
        min_cluster_size: int = 5
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """
        Perform GPU-accelerated HDBSCAN clustering.

        Parameters
        ----------
        data : np.ndarray
            Input data (n_samples, n_features)
        min_samples : int
            Minimum samples parameter for HDBSCAN
        min_cluster_size : int
            Minimum cluster size for HDBSCAN

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (-1 for noise)
        stats : dict
            Statistics including execution time and GPU info

        Raises
        ------
        RuntimeError
            If GPU is not available
        """
        if not self.is_available:
            raise RuntimeError(
                "GPU acceleration not available. "
                "Install RAPIDS: pip install cuml"
            )

        import time
        start_time = time.time()

        # Convert to GPU array
        try:
            import cupy as cp  # type: ignore
            gpu_data = cp.asarray(data)
        except ImportError:
            gpu_data = data

        # Create GPU HDBSCAN
        hdbscan_gpu = self._cuml.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size
        )

        # Fit and predict
        hdbscan_gpu.fit(gpu_data)
        labels = np.asarray(hdbscan_gpu.labels_)

        elapsed_ms = (time.time() - start_time) * 1000

        # Compute statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        stats = {
            'gpu_used': True,
            'gpu_name': self._gpu_info.gpu_name if self._gpu_info else 'Unknown',
            'execution_time_ms': elapsed_ms,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': (n_noise / len(labels) * 100) if len(labels) > 0 else 0
        }

        self.logger.debug(
            f"GPU HDBSCAN: {elapsed_ms:.1f}ms, "
            f"{n_clusters} clusters, {n_noise} noise points"
        )

        return labels, stats

    def cluster_cpu(
        self,
        data: NDArray,
        min_samples: int = 5,
        min_cluster_size: int = 5
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """
        Perform CPU HDBSCAN clustering (fallback).

        Parameters
        ----------
        data : np.ndarray
            Input data (n_samples, n_features)
        min_samples : int
            Minimum samples parameter for HDBSCAN
        min_cluster_size : int
            Minimum cluster size for HDBSCAN

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (-1 for noise)
        stats : dict
            Statistics including execution time
        """
        import time
        import hdbscan

        start_time = time.time()

        # Create CPU HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size
        )

        # Fit and predict
        labels = clusterer.fit_predict(data)

        elapsed_ms = (time.time() - start_time) * 1000

        # Compute statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        stats = {
            'gpu_used': False,
            'gpu_name': None,
            'execution_time_ms': elapsed_ms,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': (n_noise / len(labels) * 100) if len(labels) > 0 else 0
        }

        self.logger.debug(
            f"CPU HDBSCAN: {elapsed_ms:.1f}ms, "
            f"{n_clusters} clusters, {n_noise} noise points"
        )

        return labels, stats

    def cluster_adaptive(
        self,
        data: NDArray,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        prefer_gpu: bool = True
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """
        Perform clustering with adaptive GPU/CPU selection.

        Parameters
        ----------
        data : np.ndarray
            Input data (n_samples, n_features)
        min_samples : int
            Minimum samples parameter for HDBSCAN
        min_cluster_size : int
            Minimum cluster size for HDBSCAN
        prefer_gpu : bool
            If True, use GPU when available; if False, force CPU

        Returns
        -------
        labels : np.ndarray
            Cluster assignments (-1 for noise)
        stats : dict
            Statistics including which device was used
        """
        # Use GPU if available and preferred
        if prefer_gpu and self.is_available:
            try:
                return self.cluster_gpu(data, min_samples, min_cluster_size)
            except Exception as e:
                self.logger.warning(
                    f"GPU clustering failed: {e}. Falling back to CPU."
                )
                return self.cluster_cpu(data, min_samples, min_cluster_size)
        else:
            return self.cluster_cpu(data, min_samples, min_cluster_size)


def create_gpu_clustering_manager(
    logger: Optional[logging.Logger] = None
) -> GPUClusteringManager:
    """
    Factory function to create GPU clustering manager.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    GPUClusteringManager
        Initialized GPU clustering manager
    """
    return GPUClusteringManager(logger=logger)
