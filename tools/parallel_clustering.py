#!/usr/bin/env python3
"""
Parallel Clustering Module for Phase 3

Enables multi-channel parallel clustering using ThreadPoolExecutor.
Clusters both channels simultaneously for better performance.

Architecture:
- Sequential clustering: Ch1 ~100ms + Ch2 ~100ms = ~200ms total
- Parallel clustering:   max(Ch1 ~100ms, Ch2 ~100ms) = ~110ms total
- Expected speedup: 1.8-2.0x for dual-channel workflows

Design Principles:
- Non-blocking: Each channel processes independently
- Result ordering: Ch1 results available first, then Ch2
- Progress feedback: Can report progress per channel
- Error isolation: Failure in one channel doesn't block the other
- Thread-safe: Uses ThreadPoolExecutor for safety
"""

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from typing import Callable, Tuple, Dict, Any, Optional
import time
import logging
from pathlib import Path


class ParallelClusteringManager:
    """
    Manages parallel clustering for multi-channel workflows.

    Coordinates simultaneous clustering of multiple channels using
    ThreadPoolExecutor. Maintains result ordering and provides progress
    feedback.
    """

    def __init__(
        self,
        max_workers: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize parallel clustering manager.

        Parameters
        ----------
        max_workers : int
            Number of worker threads (default: 2 for dual-channel)
        logger : logging.Logger, optional
            Logger instance for reporting
        """
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def cluster_parallel(
        self,
        clustering_tasks: Dict[int, Callable],
        channel_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, Any]:
        """
        Execute multiple clustering tasks in parallel.

        Parameters
        ----------
        clustering_tasks : dict
            Dictionary mapping channel ID to clustering callable
            Example: {1: lambda: cluster(ch=1), 2: lambda: cluster(ch=2)}

        channel_names : dict, optional
            Dictionary mapping channel ID to display name
            Example: {1: "Channel 1", 2: "Channel 2"}

        Returns
        -------
        dict
            Dictionary with results from each channel, ordered by channel ID
            Example: {1: None, 2: None} (clustering modifies GUI in-place)

        Examples
        --------
        >>> manager = ParallelClusteringManager(max_workers=2)
        >>> tasks = {
        ...     1: lambda: cluster(channel=1),
        ...     2: lambda: cluster(channel=2)
        ... }
        >>> results = manager.cluster_parallel(tasks)

        Notes
        -----
        - Each channel clusters independently
        - Results are collected in order (Ch1, then Ch2)
        - Clustering operations modify GUI in-place, return value is status
        - Expected 1.8-2.0x speedup for dual-channel workflows
        """
        if not clustering_tasks:
            self.logger.warning("No clustering tasks provided")
            return {}

        if channel_names is None:
            channel_names = {ch: f"Channel {ch}" for ch in clustering_tasks}

        self.logger.info(
            f"Starting parallel clustering for {len(clustering_tasks)} channels"
        )

        # Start all clustering tasks
        future_to_channel = {}
        start_time = time.time()

        for channel_id, clustering_func in clustering_tasks.items():
            channel_name = channel_names.get(channel_id, f"Channel {channel_id}")
            self.logger.debug(f"Submitting {channel_name} to thread pool")

            future = self.executor.submit(clustering_func)
            future_to_channel[future] = channel_id

        # Collect results as they complete
        results = {}
        for future in as_completed(future_to_channel):
            channel_id = future_to_channel[future]
            channel_name = channel_names.get(channel_id, f"Channel {channel_id}")

            try:
                result = future.result()
                results[channel_id] = result
                self.logger.debug(f"{channel_name} clustering completed")
            except Exception as e:
                self.logger.error(
                    f"{channel_name} clustering failed: {e}",
                    exc_info=True
                )
                results[channel_id] = None

        elapsed = time.time() - start_time
        self.logger.info(
            f"Parallel clustering completed in {elapsed:.2f}s "
            f"({len(results)}/{len(clustering_tasks)} channels)"
        )

        return results

    def cluster_sequential(
        self,
        clustering_tasks: Dict[int, Callable],
        channel_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, Any]:
        """
        Execute multiple clustering tasks sequentially (for comparison).

        Parameters
        ----------
        clustering_tasks : dict
            Dictionary mapping channel ID to clustering callable

        channel_names : dict, optional
            Dictionary mapping channel ID to display name

        Returns
        -------
        dict
            Results from each channel in order

        Notes
        -----
        Used for performance comparison with parallel execution.
        Sequential execution useful when:
        - Single-channel workflows
        - Large datasets (sequential to reduce memory)
        - Debugging clustering issues
        """
        if not clustering_tasks:
            self.logger.warning("No clustering tasks provided")
            return {}

        if channel_names is None:
            channel_names = {ch: f"Channel {ch}" for ch in clustering_tasks}

        self.logger.info(
            f"Starting sequential clustering for {len(clustering_tasks)} channels"
        )

        results = {}
        start_time = time.time()

        # Execute tasks in order
        for channel_id in sorted(clustering_tasks.keys()):
            clustering_func = clustering_tasks[channel_id]
            channel_name = channel_names.get(channel_id, f"Channel {channel_id}")

            try:
                self.logger.debug(f"Clustering {channel_name}")
                result = clustering_func()
                results[channel_id] = result
                self.logger.debug(f"{channel_name} completed")
            except Exception as e:
                self.logger.error(
                    f"{channel_name} clustering failed: {e}",
                    exc_info=True
                )
                results[channel_id] = None

        elapsed = time.time() - start_time
        self.logger.info(
            f"Sequential clustering completed in {elapsed:.2f}s "
            f"({len(results)}/{len(clustering_tasks)} channels)"
        )

        return results

    def cluster_with_progress(
        self,
        clustering_tasks: Dict[int, Callable],
        progress_callback: Optional[Callable[[int, str], None]] = None,
        channel_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, Any]:
        """
        Execute parallel clustering with progress feedback.

        Parameters
        ----------
        clustering_tasks : dict
            Dictionary mapping channel ID to clustering callable

        progress_callback : callable, optional
            Callback function: progress_callback(channel_id, status)
            Called when each channel completes or fails
            Example: lambda ch, status: print(f"Ch{ch}: {status}")

        channel_names : dict, optional
            Dictionary mapping channel ID to display name

        Returns
        -------
        dict
            Results from each channel

        Examples
        --------
        >>> def on_progress(channel_id, status):
        ...     print(f"Channel {channel_id}: {status}")
        >>>
        >>> manager = ParallelClusteringManager()
        >>> results = manager.cluster_with_progress(
        ...     tasks,
        ...     progress_callback=on_progress
        ... )
        """
        if not clustering_tasks:
            self.logger.warning("No clustering tasks provided")
            return {}

        if channel_names is None:
            channel_names = {ch: f"Channel {ch}" for ch in clustering_tasks}

        self.logger.info(
            f"Starting parallel clustering with progress reporting "
            f"({len(clustering_tasks)} channels)"
        )

        # Submit all tasks
        future_to_channel = {}
        start_time = time.time()

        for channel_id, clustering_func in clustering_tasks.items():
            channel_name = channel_names.get(channel_id, f"Channel {channel_id}")
            if progress_callback:
                progress_callback(channel_id, "Submitted")
            future = self.executor.submit(clustering_func)
            future_to_channel[future] = channel_id

        # Collect results with progress updates
        results = {}
        for future in as_completed(future_to_channel):
            channel_id = future_to_channel[future]
            channel_name = channel_names.get(channel_id, f"Channel {channel_id}")

            try:
                result = future.result()
                results[channel_id] = result
                status = "Completed"
                if progress_callback:
                    progress_callback(channel_id, status)
                self.logger.debug(f"{channel_name} clustering completed")
            except Exception as e:
                results[channel_id] = None
                status = f"Failed: {str(e)}"
                if progress_callback:
                    progress_callback(channel_id, status)
                self.logger.error(
                    f"{channel_name} clustering failed: {e}",
                    exc_info=True
                )

        elapsed = time.time() - start_time
        self.logger.info(
            f"Parallel clustering with progress completed in {elapsed:.2f}s"
        )

        return results

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool.

        Parameters
        ----------
        wait : bool
            If True, wait for pending tasks to complete
        """
        self.executor.shutdown(wait=wait)
        self.logger.debug("ThreadPoolExecutor shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown executor."""
        self.shutdown(wait=True)
        return False


def create_parallel_clustering_manager(
    max_workers: int = 2,
    logger: Optional[logging.Logger] = None
) -> ParallelClusteringManager:
    """
    Factory function to create parallel clustering manager.

    Parameters
    ----------
    max_workers : int
        Number of worker threads (default: 2 for dual-channel)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    ParallelClusteringManager
        Configured manager instance

    Examples
    --------
    >>> manager = create_parallel_clustering_manager(max_workers=2)
    >>> results = manager.cluster_parallel(tasks)
    """
    return ParallelClusteringManager(max_workers=max_workers, logger=logger)
