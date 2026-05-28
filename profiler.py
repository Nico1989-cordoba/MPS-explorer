#!/usr/bin/env python3
"""
Performance Profiler for MPS Explorer

This module provides comprehensive profiling utilities to identify performance
bottlenecks in DBSCAN clustering and data processing operations.

Features:
  - Method-level timing with detailed statistics
  - Memory profiling for array operations
  - Complexity analysis (scalability with dataset size)
  - Detailed performance reports
  - Real-time performance monitoring

Usage:
    from profiler import PerformanceProfiler

    profiler = PerformanceProfiler()
    profiler.start_timer("operation_name")
    # ... do work ...
    elapsed = profiler.stop_timer("operation_name")

    report = profiler.generate_report()
    print(report)
"""

import time
import tracemalloc
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TimingStats:
    """Statistics for a single timed operation."""
    operation_name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    @property
    def mean_time(self) -> float:
        """Average execution time in milliseconds."""
        return (self.total_time / self.count * 1000) if self.count > 0 else 0

    @property
    def std_dev(self) -> float:
        """Standard deviation of execution times in milliseconds."""
        if self.count < 2:
            return 0.0
        mean = self.mean_time / 1000
        variance = sum((t - mean) ** 2 for t in self.times) / (self.count - 1)
        return np.sqrt(variance) * 1000

    @property
    def min_time_ms(self) -> float:
        """Minimum execution time in milliseconds."""
        return self.min_time * 1000 if self.min_time != float('inf') else 0

    @property
    def max_time_ms(self) -> float:
        """Maximum execution time in milliseconds."""
        return self.max_time * 1000


@dataclass
class MemoryStats:
    """Statistics for memory usage during operations."""
    operation_name: str
    peak_memory: float = 0.0  # bytes
    total_memory: float = 0.0  # bytes
    allocations: int = 0

    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage in MB."""
        return self.peak_memory / (1024 ** 2)

    @property
    def total_memory_mb(self) -> float:
        """Total memory allocated in MB."""
        return self.total_memory / (1024 ** 2)


class PerformanceProfiler:
    """
    Comprehensive performance profiler for MPS Explorer operations.

    Tracks timing, memory usage, and provides scalability analysis
    for DBSCAN clustering and data processing operations.
    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        self.timing_stats: Dict[str, TimingStats] = {}
        self.memory_stats: Dict[str, MemoryStats] = {}
        self.active_timers: Dict[str, float] = {}
        self.active_memory: Dict[str, Tuple[int, int]] = {}
        self.operation_sizes: Dict[str, List[int]] = {}
        self.logger: Optional[Any] = None

    def set_logger(self, logger: Any) -> None:
        """Set a logger instance for reporting."""
        self.logger = logger

    def start_timer(self, operation_name: str) -> None:
        """
        Start timing an operation.

        Parameters
        ----------
        operation_name : str
            Name of the operation to profile.
        """
        self.active_timers[operation_name] = time.perf_counter()
        tracemalloc.start()

    def stop_timer(self, operation_name: str, dataset_size: Optional[int] = None) -> float:
        """
        Stop timing an operation and record statistics.

        Parameters
        ----------
        operation_name : str
            Name of the operation being profiled.
        dataset_size : int, optional
            Size of dataset processed (for scalability analysis).

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if operation_name not in self.active_timers:
            return 0.0

        elapsed = time.perf_counter() - self.active_timers[operation_name]

        # Record timing statistics
        if operation_name not in self.timing_stats:
            self.timing_stats[operation_name] = TimingStats(operation_name)

        stats = self.timing_stats[operation_name]
        stats.count += 1
        stats.total_time += elapsed
        stats.min_time = min(stats.min_time, elapsed)
        stats.max_time = max(stats.max_time, elapsed)
        stats.times.append(elapsed)

        # Record dataset size for scalability analysis
        if dataset_size is not None:
            if operation_name not in self.operation_sizes:
                self.operation_sizes[operation_name] = []
            self.operation_sizes[operation_name].append(dataset_size)

        # Record memory statistics
        current, peak = tracemalloc.get_traced_memory()
        if operation_name not in self.memory_stats:
            self.memory_stats[operation_name] = MemoryStats(operation_name)

        mem_stats = self.memory_stats[operation_name]
        mem_stats.peak_memory = max(mem_stats.peak_memory, peak)
        mem_stats.total_memory += current
        mem_stats.allocations += 1

        tracemalloc.stop()

        # Log if logger is available
        if self.logger:
            self.logger.debug(
                f"Profiler: {operation_name} completed in {elapsed*1000:.2f}ms "
                f"(memory: {peak/(1024**2):.1f}MB)"
            )

        del self.active_timers[operation_name]
        return elapsed

    def get_statistics(self, operation_name: str) -> Optional[TimingStats]:
        """
        Get timing statistics for an operation.

        Parameters
        ----------
        operation_name : str
            Name of the operation.

        Returns
        -------
        TimingStats or None
            Statistics object if operation was profiled, None otherwise.
        """
        return self.timing_stats.get(operation_name)

    def estimate_complexity(self, operation_name: str) -> Optional[str]:
        """
        Estimate computational complexity from execution data.

        Analyzes how execution time scales with dataset size to determine
        if operation has O(n), O(n*log(n)), O(n^2) behavior, etc.

        Parameters
        ----------
        operation_name : str
            Name of the operation.

        Returns
        -------
        str or None
            Estimated complexity (e.g., "O(n)", "O(n*log(n))", "O(n^2)")
            or None if insufficient data.
        """
        if operation_name not in self.operation_sizes:
            return None

        sizes = self.operation_sizes[operation_name]
        times = self.timing_stats[operation_name].times

        if len(sizes) < 2 or len(times) < 2:
            return None

        # Fit to different complexity models
        sizes_arr = np.array(sizes, dtype=float)
        times_arr = np.array(times, dtype=float)

        # Normalize for comparison
        sizes_norm = sizes_arr / sizes_arr.min()
        times_norm = times_arr / times_arr.min()

        # Test different models
        models = {
            "O(1)": lambda n: np.ones_like(n),
            "O(log n)": lambda n: np.log(n),
            "O(n)": lambda n: n,
            "O(n*log n)": lambda n: n * np.log(n),
            "O(n^2)": lambda n: n ** 2,
            "O(n^3)": lambda n: n ** 3,
        }

        best_model = None
        best_r2 = -np.inf

        for model_name, model_func in models.items():
            try:
                predicted = model_func(sizes_norm)
                # Calculate R² (coefficient of determination)
                ss_res = np.sum((times_norm - predicted) ** 2)
                ss_tot = np.sum((times_norm - times_norm.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model_name
            except (ValueError, ZeroDivisionError):
                continue

        return best_model if best_r2 > 0.8 else None

    def generate_report(self, title: str = "Performance Profiling Report") -> str:
        """
        Generate a comprehensive performance report.

        Parameters
        ----------
        title : str
            Title for the report.

        Returns
        -------
        str
            Formatted performance report.
        """
        report = []
        report.append("\n" + "="*80)
        report.append(f"{title:^80}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)

        if not self.timing_stats:
            report.append("\nNo profiling data collected.")
            report.append("="*80 + "\n")
            return "\n".join(report)

        # Timing Summary
        report.append("\n" + "-"*80)
        report.append("EXECUTION TIME SUMMARY")
        report.append("-"*80)
        report.append(f"{'Operation':<30} {'Count':>8} {'Mean (ms)':>12} {'Std Dev':>12} {'Total (ms)':>12}")
        report.append("-"*80)

        total_all = 0
        for op_name, stats in sorted(self.timing_stats.items()):
            report.append(
                f"{op_name:<30} {stats.count:>8d} {stats.mean_time:>12.2f} "
                f"{stats.std_dev:>12.2f} {stats.total_time*1000:>12.2f}"
            )
            total_all += stats.total_time * 1000

        report.append("-"*80)
        report.append(f"{'TOTAL':<30} {sum(s.count for s in self.timing_stats.values()):>8d} "
                     f"{'':<12} {'':<12} {total_all:>12.2f}")
        report.append("="*80)

        # Memory Summary
        if self.memory_stats:
            report.append("\n" + "-"*80)
            report.append("MEMORY USAGE SUMMARY")
            report.append("-"*80)
            report.append(f"{'Operation':<30} {'Peak (MB)':>15} {'Allocations':>12}")
            report.append("-"*80)

            for op_name, stats in sorted(self.memory_stats.items()):
                report.append(
                    f"{op_name:<30} {stats.peak_memory_mb:>15.2f} {stats.allocations:>12d}"
                )
            report.append("="*80)

        # Complexity Analysis
        report.append("\n" + "-"*80)
        report.append("COMPUTATIONAL COMPLEXITY ANALYSIS")
        report.append("-"*80)
        report.append(f"{'Operation':<30} {'Estimated Complexity':>30}")
        report.append("-"*80)

        has_complexity = False
        for op_name in sorted(self.timing_stats.keys()):
            complexity = self.estimate_complexity(op_name)
            if complexity:
                report.append(f"{op_name:<30} {complexity:>30}")
                has_complexity = True

        if not has_complexity:
            report.append("(Insufficient data for complexity analysis - run with variable dataset sizes)")
        report.append("="*80)

        # Bottleneck Analysis
        if self.timing_stats:
            report.append("\n" + "-"*80)
            report.append("BOTTLENECK ANALYSIS (Top 5 Most Time-Consuming Operations)")
            report.append("-"*80)

            sorted_ops = sorted(
                self.timing_stats.items(),
                key=lambda x: x[1].total_time,
                reverse=True
            )

            total_time = sum(s.total_time for s in self.timing_stats.values())

            for i, (op_name, stats) in enumerate(sorted_ops[:5], 1):
                percentage = (stats.total_time / total_time * 100) if total_time > 0 else 0
                report.append(
                    f"{i}. {op_name:<26} {stats.total_time*1000:>10.2f}ms "
                    f"({percentage:>6.1f}% of total time)"
                )

            report.append("="*80)

        report.append("\n")
        return "\n".join(report)

    def generate_json_report(self) -> str:
        """
        Generate a JSON report of profiling data.

        Returns
        -------
        str
            JSON-formatted profiling report.
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "timing": {},
            "memory": {},
            "complexity": {}
        }

        for op_name, stats in self.timing_stats.items():
            data["timing"][op_name] = {
                "count": stats.count,
                "mean_ms": stats.mean_time,
                "min_ms": stats.min_time_ms,
                "max_ms": stats.max_time_ms,
                "std_dev_ms": stats.std_dev,
                "total_ms": stats.total_time * 1000
            }

        for op_name, stats in self.memory_stats.items():
            data["memory"][op_name] = {
                "peak_mb": stats.peak_memory_mb,
                "total_mb": stats.total_memory_mb,
                "allocations": stats.allocations
            }

        for op_name in self.timing_stats:
            complexity = self.estimate_complexity(op_name)
            if complexity:
                data["complexity"][op_name] = complexity

        return json.dumps(data, indent=2)

    def clear(self) -> None:
        """Clear all profiling data."""
        self.timing_stats.clear()
        self.memory_stats.clear()
        self.active_timers.clear()
        self.active_memory.clear()
        self.operation_sizes.clear()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_operation(operation_name: str):
    """
    Decorator for profiling function execution.

    Usage:
        @profile_operation("cluster_operation")
        def cluster_data(data):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            profiler.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Try to extract dataset size from args
                dataset_size = None
                if args and hasattr(args[0], 'shape'):
                    dataset_size = len(args[0])
                profiler.stop_timer(operation_name, dataset_size)
        return wrapper
    return decorator
