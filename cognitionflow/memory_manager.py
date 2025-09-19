"""Memory management and performance optimizations for the SDK."""

import gc
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available, memory monitoring will be limited")


class MemoryManager:
    """Manages memory usage and garbage collection for the SDK.

    This class monitors memory usage and triggers garbage collection
    when memory usage exceeds configured thresholds.

    Features:
    - Automatic memory monitoring
    - Configurable memory limits
    - Periodic garbage collection
    - Memory statistics reporting
    """

    def __init__(
        self,
        max_memory_mb: int = 100,
        gc_threshold_mb: int = 50,
        check_interval: float = 30.0,
        auto_gc: bool = True
    ):
        """Initialize the memory manager.

        Args:
            max_memory_mb: Maximum memory usage in MB before warnings
            gc_threshold_mb: Memory threshold in MB to trigger garbage collection
            check_interval: Interval in seconds between memory checks
            auto_gc: Whether to automatically trigger garbage collection
        """
        self.max_memory_mb = max_memory_mb
        self.gc_threshold_mb = gc_threshold_mb
        self.check_interval = check_interval
        self.auto_gc = auto_gc

        self._lock = threading.Lock()
        self._last_check = 0.0
        self._peak_memory_mb = 0.0
        self._gc_count = 0
        self._warnings_issued = 0

        # Set up garbage collection thresholds if auto_gc is enabled
        if self.auto_gc:
            self._setup_gc_thresholds()

    def _setup_gc_thresholds(self):
        """Set up garbage collection thresholds for better performance."""
        # Get current thresholds
        current_thresholds = gc.get_threshold()

        # Increase thresholds to reduce GC frequency for better performance
        # These values can be tuned based on application needs
        new_thresholds = (
            current_thresholds[0] * 2,  # Generation 0
            current_thresholds[1] * 2,  # Generation 1
            current_thresholds[2] * 2   # Generation 2
        )

        gc.set_threshold(*new_thresholds)
        logger.debug(f"Set GC thresholds from {current_thresholds} to {new_thresholds}")

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in MB, or 0.0 if cannot be determined
        """
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                memory_bytes = process.memory_info().rss
                return memory_bytes / 1024 / 1024
            except Exception as e:
                logger.debug(f"Failed to get memory usage with psutil: {e}")

        # Fallback: use gc module statistics (less accurate)
        try:
            stats = gc.get_stats()
            total_objects = sum(stat['collections'] for stat in stats)
            # Very rough estimate: assume 100 bytes per object on average
            estimated_mb = total_objects * 100 / 1024 / 1024
            return estimated_mb
        except Exception:
            return 0.0

    def check_memory(self) -> dict:
        """Check current memory usage and take action if needed.

        Returns:
            Dictionary with memory statistics
        """
        current_time = time.time()

        with self._lock:
            # Only check if enough time has passed
            if current_time - self._last_check < self.check_interval:
                return self._get_stats()

            self._last_check = current_time

        # Get current memory usage
        current_memory = self.get_memory_usage()

        with self._lock:
            # Update peak memory
            if current_memory > self._peak_memory_mb:
                self._peak_memory_mb = current_memory

            # Check if we should trigger garbage collection
            if self.auto_gc and current_memory > self.gc_threshold_mb:
                self._trigger_gc()

            # Issue warnings for high memory usage
            if current_memory > self.max_memory_mb:
                self._warnings_issued += 1
                if self._warnings_issued <= 5:  # Limit warning spam
                    logger.warning(
                        f"Memory usage ({current_memory:.1f} MB) exceeds limit ({self.max_memory_mb} MB)"
                    )

        return self._get_stats()

    def _trigger_gc(self):
        """Trigger garbage collection and log results."""
        logger.debug("Triggering garbage collection due to memory threshold")

        memory_before = self.get_memory_usage()

        # Force garbage collection
        collected = gc.collect()

        memory_after = self.get_memory_usage()
        memory_freed = max(0, memory_before - memory_after)

        with self._lock:
            self._gc_count += 1

        logger.debug(
            f"GC collected {collected} objects, freed {memory_freed:.1f} MB "
            f"({memory_before:.1f} -> {memory_after:.1f} MB)"
        )

    def force_gc(self) -> dict:
        """Force garbage collection and return statistics.

        Returns:
            Dictionary with garbage collection results
        """
        memory_before = self.get_memory_usage()

        # Run garbage collection for all generations
        collected = []
        for generation in range(3):
            gen_collected = gc.collect(generation)
            collected.append(gen_collected)

        memory_after = self.get_memory_usage()
        memory_freed = max(0, memory_before - memory_after)

        with self._lock:
            self._gc_count += 1

        result = {
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_freed_mb": memory_freed,
            "objects_collected": sum(collected),
            "objects_by_generation": collected
        }

        logger.info(f"Manual GC freed {memory_freed:.1f} MB, collected {sum(collected)} objects")
        return result

    def _get_stats(self) -> dict:
        """Get memory manager statistics.

        Returns:
            Dictionary containing current statistics
        """
        current_memory = self.get_memory_usage()

        with self._lock:
            return {
                "current_memory_mb": current_memory,
                "peak_memory_mb": self._peak_memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "gc_threshold_mb": self.gc_threshold_mb,
                "gc_count": self._gc_count,
                "warnings_issued": self._warnings_issued,
                "last_check": self._last_check,
                "auto_gc_enabled": self.auto_gc,
                "has_psutil": HAS_PSUTIL
            }

    def get_stats(self) -> dict:
        """Get current memory statistics.

        Returns:
            Dictionary containing memory statistics
        """
        return self.check_memory()

    def reset_stats(self):
        """Reset memory statistics."""
        with self._lock:
            self._peak_memory_mb = self.get_memory_usage()
            self._gc_count = 0
            self._warnings_issued = 0
            self._last_check = 0.0

        logger.info("Memory manager statistics reset")


class PerformanceProfiler:
    """Simple performance profiler for SDK operations.

    This class tracks timing and performance metrics for SDK operations
    to help identify bottlenecks and optimize performance.
    """

    def __init__(self, enabled: bool = True):
        """Initialize the performance profiler.

        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self._metrics = {}
        self._lock = threading.Lock()

    def record_operation(self, operation_name: str, duration_ms: float, success: bool = True):
        """Record a timed operation.

        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether the operation was successful
        """
        if not self.enabled:
            return

        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = {
                    "count": 0,
                    "total_duration_ms": 0.0,
                    "min_duration_ms": float('inf'),
                    "max_duration_ms": 0.0,
                    "success_count": 0,
                    "failure_count": 0
                }

            metrics = self._metrics[operation_name]
            metrics["count"] += 1
            metrics["total_duration_ms"] += duration_ms
            metrics["min_duration_ms"] = min(metrics["min_duration_ms"], duration_ms)
            metrics["max_duration_ms"] = max(metrics["max_duration_ms"], duration_ms)

            if success:
                metrics["success_count"] += 1
            else:
                metrics["failure_count"] += 1

    def get_stats(self) -> dict:
        """Get performance statistics.

        Returns:
            Dictionary containing performance metrics
        """
        if not self.enabled:
            return {"enabled": False}

        with self._lock:
            stats = {"enabled": True, "operations": {}}

            for operation_name, metrics in self._metrics.items():
                if metrics["count"] > 0:
                    avg_duration = metrics["total_duration_ms"] / metrics["count"]
                    success_rate = metrics["success_count"] / metrics["count"]

                    stats["operations"][operation_name] = {
                        "count": metrics["count"],
                        "avg_duration_ms": round(avg_duration, 2),
                        "min_duration_ms": round(metrics["min_duration_ms"], 2),
                        "max_duration_ms": round(metrics["max_duration_ms"], 2),
                        "total_duration_ms": round(metrics["total_duration_ms"], 2),
                        "success_rate": round(success_rate, 4),
                        "success_count": metrics["success_count"],
                        "failure_count": metrics["failure_count"]
                    }

            return stats

    def reset_stats(self):
        """Reset performance statistics."""
        with self._lock:
            self._metrics.clear()

    def get_operation_stats(self, operation_name: str) -> Optional[dict]:
        """Get statistics for a specific operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Dictionary with operation statistics or None if not found
        """
        if not self.enabled:
            return None

        with self._lock:
            if operation_name not in self._metrics:
                return None

            metrics = self._metrics[operation_name]
            if metrics["count"] == 0:
                return None

            avg_duration = metrics["total_duration_ms"] / metrics["count"]
            success_rate = metrics["success_count"] / metrics["count"]

            return {
                "count": metrics["count"],
                "avg_duration_ms": round(avg_duration, 2),
                "min_duration_ms": round(metrics["min_duration_ms"], 2),
                "max_duration_ms": round(metrics["max_duration_ms"], 2),
                "total_duration_ms": round(metrics["total_duration_ms"], 2),
                "success_rate": round(success_rate, 4),
                "success_count": metrics["success_count"],
                "failure_count": metrics["failure_count"]
            }