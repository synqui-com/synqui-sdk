"""Circuit breaker implementation for resilient API calls."""

import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast, not attempting calls
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for API calls.

    The circuit breaker pattern helps prevent cascading failures by
    temporarily disabling calls to a failing service. It has three states:

    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls fail immediately
    - HALF_OPEN: Testing if service has recovered

    Features:
    - Automatic failure detection and recovery
    - Configurable failure threshold and timeout
    - Thread-safe operation
    - Detailed logging and metrics
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker"
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that counts as failure
            name: Name for logging and identification
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._success_count = 0
        self._total_calls = 0

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"Circuit breaker '{name}' initialized with threshold={failure_threshold}, timeout={recovery_timeout}")

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count

    @property
    def success_count(self) -> int:
        """Get current success count."""
        with self._lock:
            return self._success_count

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the function call
        """
        with self._lock:
            self._total_calls += 1

            # Check current state and decide whether to proceed
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                else:
                    logger.debug(f"Circuit breaker '{self.name}' is OPEN, failing fast")
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.warning(f"Circuit breaker '{self.name}' encountered unexpected exception: {type(e).__name__}")
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker.

        Returns:
            True if enough time has passed since last failure
        """
        if self._last_failure_time is None:
            return True

        time_since_failure = time.time() - self._last_failure_time
        return time_since_failure >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit breaker '{self.name}' transitioning to CLOSED after successful call")
                self._state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, go back to OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed during recovery test, going back to OPEN")
                self._state = CircuitState.OPEN

            elif self._failure_count >= self.failure_threshold:
                # Too many failures, open the circuit
                logger.error(
                    f"Circuit breaker '{self.name}' opening after {self._failure_count} failures "
                    f"(threshold: {self.failure_threshold})"
                )
                self._state = CircuitState.OPEN

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Manually resetting circuit breaker '{self.name}'")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary containing current statistics
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
                "time_since_last_failure": (
                    time.time() - self._last_failure_time
                    if self._last_failure_time
                    else None
                ),
            }


class RetryPolicy:
    """Retry policy for failed operations.

    This class implements exponential backoff with jitter for retrying
    failed operations.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return 0.0

        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            import random
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Check if operation should be retried.

        Args:
            attempt: Current attempt number (0-based)
            exception: Exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        return attempt < self.max_retries


def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    name: Optional[str] = None
):
    """Decorator to add circuit breaker protection to a function.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that counts as failure
        name: Name for the circuit breaker

    Example:
        @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def api_call():
            # Your API call here
            pass
    """

    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=breaker_name
        )

        def wrapper(*args, **kwargs):
            return circuit_breaker.call(func, *args, **kwargs)

        # Add circuit breaker reference to wrapper
        wrapper._circuit_breaker = circuit_breaker
        return wrapper

    return decorator