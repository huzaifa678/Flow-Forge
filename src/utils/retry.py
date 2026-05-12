"""Circuit breaker and retry utilities for LLM API calls."""

import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Optional


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for handling transient failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = (Exception,),
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise RuntimeError(
                    f"Circuit breaker is OPEN. Service unavailable. "
                    f"Retry after {self.recovery_timeout}s."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as exc:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time is not None
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self) -> None:
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        retry_on_exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on_exceptions = retry_on_exceptions


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay."""
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))
    return min(delay, config.max_delay)


def with_retry(config: RetryConfig, circuit_breaker: Optional[CircuitBreaker] = None):
    """Decorator for retry with circuit breaker support."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    return func(*args, **kwargs)
                except config.retry_on_exceptions as exc:
                    last_exception = exc
                    if attempt < config.max_attempts:
                        delay = calculate_delay(attempt, config)
                        time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable (timeout, server errors, etc.)."""
    from huggingface_hub.errors import HfHubHTTPError

    if isinstance(exc, HfHubHTTPError):
        status_code = getattr(exc, "response", None)
        if status_code:
            status = getattr(status_code, "status_code", None)
            return status in (408, 429, 500, 502, 503, 504)
    return isinstance(exc, (TimeoutError, ConnectionError))


def chat_completion_with_retry(
    llm,
    messages: list,
    max_tokens: int = 500,
    retry_config: Optional[RetryConfig] = None,
) -> Any:
    """Execute chat completion with retry and circuit breaker."""
    if retry_config is None:
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            retry_on_exceptions=(Exception,),
        )

    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=Exception,
    )

    last_exception = None

    for attempt in range(1, retry_config.max_attempts + 1):
        try:
            return circuit_breaker.call(
                llm.chat_completion,
                messages=messages,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_exception = exc
            if not is_retryable_error(exc):
                raise exc
            if attempt < retry_config.max_attempts:
                delay = calculate_delay(attempt, retry_config)
                time.sleep(delay)

    raise last_exception