from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class BackoffStrategy(str, Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass(frozen=True)
class RetryState:
    attempt: int
    max_attempts: int
    exception: BaseException


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    retry_on: Tuple[Type[BaseException], ...] = (Exception,)
    strategy: BackoffStrategy = BackoffStrategy.FIXED
    delay: float = 0.0
    backoff_multiplier: float = 2.0
    max_delay: Optional[float] = None
    jitter: float = 0.0
    retry_if: Optional[Callable[[BaseException], bool]] = None
    before_retry: Optional[Callable[[RetryState, float], None]] = None
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.delay < 0:
            raise ValueError("delay must be greater than or equal to 0")
        if self.max_delay is not None and self.max_delay < 0:
            raise ValueError("max_delay must be greater than or equal to 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.jitter < 0:
            raise ValueError("jitter must be greater than or equal to 0")
        if not self.retry_on:
            raise ValueError("retry_on must contain at least one exception type")

    def should_retry(self, exception: BaseException) -> bool:
        if not isinstance(exception, self.retry_on):
            return False

        if self.retry_if is None:
            return True

        return self.retry_if(exception)

    def get_delay(self, attempt: int) -> float:
        if attempt < 1:
            raise ValueError("attempt must be at least 1")

        if self.strategy == BackoffStrategy.FIXED:
            computed_delay = self.delay
        elif self.strategy == BackoffStrategy.LINEAR:
            computed_delay = self.delay * attempt
        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            computed_delay = self.delay * (self.backoff_multiplier ** (attempt - 1))
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        if self.max_delay is not None:
            computed_delay = min(computed_delay, self.max_delay)

        if self.jitter > 0:
            computed_delay += random.uniform(0, self.jitter)

        return computed_delay


def retry(config: RetryConfig) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[BaseException] = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    if attempt >= config.max_attempts or not config.should_retry(exc):
                        raise

                    delay = config.get_delay(attempt)
                    state = RetryState(
                        attempt=attempt, max_attempts=config.max_attempts, exception=exc
                    )

                    if config.before_retry is not None:
                        config.before_retry(state, delay)

                    if delay > 0:
                        config.sleep(delay)

            if last_exception is not None:
                raise last_exception

            raise RuntimeError("retry failed without capturing an exception")

        return wrapper  # type: ignore[return-value]

    return decorator
