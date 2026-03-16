import pytest

from retryer.retry import BackoffStrategy, RetryConfig, RetryState, retry

@pytest.fixture
def calls() -> dict[str, int]:
    return {"count": 0}

def test_returns_value_without_retry_when_function_succeeds(calls: dict[str, int]) -> None:
    @retry(RetryConfig())
    def operation() -> str:
        calls["count"] += 1
        return "ok"

    result = operation()

    assert result == "ok"
    assert calls["count"] == 1


def test_retries_until_success(calls: dict[str, int]) -> None:
    @retry(RetryConfig(max_attempts=3))
    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("temporary")
        return "done"

    result = operation()

    assert result == "done"
    assert calls["count"] == 3


def test_raises_after_max_attempts_exhausted(calls: dict[str, int]) -> None:
    @retry(RetryConfig(max_attempts=3))
    def operation() -> None:
        calls["count"] += 1
        raise ValueError("still failing")

    with pytest.raises(ValueError, match="still failing"):
        operation()

    assert calls["count"] == 3


def test_no_retry_for_non_matching_exception(calls: dict[str, int]) -> None:
    @retry(RetryConfig(max_attempts=5, retry_on=(ValueError,)))
    def operation() -> None:
        calls["count"] += 1
        raise TypeError("wrong type")

    with pytest.raises(TypeError, match="wrong type"):
        operation()

    assert calls["count"] == 1


def test_retries_only_when_predicate_true(calls: dict[str, int]) -> None:
    def is_retryable(exc: BaseException) -> bool:
        return "temporary" in str(exc)

    @retry(
        RetryConfig(
            max_attempts=4,
            retry_on=(ValueError,),
            retry_if=is_retryable,
        )
    )
    def operation() -> None:
        calls["count"] += 1
        raise ValueError("temporary issue")

    with pytest.raises(ValueError, match="temporary issue"):
        operation()

    assert calls["count"] == 4


def test_no_retry_when_predicate_false(calls: dict[str, int]) -> None:
    def is_retryable(exc: BaseException) -> bool:
        return False

    @retry(
        RetryConfig(
            max_attempts=4,
            retry_on=(ValueError,),
            retry_if=is_retryable,
        )
    )
    def operation() -> None:
        calls["count"] += 1
        raise ValueError("permanent issue")

    with pytest.raises(ValueError, match="permanent issue"):
        operation()

    assert calls["count"] == 1


def test_before_retry_receives_retry_state_and_delay(calls: dict[str, int]) -> None:
    observed = []

    def before_retry(state: RetryState, delay: float) -> None:
        observed.append((state.attempt, state.max_attempts, str(state.exception), delay))

    @retry(
        RetryConfig(
            max_attempts=3,
            delay=0.5,
            before_retry=before_retry,
            sleep=lambda _: None,
        )
    )
    def operation() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("temporary")
        return "ok"

    result = operation()

    assert result == "ok"
    assert observed == [
        (1, 3, "temporary", 0.5),
        (2, 3, "temporary", 0.5),
    ]


def test_sleep_is_called_with_computed_delay() -> None:
    sleep_calls = []

    @retry(
        RetryConfig(
            max_attempts=3,
            delay=0.25,
            sleep=lambda seconds: sleep_calls.append(seconds),
        )
    )
    def operation() -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError):
        operation()

    assert sleep_calls == [0.25, 0.25]


def test_sleep_is_not_called_when_delay_is_zero() -> None:
    sleep_calls = []

    @retry(
        RetryConfig(
            max_attempts=3,
            delay=0.0,
            sleep=lambda seconds: sleep_calls.append(seconds),
        )
    )
    def operation() -> None:
        raise ValueError("fail")

    with pytest.raises(ValueError):
        operation()

    assert sleep_calls == []


def test_fixed_delay_strategy() -> None:
    config = RetryConfig(strategy=BackoffStrategy.FIXED, delay=1.5)

    assert config.get_delay(1) == 1.5
    assert config.get_delay(3) == 1.5


def test_linear_delay_strategy() -> None:
    config = RetryConfig(strategy=BackoffStrategy.LINEAR, delay=2.0)

    assert config.get_delay(1) == 2.0
    assert config.get_delay(2) == 4.0
    assert config.get_delay(3) == 6.0


def test_exponential_delay_strategy() -> None:
    config = RetryConfig(
        strategy=BackoffStrategy.EXPONENTIAL,
        delay=1.0,
        backoff_multiplier=2.0,
    )

    assert config.get_delay(1) == 1.0
    assert config.get_delay(2) == 2.0
    assert config.get_delay(3) == 4.0


def test_max_delay_caps_computed_delay() -> None:
    config = RetryConfig(
        strategy=BackoffStrategy.EXPONENTIAL,
        delay=2.0,
        backoff_multiplier=3.0,
        max_delay=5.0,
    )

    assert config.get_delay(1) == 2.0
    assert config.get_delay(2) == 5.0
    assert config.get_delay(3) == 5.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "max_attempts must be at least 1"),
        ({"delay": -1.0}, "delay must be greater than or equal to 0"),
        ({"max_delay": -1.0}, "max_delay must be greater than or equal to 0"),
        ({"backoff_multiplier": 0.5}, "backoff_multiplier must be at least 1"),
        ({"jitter": -0.1}, "jitter must be greater than or equal to 0"),
        ({"retry_on": ()}, "retry_on must contain at least one exception type"),
    ],
)
def test_retry_config_validation_errors(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RetryConfig(**kwargs)


def test_get_delay_rejects_invalid_attempt() -> None:
    config = RetryConfig()

    with pytest.raises(ValueError, match="attempt must be at least 1"):
        config.get_delay(0)


def test_decorator_preserves_function_metadata() -> None:
    @retry(RetryConfig())
    def operation() -> str:
        """Example docstring."""
        return "ok"

    assert operation.__name__ == "operation"
    assert operation.__doc__ == "Example docstring."


def test_arguments_and_keyword_arguments_are_forwarded() -> None:
    @retry(RetryConfig())
    def add(a: int, b: int, scale: int = 1) -> int:
        return (a + b) * scale

    assert add(2, 3, scale=4) == 20
