"""
Time and freshness utilities for the FT-QuPAP v5.1 project.

This module provides:

- Current Unix timestamps
- UTC datetime conversion
- Timestamp formatting
- Request-age calculation
- Freshness verification
- Future-clock tolerance
- Expiration-time generation

Protocol messages use Unix timestamps in UTC so the Mobile Station and
Authentication Server can compare freshness consistently.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from src.common.constants import (
    FRESHNESS_WINDOW_SECONDS,
)

from src.common.exceptions import (
    FreshnessError,
    ProtocolValidationError,
)

from src.common.validators import (
    validate_integer,
)


def current_timestamp() -> int:
    """
    Return the current Unix timestamp in whole seconds.

    Example:

        1785469800
    """

    return int(time.time())


def current_timestamp_milliseconds() -> int:
    """
    Return the current Unix timestamp in milliseconds.

    This is useful for logs and performance measurements, but the
    authentication protocol itself uses whole-second timestamps.
    """

    return int(time.time() * 1000)


def monotonic_time_seconds() -> float:
    """
    Return a monotonic clock value.

    Use this for measuring execution duration. Do not use it as a
    protocol timestamp because it has no relationship to UTC time.
    """

    return time.perf_counter()


def timestamp_to_datetime(
    timestamp: int,
) -> datetime:
    """
    Convert a Unix timestamp into a timezone-aware UTC datetime.
    """

    validated_timestamp = validate_integer(
        timestamp,
        field_name="timestamp",
        minimum=0,
    )

    try:
        return datetime.fromtimestamp(
            validated_timestamp,
            tz=timezone.utc,
        )
    except (
        OverflowError,
        OSError,
        ValueError,
    ) as exc:
        raise ProtocolValidationError(
            "Timestamp cannot be converted to UTC datetime.",
            details={
                "timestamp": validated_timestamp,
            },
        ) from exc


def datetime_to_timestamp(
    value: datetime,
) -> int:
    """
    Convert a datetime into a Unix timestamp.

    Naive datetimes are rejected because their timezone is ambiguous.
    """

    if not isinstance(value, datetime):
        raise ProtocolValidationError(
            "The supplied value must be a datetime object.",
            details={
                "received_type": type(value).__name__,
            },
        )

    if value.tzinfo is None:
        raise ProtocolValidationError(
            "Datetime must include timezone information."
        )

    return int(
        value.astimezone(
            timezone.utc
        ).timestamp()
    )


def format_timestamp(
    timestamp: int,
    *,
    format_string: str = "%Y-%m-%d %H:%M:%S UTC",
) -> str:
    """
    Format a Unix timestamp as readable UTC text.

    Example output:

        2026-07-31 05:50:00 UTC
    """

    utc_datetime = timestamp_to_datetime(
        timestamp
    )

    return utc_datetime.strftime(
        format_string
    )


def isoformat_timestamp(
    timestamp: int,
) -> str:
    """
    Convert a Unix timestamp into ISO 8601 UTC format.

    Example:

        2026-07-31T05:50:00+00:00
    """

    return timestamp_to_datetime(
        timestamp
    ).isoformat()


def parse_iso_timestamp(
    value: str,
) -> int:
    """
    Convert ISO 8601 datetime text into a Unix timestamp.

    The text must include timezone information.

    Example:

        2026-07-31T05:50:00+00:00
    """

    if not isinstance(value, str):
        raise ProtocolValidationError(
            "ISO timestamp must be a string.",
            details={
                "received_type": type(value).__name__,
            },
        )

    normalized = value.strip()

    if not normalized:
        raise ProtocolValidationError(
            "ISO timestamp cannot be empty."
        )

    if normalized.endswith("Z"):
        normalized = (
            normalized[:-1]
            + "+00:00"
        )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )
    except ValueError as exc:
        raise ProtocolValidationError(
            "Invalid ISO 8601 timestamp.",
            details={
                "value": value,
            },
        ) from exc

    if parsed.tzinfo is None:
        raise ProtocolValidationError(
            "ISO timestamp must contain timezone information."
        )

    return datetime_to_timestamp(
        parsed
    )


def calculate_timestamp_age(
    timestamp: int,
    *,
    reference_time: int | None = None,
) -> int:
    """
    Calculate the age of a timestamp in seconds.

    A positive result means the timestamp is in the past.

    A negative result means the timestamp is in the future.
    """

    validated_timestamp = validate_integer(
        timestamp,
        field_name="timestamp",
        minimum=0,
    )

    current = (
        current_timestamp()
        if reference_time is None
        else validate_integer(
            reference_time,
            field_name="reference_time",
            minimum=0,
        )
    )

    return current - validated_timestamp


def is_timestamp_fresh(
    timestamp: int,
    *,
    reference_time: int | None = None,
    freshness_window_seconds: int = FRESHNESS_WINDOW_SECONDS,
    future_tolerance_seconds: int = 5,
) -> bool:
    """
    Return True when a timestamp is within the accepted freshness window.

    This function does not raise an exception for stale timestamps.
    """

    validated_window = validate_integer(
        freshness_window_seconds,
        field_name="freshness_window_seconds",
        minimum=1,
    )

    validated_future_tolerance = validate_integer(
        future_tolerance_seconds,
        field_name="future_tolerance_seconds",
        minimum=0,
    )

    age = calculate_timestamp_age(
        timestamp,
        reference_time=reference_time,
    )

    return (
        -validated_future_tolerance
        <= age
        <= validated_window
    )


def require_fresh_timestamp(
    timestamp: int,
    *,
    reference_time: int | None = None,
    freshness_window_seconds: int = FRESHNESS_WINDOW_SECONDS,
    future_tolerance_seconds: int = 5,
) -> int:
    """
    Validate request freshness and return the accepted timestamp.

    Raises FreshnessError when the request is stale or too far in the
    future.
    """

    validated_timestamp = validate_integer(
        timestamp,
        field_name="timestamp",
        minimum=0,
    )

    validated_window = validate_integer(
        freshness_window_seconds,
        field_name="freshness_window_seconds",
        minimum=1,
    )

    validated_future_tolerance = validate_integer(
        future_tolerance_seconds,
        field_name="future_tolerance_seconds",
        minimum=0,
    )

    current = (
        current_timestamp()
        if reference_time is None
        else validate_integer(
            reference_time,
            field_name="reference_time",
            minimum=0,
        )
    )

    age = current - validated_timestamp

    if age > validated_window:
        raise FreshnessError(
            (
                "Authentication request is stale. "
                f"Request age is {age} seconds, while the maximum "
                f"allowed age is {validated_window} seconds."
            ),
            timestamp=validated_timestamp,
            current_time=current,
        )

    if age < -validated_future_tolerance:
        raise FreshnessError(
            (
                "Authentication request timestamp is too far in "
                f"the future by {abs(age)} seconds."
            ),
            timestamp=validated_timestamp,
            current_time=current,
        )

    return validated_timestamp


def create_expiration_timestamp(
    lifetime_seconds: int,
    *,
    start_timestamp: int | None = None,
) -> int:
    """
    Create a future expiration timestamp.

    Example:

        expiration = create_expiration_timestamp(300)
    """

    validated_lifetime = validate_integer(
        lifetime_seconds,
        field_name="lifetime_seconds",
        minimum=1,
    )

    start = (
        current_timestamp()
        if start_timestamp is None
        else validate_integer(
            start_timestamp,
            field_name="start_timestamp",
            minimum=0,
        )
    )

    return start + validated_lifetime


def is_expired(
    expiration_timestamp: int,
    *,
    reference_time: int | None = None,
) -> bool:
    """
    Return True when an expiration timestamp has passed.
    """

    validated_expiration = validate_integer(
        expiration_timestamp,
        field_name="expiration_timestamp",
        minimum=0,
    )

    current = (
        current_timestamp()
        if reference_time is None
        else validate_integer(
            reference_time,
            field_name="reference_time",
            minimum=0,
        )
    )

    return current > validated_expiration


def require_not_expired(
    expiration_timestamp: int,
    *,
    reference_time: int | None = None,
    object_name: str = "credential",
) -> int:
    """
    Raise FreshnessError when a credential or package has expired.
    """

    validated_expiration = validate_integer(
        expiration_timestamp,
        field_name="expiration_timestamp",
        minimum=0,
    )

    current = (
        current_timestamp()
        if reference_time is None
        else validate_integer(
            reference_time,
            field_name="reference_time",
            minimum=0,
        )
    )

    if current > validated_expiration:
        raise FreshnessError(
            f"{object_name} has expired.",
            timestamp=validated_expiration,
            current_time=current,
        )

    return validated_expiration


def measure_execution_time(
    function,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """
    Execute a function and return:

        result, elapsed_seconds

    Example:

        result, elapsed = measure_execution_time(
            run_protocol,
            scenario,
        )
    """

    if not callable(function):
        raise ProtocolValidationError(
            "The supplied object must be callable."
        )

    start = monotonic_time_seconds()

    result = function(
        *args,
        **kwargs,
    )

    elapsed = (
        monotonic_time_seconds()
        - start
    )

    return result, elapsed


class ExecutionTimer:
    """
    Context manager for measuring execution time.

    Example:

        with ExecutionTimer() as timer:
            run_protocol()

        print(timer.elapsed_seconds)
    """

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.elapsed_seconds: float | None = None

    def __enter__(
        self,
    ) -> "ExecutionTimer":
        self.started_at = (
            monotonic_time_seconds()
        )

        self.finished_at = None
        self.elapsed_seconds = None

        return self

    def __exit__(
        self,
        exception_type,
        exception_value,
        traceback,
    ) -> bool:
        self.finished_at = (
            monotonic_time_seconds()
        )

        if self.started_at is not None:
            self.elapsed_seconds = (
                self.finished_at
                - self.started_at
            )

        return False


__all__ = [
    "current_timestamp",
    "current_timestamp_milliseconds",
    "monotonic_time_seconds",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "format_timestamp",
    "isoformat_timestamp",
    "parse_iso_timestamp",
    "calculate_timestamp_age",
    "is_timestamp_fresh",
    "require_fresh_timestamp",
    "create_expiration_timestamp",
    "is_expired",
    "require_not_expired",
    "measure_execution_time",
    "ExecutionTimer",
]