"""
Timestamp freshness verification for FT-QuPAP v5.1.

Every Mobile Station authentication request includes a Unix timestamp.
The Authentication Server checks that the timestamp:

1. Is a valid non-negative integer.
2. Is not older than the configured freshness window.
3. Is not unreasonably far in the future.
4. Has an acceptable clock difference from the server.

A stale or future-dated request is rejected before expensive
post-quantum or quantum-processing operations begin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.common.constants import (
    FRESHNESS_WINDOW_SECONDS,
)

from src.common.enums import (
    FailureReason,
)

from src.common.exceptions import (
    FreshnessError,
    ProtocolValidationError,
)

from src.common.time_utils import (
    current_timestamp,
    format_timestamp,
)

from src.common.validators import (
    validate_integer,
)


DEFAULT_FUTURE_TOLERANCE_SECONDS = 5


@dataclass(frozen=True)
class FreshnessCheckResult:
    """
    Result of Authentication Server timestamp verification.

    Attributes
    ----------
    valid:
        True when the request timestamp is accepted.

    request_timestamp:
        Unix timestamp received from the Mobile Station.

    server_timestamp:
        Unix timestamp used by the Authentication Server.

    age_seconds:
        Difference calculated as:

            server_timestamp - request_timestamp

        Positive:
            Request timestamp is in the past.

        Negative:
            Request timestamp is in the future.

    freshness_window_seconds:
        Maximum accepted request age.

    future_tolerance_seconds:
        Maximum accepted future clock difference.

    failure_reason:
        NONE, STALE_TIMESTAMP, or FUTURE_TIMESTAMP.
    """

    valid: bool

    request_timestamp: int
    server_timestamp: int
    age_seconds: int

    freshness_window_seconds: int
    future_tolerance_seconds: int

    failure_reason: FailureReason
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ProtocolValidationError(
                "Freshness result valid flag must be Boolean."
            )

        validate_integer(
            self.request_timestamp,
            field_name="request_timestamp",
            minimum=0,
        )

        validate_integer(
            self.server_timestamp,
            field_name="server_timestamp",
            minimum=0,
        )

        if (
            isinstance(self.age_seconds, bool)
            or not isinstance(self.age_seconds, int)
        ):
            raise ProtocolValidationError(
                "age_seconds must be an integer."
            )

        validate_integer(
            self.freshness_window_seconds,
            field_name="freshness_window_seconds",
            minimum=1,
        )

        validate_integer(
            self.future_tolerance_seconds,
            field_name="future_tolerance_seconds",
            minimum=0,
        )

        if not isinstance(
            self.failure_reason,
            FailureReason,
        ):
            raise ProtocolValidationError(
                "failure_reason must be a FailureReason value."
            )

        if not isinstance(self.message, str):
            raise ProtocolValidationError(
                "Freshness result message must be a string."
            )

        if (
            self.valid
            and self.failure_reason != FailureReason.NONE
        ):
            raise ProtocolValidationError(
                (
                    "A valid freshness result cannot "
                    "contain a failure reason."
                )
            )

    @property
    def request_time_text(self) -> str:
        """Return the request timestamp as readable UTC text."""

        return format_timestamp(
            self.request_timestamp
        )

    @property
    def server_time_text(self) -> str:
        """Return the server timestamp as readable UTC text."""

        return format_timestamp(
            self.server_timestamp
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        result = asdict(self)

        result["failure_reason"] = (
            self.failure_reason.value
        )

        result["request_time_text"] = (
            self.request_time_text
        )

        result["server_time_text"] = (
            self.server_time_text
        )

        return result


def calculate_request_age(
    request_timestamp: int,
    *,
    server_timestamp: int | None = None,
) -> int:
    """
    Calculate authentication-request age in seconds.

    Formula:

        age = server_timestamp - request_timestamp

    Positive age:
        The request is from the past.

    Negative age:
        The request timestamp is ahead of the server.
    """

    validated_request_timestamp = (
        validate_integer(
            request_timestamp,
            field_name="request_timestamp",
            minimum=0,
        )
    )

    validated_server_timestamp = (
        current_timestamp()
        if server_timestamp is None
        else validate_integer(
            server_timestamp,
            field_name="server_timestamp",
            minimum=0,
        )
    )

    return (
        validated_server_timestamp
        - validated_request_timestamp
    )


def check_timestamp_freshness(
    request_timestamp: int,
    *,
    server_timestamp: int | None = None,
    freshness_window_seconds: int = (
        FRESHNESS_WINDOW_SECONDS
    ),
    future_tolerance_seconds: int = (
        DEFAULT_FUTURE_TOLERANCE_SECONDS
    ),
) -> FreshnessCheckResult:
    """
    Verify Authentication Server request freshness.

    Accepted condition:

        -future_tolerance_seconds
        <= request_age
        <= freshness_window_seconds

    A request older than the freshness window is stale.

    A request further into the future than the allowed tolerance is
    treated as an invalid future timestamp.
    """

    validated_request_timestamp = (
        validate_integer(
            request_timestamp,
            field_name="request_timestamp",
            minimum=0,
        )
    )

    validated_server_timestamp = (
        current_timestamp()
        if server_timestamp is None
        else validate_integer(
            server_timestamp,
            field_name="server_timestamp",
            minimum=0,
        )
    )

    validated_freshness_window = (
        validate_integer(
            freshness_window_seconds,
            field_name="freshness_window_seconds",
            minimum=1,
            maximum=86_400,
        )
    )

    validated_future_tolerance = (
        validate_integer(
            future_tolerance_seconds,
            field_name="future_tolerance_seconds",
            minimum=0,
            maximum=3_600,
        )
    )

    age_seconds = (
        validated_server_timestamp
        - validated_request_timestamp
    )

    if age_seconds > validated_freshness_window:
        return FreshnessCheckResult(
            valid=False,

            request_timestamp=(
                validated_request_timestamp
            ),

            server_timestamp=(
                validated_server_timestamp
            ),

            age_seconds=age_seconds,

            freshness_window_seconds=(
                validated_freshness_window
            ),

            future_tolerance_seconds=(
                validated_future_tolerance
            ),

            failure_reason=(
                FailureReason.STALE_TIMESTAMP
            ),

            message=(
                "Authentication request is stale. "
                f"Request age is {age_seconds} seconds, "
                "while the maximum accepted age is "
                f"{validated_freshness_window} seconds."
            ),
        )

    if age_seconds < -validated_future_tolerance:
        future_difference = abs(
            age_seconds
        )

        return FreshnessCheckResult(
            valid=False,

            request_timestamp=(
                validated_request_timestamp
            ),

            server_timestamp=(
                validated_server_timestamp
            ),

            age_seconds=age_seconds,

            freshness_window_seconds=(
                validated_freshness_window
            ),

            future_tolerance_seconds=(
                validated_future_tolerance
            ),

            failure_reason=(
                FailureReason.FUTURE_TIMESTAMP
            ),

            message=(
                "Authentication request timestamp is "
                f"{future_difference} seconds ahead of the server. "
                "The maximum accepted future difference is "
                f"{validated_future_tolerance} seconds."
            ),
        )

    return FreshnessCheckResult(
        valid=True,

        request_timestamp=(
            validated_request_timestamp
        ),

        server_timestamp=(
            validated_server_timestamp
        ),

        age_seconds=age_seconds,

        freshness_window_seconds=(
            validated_freshness_window
        ),

        future_tolerance_seconds=(
            validated_future_tolerance
        ),

        failure_reason=FailureReason.NONE,

        message=(
            "Authentication request timestamp "
            "is within the accepted freshness window."
        ),
    )


def is_timestamp_fresh(
    request_timestamp: int,
    *,
    server_timestamp: int | None = None,
    freshness_window_seconds: int = (
        FRESHNESS_WINDOW_SECONDS
    ),
    future_tolerance_seconds: int = (
        DEFAULT_FUTURE_TOLERANCE_SECONDS
    ),
) -> bool:
    """
    Return only the Boolean freshness decision.
    """

    result = check_timestamp_freshness(
        request_timestamp=request_timestamp,
        server_timestamp=server_timestamp,
        freshness_window_seconds=(
            freshness_window_seconds
        ),
        future_tolerance_seconds=(
            future_tolerance_seconds
        ),
    )

    return result.valid


def require_fresh_request(
    request_timestamp: int,
    *,
    server_timestamp: int | None = None,
    freshness_window_seconds: int = (
        FRESHNESS_WINDOW_SECONDS
    ),
    future_tolerance_seconds: int = (
        DEFAULT_FUTURE_TOLERANCE_SECONDS
    ),
) -> FreshnessCheckResult:
    """
    Verify freshness and raise FreshnessError when invalid.

    Returns the successful FreshnessCheckResult when accepted.
    """

    result = check_timestamp_freshness(
        request_timestamp=request_timestamp,
        server_timestamp=server_timestamp,
        freshness_window_seconds=(
            freshness_window_seconds
        ),
        future_tolerance_seconds=(
            future_tolerance_seconds
        ),
    )

    if not result.valid:
        raise FreshnessError(
            result.message,
            timestamp=result.request_timestamp,
            current_time=result.server_timestamp,
        )

    return result


def check_package_validity_period(
    *,
    issued_at: int,
    expires_at: int,
    server_timestamp: int | None = None,
    future_tolerance_seconds: int = (
        DEFAULT_FUTURE_TOLERANCE_SECONDS
    ),
) -> FreshnessCheckResult:
    """
    Verify the validity period of a signed server package.

    A package is accepted when:

        issued_at - future_tolerance
        <= server_time
        <= expires_at

    This helper may also be used by the Mobile Station when verifying
    the signed ML-KEM public-key package.
    """

    validated_issued_at = validate_integer(
        issued_at,
        field_name="issued_at",
        minimum=0,
    )

    validated_expires_at = validate_integer(
        expires_at,
        field_name="expires_at",
        minimum=0,
    )

    validated_server_timestamp = (
        current_timestamp()
        if server_timestamp is None
        else validate_integer(
            server_timestamp,
            field_name="server_timestamp",
            minimum=0,
        )
    )

    validated_future_tolerance = (
        validate_integer(
            future_tolerance_seconds,
            field_name="future_tolerance_seconds",
            minimum=0,
            maximum=3_600,
        )
    )

    if validated_expires_at <= validated_issued_at:
        raise ProtocolValidationError(
            (
                "Package expiration timestamp must "
                "be later than its issue timestamp."
            ),
            details={
                "issued_at": validated_issued_at,
                "expires_at": validated_expires_at,
            },
        )

    if (
        validated_server_timestamp
        < validated_issued_at
        - validated_future_tolerance
    ):
        age_seconds = (
            validated_server_timestamp
            - validated_issued_at
        )

        return FreshnessCheckResult(
            valid=False,
            request_timestamp=validated_issued_at,
            server_timestamp=(
                validated_server_timestamp
            ),
            age_seconds=age_seconds,
            freshness_window_seconds=(
                validated_expires_at
                - validated_issued_at
            ),
            future_tolerance_seconds=(
                validated_future_tolerance
            ),
            failure_reason=(
                FailureReason.FUTURE_TIMESTAMP
            ),
            message=(
                "Server package is not valid yet because its "
                "issue timestamp is too far in the future."
            ),
        )

    if (
        validated_server_timestamp
        > validated_expires_at
    ):
        age_seconds = (
            validated_server_timestamp
            - validated_issued_at
        )

        return FreshnessCheckResult(
            valid=False,
            request_timestamp=validated_issued_at,
            server_timestamp=(
                validated_server_timestamp
            ),
            age_seconds=age_seconds,
            freshness_window_seconds=(
                validated_expires_at
                - validated_issued_at
            ),
            future_tolerance_seconds=(
                validated_future_tolerance
            ),
            failure_reason=(
                FailureReason.STALE_TIMESTAMP
            ),
            message=(
                "Signed server package has expired."
            ),
        )

    return FreshnessCheckResult(
        valid=True,
        request_timestamp=validated_issued_at,
        server_timestamp=(
            validated_server_timestamp
        ),
        age_seconds=(
            validated_server_timestamp
            - validated_issued_at
        ),
        freshness_window_seconds=(
            validated_expires_at
            - validated_issued_at
        ),
        future_tolerance_seconds=(
            validated_future_tolerance
        ),
        failure_reason=FailureReason.NONE,
        message=(
            "Signed server package is within "
            "its accepted validity period."
        ),
    )


def run_freshness_checker_self_test() -> dict[str, Any]:
    """
    Run deterministic freshness-verification tests.

    Tests:

    - Recent request is accepted
    - Boundary request is accepted
    - Stale request is rejected
    - Small future clock difference is accepted
    - Excessive future timestamp is rejected
    - Expired package is rejected
    """

    server_time = 1_700_000_000

    recent_result = check_timestamp_freshness(
        request_timestamp=(
            server_time - 15
        ),
        server_timestamp=server_time,
        freshness_window_seconds=60,
        future_tolerance_seconds=5,
    )

    boundary_result = check_timestamp_freshness(
        request_timestamp=(
            server_time - 60
        ),
        server_timestamp=server_time,
        freshness_window_seconds=60,
        future_tolerance_seconds=5,
    )

    stale_result = check_timestamp_freshness(
        request_timestamp=(
            server_time - 61
        ),
        server_timestamp=server_time,
        freshness_window_seconds=60,
        future_tolerance_seconds=5,
    )

    acceptable_future_result = (
        check_timestamp_freshness(
            request_timestamp=(
                server_time + 5
            ),
            server_timestamp=server_time,
            freshness_window_seconds=60,
            future_tolerance_seconds=5,
        )
    )

    excessive_future_result = (
        check_timestamp_freshness(
            request_timestamp=(
                server_time + 6
            ),
            server_timestamp=server_time,
            freshness_window_seconds=60,
            future_tolerance_seconds=5,
        )
    )

    valid_package_result = (
        check_package_validity_period(
            issued_at=server_time - 10,
            expires_at=server_time + 50,
            server_timestamp=server_time,
        )
    )

    expired_package_result = (
        check_package_validity_period(
            issued_at=server_time - 100,
            expires_at=server_time - 1,
            server_timestamp=server_time,
        )
    )

    success = all(
        (
            recent_result.valid,
            boundary_result.valid,

            not stale_result.valid,

            stale_result.failure_reason
            == FailureReason.STALE_TIMESTAMP,

            acceptable_future_result.valid,

            not excessive_future_result.valid,

            excessive_future_result.failure_reason
            == FailureReason.FUTURE_TIMESTAMP,

            valid_package_result.valid,

            not expired_package_result.valid,
        )
    )

    return {
        "success": success,

        "recent_request_accepted": (
            recent_result.valid
        ),

        "boundary_request_accepted": (
            boundary_result.valid
        ),

        "stale_request_rejected": (
            not stale_result.valid
        ),

        "stale_failure_reason": (
            stale_result.failure_reason.value
        ),

        "small_future_difference_accepted": (
            acceptable_future_result.valid
        ),

        "future_request_rejected": (
            not excessive_future_result.valid
        ),

        "future_failure_reason": (
            excessive_future_result
            .failure_reason
            .value
        ),

        "valid_package_accepted": (
            valid_package_result.valid
        ),

        "expired_package_rejected": (
            not expired_package_result.valid
        ),
    }


__all__ = [
    "DEFAULT_FUTURE_TOLERANCE_SECONDS",
    "FreshnessCheckResult",
    "calculate_request_age",
    "check_timestamp_freshness",
    "is_timestamp_fresh",
    "require_fresh_request",
    "check_package_validity_period",
    "run_freshness_checker_self_test",
]