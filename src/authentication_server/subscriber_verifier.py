"""
Subscriber-request verification for FT-QuPAP v5.1.

This module verifies the Mobile Station's initial M1 authentication
request before the Authentication Server generates its M2 response.

Verification includes:

- Protocol version and domain validation
- M1 message-type validation
- Subscriber pseudonym lookup
- Subscriber active-status checking
- Timestamp freshness checking
- Mobile nonce validation
- Session and attempt-number validation
- Replay detection
- Safe subscriber-record retrieval

This stage does not verify the final KMAC authentication tag. KMAC tag
verification occurs later, after quantum payload recovery.
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.authentication_server.registration_manager import (
    RegistrationManager,
    RegistrationManagerError,
    SubscriberRegistrationRecord,
    get_default_registration_manager,
)

from src.authentication_server.replay_detector import (
    REPLAY_TYPE_NONE,
    ReplayDetector,
    ReplayCheckResult,
    get_default_replay_detector,
)

from src.common.constants import (
    FRESHNESS_WINDOW_SECONDS,
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
    decode_base64,
    encode_base64,
)

from src.common.time_utils import (
    current_timestamp,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
    validate_pseudonym_id,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SUBSCRIBER_REQUEST_VERSION = (
    "FT-QuPAP-Subscriber-Request-v1"
)

SUBSCRIBER_REQUEST_MESSAGE_TYPE = "M1"

DEFAULT_SUBSCRIBER_FRESHNESS_WINDOW_SECONDS = max(
    int(FRESHNESS_WINDOW_SECONDS),
    30,
)

DEFAULT_FUTURE_TOLERANCE_SECONDS = 5

MINIMUM_MOBILE_NONCE_BYTES = 16

MAXIMUM_MOBILE_NONCE_BYTES = 64

SUBSCRIBER_REQUEST_FINGERPRINT_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Result reasons
# ---------------------------------------------------------------------

REASON_VERIFIED = "subscriber_request_verified"

REASON_UNKNOWN_PSEUDONYM = "unknown_pseudonym"

REASON_HISTORICAL_PSEUDONYM = "historical_pseudonym"

REASON_SUBSCRIBER_INACTIVE = "subscriber_inactive"

REASON_STALE_TIMESTAMP = "stale_timestamp"

REASON_FUTURE_TIMESTAMP = "future_timestamp"

REASON_REPLAY_DETECTED = "replay_detected"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class SubscriberVerificationError(RuntimeError):
    """Raised when an M1 subscriber request cannot be accepted."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.reason = reason

        self.details = (
            {}
            if details is None
            else dict(details)
        )


# ---------------------------------------------------------------------
# Normalized request
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NormalizedSubscriberRequest:
    """
    Validated and normalized FT-QuPAP M1 request.
    """

    version: str
    protocol_domain: str
    message_type: str

    pseudonym_id: str
    session_id: str
    attempt_number: int

    mobile_nonce: bytes
    timestamp: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_non_empty_string(
            self.version,
            field_name="version",
            minimum_length=1,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.protocol_domain,
            field_name="protocol_domain",
            minimum_length=1,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.message_type,
            field_name="message_type",
            minimum_length=1,
            maximum_length=32,
        )

        validate_pseudonym_id(
            self.pseudonym_id
        )

        validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=256,
        )

        validate_integer(
            self.attempt_number,
            field_name="attempt_number",
            minimum=1,
            maximum=100,
        )

        validate_bytes(
            self.mobile_nonce,
            field_name="mobile_nonce",
            minimum_length=(
                MINIMUM_MOBILE_NONCE_BYTES
            ),
            maximum_length=(
                MAXIMUM_MOBILE_NONCE_BYTES
            ),
        )

        validate_integer(
            self.timestamp,
            field_name="timestamp",
            minimum=0,
        )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise ProtocolValidationError(
                "metadata must be a dictionary."
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        """
        Return the normalized M1 network representation.
        """

        return {
            "version": self.version,
            "protocol_domain": (
                self.protocol_domain
            ),
            "message_type": self.message_type,
            "pseudonym_id": self.pseudonym_id,
            "session_id": self.session_id,
            "attempt_number": (
                self.attempt_number
            ),
            "mobile_nonce": encode_base64(
                self.mobile_nonce
            ),
            "timestamp": self.timestamp,
            "metadata": dict(
                self.metadata
            ),
        }

    @property
    def request_fingerprint(self) -> str:
        """Return the canonical M1 request fingerprint."""

        return calculate_subscriber_request_fingerprint(
            self.to_canonical_dict()
        )

    @property
    def nonce_fingerprint(self) -> str:
        """Return the SHA3-256 nonce fingerprint."""

        return hashlib.sha3_256(
            self.mobile_nonce
        ).hexdigest()

    def __repr__(self) -> str:
        return (
            "NormalizedSubscriberRequest("
            f"version={self.version!r}, "
            f"message_type={self.message_type!r}, "
            f"pseudonym_id={self.pseudonym_id!r}, "
            f"session_id={self.session_id!r}, "
            f"attempt_number={self.attempt_number}, "
            f"timestamp={self.timestamp}, "
            f"request_fingerprint="
            f"{self.request_fingerprint!r}, "
            "mobile_nonce=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SubscriberVerificationResult:
    """
    Result of FT-QuPAP subscriber-request verification.
    """

    verified: bool

    reason: str
    message: str

    registration_id: str | None
    pseudonym_id: str
    session_id: str
    attempt_number: int

    request_timestamp: int
    checked_at: int
    request_age_seconds: int

    structure_valid: bool
    subscriber_found: bool
    subscriber_active: bool
    freshness_valid: bool
    replay_free: bool

    request_fingerprint: str
    nonce_fingerprint: str

    replay_type: str

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "verified",
            "structure_valid",
            "subscriber_found",
            "subscriber_active",
            "freshness_valid",
            "replay_free",
        ):
            if not isinstance(
                getattr(
                    self,
                    field_name,
                ),
                bool,
            ):
                raise ProtocolValidationError(
                    f"{field_name} must be Boolean."
                )

        expected_verified = all(
            (
                self.structure_valid,
                self.subscriber_found,
                self.subscriber_active,
                self.freshness_valid,
                self.replay_free,
            )
        )

        if self.verified != expected_verified:
            raise ProtocolValidationError(
                (
                    "verified does not match the "
                    "individual verification checks."
                )
            )

        validate_non_empty_string(
            self.reason,
            field_name="reason",
            minimum_length=1,
            maximum_length=256,
        )

        validate_non_empty_string(
            self.message,
            field_name="message",
            minimum_length=1,
            maximum_length=1000,
        )

        if self.registration_id is not None:
            validate_non_empty_string(
                self.registration_id,
                field_name="registration_id",
                minimum_length=8,
                maximum_length=128,
            )

        validate_pseudonym_id(
            self.pseudonym_id
        )

        validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=256,
        )

        validate_integer(
            self.attempt_number,
            field_name="attempt_number",
            minimum=1,
            maximum=100,
        )

        validate_integer(
            self.request_timestamp,
            field_name="request_timestamp",
            minimum=0,
        )

        validate_integer(
            self.checked_at,
            field_name="checked_at",
            minimum=0,
        )

        if not isinstance(
            self.request_age_seconds,
            int,
        ):
            raise ProtocolValidationError(
                "request_age_seconds must be an integer."
            )

        _validate_hex_digest(
            self.request_fingerprint,
            field_name="request_fingerprint",
        )

        _validate_hex_digest(
            self.nonce_fingerprint,
            field_name="nonce_fingerprint",
        )

        validate_non_empty_string(
            self.replay_type,
            field_name="replay_type",
            minimum_length=1,
            maximum_length=128,
        )

        if not isinstance(
            self.details,
            dict,
        ):
            raise ProtocolValidationError(
                "details must be a dictionary."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible verification result."""

        return asdict(
            self
        )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_hex_digest(
    value: str,
    *,
    field_name: str,
) -> str:
    """Validate a SHA3-256 hexadecimal digest."""

    validated = validate_non_empty_string(
        value,
        field_name=field_name,
        minimum_length=64,
        maximum_length=64,
    ).lower()

    try:
        bytes.fromhex(
            validated
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            f"{field_name} must be hexadecimal text."
        ) from exc

    return validated


def normalize_mobile_nonce(
    value: bytes | str,
) -> bytes:
    """
    Normalize an M1 Mobile Station nonce.

    Accepted forms:

    - Raw bytes
    - Hexadecimal text
    - Base64 text
    """

    if isinstance(
        value,
        bytes,
    ):
        return validate_bytes(
            value,
            field_name="mobile_nonce",
            minimum_length=(
                MINIMUM_MOBILE_NONCE_BYTES
            ),
            maximum_length=(
                MAXIMUM_MOBILE_NONCE_BYTES
            ),
        )

    if not isinstance(
        value,
        str,
    ):
        raise ProtocolValidationError(
            (
                "mobile_nonce must be bytes, "
                "hexadecimal text, or Base64 text."
            )
        )

    normalized_text = validate_non_empty_string(
        value,
        field_name="mobile_nonce",
        minimum_length=16,
        maximum_length=256,
    ).strip()

    is_hexadecimal = (
        len(normalized_text) % 2 == 0
        and all(
            character in string.hexdigits
            for character in normalized_text
        )
    )

    if is_hexadecimal:
        try:
            decoded = bytes.fromhex(
                normalized_text
            )

        except ValueError:
            decoded = b""

        if (
            MINIMUM_MOBILE_NONCE_BYTES
            <= len(decoded)
            <= MAXIMUM_MOBILE_NONCE_BYTES
        ):
            return decoded

    try:
        decoded = decode_base64(
            normalized_text
        )

    except Exception as exc:
        raise ProtocolValidationError(
            (
                "mobile_nonce is neither valid "
                "hexadecimal nor valid Base64 text."
            )
        ) from exc

    return validate_bytes(
        decoded,
        field_name="mobile_nonce",
        minimum_length=(
            MINIMUM_MOBILE_NONCE_BYTES
        ),
        maximum_length=(
            MAXIMUM_MOBILE_NONCE_BYTES
        ),
    )


def normalize_subscriber_request(
    request: Mapping[str, Any],
) -> NormalizedSubscriberRequest:
    """
    Validate and normalize an FT-QuPAP M1 request.
    """

    if not isinstance(
        request,
        Mapping,
    ):
        raise ProtocolValidationError(
            "Subscriber request must be a mapping."
        )

    required_fields = (
        "version",
        "protocol_domain",
        "message_type",
        "pseudonym_id",
        "session_id",
        "attempt_number",
        "mobile_nonce",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in request
    ]

    if (
        "timestamp" not in request
        and "issued_at" not in request
    ):
        missing_fields.append(
            "timestamp"
        )

    if missing_fields:
        raise ProtocolValidationError(
            "Subscriber request is incomplete.",
            details={
                "missing_fields": missing_fields,
            },
        )

    version = validate_non_empty_string(
        request["version"],
        field_name="version",
        minimum_length=1,
        maximum_length=128,
    )

    if version != SUBSCRIBER_REQUEST_VERSION:
        raise ProtocolValidationError(
            "Unsupported subscriber-request version.",
            details={
                "received_version": version,
                "expected_version": (
                    SUBSCRIBER_REQUEST_VERSION
                ),
            },
        )

    protocol_domain = (
        validate_non_empty_string(
            request["protocol_domain"],
            field_name="protocol_domain",
            minimum_length=1,
            maximum_length=128,
        )
    )

    expected_domain = (
        PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        )
    )

    if protocol_domain != expected_domain:
        raise ProtocolValidationError(
            "Subscriber-request protocol domain mismatch.",
            details={
                "received_domain": (
                    protocol_domain
                ),
                "expected_domain": (
                    expected_domain
                ),
            },
        )

    message_type = (
        validate_non_empty_string(
            request["message_type"],
            field_name="message_type",
            minimum_length=1,
            maximum_length=32,
        )
    )

    if (
        message_type
        != SUBSCRIBER_REQUEST_MESSAGE_TYPE
    ):
        raise ProtocolValidationError(
            "Invalid subscriber-request message type.",
            details={
                "received_message_type": (
                    message_type
                ),
                "expected_message_type": (
                    SUBSCRIBER_REQUEST_MESSAGE_TYPE
                ),
            },
        )

    pseudonym_id = validate_pseudonym_id(
        request["pseudonym_id"]
    )

    session_id = validate_non_empty_string(
        request["session_id"],
        field_name="session_id",
        minimum_length=3,
        maximum_length=256,
    )

    attempt_number = validate_integer(
        request["attempt_number"],
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    mobile_nonce = normalize_mobile_nonce(
        request["mobile_nonce"]
    )

    raw_timestamp = request.get(
        "timestamp",
        request.get(
            "issued_at"
        ),
    )

    timestamp = validate_integer(
        raw_timestamp,
        field_name="timestamp",
        minimum=0,
    )

    metadata = request.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise ProtocolValidationError(
            "Subscriber-request metadata must be a mapping."
        )

    return NormalizedSubscriberRequest(
        version=version,
        protocol_domain=protocol_domain,
        message_type=message_type,
        pseudonym_id=pseudonym_id,
        session_id=session_id,
        attempt_number=attempt_number,
        mobile_nonce=mobile_nonce,
        timestamp=timestamp,
        metadata=dict(
            metadata
        ),
    )


def calculate_subscriber_request_fingerprint(
    request: Mapping[str, Any],
) -> str:
    """
    Calculate a canonical SHA3-256 M1 request fingerprint.
    """

    if not isinstance(
        request,
        Mapping,
    ):
        raise ProtocolValidationError(
            "request must be a mapping."
        )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00subscriber-request\x00"
    )

    digest.update(
        canonical_json_bytes(
            dict(request)
        )
    )

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Freshness checking
# ---------------------------------------------------------------------

def check_request_freshness(
    *,
    request_timestamp: int,
    checked_at: int,
    freshness_window_seconds: int,
    future_tolerance_seconds: int,
) -> tuple[
    bool,
    str,
    int,
]:
    """
    Check whether an M1 timestamp is fresh.

    Returns:

        freshness_valid
        reason
        request_age_seconds
    """

    validated_request_timestamp = (
        validate_integer(
            request_timestamp,
            field_name="request_timestamp",
            minimum=0,
        )
    )

    validated_checked_at = (
        validate_integer(
            checked_at,
            field_name="checked_at",
            minimum=0,
        )
    )

    validated_window = validate_integer(
        freshness_window_seconds,
        field_name="freshness_window_seconds",
        minimum=1,
        maximum=86_400,
    )

    validated_future_tolerance = (
        validate_integer(
            future_tolerance_seconds,
            field_name=(
                "future_tolerance_seconds"
            ),
            minimum=0,
            maximum=3600,
        )
    )

    request_age = (
        validated_checked_at
        - validated_request_timestamp
    )

    if (
        validated_request_timestamp
        > (
            validated_checked_at
            + validated_future_tolerance
        )
    ):
        return (
            False,
            REASON_FUTURE_TIMESTAMP,
            request_age,
        )

    if request_age > validated_window:
        return (
            False,
            REASON_STALE_TIMESTAMP,
            request_age,
        )

    return (
        True,
        REASON_VERIFIED,
        request_age,
    )


# ---------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------

class SubscriberVerifier:
    """
    FT-QuPAP Authentication Server M1 verifier.
    """

    def __init__(
        self,
        *,
        registration_manager: RegistrationManager | None = None,
        replay_detector: ReplayDetector | None = None,
        freshness_window_seconds: int = (
            DEFAULT_SUBSCRIBER_FRESHNESS_WINDOW_SECONDS
        ),
        future_tolerance_seconds: int = (
            DEFAULT_FUTURE_TOLERANCE_SECONDS
        ),
    ) -> None:
        self.registration_manager = (
            get_default_registration_manager()
            if registration_manager is None
            else registration_manager
        )

        self.replay_detector = (
            get_default_replay_detector()
            if replay_detector is None
            else replay_detector
        )

        if not isinstance(
            self.registration_manager,
            RegistrationManager,
        ):
            raise ProtocolValidationError(
                (
                    "registration_manager must be a "
                    "RegistrationManager object."
                )
            )

        if not isinstance(
            self.replay_detector,
            ReplayDetector,
        ):
            raise ProtocolValidationError(
                (
                    "replay_detector must be a "
                    "ReplayDetector object."
                )
            )

        self.freshness_window_seconds = (
            validate_integer(
                freshness_window_seconds,
                field_name=(
                    "freshness_window_seconds"
                ),
                minimum=1,
                maximum=86_400,
            )
        )

        self.future_tolerance_seconds = (
            validate_integer(
                future_tolerance_seconds,
                field_name=(
                    "future_tolerance_seconds"
                ),
                minimum=0,
                maximum=3600,
            )
        )

    @staticmethod
    def _build_result(
        *,
        request: NormalizedSubscriberRequest,
        checked_at: int,
        request_age_seconds: int,
        verified: bool,
        reason: str,
        message: str,
        registration_id: str | None,
        subscriber_found: bool,
        subscriber_active: bool,
        freshness_valid: bool,
        replay_free: bool,
        replay_type: str = REPLAY_TYPE_NONE,
        details: Mapping[str, Any] | None = None,
    ) -> SubscriberVerificationResult:
        """Build a validated subscriber-verification result."""

        return SubscriberVerificationResult(
            verified=verified,
            reason=reason,
            message=message,
            registration_id=registration_id,
            pseudonym_id=request.pseudonym_id,
            session_id=request.session_id,
            attempt_number=(
                request.attempt_number
            ),
            request_timestamp=(
                request.timestamp
            ),
            checked_at=checked_at,
            request_age_seconds=(
                request_age_seconds
            ),
            structure_valid=True,
            subscriber_found=(
                subscriber_found
            ),
            subscriber_active=(
                subscriber_active
            ),
            freshness_valid=(
                freshness_valid
            ),
            replay_free=replay_free,
            request_fingerprint=(
                request.request_fingerprint
            ),
            nonce_fingerprint=(
                request.nonce_fingerprint
            ),
            replay_type=replay_type,
            details=(
                {}
                if details is None
                else dict(details)
            ),
        )

    def verify_request(
        self,
        request: Mapping[str, Any],
        *,
        checked_at: int | None = None,
        record_replay_on_accept: bool = True,
    ) -> SubscriberVerificationResult:
        """
        Verify one M1 request.

        Replay evidence is recorded atomically only after subscriber and
        freshness checks have succeeded.
        """

        if not isinstance(
            record_replay_on_accept,
            bool,
        ):
            raise ProtocolValidationError(
                "record_replay_on_accept must be Boolean."
            )

        normalized_request = (
            normalize_subscriber_request(
                request
            )
        )

        selected_checked_at = (
            current_timestamp()
            if checked_at is None
            else validate_integer(
                checked_at,
                field_name="checked_at",
                minimum=0,
            )
        )

        (
            freshness_valid,
            freshness_reason,
            request_age,
        ) = check_request_freshness(
            request_timestamp=(
                normalized_request.timestamp
            ),
            checked_at=selected_checked_at,
            freshness_window_seconds=(
                self.freshness_window_seconds
            ),
            future_tolerance_seconds=(
                self.future_tolerance_seconds
            ),
        )

        try:
            subscriber = (
                self.registration_manager
                .get_by_pseudonym(
                    normalized_request
                    .pseudonym_id,
                    require_active=False,
                )
            )

        except RegistrationManagerError:
            historical = (
                self.registration_manager
                .is_historical_pseudonym(
                    normalized_request
                    .pseudonym_id
                )
            )

            reason = (
                REASON_HISTORICAL_PSEUDONYM
                if historical
                else REASON_UNKNOWN_PSEUDONYM
            )

            message = (
                "The supplied subscriber pseudonym has expired."
                if historical
                else (
                    "No registered subscriber matches "
                    "the supplied pseudonym."
                )
            )

            return self._build_result(
                request=normalized_request,
                checked_at=selected_checked_at,
                request_age_seconds=request_age,
                verified=False,
                reason=reason,
                message=message,
                registration_id=None,
                subscriber_found=False,
                subscriber_active=False,
                freshness_valid=(
                    freshness_valid
                ),
                replay_free=False,
                details={
                    "historical_pseudonym": (
                        historical
                    ),
                },
            )

        if not subscriber.active:
            return self._build_result(
                request=normalized_request,
                checked_at=selected_checked_at,
                request_age_seconds=request_age,
                verified=False,
                reason=(
                    REASON_SUBSCRIBER_INACTIVE
                ),
                message=(
                    "The subscriber registration is inactive."
                ),
                registration_id=(
                    subscriber.registration_id
                ),
                subscriber_found=True,
                subscriber_active=False,
                freshness_valid=(
                    freshness_valid
                ),
                replay_free=False,
            )

        if not freshness_valid:
            message = (
                "The subscriber request timestamp is too far in the future."
                if (
                    freshness_reason
                    == REASON_FUTURE_TIMESTAMP
                )
                else (
                    "The subscriber request timestamp "
                    "is outside the freshness window."
                )
            )

            return self._build_result(
                request=normalized_request,
                checked_at=selected_checked_at,
                request_age_seconds=request_age,
                verified=False,
                reason=freshness_reason,
                message=message,
                registration_id=(
                    subscriber.registration_id
                ),
                subscriber_found=True,
                subscriber_active=True,
                freshness_valid=False,
                replay_free=False,
                details={
                    "freshness_window_seconds": (
                        self.freshness_window_seconds
                    ),
                    "future_tolerance_seconds": (
                        self.future_tolerance_seconds
                    ),
                },
            )

        replay_result: ReplayCheckResult = (
            self.replay_detector
            .check_and_record(
                pseudonym_id=(
                    normalized_request
                    .pseudonym_id
                ),
                session_id=(
                    normalized_request
                    .session_id
                ),
                attempt_number=(
                    normalized_request
                    .attempt_number
                ),
                nonce=(
                    normalized_request
                    .mobile_nonce
                ),
                message=(
                    normalized_request
                    .to_canonical_dict()
                ),
                observed_at=(
                    selected_checked_at
                ),
                metadata={
                    "request_fingerprint": (
                        normalized_request
                        .request_fingerprint
                    ),
                    "registration_id": (
                        subscriber
                        .registration_id
                    ),
                },
                record_on_accept=(
                    record_replay_on_accept
                ),
            )
        )

        if replay_result.replay_detected:
            return self._build_result(
                request=normalized_request,
                checked_at=selected_checked_at,
                request_age_seconds=request_age,
                verified=False,
                reason=(
                    REASON_REPLAY_DETECTED
                ),
                message=replay_result.message,
                registration_id=(
                    subscriber.registration_id
                ),
                subscriber_found=True,
                subscriber_active=True,
                freshness_valid=True,
                replay_free=False,
                replay_type=(
                    replay_result.replay_type
                ),
                details={
                    "replay_result": (
                        replay_result.to_dict()
                    ),
                },
            )

        return self._build_result(
            request=normalized_request,
            checked_at=selected_checked_at,
            request_age_seconds=request_age,
            verified=True,
            reason=REASON_VERIFIED,
            message=(
                "The FT-QuPAP subscriber request is valid."
            ),
            registration_id=(
                subscriber.registration_id
            ),
            subscriber_found=True,
            subscriber_active=True,
            freshness_valid=True,
            replay_free=True,
            replay_type=REPLAY_TYPE_NONE,
            details={
                "replay_recorded": (
                    replay_result.recorded
                ),
                "pseudonym_epoch": (
                    subscriber.pseudonym_epoch
                ),
            },
        )

    def require_verified_request(
        self,
        request: Mapping[str, Any],
        *,
        checked_at: int | None = None,
    ) -> SubscriberVerificationResult:
        """
        Verify an M1 request and raise when it cannot be accepted.
        """

        result = self.verify_request(
            request,
            checked_at=checked_at,
            record_replay_on_accept=True,
        )

        if not result.verified:
            raise SubscriberVerificationError(
                result.message,
                reason=result.reason,
                details=result.to_dict(),
            )

        return result

    def verify_and_get_subscriber(
        self,
        request: Mapping[str, Any],
        *,
        checked_at: int | None = None,
    ) -> tuple[
        SubscriberVerificationResult,
        SubscriberRegistrationRecord,
    ]:
        """
        Verify an M1 request and return its subscriber record.

        The returned registration record contains secret identity-key
        material and must remain inside protected server processing.
        """

        result = self.require_verified_request(
            request,
            checked_at=checked_at,
        )

        subscriber = (
            self.registration_manager
            .get_by_registration_id(
                result.registration_id
                or ""
            )
        )

        return (
            result,
            subscriber,
        )


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------

def verify_subscriber_request(
    request: Mapping[str, Any],
    *,
    checked_at: int | None = None,
    registration_manager: RegistrationManager | None = None,
    replay_detector: ReplayDetector | None = None,
    record_replay_on_accept: bool = True,
) -> SubscriberVerificationResult:
    """
    Verify an M1 request using a temporary verifier wrapper.
    """

    verifier = SubscriberVerifier(
        registration_manager=(
            registration_manager
        ),
        replay_detector=replay_detector,
    )

    return verifier.verify_request(
        request,
        checked_at=checked_at,
        record_replay_on_accept=(
            record_replay_on_accept
        ),
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_subscriber_verifier_self_test() -> dict[str, Any]:
    """
    Test valid M1 acceptance, replay rejection, retry acceptance,
    stale-request rejection, and inactive-subscriber rejection.
    """

    registration_manager = (
        RegistrationManager(
            storage_path=(
                "data/test/subscriber-verifier.json"
            ),
            auto_load=False,
        )
    )

    subscriber = (
        registration_manager
        .register_subscriber(
            permanent_identity=(
                "IMSI-470010000000099"
            ),
            pseudonym_id=(
                "PID-FTQ-SUBSCRIBER-VERIFY"
            ),
            identity_key=(
                b"K" * 32
            ),
            registered_at=(
                1_700_000_000
            ),
            registration_random_value=(
                b"R" * 16
            ),
            persist=False,
        )
    )

    replay_detector = ReplayDetector(
        replay_window_seconds=120,
        maximum_records=1000,
    )

    verifier = SubscriberVerifier(
        registration_manager=(
            registration_manager
        ),
        replay_detector=replay_detector,
        freshness_window_seconds=120,
        future_tolerance_seconds=5,
    )

    base_request = {
        "version": (
            SUBSCRIBER_REQUEST_VERSION
        ),
        "protocol_domain": (
            PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8"
            )
        ),
        "message_type": (
            SUBSCRIBER_REQUEST_MESSAGE_TYPE
        ),
        "pseudonym_id": (
            subscriber.pseudonym_id
        ),
        "session_id": (
            "FTQ-SUBSCRIBER-SESSION-001"
        ),
        "attempt_number": 1,
        "mobile_nonce": encode_base64(
            b"N" * 32
        ),
        "timestamp": 1_700_000_010,
        "metadata": {
            "context": "urban",
        },
    }

    valid_result = verifier.verify_request(
        base_request,
        checked_at=1_700_000_020,
    )

    replay_result = verifier.verify_request(
        base_request,
        checked_at=1_700_000_021,
    )

    retry_request = dict(
        base_request
    )

    retry_request.update(
        {
            "attempt_number": 2,
            "mobile_nonce": encode_base64(
                b"Q" * 32
            ),
            "timestamp": 1_700_000_022,
        }
    )

    retry_result = verifier.verify_request(
        retry_request,
        checked_at=1_700_000_023,
    )

    stale_request = dict(
        base_request
    )

    stale_request.update(
        {
            "session_id": (
                "FTQ-SUBSCRIBER-SESSION-STALE"
            ),
            "mobile_nonce": encode_base64(
                b"S" * 32
            ),
            "timestamp": 1_699_999_000,
        }
    )

    stale_result = verifier.verify_request(
        stale_request,
        checked_at=1_700_000_024,
    )

    registration_manager.deactivate_subscriber(
        registration_id=(
            subscriber.registration_id
        ),
        timestamp=1_700_000_025,
        persist=False,
    )

    inactive_request = dict(
        base_request
    )

    inactive_request.update(
        {
            "session_id": (
                "FTQ-SUBSCRIBER-SESSION-INACTIVE"
            ),
            "mobile_nonce": encode_base64(
                b"I" * 32
            ),
            "timestamp": 1_700_000_026,
        }
    )

    inactive_result = verifier.verify_request(
        inactive_request,
        checked_at=1_700_000_027,
    )

    public_output = valid_result.to_dict()

    secret_fields_absent = all(
        field_name not in public_output
        for field_name in (
            "identity_key",
            "permanent_identity_hash",
        )
    )

    success = all(
        (
            valid_result.verified,
            valid_result.subscriber_found,
            valid_result.subscriber_active,
            valid_result.freshness_valid,
            valid_result.replay_free,

            not replay_result.verified,
            not replay_result.replay_free,
            replay_result.reason
            == REASON_REPLAY_DETECTED,

            retry_result.verified,

            not stale_result.verified,
            stale_result.reason
            == REASON_STALE_TIMESTAMP,

            not inactive_result.verified,
            inactive_result.reason
            == REASON_SUBSCRIBER_INACTIVE,

            secret_fields_absent,
        )
    )

    return {
        "success": success,

        "valid_request_accepted": (
            valid_result.verified
        ),

        "subscriber_found": (
            valid_result.subscriber_found
        ),

        "subscriber_active": (
            valid_result.subscriber_active
        ),

        "freshness_valid": (
            valid_result.freshness_valid
        ),

        "replay_free": (
            valid_result.replay_free
        ),

        "duplicate_request_rejected": (
            not replay_result.verified
        ),

        "duplicate_request_reason": (
            replay_result.reason
        ),

        "retry_request_accepted": (
            retry_result.verified
        ),

        "stale_request_rejected": (
            not stale_result.verified
        ),

        "inactive_subscriber_rejected": (
            not inactive_result.verified
        ),

        "secret_fields_absent": (
            secret_fields_absent
        ),
    }


__all__ = [
    "SUBSCRIBER_REQUEST_VERSION",
    "SUBSCRIBER_REQUEST_MESSAGE_TYPE",
    "DEFAULT_SUBSCRIBER_FRESHNESS_WINDOW_SECONDS",
    "DEFAULT_FUTURE_TOLERANCE_SECONDS",
    "REASON_VERIFIED",
    "REASON_UNKNOWN_PSEUDONYM",
    "REASON_HISTORICAL_PSEUDONYM",
    "REASON_SUBSCRIBER_INACTIVE",
    "REASON_STALE_TIMESTAMP",
    "REASON_FUTURE_TIMESTAMP",
    "REASON_REPLAY_DETECTED",
    "SubscriberVerificationError",
    "NormalizedSubscriberRequest",
    "SubscriberVerificationResult",
    "normalize_mobile_nonce",
    "normalize_subscriber_request",
    "calculate_subscriber_request_fingerprint",
    "check_request_freshness",
    "SubscriberVerifier",
    "verify_subscriber_request",
    "run_subscriber_verifier_self_test",
]