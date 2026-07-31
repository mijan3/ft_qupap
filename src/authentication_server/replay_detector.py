"""
Replay detection for FT-QuPAP v5.1.

The Authentication Server rejects authentication messages that reuse:

1. A previously accepted subscriber nonce.
2. The same session and attempt combination.
3. An identical authenticated-message fingerprint.

Retries are allowed only when they use a new attempt number, a fresh
nonce, and a newly generated message.

Timestamp freshness is checked separately by `freshness_checker.py`.
This module records only replay evidence that remains valid during the
configured replay-detection window.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.common.constants import (
    FRESHNESS_WINDOW_SECONDS,
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
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

DEFAULT_REPLAY_WINDOW_SECONDS = max(
    int(FRESHNESS_WINDOW_SECONDS),
    60,
)

MINIMUM_REPLAY_WINDOW_SECONDS = 10

MAXIMUM_REPLAY_WINDOW_SECONDS = 86_400

DEFAULT_MAXIMUM_REPLAY_RECORDS = 100_000

REPLAY_FINGERPRINT_ALGORITHM = "SHA3-256"

REPLAY_TYPE_NONE = "none"

REPLAY_TYPE_NONCE = "nonce_reuse"

REPLAY_TYPE_SESSION_ATTEMPT = "session_attempt_reuse"

REPLAY_TYPE_MESSAGE = "message_reuse"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class ReplayDetectionError(RuntimeError):
    """Raised when replayed protocol data is detected."""

    def __init__(
        self,
        message: str,
        *,
        replay_type: str = REPLAY_TYPE_NONE,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.replay_type = replay_type

        self.details = (
            {}
            if details is None
            else dict(details)
        )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayRecord:
    """
    One stored replay-detection record.
    """

    token_type: str
    token_fingerprint: str

    pseudonym_id: str
    session_id: str
    attempt_number: int

    nonce_fingerprint: str
    message_fingerprint: str

    first_seen_at: int
    expires_at: int

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.token_type not in {
            REPLAY_TYPE_NONCE,
            REPLAY_TYPE_SESSION_ATTEMPT,
            REPLAY_TYPE_MESSAGE,
        }:
            raise ProtocolValidationError(
                "Unsupported replay-record token type."
            )

        _validate_hex_digest(
            self.token_fingerprint,
            field_name="token_fingerprint",
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
            maximum=1_000_000,
        )

        _validate_hex_digest(
            self.nonce_fingerprint,
            field_name="nonce_fingerprint",
        )

        _validate_hex_digest(
            self.message_fingerprint,
            field_name="message_fingerprint",
        )

        validate_integer(
            self.first_seen_at,
            field_name="first_seen_at",
            minimum=0,
        )

        validate_integer(
            self.expires_at,
            field_name="expires_at",
            minimum=0,
        )

        if self.expires_at <= self.first_seen_at:
            raise ProtocolValidationError(
                (
                    "Replay-record expiration must be later "
                    "than its first-seen timestamp."
                )
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise ProtocolValidationError(
                "Replay-record metadata must be a dictionary."
            )

    def is_expired(
        self,
        timestamp: int,
    ) -> bool:
        """Return True when this replay record has expired."""

        validated_timestamp = validate_integer(
            timestamp,
            field_name="timestamp",
            minimum=0,
        )

        return (
            validated_timestamp
            >= self.expires_at
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record dictionary."""

        return asdict(
            self
        )


@dataclass(frozen=True)
class ReplayCheckResult:
    """
    Result of one replay-detection operation.
    """

    accepted: bool
    replay_detected: bool
    recorded: bool

    replay_type: str

    pseudonym_id: str
    session_id: str
    attempt_number: int

    nonce_fingerprint: str
    message_fingerprint: str

    checked_at: int
    first_seen_at: int | None
    expires_at: int | None

    message: str

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "accepted",
            "replay_detected",
            "recorded",
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

        if self.accepted == self.replay_detected:
            raise ProtocolValidationError(
                (
                    "Exactly one of accepted or replay_detected "
                    "must be True."
                )
            )

        allowed_types = {
            REPLAY_TYPE_NONE,
            REPLAY_TYPE_NONCE,
            REPLAY_TYPE_SESSION_ATTEMPT,
            REPLAY_TYPE_MESSAGE,
        }

        if self.replay_type not in allowed_types:
            raise ProtocolValidationError(
                "Unsupported replay result type."
            )

        if (
            self.accepted
            and self.replay_type != REPLAY_TYPE_NONE
        ):
            raise ProtocolValidationError(
                (
                    "An accepted replay check cannot contain "
                    "a replay type."
                )
            )

        if (
            self.replay_detected
            and self.replay_type == REPLAY_TYPE_NONE
        ):
            raise ProtocolValidationError(
                (
                    "A rejected replay check must identify "
                    "its replay type."
                )
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
            maximum=1_000_000,
        )

        _validate_hex_digest(
            self.nonce_fingerprint,
            field_name="nonce_fingerprint",
        )

        _validate_hex_digest(
            self.message_fingerprint,
            field_name="message_fingerprint",
        )

        validate_integer(
            self.checked_at,
            field_name="checked_at",
            minimum=0,
        )

        if self.first_seen_at is not None:
            validate_integer(
                self.first_seen_at,
                field_name="first_seen_at",
                minimum=0,
            )

        if self.expires_at is not None:
            validate_integer(
                self.expires_at,
                field_name="expires_at",
                minimum=0,
            )

        if not isinstance(
            self.message,
            str,
        ):
            raise ProtocolValidationError(
                "Replay result message must be a string."
            )

        if not isinstance(
            self.details,
            dict,
        ):
            raise ProtocolValidationError(
                "Replay result details must be a dictionary."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        return asdict(
            self
        )


# ---------------------------------------------------------------------
# Validation and fingerprint helpers
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
            f"{field_name} must be valid hexadecimal text."
        ) from exc

    return validated


def normalize_nonce(
    nonce: bytes | str,
) -> bytes:
    """
    Normalize a protocol nonce.

    Accepted representations:

    - Raw bytes
    - Hexadecimal text

    The nonce must contain between 16 and 64 bytes.
    """

    if isinstance(
        nonce,
        bytes,
    ):
        return validate_bytes(
            nonce,
            field_name="nonce",
            minimum_length=16,
            maximum_length=64,
        )

    if not isinstance(
        nonce,
        str,
    ):
        raise ProtocolValidationError(
            "nonce must be bytes or hexadecimal text."
        )

    normalized_text = validate_non_empty_string(
        nonce,
        field_name="nonce",
        minimum_length=32,
        maximum_length=128,
    ).strip().lower()

    if len(normalized_text) % 2 != 0:
        raise ProtocolValidationError(
            "Hexadecimal nonce length must be even."
        )

    try:
        nonce_bytes = bytes.fromhex(
            normalized_text
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            "nonce must contain valid hexadecimal text."
        ) from exc

    return validate_bytes(
        nonce_bytes,
        field_name="nonce",
        minimum_length=16,
        maximum_length=64,
    )


def calculate_nonce_fingerprint(
    nonce: bytes | str,
) -> str:
    """Calculate a SHA3-256 nonce fingerprint."""

    normalized_nonce = normalize_nonce(
        nonce
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00replay-nonce\x00"
    )

    digest.update(
        normalized_nonce
    )

    return digest.hexdigest()


def normalize_message_bytes(
    message: bytes | Mapping[str, Any],
) -> bytes:
    """
    Convert an authenticated message into canonical bytes.
    """

    if isinstance(
        message,
        bytes,
    ):
        return validate_bytes(
            message,
            field_name="message",
            minimum_length=1,
            maximum_length=10_000_000,
        )

    if isinstance(
        message,
        Mapping,
    ):
        encoded = canonical_json_bytes(
            dict(message)
        )

        return validate_bytes(
            encoded,
            field_name="canonical_message",
            minimum_length=1,
            maximum_length=10_000_000,
        )

    raise ProtocolValidationError(
        "message must be bytes or a mapping."
    )


def calculate_message_fingerprint(
    message: bytes | Mapping[str, Any],
) -> str:
    """Calculate a SHA3-256 authenticated-message fingerprint."""

    message_bytes = normalize_message_bytes(
        message
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00replay-message\x00"
    )

    digest.update(
        message_bytes
    )

    return digest.hexdigest()


def calculate_session_attempt_fingerprint(
    *,
    pseudonym_id: str,
    session_id: str,
    attempt_number: int,
) -> str:
    """
    Calculate a subscriber-bound session-attempt fingerprint.

    A later retry with a larger attempt number receives a different
    fingerprint and is therefore not rejected solely because it uses
    the same session ID.
    """

    validated_pseudonym = validate_pseudonym_id(
        pseudonym_id
    )

    validated_session_id = (
        validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=256,
        )
    )

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=1_000_000,
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00replay-session-attempt\x00"
    )

    digest.update(
        validated_pseudonym.encode(
            "utf-8"
        )
    )

    digest.update(
        b"\x00"
    )

    digest.update(
        validated_session_id.encode(
            "utf-8"
        )
    )

    digest.update(
        validated_attempt.to_bytes(
            8,
            byteorder="big",
            signed=False,
        )
    )

    return digest.hexdigest()


def calculate_subscriber_nonce_token(
    *,
    pseudonym_id: str,
    nonce_fingerprint: str,
) -> str:
    """
    Bind a nonce fingerprint to one pseudonymous subscriber.
    """

    validated_pseudonym = validate_pseudonym_id(
        pseudonym_id
    )

    validated_nonce_fingerprint = (
        _validate_hex_digest(
            nonce_fingerprint,
            field_name="nonce_fingerprint",
        )
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00subscriber-nonce-token\x00"
    )

    digest.update(
        validated_pseudonym.encode(
            "utf-8"
        )
    )

    digest.update(
        bytes.fromhex(
            validated_nonce_fingerprint
        )
    )

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Replay detector
# ---------------------------------------------------------------------

class ReplayDetector:
    """
    Thread-safe in-memory FT-QuPAP replay detector.
    """

    def __init__(
        self,
        *,
        replay_window_seconds: int = (
            DEFAULT_REPLAY_WINDOW_SECONDS
        ),
        maximum_records: int = (
            DEFAULT_MAXIMUM_REPLAY_RECORDS
        ),
    ) -> None:
        self.replay_window_seconds = (
            validate_integer(
                replay_window_seconds,
                field_name="replay_window_seconds",
                minimum=(
                    MINIMUM_REPLAY_WINDOW_SECONDS
                ),
                maximum=(
                    MAXIMUM_REPLAY_WINDOW_SECONDS
                ),
            )
        )

        self.maximum_records = validate_integer(
            maximum_records,
            field_name="maximum_records",
            minimum=100,
            maximum=10_000_000,
        )

        self._nonce_records: dict[
            str,
            ReplayRecord,
        ] = {}

        self._session_records: dict[
            str,
            ReplayRecord,
        ] = {}

        self._message_records: dict[
            str,
            ReplayRecord,
        ] = {}

        self._lock = threading.RLock()

    def _all_record_count(self) -> int:
        """Return the number of stored replay tokens."""

        return (
            len(self._nonce_records)
            + len(self._session_records)
            + len(self._message_records)
        )

    def purge_expired(
        self,
        *,
        timestamp: int | None = None,
    ) -> int:
        """
        Delete expired replay evidence.

        Returns the total number of removed token records.
        """

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        removed = 0

        with self._lock:
            for registry in (
                self._nonce_records,
                self._session_records,
                self._message_records,
            ):
                expired_tokens = [
                    token
                    for token, record
                    in registry.items()
                    if record.is_expired(
                        selected_timestamp
                    )
                ]

                for token in expired_tokens:
                    registry.pop(
                        token,
                        None,
                    )

                    removed += 1

        return removed

    def clear(self) -> None:
        """Delete all replay-detection evidence."""

        with self._lock:
            self._nonce_records.clear()
            self._session_records.clear()
            self._message_records.clear()

    def record_count(
        self,
        *,
        purge_expired: bool = True,
        timestamp: int | None = None,
    ) -> int:
        """Return the total number of stored replay tokens."""

        if not isinstance(
            purge_expired,
            bool,
        ):
            raise ProtocolValidationError(
                "purge_expired must be Boolean."
            )

        if purge_expired:
            self.purge_expired(
                timestamp=timestamp
            )

        with self._lock:
            return self._all_record_count()

    def _enforce_capacity(
        self,
        *,
        timestamp: int,
    ) -> None:
        """
        Purge expired data and reject unsafe cache growth.
        """

        self.purge_expired(
            timestamp=timestamp
        )

        if (
            self._all_record_count() + 3
            > self.maximum_records
        ):
            raise ReplayDetectionError(
                (
                    "Replay-detection storage capacity "
                    "has been reached."
                ),
                details={
                    "maximum_records": (
                        self.maximum_records
                    ),
                    "current_records": (
                        self._all_record_count()
                    ),
                },
            )

    @staticmethod
    def _build_result_from_replay(
        *,
        replay_type: str,
        record: ReplayRecord,
        checked_at: int,
    ) -> ReplayCheckResult:
        """Build a rejected replay-check result."""

        messages = {
            REPLAY_TYPE_NONCE: (
                "Authentication request rejected because "
                "the subscriber nonce was previously used."
            ),
            REPLAY_TYPE_SESSION_ATTEMPT: (
                "Authentication request rejected because "
                "the session attempt was previously processed."
            ),
            REPLAY_TYPE_MESSAGE: (
                "Authentication request rejected because "
                "an identical authenticated message was "
                "previously processed."
            ),
        }

        return ReplayCheckResult(
            accepted=False,
            replay_detected=True,
            recorded=False,
            replay_type=replay_type,
            pseudonym_id=record.pseudonym_id,
            session_id=record.session_id,
            attempt_number=record.attempt_number,
            nonce_fingerprint=(
                record.nonce_fingerprint
            ),
            message_fingerprint=(
                record.message_fingerprint
            ),
            checked_at=checked_at,
            first_seen_at=(
                record.first_seen_at
            ),
            expires_at=record.expires_at,
            message=messages[
                replay_type
            ],
            details={
                "matched_token_type": (
                    record.token_type
                ),
                "matched_token_fingerprint": (
                    record.token_fingerprint
                ),
            },
        )

    def check_and_record(
        self,
        *,
        pseudonym_id: str,
        session_id: str,
        attempt_number: int,
        nonce: bytes | str,
        message: bytes | Mapping[str, Any],
        observed_at: int | None = None,
        replay_window_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        record_on_accept: bool = True,
    ) -> ReplayCheckResult:
        """
        Check and optionally record one authentication message.

        The operation is atomic. Concurrent duplicate requests cannot
        both be accepted.
        """

        validated_pseudonym = (
            validate_pseudonym_id(
                pseudonym_id
            )
        )

        validated_session_id = (
            validate_non_empty_string(
                session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=256,
            )
        )

        validated_attempt = validate_integer(
            attempt_number,
            field_name="attempt_number",
            minimum=1,
            maximum=1_000_000,
        )

        selected_timestamp = (
            current_timestamp()
            if observed_at is None
            else validate_integer(
                observed_at,
                field_name="observed_at",
                minimum=0,
            )
        )

        selected_window = (
            self.replay_window_seconds
            if replay_window_seconds is None
            else validate_integer(
                replay_window_seconds,
                field_name="replay_window_seconds",
                minimum=(
                    MINIMUM_REPLAY_WINDOW_SECONDS
                ),
                maximum=(
                    MAXIMUM_REPLAY_WINDOW_SECONDS
                ),
            )
        )

        if not isinstance(
            record_on_accept,
            bool,
        ):
            raise ProtocolValidationError(
                "record_on_accept must be Boolean."
            )

        normalized_metadata = (
            {}
            if metadata is None
            else dict(metadata)
        )

        nonce_fingerprint = (
            calculate_nonce_fingerprint(
                nonce
            )
        )

        message_fingerprint = (
            calculate_message_fingerprint(
                message
            )
        )

        nonce_token = (
            calculate_subscriber_nonce_token(
                pseudonym_id=(
                    validated_pseudonym
                ),
                nonce_fingerprint=(
                    nonce_fingerprint
                ),
            )
        )

        session_token = (
            calculate_session_attempt_fingerprint(
                pseudonym_id=(
                    validated_pseudonym
                ),
                session_id=(
                    validated_session_id
                ),
                attempt_number=(
                    validated_attempt
                ),
            )
        )

        expires_at = (
            selected_timestamp
            + selected_window
        )

        with self._lock:
            self.purge_expired(
                timestamp=selected_timestamp
            )

            existing_nonce = (
                self._nonce_records.get(
                    nonce_token
                )
            )

            if existing_nonce is not None:
                return self._build_result_from_replay(
                    replay_type=(
                        REPLAY_TYPE_NONCE
                    ),
                    record=existing_nonce,
                    checked_at=(
                        selected_timestamp
                    ),
                )

            existing_session = (
                self._session_records.get(
                    session_token
                )
            )

            if existing_session is not None:
                return self._build_result_from_replay(
                    replay_type=(
                        REPLAY_TYPE_SESSION_ATTEMPT
                    ),
                    record=existing_session,
                    checked_at=(
                        selected_timestamp
                    ),
                )

            existing_message = (
                self._message_records.get(
                    message_fingerprint
                )
            )

            if existing_message is not None:
                return self._build_result_from_replay(
                    replay_type=(
                        REPLAY_TYPE_MESSAGE
                    ),
                    record=existing_message,
                    checked_at=(
                        selected_timestamp
                    ),
                )

            if record_on_accept:
                self._enforce_capacity(
                    timestamp=selected_timestamp
                )

                common_fields = {
                    "pseudonym_id": (
                        validated_pseudonym
                    ),
                    "session_id": (
                        validated_session_id
                    ),
                    "attempt_number": (
                        validated_attempt
                    ),
                    "nonce_fingerprint": (
                        nonce_fingerprint
                    ),
                    "message_fingerprint": (
                        message_fingerprint
                    ),
                    "first_seen_at": (
                        selected_timestamp
                    ),
                    "expires_at": expires_at,
                    "metadata": (
                        normalized_metadata
                    ),
                }

                self._nonce_records[
                    nonce_token
                ] = ReplayRecord(
                    token_type=(
                        REPLAY_TYPE_NONCE
                    ),
                    token_fingerprint=(
                        nonce_token
                    ),
                    **common_fields,
                )

                self._session_records[
                    session_token
                ] = ReplayRecord(
                    token_type=(
                        REPLAY_TYPE_SESSION_ATTEMPT
                    ),
                    token_fingerprint=(
                        session_token
                    ),
                    **common_fields,
                )

                self._message_records[
                    message_fingerprint
                ] = ReplayRecord(
                    token_type=(
                        REPLAY_TYPE_MESSAGE
                    ),
                    token_fingerprint=(
                        message_fingerprint
                    ),
                    **common_fields,
                )

            return ReplayCheckResult(
                accepted=True,
                replay_detected=False,
                recorded=record_on_accept,
                replay_type=REPLAY_TYPE_NONE,
                pseudonym_id=(
                    validated_pseudonym
                ),
                session_id=(
                    validated_session_id
                ),
                attempt_number=(
                    validated_attempt
                ),
                nonce_fingerprint=(
                    nonce_fingerprint
                ),
                message_fingerprint=(
                    message_fingerprint
                ),
                checked_at=selected_timestamp,
                first_seen_at=(
                    selected_timestamp
                    if record_on_accept
                    else None
                ),
                expires_at=(
                    expires_at
                    if record_on_accept
                    else None
                ),
                message=(
                    "Authentication request contains no "
                    "previously observed replay evidence."
                ),
                details={
                    "replay_window_seconds": (
                        selected_window
                    ),
                },
            )

    def require_not_replayed(
        self,
        *,
        pseudonym_id: str,
        session_id: str,
        attempt_number: int,
        nonce: bytes | str,
        message: bytes | Mapping[str, Any],
        observed_at: int | None = None,
        replay_window_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReplayCheckResult:
        """
        Check, record, and raise when replay evidence is detected.
        """

        result = self.check_and_record(
            pseudonym_id=pseudonym_id,
            session_id=session_id,
            attempt_number=attempt_number,
            nonce=nonce,
            message=message,
            observed_at=observed_at,
            replay_window_seconds=(
                replay_window_seconds
            ),
            metadata=metadata,
            record_on_accept=True,
        )

        if result.replay_detected:
            raise ReplayDetectionError(
                result.message,
                replay_type=result.replay_type,
                details=result.to_dict(),
            )

        return result

    def has_seen_nonce(
        self,
        *,
        pseudonym_id: str,
        nonce: bytes | str,
        timestamp: int | None = None,
    ) -> bool:
        """
        Return True when this subscriber nonce is still recorded.
        """

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        nonce_fingerprint = (
            calculate_nonce_fingerprint(
                nonce
            )
        )

        nonce_token = (
            calculate_subscriber_nonce_token(
                pseudonym_id=pseudonym_id,
                nonce_fingerprint=(
                    nonce_fingerprint
                ),
            )
        )

        with self._lock:
            self.purge_expired(
                timestamp=selected_timestamp
            )

            return (
                nonce_token
                in self._nonce_records
            )

    def list_records(
        self,
        *,
        timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return active replay records for diagnostics.

        Only fingerprints are returned. Raw nonces and complete messages
        are never stored.
        """

        self.purge_expired(
            timestamp=timestamp
        )

        with self._lock:
            records = (
                list(
                    self._nonce_records.values()
                )
                + list(
                    self._session_records.values()
                )
                + list(
                    self._message_records.values()
                )
            )

            records.sort(
                key=lambda record: (
                    record.first_seen_at,
                    record.token_type,
                )
            )

            return [
                record.to_dict()
                for record in records
            ]


# ---------------------------------------------------------------------
# Default detector
# ---------------------------------------------------------------------

_DEFAULT_REPLAY_DETECTOR: ReplayDetector | None = None

_DEFAULT_DETECTOR_LOCK = threading.RLock()


def get_default_replay_detector() -> ReplayDetector:
    """Return the process-wide replay detector."""

    global _DEFAULT_REPLAY_DETECTOR

    with _DEFAULT_DETECTOR_LOCK:
        if _DEFAULT_REPLAY_DETECTOR is None:
            _DEFAULT_REPLAY_DETECTOR = (
                ReplayDetector()
            )

        return _DEFAULT_REPLAY_DETECTOR


def reset_default_replay_detector() -> None:
    """Reset the process-wide replay detector."""

    global _DEFAULT_REPLAY_DETECTOR

    with _DEFAULT_DETECTOR_LOCK:
        _DEFAULT_REPLAY_DETECTOR = None


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_replay_detector_self_test() -> dict[str, Any]:
    """
    Test first-use acceptance, replay rejection, retry handling,
    subscriber binding, and record expiration.
    """

    detector = ReplayDetector(
        replay_window_seconds=60,
        maximum_records=1_000,
    )

    pseudonym = "PID-FTQ-REPLAY-SELF-TEST"

    nonce_one = bytes(
        range(16)
    )

    nonce_two = bytes(
        range(
            16,
            32,
        )
    )

    nonce_three = b"\xA5" * 16

    message_one = {
        "type": "M1",
        "session_id": "SESSION-REPLAY-001",
        "attempt_number": 1,
        "nonce": nonce_one.hex(),
    }

    first_result = detector.check_and_record(
        pseudonym_id=pseudonym,
        session_id="SESSION-REPLAY-001",
        attempt_number=1,
        nonce=nonce_one,
        message=message_one,
        observed_at=1_700_000_000,
    )

    exact_replay_result = (
        detector.check_and_record(
            pseudonym_id=pseudonym,
            session_id="SESSION-REPLAY-001",
            attempt_number=1,
            nonce=nonce_one,
            message=message_one,
            observed_at=1_700_000_001,
        )
    )

    retry_message = {
        "type": "M1",
        "session_id": "SESSION-REPLAY-001",
        "attempt_number": 2,
        "nonce": nonce_two.hex(),
    }

    retry_result = detector.check_and_record(
        pseudonym_id=pseudonym,
        session_id="SESSION-REPLAY-001",
        attempt_number=2,
        nonce=nonce_two,
        message=retry_message,
        observed_at=1_700_000_002,
    )

    reused_session_attempt_result = (
        detector.check_and_record(
            pseudonym_id=pseudonym,
            session_id="SESSION-REPLAY-001",
            attempt_number=2,
            nonce=nonce_three,
            message={
                "type": "M1",
                "session_id": (
                    "SESSION-REPLAY-001"
                ),
                "attempt_number": 2,
                "nonce": nonce_three.hex(),
            },
            observed_at=1_700_000_003,
        )
    )

    other_subscriber_result = (
        detector.check_and_record(
            pseudonym_id=(
                "PID-FTQ-REPLAY-OTHER"
            ),
            session_id=(
                "SESSION-REPLAY-OTHER"
            ),
            attempt_number=1,
            nonce=nonce_one,
            message={
                "type": "M1",
                "session_id": (
                    "SESSION-REPLAY-OTHER"
                ),
                "attempt_number": 1,
                "nonce": nonce_one.hex(),
            },
            observed_at=1_700_000_004,
        )
    )

    expired_count = detector.purge_expired(
        timestamp=1_700_000_061
    )

    accepted_after_expiration = (
        detector.check_and_record(
            pseudonym_id=pseudonym,
            session_id="SESSION-REPLAY-001",
            attempt_number=1,
            nonce=nonce_one,
            message=message_one,
            observed_at=1_700_000_062,
        )
    )

    success = all(
        (
            first_result.accepted,
            first_result.recorded,

            exact_replay_result
            .replay_detected,

            exact_replay_result.replay_type
            == REPLAY_TYPE_NONCE,

            retry_result.accepted,

            reused_session_attempt_result
            .replay_detected,

            reused_session_attempt_result
            .replay_type
            == REPLAY_TYPE_SESSION_ATTEMPT,

            other_subscriber_result.accepted,

            expired_count > 0,

            accepted_after_expiration.accepted,
        )
    )

    return {
        "success": success,

        "first_request_accepted": (
            first_result.accepted
        ),

        "exact_replay_rejected": (
            exact_replay_result
            .replay_detected
        ),

        "exact_replay_type": (
            exact_replay_result
            .replay_type
        ),

        "retry_with_new_nonce_accepted": (
            retry_result.accepted
        ),

        "reused_session_attempt_rejected": (
            reused_session_attempt_result
            .replay_detected
        ),

        "session_replay_type": (
            reused_session_attempt_result
            .replay_type
        ),

        "same_nonce_other_subscriber_accepted": (
            other_subscriber_result
            .accepted
        ),

        "expired_records_removed": (
            expired_count
        ),

        "accepted_after_expiration": (
            accepted_after_expiration
            .accepted
        ),

        "active_record_count": (
            detector.record_count(
                timestamp=1_700_000_062
            )
        ),
    }


__all__ = [
    "DEFAULT_REPLAY_WINDOW_SECONDS",
    "MINIMUM_REPLAY_WINDOW_SECONDS",
    "MAXIMUM_REPLAY_WINDOW_SECONDS",
    "DEFAULT_MAXIMUM_REPLAY_RECORDS",
    "REPLAY_FINGERPRINT_ALGORITHM",
    "REPLAY_TYPE_NONE",
    "REPLAY_TYPE_NONCE",
    "REPLAY_TYPE_SESSION_ATTEMPT",
    "REPLAY_TYPE_MESSAGE",
    "ReplayDetectionError",
    "ReplayRecord",
    "ReplayCheckResult",
    "normalize_nonce",
    "calculate_nonce_fingerprint",
    "normalize_message_bytes",
    "calculate_message_fingerprint",
    "calculate_session_attempt_fingerprint",
    "calculate_subscriber_nonce_token",
    "ReplayDetector",
    "get_default_replay_detector",
    "reset_default_replay_detector",
    "run_replay_detector_self_test",
]