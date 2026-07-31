"""
KMAC authentication-tag verification for FT-QuPAP v5.1.

After Steane syndrome correction and payload decoding, the
Authentication Server recovers a 128-bit classical authentication tag
from the quantum payload blocks.

This module:

1. Builds the transcript-bound KMAC verification message.
2. Recomputes the expected KMAC tag using the derived session key.
3. Compares the recovered and expected tags in constant time.
4. Rejects altered tags, transcripts, nonces, sessions, and retries.
5. Returns secret-safe diagnostic information.

The recovered and expected KMAC tags are never included in normal public
output or object representations.
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from src.authentication_server.payload_decoder import (
    PayloadDecodingResult,
)

from src.authentication_server.session_key_derivation import (
    KMAC_AUTHENTICATION_KEY_BYTES,
    normalize_nonce,
    normalize_transcript_digest,
)

from src.common.constants import (
    KMAC_TAG_BITS,
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
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

from src.cryptography import kmac_module


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

KMAC_TAG_BYTES = KMAC_TAG_BITS // 8

KMAC_VERIFICATION_CUSTOMIZATION = (
    b"FT-QuPAP-v5.1/mobile-authentication-tag"
)

KMAC_TAG_FINGERPRINT_ALGORITHM = "SHA3-256"

KMAC_COMPARISON_ALGORITHM = "constant-time-compare"


# ---------------------------------------------------------------------
# Result reasons
# ---------------------------------------------------------------------

TAG_REASON_VALID = "kmac_tag_valid"

TAG_REASON_MISMATCH = "kmac_tag_mismatch"

TAG_REASON_INVALID_LENGTH = "invalid_kmac_tag_length"

TAG_REASON_GENERATION_FAILED = "kmac_generation_failed"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class TagVerificationError(RuntimeError):
    """Raised when KMAC tag verification cannot be completed safely."""

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
# Verification result
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TagVerificationResult:
    """
    Result of FT-QuPAP KMAC authentication-tag verification.

    The actual received and expected tag values are intentionally
    excluded from this result.
    """

    valid: bool

    reason: str
    message: str

    session_id: str
    attempt_number: int
    pseudonym_id: str

    tag_bits: int
    tag_bytes: int

    received_tag_fingerprint: str
    expected_tag_fingerprint: str

    verification_message_fingerprint: str
    transcript_digest: str

    comparison_algorithm: str

    verified_at: int

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.valid,
            bool,
        ):
            raise ProtocolValidationError(
                "valid must be Boolean."
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

        validate_pseudonym_id(
            self.pseudonym_id
        )

        validate_integer(
            self.tag_bits,
            field_name="tag_bits",
            minimum=8,
        )

        validate_integer(
            self.tag_bytes,
            field_name="tag_bytes",
            minimum=1,
        )

        if self.tag_bits != self.tag_bytes * 8:
            raise ProtocolValidationError(
                "tag_bits does not match tag_bytes."
            )

        _validate_hex_digest(
            self.received_tag_fingerprint,
            field_name="received_tag_fingerprint",
        )

        _validate_hex_digest(
            self.expected_tag_fingerprint,
            field_name="expected_tag_fingerprint",
        )

        _validate_hex_digest(
            self.verification_message_fingerprint,
            field_name=(
                "verification_message_fingerprint"
            ),
        )

        _validate_hex_digest(
            self.transcript_digest,
            field_name="transcript_digest",
        )

        validate_non_empty_string(
            self.comparison_algorithm,
            field_name="comparison_algorithm",
            minimum_length=1,
            maximum_length=128,
        )

        validate_integer(
            self.verified_at,
            field_name="verified_at",
            minimum=0,
        )

        if not isinstance(
            self.details,
            dict,
        ):
            raise ProtocolValidationError(
                "details must be a dictionary."
            )

        expected_reason = (
            TAG_REASON_VALID
            if self.valid
            else TAG_REASON_MISMATCH
        )

        if self.reason != expected_reason:
            raise ProtocolValidationError(
                (
                    "Tag-verification reason does not "
                    "match the validity result."
                )
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return secret-safe verification information.
        """

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
    """Validate a SHA3-256 hexadecimal fingerprint."""

    validated = (
        validate_non_empty_string(
            value,
            field_name=field_name,
            minimum_length=64,
            maximum_length=64,
        )
        .strip()
        .lower()
    )

    try:
        bytes.fromhex(
            validated
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            f"{field_name} must contain hexadecimal text."
        ) from exc

    return validated


def normalize_kmac_tag(
    tag: bytes | bytearray | memoryview,
    *,
    field_name: str = "kmac_tag",
) -> bytes:
    """
    Normalize a 128-bit FT-QuPAP KMAC tag.
    """

    if not isinstance(
        tag,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise ProtocolValidationError(
            f"{field_name} must be bytes."
        )

    return validate_bytes(
        bytes(tag),
        field_name=field_name,
        exact_length=KMAC_TAG_BYTES,
    )


def extract_recovered_tag(
    recovered_payload: (
        bytes
        | bytearray
        | memoryview
        | PayloadDecodingResult
    ),
) -> bytes:
    """
    Extract the recovered KMAC tag from raw bytes or a payload result.
    """

    if isinstance(
        recovered_payload,
        PayloadDecodingResult,
    ):
        if not recovered_payload.complete:
            raise TagVerificationError(
                (
                    "The quantum payload is incomplete and "
                    "cannot be verified."
                ),
                reason=TAG_REASON_INVALID_LENGTH,
                details=(
                    recovered_payload.public_dict()
                ),
            )

        return normalize_kmac_tag(
            recovered_payload.payload,
            field_name="recovered_kmac_tag",
        )

    return normalize_kmac_tag(
        bytes(recovered_payload),
        field_name="recovered_kmac_tag",
    )


def calculate_tag_fingerprint(
    tag: bytes,
) -> str:
    """
    Calculate a SHA3-256 fingerprint of a KMAC tag.
    """

    validated_tag = normalize_kmac_tag(
        tag
    )

    return hashlib.sha3_256(
        validated_tag
    ).hexdigest()


def calculate_verification_message_fingerprint(
    message: bytes,
) -> str:
    """
    Calculate a SHA3-256 verification-message fingerprint.
    """

    validated_message = validate_bytes(
        message,
        field_name="verification_message",
        minimum_length=1,
        maximum_length=10_000_000,
    )

    return hashlib.sha3_256(
        validated_message
    ).hexdigest()


# ---------------------------------------------------------------------
# KMAC verification-message construction
# ---------------------------------------------------------------------

def build_kmac_verification_message(
    *,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """
    Build the canonical message authenticated by the KMAC tag.

    The verification message binds the tag to:

    - Protocol domain
    - Session identifier
    - Authentication-attempt number
    - Subscriber pseudonym
    - Mobile Station nonce
    - Authentication Server nonce
    - Complete transcript digest
    """

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
        maximum=100,
    )

    validated_pseudonym = (
        validate_pseudonym_id(
            pseudonym_id
        )
    )

    normalized_mobile_nonce = normalize_nonce(
        mobile_nonce,
        field_name="mobile_nonce",
    )

    normalized_server_nonce = normalize_nonce(
        server_nonce,
        field_name="server_nonce",
    )

    normalized_transcript = (
        normalize_transcript_digest(
            transcript_digest
        )
    )

    normalized_metadata = (
        {}
        if metadata is None
        else dict(metadata)
    )

    message = {
        "protocol_domain": (
            PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8",
                errors="strict",
            )
        ),

        "purpose": (
            "mobile-station-authentication-tag"
        ),

        "session_id": (
            validated_session_id
        ),

        "attempt_number": (
            validated_attempt
        ),

        "pseudonym_id": (
            validated_pseudonym
        ),

        "mobile_nonce": encode_base64(
            normalized_mobile_nonce
        ),

        "server_nonce": encode_base64(
            normalized_server_nonce
        ),

        "transcript_digest": (
            normalized_transcript.hex()
        ),

        "tag_bits": KMAC_TAG_BITS,

        "metadata": normalized_metadata,
    }

    return canonical_json_bytes(
        message
    )


# ---------------------------------------------------------------------
# KMAC backend compatibility helpers
# ---------------------------------------------------------------------

def _find_kmac_generation_function() -> Callable[..., Any]:
    """
    Locate a compatible KMAC generation function.
    """

    function_names = (
        "generate_kmac_tag",
        "compute_kmac_tag",
        "create_kmac_tag",
        "kmac256_tag",
        "kmac_tag",
        "generate_tag",
    )

    for function_name in function_names:
        function = getattr(
            kmac_module,
            function_name,
            None,
        )

        if callable(function):
            return function

    raise TagVerificationError(
        (
            "No compatible KMAC generation function "
            "was found in kmac_module."
        ),
        reason=TAG_REASON_GENERATION_FAILED,
        details={
            "module": "src.cryptography.kmac_module",
            "searched_functions": list(
                function_names
            ),
        },
    )


def _invoke_compatible_call(
    function: Callable[..., Any],
    calls: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ],
) -> Any:
    """
    Call a backend function using a compatible signature.
    """

    try:
        signature = inspect.signature(
            function
        )

    except (
        TypeError,
        ValueError,
    ):
        signature = None

    binding_errors: list[str] = []

    for args, kwargs in calls:
        if signature is not None:
            try:
                signature.bind(
                    *args,
                    **kwargs,
                )

            except TypeError as exc:
                binding_errors.append(
                    str(exc)
                )

                continue

        try:
            return function(
                *args,
                **kwargs,
            )

        except Exception as exc:
            raise TagVerificationError(
                "KMAC backend tag generation failed.",
                reason=TAG_REASON_GENERATION_FAILED,
                details={
                    "function": getattr(
                        function,
                        "__name__",
                        repr(function),
                    ),
                    "reason": str(exc),
                },
            ) from exc

    raise TagVerificationError(
        (
            "Unable to call the KMAC backend using "
            "a compatible signature."
        ),
        reason=TAG_REASON_GENERATION_FAILED,
        details={
            "function": getattr(
                function,
                "__name__",
                repr(function),
            ),
            "binding_errors": (
                binding_errors
            ),
        },
    )


def _extract_generated_tag(
    backend_result: Any,
) -> bytes:
    """
    Normalize common KMAC backend return formats.

    Supported values:

    - Raw bytes
    - Dictionary containing tag or mac
    - Object containing tag or mac
    """

    generated_tag: Any = None

    if isinstance(
        backend_result,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        generated_tag = backend_result

    elif isinstance(
        backend_result,
        Mapping,
    ):
        for field_name in (
            "tag",
            "kmac_tag",
            "mac",
            "digest",
        ):
            if field_name in backend_result:
                generated_tag = (
                    backend_result[
                        field_name
                    ]
                )

                break

    else:
        for field_name in (
            "tag",
            "kmac_tag",
            "mac",
            "digest",
        ):
            if hasattr(
                backend_result,
                field_name,
            ):
                generated_tag = getattr(
                    backend_result,
                    field_name,
                )

                break

    if generated_tag is None:
        raise TagVerificationError(
            (
                "The KMAC backend did not return "
                "an authentication tag."
            ),
            reason=TAG_REASON_GENERATION_FAILED,
            details={
                "received_type": type(
                    backend_result
                ).__name__,
            },
        )

    try:
        normalized_tag = bytes(
            generated_tag
        )

    except Exception as exc:
        raise TagVerificationError(
            (
                "Unable to convert the generated "
                "KMAC tag to bytes."
            ),
            reason=TAG_REASON_GENERATION_FAILED,
            details={
                "received_type": type(
                    generated_tag
                ).__name__,
                "reason": str(exc),
            },
        ) from exc

    return normalize_kmac_tag(
        normalized_tag,
        field_name="generated_kmac_tag",
    )


def generate_expected_kmac_tag(
    *,
    authentication_key: bytes,
    verification_message: bytes,
    customization: bytes = (
        KMAC_VERIFICATION_CUSTOMIZATION
    ),
) -> bytes:
    """
    Generate the expected 128-bit FT-QuPAP KMAC tag.
    """

    validated_key = validate_bytes(
        authentication_key,
        field_name="kmac_authentication_key",
        exact_length=(
            KMAC_AUTHENTICATION_KEY_BYTES
        ),
    )

    validated_message = validate_bytes(
        verification_message,
        field_name="verification_message",
        minimum_length=1,
        maximum_length=10_000_000,
    )

    validated_customization = (
        validate_bytes(
            customization,
            field_name="kmac_customization",
            minimum_length=1,
            maximum_length=256,
        )
    )

    function = (
        _find_kmac_generation_function()
    )

    backend_result = (
        _invoke_compatible_call(
            function,
            [
                (
                    (),
                    {
                        "key": validated_key,
                        "message": (
                            validated_message
                        ),
                        "customization": (
                            validated_customization
                        ),
                        "tag_length": (
                            KMAC_TAG_BYTES
                        ),
                    },
                ),
                (
                    (),
                    {
                        "key": validated_key,
                        "data": (
                            validated_message
                        ),
                        "custom_string": (
                            validated_customization
                        ),
                        "output_length": (
                            KMAC_TAG_BYTES
                        ),
                    },
                ),
                (
                    (),
                    {
                        "key": validated_key,
                        "message": (
                            validated_message
                        ),
                        "customization_string": (
                            validated_customization
                        ),
                        "tag_bits": (
                            KMAC_TAG_BITS
                        ),
                    },
                ),
                (
                    (
                        validated_key,
                        validated_message,
                    ),
                    {
                        "customization": (
                            validated_customization
                        ),
                        "tag_length": (
                            KMAC_TAG_BYTES
                        ),
                    },
                ),
                (
                    (
                        validated_key,
                        validated_message,
                        validated_customization,
                        KMAC_TAG_BYTES,
                    ),
                    {},
                ),
                (
                    (
                        validated_key,
                        validated_message,
                    ),
                    {},
                ),
            ],
        )
    )

    return _extract_generated_tag(
        backend_result
    )


# ---------------------------------------------------------------------
# Constant-time comparison
# ---------------------------------------------------------------------

def compare_kmac_tags(
    received_tag: bytes,
    expected_tag: bytes,
) -> bool:
    """
    Compare two fixed-length KMAC tags in constant time.
    """

    normalized_received = (
        normalize_kmac_tag(
            received_tag,
            field_name="received_kmac_tag",
        )
    )

    normalized_expected = (
        normalize_kmac_tag(
            expected_tag,
            field_name="expected_kmac_tag",
        )
    )

    return hmac.compare_digest(
        normalized_received,
        normalized_expected,
    )


# ---------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------

def verify_recovered_kmac_tag(
    *,
    recovered_payload: (
        bytes
        | bytearray
        | memoryview
        | PayloadDecodingResult
    ),
    authentication_key: bytes,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
    verified_at: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    customization: bytes = (
        KMAC_VERIFICATION_CUSTOMIZATION
    ),
) -> TagVerificationResult:
    """
    Verify the KMAC tag recovered from the quantum payload.

    Authentication succeeds at this stage only when the recovered tag
    exactly matches the transcript-bound expected tag.
    """

    received_tag = extract_recovered_tag(
        recovered_payload
    )

    validated_key = validate_bytes(
        authentication_key,
        field_name="kmac_authentication_key",
        exact_length=(
            KMAC_AUTHENTICATION_KEY_BYTES
        ),
    )

    normalized_transcript = (
        normalize_transcript_digest(
            transcript_digest
        )
    )

    verification_message = (
        build_kmac_verification_message(
            session_id=session_id,
            attempt_number=attempt_number,
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                normalized_transcript
            ),
            metadata=metadata,
        )
    )

    expected_tag = (
        generate_expected_kmac_tag(
            authentication_key=validated_key,
            verification_message=(
                verification_message
            ),
            customization=customization,
        )
    )

    valid = compare_kmac_tags(
        received_tag,
        expected_tag,
    )

    selected_timestamp = (
        current_timestamp()
        if verified_at is None
        else validate_integer(
            verified_at,
            field_name="verified_at",
            minimum=0,
        )
    )

    reason = (
        TAG_REASON_VALID
        if valid
        else TAG_REASON_MISMATCH
    )

    message = (
        (
            "The quantum authentication payload contains "
            "a valid transcript-bound KMAC tag."
        )
        if valid
        else (
            "The recovered quantum authentication tag "
            "does not match the expected KMAC tag."
        )
    )

    return TagVerificationResult(
        valid=valid,
        reason=reason,
        message=message,
        session_id=session_id,
        attempt_number=attempt_number,
        pseudonym_id=pseudonym_id,
        tag_bits=KMAC_TAG_BITS,
        tag_bytes=KMAC_TAG_BYTES,
        received_tag_fingerprint=(
            calculate_tag_fingerprint(
                received_tag
            )
        ),
        expected_tag_fingerprint=(
            calculate_tag_fingerprint(
                expected_tag
            )
        ),
        verification_message_fingerprint=(
            calculate_verification_message_fingerprint(
                verification_message
            )
        ),
        transcript_digest=(
            normalized_transcript.hex()
        ),
        comparison_algorithm=(
            KMAC_COMPARISON_ALGORITHM
        ),
        verified_at=selected_timestamp,
        details={
            "customization_fingerprint": (
                hashlib.sha3_256(
                    customization
                ).hexdigest()
            ),
        },
    )


def require_valid_kmac_tag(
    *,
    recovered_payload: (
        bytes
        | bytearray
        | memoryview
        | PayloadDecodingResult
    ),
    authentication_key: bytes,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
    verified_at: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TagVerificationResult:
    """
    Verify a recovered KMAC tag and raise when it is invalid.
    """

    result = verify_recovered_kmac_tag(
        recovered_payload=(
            recovered_payload
        ),
        authentication_key=(
            authentication_key
        ),
        session_id=session_id,
        attempt_number=attempt_number,
        pseudonym_id=pseudonym_id,
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=(
            transcript_digest
        ),
        verified_at=verified_at,
        metadata=metadata,
    )

    if not result.valid:
        raise TagVerificationError(
            result.message,
            reason=result.reason,
            details=result.to_dict(),
        )

    return result


# ---------------------------------------------------------------------
# Reusable verifier
# ---------------------------------------------------------------------

class TagVerifier:
    """
    Reusable FT-QuPAP Authentication Server tag verifier.
    """

    def __init__(
        self,
        *,
        customization: bytes = (
            KMAC_VERIFICATION_CUSTOMIZATION
        ),
    ) -> None:
        self.customization = (
            validate_bytes(
                customization,
                field_name="kmac_customization",
                minimum_length=1,
                maximum_length=256,
            )
        )

    def generate_expected_tag(
        self,
        *,
        authentication_key: bytes,
        session_id: str,
        attempt_number: int,
        pseudonym_id: str,
        mobile_nonce: bytes | str,
        server_nonce: bytes | str,
        transcript_digest: bytes | str,
        metadata: Mapping[str, Any] | None = None,
    ) -> bytes:
        """
        Generate the expected tag for testing or Mobile Station logic.
        """

        verification_message = (
            build_kmac_verification_message(
                session_id=session_id,
                attempt_number=(
                    attempt_number
                ),
                pseudonym_id=pseudonym_id,
                mobile_nonce=mobile_nonce,
                server_nonce=server_nonce,
                transcript_digest=(
                    transcript_digest
                ),
                metadata=metadata,
            )
        )

        return generate_expected_kmac_tag(
            authentication_key=(
                authentication_key
            ),
            verification_message=(
                verification_message
            ),
            customization=(
                self.customization
            ),
        )

    def verify(
        self,
        *,
        recovered_payload: (
            bytes
            | bytearray
            | memoryview
            | PayloadDecodingResult
        ),
        authentication_key: bytes,
        session_id: str,
        attempt_number: int,
        pseudonym_id: str,
        mobile_nonce: bytes | str,
        server_nonce: bytes | str,
        transcript_digest: bytes | str,
        verified_at: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TagVerificationResult:
        """
        Verify one recovered authentication tag.
        """

        return verify_recovered_kmac_tag(
            recovered_payload=(
                recovered_payload
            ),
            authentication_key=(
                authentication_key
            ),
            session_id=session_id,
            attempt_number=(
                attempt_number
            ),
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                transcript_digest
            ),
            verified_at=verified_at,
            metadata=metadata,
            customization=(
                self.customization
            ),
        )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_tag_verifier_self_test() -> dict[str, Any]:
    """
    Test valid-tag acceptance, tag tampering, retry binding,
    transcript binding, and secret-safe output.
    """

    try:
        verifier = TagVerifier()

        authentication_key = (
            b"K" * KMAC_AUTHENTICATION_KEY_BYTES
        )

        session_id = (
            "FTQ-TAG-VERIFIER-SELF-TEST"
        )

        pseudonym_id = (
            "PID-FTQ-TAG-SELF-TEST"
        )

        mobile_nonce = b"M" * 32
        server_nonce = b"S" * 32

        transcript_digest = (
            hashlib.sha3_256(
                b"FT-QuPAP self-test transcript"
            ).digest()
        )

        expected_tag = (
            verifier.generate_expected_tag(
                authentication_key=(
                    authentication_key
                ),
                session_id=session_id,
                attempt_number=1,
                pseudonym_id=(
                    pseudonym_id
                ),
                mobile_nonce=(
                    mobile_nonce
                ),
                server_nonce=(
                    server_nonce
                ),
                transcript_digest=(
                    transcript_digest
                ),
            )
        )

        valid_result = verifier.verify(
            recovered_payload=(
                expected_tag
            ),
            authentication_key=(
                authentication_key
            ),
            session_id=session_id,
            attempt_number=1,
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                transcript_digest
            ),
            verified_at=1_700_000_000,
        )

        tampered_tag = bytearray(
            expected_tag
        )

        tampered_tag[0] ^= 1

        tampered_result = verifier.verify(
            recovered_payload=(
                bytes(tampered_tag)
            ),
            authentication_key=(
                authentication_key
            ),
            session_id=session_id,
            attempt_number=1,
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                transcript_digest
            ),
            verified_at=1_700_000_001,
        )

        retry_result = verifier.verify(
            recovered_payload=(
                expected_tag
            ),
            authentication_key=(
                authentication_key
            ),
            session_id=session_id,
            attempt_number=2,
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                transcript_digest
            ),
            verified_at=1_700_000_002,
        )

        changed_transcript_result = (
            verifier.verify(
                recovered_payload=(
                    expected_tag
                ),
                authentication_key=(
                    authentication_key
                ),
                session_id=session_id,
                attempt_number=1,
                pseudonym_id=(
                    pseudonym_id
                ),
                mobile_nonce=(
                    mobile_nonce
                ),
                server_nonce=(
                    server_nonce
                ),
                transcript_digest=(
                    hashlib.sha3_256(
                        b"Changed transcript"
                    ).digest()
                ),
                verified_at=(
                    1_700_000_003
                ),
            )
        )

        public_output = (
            valid_result.to_dict()
        )

        secret_values_absent = all(
            field_name not in public_output
            for field_name in (
                "received_tag",
                "expected_tag",
                "authentication_key",
            )
        )

        success = all(
            (
                valid_result.valid,

                valid_result.reason
                == TAG_REASON_VALID,

                not tampered_result.valid,

                tampered_result.reason
                == TAG_REASON_MISMATCH,

                not retry_result.valid,

                not changed_transcript_result.valid,

                secret_values_absent,

                len(expected_tag)
                == KMAC_TAG_BYTES,
            )
        )

        return {
            "success": success,

            "tag_bits": KMAC_TAG_BITS,

            "tag_bytes": len(
                expected_tag
            ),

            "valid_tag_accepted": (
                valid_result.valid
            ),

            "tampered_tag_rejected": (
                not tampered_result.valid
            ),

            "retry_attempt_changes_tag": (
                not retry_result.valid
            ),

            "transcript_changes_tag": (
                not changed_transcript_result
                .valid
            ),

            "constant_time_comparison": (
                valid_result
                .comparison_algorithm
                == KMAC_COMPARISON_ALGORITHM
            ),

            "secret_values_absent": (
                secret_values_absent
            ),

            "received_tag_fingerprint": (
                valid_result
                .received_tag_fingerprint
            ),

            "expected_tag_fingerprint": (
                valid_result
                .expected_tag_fingerprint
            ),
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(
                exc
            ).__name__,
        }


__all__ = [
    "KMAC_TAG_BYTES",
    "KMAC_VERIFICATION_CUSTOMIZATION",
    "KMAC_TAG_FINGERPRINT_ALGORITHM",
    "KMAC_COMPARISON_ALGORITHM",
    "TAG_REASON_VALID",
    "TAG_REASON_MISMATCH",
    "TAG_REASON_INVALID_LENGTH",
    "TAG_REASON_GENERATION_FAILED",
    "TagVerificationError",
    "TagVerificationResult",
    "normalize_kmac_tag",
    "extract_recovered_tag",
    "calculate_tag_fingerprint",
    "calculate_verification_message_fingerprint",
    "build_kmac_verification_message",
    "generate_expected_kmac_tag",
    "compare_kmac_tags",
    "verify_recovered_kmac_tag",
    "require_valid_kmac_tag",
    "TagVerifier",
    "run_tag_verifier_self_test",
]