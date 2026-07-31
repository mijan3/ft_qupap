"""
Constant-time comparison utilities for FT-QuPAP v5.1.

Normal equality operations may stop comparing as soon as they find a
difference. For secret values, this behavior can create timing-related
information leakage.

This module provides comparison helpers for:

- Cryptographic byte values
- Authentication tags
- Shared secrets
- Transcript hashes
- Session keys
- Text identifiers
- Classical bit sequences

Python's `hmac.compare_digest` is used for sensitive comparisons.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from typing import Any

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.validators import (
    validate_bit_sequence,
    validate_bytes,
    validate_non_empty_string,
)

from src.cryptography.crypto_models import (
    KMACTag,
    SessionKeys,
    TranscriptDigest,
)


def secure_compare_bytes(
    first: bytes,
    second: bytes,
) -> bool:
    """
    Compare two byte values using constant-time comparison.

    Different lengths return False.
    """

    validated_first = validate_bytes(
        first,
        field_name="first_value",
        minimum_length=0,
    )

    validated_second = validate_bytes(
        second,
        field_name="second_value",
        minimum_length=0,
    )

    return hmac.compare_digest(
        validated_first,
        validated_second,
    )


def secure_compare_text(
    first: str,
    second: str,
    *,
    case_sensitive: bool = True,
    strip_whitespace: bool = False,
) -> bool:
    """
    Compare two text values using constant-time comparison.

    Parameters
    ----------
    case_sensitive:
        When False, both values are converted using `casefold()`.

    strip_whitespace:
        When True, surrounding whitespace is removed before comparison.
    """

    validated_first = validate_non_empty_string(
        first,
        field_name="first_text",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    validated_second = validate_non_empty_string(
        second,
        field_name="second_text",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    if strip_whitespace:
        validated_first = validated_first.strip()
        validated_second = validated_second.strip()

    if not case_sensitive:
        validated_first = validated_first.casefold()
        validated_second = validated_second.casefold()

    return hmac.compare_digest(
        validated_first.encode("utf-8"),
        validated_second.encode("utf-8"),
    )


def secure_compare_hex(
    first_hex: str,
    second_hex: str,
) -> bool:
    """
    Compare two hexadecimal strings as decoded byte values.

    Uppercase and lowercase hexadecimal text are treated equally.
    """

    if not isinstance(first_hex, str):
        raise ProtocolValidationError(
            "first_hex must be a string.",
            details={
                "received_type": type(first_hex).__name__,
            },
        )

    if not isinstance(second_hex, str):
        raise ProtocolValidationError(
            "second_hex must be a string.",
            details={
                "received_type": type(second_hex).__name__,
            },
        )

    first_normalized = first_hex.strip()
    second_normalized = second_hex.strip()

    try:
        first_bytes = bytes.fromhex(
            first_normalized
        )

        second_bytes = bytes.fromhex(
            second_normalized
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            "Invalid hexadecimal comparison input.",
            details={
                "first_length": len(first_normalized),
                "second_length": len(second_normalized),
            },
        ) from exc

    return secure_compare_bytes(
        first_bytes,
        second_bytes,
    )


def secure_compare_bits(
    first_bits: Sequence[int],
    second_bits: Sequence[int],
) -> bool:
    """
    Compare two classical bit sequences.

    Each bit must be either 0 or 1.
    """

    validated_first = validate_bit_sequence(
        first_bits,
        field_name="first_bits",
    )

    validated_second = validate_bit_sequence(
        second_bits,
        field_name="second_bits",
    )

    if len(validated_first) != len(
        validated_second
    ):
        return False

    first_bytes = bytes(
        validated_first
    )

    second_bytes = bytes(
        validated_second
    )

    return hmac.compare_digest(
        first_bytes,
        second_bytes,
    )


def secure_compare_kmac_tags(
    first_tag: bytes | KMACTag,
    second_tag: bytes | KMACTag,
) -> bool:
    """
    Compare two KMAC authentication tags.

    KMACTag objects and raw bytes are both accepted.
    """

    first_bytes = (
        first_tag.tag
        if isinstance(first_tag, KMACTag)
        else first_tag
    )

    second_bytes = (
        second_tag.tag
        if isinstance(second_tag, KMACTag)
        else second_tag
    )

    return secure_compare_bytes(
        first_bytes,
        second_bytes,
    )


def secure_compare_transcript_digests(
    first_digest: bytes | TranscriptDigest,
    second_digest: bytes | TranscriptDigest,
) -> bool:
    """
    Compare two transcript hashes.
    """

    first_bytes = (
        first_digest.digest
        if isinstance(
            first_digest,
            TranscriptDigest,
        )
        else first_digest
    )

    second_bytes = (
        second_digest.digest
        if isinstance(
            second_digest,
            TranscriptDigest,
        )
        else second_digest
    )

    return secure_compare_bytes(
        first_bytes,
        second_bytes,
    )


def secure_compare_session_keys(
    first_keys: SessionKeys,
    second_keys: SessionKeys,
) -> bool:
    """
    Compare two complete FT-QuPAP SessionKeys objects.

    This helper is intended for:

    - Unit tests
    - Protocol demonstrations
    - Mobile/server shared-key verification

    Session keys must never be transmitted merely for comparison in a
    real deployment.
    """

    if not isinstance(
        first_keys,
        SessionKeys,
    ):
        raise ProtocolValidationError(
            "first_keys must be a SessionKeys object.",
            details={
                "received_type": type(
                    first_keys
                ).__name__,
            },
        )

    if not isinstance(
        second_keys,
        SessionKeys,
    ):
        raise ProtocolValidationError(
            "second_keys must be a SessionKeys object.",
            details={
                "received_type": type(
                    second_keys
                ).__name__,
            },
        )

    session_id_match = secure_compare_text(
        first_keys.session_id,
        second_keys.session_id,
    )

    transcript_match = secure_compare_bytes(
        first_keys.transcript_hash,
        second_keys.transcript_hash,
    )

    master_key_match = secure_compare_bytes(
        first_keys.master_key,
        second_keys.master_key,
    )

    authentication_key_match = (
        secure_compare_bytes(
            first_keys.authentication_key,
            second_keys.authentication_key,
        )
    )

    control_key_match = secure_compare_bytes(
        first_keys.control_key,
        second_keys.control_key,
    )

    return all(
        (
            session_id_match,
            transcript_match,
            master_key_match,
            authentication_key_match,
            control_key_match,
        )
    )


def require_equal_bytes(
    first: bytes,
    second: bytes,
    *,
    error_message: str = (
        "Sensitive byte values do not match."
    ),
    error_code: str = "SECURE_COMPARISON_FAILED",
    details: dict[str, Any] | None = None,
) -> None:
    """
    Require two sensitive byte values to match.

    Raises ProtocolValidationError on mismatch.
    """

    if not secure_compare_bytes(
        first,
        second,
    ):
        raise ProtocolValidationError(
            error_message,
            code=error_code,
            details=details or {},
        )


def require_equal_kmac_tags(
    first_tag: bytes | KMACTag,
    second_tag: bytes | KMACTag,
) -> None:
    """
    Require two KMAC authentication tags to match.
    """

    if not secure_compare_kmac_tags(
        first_tag,
        second_tag,
    ):
        raise ProtocolValidationError(
            "KMAC authentication tags do not match.",
            code="KMAC_TAG_MISMATCH",
        )


def require_equal_transcripts(
    first_digest: bytes | TranscriptDigest,
    second_digest: bytes | TranscriptDigest,
) -> None:
    """
    Require two transcript hashes to match.
    """

    if not secure_compare_transcript_digests(
        first_digest,
        second_digest,
    ):
        raise ProtocolValidationError(
            "Mobile Station and server transcript hashes do not match.",
            code="TRANSCRIPT_MISMATCH",
        )


def require_equal_session_keys(
    first_keys: SessionKeys,
    second_keys: SessionKeys,
) -> None:
    """
    Require two SessionKeys objects to match.
    """

    if not secure_compare_session_keys(
        first_keys,
        second_keys,
    ):
        raise ProtocolValidationError(
            "Mobile Station and server session keys do not match.",
            code="SESSION_KEY_MISMATCH",
            details={
                "first_session_id": (
                    first_keys.session_id
                ),
                "second_session_id": (
                    second_keys.session_id
                ),
            },
        )


def create_secure_fingerprint(
    value: bytes,
    *,
    output_bytes: int = 8,
) -> str:
    """
    Create a short SHA3-256 diagnostic fingerprint.

    This is suitable for logs and dashboard display. It does not expose
    the original sensitive value.
    """

    validated_value = validate_bytes(
        value,
        field_name="fingerprint_input",
        minimum_length=1,
    )

    if (
        not isinstance(output_bytes, int)
        or isinstance(output_bytes, bool)
    ):
        raise ProtocolValidationError(
            "output_bytes must be an integer."
        )

    if not 4 <= output_bytes <= 32:
        raise ProtocolValidationError(
            "output_bytes must be between 4 and 32.",
            details={
                "output_bytes": output_bytes,
            },
        )

    digest = hashlib.sha3_256(
        validated_value
    ).digest()

    return digest[
        :output_bytes
    ].hex()


def run_secure_compare_self_test() -> dict[str, Any]:
    """
    Run deterministic secure-comparison tests.
    """

    original_bytes = bytes(
        range(32)
    )

    identical_bytes = bytes(
        range(32)
    )

    changed_bytes = bytearray(
        original_bytes
    )

    changed_bytes[-1] ^= 0x01

    equal_bytes_pass = secure_compare_bytes(
        original_bytes,
        identical_bytes,
    )

    changed_bytes_rejected = (
        not secure_compare_bytes(
            original_bytes,
            bytes(changed_bytes),
        )
    )

    equal_text_pass = secure_compare_text(
        "FT-QuPAP",
        "FT-QuPAP",
    )

    case_insensitive_pass = (
        secure_compare_text(
            "URBAN",
            "urban",
            case_sensitive=False,
        )
    )

    equal_bits_pass = secure_compare_bits(
        [0, 1, 1, 0],
        [0, 1, 1, 0],
    )

    changed_bits_rejected = (
        not secure_compare_bits(
            [0, 1, 1, 0],
            [0, 1, 0, 0],
        )
    )

    tag_one = KMACTag(
        tag=b"\x01" * 16
    )

    tag_two = KMACTag(
        tag=b"\x01" * 16
    )

    tag_three = KMACTag(
        tag=b"\x02" * 16
    )

    equal_tag_pass = secure_compare_kmac_tags(
        tag_one,
        tag_two,
    )

    changed_tag_rejected = (
        not secure_compare_kmac_tags(
            tag_one,
            tag_three,
        )
    )

    fingerprint = create_secure_fingerprint(
        original_bytes
    )

    success = all(
        (
            equal_bytes_pass,
            changed_bytes_rejected,
            equal_text_pass,
            case_insensitive_pass,
            equal_bits_pass,
            changed_bits_rejected,
            equal_tag_pass,
            changed_tag_rejected,
            len(fingerprint) == 16,
        )
    )

    return {
        "success": success,
        "equal_bytes_pass": equal_bytes_pass,
        "changed_bytes_rejected": (
            changed_bytes_rejected
        ),
        "equal_text_pass": equal_text_pass,
        "case_insensitive_pass": (
            case_insensitive_pass
        ),
        "equal_bits_pass": equal_bits_pass,
        "changed_bits_rejected": (
            changed_bits_rejected
        ),
        "equal_tag_pass": equal_tag_pass,
        "changed_tag_rejected": (
            changed_tag_rejected
        ),
        "fingerprint": fingerprint,
    }


__all__ = [
    "secure_compare_bytes",
    "secure_compare_text",
    "secure_compare_hex",
    "secure_compare_bits",
    "secure_compare_kmac_tags",
    "secure_compare_transcript_digests",
    "secure_compare_session_keys",
    "require_equal_bytes",
    "require_equal_kmac_tags",
    "require_equal_transcripts",
    "require_equal_session_keys",
    "create_secure_fingerprint",
    "run_secure_compare_self_test",
]