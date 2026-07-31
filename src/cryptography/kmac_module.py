"""
KMAC authentication-tag generation for FT-QuPAP v5.1.

After ML-KEM establishes a shared secret and the KDF derives the
authentication key, this module generates a 128-bit KMAC tag.

The tag is bound to:

- Protocol version
- Session identifier
- Transcript hash
- Pseudonymous subscriber identity
- Mobile-network context
- Authentication-attempt number
- Optional protocol metadata

The resulting 16-byte tag is converted into 128 classical bits. These
bits later become the logical quantum payload before Steane encoding.
"""

from __future__ import annotations

from hmac import compare_digest
from typing import Any, Mapping

from Crypto.Hash import KMAC128, KMAC256

from src.common.constants import (
    KMAC_ALGORITHM,
    KMAC_TAG_BYTES,
    KMAC_TAG_BITS,
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    KMACError,
    KMACTagMismatchError,
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)

from src.cryptography.crypto_models import (
    KMACTag,
    SessionKeys,
    TranscriptDigest,
)


DEFAULT_KMAC_CUSTOMIZATION = b"FT-QuPAP-v5.1-Authentication"

MINIMUM_KMAC128_KEY_BYTES = 16

MINIMUM_KMAC256_KEY_BYTES = 32


def normalize_kmac_algorithm(
    algorithm: str = KMAC_ALGORITHM,
) -> str:
    """
    Normalize a KMAC algorithm name.

    Accepted examples:

        KMAC128
        KMAC-128
        kmac_128
        KMAC256
        KMAC-256
    """

    if not isinstance(algorithm, str):
        raise KMACError(
            "KMAC algorithm name must be a string.",
            details={
                "received_type": type(algorithm).__name__,
            },
        )

    normalized = (
        algorithm
        .strip()
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    aliases = {
        "KMAC128": "KMAC128",
        "KMAC256": "KMAC256",
    }

    selected = aliases.get(normalized)

    if selected is None:
        raise KMACError(
            f"Unsupported KMAC algorithm: {algorithm}",
            details={
                "algorithm": algorithm,
                "supported_algorithms": [
                    "KMAC128",
                    "KMAC256",
                ],
            },
        )

    return selected


def normalize_customization(
    customization: bytes | str,
) -> bytes:
    """
    Convert a KMAC customization value into bytes.
    """

    if isinstance(customization, bytes):
        return validate_bytes(
            customization,
            field_name="kmac_customization",
            minimum_length=0,
            maximum_length=256,
        )

    if isinstance(customization, str):
        normalized = customization.strip()

        if len(normalized) > 256:
            raise ProtocolValidationError(
                "KMAC customization text is too long.",
                details={
                    "maximum_length": 256,
                    "actual_length": len(normalized),
                },
            )

        return normalized.encode("utf-8")

    raise ProtocolValidationError(
        "KMAC customization must be bytes or a string.",
        details={
            "received_type": type(customization).__name__,
        },
    )


def normalize_transcript_hash(
    transcript_hash: bytes | TranscriptDigest,
) -> bytes:
    """
    Normalize a transcript digest into raw bytes.
    """

    if isinstance(
        transcript_hash,
        TranscriptDigest,
    ):
        digest = transcript_hash.digest
    else:
        digest = transcript_hash

    return validate_bytes(
        digest,
        field_name="transcript_hash",
        exact_length=32,
    )


def validate_kmac_key(
    key: bytes,
    *,
    algorithm: str = KMAC_ALGORITHM,
) -> bytes:
    """
    Validate the authentication key according to the KMAC variant.
    """

    selected_algorithm = normalize_kmac_algorithm(
        algorithm
    )

    minimum_length = (
        MINIMUM_KMAC128_KEY_BYTES
        if selected_algorithm == "KMAC128"
        else MINIMUM_KMAC256_KEY_BYTES
    )

    return validate_bytes(
        key,
        field_name="kmac_key",
        minimum_length=minimum_length,
        maximum_length=4096,
    )


def build_kmac_payload(
    *,
    session_id: str,
    transcript_hash: bytes | TranscriptDigest,
    pseudonym_id: str,
    context: str,
    attempt_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> bytes:
    """
    Construct the canonical message authenticated by KMAC.

    Both the Mobile Station and Authentication Server must construct
    exactly the same payload.
    """

    validated_session_id = validate_non_empty_string(
        session_id,
        field_name="session_id",
        minimum_length=3,
        maximum_length=128,
    )

    validated_transcript_hash = normalize_transcript_hash(
        transcript_hash
    )

    validated_pseudonym = validate_non_empty_string(
        pseudonym_id,
        field_name="pseudonym_id",
        minimum_length=3,
        maximum_length=128,
    )

    validated_context = validate_non_empty_string(
        context,
        field_name="context",
        minimum_length=1,
        maximum_length=32,
    ).lower()

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    if metadata is None:
        normalized_metadata: dict[str, Any] = {}
    elif isinstance(metadata, Mapping):
        normalized_metadata = dict(metadata)
    else:
        raise ProtocolValidationError(
            "KMAC metadata must be a mapping.",
            details={
                "received_type": type(metadata).__name__,
            },
        )

    payload = {
        "domain": PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        ),
        "purpose": "quantum-authentication-tag",
        "session_id": validated_session_id,
        "transcript_hash": validated_transcript_hash.hex(),
        "pseudonym_id": validated_pseudonym,
        "context": validated_context,
        "attempt_number": validated_attempt,
        "tag_length_bits": KMAC_TAG_BITS,
        "metadata": normalized_metadata,
    }

    return canonical_json_bytes(payload)


def compute_kmac(
    *,
    key: bytes,
    message: bytes,
    algorithm: str = KMAC_ALGORITHM,
    customization: bytes | str = DEFAULT_KMAC_CUSTOMIZATION,
    tag_length: int = KMAC_TAG_BYTES,
) -> bytes:
    """
    Compute a KMAC tag from raw key and message bytes.

    FT-QuPAP normally requests 16 output bytes, producing a 128-bit tag.
    """

    selected_algorithm = normalize_kmac_algorithm(
        algorithm
    )

    validated_key = validate_kmac_key(
        key,
        algorithm=selected_algorithm,
    )

    validated_message = validate_bytes(
        message,
        field_name="kmac_message",
        minimum_length=1,
        maximum_length=10_000_000,
    )

    validated_customization = normalize_customization(
        customization
    )

    validated_tag_length = validate_integer(
        tag_length,
        field_name="tag_length",
        minimum=8,
        maximum=1024,
    )

    try:
        if selected_algorithm == "KMAC128":
            kmac = KMAC128.new(
                key=validated_key,
                data=validated_message,
                mac_len=validated_tag_length,
                custom=validated_customization,
            )
        else:
            kmac = KMAC256.new(
                key=validated_key,
                data=validated_message,
                mac_len=validated_tag_length,
                custom=validated_customization,
            )

        tag = kmac.digest()

    except Exception as exc:
        raise KMACError(
            "Unable to generate the KMAC authentication tag.",
            details={
                "algorithm": selected_algorithm,
                "message_bytes": len(validated_message),
                "tag_length": validated_tag_length,
                "reason": str(exc),
            },
        ) from exc

    return validate_bytes(
        tag,
        field_name="generated_kmac_tag",
        exact_length=validated_tag_length,
    )


def generate_authentication_tag(
    *,
    authentication_key: bytes,
    session_id: str,
    transcript_hash: bytes | TranscriptDigest,
    pseudonym_id: str,
    context: str,
    attempt_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
    algorithm: str = KMAC_ALGORITHM,
    customization: bytes | str = DEFAULT_KMAC_CUSTOMIZATION,
) -> KMACTag:
    """
    Generate the complete FT-QuPAP 128-bit authentication tag.

    This function is normally executed by the Mobile Station.
    """

    payload = build_kmac_payload(
        session_id=session_id,
        transcript_hash=transcript_hash,
        pseudonym_id=pseudonym_id,
        context=context,
        attempt_number=attempt_number,
        metadata=metadata,
    )

    tag = compute_kmac(
        key=authentication_key,
        message=payload,
        algorithm=algorithm,
        customization=customization,
        tag_length=KMAC_TAG_BYTES,
    )

    if isinstance(customization, bytes):
        customization_text = customization.decode(
            "utf-8",
            errors="replace",
        )
    else:
        customization_text = customization

    return KMACTag(
        tag=tag,
        customization=customization_text,
    )


def generate_tag_from_session_keys(
    *,
    session_keys: SessionKeys,
    pseudonym_id: str,
    context: str,
    attempt_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
    algorithm: str = KMAC_ALGORITHM,
) -> KMACTag:
    """
    Generate a KMAC tag directly from a SessionKeys object.
    """

    if not isinstance(
        session_keys,
        SessionKeys,
    ):
        raise ProtocolValidationError(
            "session_keys must be a SessionKeys object.",
            details={
                "received_type": type(
                    session_keys
                ).__name__,
            },
        )

    return generate_authentication_tag(
        authentication_key=(
            session_keys.authentication_key
        ),
        session_id=session_keys.session_id,
        transcript_hash=(
            session_keys.transcript_hash
        ),
        pseudonym_id=pseudonym_id,
        context=context,
        attempt_number=attempt_number,
        metadata=metadata,
        algorithm=algorithm,
    )


def verify_authentication_tag(
    *,
    received_tag: bytes | KMACTag,
    authentication_key: bytes,
    session_id: str,
    transcript_hash: bytes | TranscriptDigest,
    pseudonym_id: str,
    context: str,
    attempt_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
    algorithm: str = KMAC_ALGORITHM,
    customization: bytes | str = DEFAULT_KMAC_CUSTOMIZATION,
) -> bool:
    """
    Recompute and verify a received FT-QuPAP KMAC tag.

    Comparison uses `compare_digest` to avoid normal early-exit byte
    comparison.
    """

    if isinstance(
        received_tag,
        KMACTag,
    ):
        received_tag_bytes = received_tag.tag
    else:
        received_tag_bytes = received_tag

    validated_received_tag = validate_bytes(
        received_tag_bytes,
        field_name="received_kmac_tag",
        exact_length=KMAC_TAG_BYTES,
    )

    expected_tag = generate_authentication_tag(
        authentication_key=authentication_key,
        session_id=session_id,
        transcript_hash=transcript_hash,
        pseudonym_id=pseudonym_id,
        context=context,
        attempt_number=attempt_number,
        metadata=metadata,
        algorithm=algorithm,
        customization=customization,
    )

    return compare_digest(
        validated_received_tag,
        expected_tag.tag,
    )


def require_valid_authentication_tag(
    *,
    received_tag: bytes | KMACTag,
    authentication_key: bytes,
    session_id: str,
    transcript_hash: bytes | TranscriptDigest,
    pseudonym_id: str,
    context: str,
    attempt_number: int = 1,
    metadata: Mapping[str, Any] | None = None,
    algorithm: str = KMAC_ALGORITHM,
    customization: bytes | str = DEFAULT_KMAC_CUSTOMIZATION,
) -> None:
    """
    Verify a KMAC tag and raise KMACTagMismatchError when it is invalid.
    """

    is_valid = verify_authentication_tag(
        received_tag=received_tag,
        authentication_key=authentication_key,
        session_id=session_id,
        transcript_hash=transcript_hash,
        pseudonym_id=pseudonym_id,
        context=context,
        attempt_number=attempt_number,
        metadata=metadata,
        algorithm=algorithm,
        customization=customization,
    )

    if not is_valid:
        raise KMACTagMismatchError(
            details={
                "session_id": session_id,
                "pseudonym_id": pseudonym_id,
                "context": context,
                "attempt_number": attempt_number,
            },
        )


def kmac_tag_to_bits(
    tag: bytes | KMACTag,
) -> list[int]:
    """
    Convert the 16-byte KMAC tag into 128 classical bits.

    Bit order is big-endian inside every byte.

    Example:

        0x05 → 00000101
    """

    if isinstance(tag, KMACTag):
        tag_bytes = tag.tag
    else:
        tag_bytes = tag

    validated_tag = validate_bytes(
        tag_bytes,
        field_name="kmac_tag",
        exact_length=KMAC_TAG_BYTES,
    )

    bits: list[int] = []

    for byte_value in validated_tag:
        for shift in range(7, -1, -1):
            bits.append(
                (byte_value >> shift) & 1
            )

    if len(bits) != KMAC_TAG_BITS:
        raise KMACError(
            "KMAC tag conversion produced an invalid number of bits.",
            details={
                "expected_bits": KMAC_TAG_BITS,
                "actual_bits": len(bits),
            },
        )

    return bits


def bits_to_kmac_tag(
    bits: list[int],
    *,
    customization: str = "FT-QuPAP-v5.1",
) -> KMACTag:
    """
    Convert 128 recovered classical bits back into a KMAC tag.
    """

    if not isinstance(bits, list):
        raise ProtocolValidationError(
            "KMAC bits must be provided as a list.",
            details={
                "received_type": type(bits).__name__,
            },
        )

    if len(bits) != KMAC_TAG_BITS:
        raise ProtocolValidationError(
            (
                f"KMAC payload must contain exactly "
                f"{KMAC_TAG_BITS} bits."
            ),
            details={
                "expected_bits": KMAC_TAG_BITS,
                "actual_bits": len(bits),
            },
        )

    output = bytearray()

    for start_index in range(
        0,
        KMAC_TAG_BITS,
        8,
    ):
        current_byte = 0

        for bit in bits[
            start_index:start_index + 8
        ]:
            if bit not in (0, 1):
                raise ProtocolValidationError(
                    "KMAC bit list contains a value other than 0 or 1.",
                    details={
                        "invalid_bit": bit,
                        "bit_index": (
                            start_index
                            + len(output)
                        ),
                    },
                )

            current_byte = (
                current_byte << 1
            ) | int(bit)

        output.append(current_byte)

    return KMACTag(
        tag=bytes(output),
        customization=customization,
    )


def run_kmac_self_test() -> dict[str, Any]:
    """
    Run a deterministic KMAC module self-test.

    The test confirms:

    - KMAC produces a 16-byte tag
    - The tag contains 128 bits
    - Same input verifies successfully
    - Changed context fails verification
    - Bit conversion is reversible
    """

    authentication_key = bytes(
        range(32)
    )

    transcript_hash = bytes(
        reversed(range(32))
    )

    tag = generate_authentication_tag(
        authentication_key=authentication_key,
        session_id="FTQ-KMAC-SELF-TEST",
        transcript_hash=transcript_hash,
        pseudonym_id="PID-SELF-TEST-001",
        context="urban",
        attempt_number=1,
        metadata={
            "test": True,
        },
    )

    valid_verification = verify_authentication_tag(
        received_tag=tag,
        authentication_key=authentication_key,
        session_id="FTQ-KMAC-SELF-TEST",
        transcript_hash=transcript_hash,
        pseudonym_id="PID-SELF-TEST-001",
        context="urban",
        attempt_number=1,
        metadata={
            "test": True,
        },
    )

    changed_context_rejected = not verify_authentication_tag(
        received_tag=tag,
        authentication_key=authentication_key,
        session_id="FTQ-KMAC-SELF-TEST",
        transcript_hash=transcript_hash,
        pseudonym_id="PID-SELF-TEST-001",
        context="rural",
        attempt_number=1,
        metadata={
            "test": True,
        },
    )

    bits = kmac_tag_to_bits(tag)

    reconstructed_tag = bits_to_kmac_tag(
        bits,
        customization=tag.customization,
    )

    conversion_reversible = compare_digest(
        tag.tag,
        reconstructed_tag.tag,
    )

    success = all(
        (
            len(tag.tag) == KMAC_TAG_BYTES,
            len(bits) == KMAC_TAG_BITS,
            valid_verification,
            changed_context_rejected,
            conversion_reversible,
        )
    )

    return {
        "success": success,
        "algorithm": normalize_kmac_algorithm(),
        "tag_bytes": len(tag.tag),
        "tag_bits": len(bits),
        "valid_verification": valid_verification,
        "changed_context_rejected": (
            changed_context_rejected
        ),
        "conversion_reversible": (
            conversion_reversible
        ),
        "tag_hex": tag.tag.hex(),
    }


__all__ = [
    "DEFAULT_KMAC_CUSTOMIZATION",
    "MINIMUM_KMAC128_KEY_BYTES",
    "MINIMUM_KMAC256_KEY_BYTES",
    "normalize_kmac_algorithm",
    "normalize_customization",
    "normalize_transcript_hash",
    "validate_kmac_key",
    "build_kmac_payload",
    "compute_kmac",
    "generate_authentication_tag",
    "generate_tag_from_session_keys",
    "verify_authentication_tag",
    "require_valid_authentication_tag",
    "kmac_tag_to_bits",
    "bits_to_kmac_tag",
    "run_kmac_self_test",
]