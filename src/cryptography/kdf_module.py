"""
Transcript-bound key derivation for FT-QuPAP v5.1.

After ML-KEM encapsulation and decapsulation, both the Mobile Station
and Authentication Server possess the same shared secret.

This module derives three independent session keys:

1. Master key
2. KMAC authentication key
3. Quantum control-schedule encryption key

The derived keys are bound to:

- ML-KEM shared secret
- Complete protocol transcript hash
- Session identifier
- Protocol domain label
- Authentication-attempt number

Changing any bound value produces completely different session keys.
"""

from __future__ import annotations

from hmac import compare_digest
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import (
    HKDF,
    HKDFExpand,
)

from src.common.constants import (
    AUTHENTICATION_KEY_LABEL,
    CONTROL_KEY_LABEL,
    PROTOCOL_DOMAIN_LABEL,
    SESSION_KEY_DERIVATION_LABEL,
)

from src.common.exceptions import (
    KeyDerivationError,
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
    SessionKeys,
    TranscriptDigest,
)


# ---------------------------------------------------------------------
# Key lengths
# ---------------------------------------------------------------------

MASTER_KEY_BYTES = 32

AUTHENTICATION_KEY_BYTES = 32

CONTROL_KEY_BYTES = 32

TRANSCRIPT_HASH_BYTES = 32


# ---------------------------------------------------------------------
# Hash configuration
# ---------------------------------------------------------------------

KDF_HASH_NAME = "SHA3-256"


def create_kdf_hash_algorithm() -> hashes.HashAlgorithm:
    """
    Return the hash algorithm used by HKDF.

    FT-QuPAP uses SHA3-256 for transcript-bound key derivation.
    """

    return hashes.SHA3_256()


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _normalize_label(
    label: bytes | str,
    *,
    field_name: str,
) -> bytes:
    """
    Convert a KDF label into bytes.
    """

    if isinstance(label, bytes):
        return validate_bytes(
            label,
            field_name=field_name,
            minimum_length=1,
            maximum_length=256,
        )

    if isinstance(label, str):
        normalized = validate_non_empty_string(
            label,
            field_name=field_name,
            minimum_length=1,
            maximum_length=256,
        )

        return normalized.encode("utf-8")

    raise ProtocolValidationError(
        f"{field_name} must be bytes or a string.",
        details={
            "field_name": field_name,
            "received_type": type(label).__name__,
        },
    )


def _normalize_transcript_hash(
    transcript_hash: bytes | TranscriptDigest,
) -> bytes:
    """
    Convert a TranscriptDigest object or raw bytes into transcript bytes.
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
        exact_length=TRANSCRIPT_HASH_BYTES,
    )


def _build_master_key_information(
    *,
    session_id: str,
    attempt_number: int,
) -> bytes:
    """
    Create deterministic HKDF information for the master key.
    """

    payload = {
        "domain": (
            PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8",
                errors="strict",
            )
        ),
        "purpose": (
            _normalize_label(
                SESSION_KEY_DERIVATION_LABEL,
                field_name=(
                    "session_key_derivation_label"
                ),
            ).decode(
                "utf-8",
                errors="strict",
            )
        ),
        "session_id": session_id,
        "attempt_number": attempt_number,
        "hash_algorithm": KDF_HASH_NAME,
        "output_length": MASTER_KEY_BYTES,
    }

    return canonical_json_bytes(payload)


def _build_subkey_information(
    *,
    session_id: str,
    transcript_hash: bytes,
    attempt_number: int,
    key_label: bytes,
    output_length: int,
) -> bytes:
    """
    Create deterministic HKDF information for a session subkey.
    """

    payload = {
        "domain": (
            PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8",
                errors="strict",
            )
        ),
        "session_id": session_id,
        "attempt_number": attempt_number,
        "transcript_hash": (
            transcript_hash.hex()
        ),
        "key_label": key_label.decode(
            "utf-8",
            errors="strict",
        ),
        "hash_algorithm": KDF_HASH_NAME,
        "output_length": output_length,
    }

    return canonical_json_bytes(payload)


# ---------------------------------------------------------------------
# HKDF extraction and expansion
# ---------------------------------------------------------------------

def derive_master_key(
    shared_secret: bytes,
    transcript_hash: bytes | TranscriptDigest,
    session_id: str,
    *,
    attempt_number: int = 1,
    output_length: int = MASTER_KEY_BYTES,
) -> bytes:
    """
    Derive the transcript-bound master key.

    HKDF inputs:

        Input key material:
            ML-KEM shared secret

        Salt:
            SHA3-256 transcript hash

        Information:
            Protocol domain, session ID, attempt number, and purpose

    Both protocol participants must supply identical values.
    """

    validated_shared_secret = validate_bytes(
        shared_secret,
        field_name="mlkem_shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    validated_transcript_hash = (
        _normalize_transcript_hash(
            transcript_hash
        )
    )

    validated_session_id = (
        validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )
    )

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    validated_output_length = validate_integer(
        output_length,
        field_name="output_length",
        minimum=16,
        maximum=1024,
    )

    information = (
        _build_master_key_information(
            session_id=validated_session_id,
            attempt_number=validated_attempt,
        )
    )

    try:
        hkdf = HKDF(
            algorithm=create_kdf_hash_algorithm(),
            length=validated_output_length,
            salt=validated_transcript_hash,
            info=information,
        )

        master_key = hkdf.derive(
            validated_shared_secret
        )

    except Exception as exc:
        raise KeyDerivationError(
            "Unable to derive the FT-QuPAP master key.",
            details={
                "session_id": validated_session_id,
                "attempt_number": validated_attempt,
                "output_length": (
                    validated_output_length
                ),
                "reason": str(exc),
            },
        ) from exc

    return validate_bytes(
        master_key,
        field_name="master_key",
        exact_length=validated_output_length,
    )


def expand_session_subkey(
    master_key: bytes,
    *,
    key_label: bytes | str,
    transcript_hash: bytes | TranscriptDigest,
    session_id: str,
    attempt_number: int = 1,
    output_length: int = 32,
) -> bytes:
    """
    Expand one independent subkey from the master key.

    Different labels produce cryptographically separated keys.
    """

    validated_master_key = validate_bytes(
        master_key,
        field_name="master_key",
        minimum_length=16,
        maximum_length=1024,
    )

    validated_key_label = _normalize_label(
        key_label,
        field_name="key_label",
    )

    validated_transcript_hash = (
        _normalize_transcript_hash(
            transcript_hash
        )
    )

    validated_session_id = (
        validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )
    )

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    validated_output_length = validate_integer(
        output_length,
        field_name="output_length",
        minimum=16,
        maximum=1024,
    )

    information = _build_subkey_information(
        session_id=validated_session_id,
        transcript_hash=validated_transcript_hash,
        attempt_number=validated_attempt,
        key_label=validated_key_label,
        output_length=validated_output_length,
    )

    try:
        hkdf_expand = HKDFExpand(
            algorithm=create_kdf_hash_algorithm(),
            length=validated_output_length,
            info=information,
        )

        subkey = hkdf_expand.derive(
            validated_master_key
        )

    except Exception as exc:
        raise KeyDerivationError(
            "Unable to expand an FT-QuPAP session subkey.",
            details={
                "session_id": validated_session_id,
                "attempt_number": validated_attempt,
                "key_label": (
                    validated_key_label.decode(
                        "utf-8",
                        errors="replace",
                    )
                ),
                "output_length": (
                    validated_output_length
                ),
                "reason": str(exc),
            },
        ) from exc

    return validate_bytes(
        subkey,
        field_name="session_subkey",
        exact_length=validated_output_length,
    )


# ---------------------------------------------------------------------
# Full session-key derivation
# ---------------------------------------------------------------------

def derive_session_keys(
    shared_secret: bytes,
    transcript_hash: bytes | TranscriptDigest,
    session_id: str,
    *,
    attempt_number: int = 1,
) -> SessionKeys:
    """
    Derive all FT-QuPAP session keys.

    Returns:

        SessionKeys(
            master_key,
            authentication_key,
            control_key,
            session_id,
            transcript_hash,
        )

    The authentication key is used for KMAC.

    The control key is used to protect the secret quantum control
    schedule.
    """

    validated_transcript_hash = (
        _normalize_transcript_hash(
            transcript_hash
        )
    )

    validated_session_id = (
        validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )
    )

    validated_attempt = validate_integer(
        attempt_number,
        field_name="attempt_number",
        minimum=1,
        maximum=100,
    )

    master_key = derive_master_key(
        shared_secret=shared_secret,
        transcript_hash=validated_transcript_hash,
        session_id=validated_session_id,
        attempt_number=validated_attempt,
        output_length=MASTER_KEY_BYTES,
    )

    authentication_key = expand_session_subkey(
        master_key=master_key,
        key_label=AUTHENTICATION_KEY_LABEL,
        transcript_hash=validated_transcript_hash,
        session_id=validated_session_id,
        attempt_number=validated_attempt,
        output_length=AUTHENTICATION_KEY_BYTES,
    )

    control_key = expand_session_subkey(
        master_key=master_key,
        key_label=CONTROL_KEY_LABEL,
        transcript_hash=validated_transcript_hash,
        session_id=validated_session_id,
        attempt_number=validated_attempt,
        output_length=CONTROL_KEY_BYTES,
    )

    if compare_digest(
        authentication_key,
        control_key,
    ):
        raise KeyDerivationError(
            (
                "Authentication and control keys must be "
                "cryptographically separated."
            ),
            details={
                "session_id": validated_session_id,
                "attempt_number": validated_attempt,
            },
        )

    return SessionKeys(
        master_key=master_key,
        authentication_key=authentication_key,
        control_key=control_key,
        session_id=validated_session_id,
        transcript_hash=validated_transcript_hash,
    )


# ---------------------------------------------------------------------
# Verification and diagnostics
# ---------------------------------------------------------------------

def compare_session_keys(
    first: SessionKeys,
    second: SessionKeys,
) -> bool:
    """
    Compare two SessionKeys objects using constant-time byte comparison.

    This is primarily intended for protocol tests and demonstrations.
    Real protocol participants do not transmit session keys.
    """

    if not isinstance(first, SessionKeys):
        raise ProtocolValidationError(
            "First value must be a SessionKeys object.",
            details={
                "received_type": type(first).__name__,
            },
        )

    if not isinstance(second, SessionKeys):
        raise ProtocolValidationError(
            "Second value must be a SessionKeys object.",
            details={
                "received_type": type(second).__name__,
            },
        )

    return all(
        (
            first.session_id == second.session_id,

            compare_digest(
                first.transcript_hash,
                second.transcript_hash,
            ),

            compare_digest(
                first.master_key,
                second.master_key,
            ),

            compare_digest(
                first.authentication_key,
                second.authentication_key,
            ),

            compare_digest(
                first.control_key,
                second.control_key,
            ),
        )
    )


def session_key_fingerprints(
    session_keys: SessionKeys,
    *,
    fingerprint_bytes: int = 8,
) -> dict[str, Any]:
    """
    Return short non-secret fingerprints for dashboard diagnostics.

    Full secret keys are never returned.

    A fingerprint is derived using SHA3-256 and truncated to the
    requested number of bytes.
    """

    if not isinstance(
        session_keys,
        SessionKeys,
    ):
        raise ProtocolValidationError(
            "session_keys must be a SessionKeys object.",
            details={
                "received_type": (
                    type(session_keys).__name__
                ),
            },
        )

    validated_length = validate_integer(
        fingerprint_bytes,
        field_name="fingerprint_bytes",
        minimum=4,
        maximum=32,
    )

    def create_fingerprint(
        key: bytes,
    ) -> str:
        digest = hashes.Hash(
            hashes.SHA3_256()
        )

        digest.update(key)

        return digest.finalize()[
            :validated_length
        ].hex()

    return {
        "session_id": session_keys.session_id,
        "transcript_hash": (
            session_keys.transcript_hash.hex()
        ),
        "master_key_fingerprint": (
            create_fingerprint(
                session_keys.master_key
            )
        ),
        "authentication_key_fingerprint": (
            create_fingerprint(
                session_keys.authentication_key
            )
        ),
        "control_key_fingerprint": (
            create_fingerprint(
                session_keys.control_key
            )
        ),
    }


def run_kdf_self_test() -> dict[str, Any]:
    """
    Run a deterministic key-derivation self-test.

    The self-test confirms:

    - Same inputs produce identical keys
    - Different transcript hashes produce different keys
    - Authentication and control keys are separated
    """

    shared_secret = bytes(
        range(32)
    )

    transcript_hash_one = bytes(
        range(32)
    )

    transcript_hash_two = bytes(
        reversed(
            range(32)
        )
    )

    session_id = "FTQ-KDF-SELF-TEST"

    first = derive_session_keys(
        shared_secret=shared_secret,
        transcript_hash=transcript_hash_one,
        session_id=session_id,
        attempt_number=1,
    )

    second = derive_session_keys(
        shared_secret=shared_secret,
        transcript_hash=transcript_hash_one,
        session_id=session_id,
        attempt_number=1,
    )

    changed_transcript = derive_session_keys(
        shared_secret=shared_secret,
        transcript_hash=transcript_hash_two,
        session_id=session_id,
        attempt_number=1,
    )

    deterministic_pass = compare_session_keys(
        first,
        second,
    )

    transcript_binding_pass = not compare_digest(
        first.master_key,
        changed_transcript.master_key,
    )

    key_separation_pass = not compare_digest(
        first.authentication_key,
        first.control_key,
    )

    success = all(
        (
            deterministic_pass,
            transcript_binding_pass,
            key_separation_pass,
        )
    )

    return {
        "success": success,
        "hash_algorithm": KDF_HASH_NAME,
        "master_key_bytes": MASTER_KEY_BYTES,
        "authentication_key_bytes": (
            AUTHENTICATION_KEY_BYTES
        ),
        "control_key_bytes": CONTROL_KEY_BYTES,
        "deterministic_pass": deterministic_pass,
        "transcript_binding_pass": (
            transcript_binding_pass
        ),
        "key_separation_pass": (
            key_separation_pass
        ),
        "fingerprints": (
            session_key_fingerprints(first)
        ),
    }


__all__ = [
    "MASTER_KEY_BYTES",
    "AUTHENTICATION_KEY_BYTES",
    "CONTROL_KEY_BYTES",
    "TRANSCRIPT_HASH_BYTES",
    "KDF_HASH_NAME",
    "create_kdf_hash_algorithm",
    "derive_master_key",
    "expand_session_subkey",
    "derive_session_keys",
    "compare_session_keys",
    "session_key_fingerprints",
    "run_kdf_self_test",
]