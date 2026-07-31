"""
Transcript-bound session-key derivation for FT-QuPAP v5.1.

After successful ML-KEM decapsulation, the Authentication Server obtains
a 32-byte shared secret. The shared secret is never used directly.

This module derives independent keys for:

- Session master-key binding
- KMAC authentication-tag verification
- AES-GCM control-schedule protection
- Key confirmation
- Retry-attempt binding

The derivation is bound to:

- FT-QuPAP protocol domain
- Session identifier
- Authentication-attempt number
- Subscriber pseudonymous identity
- Mobile Station nonce
- Authentication Server nonce
- Protocol transcript digest

Changing any bound value produces independent session keys.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Mapping

from src.common.constants import (
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
    encode_base64,
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

SESSION_KDF_ALGORITHM = "HKDF-SHA3-256"

HKDF_HASH_LENGTH = 32

MLKEM_SHARED_SECRET_BYTES = 32

SESSION_MASTER_KEY_BYTES = 32

KMAC_AUTHENTICATION_KEY_BYTES = 32

CONTROL_SCHEDULE_KEY_BYTES = 32

KEY_CONFIRMATION_KEY_BYTES = 32

RETRY_BINDING_KEY_BYTES = 32

TRANSCRIPT_DIGEST_BYTES = 32


SESSION_MASTER_LABEL = (
    b"FT-QuPAP-v5.1/session-master"
)

KMAC_AUTHENTICATION_LABEL = (
    b"FT-QuPAP-v5.1/kmac-authentication"
)

CONTROL_SCHEDULE_LABEL = (
    b"FT-QuPAP-v5.1/control-schedule"
)

KEY_CONFIRMATION_LABEL = (
    b"FT-QuPAP-v5.1/key-confirmation"
)

RETRY_BINDING_LABEL = (
    b"FT-QuPAP-v5.1/retry-binding"
)


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class SessionKeyDerivationError(RuntimeError):
    """Raised when session-key derivation cannot be completed safely."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.details = (
            {}
            if details is None
            else dict(details)
        )


# ---------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SessionKeyMaterial:
    """
    FT-QuPAP session-specific derived keys.

    Secret key bytes are excluded from public output and hidden from the
    object representation.
    """

    session_id: str
    attempt_number: int
    pseudonym_id: str

    algorithm: str

    transcript_digest: bytes
    derivation_salt: bytes

    session_master_key: bytes
    kmac_authentication_key: bytes
    control_schedule_key: bytes
    key_confirmation_key: bytes
    retry_binding_key: bytes

    key_material_fingerprint: str

    def __post_init__(self) -> None:
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

        validate_non_empty_string(
            self.algorithm,
            field_name="algorithm",
            minimum_length=1,
            maximum_length=64,
        )

        validate_bytes(
            self.transcript_digest,
            field_name="transcript_digest",
            exact_length=TRANSCRIPT_DIGEST_BYTES,
        )

        validate_bytes(
            self.derivation_salt,
            field_name="derivation_salt",
            exact_length=HKDF_HASH_LENGTH,
        )

        key_fields = {
            "session_master_key": (
                self.session_master_key,
                SESSION_MASTER_KEY_BYTES,
            ),

            "kmac_authentication_key": (
                self.kmac_authentication_key,
                KMAC_AUTHENTICATION_KEY_BYTES,
            ),

            "control_schedule_key": (
                self.control_schedule_key,
                CONTROL_SCHEDULE_KEY_BYTES,
            ),

            "key_confirmation_key": (
                self.key_confirmation_key,
                KEY_CONFIRMATION_KEY_BYTES,
            ),

            "retry_binding_key": (
                self.retry_binding_key,
                RETRY_BINDING_KEY_BYTES,
            ),
        }

        for (
            field_name,
            (
                key_value,
                expected_length,
            ),
        ) in key_fields.items():
            validate_bytes(
                key_value,
                field_name=field_name,
                exact_length=expected_length,
            )

        validated_fingerprint = (
            validate_non_empty_string(
                self.key_material_fingerprint,
                field_name=(
                    "key_material_fingerprint"
                ),
                minimum_length=64,
                maximum_length=64,
            )
            .lower()
        )

        try:
            bytes.fromhex(
                validated_fingerprint
            )

        except ValueError as exc:
            raise ProtocolValidationError(
                (
                    "key_material_fingerprint must be "
                    "valid hexadecimal text."
                )
            ) from exc

        expected_fingerprint = (
            calculate_key_material_fingerprint(
                session_master_key=(
                    self.session_master_key
                ),
                kmac_authentication_key=(
                    self.kmac_authentication_key
                ),
                control_schedule_key=(
                    self.control_schedule_key
                ),
                key_confirmation_key=(
                    self.key_confirmation_key
                ),
                retry_binding_key=(
                    self.retry_binding_key
                ),
            )
        )

        if not hmac.compare_digest(
            validated_fingerprint,
            expected_fingerprint,
        ):
            raise ProtocolValidationError(
                (
                    "Derived-key fingerprint does not match "
                    "the session key material."
                )
            )

    def public_dict(self) -> dict[str, Any]:
        """
        Return non-secret session-key metadata.

        No derived key is included.
        """

        return {
            "session_id": self.session_id,
            "attempt_number": (
                self.attempt_number
            ),
            "pseudonym_id": (
                self.pseudonym_id
            ),
            "algorithm": self.algorithm,
            "transcript_digest": (
                self.transcript_digest.hex()
            ),
            "derivation_salt": (
                self.derivation_salt.hex()
            ),
            "key_material_fingerprint": (
                self.key_material_fingerprint
            ),
            "session_master_key_bytes": len(
                self.session_master_key
            ),
            "kmac_authentication_key_bytes": len(
                self.kmac_authentication_key
            ),
            "control_schedule_key_bytes": len(
                self.control_schedule_key
            ),
            "key_confirmation_key_bytes": len(
                self.key_confirmation_key
            ),
            "retry_binding_key_bytes": len(
                self.retry_binding_key
            ),
        }

    def protected_dict(self) -> dict[str, Any]:
        """
        Return complete key material for protected internal processing.

        This output must never be written to ordinary logs.
        """

        result = self.public_dict()

        result.update(
            {
                "session_master_key": (
                    encode_base64(
                        self.session_master_key
                    )
                ),

                "kmac_authentication_key": (
                    encode_base64(
                        self.kmac_authentication_key
                    )
                ),

                "control_schedule_key": (
                    encode_base64(
                        self.control_schedule_key
                    )
                ),

                "key_confirmation_key": (
                    encode_base64(
                        self.key_confirmation_key
                    )
                ),

                "retry_binding_key": (
                    encode_base64(
                        self.retry_binding_key
                    )
                ),
            }
        )

        return result

    def __repr__(self) -> str:
        return (
            "SessionKeyMaterial("
            f"session_id={self.session_id!r}, "
            f"attempt_number={self.attempt_number}, "
            f"pseudonym_id={self.pseudonym_id!r}, "
            f"algorithm={self.algorithm!r}, "
            f"transcript_digest="
            f"{self.transcript_digest.hex()!r}, "
            f"key_material_fingerprint="
            f"{self.key_material_fingerprint!r}, "
            "session_master_key=<hidden>, "
            "kmac_authentication_key=<hidden>, "
            "control_schedule_key=<hidden>, "
            "key_confirmation_key=<hidden>, "
            "retry_binding_key=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------

def normalize_nonce(
    nonce: bytes | str,
    *,
    field_name: str,
) -> bytes:
    """
    Normalize a protocol nonce from raw bytes or hexadecimal text.
    """

    if isinstance(
        nonce,
        bytes,
    ):
        return validate_bytes(
            nonce,
            field_name=field_name,
            minimum_length=16,
            maximum_length=64,
        )

    if not isinstance(
        nonce,
        str,
    ):
        raise ProtocolValidationError(
            (
                f"{field_name} must be bytes or "
                "hexadecimal text."
            )
        )

    normalized_text = (
        validate_non_empty_string(
            nonce,
            field_name=field_name,
            minimum_length=32,
            maximum_length=128,
        )
        .strip()
        .lower()
    )

    if len(normalized_text) % 2 != 0:
        raise ProtocolValidationError(
            (
                f"{field_name} hexadecimal length "
                "must be even."
            )
        )

    try:
        nonce_bytes = bytes.fromhex(
            normalized_text
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            (
                f"{field_name} must contain valid "
                "hexadecimal text."
            )
        ) from exc

    return validate_bytes(
        nonce_bytes,
        field_name=field_name,
        minimum_length=16,
        maximum_length=64,
    )


def normalize_transcript_digest(
    transcript_digest: bytes | str,
) -> bytes:
    """
    Normalize a SHA3-256 transcript digest.
    """

    if isinstance(
        transcript_digest,
        bytes,
    ):
        return validate_bytes(
            transcript_digest,
            field_name="transcript_digest",
            exact_length=TRANSCRIPT_DIGEST_BYTES,
        )

    if not isinstance(
        transcript_digest,
        str,
    ):
        raise ProtocolValidationError(
            (
                "transcript_digest must be bytes "
                "or hexadecimal text."
            )
        )

    normalized_text = (
        validate_non_empty_string(
            transcript_digest,
            field_name="transcript_digest",
            minimum_length=64,
            maximum_length=64,
        )
        .strip()
        .lower()
    )

    try:
        digest_bytes = bytes.fromhex(
            normalized_text
        )

    except ValueError as exc:
        raise ProtocolValidationError(
            (
                "transcript_digest must contain "
                "valid hexadecimal text."
            )
        ) from exc

    return validate_bytes(
        digest_bytes,
        field_name="transcript_digest",
        exact_length=TRANSCRIPT_DIGEST_BYTES,
    )


# ---------------------------------------------------------------------
# HKDF implementation
# ---------------------------------------------------------------------

def hkdf_extract_sha3_256(
    *,
    salt: bytes,
    input_key_material: bytes,
) -> bytes:
    """
    Perform HKDF-Extract using HMAC-SHA3-256.

    Formula:

        PRK = HMAC-SHA3-256(salt, IKM)
    """

    validated_salt = validate_bytes(
        salt,
        field_name="hkdf_salt",
        exact_length=HKDF_HASH_LENGTH,
    )

    validated_ikm = validate_bytes(
        input_key_material,
        field_name="input_key_material",
        minimum_length=16,
        maximum_length=1_000_000,
    )

    return hmac.new(
        validated_salt,
        validated_ikm,
        hashlib.sha3_256,
    ).digest()


def hkdf_expand_sha3_256(
    *,
    pseudorandom_key: bytes,
    info: bytes,
    output_length: int,
) -> bytes:
    """
    Perform HKDF-Expand using HMAC-SHA3-256.

    Formula:

        T(0) = empty
        T(i) = HMAC(PRK, T(i-1) || info || i)
        OKM  = T(1) || T(2) || ...
    """

    validated_prk = validate_bytes(
        pseudorandom_key,
        field_name="pseudorandom_key",
        exact_length=HKDF_HASH_LENGTH,
    )

    validated_info = validate_bytes(
        info,
        field_name="hkdf_info",
        minimum_length=1,
        maximum_length=10_000,
    )

    validated_length = validate_integer(
        output_length,
        field_name="output_length",
        minimum=1,
        maximum=(
            255 * HKDF_HASH_LENGTH
        ),
    )

    output = bytearray()
    previous_block = b""

    block_number = 1

    while len(output) < validated_length:
        previous_block = hmac.new(
            validated_prk,
            (
                previous_block
                + validated_info
                + bytes(
                    [block_number]
                )
            ),
            hashlib.sha3_256,
        ).digest()

        output.extend(
            previous_block
        )

        block_number += 1

    return bytes(
        output[:validated_length]
    )


def hkdf_sha3_256(
    *,
    input_key_material: bytes,
    salt: bytes,
    info: bytes,
    output_length: int,
) -> bytes:
    """
    Perform complete HKDF-SHA3-256 derivation.
    """

    pseudorandom_key = (
        hkdf_extract_sha3_256(
            salt=salt,
            input_key_material=(
                input_key_material
            ),
        )
    )

    return hkdf_expand_sha3_256(
        pseudorandom_key=(
            pseudorandom_key
        ),
        info=info,
        output_length=output_length,
    )


# ---------------------------------------------------------------------
# Context and salt construction
# ---------------------------------------------------------------------

def build_session_derivation_context(
    *,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
) -> dict[str, Any]:
    """
    Build the canonical transcript-bound KDF context.
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

    return {
        "protocol_domain": (
            PROTOCOL_DOMAIN_LABEL.decode(
                "utf-8",
                errors="strict",
            )
        ),

        "purpose": (
            "ft-qupap-session-key-derivation"
        ),

        "algorithm": SESSION_KDF_ALGORITHM,

        "session_id": validated_session_id,

        "attempt_number": (
            validated_attempt
        ),

        "pseudonym_id": (
            validated_pseudonym
        ),

        "mobile_nonce": (
            encode_base64(
                normalized_mobile_nonce
            )
        ),

        "server_nonce": (
            encode_base64(
                normalized_server_nonce
            )
        ),

        "transcript_digest": (
            normalized_transcript.hex()
        ),
    }


def calculate_session_derivation_salt(
    context: Mapping[str, Any],
) -> bytes:
    """
    Calculate the 32-byte KDF salt from canonical session context.
    """

    if not isinstance(
        context,
        Mapping,
    ):
        raise ProtocolValidationError(
            "context must be a mapping."
        )

    encoded_context = canonical_json_bytes(
        dict(context)
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00session-kdf-salt\x00"
    )

    digest.update(
        encoded_context
    )

    return digest.digest()


def build_key_information(
    *,
    label: bytes,
    context: Mapping[str, Any],
) -> bytes:
    """
    Build a canonical HKDF information field for one derived key.
    """

    validated_label = validate_bytes(
        label,
        field_name="key_label",
        minimum_length=1,
        maximum_length=256,
    )

    if not isinstance(
        context,
        Mapping,
    ):
        raise ProtocolValidationError(
            "context must be a mapping."
        )

    return canonical_json_bytes(
        {
            "label": (
                validated_label.decode(
                    "utf-8",
                    errors="strict",
                )
            ),

            "context": dict(
                context
            ),
        }
    )


# ---------------------------------------------------------------------
# Key-material fingerprint
# ---------------------------------------------------------------------

def calculate_key_material_fingerprint(
    *,
    session_master_key: bytes,
    kmac_authentication_key: bytes,
    control_schedule_key: bytes,
    key_confirmation_key: bytes,
    retry_binding_key: bytes,
) -> str:
    """
    Calculate a diagnostic fingerprint of the complete key set.

    The fingerprint cannot be used to reconstruct the keys.
    """

    key_values = (
        validate_bytes(
            session_master_key,
            field_name="session_master_key",
            exact_length=(
                SESSION_MASTER_KEY_BYTES
            ),
        ),

        validate_bytes(
            kmac_authentication_key,
            field_name=(
                "kmac_authentication_key"
            ),
            exact_length=(
                KMAC_AUTHENTICATION_KEY_BYTES
            ),
        ),

        validate_bytes(
            control_schedule_key,
            field_name=(
                "control_schedule_key"
            ),
            exact_length=(
                CONTROL_SCHEDULE_KEY_BYTES
            ),
        ),

        validate_bytes(
            key_confirmation_key,
            field_name=(
                "key_confirmation_key"
            ),
            exact_length=(
                KEY_CONFIRMATION_KEY_BYTES
            ),
        ),

        validate_bytes(
            retry_binding_key,
            field_name=(
                "retry_binding_key"
            ),
            exact_length=(
                RETRY_BINDING_KEY_BYTES
            ),
        ),
    )

    digest = hashlib.sha3_256()

    digest.update(
        PROTOCOL_DOMAIN_LABEL
    )

    digest.update(
        b"\x00derived-key-fingerprint\x00"
    )

    for key_value in key_values:
        digest.update(
            len(key_value).to_bytes(
                4,
                byteorder="big",
                signed=False,
            )
        )

        digest.update(
            key_value
        )

    return digest.hexdigest()


# ---------------------------------------------------------------------
# Main derivation
# ---------------------------------------------------------------------

def derive_session_keys(
    *,
    shared_secret: bytes,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
) -> SessionKeyMaterial:
    """
    Derive all FT-QuPAP session-specific keys.

    The ML-KEM shared secret is used only as HKDF input key material.
    """

    validated_shared_secret = (
        validate_bytes(
            shared_secret,
            field_name="mlkem_shared_secret",
            exact_length=(
                MLKEM_SHARED_SECRET_BYTES
            ),
        )
    )

    context = (
        build_session_derivation_context(
            session_id=session_id,
            attempt_number=attempt_number,
            pseudonym_id=pseudonym_id,
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                transcript_digest
            ),
        )
    )

    normalized_transcript = bytes.fromhex(
        context[
            "transcript_digest"
        ]
    )

    derivation_salt = (
        calculate_session_derivation_salt(
            context
        )
    )

    pseudorandom_key = (
        hkdf_extract_sha3_256(
            salt=derivation_salt,
            input_key_material=(
                validated_shared_secret
            ),
        )
    )

    def derive_key(
        label: bytes,
        length: int,
    ) -> bytes:
        info = build_key_information(
            label=label,
            context=context,
        )

        return hkdf_expand_sha3_256(
            pseudorandom_key=(
                pseudorandom_key
            ),
            info=info,
            output_length=length,
        )

    session_master_key = derive_key(
        SESSION_MASTER_LABEL,
        SESSION_MASTER_KEY_BYTES,
    )

    kmac_authentication_key = derive_key(
        KMAC_AUTHENTICATION_LABEL,
        KMAC_AUTHENTICATION_KEY_BYTES,
    )

    control_schedule_key = derive_key(
        CONTROL_SCHEDULE_LABEL,
        CONTROL_SCHEDULE_KEY_BYTES,
    )

    key_confirmation_key = derive_key(
        KEY_CONFIRMATION_LABEL,
        KEY_CONFIRMATION_KEY_BYTES,
    )

    retry_binding_key = derive_key(
        RETRY_BINDING_LABEL,
        RETRY_BINDING_KEY_BYTES,
    )

    fingerprint = (
        calculate_key_material_fingerprint(
            session_master_key=(
                session_master_key
            ),
            kmac_authentication_key=(
                kmac_authentication_key
            ),
            control_schedule_key=(
                control_schedule_key
            ),
            key_confirmation_key=(
                key_confirmation_key
            ),
            retry_binding_key=(
                retry_binding_key
            ),
        )
    )

    return SessionKeyMaterial(
        session_id=context[
            "session_id"
        ],

        attempt_number=context[
            "attempt_number"
        ],

        pseudonym_id=context[
            "pseudonym_id"
        ],

        algorithm=SESSION_KDF_ALGORITHM,

        transcript_digest=(
            normalized_transcript
        ),

        derivation_salt=(
            derivation_salt
        ),

        session_master_key=(
            session_master_key
        ),

        kmac_authentication_key=(
            kmac_authentication_key
        ),

        control_schedule_key=(
            control_schedule_key
        ),

        key_confirmation_key=(
            key_confirmation_key
        ),

        retry_binding_key=(
            retry_binding_key
        ),

        key_material_fingerprint=(
            fingerprint
        ),
    )


def derive_kmac_authentication_key(
    *,
    shared_secret: bytes,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
) -> bytes:
    """
    Derive and return only the KMAC authentication key.
    """

    material = derive_session_keys(
        shared_secret=shared_secret,
        session_id=session_id,
        attempt_number=attempt_number,
        pseudonym_id=pseudonym_id,
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=transcript_digest,
    )

    return bytes(
        material.kmac_authentication_key
    )


def derive_control_schedule_key(
    *,
    shared_secret: bytes,
    session_id: str,
    attempt_number: int,
    pseudonym_id: str,
    mobile_nonce: bytes | str,
    server_nonce: bytes | str,
    transcript_digest: bytes | str,
) -> bytes:
    """
    Derive and return only the AES-256-GCM control-schedule key.
    """

    material = derive_session_keys(
        shared_secret=shared_secret,
        session_id=session_id,
        attempt_number=attempt_number,
        pseudonym_id=pseudonym_id,
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=transcript_digest,
    )

    return bytes(
        material.control_schedule_key
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_session_key_derivation_self_test() -> dict[str, Any]:
    """
    Test determinism, key separation, attempt binding, transcript
    binding, and secret-safe public output.
    """

    shared_secret = bytes(
        range(32)
    )

    mobile_nonce = b"M" * 32
    server_nonce = b"S" * 32

    transcript_digest = hashlib.sha3_256(
        b"FT-QuPAP transcript"
    ).digest()

    first_result = derive_session_keys(
        shared_secret=shared_secret,
        session_id=(
            "FTQ-SESSION-KDF-SELF-TEST"
        ),
        attempt_number=1,
        pseudonym_id=(
            "PID-FTQ-KDF-SELF-TEST"
        ),
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=(
            transcript_digest
        ),
    )

    repeated_result = derive_session_keys(
        shared_secret=shared_secret,
        session_id=(
            "FTQ-SESSION-KDF-SELF-TEST"
        ),
        attempt_number=1,
        pseudonym_id=(
            "PID-FTQ-KDF-SELF-TEST"
        ),
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=(
            transcript_digest
        ),
    )

    retry_result = derive_session_keys(
        shared_secret=shared_secret,
        session_id=(
            "FTQ-SESSION-KDF-SELF-TEST"
        ),
        attempt_number=2,
        pseudonym_id=(
            "PID-FTQ-KDF-SELF-TEST"
        ),
        mobile_nonce=mobile_nonce,
        server_nonce=server_nonce,
        transcript_digest=(
            transcript_digest
        ),
    )

    changed_transcript_result = (
        derive_session_keys(
            shared_secret=shared_secret,
            session_id=(
                "FTQ-SESSION-KDF-SELF-TEST"
            ),
            attempt_number=1,
            pseudonym_id=(
                "PID-FTQ-KDF-SELF-TEST"
            ),
            mobile_nonce=mobile_nonce,
            server_nonce=server_nonce,
            transcript_digest=(
                hashlib.sha3_256(
                    b"Changed transcript"
                ).digest()
            ),
        )
    )

    deterministic_pass = all(
        (
            hmac.compare_digest(
                first_result.session_master_key,
                repeated_result.session_master_key,
            ),

            hmac.compare_digest(
                first_result
                .kmac_authentication_key,
                repeated_result
                .kmac_authentication_key,
            ),

            hmac.compare_digest(
                first_result
                .control_schedule_key,
                repeated_result
                .control_schedule_key,
            ),

            first_result
            .key_material_fingerprint
            == repeated_result
            .key_material_fingerprint,
        )
    )

    separated_keys = {
        first_result.session_master_key,
        first_result.kmac_authentication_key,
        first_result.control_schedule_key,
        first_result.key_confirmation_key,
        first_result.retry_binding_key,
    }

    key_separation_pass = (
        len(separated_keys) == 5
    )

    retry_binding_pass = (
        first_result.key_material_fingerprint
        != retry_result.key_material_fingerprint
    )

    transcript_binding_pass = (
        first_result.key_material_fingerprint
        != changed_transcript_result
        .key_material_fingerprint
    )

    public_output = (
        first_result.public_dict()
    )

    secret_fields_absent = all(
        field_name not in public_output
        for field_name in (
            "session_master_key",
            "kmac_authentication_key",
            "control_schedule_key",
            "key_confirmation_key",
            "retry_binding_key",
        )
    )

    correct_lengths = all(
        (
            len(
                first_result.session_master_key
            )
            == SESSION_MASTER_KEY_BYTES,

            len(
                first_result
                .kmac_authentication_key
            )
            == KMAC_AUTHENTICATION_KEY_BYTES,

            len(
                first_result
                .control_schedule_key
            )
            == CONTROL_SCHEDULE_KEY_BYTES,

            len(
                first_result
                .key_confirmation_key
            )
            == KEY_CONFIRMATION_KEY_BYTES,

            len(
                first_result
                .retry_binding_key
            )
            == RETRY_BINDING_KEY_BYTES,
        )
    )

    success = all(
        (
            deterministic_pass,
            key_separation_pass,
            retry_binding_pass,
            transcript_binding_pass,
            secret_fields_absent,
            correct_lengths,
        )
    )

    return {
        "success": success,

        "algorithm": (
            first_result.algorithm
        ),

        "deterministic_derivation": (
            deterministic_pass
        ),

        "key_separation_pass": (
            key_separation_pass
        ),

        "retry_attempt_changes_keys": (
            retry_binding_pass
        ),

        "transcript_changes_keys": (
            transcript_binding_pass
        ),

        "secret_fields_absent": (
            secret_fields_absent
        ),

        "correct_key_lengths": (
            correct_lengths
        ),

        "session_master_key_bytes": len(
            first_result.session_master_key
        ),

        "kmac_authentication_key_bytes": len(
            first_result
            .kmac_authentication_key
        ),

        "control_schedule_key_bytes": len(
            first_result.control_schedule_key
        ),

        "key_confirmation_key_bytes": len(
            first_result.key_confirmation_key
        ),

        "retry_binding_key_bytes": len(
            first_result.retry_binding_key
        ),

        "key_material_fingerprint": (
            first_result
            .key_material_fingerprint
        ),
    }


__all__ = [
    "SESSION_KDF_ALGORITHM",
    "HKDF_HASH_LENGTH",
    "MLKEM_SHARED_SECRET_BYTES",
    "SESSION_MASTER_KEY_BYTES",
    "KMAC_AUTHENTICATION_KEY_BYTES",
    "CONTROL_SCHEDULE_KEY_BYTES",
    "KEY_CONFIRMATION_KEY_BYTES",
    "RETRY_BINDING_KEY_BYTES",
    "TRANSCRIPT_DIGEST_BYTES",
    "SessionKeyDerivationError",
    "SessionKeyMaterial",
    "normalize_nonce",
    "normalize_transcript_digest",
    "hkdf_extract_sha3_256",
    "hkdf_expand_sha3_256",
    "hkdf_sha3_256",
    "build_session_derivation_context",
    "calculate_session_derivation_salt",
    "build_key_information",
    "calculate_key_material_fingerprint",
    "derive_session_keys",
    "derive_kmac_authentication_key",
    "derive_control_schedule_key",
    "run_session_key_derivation_self_test",
]