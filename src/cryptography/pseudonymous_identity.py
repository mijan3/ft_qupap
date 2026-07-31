"""
Pseudonymous identity management for FT-QuPAP v5.1.

The Mobile Station must not transmit its permanent subscriber identity
during authentication. Instead, it sends a pseudonymous identity.

This module provides:

- Random pseudonymous identity generation
- Secret-key-based deterministic pseudonym derivation
- Periodic pseudonym rotation
- Pseudonym-binding verification
- Safe identity masking
- Pseudonymous identity self-testing

A pseudonym is not an encryption of the permanent identity. It is a
keyed, one-way identifier derived using HMAC-SHA3-256.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from typing import Any

from src.common.exceptions import (
    CryptographicError,
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
    validate_pseudonym_id,
)


DEFAULT_PSEUDONYM_PREFIX = "PID-6G-UE"

DEFAULT_PSEUDONYM_DIGEST_BYTES = 12

MINIMUM_IDENTITY_KEY_BYTES = 32

MAXIMUM_PERMANENT_IDENTITY_LENGTH = 128


PERMANENT_IDENTITY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:+@-]{2,127}$"
)


@dataclass(frozen=True)
class PseudonymousIdentity:
    """
    One pseudonymous subscriber identity.

    Attributes
    ----------
    pseudonym_id:
        Public identity sent during authentication.

    scope:
        Network or application scope for which the pseudonym is valid.

    epoch:
        Rotation period identifier.

    version:
        Pseudonym-generation format version.
    """

    pseudonym_id: str
    scope: str
    epoch: int
    version: str = "FT-QuPAP-PID-v1"

    def __post_init__(self) -> None:
        validate_pseudonym_id(
            self.pseudonym_id
        )

        validate_non_empty_string(
            self.scope,
            field_name="scope",
            minimum_length=1,
            maximum_length=128,
        )

        validate_integer(
            self.epoch,
            field_name="epoch",
            minimum=0,
        )

        validate_non_empty_string(
            self.version,
            field_name="version",
            minimum_length=1,
            maximum_length=64,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the pseudonymous identity into JSON-compatible data.
        """

        return {
            "pseudonym_id": self.pseudonym_id,
            "scope": self.scope,
            "epoch": self.epoch,
            "version": self.version,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "PseudonymousIdentity":
        """
        Restore a pseudonymous identity from dictionary data.
        """

        required_fields = (
            "pseudonym_id",
            "scope",
            "epoch",
        )

        missing_fields = [
            field_name
            for field_name in required_fields
            if field_name not in data
        ]

        if missing_fields:
            raise ProtocolValidationError(
                "Pseudonymous identity data is incomplete.",
                details={
                    "missing_fields": missing_fields,
                },
            )

        return cls(
            pseudonym_id=data["pseudonym_id"],
            scope=data["scope"],
            epoch=data["epoch"],
            version=data.get(
                "version",
                "FT-QuPAP-PID-v1",
            ),
        )


def normalize_permanent_identity(
    permanent_identity: str,
) -> str:
    """
    Validate and normalize a permanent subscriber identity.

    The permanent identity remains local and must never be inserted
    directly into an authentication message.
    """

    normalized = validate_non_empty_string(
        permanent_identity,
        field_name="permanent_identity",
        minimum_length=3,
        maximum_length=(
            MAXIMUM_PERMANENT_IDENTITY_LENGTH
        ),
    )

    if (
        PERMANENT_IDENTITY_PATTERN.fullmatch(
            normalized
        )
        is None
    ):
        raise ProtocolValidationError(
            "Permanent identity contains unsupported characters.",
            details={
                "allowed_pattern": (
                    PERMANENT_IDENTITY_PATTERN.pattern
                ),
            },
        )

    return normalized


def normalize_scope(
    scope: str,
) -> str:
    """
    Validate and normalize the pseudonym scope.
    """

    return validate_non_empty_string(
        scope,
        field_name="scope",
        minimum_length=1,
        maximum_length=128,
    ).lower()


def normalize_pseudonym_prefix(
    prefix: str,
) -> str:
    """
    Validate a pseudonym prefix.

    Example:

        PID-6G-UE
    """

    normalized = validate_non_empty_string(
        prefix,
        field_name="pseudonym_prefix",
        minimum_length=2,
        maximum_length=32,
    ).upper()

    if (
        re.fullmatch(
            r"[A-Z0-9][A-Z0-9-]{1,31}",
            normalized,
        )
        is None
    ):
        raise ProtocolValidationError(
            "Pseudonym prefix contains invalid characters.",
            details={
                "prefix": normalized,
            },
        )

    return normalized


def generate_identity_key(
    byte_length: int = MINIMUM_IDENTITY_KEY_BYTES,
) -> bytes:
    """
    Generate a secret key for deterministic pseudonym derivation.

    This key must be stored securely and must not be transmitted.
    """

    validated_length = validate_integer(
        byte_length,
        field_name="byte_length",
        minimum=MINIMUM_IDENTITY_KEY_BYTES,
        maximum=128,
    )

    return secrets.token_bytes(
        validated_length
    )


def generate_random_pseudonym(
    *,
    prefix: str = DEFAULT_PSEUDONYM_PREFIX,
    random_bytes: int = DEFAULT_PSEUDONYM_DIGEST_BYTES,
) -> str:
    """
    Generate a cryptographically random pseudonymous identity.

    Example:

        PID-6G-UE-A81F3904B12C9E447A503811
    """

    normalized_prefix = (
        normalize_pseudonym_prefix(
            prefix
        )
    )

    validated_random_bytes = validate_integer(
        random_bytes,
        field_name="random_bytes",
        minimum=8,
        maximum=32,
    )

    random_identifier = secrets.token_hex(
        validated_random_bytes
    ).upper()

    pseudonym_id = (
        f"{normalized_prefix}-"
        f"{random_identifier}"
    )

    return validate_pseudonym_id(
        pseudonym_id
    )


def build_pseudonym_derivation_message(
    *,
    permanent_identity: str,
    scope: str,
    epoch: int,
    version: str = "FT-QuPAP-PID-v1",
) -> bytes:
    """
    Build the canonical message used for pseudonym derivation.

    The identity key is not included in this message. It is used as the
    HMAC key.
    """

    validated_identity = (
        normalize_permanent_identity(
            permanent_identity
        )
    )

    validated_scope = normalize_scope(
        scope
    )

    validated_epoch = validate_integer(
        epoch,
        field_name="epoch",
        minimum=0,
    )

    validated_version = validate_non_empty_string(
        version,
        field_name="version",
        minimum_length=1,
        maximum_length=64,
    )

    payload = {
        "domain": "FT-QuPAP",
        "purpose": "pseudonymous-subscriber-identity",
        "version": validated_version,
        "permanent_identity": validated_identity,
        "scope": validated_scope,
        "epoch": validated_epoch,
    }

    return canonical_json_bytes(
        payload
    )


def derive_pseudonymous_identity(
    *,
    permanent_identity: str,
    identity_key: bytes,
    scope: str,
    epoch: int,
    prefix: str = DEFAULT_PSEUDONYM_PREFIX,
    digest_bytes: int = DEFAULT_PSEUDONYM_DIGEST_BYTES,
) -> PseudonymousIdentity:
    """
    Derive a deterministic pseudonymous identity using HMAC-SHA3-256.

    The same values produce the same pseudonym:

        permanent identity
        identity key
        scope
        epoch

    Changing the epoch rotates the pseudonym.
    """

    validated_key = validate_bytes(
        identity_key,
        field_name="identity_key",
        minimum_length=MINIMUM_IDENTITY_KEY_BYTES,
        maximum_length=4096,
    )

    normalized_prefix = (
        normalize_pseudonym_prefix(
            prefix
        )
    )

    validated_digest_bytes = validate_integer(
        digest_bytes,
        field_name="digest_bytes",
        minimum=8,
        maximum=32,
    )

    validated_scope = normalize_scope(
        scope
    )

    validated_epoch = validate_integer(
        epoch,
        field_name="epoch",
        minimum=0,
    )

    message = build_pseudonym_derivation_message(
        permanent_identity=permanent_identity,
        scope=validated_scope,
        epoch=validated_epoch,
    )

    try:
        digest = hmac.new(
            validated_key,
            message,
            hashlib.sha3_256,
        ).digest()

    except Exception as exc:
        raise CryptographicError(
            "Unable to derive pseudonymous identity.",
            code="PSEUDONYM_DERIVATION_ERROR",
            details={
                "scope": validated_scope,
                "epoch": validated_epoch,
                "reason": str(exc),
            },
        ) from exc

    pseudonym_id = (
        f"{normalized_prefix}-"
        f"{digest[:validated_digest_bytes].hex().upper()}"
    )

    return PseudonymousIdentity(
        pseudonym_id=validate_pseudonym_id(
            pseudonym_id
        ),
        scope=validated_scope,
        epoch=validated_epoch,
    )


def rotate_pseudonymous_identity(
    *,
    permanent_identity: str,
    identity_key: bytes,
    scope: str,
    current_epoch: int,
    prefix: str = DEFAULT_PSEUDONYM_PREFIX,
) -> PseudonymousIdentity:
    """
    Create the pseudonym for the next rotation epoch.
    """

    validated_epoch = validate_integer(
        current_epoch,
        field_name="current_epoch",
        minimum=0,
    )

    return derive_pseudonymous_identity(
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope=scope,
        epoch=validated_epoch + 1,
        prefix=prefix,
    )


def verify_pseudonym_binding(
    *,
    pseudonym_id: str,
    permanent_identity: str,
    identity_key: bytes,
    scope: str,
    epoch: int,
    prefix: str = DEFAULT_PSEUDONYM_PREFIX,
) -> bool:
    """
    Verify that a pseudonym belongs to the supplied secret identity data.

    Comparison uses constant-time `compare_digest`.
    """

    validated_pseudonym = validate_pseudonym_id(
        pseudonym_id
    )

    expected = derive_pseudonymous_identity(
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope=scope,
        epoch=epoch,
        prefix=prefix,
    )

    return hmac.compare_digest(
        validated_pseudonym,
        expected.pseudonym_id,
    )


def require_valid_pseudonym_binding(
    *,
    pseudonym_id: str,
    permanent_identity: str,
    identity_key: bytes,
    scope: str,
    epoch: int,
    prefix: str = DEFAULT_PSEUDONYM_PREFIX,
) -> None:
    """
    Verify pseudonym ownership and raise an exception when it is invalid.
    """

    valid = verify_pseudonym_binding(
        pseudonym_id=pseudonym_id,
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope=scope,
        epoch=epoch,
        prefix=prefix,
    )

    if not valid:
        raise ProtocolValidationError(
            "Pseudonymous identity binding verification failed.",
            code="INVALID_PSEUDONYM_BINDING",
            details={
                "pseudonym_id": pseudonym_id,
                "scope": scope,
                "epoch": epoch,
            },
        )


def mask_permanent_identity(
    permanent_identity: str,
    *,
    visible_start: int = 2,
    visible_end: int = 2,
    mask_character: str = "*",
) -> str:
    """
    Mask a permanent identity for diagnostic display.

    Example:

        123456789012345
        12***********45
    """

    validated_identity = (
        normalize_permanent_identity(
            permanent_identity
        )
    )

    validated_start = validate_integer(
        visible_start,
        field_name="visible_start",
        minimum=0,
    )

    validated_end = validate_integer(
        visible_end,
        field_name="visible_end",
        minimum=0,
    )

    validated_mask = validate_non_empty_string(
        mask_character,
        field_name="mask_character",
        minimum_length=1,
        maximum_length=1,
    )

    if (
        validated_start
        + validated_end
        >= len(validated_identity)
    ):
        return validated_mask * len(
            validated_identity
        )

    hidden_length = (
        len(validated_identity)
        - validated_start
        - validated_end
    )

    start_text = validated_identity[
        :validated_start
    ]

    end_text = (
        validated_identity[
            -validated_end:
        ]
        if validated_end > 0
        else ""
    )

    return (
        start_text
        + validated_mask * hidden_length
        + end_text
    )


def pseudonym_fingerprint(
    pseudonym_id: str,
    *,
    fingerprint_bytes: int = 8,
) -> str:
    """
    Create a short SHA3-256 fingerprint for logging.
    """

    validated_pseudonym = validate_pseudonym_id(
        pseudonym_id
    )

    validated_length = validate_integer(
        fingerprint_bytes,
        field_name="fingerprint_bytes",
        minimum=4,
        maximum=32,
    )

    digest = hashlib.sha3_256(
        validated_pseudonym.encode("utf-8")
    ).digest()

    return digest[
        :validated_length
    ].hex()


def run_pseudonymous_identity_self_test() -> dict[str, Any]:
    """
    Run a deterministic pseudonymous identity self-test.

    The test confirms:

    - Same inputs create the same pseudonym
    - A changed epoch rotates the pseudonym
    - Correct binding verification succeeds
    - An incorrect identity fails verification
    - Permanent identity masking works
    """

    permanent_identity = "SUBSCRIBER-015001"

    identity_key = bytes(
        range(32)
    )

    first = derive_pseudonymous_identity(
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope="6g-authentication",
        epoch=1,
    )

    second = derive_pseudonymous_identity(
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope="6g-authentication",
        epoch=1,
    )

    rotated = rotate_pseudonymous_identity(
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope="6g-authentication",
        current_epoch=1,
    )

    deterministic_pass = hmac.compare_digest(
        first.pseudonym_id,
        second.pseudonym_id,
    )

    rotation_pass = not hmac.compare_digest(
        first.pseudonym_id,
        rotated.pseudonym_id,
    )

    binding_pass = verify_pseudonym_binding(
        pseudonym_id=first.pseudonym_id,
        permanent_identity=permanent_identity,
        identity_key=identity_key,
        scope="6g-authentication",
        epoch=1,
    )

    invalid_identity_rejected = (
        not verify_pseudonym_binding(
            pseudonym_id=first.pseudonym_id,
            permanent_identity="ATTACKER-IDENTITY",
            identity_key=identity_key,
            scope="6g-authentication",
            epoch=1,
        )
    )

    masked_identity = mask_permanent_identity(
        permanent_identity
    )

    identity_hidden = (
        permanent_identity
        not in masked_identity
    )

    success = all(
        (
            deterministic_pass,
            rotation_pass,
            binding_pass,
            invalid_identity_rejected,
            identity_hidden,
        )
    )

    return {
        "success": success,
        "pseudonym_id": first.pseudonym_id,
        "rotated_pseudonym_id": (
            rotated.pseudonym_id
        ),
        "pseudonym_fingerprint": (
            pseudonym_fingerprint(
                first.pseudonym_id
            )
        ),
        "deterministic_pass": (
            deterministic_pass
        ),
        "rotation_pass": rotation_pass,
        "binding_pass": binding_pass,
        "invalid_identity_rejected": (
            invalid_identity_rejected
        ),
        "masked_identity": masked_identity,
        "identity_hidden": identity_hidden,
    }


__all__ = [
    "DEFAULT_PSEUDONYM_PREFIX",
    "DEFAULT_PSEUDONYM_DIGEST_BYTES",
    "MINIMUM_IDENTITY_KEY_BYTES",
    "PseudonymousIdentity",
    "normalize_permanent_identity",
    "normalize_scope",
    "normalize_pseudonym_prefix",
    "generate_identity_key",
    "generate_random_pseudonym",
    "build_pseudonym_derivation_message",
    "derive_pseudonymous_identity",
    "rotate_pseudonymous_identity",
    "verify_pseudonym_binding",
    "require_valid_pseudonym_binding",
    "mask_permanent_identity",
    "pseudonym_fingerprint",
    "run_pseudonymous_identity_self_test",
]