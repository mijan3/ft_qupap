"""
Session Key Derivation Module
FT-QuPAP Mobile Station

This module implements FT-QuPAP notebook Cells 33 and 34.

The ML-KEM shared secret K_ss is bound to the complete protocol
transcript using HKDF-SHA3-256.

Derivation:

    key_material = HKDF-SHA3-256(
        input_key_material = K_ss,
        salt = H(Transcript),
        info = b"FT-QuPAP/session-key-separation/v1",
        length = 64 bytes,
    )

Key separation:

    K_auth = key_material[0:32]
    K_ctrl = key_material[32:64]

K_auth:
    Used only for the transcript-bound KMAC256 authentication tag.

K_ctrl:
    Used only for AES-GCM protection of the control/check schedule.

Neither key should be transmitted, logged, exported, or reused in
another session.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


HKDF_INFO = b"FT-QuPAP/session-key-separation/v1"

MLKEM_SHARED_SECRET_LENGTH = 32
TRANSCRIPT_HASH_LENGTH = 32

SESSION_KEY_MATERIAL_LENGTH = 64
AUTHENTICATION_KEY_LENGTH = 32
CONTROL_KEY_LENGTH = 32


class SessionKeyDerivationError(Exception):
    """Raised when FT-QuPAP session-key derivation fails."""


@dataclass(frozen=True)
class SessionKeys:
    """
    Purpose-separated FT-QuPAP session keys.

    Attributes:
        authentication_key:
            K_auth, used exclusively for KMAC256.

        control_key:
            K_ctrl, used exclusively for control-schedule protection.
    """

    authentication_key: bytes
    control_key: bytes

    def __post_init__(self) -> None:
        validate_authentication_key(
            self.authentication_key
        )

        validate_control_key(
            self.control_key
        )

        if hmac.compare_digest(
            self.authentication_key,
            self.control_key,
        ):
            raise SessionKeyDerivationError(
                "K_auth and K_ctrl must be different."
            )

    @property
    def k_auth(self) -> bytes:
        """Return the authentication key."""

        return self.authentication_key

    @property
    def k_ctrl(self) -> bytes:
        """Return the control key."""

        return self.control_key

    def as_tuple(self) -> tuple[bytes, bytes]:
        """
        Return keys in notebook-compatible order.

        Returns:
            (K_auth, K_ctrl)
        """

        return (
            self.authentication_key,
            self.control_key,
        )

    def safe_summary(self) -> dict[str, Any]:
        """
        Return only non-secret diagnostic information.

        The real keys are never included.
        """

        return {
            "authentication_key_length":
                len(self.authentication_key),
            "control_key_length":
                len(self.control_key),
            "authentication_key_fingerprint":
                key_fingerprint(
                    self.authentication_key
                ),
            "control_key_fingerprint":
                key_fingerprint(
                    self.control_key
                ),
            "keys_are_separated":
                not hmac.compare_digest(
                    self.authentication_key,
                    self.control_key,
                ),
        }


def validate_shared_secret(
    shared_secret: bytes,
) -> None:
    """
    Validate the ML-KEM shared secret.

    ML-KEM-768 produces a 32-byte shared secret in the notebook.
    """

    if not isinstance(shared_secret, bytes):
        raise TypeError(
            "shared_secret must be bytes."
        )

    if len(shared_secret) != (
        MLKEM_SHARED_SECRET_LENGTH
    ):
        raise ValueError(
            "ML-KEM-768 shared_secret must contain "
            f"exactly {MLKEM_SHARED_SECRET_LENGTH} bytes. "
            f"Received {len(shared_secret)} bytes."
        )


def validate_transcript_hash(
    transcript_hash: bytes,
) -> None:
    """
    Validate H(Transcript).

    FT-QuPAP uses SHA3-256, so the transcript hash must contain
    exactly 32 bytes.
    """

    if not isinstance(transcript_hash, bytes):
        raise TypeError(
            "transcript_hash must be bytes."
        )

    if len(transcript_hash) != (
        TRANSCRIPT_HASH_LENGTH
    ):
        raise ValueError(
            "SHA3-256 transcript_hash must contain "
            f"exactly {TRANSCRIPT_HASH_LENGTH} bytes. "
            f"Received {len(transcript_hash)} bytes."
        )


def validate_key_material(
    key_material: bytes,
) -> None:
    """Validate the 64-byte HKDF output."""

    if not isinstance(key_material, bytes):
        raise TypeError(
            "key_material must be bytes."
        )

    if len(key_material) < (
        SESSION_KEY_MATERIAL_LENGTH
    ):
        raise ValueError(
            "Expected at least "
            f"{SESSION_KEY_MATERIAL_LENGTH} bytes "
            "of key material."
        )


def validate_authentication_key(
    authentication_key: bytes,
) -> None:
    """Validate K_auth."""

    if not isinstance(
        authentication_key,
        bytes,
    ):
        raise TypeError(
            "authentication_key must be bytes."
        )

    if len(authentication_key) != (
        AUTHENTICATION_KEY_LENGTH
    ):
        raise ValueError(
            "K_auth must contain exactly "
            f"{AUTHENTICATION_KEY_LENGTH} bytes."
        )


def validate_control_key(
    control_key: bytes,
) -> None:
    """Validate K_ctrl."""

    if not isinstance(control_key, bytes):
        raise TypeError(
            "control_key must be bytes."
        )

    if len(control_key) != (
        CONTROL_KEY_LENGTH
    ):
        raise ValueError(
            "K_ctrl must contain exactly "
            f"{CONTROL_KEY_LENGTH} bytes."
        )


def derive_session_key_material(
    shared_secret: bytes,
    transcript_hash: bytes,
) -> bytes:
    """
    Derive 64 bytes of transcript-bound session-key material.

    This function matches the FT-QuPAP notebook implementation.

    Args:
        shared_secret:
            The 32-byte ML-KEM shared secret K_ss.

        transcript_hash:
            The 32-byte SHA3-256 hash of the request/server
            transcript.

    Returns:
        Exactly 64 bytes of derived key material.
    """

    validate_shared_secret(
        shared_secret
    )

    validate_transcript_hash(
        transcript_hash
    )

    try:
        key_material = HKDF(
            algorithm=hashes.SHA3_256(),
            length=SESSION_KEY_MATERIAL_LENGTH,
            salt=transcript_hash,
            info=HKDF_INFO,
        ).derive(shared_secret)

    except Exception as error:
        raise SessionKeyDerivationError(
            "HKDF-SHA3-256 session-key derivation failed."
        ) from error

    if len(key_material) != (
        SESSION_KEY_MATERIAL_LENGTH
    ):
        raise SessionKeyDerivationError(
            "HKDF returned an unexpected key-material length."
        )

    return key_material


def split_session_keys(
    key_material: bytes,
) -> tuple[bytes, bytes]:
    """
    Split HKDF output into K_auth and K_ctrl.

    Notebook-compatible output:

        K_auth = key_material[:32]
        K_ctrl = key_material[32:64]

    Args:
        key_material:
            At least 64 bytes of HKDF output.

    Returns:
        Tuple containing:

            (K_auth, K_ctrl)
    """

    validate_key_material(
        key_material
    )

    k_auth = bytes(
        key_material[
            :AUTHENTICATION_KEY_LENGTH
        ]
    )

    k_ctrl = bytes(
        key_material[
            AUTHENTICATION_KEY_LENGTH:
            SESSION_KEY_MATERIAL_LENGTH
        ]
    )

    validate_authentication_key(
        k_auth
    )

    validate_control_key(
        k_ctrl
    )

    if hmac.compare_digest(
        k_auth,
        k_ctrl,
    ):
        raise SessionKeyDerivationError(
            "Key separation failed because "
            "K_auth equals K_ctrl."
        )

    return k_auth, k_ctrl


def derive_session_keys(
    shared_secret: bytes,
    transcript_hash: bytes,
) -> SessionKeys:
    """
    Derive and return both FT-QuPAP session keys.

    This is a convenience wrapper around:

        derive_session_key_material()
        split_session_keys()
    """

    key_material = (
        derive_session_key_material(
            shared_secret=shared_secret,
            transcript_hash=transcript_hash,
        )
    )

    k_auth, k_ctrl = split_session_keys(
        key_material
    )

    return SessionKeys(
        authentication_key=k_auth,
        control_key=k_ctrl,
    )


def verify_same_session_keys(
    first: SessionKeys,
    second: SessionKeys,
) -> bool:
    """
    Verify that the MS and AS derived identical session keys.

    Constant-time comparisons are used.
    """

    if not isinstance(first, SessionKeys):
        raise TypeError(
            "first must be a SessionKeys object."
        )

    if not isinstance(second, SessionKeys):
        raise TypeError(
            "second must be a SessionKeys object."
        )

    authentication_keys_match = (
        hmac.compare_digest(
            first.authentication_key,
            second.authentication_key,
        )
    )

    control_keys_match = (
        hmac.compare_digest(
            first.control_key,
            second.control_key,
        )
    )

    return (
        authentication_keys_match
        and control_keys_match
    )


def key_fingerprint(
    key: bytes,
    length: int = 16,
) -> str:
    """
    Return a short SHA3-256 fingerprint for testing.

    The key itself is not returned.
    """

    if not isinstance(key, bytes):
        raise TypeError(
            "key must be bytes."
        )

    if not key:
        raise ValueError(
            "key cannot be empty."
        )

    if not isinstance(length, int):
        raise TypeError(
            "length must be an integer."
        )

    if not 1 <= length <= 64:
        raise ValueError(
            "length must be between 1 and 64."
        )

    return hashlib.sha3_256(
        key
    ).hexdigest()[:length]


def run_self_test() -> None:
    """
    Test deterministic MS/AS session-key derivation.
    """

    print("=" * 70)
    print("FT-QuPAP Session-Key Derivation Self-Test")
    print("=" * 70)

    # Represents the shared secret independently recovered by
    # ML-KEM encapsulation and decapsulation.
    shared_secret_ms = hashlib.sha3_256(
        b"FT-QuPAP self-test shared secret"
    ).digest()

    shared_secret_as = bytes(
        shared_secret_ms
    )

    transcript_hash = hashlib.sha3_256(
        b"FT-QuPAP self-test transcript"
    ).digest()

    mobile_keys = derive_session_keys(
        shared_secret=shared_secret_ms,
        transcript_hash=transcript_hash,
    )

    server_keys = derive_session_keys(
        shared_secret=shared_secret_as,
        transcript_hash=transcript_hash,
    )

    keys_match = verify_same_session_keys(
        mobile_keys,
        server_keys,
    )

    key_material = (
        derive_session_key_material(
            shared_secret_ms,
            transcript_hash,
        )
    )

    k_auth, k_ctrl = split_session_keys(
        key_material
    )

    deterministic_repeat = (
        hmac.compare_digest(
            k_auth,
            mobile_keys.authentication_key,
        )
        and hmac.compare_digest(
            k_ctrl,
            mobile_keys.control_key,
        )
    )

    changed_transcript_hash = (
        hashlib.sha3_256(
            b"Different FT-QuPAP transcript"
        ).digest()
    )

    changed_session_keys = (
        derive_session_keys(
            shared_secret=shared_secret_ms,
            transcript_hash=
                changed_transcript_hash,
        )
    )

    transcript_binding_valid = not (
        hmac.compare_digest(
            mobile_keys.authentication_key,
            changed_session_keys.authentication_key,
        )
        or hmac.compare_digest(
            mobile_keys.control_key,
            changed_session_keys.control_key,
        )
    )

    print(
        f"Shared-secret bytes       : "
        f"{len(shared_secret_ms)}"
    )
    print(
        f"Transcript-hash bytes     : "
        f"{len(transcript_hash)}"
    )
    print(
        f"Derived material bytes    : "
        f"{len(key_material)}"
    )
    print(
        f"K_auth bytes              : "
        f"{len(k_auth)}"
    )
    print(
        f"K_ctrl bytes              : "
        f"{len(k_ctrl)}"
    )
    print(
        f"K_auth fingerprint        : "
        f"{key_fingerprint(k_auth)}"
    )
    print(
        f"K_ctrl fingerprint        : "
        f"{key_fingerprint(k_ctrl)}"
    )
    print(
        f"MS/AS keys match          : "
        f"{keys_match}"
    )
    print(
        f"Deterministic derivation  : "
        f"{deterministic_repeat}"
    )
    print(
        f"Transcript binding valid  : "
        f"{transcript_binding_valid}"
    )
    print(
        f"K_auth differs from K_ctrl: "
        f"{not hmac.compare_digest(k_auth, k_ctrl)}"
    )

    if not keys_match:
        raise SessionKeyDerivationError(
            "Mobile Station and Authentication Server "
            "derived different keys."
        )

    if not deterministic_repeat:
        raise SessionKeyDerivationError(
            "Repeated derivation was not deterministic."
        )

    if not transcript_binding_valid:
        raise SessionKeyDerivationError(
            "Changing the transcript did not change "
            "the derived keys."
        )

    if hmac.compare_digest(
        k_auth,
        k_ctrl,
    ):
        raise SessionKeyDerivationError(
            "K_auth and K_ctrl are not separated."
        )

    print(
        "\nSession-key derivation self-test "
        "completed successfully."
    )


__all__ = [
    "SessionKeys",
    "SessionKeyDerivationError",
    "derive_session_key_material",
    "split_session_keys",
    "derive_session_keys",
    "verify_same_session_keys",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        SessionKeyDerivationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[SESSION KEY DERIVATION ERROR] "
            f"{error}"
        )