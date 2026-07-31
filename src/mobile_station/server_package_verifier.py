"""
Server Package Verifier
FT-QuPAP Mobile Station

This module verifies the Authentication Server's signed ephemeral
ML-KEM public-key package before the Mobile Station performs
ML-KEM encapsulation.

Verification checks:

1. Required package fields
2. Expected server identity
3. ML-DSA algorithm
4. ML-KEM algorithm
5. Request-nonce binding
6. Service-context binding
7. Server timestamp freshness
8. ML-KEM public-key encoding and length
9. ML-DSA signature validity

Notebook-compatible result:

    (True, "credential_valid")

or one of the notebook-defined failure reasons.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from pqcrypto.kem import ml_kem_768
    from pqcrypto.sign import ml_dsa_65
except ImportError as error:
    raise ImportError(
        "The 'pqcrypto' package is required. Install it using: "
        "python -m pip install --upgrade pqcrypto"
    ) from error


ML_KEM_ALGORITHM = "ML-KEM-768"
ML_DSA_ALGORITHM = "ML-DSA-65"

DEFAULT_FRESHNESS_WINDOW_SECONDS = 60

REQUIRED_SERVER_FIELDS = frozenset(
    {
        "server_id",
        "timestamp",
        "request_nonce",
        "ml_kem_algorithm",
        "ml_kem_public_key",
        "ml_dsa_algorithm",
        "service_context",
        "signature",
    }
)

REQUIRED_TRUST_ANCHOR_FIELDS = frozenset(
    {
        "server_id",
        "algorithm",
        "public_key",
    }
)

REQUIRED_REQUEST_FIELDS = frozenset(
    {
        "nonce",
        "service_context",
    }
)


class ServerPackageVerificationError(Exception):
    """Base exception for server-package verification."""


@dataclass(frozen=True)
class ServerPackageVerificationResult:
    """
    Detailed server-package verification result.

    The ML-KEM public key is returned only after every mandatory
    verification step succeeds.
    """

    valid: bool
    reason: str
    server_id: str | None = None
    server_timestamp: int | None = None
    ml_kem_public_key: bytes | None = None

    def require_valid_public_key(self) -> bytes:
        """
        Return the verified ML-KEM public key.

        Raises:
            ServerPackageVerificationError:
                If verification failed.
        """

        if not self.valid:
            raise ServerPackageVerificationError(
                "Server package verification failed: "
                f"{self.reason}"
            )

        if self.ml_kem_public_key is None:
            raise ServerPackageVerificationError(
                "Verified result contains no ML-KEM public key."
            )

        return self.ml_kem_public_key

    def public_summary(self) -> dict[str, Any]:
        """Return a non-secret result summary."""

        return {
            "valid": self.valid,
            "reason": self.reason,
            "server_id": self.server_id,
            "server_timestamp": self.server_timestamp,
            "ml_dsa_algorithm": (
                ML_DSA_ALGORITHM
                if self.valid
                else None
            ),
            "ml_kem_algorithm": (
                ML_KEM_ALGORITHM
                if self.valid
                else None
            ),
            "ml_kem_public_key_bytes": (
                len(self.ml_kem_public_key)
                if self.ml_kem_public_key is not None
                else None
            ),
        }


def current_timestamp() -> int:
    """Return the current Unix timestamp."""

    return int(time.time())


def canonical_json_bytes(
    value: Mapping[str, Any],
) -> bytes:
    """
    Serialize data exactly as used by the notebook.

    The Authentication Server signs this canonical representation,
    excluding the signature field.
    """

    if not isinstance(value, Mapping):
        raise TypeError(
            "value must be a mapping."
        )

    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def encode_base64(data: bytes) -> str:
    """Encode non-empty bytes as Base64 text."""

    if not isinstance(data, bytes):
        raise TypeError(
            "data must be bytes."
        )

    if not data:
        raise ValueError(
            "data cannot be empty."
        )

    return base64.b64encode(
        data
    ).decode("ascii")


def decode_base64(
    encoded_value: str,
) -> bytes:
    """Decode strict Base64 text."""

    if not isinstance(encoded_value, str):
        raise TypeError(
            "encoded_value must be a string."
        )

    if not encoded_value:
        raise ValueError(
            "encoded_value cannot be empty."
        )

    return base64.b64decode(
        encoded_value.encode("ascii"),
        validate=True,
    )


def unsigned_server_information(
    server_info: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Remove the signature field from the server package.

    The remaining fields form the exact ML-DSA-signed message.
    """

    if not isinstance(server_info, Mapping):
        raise TypeError(
            "server_info must be a mapping."
        )

    return {
        key: value
        for key, value in server_info.items()
        if key != "signature"
    }


def _failure(
    reason: str,
    server_id: str | None = None,
    server_timestamp: int | None = None,
) -> ServerPackageVerificationResult:
    """Construct a failed verification result."""

    return ServerPackageVerificationResult(
        valid=False,
        reason=reason,
        server_id=server_id,
        server_timestamp=server_timestamp,
        ml_kem_public_key=None,
    )


def verify_server_credential_detailed(
    server_info: Mapping[str, Any],
    trust_anchor: Mapping[str, Any],
    request: Mapping[str, Any],
    now: int | None = None,
    freshness_window_seconds: int = (
        DEFAULT_FRESHNESS_WINDOW_SECONDS
    ),
) -> ServerPackageVerificationResult:
    """
    Verify the Authentication Server credential package.

    Args:
        server_info:
            Signed Authentication Server information package.

        trust_anchor:
            Operator-installed Mobile Station trust anchor:

                {
                    "server_id": "AS-6G-001",
                    "algorithm": "ML-DSA-65",
                    "public_key": <bytes>
                }

        request:
            Original Mobile Station authentication request.

        now:
            Optional verification timestamp for deterministic testing.

        freshness_window_seconds:
            Maximum accepted server-package age.

    Returns:
        Detailed verification result.
    """

    # ---------------------------------------------------------
    # Basic mapping checks
    # ---------------------------------------------------------
    if not isinstance(server_info, Mapping):
        return _failure(
            "credential_not_mapping"
        )

    if not isinstance(trust_anchor, Mapping):
        return _failure(
            "invalid_trust_anchor"
        )

    if not isinstance(request, Mapping):
        return _failure(
            "invalid_authentication_request"
        )

    if not REQUIRED_TRUST_ANCHOR_FIELDS.issubset(
        trust_anchor.keys()
    ):
        return _failure(
            "invalid_trust_anchor"
        )

    if not REQUIRED_REQUEST_FIELDS.issubset(
        request.keys()
    ):
        return _failure(
            "invalid_authentication_request"
        )

    # ---------------------------------------------------------
    # Notebook check 1: required server-package fields
    # ---------------------------------------------------------
    if not REQUIRED_SERVER_FIELDS.issubset(
        server_info.keys()
    ):
        return _failure(
            "credential_missing_required_field"
        )

    received_server_id = server_info.get(
        "server_id"
    )

    safe_server_id = (
        received_server_id
        if isinstance(received_server_id, str)
        else None
    )

    # ---------------------------------------------------------
    # Notebook check 2: server identity
    # ---------------------------------------------------------
    if (
        server_info["server_id"]
        != trust_anchor["server_id"]
    ):
        return _failure(
            "server_id_mismatch",
            server_id=safe_server_id,
        )

    # ---------------------------------------------------------
    # Notebook check 3: ML-DSA algorithm
    # ---------------------------------------------------------
    if (
        server_info["ml_dsa_algorithm"]
        != trust_anchor["algorithm"]
    ):
        return _failure(
            "ml_dsa_algorithm_mismatch",
            server_id=safe_server_id,
        )

    if (
        trust_anchor["algorithm"]
        != ML_DSA_ALGORITHM
    ):
        return _failure(
            "ml_dsa_algorithm_mismatch",
            server_id=safe_server_id,
        )

    # ---------------------------------------------------------
    # Notebook check 4: ML-KEM algorithm
    # ---------------------------------------------------------
    if (
        server_info["ml_kem_algorithm"]
        != ML_KEM_ALGORITHM
    ):
        return _failure(
            "ml_kem_algorithm_mismatch",
            server_id=safe_server_id,
        )

    # ---------------------------------------------------------
    # Notebook check 5: request nonce binding
    # ---------------------------------------------------------
    if (
        server_info["request_nonce"]
        != request["nonce"]
    ):
        return _failure(
            "credential_nonce_mismatch",
            server_id=safe_server_id,
        )

    # ---------------------------------------------------------
    # Notebook check 6: service-context binding
    # ---------------------------------------------------------
    if (
        server_info["service_context"]
        != request["service_context"]
    ):
        return _failure(
            "credential_context_mismatch",
            server_id=safe_server_id,
        )

    # ---------------------------------------------------------
    # Notebook check 7: server timestamp
    # ---------------------------------------------------------
    try:
        server_timestamp = int(
            server_info["timestamp"]
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return _failure(
            "invalid_server_timestamp",
            server_id=safe_server_id,
        )

    try:
        verification_time = (
            current_timestamp()
            if now is None
            else int(now)
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return _failure(
            "invalid_verification_timestamp",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    if isinstance(
        freshness_window_seconds,
        bool,
    ) or not isinstance(
        freshness_window_seconds,
        int,
    ):
        return _failure(
            "invalid_freshness_window",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    if freshness_window_seconds <= 0:
        return _failure(
            "invalid_freshness_window",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    if abs(
        verification_time
        - server_timestamp
    ) > freshness_window_seconds:
        return _failure(
            "stale_server_credential",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    # ---------------------------------------------------------
    # Notebook check 8: ML-KEM public-key encoding
    # ---------------------------------------------------------
    try:
        kem_public_key = decode_base64(
            server_info[
                "ml_kem_public_key"
            ]
        )

    except Exception:
        return _failure(
            "invalid_ml_kem_public_key_encoding",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    if len(kem_public_key) != (
        ml_kem_768.PUBLIC_KEY_SIZE
    ):
        return _failure(
            "invalid_ml_kem_public_key_length",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    # ---------------------------------------------------------
    # Validate trust-anchor public key
    # ---------------------------------------------------------
    trust_public_key = trust_anchor.get(
        "public_key"
    )

    if not isinstance(
        trust_public_key,
        (bytes, bytearray),
    ):
        return _failure(
            "invalid_trust_anchor_public_key",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    if len(trust_public_key) != (
        ml_dsa_65.PUBLIC_KEY_SIZE
    ):
        return _failure(
            "invalid_trust_anchor_public_key",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    # ---------------------------------------------------------
    # Notebook check 9: ML-DSA signature
    # ---------------------------------------------------------
    try:
        signature = decode_base64(
            server_info["signature"]
        )

        unsigned = unsigned_server_information(
            server_info
        )

        verified = ml_dsa_65.verify(
            bytes(trust_public_key),
            canonical_json_bytes(
                unsigned
            ),
            signature,
        )

    except Exception:
        verified = False

    if not verified:
        return _failure(
            "invalid_server_credential",
            server_id=safe_server_id,
            server_timestamp=server_timestamp,
        )

    return ServerPackageVerificationResult(
        valid=True,
        reason="credential_valid",
        server_id=safe_server_id,
        server_timestamp=server_timestamp,
        ml_kem_public_key=kem_public_key,
    )


def verify_server_credential(
    server_info: Mapping[str, Any],
    trust_anchor: Mapping[str, Any],
    request: Mapping[str, Any],
    now: int | None = None,
    freshness_window_seconds: int = (
        DEFAULT_FRESHNESS_WINDOW_SECONDS
    ),
) -> tuple[bool, str]:
    """
    Notebook-compatible verification function.

    Returns:
        Tuple containing:

            credential_ok
            credential_reason
    """

    result = verify_server_credential_detailed(
        server_info=server_info,
        trust_anchor=trust_anchor,
        request=request,
        now=now,
        freshness_window_seconds=(
            freshness_window_seconds
        ),
    )

    return result.valid, result.reason


def get_verified_mlkem_public_key(
    server_info: Mapping[str, Any],
    trust_anchor: Mapping[str, Any],
    request: Mapping[str, Any],
    now: int | None = None,
    freshness_window_seconds: int = (
        DEFAULT_FRESHNESS_WINDOW_SECONDS
    ),
) -> bytes:
    """
    Verify the package and return its ML-KEM public key.

    This function should be called immediately before ML-KEM
    encapsulation.
    """

    result = verify_server_credential_detailed(
        server_info=server_info,
        trust_anchor=trust_anchor,
        request=request,
        now=now,
        freshness_window_seconds=(
            freshness_window_seconds
        ),
    )

    return result.require_valid_public_key()


def build_test_server_package(
    request: Mapping[str, Any],
    server_id: str,
    kem_public_key: bytes,
    signing_secret_key: bytes,
    timestamp: int,
) -> dict[str, Any]:
    """
    Build a signed server package for local self-testing.

    Real package creation belongs to the Authentication Server.
    """

    unsigned = {
        "server_id": server_id,
        "timestamp": timestamp,
        "request_nonce": request["nonce"],
        "ml_kem_algorithm":
            ML_KEM_ALGORITHM,
        "ml_kem_public_key":
            encode_base64(
                kem_public_key
            ),
        "ml_dsa_algorithm":
            ML_DSA_ALGORITHM,
        "service_context":
            request["service_context"],
    }

    signature = ml_dsa_65.sign(
        signing_secret_key,
        canonical_json_bytes(
            unsigned
        ),
    )

    return {
        **unsigned,
        "signature":
            encode_base64(signature),
    }


def run_self_test() -> None:
    """
    Test valid, tampered, stale, nonce-mismatched, and
    context-mismatched server packages.
    """

    print("=" * 70)
    print("FT-QuPAP Server Package Verifier Self-Test")
    print("=" * 70)

    signing_public_key, signing_secret_key = (
        ml_dsa_65.generate_keypair()
    )

    kem_public_key, _kem_secret_key = (
        ml_kem_768.generate_keypair()
    )

    test_time = current_timestamp()

    request = {
        "pseudonym_id":
            "PID-6G-UE-0001",
        "timestamp":
            test_time,
        "nonce":
            encode_base64(
                b"0123456789ABCDEF"
            ),
        "service_context":
            "urban",
        "request_type":
            "FT-QuPAP-Authentication",
    }

    trust_anchor = {
        "server_id":
            "AS-6G-001",
        "algorithm":
            ML_DSA_ALGORITHM,
        "public_key":
            signing_public_key,
        "trust_anchor_version":
            1,
    }

    server_info = build_test_server_package(
        request=request,
        server_id=
            trust_anchor["server_id"],
        kem_public_key=
            kem_public_key,
        signing_secret_key=
            signing_secret_key,
        timestamp=test_time,
    )

    valid_result = (
        verify_server_credential_detailed(
            server_info=server_info,
            trust_anchor=trust_anchor,
            request=request,
            now=test_time,
        )
    )

    tampered_package = dict(
        server_info
    )

    tampered_signature = bytearray(
        decode_base64(
            tampered_package["signature"]
        )
    )

    tampered_signature[0] ^= 0x01

    tampered_package["signature"] = (
        encode_base64(
            bytes(tampered_signature)
        )
    )

    tampered_result = (
        verify_server_credential_detailed(
            server_info=tampered_package,
            trust_anchor=trust_anchor,
            request=request,
            now=test_time,
        )
    )

    mismatched_request = dict(
        request
    )

    mismatched_request["nonce"] = (
        encode_base64(
            b"FEDCBA9876543210"
        )
    )

    nonce_result = (
        verify_server_credential_detailed(
            server_info=server_info,
            trust_anchor=trust_anchor,
            request=mismatched_request,
            now=test_time,
        )
    )

    context_request = dict(
        request
    )

    context_request[
        "service_context"
    ] = "rural"

    context_result = (
        verify_server_credential_detailed(
            server_info=server_info,
            trust_anchor=trust_anchor,
            request=context_request,
            now=test_time,
        )
    )

    stale_result = (
        verify_server_credential_detailed(
            server_info=server_info,
            trust_anchor=trust_anchor,
            request=request,
            now=(
                test_time
                + DEFAULT_FRESHNESS_WINDOW_SECONDS
                + 1
            ),
        )
    )

    print(
        f"ML-DSA algorithm          : "
        f"{ML_DSA_ALGORITHM}"
    )
    print(
        f"ML-KEM algorithm          : "
        f"{ML_KEM_ALGORITHM}"
    )
    print(
        f"ML-DSA public-key bytes   : "
        f"{len(signing_public_key)}"
    )
    print(
        f"ML-KEM public-key bytes   : "
        f"{len(kem_public_key)}"
    )
    print(
        f"Valid package             : "
        f"{valid_result.valid}, "
        f"{valid_result.reason}"
    )
    print(
        f"Tampered package          : "
        f"{tampered_result.valid}, "
        f"{tampered_result.reason}"
    )
    print(
        f"Nonce mismatch            : "
        f"{nonce_result.valid}, "
        f"{nonce_result.reason}"
    )
    print(
        f"Context mismatch          : "
        f"{context_result.valid}, "
        f"{context_result.reason}"
    )
    print(
        f"Stale package             : "
        f"{stale_result.valid}, "
        f"{stale_result.reason}"
    )

    if not valid_result.valid:
        raise ServerPackageVerificationError(
            "Valid server package was rejected."
        )

    if (
        valid_result.require_valid_public_key()
        != kem_public_key
    ):
        raise ServerPackageVerificationError(
            "Verified ML-KEM public key does not match."
        )

    if tampered_result.reason != (
        "invalid_server_credential"
    ):
        raise ServerPackageVerificationError(
            "Tampered credential test failed."
        )

    if nonce_result.reason != (
        "credential_nonce_mismatch"
    ):
        raise ServerPackageVerificationError(
            "Nonce-binding test failed."
        )

    if context_result.reason != (
        "credential_context_mismatch"
    ):
        raise ServerPackageVerificationError(
            "Context-binding test failed."
        )

    if stale_result.reason != (
        "stale_server_credential"
    ):
        raise ServerPackageVerificationError(
            "Credential freshness test failed."
        )

    print(
        "\nServer package verifier self-test "
        "completed successfully."
    )


__all__ = [
    "ML_KEM_ALGORITHM",
    "ML_DSA_ALGORITHM",
    "ServerPackageVerificationResult",
    "ServerPackageVerificationError",
    "verify_server_credential",
    "verify_server_credential_detailed",
    "get_verified_mlkem_public_key",
    "unsigned_server_information",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ServerPackageVerificationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[SERVER PACKAGE VERIFICATION ERROR] "
            f"{error}"
        )