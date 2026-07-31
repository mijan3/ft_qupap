"""
ML-KEM Encapsulation Module
FT-QuPAP Mobile Station

This module implements the Mobile Station side of ML-KEM-768
encapsulation.

Protocol order:

1. The Mobile Station receives the Authentication Server package.
2. server_package_verifier.py verifies its ML-DSA signature.
3. The verified ephemeral ML-KEM public key is passed here.
4. ML-KEM encapsulation produces:

       ciphertext, shared_secret

5. The ciphertext is sent to the Authentication Server.
6. The shared secret remains private inside the Mobile Station.
7. session_key_derivation.py derives K_auth and K_ctrl from the
   shared secret and H(Transcript).

Important:
    This module does not authenticate the server public key.
    Server-package verification must succeed before encapsulation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from pqcrypto.kem import ml_kem_768
except ImportError as error:
    raise ImportError(
        "The 'pqcrypto' package is required. Install it using: "
        "python -m pip install pqcrypto"
    ) from error


ML_KEM_ALGORITHM = "ML-KEM-768"

PUBLIC_KEY_SIZE = ml_kem_768.PUBLIC_KEY_SIZE
SECRET_KEY_SIZE = ml_kem_768.SECRET_KEY_SIZE
CIPHERTEXT_SIZE = ml_kem_768.CIPHERTEXT_SIZE

# The pqcrypto package names the ML-KEM shared-secret size
# PLAINTEXT_SIZE.
SHARED_SECRET_SIZE = ml_kem_768.PLAINTEXT_SIZE


class MLKEMEncapsulationError(Exception):
    """Raised when Mobile Station ML-KEM encapsulation fails."""


class MLKEMPublicKeyError(MLKEMEncapsulationError):
    """Raised when the server ML-KEM public key is invalid."""


class MLKEMCiphertextError(MLKEMEncapsulationError):
    """Raised when the generated ML-KEM ciphertext is invalid."""


@dataclass(frozen=True)
class MLKEMEncapsulationResult:
    """
    Result produced by ML-KEM-768 encapsulation.

    Attributes:
        algorithm:
            ML-KEM parameter set used by the session.

        ciphertext:
            Public ciphertext transmitted to the Authentication Server.

        shared_secret:
            Private Mobile Station shared secret.

        encapsulation_time_ms:
            Encapsulation runtime measured for the simulation.
    """

    ciphertext: bytes
    shared_secret: bytes
    algorithm: str = ML_KEM_ALGORITHM
    encapsulation_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.algorithm != ML_KEM_ALGORITHM:
            raise ValueError(
                f"algorithm must be {ML_KEM_ALGORITHM!r}."
            )

        validate_mlkem_ciphertext_or_raise(
            self.ciphertext
        )

        validate_shared_secret(
            self.shared_secret
        )

        if not isinstance(
            self.encapsulation_time_ms,
            (int, float),
        ):
            raise TypeError(
                "encapsulation_time_ms must be numeric."
            )

        if self.encapsulation_time_ms < 0:
            raise ValueError(
                "encapsulation_time_ms cannot be negative."
            )

    @property
    def ciphertext_base64(self) -> str:
        """Return the ciphertext in Base64 transport format."""

        return encode_base64(
            self.ciphertext
        )

    @property
    def ciphertext_fingerprint(self) -> str:
        """Return a short ciphertext fingerprint."""

        return hashlib.sha3_256(
            self.ciphertext
        ).hexdigest()[:16]

    @property
    def shared_secret_fingerprint(self) -> str:
        """
        Return a short testing fingerprint.

        The shared secret itself is never displayed.
        """

        return hashlib.sha3_256(
            self.shared_secret
        ).hexdigest()[:16]

    def to_transport_dictionary(self) -> dict[str, Any]:
        """
        Return the public ML-KEM transport fields.

        The shared secret is deliberately excluded.
        """

        return {
            "kem_algorithm": self.algorithm,
            "kem_ciphertext":
                self.ciphertext_base64,
        }

    def safe_summary(self) -> dict[str, Any]:
        """Return non-secret encapsulation information."""

        return {
            "algorithm": self.algorithm,
            "public_key_size":
                PUBLIC_KEY_SIZE,
            "ciphertext_size":
                len(self.ciphertext),
            "shared_secret_size":
                len(self.shared_secret),
            "ciphertext_fingerprint":
                self.ciphertext_fingerprint,
            "shared_secret_fingerprint":
                self.shared_secret_fingerprint,
            "encapsulation_time_ms":
                self.encapsulation_time_ms,
        }


def encode_base64(data: bytes) -> str:
    """Encode non-empty bytes as Base64 ASCII text."""

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
    field_name: str = "encoded_value",
) -> bytes:
    """Decode strict Base64 ASCII text."""

    if not isinstance(encoded_value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    if not encoded_value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    try:
        return base64.b64decode(
            encoded_value.encode("ascii"),
            validate=True,
        )

    except Exception as error:
        raise ValueError(
            f"{field_name} is not valid Base64."
        ) from error


def validate_server_public_key(
    server_public_key: bytes,
) -> None:
    """
    Validate the verified ephemeral ML-KEM-768 public key.
    """

    if not isinstance(
        server_public_key,
        bytes,
    ):
        raise TypeError(
            "server_public_key must be bytes."
        )

    if len(server_public_key) != PUBLIC_KEY_SIZE:
        raise MLKEMPublicKeyError(
            "Invalid ML-KEM-768 public-key length. "
            f"Expected {PUBLIC_KEY_SIZE} bytes, "
            f"received {len(server_public_key)}."
        )


def validate_shared_secret(
    shared_secret: bytes,
) -> None:
    """Validate the ML-KEM shared secret."""

    if not isinstance(shared_secret, bytes):
        raise TypeError(
            "shared_secret must be bytes."
        )

    if len(shared_secret) != SHARED_SECRET_SIZE:
        raise MLKEMEncapsulationError(
            "Invalid ML-KEM shared-secret length. "
            f"Expected {SHARED_SECRET_SIZE} bytes, "
            f"received {len(shared_secret)}."
        )


def validate_mlkem_ciphertext(
    ciphertext: bytes,
) -> tuple[bool, str]:
    """
    Notebook-compatible ML-KEM ciphertext validation.

    Returns:
        (True, "ciphertext_format_valid")

    or:

        (False, "ciphertext_not_bytes")
        (False, "malformed_ciphertext_length")
    """

    if not isinstance(
        ciphertext,
        (bytes, bytearray),
    ):
        return False, "ciphertext_not_bytes"

    if len(ciphertext) != CIPHERTEXT_SIZE:
        return False, "malformed_ciphertext_length"

    return True, "ciphertext_format_valid"


def validate_mlkem_ciphertext_or_raise(
    ciphertext: bytes,
) -> None:
    """Validate a ciphertext and raise on failure."""

    valid, reason = validate_mlkem_ciphertext(
        ciphertext
    )

    if not valid:
        raise MLKEMCiphertextError(
            reason
        )


def encapsulate_session_secret(
    server_public_key: bytes,
) -> MLKEMEncapsulationResult:
    """
    Perform Mobile Station ML-KEM-768 encapsulation.

    Args:
        server_public_key:
            The Authentication Server's already verified,
            ephemeral ML-KEM-768 public key.

    Returns:
        MLKEMEncapsulationResult containing:

            ciphertext
            shared_secret
            encapsulation runtime
    """

    validate_server_public_key(
        server_public_key
    )

    started_at = time.perf_counter()

    try:
        ciphertext, shared_secret = (
            ml_kem_768.encrypt(
                server_public_key
            )
        )

    except Exception as error:
        raise MLKEMEncapsulationError(
            "ML-KEM-768 encapsulation failed."
        ) from error

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000.0

    ciphertext = bytes(ciphertext)
    shared_secret = bytes(shared_secret)

    validate_mlkem_ciphertext_or_raise(
        ciphertext
    )

    validate_shared_secret(
        shared_secret
    )

    return MLKEMEncapsulationResult(
        ciphertext=ciphertext,
        shared_secret=shared_secret,
        algorithm=ML_KEM_ALGORITHM,
        encapsulation_time_ms=elapsed_ms,
    )


def mlkem_encapsulate(
    server_info: Mapping[str, Any],
) -> tuple[bytes, bytes]:
    """
    Notebook-compatible encapsulation function.

    Expected signed server-information field:

        server_info["ml_kem_public_key"]

    Security note:
        server_package_verifier.py must verify the package before
        this function is called.

    Returns:
        Tuple containing:

            ciphertext, shared_secret
    """

    if not isinstance(server_info, Mapping):
        raise TypeError(
            "server_info must be a mapping."
        )

    if "ml_kem_public_key" not in server_info:
        raise MLKEMPublicKeyError(
            "server_info is missing "
            "'ml_kem_public_key'."
        )

    if (
        "ml_kem_algorithm" in server_info
        and server_info["ml_kem_algorithm"]
        != ML_KEM_ALGORITHM
    ):
        raise MLKEMPublicKeyError(
            "Server package does not declare "
            f"{ML_KEM_ALGORITHM}."
        )

    try:
        server_public_key = decode_base64(
            server_info["ml_kem_public_key"],
            "ml_kem_public_key",
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise MLKEMPublicKeyError(
            "Invalid encoded ML-KEM public key."
        ) from error

    result = encapsulate_session_secret(
        server_public_key
    )

    return (
        result.ciphertext,
        result.shared_secret,
    )


def encapsulate_base64_public_key(
    server_public_key_base64: str,
) -> MLKEMEncapsulationResult:
    """
    Encapsulate using a Base64-encoded verified public key.
    """

    try:
        server_public_key = decode_base64(
            server_public_key_base64,
            "server_public_key_base64",
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise MLKEMPublicKeyError(
            "Invalid Base64 ML-KEM public key."
        ) from error

    return encapsulate_session_secret(
        server_public_key
    )


def verify_shared_secrets(
    mobile_shared_secret: bytes,
    server_shared_secret: bytes,
) -> bool:
    """
    Compare Mobile Station and Authentication Server secrets.

    This helper is for tests only.
    """

    validate_shared_secret(
        mobile_shared_secret
    )

    validate_shared_secret(
        server_shared_secret
    )

    return hmac.compare_digest(
        mobile_shared_secret,
        server_shared_secret,
    )


def run_self_test() -> None:
    """
    Run a complete ML-KEM encapsulation/decapsulation test.

    Test sequence:

        1. AS generates an ephemeral ML-KEM key pair.
        2. MS encapsulates using the AS public key.
        3. AS decapsulates using the AS secret key.
        4. Both shared secrets are compared.
    """

    print("=" * 70)
    print("FT-QuPAP ML-KEM Encapsulation Self-Test")
    print("=" * 70)

    server_public_key, server_secret_key = (
        ml_kem_768.generate_keypair()
    )

    result = encapsulate_session_secret(
        server_public_key
    )

    ciphertext_valid, ciphertext_reason = (
        validate_mlkem_ciphertext(
            result.ciphertext
        )
    )

    try:
        server_shared_secret = (
            ml_kem_768.decrypt(
                server_secret_key,
                result.ciphertext,
            )
        )

    except Exception as error:
        raise MLKEMEncapsulationError(
            "ML-KEM-768 self-test decapsulation failed."
        ) from error

    server_shared_secret = bytes(
        server_shared_secret
    )

    secrets_match = verify_shared_secrets(
        result.shared_secret,
        server_shared_secret,
    )

    server_info = {
        "ml_kem_algorithm":
            ML_KEM_ALGORITHM,
        "ml_kem_public_key":
            encode_base64(
                server_public_key
            ),
    }

    notebook_ciphertext, notebook_secret = (
        mlkem_encapsulate(
            server_info
        )
    )

    notebook_server_secret = (
        ml_kem_768.decrypt(
            server_secret_key,
            notebook_ciphertext,
        )
    )

    notebook_interface_valid = (
        hmac.compare_digest(
            notebook_secret,
            bytes(notebook_server_secret),
        )
    )

    print(
        f"Algorithm                 : "
        f"{ML_KEM_ALGORITHM}"
    )
    print(
        f"Public-key bytes          : "
        f"{len(server_public_key)}"
    )
    print(
        f"Secret-key bytes          : "
        f"{len(server_secret_key)}"
    )
    print(
        f"Ciphertext bytes          : "
        f"{len(result.ciphertext)}"
    )
    print(
        f"Shared-secret bytes       : "
        f"{len(result.shared_secret)}"
    )
    print(
        f"Ciphertext format valid   : "
        f"{ciphertext_valid}, "
        f"{ciphertext_reason}"
    )
    print(
        f"Shared secrets equal      : "
        f"{secrets_match}"
    )
    print(
        f"Notebook interface valid  : "
        f"{notebook_interface_valid}"
    )
    print(
        f"Ciphertext fingerprint    : "
        f"{result.ciphertext_fingerprint}"
    )
    print(
        f"Shared-secret fingerprint : "
        f"{result.shared_secret_fingerprint}"
    )
    print(
        f"Encapsulation time        : "
        f"{result.encapsulation_time_ms:.4f} ms"
    )

    if not ciphertext_valid:
        raise MLKEMEncapsulationError(
            ciphertext_reason
        )

    if not secrets_match:
        raise MLKEMEncapsulationError(
            "Mobile Station and Authentication Server "
            "shared secrets do not match."
        )

    if not notebook_interface_valid:
        raise MLKEMEncapsulationError(
            "Notebook-compatible interface failed."
        )

    print(
        "\nML-KEM encapsulation self-test "
        "completed successfully."
    )


__all__ = [
    "ML_KEM_ALGORITHM",
    "MLKEMEncapsulationResult",
    "MLKEMEncapsulationError",
    "MLKEMPublicKeyError",
    "MLKEMCiphertextError",
    "encapsulate_session_secret",
    "encapsulate_base64_public_key",
    "mlkem_encapsulate",
    "validate_mlkem_ciphertext",
    "verify_shared_secrets",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        MLKEMEncapsulationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"\n[ML-KEM ENCAPSULATION ERROR] "
            f"{error}"
        )