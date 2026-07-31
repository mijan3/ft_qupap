"""
ML-KEM key-establishment operations for FT-QuPAP v5.1.

Protocol use:

1. Authentication Server generates an ML-KEM key pair.
2. The server signs its ML-KEM public key using ML-DSA.
3. Mobile Station verifies the ML-DSA signature.
4. Mobile Station encapsulates a shared secret with the public key.
5. Mobile Station sends the ML-KEM ciphertext to the server.
6. Authentication Server decapsulates the same shared secret.
7. Both sides derive transcript-bound session keys.

This module provides:

- ML-KEM key-pair generation
- Shared-secret encapsulation
- Shared-secret decapsulation
- Constant-time shared-secret comparison
- Public-key fingerprints
- Backend diagnostics
- ML-KEM self-testing
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from hmac import compare_digest
from importlib import import_module
from types import ModuleType
from typing import Any

from src.common.constants import ML_KEM_ALGORITHM

from src.common.exceptions import (
    MLKEMCiphertextError,
    MLKEMDecapsulationError,
    MLKEMError,
    ProtocolValidationError,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)

from src.cryptography.crypto_models import (
    MLKEMDecapsulationResult,
    MLKEMEncapsulationResult,
    MLKEMKeyPair,
)


# ---------------------------------------------------------------------
# Supported ML-KEM algorithms
# ---------------------------------------------------------------------

MLKEM_MODULE_PATHS: dict[str, str] = {
    "ML-KEM-512": "pqcrypto.kem.ml_kem_512",
    "ML-KEM-768": "pqcrypto.kem.ml_kem_768",
    "ML-KEM-1024": "pqcrypto.kem.ml_kem_1024",
}


DEFAULT_MLKEM_ALGORITHM = ML_KEM_ALGORITHM


# ---------------------------------------------------------------------
# Algorithm-name handling
# ---------------------------------------------------------------------

def normalize_mlkem_algorithm(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> str:
    """
    Normalize an ML-KEM algorithm name.

    Accepted examples:

        ML-KEM-768
        ml-kem-768
        ML_KEM_768
        MLKEM768
        Kyber768

    All corresponding aliases return:

        ML-KEM-768
    """

    normalized_input = validate_non_empty_string(
        algorithm,
        field_name="mlkem_algorithm",
        minimum_length=1,
        maximum_length=64,
    )

    normalized = (
        normalized_input
        .strip()
        .upper()
        .replace("_", "-")
        .replace(" ", "-")
    )

    while "--" in normalized:
        normalized = normalized.replace(
            "--",
            "-",
        )

    compact = normalized.replace(
        "-",
        "",
    )

    aliases = {
        "MLKEM512": "ML-KEM-512",
        "MLKEM768": "ML-KEM-768",
        "MLKEM1024": "ML-KEM-1024",

        "KYBER512": "ML-KEM-512",
        "KYBER768": "ML-KEM-768",
        "KYBER1024": "ML-KEM-1024",
    }

    selected_algorithm = aliases.get(
        compact,
        normalized,
    )

    if selected_algorithm not in MLKEM_MODULE_PATHS:
        raise MLKEMError(
            f"Unsupported ML-KEM algorithm: {algorithm}",
            details={
                "received_algorithm": algorithm,
                "normalized_algorithm": selected_algorithm,
                "supported_algorithms": list(
                    MLKEM_MODULE_PATHS.keys()
                ),
            },
        )

    return selected_algorithm


def get_mlkem_module_path(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> str:
    """
    Return the pqcrypto module path for an ML-KEM algorithm.
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    return MLKEM_MODULE_PATHS[
        normalized_algorithm
    ]


# ---------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_mlkem_backend(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> ModuleType:
    """
    Import and validate the selected pqcrypto ML-KEM backend.
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    module_path = get_mlkem_module_path(
        normalized_algorithm
    )

    try:
        module = import_module(
            module_path
        )

    except ModuleNotFoundError as exc:
        raise MLKEMError(
            (
                f"Unable to import {module_path}. "
                "Install pqcrypto in the active environment."
            ),
            details={
                "algorithm": normalized_algorithm,
                "module_path": module_path,
                "installation_command": (
                    "python -m pip install pqcrypto"
                ),
                "reason": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise MLKEMError(
            "Unable to initialize the ML-KEM backend.",
            details={
                "algorithm": normalized_algorithm,
                "module_path": module_path,
                "reason": str(exc),
            },
        ) from exc

    required_functions = (
        "generate_keypair",
        "encrypt",
        "decrypt",
    )

    missing_functions = [
        function_name
        for function_name in required_functions
        if not callable(
            getattr(
                module,
                function_name,
                None,
            )
        )
    ]

    if missing_functions:
        raise MLKEMError(
            "The installed ML-KEM backend has an incompatible API.",
            details={
                "algorithm": normalized_algorithm,
                "module_path": module_path,
                "missing_functions": missing_functions,
            },
        )

    return module


# ---------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------

def generate_mlkem_keypair(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> MLKEMKeyPair:
    """
    Generate an ML-KEM public and secret key pair.

    The Authentication Server keeps the secret key private.

    The public key is included in the server package and authenticated
    using an ML-DSA signature.
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    backend = load_mlkem_backend(
        normalized_algorithm
    )

    try:
        public_key, secret_key = (
            backend.generate_keypair()
        )

    except Exception as exc:
        raise MLKEMError(
            "ML-KEM key-pair generation failed.",
            details={
                "algorithm": normalized_algorithm,
                "reason": str(exc),
            },
        ) from exc

    validated_public_key = validate_bytes(
        public_key,
        field_name="mlkem_public_key",
        minimum_length=1,
    )

    validated_secret_key = validate_bytes(
        secret_key,
        field_name="mlkem_secret_key",
        minimum_length=1,
    )

    return MLKEMKeyPair(
        public_key=validated_public_key,
        secret_key=validated_secret_key,
        algorithm=normalized_algorithm,
    )


# ---------------------------------------------------------------------
# Encapsulation
# ---------------------------------------------------------------------

def encapsulate_shared_secret(
    public_key: bytes,
    *,
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> MLKEMEncapsulationResult:
    """
    Encapsulate a new shared secret using an ML-KEM public key.

    This operation is performed by the Mobile Station.

    Returns:

        ciphertext:
            Sent to the Authentication Server.

        shared_secret:
            Retained locally by the Mobile Station.
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    validated_public_key = validate_bytes(
        public_key,
        field_name="mlkem_public_key",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    backend = load_mlkem_backend(
        normalized_algorithm
    )

    try:
        ciphertext, shared_secret = backend.encrypt(
            validated_public_key
        )

    except Exception as exc:
        raise MLKEMError(
            "ML-KEM shared-secret encapsulation failed.",
            details={
                "algorithm": normalized_algorithm,
                "public_key_bytes": len(
                    validated_public_key
                ),
                "reason": str(exc),
            },
        ) from exc

    validated_ciphertext = validate_bytes(
        ciphertext,
        field_name="mlkem_ciphertext",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    validated_shared_secret = validate_bytes(
        shared_secret,
        field_name="mlkem_shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    return MLKEMEncapsulationResult(
        ciphertext=validated_ciphertext,
        shared_secret=validated_shared_secret,
        algorithm=normalized_algorithm,
    )


def encapsulate_with_public_key(
    keypair: MLKEMKeyPair,
) -> MLKEMEncapsulationResult:
    """
    Encapsulate using the public key stored in an MLKEMKeyPair.

    This helper is mainly useful in tests. In the actual protocol, the
    Mobile Station normally receives only the server's public key.
    """

    if not isinstance(
        keypair,
        MLKEMKeyPair,
    ):
        raise ProtocolValidationError(
            "keypair must be an MLKEMKeyPair object.",
            details={
                "received_type": type(
                    keypair
                ).__name__,
            },
        )

    return encapsulate_shared_secret(
        public_key=keypair.public_key,
        algorithm=keypair.algorithm,
    )


# ---------------------------------------------------------------------
# Decapsulation
# ---------------------------------------------------------------------

def decapsulate_shared_secret(
    secret_key: bytes,
    ciphertext: bytes,
    *,
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> MLKEMDecapsulationResult:
    """
    Recover the shared secret from an ML-KEM ciphertext.

    This operation is performed by the Authentication Server.

    pqcrypto argument order:

        decrypt(secret_key, ciphertext)
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    validated_secret_key = validate_bytes(
        secret_key,
        field_name="mlkem_secret_key",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    validated_ciphertext = validate_bytes(
        ciphertext,
        field_name="mlkem_ciphertext",
        minimum_length=1,
        maximum_length=1_000_000,
    )

    backend = load_mlkem_backend(
        normalized_algorithm
    )

    try:
        shared_secret = backend.decrypt(
            validated_secret_key,
            validated_ciphertext,
        )

    except ValueError as exc:
        raise MLKEMCiphertextError(
            message=(
                "ML-KEM ciphertext validation or "
                "decapsulation failed."
            ),
            details={
                "algorithm": normalized_algorithm,
                "ciphertext_bytes": len(
                    validated_ciphertext
                ),
                "reason": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise MLKEMDecapsulationError(
            message=(
                "The Authentication Server could not "
                "decapsulate the ML-KEM ciphertext."
            ),
            details={
                "algorithm": normalized_algorithm,
                "secret_key_bytes": len(
                    validated_secret_key
                ),
                "ciphertext_bytes": len(
                    validated_ciphertext
                ),
                "reason": str(exc),
            },
        ) from exc

    validated_shared_secret = validate_bytes(
        shared_secret,
        field_name="decapsulated_shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    return MLKEMDecapsulationResult(
        shared_secret=validated_shared_secret,
        algorithm=normalized_algorithm,
        success=True,
    )


def decapsulate_with_keypair(
    keypair: MLKEMKeyPair,
    ciphertext: bytes,
) -> MLKEMDecapsulationResult:
    """
    Decapsulate a ciphertext using an MLKEMKeyPair object.
    """

    if not isinstance(
        keypair,
        MLKEMKeyPair,
    ):
        raise ProtocolValidationError(
            "keypair must be an MLKEMKeyPair object.",
            details={
                "received_type": type(
                    keypair
                ).__name__,
            },
        )

    return decapsulate_shared_secret(
        secret_key=keypair.secret_key,
        ciphertext=ciphertext,
        algorithm=keypair.algorithm,
    )


# ---------------------------------------------------------------------
# Shared-secret comparison
# ---------------------------------------------------------------------

def compare_shared_secrets(
    first_shared_secret: bytes,
    second_shared_secret: bytes,
) -> bool:
    """
    Compare two shared secrets using constant-time comparison.

    This helper is used by tests and demonstration diagnostics.

    In the real network protocol, the participants do not send their
    shared secrets to each other.
    """

    validated_first_secret = validate_bytes(
        first_shared_secret,
        field_name="first_shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    validated_second_secret = validate_bytes(
        second_shared_secret,
        field_name="second_shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    return compare_digest(
        validated_first_secret,
        validated_second_secret,
    )


# ---------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------

def mlkem_public_key_fingerprint(
    public_key: bytes,
    *,
    fingerprint_bytes: int = 16,
) -> str:
    """
    Create a truncated SHA3-256 fingerprint of an ML-KEM public key.

    This fingerprint may be displayed in logs and the dashboard.
    """

    validated_public_key = validate_bytes(
        public_key,
        field_name="mlkem_public_key",
        minimum_length=1,
    )

    validated_length = validate_integer(
        fingerprint_bytes,
        field_name="fingerprint_bytes",
        minimum=4,
        maximum=32,
    )

    digest = hashlib.sha3_256(
        validated_public_key
    ).digest()

    return digest[
        :validated_length
    ].hex()


def shared_secret_fingerprint(
    shared_secret: bytes,
    *,
    fingerprint_bytes: int = 8,
) -> str:
    """
    Create a short diagnostic fingerprint of a shared secret.

    The complete shared secret is never returned.
    """

    validated_shared_secret = validate_bytes(
        shared_secret,
        field_name="shared_secret",
        minimum_length=16,
        maximum_length=4096,
    )

    validated_length = validate_integer(
        fingerprint_bytes,
        field_name="fingerprint_bytes",
        minimum=4,
        maximum=32,
    )

    digest = hashlib.sha3_256(
        validated_shared_secret
    ).digest()

    return digest[
        :validated_length
    ].hex()


# ---------------------------------------------------------------------
# Backend information
# ---------------------------------------------------------------------

def _read_backend_integer(
    backend: ModuleType,
    attribute_names: tuple[str, ...],
) -> int | None:
    """
    Read an optional integer-size constant from a pqcrypto backend.

    Different package versions may expose different constant names.
    """

    for attribute_name in attribute_names:
        value = getattr(
            backend,
            attribute_name,
            None,
        )

        if (
            isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return value

    return None


def get_mlkem_backend_information(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> dict[str, Any]:
    """
    Return information about the configured ML-KEM backend.
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    module_path = get_mlkem_module_path(
        normalized_algorithm
    )

    try:
        backend = load_mlkem_backend(
            normalized_algorithm
        )

        available = True
        error = None
        loaded_module = backend.__name__

        public_key_size = _read_backend_integer(
            backend,
            (
                "PUBLIC_KEY_SIZE",
                "PUBLICKEYBYTES",
            ),
        )

        secret_key_size = _read_backend_integer(
            backend,
            (
                "SECRET_KEY_SIZE",
                "SECRETKEYBYTES",
            ),
        )

        ciphertext_size = _read_backend_integer(
            backend,
            (
                "CIPHERTEXT_SIZE",
                "CIPHERTEXTBYTES",
            ),
        )

        shared_secret_size = _read_backend_integer(
            backend,
            (
                "PLAINTEXT_SIZE",
                "SHARED_SECRET_SIZE",
                "BYTES",
            ),
        )

    except MLKEMError as exc:
        available = False
        error = str(exc)
        loaded_module = None
        public_key_size = None
        secret_key_size = None
        ciphertext_size = None
        shared_secret_size = None

    return {
        "algorithm": normalized_algorithm,
        "module_path": module_path,
        "available": available,
        "loaded_module": loaded_module,
        "public_key_bytes": public_key_size,
        "secret_key_bytes": secret_key_size,
        "ciphertext_bytes": ciphertext_size,
        "shared_secret_bytes": shared_secret_size,
        "error": error,
    }


# ---------------------------------------------------------------------
# ML-KEM self-test
# ---------------------------------------------------------------------

def run_mlkem_self_test(
    algorithm: str = DEFAULT_MLKEM_ALGORITHM,
) -> dict[str, Any]:
    """
    Run a real ML-KEM key-establishment self-test.

    The test confirms:

    - Key generation succeeds
    - Encapsulation succeeds
    - Decapsulation succeeds
    - Mobile and server shared secrets match
    - A modified ciphertext does not recover the original secret
    """

    normalized_algorithm = normalize_mlkem_algorithm(
        algorithm
    )

    try:
        keypair = generate_mlkem_keypair(
            normalized_algorithm
        )

        encapsulation = encapsulate_shared_secret(
            public_key=keypair.public_key,
            algorithm=normalized_algorithm,
        )

        decapsulation = decapsulate_shared_secret(
            secret_key=keypair.secret_key,
            ciphertext=encapsulation.ciphertext,
            algorithm=normalized_algorithm,
        )

        shared_secret_match = compare_shared_secrets(
            encapsulation.shared_secret,
            decapsulation.shared_secret,
        )

        tampered_ciphertext = bytearray(
            encapsulation.ciphertext
        )

        tampered_ciphertext[0] ^= 0x01

        try:
            tampered_decapsulation = (
                decapsulate_shared_secret(
                    secret_key=keypair.secret_key,
                    ciphertext=bytes(
                        tampered_ciphertext
                    ),
                    algorithm=normalized_algorithm,
                )
            )

            tampered_ciphertext_rejected = (
                not compare_shared_secrets(
                    encapsulation.shared_secret,
                    tampered_decapsulation.shared_secret,
                )
            )

        except (
            MLKEMCiphertextError,
            MLKEMDecapsulationError,
        ):
            tampered_ciphertext_rejected = True

        success = all(
            (
                len(keypair.public_key) > 0,
                len(keypair.secret_key) > 0,
                len(encapsulation.ciphertext) > 0,
                len(encapsulation.shared_secret) > 0,
                len(decapsulation.shared_secret) > 0,
                shared_secret_match,
                tampered_ciphertext_rejected,
            )
        )

        return {
            "success": success,
            "algorithm": normalized_algorithm,

            "public_key_bytes": len(
                keypair.public_key
            ),

            "secret_key_bytes": len(
                keypair.secret_key
            ),

            "ciphertext_bytes": len(
                encapsulation.ciphertext
            ),

            "shared_secret_bytes": len(
                encapsulation.shared_secret
            ),

            "public_key_fingerprint": (
                mlkem_public_key_fingerprint(
                    keypair.public_key
                )
            ),

            "mobile_shared_secret_fingerprint": (
                shared_secret_fingerprint(
                    encapsulation.shared_secret
                )
            ),

            "server_shared_secret_fingerprint": (
                shared_secret_fingerprint(
                    decapsulation.shared_secret
                )
            ),

            "shared_secret_match": (
                shared_secret_match
            ),

            "tampered_ciphertext_rejected": (
                tampered_ciphertext_rejected
            ),

            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "algorithm": normalized_algorithm,
            "public_key_bytes": 0,
            "secret_key_bytes": 0,
            "ciphertext_bytes": 0,
            "shared_secret_bytes": 0,
            "public_key_fingerprint": None,
            "mobile_shared_secret_fingerprint": None,
            "server_shared_secret_fingerprint": None,
            "shared_secret_match": False,
            "tampered_ciphertext_rejected": False,
            "error": str(exc),
        }


__all__ = [
    "MLKEM_MODULE_PATHS",
    "DEFAULT_MLKEM_ALGORITHM",
    "normalize_mlkem_algorithm",
    "get_mlkem_module_path",
    "load_mlkem_backend",
    "generate_mlkem_keypair",
    "encapsulate_shared_secret",
    "encapsulate_with_public_key",
    "decapsulate_shared_secret",
    "decapsulate_with_keypair",
    "compare_shared_secrets",
    "mlkem_public_key_fingerprint",
    "shared_secret_fingerprint",
    "get_mlkem_backend_information",
    "run_mlkem_self_test",
]