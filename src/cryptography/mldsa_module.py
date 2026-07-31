"""
ML-DSA digital-signature operations for FT-QuPAP v5.1.

The Authentication Server uses ML-DSA to sign the temporary ML-KEM
public-key package. The Mobile Station verifies the signature before
performing ML-KEM encapsulation.

This module provides:

- ML-DSA key-pair generation
- Byte-message signing
- Canonical dictionary signing
- Signature verification
- Required-verification helpers
- Public-key fingerprint generation
- ML-DSA self-testing
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any, Mapping

from src.common.constants import (
    ML_DSA_ALGORITHM,
)

from src.common.exceptions import (
    MLDSAError,
    MLDSAVerificationError,
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
)

from src.common.validators import (
    validate_bytes,
    validate_non_empty_string,
)

from src.cryptography.crypto_models import (
    MLDSAKeyPair,
    MLDSASignature,
)


# ---------------------------------------------------------------------
# Supported ML-DSA algorithms
# ---------------------------------------------------------------------

MLDSA_MODULE_PATHS: dict[str, str] = {
    "ML-DSA-44": "pqcrypto.sign.ml_dsa_44",
    "ML-DSA-65": "pqcrypto.sign.ml_dsa_65",
    "ML-DSA-87": "pqcrypto.sign.ml_dsa_87",
}


DEFAULT_MLDSA_ALGORITHM = ML_DSA_ALGORITHM


# ---------------------------------------------------------------------
# Algorithm handling
# ---------------------------------------------------------------------

def normalize_mldsa_algorithm(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> str:
    """
    Normalize an ML-DSA algorithm name.

    Accepted examples:

        ML-DSA-65
        ml-dsa-65
        ML_DSA_65
        MLDSA65
        Dilithium3

    All corresponding aliases return:

        ML-DSA-65
    """

    normalized_input = validate_non_empty_string(
        algorithm,
        field_name="mldsa_algorithm",
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
        "MLDSA44": "ML-DSA-44",
        "MLDSA65": "ML-DSA-65",
        "MLDSA87": "ML-DSA-87",

        "DILITHIUM2": "ML-DSA-44",
        "DILITHIUM3": "ML-DSA-65",
        "DILITHIUM5": "ML-DSA-87",
    }

    selected_algorithm = aliases.get(
        compact,
        normalized,
    )

    if selected_algorithm not in MLDSA_MODULE_PATHS:
        raise MLDSAError(
            f"Unsupported ML-DSA algorithm: {algorithm}",
            details={
                "received_algorithm": algorithm,
                "normalized_algorithm": (
                    selected_algorithm
                ),
                "supported_algorithms": list(
                    MLDSA_MODULE_PATHS.keys()
                ),
            },
        )

    return selected_algorithm


def get_mldsa_module_path(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> str:
    """
    Return the pqcrypto module path for the selected algorithm.
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    return MLDSA_MODULE_PATHS[
        normalized_algorithm
    ]


# ---------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_mldsa_backend(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> ModuleType:
    """
    Import and validate the configured pqcrypto ML-DSA module.
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    module_path = get_mldsa_module_path(
        normalized_algorithm
    )

    try:
        module = import_module(
            module_path
        )
    except ModuleNotFoundError as exc:
        raise MLDSAError(
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
        raise MLDSAError(
            "Unable to initialize the ML-DSA backend.",
            details={
                "algorithm": normalized_algorithm,
                "module_path": module_path,
                "reason": str(exc),
            },
        ) from exc

    required_functions = (
        "generate_keypair",
        "sign",
        "verify",
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
        raise MLDSAError(
            "The installed ML-DSA backend has an incompatible API.",
            details={
                "algorithm": normalized_algorithm,
                "module_path": module_path,
                "missing_functions": (
                    missing_functions
                ),
            },
        )

    return module


# ---------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------

def generate_mldsa_keypair(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> MLDSAKeyPair:
    """
    Generate a real ML-DSA public and secret key pair.

    The Authentication Server retains the secret key. The public key is
    distributed to or provisioned at the Mobile Station.
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    backend = load_mldsa_backend(
        normalized_algorithm
    )

    try:
        public_key, secret_key = (
            backend.generate_keypair()
        )
    except Exception as exc:
        raise MLDSAError(
            "ML-DSA key-pair generation failed.",
            details={
                "algorithm": normalized_algorithm,
                "reason": str(exc),
            },
        ) from exc

    validated_public_key = validate_bytes(
        public_key,
        field_name="mldsa_public_key",
        minimum_length=1,
    )

    validated_secret_key = validate_bytes(
        secret_key,
        field_name="mldsa_secret_key",
        minimum_length=1,
    )

    return MLDSAKeyPair(
        public_key=validated_public_key,
        secret_key=validated_secret_key,
        algorithm=normalized_algorithm,
    )


# ---------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------

def sign_message(
    secret_key: bytes,
    message: bytes,
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> MLDSASignature:
    """
    Sign raw message bytes using an ML-DSA secret key.
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    validated_secret_key = validate_bytes(
        secret_key,
        field_name="mldsa_secret_key",
        minimum_length=1,
    )

    validated_message = validate_bytes(
        message,
        field_name="message",
        minimum_length=1,
        maximum_length=100_000_000,
    )

    backend = load_mldsa_backend(
        normalized_algorithm
    )

    try:
        signature = backend.sign(
            validated_secret_key,
            validated_message,
        )
    except Exception as exc:
        raise MLDSAError(
            "ML-DSA message signing failed.",
            details={
                "algorithm": normalized_algorithm,
                "message_bytes": len(
                    validated_message
                ),
                "reason": str(exc),
            },
        ) from exc

    validated_signature = validate_bytes(
        signature,
        field_name="mldsa_signature",
        minimum_length=1,
    )

    return MLDSASignature(
        signature=validated_signature,
        algorithm=normalized_algorithm,
    )


def sign_payload(
    secret_key: bytes,
    payload: Mapping[str, Any],
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> MLDSASignature:
    """
    Canonically serialize and sign a dictionary-like protocol payload.

    Canonical serialization ensures that both protocol participants use
    identical field ordering and byte representation.
    """

    if not isinstance(payload, Mapping):
        raise ProtocolValidationError(
            "ML-DSA payload must be a mapping.",
            details={
                "received_type": type(
                    payload
                ).__name__,
            },
        )

    serialized_payload = (
        canonical_json_bytes(
            dict(payload)
        )
    )

    return sign_message(
        secret_key=secret_key,
        message=serialized_payload,
        algorithm=algorithm,
    )


def sign_with_keypair(
    keypair: MLDSAKeyPair,
    message: bytes,
) -> MLDSASignature:
    """
    Sign raw bytes using an MLDSAKeyPair object.
    """

    if not isinstance(
        keypair,
        MLDSAKeyPair,
    ):
        raise ProtocolValidationError(
            "keypair must be an MLDSAKeyPair object.",
            details={
                "received_type": type(
                    keypair
                ).__name__,
            },
        )

    return sign_message(
        secret_key=keypair.secret_key,
        message=message,
        algorithm=keypair.algorithm,
    )


def sign_payload_with_keypair(
    keypair: MLDSAKeyPair,
    payload: Mapping[str, Any],
) -> MLDSASignature:
    """
    Canonically serialize and sign a payload using an MLDSAKeyPair.
    """

    if not isinstance(
        keypair,
        MLDSAKeyPair,
    ):
        raise ProtocolValidationError(
            "keypair must be an MLDSAKeyPair object.",
            details={
                "received_type": type(
                    keypair
                ).__name__,
            },
        )

    return sign_payload(
        secret_key=keypair.secret_key,
        payload=payload,
        algorithm=keypair.algorithm,
    )


# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

def verify_signature(
    public_key: bytes,
    message: bytes,
    signature: bytes | MLDSASignature,
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> bool:
    """
    Verify an ML-DSA signature.

    Returns:

        True:
            The signature is valid.

        False:
            The signature, message, key, or algorithm does not match.
    """

    if isinstance(
        signature,
        MLDSASignature,
    ):
        signature_bytes = signature.signature
        signature_algorithm = (
            normalize_mldsa_algorithm(
                signature.algorithm
            )
        )

        requested_algorithm = (
            normalize_mldsa_algorithm(
                algorithm
            )
        )

        if (
            signature_algorithm
            != requested_algorithm
        ):
            return False

        normalized_algorithm = (
            signature_algorithm
        )
    else:
        signature_bytes = signature

        normalized_algorithm = (
            normalize_mldsa_algorithm(
                algorithm
            )
        )

    validated_public_key = validate_bytes(
        public_key,
        field_name="mldsa_public_key",
        minimum_length=1,
    )

    validated_message = validate_bytes(
        message,
        field_name="message",
        minimum_length=1,
        maximum_length=100_000_000,
    )

    validated_signature = validate_bytes(
        signature_bytes,
        field_name="mldsa_signature",
        minimum_length=1,
    )

    backend = load_mldsa_backend(
        normalized_algorithm
    )

    try:
        verification_result = backend.verify(
            validated_public_key,
            validated_message,
            validated_signature,
        )

        return bool(
            verification_result
        )

    except Exception:
        # Some backend versions may raise an exception when the
        # signature is malformed or invalid.
        return False


def verify_payload_signature(
    public_key: bytes,
    payload: Mapping[str, Any],
    signature: bytes | MLDSASignature,
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> bool:
    """
    Canonically serialize a payload and verify its ML-DSA signature.
    """

    if not isinstance(payload, Mapping):
        raise ProtocolValidationError(
            "ML-DSA payload must be a mapping.",
            details={
                "received_type": type(
                    payload
                ).__name__,
            },
        )

    serialized_payload = (
        canonical_json_bytes(
            dict(payload)
        )
    )

    return verify_signature(
        public_key=public_key,
        message=serialized_payload,
        signature=signature,
        algorithm=algorithm,
    )


def require_valid_signature(
    public_key: bytes,
    message: bytes,
    signature: bytes | MLDSASignature,
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> None:
    """
    Verify a signature and raise MLDSAVerificationError on failure.
    """

    valid = verify_signature(
        public_key=public_key,
        message=message,
        signature=signature,
        algorithm=algorithm,
    )

    if not valid:
        raise MLDSAVerificationError(
            details={
                "algorithm": (
                    normalize_mldsa_algorithm(
                        algorithm
                    )
                ),
                "message_bytes": len(message),
            },
        )


def require_valid_payload_signature(
    public_key: bytes,
    payload: Mapping[str, Any],
    signature: bytes | MLDSASignature,
    *,
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> None:
    """
    Verify a signed protocol payload and raise an exception on failure.
    """

    valid = verify_payload_signature(
        public_key=public_key,
        payload=payload,
        signature=signature,
        algorithm=algorithm,
    )

    if not valid:
        raise MLDSAVerificationError(
            message=(
                "ML-DSA protocol-payload "
                "signature verification failed."
            ),
            details={
                "algorithm": (
                    normalize_mldsa_algorithm(
                        algorithm
                    )
                ),
                "payload_fields": sorted(
                    str(field_name)
                    for field_name in payload.keys()
                ),
            },
        )


# ---------------------------------------------------------------------
# Public-key fingerprint
# ---------------------------------------------------------------------

def mldsa_public_key_fingerprint(
    public_key: bytes,
    *,
    fingerprint_bytes: int = 16,
) -> str:
    """
    Create a SHA3-256 fingerprint of an ML-DSA public key.

    The fingerprint is suitable for logs and dashboard display. It is
    not a replacement for the complete trusted public key.
    """

    validated_public_key = validate_bytes(
        public_key,
        field_name="mldsa_public_key",
        minimum_length=1,
    )

    if (
        not isinstance(
            fingerprint_bytes,
            int,
        )
        or isinstance(
            fingerprint_bytes,
            bool,
        )
    ):
        raise ProtocolValidationError(
            "fingerprint_bytes must be an integer."
        )

    if not 4 <= fingerprint_bytes <= 32:
        raise ProtocolValidationError(
            (
                "fingerprint_bytes must be "
                "between 4 and 32."
            ),
            details={
                "fingerprint_bytes": (
                    fingerprint_bytes
                ),
            },
        )

    digest = hashlib.sha3_256(
        validated_public_key
    ).digest()

    return digest[
        :fingerprint_bytes
    ].hex()


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def get_mldsa_backend_information(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> dict[str, Any]:
    """
    Return information about the selected ML-DSA backend.
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    module_path = get_mldsa_module_path(
        normalized_algorithm
    )

    try:
        backend = load_mldsa_backend(
            normalized_algorithm
        )

        available = True
        error = None
        loaded_module = backend.__name__

    except MLDSAError as exc:
        available = False
        error = str(exc)
        loaded_module = None

    return {
        "algorithm": normalized_algorithm,
        "module_path": module_path,
        "available": available,
        "loaded_module": loaded_module,
        "error": error,
    }


def run_mldsa_self_test(
    algorithm: str = DEFAULT_MLDSA_ALGORITHM,
) -> dict[str, Any]:
    """
    Run a real ML-DSA signing and verification self-test.

    The test confirms:

    - Key generation succeeds
    - Signing succeeds
    - The original message verifies
    - A modified message is rejected
    - A modified signature is rejected
    - Canonical payload signing works
    """

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    message = (
        b"FT-QuPAP-v5.1-ML-DSA-self-test"
    )

    payload = {
        "server_id": "AS-6G-001",
        "session_id": "FTQ-MLDSA-SELF-TEST",
        "purpose": "server-authentication",
        "algorithm": normalized_algorithm,
    }

    try:
        keypair = generate_mldsa_keypair(
            normalized_algorithm
        )

        signature = sign_message(
            secret_key=keypair.secret_key,
            message=message,
            algorithm=normalized_algorithm,
        )

        original_message_valid = (
            verify_signature(
                public_key=keypair.public_key,
                message=message,
                signature=signature,
                algorithm=normalized_algorithm,
            )
        )

        modified_message_rejected = (
            not verify_signature(
                public_key=keypair.public_key,
                message=(
                    message
                    + b"-modified"
                ),
                signature=signature,
                algorithm=normalized_algorithm,
            )
        )

        modified_signature_bytes = bytearray(
            signature.signature
        )

        modified_signature_bytes[0] ^= 0x01

        modified_signature_rejected = (
            not verify_signature(
                public_key=keypair.public_key,
                message=message,
                signature=bytes(
                    modified_signature_bytes
                ),
                algorithm=normalized_algorithm,
            )
        )

        payload_signature = sign_payload(
            secret_key=keypair.secret_key,
            payload=payload,
            algorithm=normalized_algorithm,
        )

        payload_signature_valid = (
            verify_payload_signature(
                public_key=keypair.public_key,
                payload=payload,
                signature=payload_signature,
                algorithm=normalized_algorithm,
            )
        )

        changed_payload = dict(payload)

        changed_payload["server_id"] = (
            "ATTACKER-SERVER"
        )

        changed_payload_rejected = (
            not verify_payload_signature(
                public_key=keypair.public_key,
                payload=changed_payload,
                signature=payload_signature,
                algorithm=normalized_algorithm,
            )
        )

        success = all(
            (
                original_message_valid,
                modified_message_rejected,
                modified_signature_rejected,
                payload_signature_valid,
                changed_payload_rejected,
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
            "signature_bytes": len(
                signature.signature
            ),
            "public_key_fingerprint": (
                mldsa_public_key_fingerprint(
                    keypair.public_key
                )
            ),
            "original_message_valid": (
                original_message_valid
            ),
            "modified_message_rejected": (
                modified_message_rejected
            ),
            "modified_signature_rejected": (
                modified_signature_rejected
            ),
            "payload_signature_valid": (
                payload_signature_valid
            ),
            "changed_payload_rejected": (
                changed_payload_rejected
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "algorithm": normalized_algorithm,
            "public_key_bytes": 0,
            "secret_key_bytes": 0,
            "signature_bytes": 0,
            "public_key_fingerprint": None,
            "original_message_valid": False,
            "modified_message_rejected": False,
            "modified_signature_rejected": False,
            "payload_signature_valid": False,
            "changed_payload_rejected": False,
            "error": str(exc),
        }


__all__ = [
    "MLDSA_MODULE_PATHS",
    "DEFAULT_MLDSA_ALGORITHM",
    "normalize_mldsa_algorithm",
    "get_mldsa_module_path",
    "load_mldsa_backend",
    "generate_mldsa_keypair",
    "sign_message",
    "sign_payload",
    "sign_with_keypair",
    "sign_payload_with_keypair",
    "verify_signature",
    "verify_payload_signature",
    "require_valid_signature",
    "require_valid_payload_signature",
    "mldsa_public_key_fingerprint",
    "get_mldsa_backend_information",
    "run_mldsa_self_test",
]