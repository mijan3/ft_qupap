"""
Authentication Server ML-DSA key management for FT-QuPAP v5.1.

The Authentication Server uses a long-term ML-DSA key pair to sign the
ephemeral ML-KEM public-key package sent to the Mobile Station.

This module provides:

- ML-DSA key-pair generation
- Secure private-key storage
- Public-key loading and exporting
- Key fingerprint and key-ID generation
- Message signing
- Signature verification
- Controlled key rotation
- Thread-safe key access

The ML-DSA cryptographic operations are delegated to:

    src.cryptography.mldsa_module

The secret key is never returned by public information methods.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

from src.common.constants import (
    ML_DSA_ALGORITHM,
    PROTOCOL_DOMAIN_LABEL,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
    canonical_json_bytes,
    decode_base64,
    encode_base64,
)

from src.common.time_utils import (
    current_timestamp,
)

from src.common.validators import (
    validate_bytes,
    validate_integer,
    validate_non_empty_string,
)

from src.cryptography import mldsa_module


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MLDSA_KEY_FILE_VERSION = "FT-QuPAP-MLDSA-Key-v1"

MLDSA_SIGNING_CONTEXT = b"FT-QuPAP-MLDSA-Server-Signature"

DEFAULT_MLDSA_PUBLIC_KEY_PATH = Path(
    "data/keys/authentication_server_mldsa_public.json"
)

DEFAULT_MLDSA_SECRET_KEY_PATH = Path(
    "data/keys/authentication_server_mldsa_secret.json"
)

PUBLIC_KEY_FILE_PERMISSION = 0o644

SECRET_KEY_FILE_PERMISSION = 0o600


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class MLDSAKeyManagerError(RuntimeError):
    """Raised when server ML-DSA key management fails."""

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
# Key-pair model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ManagedMLDSAKeyPair:
    """
    Validated Authentication Server ML-DSA key pair.

    Attributes
    ----------
    algorithm:
        ML-DSA parameter set used by the server.

    public_key:
        Public verification key distributed to Mobile Stations.

    secret_key:
        Private signing key retained by the Authentication Server.

    key_id:
        Short identifier derived from the public-key fingerprint.

    fingerprint:
        Full SHA3-256 public-key fingerprint.

    created_at:
        Unix timestamp when the key pair was generated.
    """

    algorithm: str
    public_key: bytes
    secret_key: bytes
    key_id: str
    fingerprint: str
    created_at: int

    def __post_init__(self) -> None:
        validate_non_empty_string(
            self.algorithm,
            field_name="algorithm",
            minimum_length=1,
            maximum_length=64,
        )

        validate_bytes(
            self.public_key,
            field_name="mldsa_public_key",
            minimum_length=32,
            maximum_length=100_000,
        )

        validate_bytes(
            self.secret_key,
            field_name="mldsa_secret_key",
            minimum_length=32,
            maximum_length=200_000,
        )

        validate_non_empty_string(
            self.key_id,
            field_name="key_id",
            minimum_length=8,
            maximum_length=128,
        )

        validate_non_empty_string(
            self.fingerprint,
            field_name="fingerprint",
            minimum_length=64,
            maximum_length=64,
        )

        validate_integer(
            self.created_at,
            field_name="created_at",
            minimum=0,
        )

        expected_fingerprint = calculate_public_key_fingerprint(
            self.public_key
        )

        if self.fingerprint.lower() != expected_fingerprint:
            raise ProtocolValidationError(
                "ML-DSA public-key fingerprint is invalid.",
                details={
                    "received_fingerprint": self.fingerprint,
                    "expected_fingerprint": expected_fingerprint,
                },
            )

        expected_key_id = create_mldsa_key_id(
            self.public_key
        )

        if self.key_id != expected_key_id:
            raise ProtocolValidationError(
                "ML-DSA key ID does not match the public key.",
                details={
                    "received_key_id": self.key_id,
                    "expected_key_id": expected_key_id,
                },
            )

    def public_dict(self) -> dict[str, Any]:
        """Return non-sensitive key information."""

        return {
            "version": MLDSA_KEY_FILE_VERSION,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "public_key": encode_base64(
                self.public_key
            ),
        }

    def private_dict(self) -> dict[str, Any]:
        """
        Return the protected private-key record.

        This dictionary must never be sent over the network.
        """

        return {
            "version": MLDSA_KEY_FILE_VERSION,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "public_key": encode_base64(
                self.public_key
            ),
            "secret_key": encode_base64(
                self.secret_key
            ),
        }

    def __repr__(self) -> str:
        return (
            "ManagedMLDSAKeyPair("
            f"algorithm={self.algorithm!r}, "
            f"key_id={self.key_id!r}, "
            f"fingerprint={self.fingerprint!r}, "
            f"created_at={self.created_at}, "
            f"public_key_bytes={len(self.public_key)}, "
            "secret_key=<hidden>"
            ")"
        )


@dataclass(frozen=True)
class MLDSAKeyRotationResult:
    """Result of Authentication Server ML-DSA key rotation."""

    previous_key_id: str | None
    previous_fingerprint: str | None

    new_key_id: str
    new_fingerprint: str

    rotated_at: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_key_id": self.previous_key_id,
            "previous_fingerprint": (
                self.previous_fingerprint
            ),
            "new_key_id": self.new_key_id,
            "new_fingerprint": self.new_fingerprint,
            "rotated_at": self.rotated_at,
        }


# ---------------------------------------------------------------------
# Algorithm normalization
# ---------------------------------------------------------------------

def normalize_mldsa_algorithm(
    algorithm: str = ML_DSA_ALGORITHM,
) -> str:
    """
    Normalize the configured ML-DSA algorithm name.
    """

    validated = validate_non_empty_string(
        algorithm,
        field_name="mldsa_algorithm",
        minimum_length=1,
        maximum_length=64,
    )

    compact = (
        validated
        .strip()
        .upper()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    aliases = {
        "MLDSA44": "ML-DSA-44",
        "MLDSA65": "ML-DSA-65",
        "MLDSA87": "ML-DSA-87",
        "DILITHIUM2": "ML-DSA-44",
        "DILITHIUM3": "ML-DSA-65",
        "DILITHIUM5": "ML-DSA-87",
    }

    normalized = aliases.get(
        compact
    )

    if normalized is None:
        raise ProtocolValidationError(
            f"Unsupported ML-DSA algorithm: {algorithm}",
            details={
                "supported_algorithms": [
                    "ML-DSA-44",
                    "ML-DSA-65",
                    "ML-DSA-87",
                ],
            },
        )

    return normalized


# ---------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------

def calculate_public_key_fingerprint(
    public_key: bytes,
) -> str:
    """
    Calculate the SHA3-256 fingerprint of an ML-DSA public key.
    """

    validated_public_key = validate_bytes(
        public_key,
        field_name="mldsa_public_key",
        minimum_length=32,
        maximum_length=100_000,
    )

    return hashlib.sha3_256(
        validated_public_key
    ).hexdigest()


def create_mldsa_key_id(
    public_key: bytes,
) -> str:
    """
    Create a short stable key identifier.

    The key ID is based on the first 16 bytes of the SHA3-256
    public-key fingerprint.
    """

    fingerprint = calculate_public_key_fingerprint(
        public_key
    )

    return (
        "MLDSA-"
        + fingerprint[:32].upper()
    )


# ---------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------

def _find_backend_function(
    names: tuple[str, ...],
) -> Callable[..., Any]:
    """
    Find a supported function in mldsa_module.

    The adapter permits small API-name differences while keeping this
    key manager compatible with the project cryptography layer.
    """

    for function_name in names:
        function = getattr(
            mldsa_module,
            function_name,
            None,
        )

        if callable(function):
            return function

    raise MLDSAKeyManagerError(
        "Required ML-DSA backend function was not found.",
        details={
            "searched_functions": list(names),
            "module": (
                "src.cryptography.mldsa_module"
            ),
        },
    )


def _invoke_compatible_call(
    function: Callable[..., Any],
    calls: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ],
) -> Any:
    """
    Invoke the first argument combination compatible with a function.
    """

    try:
        signature = inspect.signature(
            function
        )

    except (
        TypeError,
        ValueError,
    ):
        signature = None

    binding_errors: list[str] = []

    for args, kwargs in calls:
        if signature is not None:
            try:
                signature.bind(
                    *args,
                    **kwargs,
                )

            except TypeError as exc:
                binding_errors.append(
                    str(exc)
                )

                continue

        return function(
            *args,
            **kwargs,
        )

    raise MLDSAKeyManagerError(
        "Unable to call the ML-DSA backend function.",
        details={
            "function": getattr(
                function,
                "__name__",
                repr(function),
            ),
            "binding_errors": binding_errors,
        },
    )


def _normalize_generated_keypair(
    generated: Any,
    *,
    algorithm: str,
    created_at: int,
) -> ManagedMLDSAKeyPair:
    """
    Normalize common ML-DSA backend key-pair return formats.

    Supported formats:

    - Tuple: (public_key, secret_key)
    - Dictionary
    - Object with public_key and secret_key/private_key attributes
    """

    public_key: Any = None
    secret_key: Any = None

    if isinstance(
        generated,
        Mapping,
    ):
        public_key = generated.get(
            "public_key"
        )

        secret_key = generated.get(
            "secret_key",
            generated.get(
                "private_key"
            ),
        )

    elif (
        isinstance(
            generated,
            tuple,
        )
        and len(generated) == 2
    ):
        public_key = generated[0]
        secret_key = generated[1]

    else:
        public_key = getattr(
            generated,
            "public_key",
            None,
        )

        secret_key = getattr(
            generated,
            "secret_key",
            getattr(
                generated,
                "private_key",
                None,
            ),
        )

    if (
        public_key is None
        or secret_key is None
    ):
        raise MLDSAKeyManagerError(
            "ML-DSA backend returned an invalid key pair.",
            details={
                "received_type": type(
                    generated
                ).__name__,
            },
        )

    validated_public_key = validate_bytes(
        bytes(public_key),
        field_name="mldsa_public_key",
        minimum_length=32,
        maximum_length=100_000,
    )

    validated_secret_key = validate_bytes(
        bytes(secret_key),
        field_name="mldsa_secret_key",
        minimum_length=32,
        maximum_length=200_000,
    )

    fingerprint = (
        calculate_public_key_fingerprint(
            validated_public_key
        )
    )

    key_id = create_mldsa_key_id(
        validated_public_key
    )

    return ManagedMLDSAKeyPair(
        algorithm=algorithm,
        public_key=validated_public_key,
        secret_key=validated_secret_key,
        key_id=key_id,
        fingerprint=fingerprint,
        created_at=created_at,
    )


def _backend_generate_keypair(
    algorithm: str,
) -> tuple[bytes, bytes]:
    """
    Generate an ML-DSA key pair through mldsa_module.
    """

    function = _find_backend_function(
        (
            "generate_mldsa_keypair",
            "generate_keypair",
            "mldsa_generate_keypair",
            "mldsa_keygen",
        )
    )

    generated = _invoke_compatible_call(
        function,
        [
            (
                (),
                {
                    "algorithm": algorithm,
                },
            ),
            (
                (),
                {
                    "parameter_set": algorithm,
                },
            ),
            (
                (
                    algorithm,
                ),
                {},
            ),
            (
                (),
                {},
            ),
        ],
    )

    normalized = _normalize_generated_keypair(
        generated,
        algorithm=algorithm,
        created_at=current_timestamp(),
    )

    return (
        normalized.public_key,
        normalized.secret_key,
    )


def _backend_sign(
    *,
    message: bytes,
    secret_key: bytes,
    algorithm: str,
) -> bytes:
    """
    Sign bytes through the project ML-DSA backend.
    """

    function = _find_backend_function(
        (
            "sign_message",
            "mldsa_sign",
            "sign",
            "create_signature",
        )
    )

    signature = _invoke_compatible_call(
        function,
        [
            (
                (),
                {
                    "message": message,
                    "secret_key": secret_key,
                    "algorithm": algorithm,
                },
            ),
            (
                (),
                {
                    "message": message,
                    "private_key": secret_key,
                    "algorithm": algorithm,
                },
            ),
            (
                (),
                {
                    "data": message,
                    "secret_key": secret_key,
                    "algorithm": algorithm,
                },
            ),
            (
                (
                    message,
                    secret_key,
                ),
                {
                    "algorithm": algorithm,
                },
            ),
            (
                (
                    secret_key,
                    message,
                ),
                {},
            ),
        ],
    )

    return validate_bytes(
        bytes(signature),
        field_name="mldsa_signature",
        minimum_length=32,
        maximum_length=100_000,
    )


def _backend_verify(
    *,
    message: bytes,
    signature: bytes,
    public_key: bytes,
    algorithm: str,
) -> bool:
    """
    Verify an ML-DSA signature through the project backend.
    """

    function = _find_backend_function(
        (
            "verify_signature",
            "mldsa_verify",
            "verify",
            "verify_message",
        )
    )

    try:
        result = _invoke_compatible_call(
            function,
            [
                (
                    (),
                    {
                        "message": message,
                        "signature": signature,
                        "public_key": public_key,
                        "algorithm": algorithm,
                    },
                ),
                (
                    (),
                    {
                        "data": message,
                        "signature": signature,
                        "public_key": public_key,
                        "algorithm": algorithm,
                    },
                ),
                (
                    (
                        message,
                        signature,
                        public_key,
                    ),
                    {
                        "algorithm": algorithm,
                    },
                ),
                (
                    (
                        public_key,
                        message,
                        signature,
                    ),
                    {},
                ),
            ],
        )

    except Exception:
        return False

    return bool(result)


# ---------------------------------------------------------------------
# Signing-data construction
# ---------------------------------------------------------------------

def build_mldsa_signing_message(
    *,
    message: bytes,
    key_id: str,
    algorithm: str,
    context: bytes = MLDSA_SIGNING_CONTEXT,
) -> bytes:
    """
    Build a deterministic domain-separated ML-DSA signing message.

    This prevents the same signature from being valid in an unrelated
    protocol or application context.
    """

    validated_message = validate_bytes(
        message,
        field_name="message",
        minimum_length=1,
        maximum_length=10_000_000,
    )

    validated_context = validate_bytes(
        context,
        field_name="mldsa_signing_context",
        minimum_length=1,
        maximum_length=256,
    )

    validated_key_id = validate_non_empty_string(
        key_id,
        field_name="key_id",
        minimum_length=8,
        maximum_length=128,
    )

    normalized_algorithm = (
        normalize_mldsa_algorithm(
            algorithm
        )
    )

    payload = {
        "domain": PROTOCOL_DOMAIN_LABEL.decode(
            "utf-8",
            errors="strict",
        ),
        "purpose": "authentication-server-signature",
        "algorithm": normalized_algorithm,
        "key_id": validated_key_id,
        "context": encode_base64(
            validated_context
        ),
        "message": encode_base64(
            validated_message
        ),
    }

    return canonical_json_bytes(
        payload
    )


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def _write_json_atomically(
    path: Path,
    data: Mapping[str, Any],
    *,
    permission: int,
) -> None:
    """
    Write a JSON object atomically and apply file permissions.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    encoded = canonical_json_bytes(
        dict(data)
    )

    try:
        with temporary_path.open(
            "wb",
        ) as file:
            file.write(encoded)
            file.flush()

            try:
                os.fsync(
                    file.fileno()
                )

            except OSError:
                pass

        try:
            os.chmod(
                temporary_path,
                permission,
            )

        except OSError:
            pass

        os.replace(
            temporary_path,
            path,
        )

        try:
            os.chmod(
                path,
                permission,
            )

        except OSError:
            pass

    except Exception as exc:
        try:
            temporary_path.unlink(
                missing_ok=True
            )

        except OSError:
            pass

        raise MLDSAKeyManagerError(
            "Unable to save the ML-DSA key file.",
            details={
                "path": str(path),
                "reason": str(exc),
            },
        ) from exc


def _read_json_file(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object from disk."""

    if not path.exists():
        raise MLDSAKeyManagerError(
            "ML-DSA key file does not exist.",
            details={
                "path": str(path),
            },
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as exc:
        raise MLDSAKeyManagerError(
            "Unable to read the ML-DSA key file.",
            details={
                "path": str(path),
                "reason": str(exc),
            },
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise MLDSAKeyManagerError(
            "ML-DSA key file must contain a JSON object.",
            details={
                "path": str(path),
            },
        )

    return data


# ---------------------------------------------------------------------
# Main key manager
# ---------------------------------------------------------------------

class MLDSAKeyManager:
    """
    Thread-safe Authentication Server ML-DSA key manager.
    """

    def __init__(
        self,
        *,
        public_key_path: Path | str = (
            DEFAULT_MLDSA_PUBLIC_KEY_PATH
        ),
        secret_key_path: Path | str = (
            DEFAULT_MLDSA_SECRET_KEY_PATH
        ),
        algorithm: str = ML_DSA_ALGORITHM,
    ) -> None:
        self.public_key_path = Path(
            public_key_path
        )

        self.secret_key_path = Path(
            secret_key_path
        )

        self.algorithm = (
            normalize_mldsa_algorithm(
                algorithm
            )
        )

        if (
            self.public_key_path.resolve()
            == self.secret_key_path.resolve()
        ):
            raise ProtocolValidationError(
                (
                    "ML-DSA public and secret keys must "
                    "use different files."
                )
            )

        self._key_pair: (
            ManagedMLDSAKeyPair | None
        ) = None

        self._lock = threading.RLock()

    @property
    def keys_loaded(self) -> bool:
        """Return True when a key pair is loaded in memory."""

        with self._lock:
            return self._key_pair is not None

    def key_files_exist(self) -> bool:
        """Return True when both key files exist."""

        return (
            self.public_key_path.exists()
            and self.secret_key_path.exists()
        )

    def generate_keypair(
        self,
        *,
        overwrite: bool = False,
    ) -> ManagedMLDSAKeyPair:
        """
        Generate and save a new Authentication Server key pair.

        Existing key files are protected unless overwrite=True.
        """

        if not isinstance(
            overwrite,
            bool,
        ):
            raise ProtocolValidationError(
                "overwrite must be Boolean."
            )

        with self._lock:
            if (
                not overwrite
                and (
                    self.public_key_path.exists()
                    or self.secret_key_path.exists()
                )
            ):
                raise MLDSAKeyManagerError(
                    (
                        "ML-DSA key files already exist. "
                        "Use rotation or overwrite=True."
                    ),
                    details={
                        "public_key_path": str(
                            self.public_key_path
                        ),
                        "secret_key_path": str(
                            self.secret_key_path
                        ),
                    },
                )

            public_key, secret_key = (
                _backend_generate_keypair(
                    self.algorithm
                )
            )

            created_at = current_timestamp()

            key_pair = ManagedMLDSAKeyPair(
                algorithm=self.algorithm,
                public_key=public_key,
                secret_key=secret_key,
                key_id=create_mldsa_key_id(
                    public_key
                ),
                fingerprint=(
                    calculate_public_key_fingerprint(
                        public_key
                    )
                ),
                created_at=created_at,
            )

            _write_json_atomically(
                self.public_key_path,
                key_pair.public_dict(),
                permission=(
                    PUBLIC_KEY_FILE_PERMISSION
                ),
            )

            _write_json_atomically(
                self.secret_key_path,
                key_pair.private_dict(),
                permission=(
                    SECRET_KEY_FILE_PERMISSION
                ),
            )

            self._key_pair = key_pair

            return key_pair

    def load_keypair(
        self,
    ) -> ManagedMLDSAKeyPair:
        """
        Load and validate the persisted ML-DSA key pair.
        """

        with self._lock:
            public_record = _read_json_file(
                self.public_key_path
            )

            private_record = _read_json_file(
                self.secret_key_path
            )

            for record_name, record in (
                (
                    "public",
                    public_record,
                ),
                (
                    "private",
                    private_record,
                ),
            ):
                if (
                    record.get("version")
                    != MLDSA_KEY_FILE_VERSION
                ):
                    raise MLDSAKeyManagerError(
                        (
                            f"Unsupported {record_name} "
                            "ML-DSA key-file version."
                        ),
                        details={
                            "received_version": (
                                record.get(
                                    "version"
                                )
                            ),
                            "expected_version": (
                                MLDSA_KEY_FILE_VERSION
                            ),
                        },
                    )

            public_algorithm = (
                normalize_mldsa_algorithm(
                    public_record.get(
                        "algorithm",
                        "",
                    )
                )
            )

            private_algorithm = (
                normalize_mldsa_algorithm(
                    private_record.get(
                        "algorithm",
                        "",
                    )
                )
            )

            if (
                public_algorithm
                != private_algorithm
                or public_algorithm
                != self.algorithm
            ):
                raise MLDSAKeyManagerError(
                    "ML-DSA key-file algorithms do not match.",
                    details={
                        "manager_algorithm": self.algorithm,
                        "public_algorithm": public_algorithm,
                        "private_algorithm": private_algorithm,
                    },
                )

            try:
                public_key = decode_base64(
                    public_record[
                        "public_key"
                    ]
                )

                private_public_key = decode_base64(
                    private_record[
                        "public_key"
                    ]
                )

                secret_key = decode_base64(
                    private_record[
                        "secret_key"
                    ]
                )

            except Exception as exc:
                raise MLDSAKeyManagerError(
                    "Unable to decode stored ML-DSA key material.",
                    details={
                        "reason": str(exc),
                    },
                ) from exc

            if public_key != private_public_key:
                raise MLDSAKeyManagerError(
                    (
                        "Public key does not match the public "
                        "key stored with the private record."
                    )
                )

            key_id = validate_non_empty_string(
                public_record.get(
                    "key_id",
                    "",
                ),
                field_name="key_id",
                minimum_length=8,
                maximum_length=128,
            )

            private_key_id = (
                validate_non_empty_string(
                    private_record.get(
                        "key_id",
                        "",
                    ),
                    field_name="private_key_id",
                    minimum_length=8,
                    maximum_length=128,
                )
            )

            if key_id != private_key_id:
                raise MLDSAKeyManagerError(
                    "Public and private key IDs do not match."
                )

            fingerprint = (
                validate_non_empty_string(
                    public_record.get(
                        "fingerprint",
                        "",
                    ),
                    field_name="fingerprint",
                    minimum_length=64,
                    maximum_length=64,
                )
                .lower()
            )

            private_fingerprint = (
                validate_non_empty_string(
                    private_record.get(
                        "fingerprint",
                        "",
                    ),
                    field_name="private_fingerprint",
                    minimum_length=64,
                    maximum_length=64,
                )
                .lower()
            )

            if fingerprint != private_fingerprint:
                raise MLDSAKeyManagerError(
                    (
                        "Public and private key "
                        "fingerprints do not match."
                    )
                )

            created_at = validate_integer(
                public_record.get(
                    "created_at",
                    0,
                ),
                field_name="created_at",
                minimum=0,
            )

            private_created_at = (
                validate_integer(
                    private_record.get(
                        "created_at",
                        0,
                    ),
                    field_name=(
                        "private_created_at"
                    ),
                    minimum=0,
                )
            )

            if created_at != private_created_at:
                raise MLDSAKeyManagerError(
                    (
                        "Public and private key creation "
                        "timestamps do not match."
                    )
                )

            key_pair = ManagedMLDSAKeyPair(
                algorithm=self.algorithm,
                public_key=public_key,
                secret_key=secret_key,
                key_id=key_id,
                fingerprint=fingerprint,
                created_at=created_at,
            )

            self._key_pair = key_pair

            return key_pair

    def ensure_keypair(
        self,
    ) -> ManagedMLDSAKeyPair:
        """
        Return the loaded key pair, load it, or create it.

        A new key pair is generated only when neither key file exists.
        """

        with self._lock:
            if self._key_pair is not None:
                return self._key_pair

            public_exists = (
                self.public_key_path.exists()
            )

            secret_exists = (
                self.secret_key_path.exists()
            )

            if (
                public_exists
                and secret_exists
            ):
                return self.load_keypair()

            if (
                public_exists
                != secret_exists
            ):
                raise MLDSAKeyManagerError(
                    (
                        "Only one ML-DSA key file exists. "
                        "Automatic regeneration was blocked."
                    ),
                    details={
                        "public_key_exists": public_exists,
                        "secret_key_exists": secret_exists,
                    },
                )

            return self.generate_keypair()

    def unload_keypair(self) -> None:
        """
        Remove the key-pair reference from this manager instance.
        """

        with self._lock:
            self._key_pair = None

    def get_public_key(
        self,
    ) -> bytes:
        """Return the server ML-DSA public key."""

        return bytes(
            self.ensure_keypair().public_key
        )

    def get_key_id(
        self,
    ) -> str:
        """Return the active ML-DSA key identifier."""

        return self.ensure_keypair().key_id

    def get_public_information(
        self,
    ) -> dict[str, Any]:
        """Return non-sensitive active-key information."""

        key_pair = self.ensure_keypair()

        return {
            "algorithm": key_pair.algorithm,
            "key_id": key_pair.key_id,
            "fingerprint": key_pair.fingerprint,
            "created_at": key_pair.created_at,
            "public_key_bytes": len(
                key_pair.public_key
            ),
            "public_key_path": str(
                self.public_key_path
            ),
        }

    def sign(
        self,
        message: bytes,
        *,
        context: bytes = MLDSA_SIGNING_CONTEXT,
    ) -> bytes:
        """
        Sign a domain-separated server message.
        """

        key_pair = self.ensure_keypair()

        signing_message = (
            build_mldsa_signing_message(
                message=message,
                key_id=key_pair.key_id,
                algorithm=key_pair.algorithm,
                context=context,
            )
        )

        return _backend_sign(
            message=signing_message,
            secret_key=key_pair.secret_key,
            algorithm=key_pair.algorithm,
        )

    def verify(
        self,
        message: bytes,
        signature: bytes,
        *,
        public_key: bytes | None = None,
        context: bytes = MLDSA_SIGNING_CONTEXT,
    ) -> bool:
        """
        Verify a server ML-DSA signature.

        An external public key may be supplied for testing or historical
        key verification. Otherwise, the active public key is used.
        """

        validated_message = validate_bytes(
            message,
            field_name="message",
            minimum_length=1,
            maximum_length=10_000_000,
        )

        validated_signature = validate_bytes(
            signature,
            field_name="mldsa_signature",
            minimum_length=32,
            maximum_length=100_000,
        )

        if public_key is None:
            key_pair = self.ensure_keypair()

            selected_public_key = (
                key_pair.public_key
            )

            selected_key_id = (
                key_pair.key_id
            )

            selected_algorithm = (
                key_pair.algorithm
            )

        else:
            selected_public_key = validate_bytes(
                public_key,
                field_name="mldsa_public_key",
                minimum_length=32,
                maximum_length=100_000,
            )

            selected_key_id = (
                create_mldsa_key_id(
                    selected_public_key
                )
            )

            selected_algorithm = (
                self.algorithm
            )

        signing_message = (
            build_mldsa_signing_message(
                message=validated_message,
                key_id=selected_key_id,
                algorithm=selected_algorithm,
                context=context,
            )
        )

        return _backend_verify(
            message=signing_message,
            signature=validated_signature,
            public_key=selected_public_key,
            algorithm=selected_algorithm,
        )

    def rotate_keypair(
        self,
    ) -> MLDSAKeyRotationResult:
        """
        Replace the active server ML-DSA key pair.

        Production deployments should publish the new public key through
        a trusted provisioning or certificate mechanism.
        """

        with self._lock:
            previous_key_id: str | None = None
            previous_fingerprint: str | None = None

            if self.key_files_exist():
                previous = self.load_keypair()

                previous_key_id = (
                    previous.key_id
                )

                previous_fingerprint = (
                    previous.fingerprint
                )

            new_key_pair = (
                self.generate_keypair(
                    overwrite=True
                )
            )

            return MLDSAKeyRotationResult(
                previous_key_id=(
                    previous_key_id
                ),
                previous_fingerprint=(
                    previous_fingerprint
                ),
                new_key_id=(
                    new_key_pair.key_id
                ),
                new_fingerprint=(
                    new_key_pair.fingerprint
                ),
                rotated_at=current_timestamp(),
            )


# ---------------------------------------------------------------------
# Default manager
# ---------------------------------------------------------------------

_DEFAULT_MANAGER: MLDSAKeyManager | None = None

_DEFAULT_MANAGER_LOCK = threading.RLock()


def get_default_mldsa_key_manager() -> MLDSAKeyManager:
    """
    Return the process-wide Authentication Server key manager.
    """

    global _DEFAULT_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = (
                MLDSAKeyManager()
            )

        return _DEFAULT_MANAGER


def reset_default_mldsa_key_manager() -> None:
    """Reset the process-wide key-manager instance."""

    global _DEFAULT_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        _DEFAULT_MANAGER = None


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_mldsa_key_manager_self_test() -> dict[str, Any]:
    """
    Run key generation, persistence, signing, and verification tests.
    """

    try:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            manager = MLDSAKeyManager(
                public_key_path=(
                    root / "public.json"
                ),
                secret_key_path=(
                    root / "secret.json"
                ),
            )

            generated = (
                manager.generate_keypair()
            )

            message = (
                b"FT-QuPAP ML-DSA key manager self-test"
            )

            signature = manager.sign(
                message
            )

            valid_signature = manager.verify(
                message,
                signature,
            )

            tampered_message_rejected = (
                not manager.verify(
                    message + b"-tampered",
                    signature,
                )
            )

            public_file_exists = (
                manager.public_key_path.exists()
            )

            secret_file_exists = (
                manager.secret_key_path.exists()
            )

            original_fingerprint = (
                generated.fingerprint
            )

            manager.unload_keypair()

            reloaded = manager.load_keypair()

            reload_pass = (
                reloaded.fingerprint
                == original_fingerprint
                and reloaded.public_key
                == generated.public_key
                and reloaded.secret_key
                == generated.secret_key
            )

            success = all(
                (
                    valid_signature,
                    tampered_message_rejected,
                    public_file_exists,
                    secret_file_exists,
                    reload_pass,
                )
            )

            return {
                "success": success,
                "algorithm": generated.algorithm,
                "key_id": generated.key_id,
                "fingerprint": generated.fingerprint,
                "public_key_bytes": len(
                    generated.public_key
                ),
                "secret_key_bytes": len(
                    generated.secret_key
                ),
                "signature_bytes": len(
                    signature
                ),
                "valid_signature": (
                    valid_signature
                ),
                "tampered_message_rejected": (
                    tampered_message_rejected
                ),
                "public_file_exists": (
                    public_file_exists
                ),
                "secret_file_exists": (
                    secret_file_exists
                ),
                "reload_pass": reload_pass,
            }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(
                exc
            ).__name__,
        }


__all__ = [
    "MLDSA_KEY_FILE_VERSION",
    "MLDSA_SIGNING_CONTEXT",
    "DEFAULT_MLDSA_PUBLIC_KEY_PATH",
    "DEFAULT_MLDSA_SECRET_KEY_PATH",
    "MLDSAKeyManagerError",
    "ManagedMLDSAKeyPair",
    "MLDSAKeyRotationResult",
    "normalize_mldsa_algorithm",
    "calculate_public_key_fingerprint",
    "create_mldsa_key_id",
    "build_mldsa_signing_message",
    "MLDSAKeyManager",
    "get_default_mldsa_key_manager",
    "reset_default_mldsa_key_manager",
    "run_mldsa_key_manager_self_test",
]