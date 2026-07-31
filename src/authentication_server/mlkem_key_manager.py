"""
Ephemeral ML-KEM key management for the FT-QuPAP v5.1
Authentication Server.

For every authentication session or retry attempt, the Authentication
Server generates a fresh ML-KEM key pair.

The public encapsulation key is included in the signed M2 server
package. The secret decapsulation key remains only inside the
Authentication Server and is used when the M3 ciphertext arrives.

This module provides:

- Session-bound ephemeral ML-KEM key generation
- Attempt-number binding
- Public-key fingerprinting
- Expiration enforcement
- One-time key-consumption tracking
- Thread-safe key retrieval and deletion
- Secret-safe public representations

Ephemeral ML-KEM secret keys are never written to normal log output.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.authentication_server.mlkem_decapsulation import (
    normalize_mlkem_algorithm,
)

from src.common.constants import (
    ML_KEM_ALGORITHM,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.serialization import (
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

from src.cryptography import mlkem_module


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MLKEM_KEY_TTL_SECONDS = 120

MINIMUM_MLKEM_KEY_TTL_SECONDS = 10

MAXIMUM_MLKEM_KEY_TTL_SECONDS = 3600

MLKEM_PUBLIC_KEY_FINGERPRINT_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class MLKEMKeyManagerError(RuntimeError):
    """Raised when ephemeral ML-KEM key management fails."""

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
# Ephemeral key-pair model
# ---------------------------------------------------------------------

@dataclass
class EphemeralMLKEMKeyPair:
    """
    Session-bound Authentication Server ML-KEM key pair.

    Attributes
    ----------
    session_id:
        Authentication session to which this key pair belongs.

    attempt_number:
        Current authentication or retry attempt.

    public_key:
        ML-KEM encapsulation key included in the signed M2 package.

    secret_key:
        ML-KEM decapsulation key retained by the server.

    consumed:
        Indicates that the secret key has already been used for a
        completed decapsulation attempt.
    """

    session_id: str
    attempt_number: int

    algorithm: str

    public_key: bytes
    secret_key: bytes

    key_id: str
    public_key_fingerprint: str

    created_at: int
    expires_at: int

    consumed: bool = False
    consumed_at: int | None = None

    def __post_init__(self) -> None:
        self.session_id = validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=128,
        )

        self.attempt_number = validate_integer(
            self.attempt_number,
            field_name="attempt_number",
            minimum=1,
            maximum=100,
        )

        self.algorithm = normalize_mlkem_algorithm(
            self.algorithm
        )

        self.public_key = validate_bytes(
            self.public_key,
            field_name="mlkem_public_key",
            minimum_length=32,
            maximum_length=100_000,
        )

        self.secret_key = validate_bytes(
            self.secret_key,
            field_name="mlkem_secret_key",
            minimum_length=32,
            maximum_length=200_000,
        )

        self.key_id = validate_non_empty_string(
            self.key_id,
            field_name="mlkem_key_id",
            minimum_length=8,
            maximum_length=128,
        )

        self.public_key_fingerprint = (
            validate_non_empty_string(
                self.public_key_fingerprint,
                field_name=(
                    "public_key_fingerprint"
                ),
                minimum_length=64,
                maximum_length=64,
            ).lower()
        )

        self.created_at = validate_integer(
            self.created_at,
            field_name="created_at",
            minimum=0,
        )

        self.expires_at = validate_integer(
            self.expires_at,
            field_name="expires_at",
            minimum=0,
        )

        if self.expires_at <= self.created_at:
            raise ProtocolValidationError(
                (
                    "ML-KEM key expiration time must be "
                    "later than its creation time."
                )
            )

        if not isinstance(
            self.consumed,
            bool,
        ):
            raise ProtocolValidationError(
                "consumed must be Boolean."
            )

        if self.consumed_at is not None:
            self.consumed_at = validate_integer(
                self.consumed_at,
                field_name="consumed_at",
                minimum=0,
            )

        expected_fingerprint = (
            calculate_mlkem_public_key_fingerprint(
                self.public_key
            )
        )

        if (
            self.public_key_fingerprint
            != expected_fingerprint
        ):
            raise ProtocolValidationError(
                (
                    "ML-KEM public-key fingerprint does "
                    "not match the public key."
                ),
                details={
                    "received_fingerprint": (
                        self.public_key_fingerprint
                    ),
                    "expected_fingerprint": (
                        expected_fingerprint
                    ),
                },
            )

        expected_key_id = create_mlkem_key_id(
            public_key=self.public_key,
            session_id=self.session_id,
            attempt_number=self.attempt_number,
        )

        if self.key_id != expected_key_id:
            raise ProtocolValidationError(
                (
                    "ML-KEM key ID does not match its "
                    "session, attempt, and public key."
                ),
                details={
                    "received_key_id": self.key_id,
                    "expected_key_id": expected_key_id,
                },
            )

    def is_expired(
        self,
        *,
        timestamp: int | None = None,
    ) -> bool:
        """Return True when the ephemeral key has expired."""

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        return (
            selected_timestamp
            >= self.expires_at
        )

    def is_usable(
        self,
        *,
        timestamp: int | None = None,
    ) -> bool:
        """
        Return True when the key is unexpired and unconsumed.
        """

        return (
            not self.consumed
            and not self.is_expired(
                timestamp=timestamp
            )
        )

    def public_dict(self) -> dict[str, Any]:
        """
        Return the public part used by the M2 server package.
        """

        return {
            "session_id": self.session_id,
            "attempt_number": self.attempt_number,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "public_key_fingerprint": (
                self.public_key_fingerprint
            ),
            "public_key": encode_base64(
                self.public_key
            ),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    def diagnostic_dict(self) -> dict[str, Any]:
        """
        Return non-secret server diagnostic information.
        """

        return {
            "session_id": self.session_id,
            "attempt_number": self.attempt_number,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "public_key_fingerprint": (
                self.public_key_fingerprint
            ),
            "public_key_bytes": len(
                self.public_key
            ),
            "secret_key_bytes": len(
                self.secret_key
            ),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "consumed": self.consumed,
            "consumed_at": self.consumed_at,
        }

    def __repr__(self) -> str:
        return (
            "EphemeralMLKEMKeyPair("
            f"session_id={self.session_id!r}, "
            f"attempt_number={self.attempt_number}, "
            f"algorithm={self.algorithm!r}, "
            f"key_id={self.key_id!r}, "
            f"public_key_fingerprint="
            f"{self.public_key_fingerprint!r}, "
            f"public_key_bytes={len(self.public_key)}, "
            f"created_at={self.created_at}, "
            f"expires_at={self.expires_at}, "
            f"consumed={self.consumed}, "
            "secret_key=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Fingerprint and key-ID helpers
# ---------------------------------------------------------------------

def calculate_mlkem_public_key_fingerprint(
    public_key: bytes,
) -> str:
    """
    Calculate the SHA3-256 ML-KEM public-key fingerprint.
    """

    validated_public_key = validate_bytes(
        public_key,
        field_name="mlkem_public_key",
        minimum_length=32,
        maximum_length=100_000,
    )

    return hashlib.sha3_256(
        validated_public_key
    ).hexdigest()


def create_mlkem_key_id(
    *,
    public_key: bytes,
    session_id: str,
    attempt_number: int,
) -> str:
    """
    Create a session-bound ML-KEM key identifier.

    The key ID binds:

    - Public key
    - Session identifier
    - Authentication-attempt number
    """

    validated_public_key = validate_bytes(
        public_key,
        field_name="mlkem_public_key",
        minimum_length=32,
        maximum_length=100_000,
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

    digest = hashlib.sha3_256()

    digest.update(
        validated_session_id.encode(
            "utf-8"
        )
    )

    digest.update(
        validated_attempt.to_bytes(
            4,
            byteorder="big",
            signed=False,
        )
    )

    digest.update(
        validated_public_key
    )

    return (
        "MLKEM-"
        + digest.hexdigest()[:32].upper()
    )


# ---------------------------------------------------------------------
# Backend compatibility helpers
# ---------------------------------------------------------------------

def _find_key_generation_function() -> Callable[..., Any]:
    """
    Find the ML-KEM key-generation function.
    """

    function_names = (
        "generate_mlkem_keypair",
        "generate_keypair",
        "mlkem_generate_keypair",
        "mlkem_keygen",
    )

    for function_name in function_names:
        function = getattr(
            mlkem_module,
            function_name,
            None,
        )

        if callable(function):
            return function

    raise MLKEMKeyManagerError(
        "No ML-KEM key-generation function was found.",
        details={
            "module": (
                "src.cryptography.mlkem_module"
            ),
            "searched_functions": list(
                function_names
            ),
        },
    )


def _invoke_compatible_keygen(
    function: Callable[..., Any],
    algorithm: str,
) -> Any:
    """
    Invoke the ML-KEM backend using a compatible signature.
    """

    calls: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ] = [
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
    ]

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

        try:
            return function(
                *args,
                **kwargs,
            )

        except Exception as exc:
            raise MLKEMKeyManagerError(
                "ML-KEM backend key generation failed.",
                details={
                    "function": getattr(
                        function,
                        "__name__",
                        repr(function),
                    ),
                    "reason": str(exc),
                },
            ) from exc

    raise MLKEMKeyManagerError(
        (
            "Unable to call the ML-KEM key-generation "
            "function with a compatible signature."
        ),
        details={
            "binding_errors": binding_errors,
        },
    )


def _normalize_generated_keypair(
    generated: Any,
) -> tuple[bytes, bytes]:
    """
    Normalize common ML-KEM key-generation return formats.

    Supported forms:

    - (public_key, secret_key)
    - Dictionary
    - Object with public and secret key attributes
    """

    public_key: Any = None
    secret_key: Any = None

    if isinstance(
        generated,
        Mapping,
    ):
        public_key = generated.get(
            "public_key",
            generated.get(
                "encapsulation_key"
            ),
        )

        secret_key = generated.get(
            "secret_key",
            generated.get(
                "decapsulation_key",
                generated.get(
                    "private_key"
                ),
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
            getattr(
                generated,
                "encapsulation_key",
                None,
            ),
        )

        secret_key = getattr(
            generated,
            "secret_key",
            getattr(
                generated,
                "decapsulation_key",
                getattr(
                    generated,
                    "private_key",
                    None,
                ),
            ),
        )

    if (
        public_key is None
        or secret_key is None
    ):
        raise MLKEMKeyManagerError(
            (
                "ML-KEM backend returned an invalid "
                "key-pair structure."
            ),
            details={
                "received_type": type(
                    generated
                ).__name__,
            },
        )

    try:
        normalized_public_key = bytes(
            public_key
        )

        normalized_secret_key = bytes(
            secret_key
        )

    except Exception as exc:
        raise MLKEMKeyManagerError(
            "Unable to convert ML-KEM key material to bytes.",
            details={
                "reason": str(exc),
            },
        ) from exc

    return (
        validate_bytes(
            normalized_public_key,
            field_name="mlkem_public_key",
            minimum_length=32,
            maximum_length=100_000,
        ),
        validate_bytes(
            normalized_secret_key,
            field_name="mlkem_secret_key",
            minimum_length=32,
            maximum_length=200_000,
        ),
    )


def generate_mlkem_key_material(
    *,
    algorithm: str = ML_KEM_ALGORITHM,
) -> tuple[bytes, bytes]:
    """
    Generate raw ML-KEM public and secret key bytes.
    """

    normalized_algorithm = (
        normalize_mlkem_algorithm(
            algorithm
        )
    )

    function = (
        _find_key_generation_function()
    )

    generated = _invoke_compatible_keygen(
        function,
        normalized_algorithm,
    )

    return _normalize_generated_keypair(
        generated
    )


# ---------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------

class MLKEMKeyManager:
    """
    Thread-safe manager for ephemeral session-bound ML-KEM keys.
    """

    def __init__(
        self,
        *,
        algorithm: str = ML_KEM_ALGORITHM,
        default_ttl_seconds: int = (
            DEFAULT_MLKEM_KEY_TTL_SECONDS
        ),
    ) -> None:
        self.algorithm = normalize_mlkem_algorithm(
            algorithm
        )

        self.default_ttl_seconds = (
            validate_integer(
                default_ttl_seconds,
                field_name=(
                    "default_ttl_seconds"
                ),
                minimum=(
                    MINIMUM_MLKEM_KEY_TTL_SECONDS
                ),
                maximum=(
                    MAXIMUM_MLKEM_KEY_TTL_SECONDS
                ),
            )
        )

        self._keys: dict[
            tuple[str, int],
            EphemeralMLKEMKeyPair,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _record_key(
        session_id: str,
        attempt_number: int,
    ) -> tuple[str, int]:
        """Create the internal dictionary key."""

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

        return (
            validated_session_id,
            validated_attempt,
        )

    def generate_ephemeral_keypair(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
        ttl_seconds: int | None = None,
        replace: bool = False,
        generated_at: int | None = None,
    ) -> EphemeralMLKEMKeyPair:
        """
        Generate a fresh ML-KEM key pair for one attempt.

        Existing unexpired key material is protected unless
        `replace=True`.
        """

        if not isinstance(
            replace,
            bool,
        ):
            raise ProtocolValidationError(
                "replace must be Boolean."
            )

        record_key = self._record_key(
            session_id,
            attempt_number,
        )

        selected_ttl = (
            self.default_ttl_seconds
            if ttl_seconds is None
            else validate_integer(
                ttl_seconds,
                field_name="ttl_seconds",
                minimum=(
                    MINIMUM_MLKEM_KEY_TTL_SECONDS
                ),
                maximum=(
                    MAXIMUM_MLKEM_KEY_TTL_SECONDS
                ),
            )
        )

        selected_created_at = (
            current_timestamp()
            if generated_at is None
            else validate_integer(
                generated_at,
                field_name="generated_at",
                minimum=0,
            )
        )

        with self._lock:
            existing = self._keys.get(
                record_key
            )

            if (
                existing is not None
                and not replace
                and not existing.is_expired(
                    timestamp=selected_created_at
                )
            ):
                raise MLKEMKeyManagerError(
                    (
                        "An active ML-KEM key pair already "
                        "exists for this session attempt."
                    ),
                    details={
                        "session_id": record_key[0],
                        "attempt_number": record_key[1],
                        "key_id": existing.key_id,
                    },
                )

            public_key, secret_key = (
                generate_mlkem_key_material(
                    algorithm=self.algorithm
                )
            )

            fingerprint = (
                calculate_mlkem_public_key_fingerprint(
                    public_key
                )
            )

            key_id = create_mlkem_key_id(
                public_key=public_key,
                session_id=record_key[0],
                attempt_number=record_key[1],
            )

            key_pair = EphemeralMLKEMKeyPair(
                session_id=record_key[0],
                attempt_number=record_key[1],
                algorithm=self.algorithm,
                public_key=public_key,
                secret_key=secret_key,
                key_id=key_id,
                public_key_fingerprint=(
                    fingerprint
                ),
                created_at=selected_created_at,
                expires_at=(
                    selected_created_at
                    + selected_ttl
                ),
            )

            self._keys[
                record_key
            ] = key_pair

            return key_pair

    def get_keypair(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
        timestamp: int | None = None,
        require_unconsumed: bool = True,
    ) -> EphemeralMLKEMKeyPair:
        """
        Retrieve and validate one ephemeral key pair.
        """

        if not isinstance(
            require_unconsumed,
            bool,
        ):
            raise ProtocolValidationError(
                "require_unconsumed must be Boolean."
            )

        record_key = self._record_key(
            session_id,
            attempt_number,
        )

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        with self._lock:
            key_pair = self._keys.get(
                record_key
            )

            if key_pair is None:
                raise MLKEMKeyManagerError(
                    (
                        "No ML-KEM key pair exists for "
                        "this session attempt."
                    ),
                    details={
                        "session_id": record_key[0],
                        "attempt_number": record_key[1],
                    },
                )

            if key_pair.is_expired(
                timestamp=selected_timestamp
            ):
                self._keys.pop(
                    record_key,
                    None,
                )

                raise MLKEMKeyManagerError(
                    (
                        "The ephemeral ML-KEM key pair "
                        "has expired."
                    ),
                    details={
                        "session_id": record_key[0],
                        "attempt_number": record_key[1],
                        "expired_at": key_pair.expires_at,
                        "current_timestamp": (
                            selected_timestamp
                        ),
                    },
                )

            if (
                require_unconsumed
                and key_pair.consumed
            ):
                raise MLKEMKeyManagerError(
                    (
                        "The ephemeral ML-KEM secret key "
                        "has already been consumed."
                    ),
                    details={
                        "session_id": record_key[0],
                        "attempt_number": record_key[1],
                        "consumed_at": (
                            key_pair.consumed_at
                        ),
                    },
                )

            return key_pair

    def get_public_package(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
    ) -> dict[str, Any]:
        """
        Return the public ML-KEM package for M2 signing.
        """

        key_pair = self.get_keypair(
            session_id=session_id,
            attempt_number=attempt_number,
            require_unconsumed=False,
        )

        return key_pair.public_dict()

    def get_public_key(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
    ) -> bytes:
        """Return a copy of the public encapsulation key."""

        return bytes(
            self.get_keypair(
                session_id=session_id,
                attempt_number=attempt_number,
                require_unconsumed=False,
            ).public_key
        )

    def get_secret_key(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
    ) -> bytes:
        """
        Return a copy of the unconsumed secret decapsulation key.

        This method is for internal Authentication Server use only.
        """

        return bytes(
            self.get_keypair(
                session_id=session_id,
                attempt_number=attempt_number,
                require_unconsumed=True,
            ).secret_key
        )

    def mark_consumed(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
        consumed_at: int | None = None,
    ) -> EphemeralMLKEMKeyPair:
        """
        Mark a secret key as successfully consumed.

        Call this only after ML-KEM decapsulation and required
        consistency checks have succeeded.
        """

        selected_consumed_at = (
            current_timestamp()
            if consumed_at is None
            else validate_integer(
                consumed_at,
                field_name="consumed_at",
                minimum=0,
            )
        )

        with self._lock:
            key_pair = self.get_keypair(
                session_id=session_id,
                attempt_number=attempt_number,
                timestamp=selected_consumed_at,
                require_unconsumed=True,
            )

            key_pair.consumed = True
            key_pair.consumed_at = (
                selected_consumed_at
            )

            return key_pair

    def delete_keypair(
        self,
        *,
        session_id: str,
        attempt_number: int = 1,
    ) -> bool:
        """
        Delete a session-attempt key-pair reference.

        Python byte objects cannot be reliably overwritten in place, but
        removing references allows the runtime to reclaim the material.
        """

        record_key = self._record_key(
            session_id,
            attempt_number,
        )

        with self._lock:
            return (
                self._keys.pop(
                    record_key,
                    None,
                )
                is not None
            )

    def delete_session_keys(
        self,
        session_id: str,
    ) -> int:
        """
        Delete all ML-KEM key pairs associated with a session.
        """

        validated_session_id = (
            validate_non_empty_string(
                session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=128,
            )
        )

        with self._lock:
            matching_keys = [
                record_key
                for record_key
                in self._keys
                if record_key[0]
                == validated_session_id
            ]

            for record_key in matching_keys:
                self._keys.pop(
                    record_key,
                    None,
                )

            return len(
                matching_keys
            )

    def purge_expired(
        self,
        *,
        timestamp: int | None = None,
    ) -> int:
        """
        Delete all expired ephemeral key pairs.
        """

        selected_timestamp = (
            current_timestamp()
            if timestamp is None
            else validate_integer(
                timestamp,
                field_name="timestamp",
                minimum=0,
            )
        )

        with self._lock:
            expired_keys = [
                record_key
                for record_key, key_pair
                in self._keys.items()
                if key_pair.is_expired(
                    timestamp=selected_timestamp
                )
            ]

            for record_key in expired_keys:
                self._keys.pop(
                    record_key,
                    None,
                )

            return len(
                expired_keys
            )

    def active_key_count(
        self,
        *,
        timestamp: int | None = None,
    ) -> int:
        """
        Return the number of unexpired stored key pairs.
        """

        self.purge_expired(
            timestamp=timestamp
        )

        with self._lock:
            return len(
                self._keys
            )

    def list_key_information(
        self,
        *,
        timestamp: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return non-secret information for active key pairs.
        """

        self.purge_expired(
            timestamp=timestamp
        )

        with self._lock:
            return [
                key_pair.diagnostic_dict()
                for key_pair
                in self._keys.values()
            ]


# ---------------------------------------------------------------------
# Default process-wide manager
# ---------------------------------------------------------------------

_DEFAULT_MLKEM_KEY_MANAGER: MLKEMKeyManager | None = None

_DEFAULT_MANAGER_LOCK = threading.RLock()


def get_default_mlkem_key_manager() -> MLKEMKeyManager:
    """
    Return the process-wide ephemeral ML-KEM key manager.
    """

    global _DEFAULT_MLKEM_KEY_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        if (
            _DEFAULT_MLKEM_KEY_MANAGER
            is None
        ):
            _DEFAULT_MLKEM_KEY_MANAGER = (
                MLKEMKeyManager()
            )

        return _DEFAULT_MLKEM_KEY_MANAGER


def reset_default_mlkem_key_manager() -> None:
    """Reset the process-wide ML-KEM manager."""

    global _DEFAULT_MLKEM_KEY_MANAGER

    with _DEFAULT_MANAGER_LOCK:
        _DEFAULT_MLKEM_KEY_MANAGER = None


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_mlkem_key_manager_self_test() -> dict[str, Any]:
    """
    Test generation, session binding, consumption, and expiration.
    """

    try:
        manager = MLKEMKeyManager(
            default_ttl_seconds=60
        )

        key_pair = (
            manager.generate_ephemeral_keypair(
                session_id=(
                    "FTQ-MLKEM-KEY-SELF-TEST"
                ),
                attempt_number=1,
                generated_at=1_700_000_000,
                ttl_seconds=60,
            )
        )

        public_package = (
            key_pair.public_dict()
        )

        public_key_retrieved = (
            manager.get_public_key(
                session_id=(
                    "FTQ-MLKEM-KEY-SELF-TEST"
                ),
                attempt_number=1,
            )
            == key_pair.public_key
        )

        secret_key_retrieved = (
            manager.get_secret_key(
                session_id=(
                    "FTQ-MLKEM-KEY-SELF-TEST"
                ),
                attempt_number=1,
            )
            == key_pair.secret_key
        )

        duplicate_rejected = False

        try:
            manager.generate_ephemeral_keypair(
                session_id=(
                    "FTQ-MLKEM-KEY-SELF-TEST"
                ),
                attempt_number=1,
                generated_at=1_700_000_001,
                ttl_seconds=60,
            )

        except MLKEMKeyManagerError:
            duplicate_rejected = True

        manager.mark_consumed(
            session_id=(
                "FTQ-MLKEM-KEY-SELF-TEST"
            ),
            attempt_number=1,
            consumed_at=1_700_000_010,
        )

        reuse_rejected = False

        try:
            manager.get_secret_key(
                session_id=(
                    "FTQ-MLKEM-KEY-SELF-TEST"
                ),
                attempt_number=1,
            )

        except MLKEMKeyManagerError:
            reuse_rejected = True

        manager.generate_ephemeral_keypair(
            session_id=(
                "FTQ-MLKEM-EXPIRED-SELF-TEST"
            ),
            attempt_number=1,
            generated_at=1_700_000_000,
            ttl_seconds=60,
        )

        expired_removed = (
            manager.purge_expired(
                timestamp=1_700_000_061
            )
            >= 1
        )

        no_secret_in_public_package = (
            "secret_key"
            not in public_package
        )

        success = all(
            (
                public_key_retrieved,
                secret_key_retrieved,
                duplicate_rejected,
                key_pair.consumed,
                reuse_rejected,
                expired_removed,
                no_secret_in_public_package,
            )
        )

        return {
            "success": success,
            "algorithm": key_pair.algorithm,
            "key_id": key_pair.key_id,
            "public_key_bytes": len(
                key_pair.public_key
            ),
            "secret_key_bytes": len(
                key_pair.secret_key
            ),
            "public_key_retrieved": (
                public_key_retrieved
            ),
            "secret_key_retrieved": (
                secret_key_retrieved
            ),
            "duplicate_rejected": (
                duplicate_rejected
            ),
            "consumed": key_pair.consumed,
            "reuse_rejected": (
                reuse_rejected
            ),
            "expired_removed": (
                expired_removed
            ),
            "no_secret_in_public_package": (
                no_secret_in_public_package
            ),
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
    "DEFAULT_MLKEM_KEY_TTL_SECONDS",
    "MINIMUM_MLKEM_KEY_TTL_SECONDS",
    "MAXIMUM_MLKEM_KEY_TTL_SECONDS",
    "MLKEM_PUBLIC_KEY_FINGERPRINT_ALGORITHM",
    "MLKEMKeyManagerError",
    "EphemeralMLKEMKeyPair",
    "calculate_mlkem_public_key_fingerprint",
    "create_mlkem_key_id",
    "generate_mlkem_key_material",
    "MLKEMKeyManager",
    "get_default_mlkem_key_manager",
    "reset_default_mlkem_key_manager",
    "run_mlkem_key_manager_self_test",
]