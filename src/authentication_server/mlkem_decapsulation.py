"""
ML-KEM decapsulation for the FT-QuPAP v5.1 Authentication Server.

After receiving the M3 message, the Authentication Server uses its
ephemeral ML-KEM secret key to decapsulate the Mobile Station's
ciphertext.

The resulting 256-bit shared secret is not used directly as an
authentication key. It is passed to the transcript-bound session-key
derivation module.

This module provides:

- ML-KEM algorithm normalization
- Ciphertext fingerprint verification
- Backend-compatible decapsulation
- Shared-secret validation
- Secret-safe result reporting
- Deterministic self-testing
"""

from __future__ import annotations

import hashlib
import hmac
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping

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

MLKEM_SHARED_SECRET_BYTES = 32

MLKEM_CIPHERTEXT_FINGERPRINT_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class MLKEMDecapsulationError(RuntimeError):
    """Raised when ML-KEM decapsulation cannot be completed safely."""

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
class MLKEMDecapsulationResult:
    """
    Successful Authentication Server ML-KEM decapsulation result.

    The shared secret is intentionally excluded from `public_dict()`
    and hidden from `repr()`.
    """

    algorithm: str
    shared_secret: bytes

    ciphertext_fingerprint: str
    ciphertext_bytes: int
    shared_secret_bytes: int

    decapsulated_at: int
    success: bool = True

    def __post_init__(self) -> None:
        normalize_mlkem_algorithm(
            self.algorithm
        )

        validate_bytes(
            self.shared_secret,
            field_name="shared_secret",
            exact_length=MLKEM_SHARED_SECRET_BYTES,
        )

        validate_non_empty_string(
            self.ciphertext_fingerprint,
            field_name="ciphertext_fingerprint",
            minimum_length=64,
            maximum_length=64,
        )

        validate_integer(
            self.ciphertext_bytes,
            field_name="ciphertext_bytes",
            minimum=1,
        )

        validate_integer(
            self.shared_secret_bytes,
            field_name="shared_secret_bytes",
            minimum=1,
        )

        validate_integer(
            self.decapsulated_at,
            field_name="decapsulated_at",
            minimum=0,
        )

        if not isinstance(
            self.success,
            bool,
        ):
            raise ProtocolValidationError(
                "success must be Boolean."
            )

        if not self.success:
            raise ProtocolValidationError(
                (
                    "MLKEMDecapsulationResult can only represent "
                    "successful decapsulation."
                )
            )

        if (
            self.shared_secret_bytes
            != len(self.shared_secret)
        ):
            raise ProtocolValidationError(
                (
                    "shared_secret_bytes does not match "
                    "the actual shared-secret length."
                )
            )

    def public_dict(self) -> dict[str, Any]:
        """
        Return non-secret decapsulation information.
        """

        return {
            "success": self.success,
            "algorithm": self.algorithm,
            "ciphertext_fingerprint": (
                self.ciphertext_fingerprint
            ),
            "ciphertext_bytes": (
                self.ciphertext_bytes
            ),
            "shared_secret_bytes": (
                self.shared_secret_bytes
            ),
            "decapsulated_at": (
                self.decapsulated_at
            ),
        }

    def protected_dict(self) -> dict[str, Any]:
        """
        Return the full result for protected internal processing.

        This output must not be logged or sent over the network.
        """

        result = self.public_dict()

        result["shared_secret"] = encode_base64(
            self.shared_secret
        )

        return result

    def get_shared_secret(self) -> bytes:
        """
        Return an independent copy of the shared secret.
        """

        return bytes(
            self.shared_secret
        )

    def __repr__(self) -> str:
        return (
            "MLKEMDecapsulationResult("
            f"algorithm={self.algorithm!r}, "
            f"ciphertext_fingerprint="
            f"{self.ciphertext_fingerprint!r}, "
            f"ciphertext_bytes={self.ciphertext_bytes}, "
            f"shared_secret_bytes={self.shared_secret_bytes}, "
            f"decapsulated_at={self.decapsulated_at}, "
            f"success={self.success}, "
            "shared_secret=<hidden>"
            ")"
        )


# ---------------------------------------------------------------------
# Algorithm normalization
# ---------------------------------------------------------------------

def normalize_mlkem_algorithm(
    algorithm: str = ML_KEM_ALGORITHM,
) -> str:
    """
    Normalize ML-KEM and legacy Kyber parameter-set names.

    Supported values:

    - ML-KEM-512
    - ML-KEM-768
    - ML-KEM-1024
    """

    validated = validate_non_empty_string(
        algorithm,
        field_name="mlkem_algorithm",
        minimum_length=1,
        maximum_length=64,
    )

    compact = (
        validated
        .strip()
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )

    aliases = {
        "MLKEM512": "ML-KEM-512",
        "MLKEM768": "ML-KEM-768",
        "MLKEM1024": "ML-KEM-1024",
        "KYBER512": "ML-KEM-512",
        "KYBER768": "ML-KEM-768",
        "KYBER1024": "ML-KEM-1024",
    }

    normalized = aliases.get(
        compact
    )

    if normalized is None:
        raise ProtocolValidationError(
            f"Unsupported ML-KEM algorithm: {algorithm}",
            details={
                "supported_algorithms": [
                    "ML-KEM-512",
                    "ML-KEM-768",
                    "ML-KEM-1024",
                ],
            },
        )

    return normalized


# ---------------------------------------------------------------------
# Ciphertext fingerprint
# ---------------------------------------------------------------------

def calculate_ciphertext_fingerprint(
    ciphertext: bytes,
) -> str:
    """
    Calculate a SHA3-256 fingerprint of an ML-KEM ciphertext.

    The fingerprint can be stored in protocol diagnostics without
    exposing the entire ciphertext.
    """

    validated_ciphertext = validate_bytes(
        ciphertext,
        field_name="mlkem_ciphertext",
        minimum_length=32,
        maximum_length=100_000,
    )

    return hashlib.sha3_256(
        validated_ciphertext
    ).hexdigest()


def verify_ciphertext_fingerprint(
    ciphertext: bytes,
    expected_fingerprint: str,
) -> bool:
    """
    Verify a ciphertext fingerprint in constant time.
    """

    calculated = calculate_ciphertext_fingerprint(
        ciphertext
    )

    validated_expected = (
        validate_non_empty_string(
            expected_fingerprint,
            field_name=(
                "expected_ciphertext_fingerprint"
            ),
            minimum_length=64,
            maximum_length=64,
        )
        .lower()
    )

    return hmac.compare_digest(
        calculated,
        validated_expected,
    )


# ---------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------

def _find_backend_function(
    function_names: tuple[str, ...],
) -> Callable[..., Any]:
    """
    Locate a compatible function in `mlkem_module`.
    """

    for function_name in function_names:
        function = getattr(
            mlkem_module,
            function_name,
            None,
        )

        if callable(function):
            return function

    raise MLKEMDecapsulationError(
        "Required ML-KEM backend function was not found.",
        details={
            "module": (
                "src.cryptography.mlkem_module"
            ),
            "searched_functions": list(
                function_names
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
    Invoke the first argument combination compatible with the backend.
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

    raise MLKEMDecapsulationError(
        "Unable to call the ML-KEM backend function.",
        details={
            "function": getattr(
                function,
                "__name__",
                repr(function),
            ),
            "binding_errors": binding_errors,
        },
    )


def _extract_shared_secret(
    backend_result: Any,
) -> bytes:
    """
    Normalize common backend decapsulation return formats.

    Supported forms include:

    - bytes
    - {"shared_secret": bytes}
    - {"secret": bytes}
    - object.shared_secret
    - (shared_secret,)
    - (shared_secret, valid)
    - (valid, shared_secret)
    """

    shared_secret: Any = None

    if isinstance(
        backend_result,
        (bytes, bytearray, memoryview),
    ):
        shared_secret = bytes(
            backend_result
        )

    elif isinstance(
        backend_result,
        Mapping,
    ):
        for field_name in (
            "shared_secret",
            "secret",
            "ss",
        ):
            if field_name in backend_result:
                shared_secret = backend_result[
                    field_name
                ]

                break

        explicit_validity = backend_result.get(
            "valid"
        )

        if (
            explicit_validity is not None
            and explicit_validity is False
        ):
            raise MLKEMDecapsulationError(
                (
                    "The ML-KEM backend reported "
                    "an invalid ciphertext."
                )
            )

    elif isinstance(
        backend_result,
        tuple,
    ):
        if len(backend_result) == 1:
            shared_secret = backend_result[0]

        elif len(backend_result) == 2:
            first, second = backend_result

            if isinstance(first, bool):
                if not first:
                    raise MLKEMDecapsulationError(
                        (
                            "The ML-KEM backend reported "
                            "decapsulation failure."
                        )
                    )

                shared_secret = second

            elif isinstance(second, bool):
                if not second:
                    raise MLKEMDecapsulationError(
                        (
                            "The ML-KEM backend reported "
                            "decapsulation failure."
                        )
                    )

                shared_secret = first

            else:
                for candidate in (
                    first,
                    second,
                ):
                    if isinstance(
                        candidate,
                        (
                            bytes,
                            bytearray,
                            memoryview,
                        ),
                    ):
                        shared_secret = candidate
                        break

    else:
        shared_secret = getattr(
            backend_result,
            "shared_secret",
            getattr(
                backend_result,
                "secret",
                None,
            ),
        )

        explicit_validity = getattr(
            backend_result,
            "valid",
            None,
        )

        if (
            explicit_validity is not None
            and explicit_validity is False
        ):
            raise MLKEMDecapsulationError(
                (
                    "The ML-KEM backend reported "
                    "an invalid ciphertext."
                )
            )

    if shared_secret is None:
        raise MLKEMDecapsulationError(
            (
                "The ML-KEM backend did not return "
                "a shared secret."
            ),
            details={
                "received_type": type(
                    backend_result
                ).__name__,
            },
        )

    try:
        normalized_secret = bytes(
            shared_secret
        )

    except Exception as exc:
        raise MLKEMDecapsulationError(
            "Unable to convert the shared secret to bytes.",
            details={
                "received_type": type(
                    shared_secret
                ).__name__,
                "reason": str(exc),
            },
        ) from exc

    return validate_bytes(
        normalized_secret,
        field_name="shared_secret",
        exact_length=MLKEM_SHARED_SECRET_BYTES,
    )


def _backend_decapsulate(
    *,
    secret_key: bytes,
    ciphertext: bytes,
    algorithm: str,
) -> bytes:
    """
    Perform ML-KEM decapsulation through the cryptography module.
    """

    function = _find_backend_function(
        (
            "decapsulate_shared_secret",
            "decapsulate_ciphertext",
            "mlkem_decapsulate",
            "decapsulate",
        )
    )

    try:
        backend_result = _invoke_compatible_call(
            function,
            [
                (
                    (),
                    {
                        "secret_key": secret_key,
                        "ciphertext": ciphertext,
                        "algorithm": algorithm,
                    },
                ),
                (
                    (),
                    {
                        "private_key": secret_key,
                        "ciphertext": ciphertext,
                        "algorithm": algorithm,
                    },
                ),
                (
                    (),
                    {
                        "decapsulation_key": secret_key,
                        "ciphertext": ciphertext,
                        "algorithm": algorithm,
                    },
                ),
                (
                    (
                        secret_key,
                        ciphertext,
                    ),
                    {
                        "algorithm": algorithm,
                    },
                ),
                (
                    (
                        ciphertext,
                        secret_key,
                    ),
                    {
                        "algorithm": algorithm,
                    },
                ),
                (
                    (
                        secret_key,
                        ciphertext,
                    ),
                    {},
                ),
                (
                    (
                        ciphertext,
                        secret_key,
                    ),
                    {},
                ),
            ],
        )

    except MLKEMDecapsulationError:
        raise

    except Exception as exc:
        raise MLKEMDecapsulationError(
            "ML-KEM backend decapsulation failed.",
            details={
                "algorithm": algorithm,
                "reason": str(exc),
            },
        ) from exc

    return _extract_shared_secret(
        backend_result
    )


# ---------------------------------------------------------------------
# Main decapsulation operation
# ---------------------------------------------------------------------

def decapsulate_mlkem_ciphertext(
    *,
    secret_key: bytes,
    ciphertext: bytes,
    algorithm: str = ML_KEM_ALGORITHM,
    expected_ciphertext_fingerprint: str | None = None,
    decapsulation_timestamp: int | None = None,
) -> MLKEMDecapsulationResult:
    """
    Decapsulate an ML-KEM ciphertext at the Authentication Server.

    Parameters
    ----------
    secret_key:
        Ephemeral ML-KEM secret decapsulation key generated by the
        Authentication Server.

    ciphertext:
        M3 ciphertext received from the Mobile Station.

    algorithm:
        ML-KEM parameter set.

    expected_ciphertext_fingerprint:
        Optional fingerprint recorded from the authenticated M3
        envelope. A mismatch causes fail-closed rejection.

    decapsulation_timestamp:
        Optional deterministic timestamp for testing.
    """

    normalized_algorithm = (
        normalize_mlkem_algorithm(
            algorithm
        )
    )

    validated_secret_key = validate_bytes(
        secret_key,
        field_name="mlkem_secret_key",
        minimum_length=32,
        maximum_length=100_000,
    )

    validated_ciphertext = validate_bytes(
        ciphertext,
        field_name="mlkem_ciphertext",
        minimum_length=32,
        maximum_length=100_000,
    )

    ciphertext_fingerprint = (
        calculate_ciphertext_fingerprint(
            validated_ciphertext
        )
    )

    if (
        expected_ciphertext_fingerprint
        is not None
        and not verify_ciphertext_fingerprint(
            validated_ciphertext,
            expected_ciphertext_fingerprint,
        )
    ):
        raise MLKEMDecapsulationError(
            (
                "ML-KEM ciphertext fingerprint does not "
                "match the authenticated M3 message."
            ),
            details={
                "calculated_fingerprint": (
                    ciphertext_fingerprint
                ),
                "expected_fingerprint": (
                    expected_ciphertext_fingerprint
                ),
            },
        )

    decapsulated_at = (
        current_timestamp()
        if decapsulation_timestamp is None
        else validate_integer(
            decapsulation_timestamp,
            field_name=(
                "decapsulation_timestamp"
            ),
            minimum=0,
        )
    )

    shared_secret = _backend_decapsulate(
        secret_key=validated_secret_key,
        ciphertext=validated_ciphertext,
        algorithm=normalized_algorithm,
    )

    return MLKEMDecapsulationResult(
        algorithm=normalized_algorithm,
        shared_secret=shared_secret,
        ciphertext_fingerprint=(
            ciphertext_fingerprint
        ),
        ciphertext_bytes=len(
            validated_ciphertext
        ),
        shared_secret_bytes=len(
            shared_secret
        ),
        decapsulated_at=decapsulated_at,
        success=True,
    )


def decapsulate_ciphertext(
    *,
    secret_key: bytes,
    ciphertext: bytes,
    algorithm: str = ML_KEM_ALGORITHM,
    expected_ciphertext_fingerprint: str | None = None,
) -> bytes:
    """
    Decapsulate and return only the 32-byte shared secret.
    """

    result = decapsulate_mlkem_ciphertext(
        secret_key=secret_key,
        ciphertext=ciphertext,
        algorithm=algorithm,
        expected_ciphertext_fingerprint=(
            expected_ciphertext_fingerprint
        ),
    )

    return result.get_shared_secret()


def mlkem_decapsulate(
    secret_key: bytes,
    ciphertext: bytes,
    *,
    algorithm: str = ML_KEM_ALGORITHM,
) -> bytes:
    """
    Compatibility alias for direct ML-KEM decapsulation.
    """

    return decapsulate_ciphertext(
        secret_key=secret_key,
        ciphertext=ciphertext,
        algorithm=algorithm,
    )


# ---------------------------------------------------------------------
# Reusable decapsulator
# ---------------------------------------------------------------------

class MLKEMDecapsulator:
    """
    Reusable Authentication Server ML-KEM decapsulation component.
    """

    def __init__(
        self,
        *,
        algorithm: str = ML_KEM_ALGORITHM,
    ) -> None:
        self.algorithm = (
            normalize_mlkem_algorithm(
                algorithm
            )
        )

    def decapsulate(
        self,
        *,
        secret_key: bytes,
        ciphertext: bytes,
        expected_ciphertext_fingerprint: str | None = None,
    ) -> MLKEMDecapsulationResult:
        """
        Decapsulate one M3 ciphertext.
        """

        return decapsulate_mlkem_ciphertext(
            secret_key=secret_key,
            ciphertext=ciphertext,
            algorithm=self.algorithm,
            expected_ciphertext_fingerprint=(
                expected_ciphertext_fingerprint
            ),
        )


# ---------------------------------------------------------------------
# Self-test backend helpers
# ---------------------------------------------------------------------

def _self_test_generate_keypair(
    algorithm: str,
) -> tuple[bytes, bytes]:
    """
    Generate a temporary ML-KEM key pair through mlkem_module.
    """

    function = _find_backend_function(
        (
            "generate_mlkem_keypair",
            "generate_keypair",
            "mlkem_generate_keypair",
            "mlkem_keygen",
        )
    )

    result = _invoke_compatible_call(
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

    public_key: Any = None
    secret_key: Any = None

    if isinstance(
        result,
        Mapping,
    ):
        public_key = result.get(
            "public_key",
            result.get(
                "encapsulation_key"
            ),
        )

        secret_key = result.get(
            "secret_key",
            result.get(
                "decapsulation_key",
                result.get(
                    "private_key"
                ),
            ),
        )

    elif (
        isinstance(result, tuple)
        and len(result) == 2
    ):
        public_key = result[0]
        secret_key = result[1]

    else:
        public_key = getattr(
            result,
            "public_key",
            getattr(
                result,
                "encapsulation_key",
                None,
            ),
        )

        secret_key = getattr(
            result,
            "secret_key",
            getattr(
                result,
                "decapsulation_key",
                getattr(
                    result,
                    "private_key",
                    None,
                ),
            ),
        )

    if (
        public_key is None
        or secret_key is None
    ):
        raise MLKEMDecapsulationError(
            "ML-KEM key generation returned an invalid result."
        )

    return (
        validate_bytes(
            bytes(public_key),
            field_name="mlkem_public_key",
            minimum_length=32,
            maximum_length=100_000,
        ),
        validate_bytes(
            bytes(secret_key),
            field_name="mlkem_secret_key",
            minimum_length=32,
            maximum_length=100_000,
        ),
    )


def _self_test_encapsulate(
    *,
    public_key: bytes,
    algorithm: str,
) -> tuple[bytes, bytes]:
    """
    Encapsulate a temporary shared secret through mlkem_module.
    """

    function = _find_backend_function(
        (
            "encapsulate_shared_secret",
            "encapsulate",
            "mlkem_encapsulate",
            "encapsulate_key",
        )
    )

    result = _invoke_compatible_call(
        function,
        [
            (
                (),
                {
                    "public_key": public_key,
                    "algorithm": algorithm,
                },
            ),
            (
                (),
                {
                    "encapsulation_key": public_key,
                    "algorithm": algorithm,
                },
            ),
            (
                (
                    public_key,
                ),
                {
                    "algorithm": algorithm,
                },
            ),
            (
                (
                    public_key,
                ),
                {},
            ),
        ],
    )

    ciphertext: Any = None
    shared_secret: Any = None

    if isinstance(
        result,
        Mapping,
    ):
        ciphertext = result.get(
            "ciphertext"
        )

        shared_secret = result.get(
            "shared_secret",
            result.get(
                "secret"
            ),
        )

    elif (
        isinstance(result, tuple)
        and len(result) == 2
    ):
        first, second = result

        first_bytes = bytes(first)
        second_bytes = bytes(second)

        if len(first_bytes) == MLKEM_SHARED_SECRET_BYTES:
            shared_secret = first_bytes
            ciphertext = second_bytes

        elif len(second_bytes) == MLKEM_SHARED_SECRET_BYTES:
            ciphertext = first_bytes
            shared_secret = second_bytes

        else:
            ciphertext = first_bytes
            shared_secret = second_bytes

    else:
        ciphertext = getattr(
            result,
            "ciphertext",
            None,
        )

        shared_secret = getattr(
            result,
            "shared_secret",
            getattr(
                result,
                "secret",
                None,
            ),
        )

    if (
        ciphertext is None
        or shared_secret is None
    ):
        raise MLKEMDecapsulationError(
            "ML-KEM encapsulation returned an invalid result."
        )

    return (
        validate_bytes(
            bytes(ciphertext),
            field_name="mlkem_ciphertext",
            minimum_length=32,
            maximum_length=100_000,
        ),
        validate_bytes(
            bytes(shared_secret),
            field_name="shared_secret",
            exact_length=MLKEM_SHARED_SECRET_BYTES,
        ),
    )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_mlkem_decapsulation_self_test() -> dict[str, Any]:
    """
    Test key generation, encapsulation, and decapsulation.

    The test confirms that the Mobile Station and Authentication Server
    obtain the same 32-byte shared secret.
    """

    try:
        algorithm = normalize_mlkem_algorithm(
            ML_KEM_ALGORITHM
        )

        public_key, secret_key = (
            _self_test_generate_keypair(
                algorithm
            )
        )

        ciphertext, mobile_shared_secret = (
            _self_test_encapsulate(
                public_key=public_key,
                algorithm=algorithm,
            )
        )

        expected_fingerprint = (
            calculate_ciphertext_fingerprint(
                ciphertext
            )
        )

        result = decapsulate_mlkem_ciphertext(
            secret_key=secret_key,
            ciphertext=ciphertext,
            algorithm=algorithm,
            expected_ciphertext_fingerprint=(
                expected_fingerprint
            ),
            decapsulation_timestamp=(
                1_700_000_000
            ),
        )

        shared_secret_match = (
            hmac.compare_digest(
                mobile_shared_secret,
                result.shared_secret,
            )
        )

        wrong_fingerprint_rejected = False

        try:
            decapsulate_mlkem_ciphertext(
                secret_key=secret_key,
                ciphertext=ciphertext,
                algorithm=algorithm,
                expected_ciphertext_fingerprint=(
                    "00" * 32
                ),
            )

        except MLKEMDecapsulationError:
            wrong_fingerprint_rejected = True

        success = all(
            (
                result.success,
                shared_secret_match,
                result.shared_secret_bytes
                == MLKEM_SHARED_SECRET_BYTES,
                wrong_fingerprint_rejected,
            )
        )

        return {
            "success": success,
            "algorithm": result.algorithm,
            "public_key_bytes": len(
                public_key
            ),
            "secret_key_bytes": len(
                secret_key
            ),
            "ciphertext_bytes": (
                result.ciphertext_bytes
            ),
            "shared_secret_bytes": (
                result.shared_secret_bytes
            ),
            "shared_secret_match": (
                shared_secret_match
            ),
            "ciphertext_fingerprint": (
                result.ciphertext_fingerprint
            ),
            "wrong_fingerprint_rejected": (
                wrong_fingerprint_rejected
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
    "MLKEM_SHARED_SECRET_BYTES",
    "MLKEM_CIPHERTEXT_FINGERPRINT_ALGORITHM",
    "MLKEMDecapsulationError",
    "MLKEMDecapsulationResult",
    "normalize_mlkem_algorithm",
    "calculate_ciphertext_fingerprint",
    "verify_ciphertext_fingerprint",
    "decapsulate_mlkem_ciphertext",
    "decapsulate_ciphertext",
    "mlkem_decapsulate",
    "MLKEMDecapsulator",
    "run_mlkem_decapsulation_self_test",
]