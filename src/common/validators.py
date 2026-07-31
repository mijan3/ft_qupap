"""
Validation helpers for the FT-QuPAP v5.1 project.

This module validates:

- Required dictionary fields
- Pseudonymous identities
- Timestamps and freshness
- Nonces
- Mobile-network contexts
- Cryptographic byte values
- Bit strings
- Probabilities
- QBER and loss values
- Quantum block counts
- Gaussian Process feature dictionaries
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from src.common.constants import (
    CHECK_LOGICAL_QUBITS,
    FEATURE_COLUMNS,
    FRESHNESS_WINDOW_SECONDS,
    KMAC_TAG_BYTES,
    MINIMUM_OBSERVED_CHECK_BLOCKS,
    NONCE_SIZE_BYTES,
    PAYLOAD_LOGICAL_QUBITS,
    SUPPORTED_CONTEXTS,
    TOTAL_LOGICAL_QUBITS,
    TOTAL_PHYSICAL_QUBITS,
)

from src.common.exceptions import (
    ConfigurationError,
    FeatureSchemaError,
    FreshnessError,
    ProtocolValidationError,
)


PSEUDONYM_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$"
)

HEX_PATTERN = re.compile(
    r"^[0-9a-fA-F]+$"
)

BIT_STRING_PATTERN = re.compile(
    r"^[01]+$"
)


def require_mapping(
    value: Any,
    *,
    field_name: str = "value",
) -> Mapping[str, Any]:
    """
    Ensure that a value is dictionary-like.
    """

    if not isinstance(value, Mapping):
        raise ProtocolValidationError(
            f"{field_name} must be a mapping.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def require_sequence(
    value: Any,
    *,
    field_name: str = "value",
) -> Sequence[Any]:
    """
    Ensure that a value is a non-string sequence.
    """

    if isinstance(value, (str, bytes, bytearray)):
        raise ProtocolValidationError(
            f"{field_name} must be a sequence, not text or bytes.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    if not isinstance(value, Sequence):
        raise ProtocolValidationError(
            f"{field_name} must be a sequence.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    return value


def require_fields(
    payload: Mapping[str, Any],
    required_fields: Sequence[str],
    *,
    payload_name: str = "payload",
) -> None:
    """
    Ensure that all required fields exist in a mapping.

    Example:

        require_fields(
            request,
            ("pseudonym_id", "timestamp", "nonce", "context"),
            payload_name="authentication request",
        )
    """

    require_mapping(
        payload,
        field_name=payload_name,
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in payload
    ]

    if missing_fields:
        raise ProtocolValidationError(
            f"{payload_name} is missing required fields.",
            details={
                "missing_fields": missing_fields,
                "required_fields": list(required_fields),
            },
        )


def reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed_fields: Sequence[str],
    *,
    payload_name: str = "payload",
) -> None:
    """
    Reject fields that are not part of the expected message schema.
    """

    allowed = set(allowed_fields)

    unknown_fields = sorted(
        key
        for key in payload.keys()
        if key not in allowed
    )

    if unknown_fields:
        raise ProtocolValidationError(
            f"{payload_name} contains unsupported fields.",
            details={
                "unknown_fields": unknown_fields,
                "allowed_fields": sorted(allowed),
            },
        )


def validate_non_empty_string(
    value: Any,
    *,
    field_name: str,
    minimum_length: int = 1,
    maximum_length: int = 1024,
) -> str:
    """
    Validate a trimmed, non-empty string.
    """

    if not isinstance(value, str):
        raise ProtocolValidationError(
            f"{field_name} must be a string.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    normalized = value.strip()

    if len(normalized) < minimum_length:
        raise ProtocolValidationError(
            f"{field_name} is too short.",
            details={
                "minimum_length": minimum_length,
                "actual_length": len(normalized),
            },
        )

    if len(normalized) > maximum_length:
        raise ProtocolValidationError(
            f"{field_name} is too long.",
            details={
                "maximum_length": maximum_length,
                "actual_length": len(normalized),
            },
        )

    return normalized


def validate_pseudonym_id(
    pseudonym_id: Any,
) -> str:
    """
    Validate a pseudonymous subscriber identity.

    Valid examples:

        PID-6G-UE-0001
        subscriber:001
        UE_test_01
    """

    normalized = validate_non_empty_string(
        pseudonym_id,
        field_name="pseudonym_id",
        minimum_length=3,
        maximum_length=128,
    )

    if PSEUDONYM_PATTERN.fullmatch(normalized) is None:
        raise ProtocolValidationError(
            "The pseudonymous identity contains invalid characters.",
            details={
                "pseudonym_id": normalized,
                "allowed_pattern": PSEUDONYM_PATTERN.pattern,
            },
        )

    return normalized


def validate_context(
    context: Any,
) -> str:
    """
    Validate and normalize the channel context.
    """

    normalized = validate_non_empty_string(
        context,
        field_name="context",
        minimum_length=1,
        maximum_length=32,
    ).lower()

    if normalized not in SUPPORTED_CONTEXTS:
        raise ProtocolValidationError(
            f"Unsupported context: {normalized}",
            details={
                "context": normalized,
                "supported_contexts": list(SUPPORTED_CONTEXTS),
            },
        )

    return normalized


def validate_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """
    Validate an integer while rejecting Boolean values.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(
            f"{field_name} must be an integer.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    if minimum is not None and value < minimum:
        raise ProtocolValidationError(
            f"{field_name} must be at least {minimum}.",
            details={
                "field_name": field_name,
                "minimum": minimum,
                "received": value,
            },
        )

    if maximum is not None and value > maximum:
        raise ProtocolValidationError(
            f"{field_name} must not exceed {maximum}.",
            details={
                "field_name": field_name,
                "maximum": maximum,
                "received": value,
            },
        )

    return value


def validate_number(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """
    Validate a finite integer or floating-point value.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ProtocolValidationError(
            f"{field_name} must be numeric.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ProtocolValidationError(
            f"{field_name} must be finite.",
            details={
                "field_name": field_name,
                "received": converted,
            },
        )

    if minimum is not None and converted < minimum:
        raise ProtocolValidationError(
            f"{field_name} must be at least {minimum}.",
            details={
                "field_name": field_name,
                "minimum": minimum,
                "received": converted,
            },
        )

    if maximum is not None and converted > maximum:
        raise ProtocolValidationError(
            f"{field_name} must not exceed {maximum}.",
            details={
                "field_name": field_name,
                "maximum": maximum,
                "received": converted,
            },
        )

    return converted


def validate_probability(
    value: Any,
    *,
    field_name: str = "probability",
) -> float:
    """
    Validate a probability in the inclusive interval [0, 1].
    """

    return validate_number(
        value,
        field_name=field_name,
        minimum=0.0,
        maximum=1.0,
    )


def validate_qber(
    value: Any,
) -> float:
    """
    Validate a QBER value in the interval [0, 1].
    """

    return validate_probability(
        value,
        field_name="qber",
    )


def validate_loss_rate(
    value: Any,
) -> float:
    """
    Validate a channel loss rate in the interval [0, 1].
    """

    return validate_probability(
        value,
        field_name="loss_rate",
    )


def validate_timestamp(
    timestamp: Any,
    *,
    current_time: int | None = None,
    freshness_window_seconds: int = FRESHNESS_WINDOW_SECONDS,
    future_tolerance_seconds: int = 5,
) -> int:
    """
    Validate authentication-request freshness.

    The timestamp must not be older than the configured freshness window.
    A small future tolerance is allowed for clock differences.
    """

    validated_timestamp = validate_integer(
        timestamp,
        field_name="timestamp",
        minimum=0,
    )

    validated_window = validate_integer(
        freshness_window_seconds,
        field_name="freshness_window_seconds",
        minimum=1,
    )

    validated_future_tolerance = validate_integer(
        future_tolerance_seconds,
        field_name="future_tolerance_seconds",
        minimum=0,
    )

    now = (
        int(time.time())
        if current_time is None
        else validate_integer(
            current_time,
            field_name="current_time",
            minimum=0,
        )
    )

    age = now - validated_timestamp

    if age > validated_window:
        raise FreshnessError(
            (
                f"Authentication request is stale by {age} seconds. "
                f"Maximum age is {validated_window} seconds."
            ),
            timestamp=validated_timestamp,
            current_time=now,
        )

    if age < -validated_future_tolerance:
        raise FreshnessError(
            (
                "Authentication request timestamp is too far "
                "in the future."
            ),
            timestamp=validated_timestamp,
            current_time=now,
        )

    return validated_timestamp


def validate_nonce_hex(
    nonce: Any,
    *,
    expected_bytes: int = NONCE_SIZE_BYTES,
) -> str:
    """
    Validate a hexadecimal nonce.

    The default protocol nonce contains 16 bytes, represented by
    32 hexadecimal characters.
    """

    normalized = validate_non_empty_string(
        nonce,
        field_name="nonce",
        minimum_length=2,
        maximum_length=512,
    )

    if len(normalized) % 2 != 0:
        raise ProtocolValidationError(
            "The hexadecimal nonce length must be even.",
            details={
                "nonce_length": len(normalized),
            },
        )

    if HEX_PATTERN.fullmatch(normalized) is None:
        raise ProtocolValidationError(
            "The nonce must contain only hexadecimal characters.",
            details={
                "nonce": normalized,
            },
        )

    actual_bytes = len(normalized) // 2

    if actual_bytes != expected_bytes:
        raise ProtocolValidationError(
            (
                f"The nonce must contain exactly "
                f"{expected_bytes} bytes."
            ),
            details={
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
            },
        )

    return normalized.lower()


def validate_nonce_bytes(
    nonce: Any,
    *,
    expected_bytes: int = NONCE_SIZE_BYTES,
) -> bytes:
    """
    Validate a nonce stored as raw bytes.
    """

    return validate_bytes(
        nonce,
        field_name="nonce",
        exact_length=expected_bytes,
    )


def validate_bytes(
    value: Any,
    *,
    field_name: str,
    minimum_length: int = 1,
    maximum_length: int | None = None,
    exact_length: int | None = None,
) -> bytes:
    """
    Validate immutable cryptographic byte material.
    """

    if not isinstance(value, bytes):
        raise ProtocolValidationError(
            f"{field_name} must be bytes.",
            details={
                "field_name": field_name,
                "received_type": type(value).__name__,
            },
        )

    length = len(value)

    if exact_length is not None and length != exact_length:
        raise ProtocolValidationError(
            (
                f"{field_name} must contain exactly "
                f"{exact_length} bytes."
            ),
            details={
                "field_name": field_name,
                "expected_length": exact_length,
                "actual_length": length,
            },
        )

    if length < minimum_length:
        raise ProtocolValidationError(
            (
                f"{field_name} must contain at least "
                f"{minimum_length} byte(s)."
            ),
            details={
                "field_name": field_name,
                "minimum_length": minimum_length,
                "actual_length": length,
            },
        )

    if maximum_length is not None and length > maximum_length:
        raise ProtocolValidationError(
            (
                f"{field_name} must not exceed "
                f"{maximum_length} bytes."
            ),
            details={
                "field_name": field_name,
                "maximum_length": maximum_length,
                "actual_length": length,
            },
        )

    return value


def validate_kmac_tag(
    tag: Any,
) -> bytes:
    """
    Validate the 128-bit KMAC authentication tag.
    """

    return validate_bytes(
        tag,
        field_name="kmac_tag",
        exact_length=KMAC_TAG_BYTES,
    )


def validate_bit(
    value: Any,
    *,
    field_name: str = "bit",
) -> int:
    """
    Validate a single classical bit.
    """

    if value not in (0, 1):
        raise ProtocolValidationError(
            f"{field_name} must be 0 or 1.",
            details={
                "field_name": field_name,
                "received": value,
            },
        )

    return int(value)


def validate_bit_string(
    bit_string: Any,
    *,
    field_name: str = "bit_string",
    exact_length: int | None = None,
) -> str:
    """
    Validate a text bit string such as '010101'.
    """

    normalized = validate_non_empty_string(
        bit_string,
        field_name=field_name,
        minimum_length=1,
        maximum_length=1_000_000,
    )

    if BIT_STRING_PATTERN.fullmatch(normalized) is None:
        raise ProtocolValidationError(
            f"{field_name} must contain only 0 and 1.",
            details={
                "field_name": field_name,
            },
        )

    if (
        exact_length is not None
        and len(normalized) != exact_length
    ):
        raise ProtocolValidationError(
            (
                f"{field_name} must contain exactly "
                f"{exact_length} bits."
            ),
            details={
                "field_name": field_name,
                "expected_length": exact_length,
                "actual_length": len(normalized),
            },
        )

    return normalized


def validate_bit_sequence(
    bits: Any,
    *,
    field_name: str = "bits",
    exact_length: int | None = None,
) -> list[int]:
    """
    Validate a sequence containing only integer bits.
    """

    sequence = require_sequence(
        bits,
        field_name=field_name,
    )

    normalized = [
        validate_bit(
            bit,
            field_name=f"{field_name}[{index}]",
        )
        for index, bit in enumerate(sequence)
    ]

    if (
        exact_length is not None
        and len(normalized) != exact_length
    ):
        raise ProtocolValidationError(
            (
                f"{field_name} must contain exactly "
                f"{exact_length} values."
            ),
            details={
                "field_name": field_name,
                "expected_length": exact_length,
                "actual_length": len(normalized),
            },
        )

    return normalized


def validate_payload_bit_count(
    payload_bits: Any,
) -> list[int]:
    """
    Validate the 128 logical payload bits.
    """

    return validate_bit_sequence(
        payload_bits,
        field_name="payload_bits",
        exact_length=PAYLOAD_LOGICAL_QUBITS,
    )


def validate_check_bit_count(
    check_bits: Any,
) -> list[int]:
    """
    Validate the 32 independent logical check bits.
    """

    return validate_bit_sequence(
        check_bits,
        field_name="check_bits",
        exact_length=CHECK_LOGICAL_QUBITS,
    )


def validate_total_logical_block_count(
    blocks: Any,
) -> Sequence[Any]:
    """
    Validate that the frame contains 160 logical blocks.
    """

    sequence = require_sequence(
        blocks,
        field_name="logical_blocks",
    )

    if len(sequence) != TOTAL_LOGICAL_QUBITS:
        raise ProtocolValidationError(
            (
                "The quantum frame must contain exactly "
                f"{TOTAL_LOGICAL_QUBITS} logical blocks."
            ),
            details={
                "expected_logical_blocks": TOTAL_LOGICAL_QUBITS,
                "actual_logical_blocks": len(sequence),
            },
        )

    return sequence


def validate_physical_qubit_count(
    physical_qubit_count: Any,
) -> int:
    """
    Validate the full Steane-encoded frame size.
    """

    count = validate_integer(
        physical_qubit_count,
        field_name="physical_qubit_count",
        minimum=0,
    )

    if count != TOTAL_PHYSICAL_QUBITS:
        raise ProtocolValidationError(
            (
                "The complete Steane frame must contain exactly "
                f"{TOTAL_PHYSICAL_QUBITS} physical qubits."
            ),
            details={
                "expected_physical_qubits": TOTAL_PHYSICAL_QUBITS,
                "actual_physical_qubits": count,
            },
        )

    return count


def validate_observed_check_blocks(
    observed_count: Any,
    *,
    minimum_required: int = MINIMUM_OBSERVED_CHECK_BLOCKS,
) -> int:
    """
    Validate the number of successfully observed check blocks.
    """

    count = validate_integer(
        observed_count,
        field_name="observed_check_blocks",
        minimum=0,
        maximum=CHECK_LOGICAL_QUBITS,
    )

    minimum = validate_integer(
        minimum_required,
        field_name="minimum_required",
        minimum=1,
        maximum=CHECK_LOGICAL_QUBITS,
    )

    if count < minimum:
        raise ProtocolValidationError(
            (
                f"Only {count} check blocks were observed; "
                f"at least {minimum} are required."
            ),
            code="INSUFFICIENT_CHECK_BLOCKS",
            details={
                "observed_check_blocks": count,
                "minimum_required": minimum,
            },
        )

    return count


def validate_feature_dictionary(
    features: Mapping[str, Any],
    *,
    reject_extra_features: bool = True,
) -> dict[str, float]:
    """
    Validate the complete Gaussian Process input feature dictionary.

    Required order and names:

        qber_raw
        mean_syndrome_weight
        max_syndrome_weight
        correction_failure_rate
        loss_rate
        noise_estimate
        ctx_urban
        ctx_suburban
        ctx_rural
    """

    require_mapping(
        features,
        field_name="features",
    )

    feature_names = set(features.keys())
    required_names = set(FEATURE_COLUMNS)

    missing_features = sorted(
        required_names - feature_names
    )

    unexpected_features = sorted(
        feature_names - required_names
    )

    if missing_features or (
        reject_extra_features
        and unexpected_features
    ):
        raise FeatureSchemaError(
            missing_features=missing_features,
            unexpected_features=unexpected_features,
        )

    normalized: dict[str, float] = {}

    for feature_name in FEATURE_COLUMNS:
        feature_value = validate_number(
            features[feature_name],
            field_name=feature_name,
        )

        normalized[feature_name] = feature_value

    for probability_feature in (
        "qber_raw",
        "correction_failure_rate",
        "loss_rate",
        "noise_estimate",
        "ctx_urban",
        "ctx_suburban",
        "ctx_rural",
    ):
        validate_probability(
            normalized[probability_feature],
            field_name=probability_feature,
        )

    validate_number(
        normalized["mean_syndrome_weight"],
        field_name="mean_syndrome_weight",
        minimum=0.0,
    )

    validate_number(
        normalized["max_syndrome_weight"],
        field_name="max_syndrome_weight",
        minimum=0.0,
    )

    context_sum = (
        normalized["ctx_urban"]
        + normalized["ctx_suburban"]
        + normalized["ctx_rural"]
    )

    if not math.isclose(
        context_sum,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise FeatureSchemaError(
            missing_features=[],
            unexpected_features=[
                (
                    "Context one-hot features must sum "
                    f"to 1.0; received {context_sum}."
                )
            ],
        )

    context_values = (
        normalized["ctx_urban"],
        normalized["ctx_suburban"],
        normalized["ctx_rural"],
    )

    if any(value not in (0.0, 1.0) for value in context_values):
        raise FeatureSchemaError(
            missing_features=[],
            unexpected_features=[
                "Context features must be exact one-hot values."
            ],
        )

    return normalized


def validate_threshold_order(
    *,
    operational_threshold: float,
    retry_upper: float,
) -> tuple[float, float]:
    """
    Ensure that the retry gray zone is correctly ordered.

    Expected policy:

        p_attack < operational_threshold
            → Accept

        operational_threshold <= p_attack < retry_upper
            → Retry when deterministic conditions allow it

        p_attack >= retry_upper
            → Reject
    """

    threshold = validate_probability(
        operational_threshold,
        field_name="operational_threshold",
    )

    upper = validate_probability(
        retry_upper,
        field_name="retry_upper",
    )

    if upper < threshold:
        raise ConfigurationError(
            (
                "Retry upper boundary cannot be lower than "
                "the operational GP threshold."
            ),
            details={
                "operational_threshold": threshold,
                "retry_upper": upper,
            },
        )

    return threshold, upper


def validate_authentication_request(
    request: Mapping[str, Any],
    *,
    current_time: int | None = None,
) -> dict[str, Any]:
    """
    Validate the basic M1 authentication-request structure.

    Expected fields:

        pseudonym_id
        timestamp
        nonce
        context
        request_type
    """

    expected_fields = (
        "pseudonym_id",
        "timestamp",
        "nonce",
        "context",
        "request_type",
    )

    require_fields(
        request,
        expected_fields,
        payload_name="authentication request",
    )

    reject_unknown_fields(
        request,
        expected_fields,
        payload_name="authentication request",
    )

    validated_request = {
        "pseudonym_id": validate_pseudonym_id(
            request["pseudonym_id"]
        ),
        "timestamp": validate_timestamp(
            request["timestamp"],
            current_time=current_time,
        ),
        "nonce": validate_nonce_hex(
            request["nonce"]
        ),
        "context": validate_context(
            request["context"]
        ),
        "request_type": validate_non_empty_string(
            request["request_type"],
            field_name="request_type",
            minimum_length=1,
            maximum_length=128,
        ),
    }

    return validated_request


__all__ = [
    "require_mapping",
    "require_sequence",
    "require_fields",
    "reject_unknown_fields",
    "validate_non_empty_string",
    "validate_pseudonym_id",
    "validate_context",
    "validate_integer",
    "validate_number",
    "validate_probability",
    "validate_qber",
    "validate_loss_rate",
    "validate_timestamp",
    "validate_nonce_hex",
    "validate_nonce_bytes",
    "validate_bytes",
    "validate_kmac_tag",
    "validate_bit",
    "validate_bit_string",
    "validate_bit_sequence",
    "validate_payload_bit_count",
    "validate_check_bit_count",
    "validate_total_logical_block_count",
    "validate_physical_qubit_count",
    "validate_observed_check_blocks",
    "validate_feature_dictionary",
    "validate_threshold_order",
    "validate_authentication_request",
]