"""
FT-QuPAP Gaussian Process Feature Schema

This module defines the exact receiver-observable feature order used by
the FT-QuPAP Gaussian Process attack detector.

The feature order must remain identical during:

- model training
- validation
- probability calibration
- independent testing
- model export
- model loading
- live attack prediction

Security boundary:

Only Authentication Server observable evidence may be used as model
input. Hidden simulator information such as Eve's configured attack
fraction, attack mode, attack positions, or ground-truth label must
never enter FEATURE_COLUMNS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CONTEXT_CATEGORIES: tuple[str, ...] = (
    "urban",
    "suburban",
    "rural",
)


BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    "qber_raw",
    "mean_syndrome_weight",
    "max_syndrome_weight",
    "correction_failure_rate",
    "loss_rate",
    "noise_estimate",
)


CONTEXT_FEATURE_COLUMNS: tuple[str, ...] = (
    "ctx_urban",
    "ctx_suburban",
    "ctx_rural",
)


# Keep FEATURE_COLUMNS as a list because the notebook and exported
# model bundle compare their stored feature_columns value directly
# against this object.
FEATURE_COLUMNS: list[str] = [
    "qber_raw",
    "mean_syndrome_weight",
    "max_syndrome_weight",
    "correction_failure_rate",
    "loss_rate",
    "noise_estimate",
    "ctx_urban",
    "ctx_suburban",
    "ctx_rural",
]


FEATURE_COUNT = len(FEATURE_COLUMNS)


PROBABILITY_FEATURE_COLUMNS = frozenset(
    {
        "qber_raw",
        "correction_failure_rate",
        "loss_rate",
        "noise_estimate",
    }
)


NONNEGATIVE_FEATURE_COLUMNS = frozenset(
    {
        "mean_syndrome_weight",
        "max_syndrome_weight",
    }
)


FEATURE_DEFINITIONS: dict[str, str] = {
    "qber_raw": (
        "Raw physical check-bit error rate calculated from declared "
        "check blocks before CSS correction."
    ),
    "mean_syndrome_weight": (
        "Mean receiver-observed syndrome weight across decoded blocks."
    ),
    "max_syndrome_weight": (
        "Maximum receiver-observed syndrome weight across decoded blocks."
    ),
    "correction_failure_rate": (
        "Fraction of decoder records marked uncorrectable."
    ),
    "loss_rate": (
        "Fraction of erased physical qubits in the received frame."
    ),
    "noise_estimate": (
        "Receiver-side proxy derived from observable QBER, syndrome, "
        "correction-failure, and loss evidence."
    ),
    "ctx_urban": (
        "One-hot indicator for the urban channel context."
    ),
    "ctx_suburban": (
        "One-hot indicator for the suburban channel context."
    ),
    "ctx_rural": (
        "One-hot indicator for the rural channel context."
    ),
}


# These values may be useful as offline labels or simulation metadata,
# but they are forbidden as Authentication Server GP input features.
HIDDEN_SIMULATOR_FIELDS = frozenset(
    {
        "eve_fraction",
        "eve_mode",
        "eve_positions",
        "eve_basis",
        "attacked_mask",
        "attack_positions",
        "intercepted_positions",
        "scenario",
        "scenario_severity",
        "actual_attack",
        "label_attack",
        "attack_label",
        "ground_truth",
        "is_attack",
    }
)


TRAINING_METADATA_FIELDS = frozenset(
    {
        "scenario",
        "scenario_severity",
        "label_attack",
        "attack_label",
        "split",
        "seed",
        "session_id",
    }
)


class FeatureSchemaError(Exception):
    """Base exception for GP feature-schema failures."""


class UnknownContextError(FeatureSchemaError):
    """Raised when a channel context is unsupported."""


class MissingFeatureError(FeatureSchemaError):
    """Raised when a required model feature is missing."""


class UnexpectedFeatureError(FeatureSchemaError):
    """Raised when an unexpected model feature is supplied."""


class HiddenSimulatorFeatureError(FeatureSchemaError):
    """Raised when simulator-only knowledge enters model input."""


class InvalidFeatureValueError(FeatureSchemaError):
    """Raised when a feature value is malformed."""


class FeatureOrderMismatchError(FeatureSchemaError):
    """Raised when model metadata uses a different feature order."""


def validate_context(
    context: Any,
    categories: Sequence[str] = CONTEXT_CATEGORIES,
) -> str:
    """
    Validate and normalize a channel context.
    """

    if not isinstance(context, str):
        raise TypeError(
            "context must be a string."
        )

    normalized = context.strip().lower()

    normalized_categories = tuple(
        str(category).strip().lower()
        for category in categories
    )

    if normalized not in normalized_categories:
        raise UnknownContextError(
            f"Unknown context: {context!r}. "
            f"Expected one of {normalized_categories!r}."
        )

    return normalized


def encode_context(
    context: str,
    categories: Sequence[str] = CONTEXT_CATEGORIES,
) -> dict[str, float]:
    """
    Encode a channel context using notebook-compatible one-hot values.

    Example:

        encode_context("urban")

    returns:

        {
            "ctx_urban": 1.0,
            "ctx_suburban": 0.0,
            "ctx_rural": 0.0,
        }
    """

    normalized_context = validate_context(
        context,
        categories,
    )

    return {
        f"ctx_{name}": float(
            normalized_context == name
        )
        for name in categories
    }


def validate_numeric_feature(
    value: Any,
    feature_name: str,
) -> float:
    """
    Validate and normalize one numeric feature.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise InvalidFeatureValueError(
            f"{feature_name} must be numeric."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise InvalidFeatureValueError(
            f"{feature_name} must be finite."
        )

    if feature_name in PROBABILITY_FEATURE_COLUMNS:
        if not 0.0 <= normalized <= 1.0:
            raise InvalidFeatureValueError(
                f"{feature_name} must be between 0 and 1."
            )

    if feature_name in NONNEGATIVE_FEATURE_COLUMNS:
        if normalized < 0.0:
            raise InvalidFeatureValueError(
                f"{feature_name} cannot be negative."
            )

    if feature_name in CONTEXT_FEATURE_COLUMNS:
        if normalized not in {
            0.0,
            1.0,
        }:
            raise InvalidFeatureValueError(
                f"{feature_name} must be exactly 0.0 or 1.0."
            )

    return normalized


def find_hidden_simulator_fields(
    features: Mapping[str, Any],
) -> tuple[str, ...]:
    """
    Return simulator-only fields found in a feature mapping.
    """

    if not isinstance(features, Mapping):
        raise TypeError(
            "features must be a mapping."
        )

    hidden_fields = [
        str(field_name)
        for field_name in features
        if str(field_name).strip().lower()
        in HIDDEN_SIMULATOR_FIELDS
    ]

    return tuple(
        hidden_fields
    )


def validate_context_encoding(
    features: Mapping[str, float],
) -> None:
    """
    Require exactly one active context indicator.
    """

    context_values = [
        float(
            features[column]
        )
        for column in CONTEXT_FEATURE_COLUMNS
    ]

    active_contexts = sum(
        value == 1.0
        for value in context_values
    )

    if active_contexts != 1:
        raise InvalidFeatureValueError(
            "Exactly one context feature must equal 1.0."
        )


def validate_feature_mapping(
    features: Mapping[str, Any],
    *,
    require_exact_schema: bool = True,
    reject_hidden_fields: bool = True,
) -> dict[str, float]:
    """
    Validate one complete FT-QuPAP GP feature mapping.

    Args:
        features:
            Receiver-observable feature mapping.

        require_exact_schema:
            When True, reject every field not present in
            FEATURE_COLUMNS.

        reject_hidden_fields:
            When True, reject simulator-only fields such as
            eve_fraction and attack labels.

    Returns:
        A newly allocated dictionary in canonical model order.
    """

    if not isinstance(features, Mapping):
        raise TypeError(
            "features must be a mapping."
        )

    if reject_hidden_fields:
        hidden_fields = find_hidden_simulator_fields(
            features
        )

        if hidden_fields:
            raise HiddenSimulatorFeatureError(
                "Simulator-only fields cannot be used as GP input: "
                f"{list(hidden_fields)!r}"
            )

    supplied_columns = {
        str(column)
        for column in features
    }

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in supplied_columns
    ]

    if missing_columns:
        raise MissingFeatureError(
            f"Missing GP features: {missing_columns!r}"
        )

    if require_exact_schema:
        unexpected_columns = sorted(
            supplied_columns.difference(
                FEATURE_COLUMNS
            )
        )

        if unexpected_columns:
            raise UnexpectedFeatureError(
                "Unexpected GP features: "
                f"{unexpected_columns!r}"
            )

    normalized_features = {
        column: validate_numeric_feature(
            features[column],
            column,
        )
        for column in FEATURE_COLUMNS
    }

    validate_context_encoding(
        normalized_features
    )

    return normalized_features


def ordered_feature_values(
    features: Mapping[str, Any],
    *,
    require_exact_schema: bool = True,
) -> list[float]:
    """
    Return values in the trained GP model's exact feature order.
    """

    normalized = validate_feature_mapping(
        features,
        require_exact_schema=require_exact_schema,
    )

    return [
        normalized[column]
        for column in FEATURE_COLUMNS
    ]


def build_feature_row(
    features: Mapping[str, Any],
    *,
    require_exact_schema: bool = True,
) -> list[list[float]]:
    """
    Return a two-dimensional single-row model input.

    The result can be passed to a compatible scikit-learn estimator or
    used to construct a pandas DataFrame.
    """

    return [
        ordered_feature_values(
            features,
            require_exact_schema=require_exact_schema,
        )
    ]


def merge_observables_and_context(
    observables: Mapping[str, Any],
    context: str,
) -> dict[str, float]:
    """
    Combine six receiver observables with one-hot context features.
    """

    if not isinstance(observables, Mapping):
        raise TypeError(
            "observables must be a mapping."
        )

    merged = {
        str(key): value
        for key, value in observables.items()
    }

    merged.update(
        encode_context(context)
    )

    return validate_feature_mapping(
        merged,
        require_exact_schema=True,
    )


def validate_feature_order(
    feature_columns: Sequence[str],
) -> tuple[str, ...]:
    """
    Verify exact model feature names and ordering.

    The notebook rejects a loaded model bundle when its stored
    feature_columns list differs from FEATURE_COLUMNS.
    """

    if isinstance(
        feature_columns,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "feature_columns must be a sequence of strings."
        )

    if not isinstance(feature_columns, Sequence):
        raise TypeError(
            "feature_columns must be a sequence."
        )

    normalized_columns = tuple(
        str(column)
        for column in feature_columns
    )

    expected_columns = tuple(
        FEATURE_COLUMNS
    )

    if normalized_columns != expected_columns:
        raise FeatureOrderMismatchError(
            "GP feature schema mismatch. "
            f"Expected {expected_columns!r}, "
            f"received {normalized_columns!r}."
        )

    return normalized_columns


def schema_matches(
    feature_columns: Sequence[str],
) -> bool:
    """
    Return whether supplied columns exactly match the schema.
    """

    try:
        validate_feature_order(
            feature_columns
        )

    except (
        FeatureOrderMismatchError,
        TypeError,
    ):
        return False

    return True


def feature_schema_metadata() -> dict[str, Any]:
    """
    Return JSON-safe feature-schema metadata.
    """

    return {
        "feature_columns": list(
            FEATURE_COLUMNS
        ),
        "feature_count": FEATURE_COUNT,
        "context_categories": list(
            CONTEXT_CATEGORIES
        ),
        "feature_definitions": dict(
            FEATURE_DEFINITIONS
        ),
    }


@dataclass(frozen=True)
class GPFeatureSchema:
    """
    Reusable FT-QuPAP GP feature-schema service.
    """

    feature_columns: tuple[str, ...] = tuple(
        FEATURE_COLUMNS
    )

    context_categories: tuple[str, ...] = (
        CONTEXT_CATEGORIES
    )

    def __post_init__(self) -> None:
        validate_feature_order(
            self.feature_columns
        )

        if self.context_categories != CONTEXT_CATEGORIES:
            raise FeatureSchemaError(
                "Context categories must match the trained schema."
            )

    @property
    def feature_count(self) -> int:
        """Return the total number of GP input features."""

        return len(
            self.feature_columns
        )

    def encode_context(
        self,
        context: str,
    ) -> dict[str, float]:
        """Encode a supported channel context."""

        return encode_context(
            context,
            self.context_categories,
        )

    def validate(
        self,
        features: Mapping[str, Any],
        *,
        require_exact_schema: bool = True,
    ) -> dict[str, float]:
        """Validate one complete feature mapping."""

        return validate_feature_mapping(
            features,
            require_exact_schema=require_exact_schema,
        )

    def vector(
        self,
        features: Mapping[str, Any],
        *,
        require_exact_schema: bool = True,
    ) -> list[float]:
        """Return one ordered model feature vector."""

        return ordered_feature_values(
            features,
            require_exact_schema=require_exact_schema,
        )

    def row(
        self,
        features: Mapping[str, Any],
        *,
        require_exact_schema: bool = True,
    ) -> list[list[float]]:
        """Return one two-dimensional model input row."""

        return build_feature_row(
            features,
            require_exact_schema=require_exact_schema,
        )

    def metadata(self) -> dict[str, Any]:
        """Return serializable feature-schema metadata."""

        return feature_schema_metadata()


DEFAULT_FEATURE_SCHEMA = GPFeatureSchema()


def run_self_test() -> None:
    """
    Verify feature ordering, context encoding, and hidden-field rejection.
    """

    observables = {
        "qber_raw": 0.01,
        "mean_syndrome_weight": 0.05,
        "max_syndrome_weight": 1.0,
        "correction_failure_rate": 0.0,
        "loss_rate": 0.01,
        "noise_estimate": 0.02,
    }

    features = merge_observables_and_context(
        observables,
        "urban",
    )

    if list(features) != FEATURE_COLUMNS:
        raise FeatureSchemaError(
            "Feature ordering is incorrect."
        )

    feature_vector = ordered_feature_values(
        features
    )

    if len(feature_vector) != FEATURE_COUNT:
        raise FeatureSchemaError(
            "Feature-vector length is incorrect."
        )

    if feature_vector[-3:] != [
        1.0,
        0.0,
        0.0,
    ]:
        raise FeatureSchemaError(
            "Urban context encoding is incorrect."
        )

    if not schema_matches(
        FEATURE_COLUMNS
    ):
        raise FeatureSchemaError(
            "Canonical schema did not match itself."
        )

    hidden_features = dict(
        features
    )

    hidden_features[
        "eve_fraction"
    ] = 0.50

    hidden_field_rejected = False

    try:
        validate_feature_mapping(
            hidden_features,
            require_exact_schema=False,
        )

    except HiddenSimulatorFeatureError:
        hidden_field_rejected = True

    if not hidden_field_rejected:
        raise FeatureSchemaError(
            "Hidden Eve evidence entered the GP schema."
        )

    unknown_context_rejected = False

    try:
        encode_context(
            "unknown"
        )

    except UnknownContextError:
        unknown_context_rejected = True

    if not unknown_context_rejected:
        raise FeatureSchemaError(
            "Unknown context was accepted."
        )

    print(
        "Feature schema self-test completed successfully."
    )

    print(
        "Feature count:",
        FEATURE_COUNT,
    )

    print(
        "Feature columns:",
        FEATURE_COLUMNS,
    )


__all__ = [
    "CONTEXT_CATEGORIES",
    "BASE_FEATURE_COLUMNS",
    "CONTEXT_FEATURE_COLUMNS",
    "FEATURE_COLUMNS",
    "FEATURE_COUNT",
    "PROBABILITY_FEATURE_COLUMNS",
    "NONNEGATIVE_FEATURE_COLUMNS",
    "FEATURE_DEFINITIONS",
    "HIDDEN_SIMULATOR_FIELDS",
    "TRAINING_METADATA_FIELDS",
    "FeatureSchemaError",
    "UnknownContextError",
    "MissingFeatureError",
    "UnexpectedFeatureError",
    "HiddenSimulatorFeatureError",
    "InvalidFeatureValueError",
    "FeatureOrderMismatchError",
    "GPFeatureSchema",
    "DEFAULT_FEATURE_SCHEMA",
    "validate_context",
    "encode_context",
    "validate_numeric_feature",
    "find_hidden_simulator_fields",
    "validate_context_encoding",
    "validate_feature_mapping",
    "ordered_feature_values",
    "build_feature_row",
    "merge_observables_and_context",
    "validate_feature_order",
    "schema_matches",
    "feature_schema_metadata",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        FeatureSchemaError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[FEATURE SCHEMA ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error