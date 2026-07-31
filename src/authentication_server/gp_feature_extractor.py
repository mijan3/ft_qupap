"""
Gaussian Process feature extraction for FT-QuPAP v5.1.

This module converts receiver-observable quantum-session measurements
into the exact ordered feature schema expected by the trained Gaussian
Process attack detector.

Feature schema:

1. qber_raw
2. mean_syndrome_weight
3. max_syndrome_weight
4. correction_failure_rate
5. loss_rate
6. noise_estimate
7. ctx_urban
8. ctx_suburban
9. ctx_rural

No sender-secret information, permanent identity, secret check schedule,
or cryptographic key is included in the GP feature vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from src.authentication_server.check_block_analyzer import (
    CheckBlockAnalysisResult,
)

from src.common.constants import (
    FEATURE_COLUMNS,
)

from src.common.enums import (
    ChannelContext,
)

from src.common.exceptions import (
    FeatureSchemaError,
    ProtocolValidationError,
)

from src.common.validators import (
    validate_context,
    validate_feature_dictionary,
    validate_integer,
    validate_probability,
    validate_qber,
)


EXPECTED_GP_FEATURES = (
    "qber_raw",
    "mean_syndrome_weight",
    "max_syndrome_weight",
    "correction_failure_rate",
    "loss_rate",
    "noise_estimate",
    "ctx_urban",
    "ctx_suburban",
    "ctx_rural",
)


@dataclass(frozen=True)
class GPFeatureVector:
    """
    Validated receiver-observable GP feature vector.
    """

    qber_raw: float
    mean_syndrome_weight: float
    max_syndrome_weight: float
    correction_failure_rate: float
    loss_rate: float
    noise_estimate: float

    ctx_urban: float
    ctx_suburban: float
    ctx_rural: float

    context: str
    noise_estimate_derived: bool

    def __post_init__(self) -> None:
        validate_qber(
            self.qber_raw
        )

        _validate_non_negative_number(
            self.mean_syndrome_weight,
            field_name="mean_syndrome_weight",
        )

        _validate_non_negative_number(
            self.max_syndrome_weight,
            field_name="max_syndrome_weight",
        )

        validate_probability(
            self.correction_failure_rate,
            field_name="correction_failure_rate",
        )

        validate_probability(
            self.loss_rate,
            field_name="loss_rate",
        )

        validate_probability(
            self.noise_estimate,
            field_name="noise_estimate",
        )

        for field_name in (
            "ctx_urban",
            "ctx_suburban",
            "ctx_rural",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value not in (
                0.0,
                1.0,
            ):
                raise ProtocolValidationError(
                    f"{field_name} must be 0.0 or 1.0."
                )

        if (
            self.ctx_urban
            + self.ctx_suburban
            + self.ctx_rural
            != 1.0
        ):
            raise ProtocolValidationError(
                (
                    "Exactly one channel-context "
                    "indicator must be active."
                )
            )

        validate_context(
            self.context
        )

        if not isinstance(
            self.noise_estimate_derived,
            bool,
        ):
            raise ProtocolValidationError(
                "noise_estimate_derived must be Boolean."
            )

    def to_feature_dict(self) -> dict[str, float]:
        """
        Return only the nine model input features.
        """

        features = {
            "qber_raw": float(
                self.qber_raw
            ),
            "mean_syndrome_weight": float(
                self.mean_syndrome_weight
            ),
            "max_syndrome_weight": float(
                self.max_syndrome_weight
            ),
            "correction_failure_rate": float(
                self.correction_failure_rate
            ),
            "loss_rate": float(
                self.loss_rate
            ),
            "noise_estimate": float(
                self.noise_estimate
            ),
            "ctx_urban": float(
                self.ctx_urban
            ),
            "ctx_suburban": float(
                self.ctx_suburban
            ),
            "ctx_rural": float(
                self.ctx_rural
            ),
        }

        return validate_feature_dictionary(
            features
        )

    def to_ordered_list(self) -> list[float]:
        """
        Return feature values in trained-model column order.
        """

        features = self.to_feature_dict()

        return [
            features[feature_name]
            for feature_name in FEATURE_COLUMNS
        ]

    def to_dict(self) -> dict[str, Any]:
        """
        Return features and extraction metadata.
        """

        result = asdict(self)

        result["features"] = (
            self.to_feature_dict()
        )

        result["ordered_feature_columns"] = list(
            FEATURE_COLUMNS
        )

        result["ordered_feature_values"] = (
            self.to_ordered_list()
        )

        return result


def _validate_non_negative_number(
    value: Any,
    *,
    field_name: str,
) -> float:
    """
    Validate a finite non-negative numeric value.
    """

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise ProtocolValidationError(
            f"{field_name} must be numeric.",
            details={
                "field_name": field_name,
                "received_type": type(
                    value
                ).__name__,
            },
        )

    normalized = float(
        value
    )

    if not np.isfinite(
        normalized
    ):
        raise ProtocolValidationError(
            f"{field_name} must be finite."
        )

    if normalized < 0.0:
        raise ProtocolValidationError(
            f"{field_name} cannot be negative."
        )

    return normalized


def _require_mapping(
    value: Any,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """
    Require a dictionary-like value.
    """

    if not isinstance(
        value,
        Mapping,
    ):
        raise ProtocolValidationError(
            f"{field_name} must be a mapping.",
            details={
                "field_name": field_name,
                "received_type": type(
                    value
                ).__name__,
            },
        )

    return value


def _require_numeric_sequence(
    value: Any,
    *,
    field_name: str,
) -> list[float]:
    """
    Validate a sequence of finite non-negative numbers.
    """

    if isinstance(
        value,
        (str, bytes, bytearray),
    ) or not isinstance(
        value,
        Sequence,
    ):
        raise ProtocolValidationError(
            f"{field_name} must be a numeric sequence."
        )

    normalized = [
        _validate_non_negative_number(
            item,
            field_name=(
                f"{field_name}[{index}]"
            ),
        )
        for index, item in enumerate(
            value
        )
    ]

    return normalized


def normalize_channel_context(
    context: ChannelContext | str,
) -> str:
    """
    Normalize channel context to urban, suburban, or rural.
    """

    raw_context = (
        context.value
        if isinstance(
            context,
            ChannelContext,
        )
        else context
    )

    normalized = validate_context(
        raw_context
    )

    return str(
        normalized
    ).strip().lower()


def encode_channel_context(
    context: ChannelContext | str,
) -> dict[str, float]:
    """
    Convert channel context into three one-hot indicators.
    """

    normalized = normalize_channel_context(
        context
    )

    indicators = {
        "ctx_urban": 0.0,
        "ctx_suburban": 0.0,
        "ctx_rural": 0.0,
    }

    indicator_name = (
        f"ctx_{normalized}"
    )

    if indicator_name not in indicators:
        raise ProtocolValidationError(
            "Unsupported channel context.",
            details={
                "context": normalized,
                "supported_contexts": [
                    "urban",
                    "suburban",
                    "rural",
                ],
            },
        )

    indicators[
        indicator_name
    ] = 1.0

    return indicators


def calculate_correction_failure_rate(
    *,
    correction_failures: int,
    processed_payload_blocks: int,
) -> float:
    """
    Calculate CSS correction-failure rate.

    Formula:

        correction_failure_rate =
            correction_failures
            ------------------------
            processed_payload_blocks

    When no payload block was processed, the result is conservatively
    set to 1.0.
    """

    validated_failures = validate_integer(
        correction_failures,
        field_name="correction_failures",
        minimum=0,
    )

    validated_processed = validate_integer(
        processed_payload_blocks,
        field_name="processed_payload_blocks",
        minimum=0,
    )

    if (
        validated_failures
        > validated_processed
        and validated_processed > 0
    ):
        raise ProtocolValidationError(
            (
                "Correction failures cannot exceed "
                "processed payload blocks."
            )
        )

    if validated_processed == 0:
        return 1.0

    return validate_probability(
        validated_failures
        / validated_processed,
        field_name="correction_failure_rate",
    )


def calculate_syndrome_statistics(
    syndrome_weights: Sequence[Any],
) -> tuple[float, float]:
    """
    Calculate mean and maximum syndrome weights.

    Empty syndrome evidence returns:

        mean_syndrome_weight = 0.0
        max_syndrome_weight = 0.0
    """

    normalized = _require_numeric_sequence(
        syndrome_weights,
        field_name="syndrome_weights",
    )

    if not normalized:
        return (
            0.0,
            0.0,
        )

    values = np.asarray(
        normalized,
        dtype=float,
    )

    mean_weight = float(
        np.mean(
            values
        )
    )

    max_weight = float(
        np.max(
            values
        )
    )

    return (
        mean_weight,
        max_weight,
    )


def estimate_receiver_noise(
    *,
    qber_raw: float,
    mean_syndrome_weight: float,
    correction_failure_rate: float,
    loss_rate: float,
) -> float:
    """
    Derive a bounded receiver-side noise estimate.

    This fallback is used only when the channel simulator or receiver
    does not provide an explicit noise estimate.

    The estimate combines:

    - Raw QBER
    - Normalized mean syndrome weight
    - Correction-failure rate
    - Loss rate
    """

    validated_qber = validate_qber(
        qber_raw
    )

    validated_mean_syndrome = (
        _validate_non_negative_number(
            mean_syndrome_weight,
            field_name="mean_syndrome_weight",
        )
    )

    validated_failure_rate = (
        validate_probability(
            correction_failure_rate,
            field_name=(
                "correction_failure_rate"
            ),
        )
    )

    validated_loss = validate_probability(
        loss_rate,
        field_name="loss_rate",
    )

    normalized_syndrome = float(
        np.clip(
            validated_mean_syndrome / 6.0,
            0.0,
            1.0,
        )
    )

    estimated_noise = (
        0.50 * validated_qber
        + 0.20 * normalized_syndrome
        + 0.20 * validated_failure_rate
        + 0.10 * validated_loss
    )

    return float(
        np.clip(
            estimated_noise,
            0.0,
            1.0,
        )
    )


def _normalize_check_analysis(
    check_analysis: CheckBlockAnalysisResult
    | Mapping[str, Any],
) -> dict[str, Any]:
    """
    Normalize check-block analysis input.
    """

    if isinstance(
        check_analysis,
        CheckBlockAnalysisResult,
    ):
        return check_analysis.to_dict()

    mapping = _require_mapping(
        check_analysis,
        field_name="check_analysis",
    )

    if "qber_raw" not in mapping:
        raise ProtocolValidationError(
            (
                "check_analysis must contain "
                "'qber_raw'."
            )
        )

    return dict(
        mapping
    )


def _extract_syndrome_values(
    syndrome_summary: Mapping[str, Any],
) -> tuple[
    float,
    float,
    int,
    int,
]:
    """
    Extract syndrome statistics and correction counts.

    Supported input formats include:

        {
            "syndrome_weights": [...]
        }

    or:

        {
            "mean_syndrome_weight": 0.4,
            "max_syndrome_weight": 2,
            "correction_failures": 1,
            "processed_payload_blocks": 128
        }
    """

    summary = _require_mapping(
        syndrome_summary,
        field_name="syndrome_summary",
    )

    if "syndrome_weights" in summary:
        mean_weight, max_weight = (
            calculate_syndrome_statistics(
                summary["syndrome_weights"]
            )
        )

    else:
        required_statistics = (
            "mean_syndrome_weight",
            "max_syndrome_weight",
        )

        missing_fields = [
            field_name
            for field_name
            in required_statistics
            if field_name not in summary
        ]

        if missing_fields:
            raise ProtocolValidationError(
                "Syndrome summary is incomplete.",
                details={
                    "missing_fields": (
                        missing_fields
                    ),
                },
            )

        mean_weight = (
            _validate_non_negative_number(
                summary[
                    "mean_syndrome_weight"
                ],
                field_name=(
                    "mean_syndrome_weight"
                ),
            )
        )

        max_weight = (
            _validate_non_negative_number(
                summary[
                    "max_syndrome_weight"
                ],
                field_name=(
                    "max_syndrome_weight"
                ),
            )
        )

    correction_failures = (
        validate_integer(
            summary.get(
                "correction_failures",
                0,
            ),
            field_name="correction_failures",
            minimum=0,
        )
    )

    processed_payload_blocks = (
        validate_integer(
            summary.get(
                "processed_payload_blocks",
                summary.get(
                    "total_payload_blocks",
                    0,
                ),
            ),
            field_name=(
                "processed_payload_blocks"
            ),
            minimum=0,
        )
    )

    if max_weight < mean_weight:
        raise ProtocolValidationError(
            (
                "Maximum syndrome weight cannot be "
                "smaller than mean syndrome weight."
            )
        )

    return (
        mean_weight,
        max_weight,
        correction_failures,
        processed_payload_blocks,
    )


def extract_gp_features(
    *,
    check_analysis: CheckBlockAnalysisResult
    | Mapping[str, Any],
    syndrome_summary: Mapping[str, Any],
    loss_rate: float,
    context: ChannelContext | str,
    noise_estimate: float | None = None,
) -> GPFeatureVector:
    """
    Extract all nine GP model features.

    Parameters
    ----------
    check_analysis:
        Output from check_block_analyzer.py.

    syndrome_summary:
        Output from syndrome processing or a compatible dictionary.

    loss_rate:
        Receiver-observed quantum-frame loss rate.

    context:
        Urban, suburban, or rural channel context.

    noise_estimate:
        Explicit receiver-side noise estimate. When omitted, a bounded
        fallback estimate is calculated from observable measurements.
    """

    analysis = _normalize_check_analysis(
        check_analysis
    )

    qber_raw = validate_qber(
        analysis["qber_raw"]
    )

    (
        mean_syndrome_weight,
        max_syndrome_weight,
        correction_failures,
        processed_payload_blocks,
    ) = _extract_syndrome_values(
        syndrome_summary
    )

    correction_failure_rate = (
        calculate_correction_failure_rate(
            correction_failures=(
                correction_failures
            ),
            processed_payload_blocks=(
                processed_payload_blocks
            ),
        )
    )

    validated_loss_rate = (
        validate_probability(
            loss_rate,
            field_name="loss_rate",
        )
    )

    normalized_context = (
        normalize_channel_context(
            context
        )
    )

    context_features = (
        encode_channel_context(
            normalized_context
        )
    )

    if noise_estimate is None:
        final_noise_estimate = (
            estimate_receiver_noise(
                qber_raw=qber_raw,
                mean_syndrome_weight=(
                    mean_syndrome_weight
                ),
                correction_failure_rate=(
                    correction_failure_rate
                ),
                loss_rate=(
                    validated_loss_rate
                ),
            )
        )

        noise_estimate_derived = True

    else:
        final_noise_estimate = (
            validate_probability(
                noise_estimate,
                field_name="noise_estimate",
            )
        )

        noise_estimate_derived = False

    feature_vector = GPFeatureVector(
        qber_raw=qber_raw,

        mean_syndrome_weight=(
            mean_syndrome_weight
        ),

        max_syndrome_weight=(
            max_syndrome_weight
        ),

        correction_failure_rate=(
            correction_failure_rate
        ),

        loss_rate=validated_loss_rate,

        noise_estimate=(
            final_noise_estimate
        ),

        ctx_urban=(
            context_features[
                "ctx_urban"
            ]
        ),

        ctx_suburban=(
            context_features[
                "ctx_suburban"
            ]
        ),

        ctx_rural=(
            context_features[
                "ctx_rural"
            ]
        ),

        context=normalized_context,

        noise_estimate_derived=(
            noise_estimate_derived
        ),
    )

    validate_gp_feature_schema(
        feature_vector.to_feature_dict()
    )

    return feature_vector


def validate_gp_feature_schema(
    features: Mapping[str, Any],
) -> dict[str, float]:
    """
    Require the exact GP feature names and ordering schema.
    """

    if not isinstance(
        features,
        Mapping,
    ):
        raise FeatureSchemaError(
            missing_features=list(
                FEATURE_COLUMNS
            ),
            unexpected_features=[],
        )

    received_features = list(
        features.keys()
    )

    missing_features = [
        feature_name
        for feature_name in FEATURE_COLUMNS
        if feature_name not in features
    ]

    unexpected_features = [
        feature_name
        for feature_name in received_features
        if feature_name not in FEATURE_COLUMNS
    ]

    if (
        missing_features
        or unexpected_features
    ):
        raise FeatureSchemaError(
            missing_features=missing_features,
            unexpected_features=unexpected_features,
        )

    if set(FEATURE_COLUMNS) != set(
        EXPECTED_GP_FEATURES
    ):
        raise FeatureSchemaError(
            missing_features=[
                feature_name
                for feature_name
                in EXPECTED_GP_FEATURES
                if feature_name
                not in FEATURE_COLUMNS
            ],
            unexpected_features=[
                feature_name
                for feature_name
                in FEATURE_COLUMNS
                if feature_name
                not in EXPECTED_GP_FEATURES
            ],
        )

    validated = (
        validate_feature_dictionary(
            features
        )
    )

    return {
        feature_name: float(
            validated[
                feature_name
            ]
        )
        for feature_name in FEATURE_COLUMNS
    }


def extract_gp_feature_dictionary(
    *,
    check_analysis: CheckBlockAnalysisResult
    | Mapping[str, Any],
    syndrome_summary: Mapping[str, Any],
    loss_rate: float,
    context: ChannelContext | str,
    noise_estimate: float | None = None,
) -> dict[str, float]:
    """
    Extract and return only the model-ready feature dictionary.
    """

    return extract_gp_features(
        check_analysis=check_analysis,
        syndrome_summary=syndrome_summary,
        loss_rate=loss_rate,
        context=context,
        noise_estimate=noise_estimate,
    ).to_feature_dict()


def run_gp_feature_extractor_self_test() -> dict[str, Any]:
    """
    Run deterministic feature-extraction tests.
    """

    check_analysis = {
        "qber_raw": 0.04,
        "observed_check_blocks": 30,
    }

    syndrome_summary = {
        "syndrome_weights": [
            0,
            1,
            0,
            2,
        ],
        "correction_failures": 1,
        "processed_payload_blocks": 128,
    }

    explicit_result = extract_gp_features(
        check_analysis=check_analysis,
        syndrome_summary=syndrome_summary,
        loss_rate=0.05,
        context="urban",
        noise_estimate=0.08,
    )

    derived_result = extract_gp_features(
        check_analysis=check_analysis,
        syndrome_summary=syndrome_summary,
        loss_rate=0.05,
        context="rural",
        noise_estimate=None,
    )

    explicit_features = (
        explicit_result.to_feature_dict()
    )

    derived_features = (
        derived_result.to_feature_dict()
    )

    expected_mean_syndrome = (
        0.0
        + 1.0
        + 0.0
        + 2.0
    ) / 4.0

    expected_failure_rate = (
        1.0 / 128.0
    )

    success = all(
        (
            list(
                explicit_features.keys()
            )
            == list(
                FEATURE_COLUMNS
            ),

            np.isclose(
                explicit_features[
                    "mean_syndrome_weight"
                ],
                expected_mean_syndrome,
            ),

            np.isclose(
                explicit_features[
                    "max_syndrome_weight"
                ],
                2.0,
            ),

            np.isclose(
                explicit_features[
                    "correction_failure_rate"
                ],
                expected_failure_rate,
            ),

            explicit_features[
                "ctx_urban"
            ]
            == 1.0,

            explicit_features[
                "ctx_suburban"
            ]
            == 0.0,

            explicit_features[
                "ctx_rural"
            ]
            == 0.0,

            not explicit_result
            .noise_estimate_derived,

            derived_result
            .noise_estimate_derived,

            derived_features[
                "ctx_rural"
            ]
            == 1.0,

            0.0
            <= derived_features[
                "noise_estimate"
            ]
            <= 1.0,
        )
    )

    return {
        "success": success,

        "feature_columns": list(
            explicit_features.keys()
        ),

        "qber_raw": (
            explicit_features[
                "qber_raw"
            ]
        ),

        "mean_syndrome_weight": (
            explicit_features[
                "mean_syndrome_weight"
            ]
        ),

        "max_syndrome_weight": (
            explicit_features[
                "max_syndrome_weight"
            ]
        ),

        "correction_failure_rate": (
            explicit_features[
                "correction_failure_rate"
            ]
        ),

        "explicit_noise_estimate": (
            explicit_features[
                "noise_estimate"
            ]
        ),

        "derived_noise_estimate": (
            derived_features[
                "noise_estimate"
            ]
        ),

        "explicit_noise_derived": (
            explicit_result
            .noise_estimate_derived
        ),

        "fallback_noise_derived": (
            derived_result
            .noise_estimate_derived
        ),

        "urban_indicator": (
            explicit_features[
                "ctx_urban"
            ]
        ),

        "rural_indicator": (
            derived_features[
                "ctx_rural"
            ]
        ),
    }


__all__ = [
    "EXPECTED_GP_FEATURES",
    "GPFeatureVector",
    "normalize_channel_context",
    "encode_channel_context",
    "calculate_correction_failure_rate",
    "calculate_syndrome_statistics",
    "estimate_receiver_noise",
    "extract_gp_features",
    "validate_gp_feature_schema",
    "extract_gp_feature_dictionary",
    "run_gp_feature_extractor_self_test",
]