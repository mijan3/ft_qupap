"""
Gaussian Process attack detection for FT-QuPAP v5.1.

The Authentication Server uses a trained Gaussian Process classifier to
estimate the probability that the current quantum authentication session
contains eavesdropping or malicious channel behaviour.

The model uses only receiver-observable features:

- Raw QBER
- Mean syndrome weight
- Maximum syndrome weight
- Correction-failure rate
- Loss rate
- Receiver-side noise estimate
- Urban context indicator
- Suburban context indicator
- Rural context indicator

The detector performs:

1. GP model-bundle loading
2. Feature-schema verification
3. Raw attack-probability prediction
4. Optional isotonic calibration
5. Binary-entropy uncertainty calculation
6. Operational threshold resolution
7. ACCEPT, RETRY, or REJECT recommendation

Mandatory deterministic checks always take priority over the GP model.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.common.constants import (
    BAYES_COST_FALSE_ACCEPT,
    BAYES_COST_FALSE_REJECT,
    DEFAULT_GP_ATTACK_THRESHOLD,
    FEATURE_COLUMNS,
    GP_GRAY_ZONE_RETRY_UPPER,
    GP_MAXIMUM_UNCERTAINTY,
    GP_MODEL_PATH,
    MINIMUM_OPERATIONAL_GP_THRESHOLD,
    MODEL_METADATA_PATH,
    THRESHOLD_PATH,
)

from src.common.enums import (
    AuthenticationDecision,
)

from src.common.exceptions import (
    FeatureSchemaError,
    MachineLearningError,
    ModelLoadingError,
    ModelNotGeneratedError,
    ProtocolValidationError,
)

from src.common.validators import (
    validate_feature_dictionary,
    validate_probability,
)


# ---------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------

ATTACK_CLASS_LABEL = 1

DEFAULT_MODEL_NAME = "FT-QuPAP Gaussian Process Detector"

DEFAULT_MODEL_VERSION = "unknown"

UNCERTAINTY_METHOD = "binary_entropy"

DECISION_REASON_ACCEPT = (
    "accepted_by_calibrated_gp_policy"
)

DECISION_REASON_RETRY = (
    "gp_probability_in_retry_gray_zone"
)

DECISION_REASON_REJECT = (
    "rejected_by_calibrated_gp_policy"
)

DECISION_REASON_UNCERTAINTY = (
    "gp_uncertainty_too_high"
)

DECISION_REASON_DETERMINISTIC_FAILURE = (
    "deterministic_protocol_check_failed"
)


# ---------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class GPThresholdPolicy:
    """
    Threshold values used by the operational GP policy.

    raw_calibration_threshold:
        Threshold selected using calibration-only sessions.

    operational_threshold:
        Deployment threshold after applying the configured operational
        lower bound.

    gray_zone_retry_upper:
        Upper probability boundary for retry consideration.

    maximum_uncertainty:
        Optional maximum accepted predictive entropy.
    """

    raw_calibration_threshold: float
    operational_threshold: float
    gray_zone_retry_upper: float
    minimum_operational_threshold: float

    maximum_uncertainty: float | None

    threshold_source: str

    def __post_init__(self) -> None:
        validate_probability(
            self.raw_calibration_threshold,
            field_name=(
                "raw_calibration_threshold"
            ),
        )

        validate_probability(
            self.operational_threshold,
            field_name=(
                "operational_threshold"
            ),
        )

        validate_probability(
            self.gray_zone_retry_upper,
            field_name=(
                "gray_zone_retry_upper"
            ),
        )

        validate_probability(
            self.minimum_operational_threshold,
            field_name=(
                "minimum_operational_threshold"
            ),
        )

        if self.maximum_uncertainty is not None:
            validate_probability(
                self.maximum_uncertainty,
                field_name="maximum_uncertainty",
            )

        if not isinstance(
            self.threshold_source,
            str,
        ) or not self.threshold_source:
            raise ProtocolValidationError(
                "threshold_source must be a non-empty string."
            )

    @property
    def gray_zone_enabled(self) -> bool:
        """
        Return True when the configured retry interval is non-empty.
        """

        return (
            self.gray_zone_retry_upper
            > self.operational_threshold
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["gray_zone_enabled"] = (
            self.gray_zone_enabled
        )

        return result


@dataclass(frozen=True)
class GPAttackDetectionResult:
    """
    Complete Gaussian Process detection result.
    """

    evaluated: bool

    deterministic_pass: bool

    raw_attack_probability: float | None
    attack_probability: float | None
    uncertainty: float | None

    predicted_attack: bool | None
    within_retry_gray_zone: bool

    recommended_decision: AuthenticationDecision
    reason: str

    threshold_policy: GPThresholdPolicy

    calibrated: bool

    model_name: str
    model_version: str
    training_source: str | None
    calibration_method: str | None

    features: dict[str, float]

    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.evaluated,
            bool,
        ):
            raise ProtocolValidationError(
                "evaluated must be Boolean."
            )

        if not isinstance(
            self.deterministic_pass,
            bool,
        ):
            raise ProtocolValidationError(
                "deterministic_pass must be Boolean."
            )

        if self.raw_attack_probability is not None:
            validate_probability(
                self.raw_attack_probability,
                field_name=(
                    "raw_attack_probability"
                ),
            )

        if self.attack_probability is not None:
            validate_probability(
                self.attack_probability,
                field_name="attack_probability",
            )

        if self.uncertainty is not None:
            validate_probability(
                self.uncertainty,
                field_name="uncertainty",
            )

        if (
            self.predicted_attack is not None
            and not isinstance(
                self.predicted_attack,
                bool,
            )
        ):
            raise ProtocolValidationError(
                "predicted_attack must be Boolean or None."
            )

        if not isinstance(
            self.within_retry_gray_zone,
            bool,
        ):
            raise ProtocolValidationError(
                "within_retry_gray_zone must be Boolean."
            )

        if not isinstance(
            self.recommended_decision,
            AuthenticationDecision,
        ):
            raise ProtocolValidationError(
                (
                    "recommended_decision must be an "
                    "AuthenticationDecision value."
                )
            )

        if not isinstance(
            self.threshold_policy,
            GPThresholdPolicy,
        ):
            raise ProtocolValidationError(
                (
                    "threshold_policy must be a "
                    "GPThresholdPolicy object."
                )
            )

        if not isinstance(
            self.features,
            dict,
        ):
            raise ProtocolValidationError(
                "features must be a dictionary."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise ProtocolValidationError(
                "metadata must be a dictionary."
            )

    @property
    def accepted_by_gp(self) -> bool:
        return (
            self.recommended_decision
            == AuthenticationDecision.ACCEPT
        )

    @property
    def retry_recommended(self) -> bool:
        return (
            self.recommended_decision
            == AuthenticationDecision.RETRY
        )

    @property
    def rejected_by_gp(self) -> bool:
        return (
            self.recommended_decision
            == AuthenticationDecision.REJECT
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible result dictionary.
        """

        return {
            "evaluated": self.evaluated,

            "deterministic_pass": (
                self.deterministic_pass
            ),

            "raw_attack_probability": (
                self.raw_attack_probability
            ),

            "attack_probability": (
                self.attack_probability
            ),

            "uncertainty": self.uncertainty,

            "predicted_attack": (
                self.predicted_attack
            ),

            "within_retry_gray_zone": (
                self.within_retry_gray_zone
            ),

            "recommended_decision": (
                self.recommended_decision.value
            ),

            "accepted_by_gp": (
                self.accepted_by_gp
            ),

            "retry_recommended": (
                self.retry_recommended
            ),

            "rejected_by_gp": (
                self.rejected_by_gp
            ),

            "reason": self.reason,

            "threshold_policy": (
                self.threshold_policy.to_dict()
            ),

            "calibrated": self.calibrated,

            "model_name": self.model_name,
            "model_version": self.model_version,

            "training_source": (
                self.training_source
            ),

            "calibration_method": (
                self.calibration_method
            ),

            "features": dict(
                self.features
            ),

            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------

_GP_BUNDLE: dict[str, Any] | None = None

_GP_BUNDLE_PATH: Path | None = None

_GP_BUNDLE_LOCK = threading.RLock()


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def _load_json_mapping(
    path: Path | str,
) -> dict[str, Any]:
    """
    Load a JSON object from disk.

    A missing file returns an empty dictionary.
    """

    source_path = Path(path)

    if not source_path.exists():
        return {}

    try:
        with source_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as exc:
        raise ModelLoadingError(
            model_path=str(source_path),
            reason=(
                "Unable to read JSON metadata: "
                f"{exc}"
            ),
        ) from exc

    if not isinstance(data, dict):
        raise ModelLoadingError(
            model_path=str(source_path),
            reason=(
                "The JSON file must contain an object."
            ),
        )

    return data


# ---------------------------------------------------------------------
# Bundle validation
# ---------------------------------------------------------------------

def validate_gp_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate a loaded GP model bundle.

    Required fields:

        model
        feature_columns
        protocol_version

    Optional fields:

        calibrator
        raw_calibration_gp_attack_threshold
        gp_attack_threshold
        training_source
        calibration_method
        metrics
    """

    if not isinstance(
        bundle,
        Mapping,
    ):
        raise ModelLoadingError(
            model_path="<memory>",
            reason=(
                "The GP bundle must be a mapping."
            ),
        )

    required_fields = (
        "model",
        "feature_columns",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in bundle
    ]

    if missing_fields:
        raise ModelLoadingError(
            model_path="<memory>",
            reason=(
                "The GP bundle is missing fields: "
                + ", ".join(missing_fields)
            ),
        )

    model = bundle["model"]

    if not callable(
        getattr(
            model,
            "predict_proba",
            None,
        )
    ):
        raise ModelLoadingError(
            model_path="<memory>",
            reason=(
                "The GP model does not provide "
                "predict_proba()."
            ),
        )

    bundle_feature_columns = list(
        bundle["feature_columns"]
    )

    if bundle_feature_columns != list(
        FEATURE_COLUMNS
    ):
        raise FeatureSchemaError(
            missing_features=[
                feature_name
                for feature_name in FEATURE_COLUMNS
                if feature_name
                not in bundle_feature_columns
            ],
            unexpected_features=[
                feature_name
                for feature_name
                in bundle_feature_columns
                if feature_name
                not in FEATURE_COLUMNS
            ],
        )

    calibrator = bundle.get(
        "calibrator"
    )

    if (
        calibrator is not None
        and not callable(
            getattr(
                calibrator,
                "predict",
                None,
            )
        )
    ):
        raise ModelLoadingError(
            model_path="<memory>",
            reason=(
                "The GP calibrator does not provide "
                "predict()."
            ),
        )

    normalized_bundle = dict(bundle)

    normalized_bundle[
        "feature_columns"
    ] = list(FEATURE_COLUMNS)

    normalized_bundle.setdefault(
        "protocol_version",
        DEFAULT_MODEL_VERSION,
    )

    return normalized_bundle


# ---------------------------------------------------------------------
# Model loading and cache
# ---------------------------------------------------------------------

def load_gp_bundle(
    path: Path | str = GP_MODEL_PATH,
) -> dict[str, Any]:
    """
    Load and validate the versioned Gaussian Process model bundle.
    """

    model_path = Path(path)

    if not model_path.exists():
        raise ModelNotGeneratedError(
            str(model_path)
        )

    try:
        bundle = joblib.load(
            model_path
        )

    except Exception as exc:
        raise ModelLoadingError(
            model_path=str(model_path),
            reason=str(exc),
        ) from exc

    try:
        return validate_gp_bundle(
            bundle
        )

    except Exception as exc:
        if isinstance(
            exc,
            (
                ModelLoadingError,
                FeatureSchemaError,
            ),
        ):
            raise

        raise ModelLoadingError(
            model_path=str(model_path),
            reason=str(exc),
        ) from exc


def set_gp_bundle(
    bundle: Mapping[str, Any],
    *,
    source_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Install a validated GP bundle in the module cache.
    """

    global _GP_BUNDLE
    global _GP_BUNDLE_PATH

    validated_bundle = (
        validate_gp_bundle(
            bundle
        )
    )

    with _GP_BUNDLE_LOCK:
        _GP_BUNDLE = (
            validated_bundle
        )

        _GP_BUNDLE_PATH = (
            Path(source_path)
            if source_path is not None
            else None
        )

    return validated_bundle


def ensure_gp_bundle(
    path: Path | str = GP_MODEL_PATH,
) -> dict[str, Any]:
    """
    Return the cached model bundle or load it on first use.
    """

    global _GP_BUNDLE
    global _GP_BUNDLE_PATH

    requested_path = Path(
        path
    )

    with _GP_BUNDLE_LOCK:
        if (
            _GP_BUNDLE is not None
            and (
                _GP_BUNDLE_PATH is None
                or _GP_BUNDLE_PATH
                == requested_path
            )
        ):
            return _GP_BUNDLE

        loaded_bundle = load_gp_bundle(
            requested_path
        )

        _GP_BUNDLE = loaded_bundle
        _GP_BUNDLE_PATH = (
            requested_path
        )

        return loaded_bundle


def clear_gp_bundle_cache() -> None:
    """
    Clear the loaded GP model from memory.
    """

    global _GP_BUNDLE
    global _GP_BUNDLE_PATH

    with _GP_BUNDLE_LOCK:
        _GP_BUNDLE = None
        _GP_BUNDLE_PATH = None


# ---------------------------------------------------------------------
# Threshold policy
# ---------------------------------------------------------------------

def calculate_bayes_risk_threshold(
    *,
    false_accept_cost: float = (
        BAYES_COST_FALSE_ACCEPT
    ),
    false_reject_cost: float = (
        BAYES_COST_FALSE_REJECT
    ),
) -> float:
    """
    Calculate the theoretical cost-sensitive Bayes threshold.

    Formula:

        threshold =
            false_reject_cost
            -------------------------------
            false_accept_cost + false_reject_cost
    """

    if (
        isinstance(false_accept_cost, bool)
        or not isinstance(
            false_accept_cost,
            (int, float),
        )
        or float(false_accept_cost) <= 0.0
    ):
        raise ProtocolValidationError(
            (
                "false_accept_cost must be "
                "a positive number."
            )
        )

    if (
        isinstance(false_reject_cost, bool)
        or not isinstance(
            false_reject_cost,
            (int, float),
        )
        or float(false_reject_cost) <= 0.0
    ):
        raise ProtocolValidationError(
            (
                "false_reject_cost must be "
                "a positive number."
            )
        )

    threshold = (
        float(false_reject_cost)
        / (
            float(false_accept_cost)
            + float(false_reject_cost)
        )
    )

    return validate_probability(
        threshold,
        field_name="bayes_risk_threshold",
    )


def resolve_gp_threshold_policy(
    *,
    bundle: Mapping[str, Any] | None = None,
    explicit_threshold: float | None = None,
    threshold_path: Path | str = THRESHOLD_PATH,
    minimum_operational_threshold: float = (
        MINIMUM_OPERATIONAL_GP_THRESHOLD
    ),
    gray_zone_retry_upper: float | None = None,
    maximum_uncertainty: float | None = (
        GP_MAXIMUM_UNCERTAINTY
    ),
) -> GPThresholdPolicy:
    """
    Resolve the threshold policy using this priority order:

    1. Explicit function argument
    2. Loaded model bundle
    3. threshold.json
    4. Configuration value
    5. Theoretical Bayes-risk threshold

    The operational threshold is never lower than the configured
    operational minimum.
    """

    validated_minimum = (
        validate_probability(
            minimum_operational_threshold,
            field_name=(
                "minimum_operational_threshold"
            ),
        )
    )

    threshold_data = _load_json_mapping(
        threshold_path
    )

    selected_threshold: float | None = None

    raw_calibration_threshold: float | None = None

    threshold_source = (
        "bayes_risk_fallback"
    )

    if explicit_threshold is not None:
        selected_threshold = (
            validate_probability(
                explicit_threshold,
                field_name=(
                    "explicit_gp_threshold"
                ),
            )
        )

        threshold_source = (
            "explicit_argument"
        )

    if bundle is not None:
        bundle_raw_threshold = bundle.get(
            "raw_calibration_gp_attack_threshold"
        )

        if bundle_raw_threshold is not None:
            raw_calibration_threshold = (
                validate_probability(
                    bundle_raw_threshold,
                    field_name=(
                        "raw_calibration_gp_attack_threshold"
                    ),
                )
            )

        if selected_threshold is None:
            bundle_threshold = bundle.get(
                "gp_attack_threshold"
            )

            if bundle_threshold is not None:
                selected_threshold = (
                    validate_probability(
                        bundle_threshold,
                        field_name=(
                            "bundle_gp_attack_threshold"
                        ),
                    )
                )

                threshold_source = (
                    "model_bundle"
                )

    if raw_calibration_threshold is None:
        threshold_file_raw = threshold_data.get(
            "raw_calibration_gp_attack_threshold"
        )

        if threshold_file_raw is not None:
            raw_calibration_threshold = (
                validate_probability(
                    threshold_file_raw,
                    field_name=(
                        "threshold_file_raw_threshold"
                    ),
                )
            )

    if selected_threshold is None:
        threshold_file_value = (
            threshold_data.get(
                "gp_attack_threshold"
            )
        )

        if threshold_file_value is not None:
            selected_threshold = (
                validate_probability(
                    threshold_file_value,
                    field_name=(
                        "threshold_file_gp_threshold"
                    ),
                )
            )

            threshold_source = (
                "threshold_file"
            )

    if (
        selected_threshold is None
        and DEFAULT_GP_ATTACK_THRESHOLD
        is not None
    ):
        selected_threshold = (
            validate_probability(
                DEFAULT_GP_ATTACK_THRESHOLD,
                field_name=(
                    "configured_gp_attack_threshold"
                ),
            )
        )

        threshold_source = (
            "configuration"
        )

    if selected_threshold is None:
        selected_threshold = (
            calculate_bayes_risk_threshold()
        )

        threshold_source = (
            "bayes_risk_fallback"
        )

    if raw_calibration_threshold is None:
        raw_calibration_threshold = (
            selected_threshold
        )

    operational_threshold = max(
        selected_threshold,
        validated_minimum,
    )

    operational_threshold = float(
        np.clip(
            operational_threshold,
            0.0,
            1.0,
        )
    )

    if gray_zone_retry_upper is None:
        threshold_file_gray_upper = (
            threshold_data.get(
                "gp_gray_zone_retry_upper"
            )
        )

        if threshold_file_gray_upper is None:
            selected_gray_upper = (
                GP_GRAY_ZONE_RETRY_UPPER
            )

        else:
            selected_gray_upper = (
                threshold_file_gray_upper
            )

    else:
        selected_gray_upper = (
            gray_zone_retry_upper
        )

    validated_gray_upper = (
        validate_probability(
            selected_gray_upper,
            field_name=(
                "gray_zone_retry_upper"
            ),
        )
    )

    validated_maximum_uncertainty = (
        None
        if maximum_uncertainty is None
        else validate_probability(
            maximum_uncertainty,
            field_name=(
                "maximum_uncertainty"
            ),
        )
    )

    return GPThresholdPolicy(
        raw_calibration_threshold=(
            raw_calibration_threshold
        ),

        operational_threshold=(
            operational_threshold
        ),

        gray_zone_retry_upper=(
            validated_gray_upper
        ),

        minimum_operational_threshold=(
            validated_minimum
        ),

        maximum_uncertainty=(
            validated_maximum_uncertainty
        ),

        threshold_source=(
            threshold_source
        ),
    )


# ---------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------

def build_gp_feature_row(
    features: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Validate and create one ordered GP feature row.
    """

    validated_features = (
        validate_feature_dictionary(
            features
        )
    )

    row = pd.DataFrame(
        [
            [
                validated_features[
                    feature_name
                ]
                for feature_name
                in FEATURE_COLUMNS
            ]
        ],
        columns=list(
            FEATURE_COLUMNS
        ),
    )

    return row


# ---------------------------------------------------------------------
# Probability prediction
# ---------------------------------------------------------------------

def _get_attack_class_index(
    model: Any,
) -> int:
    """
    Find the probability-column index for attack class label 1.

    The notebook normally uses column index 1. This helper also supports
    models whose class ordering is explicitly stored.
    """

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        return 1

    normalized_classes = list(
        np.asarray(
            classes
        ).reshape(-1)
    )

    if ATTACK_CLASS_LABEL not in normalized_classes:
        raise MachineLearningError(
            (
                "The GP model does not contain "
                "the attack class label 1."
            ),
            code="GP_ATTACK_CLASS_MISSING",
            details={
                "classes": [
                    str(value)
                    for value
                    in normalized_classes
                ],
            },
        )

    return normalized_classes.index(
        ATTACK_CLASS_LABEL
    )


def predict_raw_attack_probability(
    features: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any] | None = None,
    model_path: Path | str = GP_MODEL_PATH,
) -> float:
    """
    Predict the uncalibrated GP attack probability.
    """

    selected_bundle = (
        ensure_gp_bundle(
            model_path
        )
        if bundle is None
        else validate_gp_bundle(
            bundle
        )
    )

    feature_row = build_gp_feature_row(
        features
    )

    model = selected_bundle[
        "model"
    ]

    try:
        probabilities = np.asarray(
            model.predict_proba(
                feature_row
            ),
            dtype=float,
        )

    except Exception as exc:
        raise MachineLearningError(
            "Gaussian Process prediction failed.",
            code="GP_PREDICTION_ERROR",
            details={
                "reason": str(exc),
                "feature_columns": list(
                    FEATURE_COLUMNS
                ),
            },
        ) from exc

    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != 1
        or probabilities.shape[1] < 2
    ):
        raise MachineLearningError(
            (
                "Gaussian Process predict_proba returned "
                "an invalid probability matrix."
            ),
            code="GP_INVALID_PROBABILITY_OUTPUT",
            details={
                "shape": list(
                    probabilities.shape
                ),
            },
        )

    attack_class_index = (
        _get_attack_class_index(
            model
        )
    )

    if attack_class_index >= (
        probabilities.shape[1]
    ):
        raise MachineLearningError(
            (
                "Attack-class index exceeds the "
                "probability matrix width."
            ),
            code="GP_INVALID_ATTACK_CLASS_INDEX",
        )

    raw_probability = float(
        probabilities[
            0,
            attack_class_index,
        ]
    )

    return float(
        np.clip(
            raw_probability,
            0.0,
            1.0,
        )
    )


def calibrate_attack_probability(
    raw_probability: float,
    *,
    calibrator: Any | None,
) -> tuple[float, bool]:
    """
    Apply the optional isotonic probability calibrator.

    Returns:

        calibrated_probability,
        calibration_applied
    """

    validated_raw_probability = (
        validate_probability(
            raw_probability,
            field_name=(
                "raw_attack_probability"
            ),
        )
    )

    if calibrator is None:
        return (
            validated_raw_probability,
            False,
        )

    if not callable(
        getattr(
            calibrator,
            "predict",
            None,
        )
    ):
        raise MachineLearningError(
            (
                "The probability calibrator does "
                "not provide predict()."
            ),
            code="GP_CALIBRATOR_API_ERROR",
        )

    try:
        calibrated_output = np.asarray(
            calibrator.predict(
                np.asarray(
                    [
                        validated_raw_probability
                    ],
                    dtype=float,
                )
            ),
            dtype=float,
        ).reshape(-1)

    except Exception as exc:
        raise MachineLearningError(
            "GP probability calibration failed.",
            code="GP_CALIBRATION_ERROR",
            details={
                "reason": str(exc),
            },
        ) from exc

    if len(calibrated_output) != 1:
        raise MachineLearningError(
            (
                "The GP calibrator returned an "
                "invalid number of probabilities."
            ),
            code="GP_INVALID_CALIBRATION_OUTPUT",
            details={
                "output_length": len(
                    calibrated_output
                ),
            },
        )

    calibrated_probability = float(
        np.clip(
            calibrated_output[0],
            0.0,
            1.0,
        )
    )

    return (
        calibrated_probability,
        True,
    )


def predict_attack_probability(
    features: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any] | None = None,
    model_path: Path | str = GP_MODEL_PATH,
) -> tuple[float, float, bool]:
    """
    Predict raw and calibrated attack probabilities.

    Returns:

        raw_probability,
        final_probability,
        calibration_applied
    """

    selected_bundle = (
        ensure_gp_bundle(
            model_path
        )
        if bundle is None
        else validate_gp_bundle(
            bundle
        )
    )

    raw_probability = (
        predict_raw_attack_probability(
            features,
            bundle=selected_bundle,
            model_path=model_path,
        )
    )

    final_probability, calibrated = (
        calibrate_attack_probability(
            raw_probability,
            calibrator=selected_bundle.get(
                "calibrator"
            ),
        )
    )

    return (
        raw_probability,
        final_probability,
        calibrated,
    )


# ---------------------------------------------------------------------
# Predictive uncertainty
# ---------------------------------------------------------------------

def gp_predictive_uncertainty(
    attack_probability: float,
) -> float:
    """
    Calculate normalized binary entropy.

    Formula:

        H(p) =
            -p log2(p)
            -(1-p) log2(1-p)

    The maximum binary entropy is 1 bit.
    """

    validated_probability = (
        validate_probability(
            attack_probability,
            field_name="attack_probability",
        )
    )

    clipped_probability = float(
        np.clip(
            validated_probability,
            1e-12,
            1.0 - 1e-12,
        )
    )

    uncertainty = -(
        clipped_probability
        * math.log2(
            clipped_probability
        )
        + (
            1.0
            - clipped_probability
        )
        * math.log2(
            1.0
            - clipped_probability
        )
    )

    return float(
        np.clip(
            uncertainty,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------
# GP policy evaluation
# ---------------------------------------------------------------------

def apply_gp_policy(
    *,
    deterministic_pass: bool,
    attack_probability: float,
    uncertainty: float,
    threshold_policy: GPThresholdPolicy,
) -> tuple[
    AuthenticationDecision,
    str,
    bool,
    bool,
]:
    """
    Apply deterministic-first GP policy.

    Returns:

        recommended_decision,
        reason,
        predicted_attack,
        within_retry_gray_zone
    """

    if not isinstance(
        deterministic_pass,
        bool,
    ):
        raise ProtocolValidationError(
            "deterministic_pass must be Boolean."
        )

    validated_probability = (
        validate_probability(
            attack_probability,
            field_name="attack_probability",
        )
    )

    validated_uncertainty = (
        validate_probability(
            uncertainty,
            field_name="uncertainty",
        )
    )

    if not isinstance(
        threshold_policy,
        GPThresholdPolicy,
    ):
        raise ProtocolValidationError(
            (
                "threshold_policy must be a "
                "GPThresholdPolicy object."
            )
        )

    if not deterministic_pass:
        return (
            AuthenticationDecision.REJECT,
            DECISION_REASON_DETERMINISTIC_FAILURE,
            True,
            False,
        )

    if (
        threshold_policy.maximum_uncertainty
        is not None
        and validated_uncertainty
        > threshold_policy.maximum_uncertainty
    ):
        return (
            AuthenticationDecision.REJECT,
            DECISION_REASON_UNCERTAINTY,
            True,
            False,
        )

    predicted_attack = (
        validated_probability
        >= threshold_policy.operational_threshold
    )

    if not predicted_attack:
        return (
            AuthenticationDecision.ACCEPT,
            DECISION_REASON_ACCEPT,
            False,
            False,
        )

    within_gray_zone = (
        threshold_policy.gray_zone_enabled
        and validated_probability
        < threshold_policy.gray_zone_retry_upper
    )

    if within_gray_zone:
        return (
            AuthenticationDecision.RETRY,
            DECISION_REASON_RETRY,
            True,
            True,
        )

    return (
        AuthenticationDecision.REJECT,
        DECISION_REASON_REJECT,
        True,
        False,
    )


# ---------------------------------------------------------------------
# Complete detector
# ---------------------------------------------------------------------

def detect_gp_attack(
    *,
    features: Mapping[str, Any],
    deterministic_pass: bool,
    bundle: Mapping[str, Any] | None = None,
    model_path: Path | str = GP_MODEL_PATH,
    threshold_path: Path | str = THRESHOLD_PATH,
    explicit_threshold: float | None = None,
    gray_zone_retry_upper: float | None = None,
    maximum_uncertainty: float | None = (
        GP_MAXIMUM_UNCERTAINTY
    ),
) -> GPAttackDetectionResult:
    """
    Run complete GP attack detection and policy evaluation.

    When deterministic_pass is False, the GP model is not evaluated and
    the protocol fails closed.
    """

    if not isinstance(
        deterministic_pass,
        bool,
    ):
        raise ProtocolValidationError(
            "deterministic_pass must be Boolean."
        )

    validated_features = (
        validate_feature_dictionary(
            features
        )
    )

    selected_bundle = (
        ensure_gp_bundle(
            model_path
        )
        if bundle is None
        else validate_gp_bundle(
            bundle
        )
    )

    threshold_policy = (
        resolve_gp_threshold_policy(
            bundle=selected_bundle,
            explicit_threshold=(
                explicit_threshold
            ),
            threshold_path=threshold_path,
            gray_zone_retry_upper=(
                gray_zone_retry_upper
            ),
            maximum_uncertainty=(
                maximum_uncertainty
            ),
        )
    )

    model_name = str(
        selected_bundle.get(
            "model_name",
            DEFAULT_MODEL_NAME,
        )
    )

    model_version = str(
        selected_bundle.get(
            "protocol_version",
            DEFAULT_MODEL_VERSION,
        )
    )

    training_source = (
        selected_bundle.get(
            "training_source"
        )
    )

    calibration_method = (
        selected_bundle.get(
            "calibration_method"
        )
    )

    metadata: dict[str, Any] = {
        "model_path": str(
            model_path
        ),
        "feature_columns": list(
            FEATURE_COLUMNS
        ),
        "uncertainty_method": (
            UNCERTAINTY_METHOD
        ),
        "bundle_seed": (
            selected_bundle.get(
                "seed"
            )
        ),
        "bundle_metrics": (
            selected_bundle.get(
                "metrics",
                {},
            )
        ),
    }

    if not deterministic_pass:
        return GPAttackDetectionResult(
            evaluated=False,
            deterministic_pass=False,

            raw_attack_probability=None,
            attack_probability=None,
            uncertainty=None,

            predicted_attack=None,
            within_retry_gray_zone=False,

            recommended_decision=(
                AuthenticationDecision.REJECT
            ),

            reason=(
                DECISION_REASON_DETERMINISTIC_FAILURE
            ),

            threshold_policy=(
                threshold_policy
            ),

            calibrated=(
                selected_bundle.get(
                    "calibrator"
                )
                is not None
            ),

            model_name=model_name,
            model_version=model_version,

            training_source=(
                None
                if training_source is None
                else str(training_source)
            ),

            calibration_method=(
                None
                if calibration_method is None
                else str(calibration_method)
            ),

            features=validated_features,
            metadata=metadata,
        )

    (
        raw_probability,
        attack_probability,
        calibrated,
    ) = predict_attack_probability(
        validated_features,
        bundle=selected_bundle,
        model_path=model_path,
    )

    uncertainty = (
        gp_predictive_uncertainty(
            attack_probability
        )
    )

    (
        recommended_decision,
        reason,
        predicted_attack,
        within_retry_gray_zone,
    ) = apply_gp_policy(
        deterministic_pass=True,
        attack_probability=(
            attack_probability
        ),
        uncertainty=uncertainty,
        threshold_policy=(
            threshold_policy
        ),
    )

    return GPAttackDetectionResult(
        evaluated=True,
        deterministic_pass=True,

        raw_attack_probability=(
            raw_probability
        ),

        attack_probability=(
            attack_probability
        ),

        uncertainty=uncertainty,

        predicted_attack=(
            predicted_attack
        ),

        within_retry_gray_zone=(
            within_retry_gray_zone
        ),

        recommended_decision=(
            recommended_decision
        ),

        reason=reason,

        threshold_policy=(
            threshold_policy
        ),

        calibrated=calibrated,

        model_name=model_name,
        model_version=model_version,

        training_source=(
            None
            if training_source is None
            else str(training_source)
        ),

        calibration_method=(
            None
            if calibration_method is None
            else str(calibration_method)
        ),

        features=validated_features,
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def get_gp_detector_information(
    *,
    model_path: Path | str = GP_MODEL_PATH,
    threshold_path: Path | str = THRESHOLD_PATH,
    metadata_path: Path | str = MODEL_METADATA_PATH,
) -> dict[str, Any]:
    """
    Return model, threshold, and metadata availability information.
    """

    selected_model_path = Path(
        model_path
    )

    selected_threshold_path = Path(
        threshold_path
    )

    selected_metadata_path = Path(
        metadata_path
    )

    information: dict[str, Any] = {
        "model_path": str(
            selected_model_path
        ),

        "model_exists": (
            selected_model_path.exists()
        ),

        "threshold_path": str(
            selected_threshold_path
        ),

        "threshold_exists": (
            selected_threshold_path.exists()
        ),

        "metadata_path": str(
            selected_metadata_path
        ),

        "metadata_exists": (
            selected_metadata_path.exists()
        ),

        "expected_feature_columns": list(
            FEATURE_COLUMNS
        ),
    }

    if selected_threshold_path.exists():
        information["threshold_data"] = (
            _load_json_mapping(
                selected_threshold_path
            )
        )

    if selected_metadata_path.exists():
        information["model_metadata"] = (
            _load_json_mapping(
                selected_metadata_path
            )
        )

    if selected_model_path.exists():
        try:
            bundle = load_gp_bundle(
                selected_model_path
            )

            information.update(
                {
                    "bundle_valid": True,

                    "protocol_version": (
                        bundle.get(
                            "protocol_version"
                        )
                    ),

                    "training_source": (
                        bundle.get(
                            "training_source"
                        )
                    ),

                    "calibration_method": (
                        bundle.get(
                            "calibration_method"
                        )
                    ),

                    "has_calibrator": (
                        bundle.get(
                            "calibrator"
                        )
                        is not None
                    ),

                    "bundle_threshold": (
                        bundle.get(
                            "gp_attack_threshold"
                        )
                    ),
                }
            )

        except Exception as exc:
            information.update(
                {
                    "bundle_valid": False,
                    "bundle_error": str(exc),
                }
            )

    return information


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

class _SelfTestModel:
    """
    Minimal predict_proba model used without an external model file.
    """

    classes_ = np.asarray(
        [0, 1],
        dtype=int,
    )

    def __init__(
        self,
        attack_probability: float,
    ) -> None:
        self.attack_probability = float(
            attack_probability
        )

    def predict_proba(
        self,
        feature_row: pd.DataFrame,
    ) -> np.ndarray:
        if list(
            feature_row.columns
        ) != list(
            FEATURE_COLUMNS
        ):
            raise ValueError(
                "Feature order mismatch."
            )

        probability = float(
            np.clip(
                self.attack_probability,
                0.0,
                1.0,
            )
        )

        return np.asarray(
            [
                [
                    1.0 - probability,
                    probability,
                ]
            ],
            dtype=float,
        )


class _IdentityCalibrator:
    """
    Self-test calibrator that leaves probabilities unchanged.
    """

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            probabilities,
            dtype=float,
        )


def run_gp_attack_detector_self_test() -> dict[str, Any]:
    """
    Run deterministic ACCEPT, RETRY, REJECT, and fail-closed tests.
    """

    features = {
        "qber_raw": 0.03,
        "mean_syndrome_weight": 0.20,
        "max_syndrome_weight": 1.00,
        "correction_failure_rate": 0.00,
        "loss_rate": 0.02,
        "noise_estimate": 0.03,
        "ctx_urban": 1.0,
        "ctx_suburban": 0.0,
        "ctx_rural": 0.0,
    }

    common_bundle_data = {
        "feature_columns": list(
            FEATURE_COLUMNS
        ),
        "protocol_version": (
            "self-test-v1"
        ),
        "calibrator": (
            _IdentityCalibrator()
        ),
        "raw_calibration_gp_attack_threshold": (
            0.15
        ),
        "gp_attack_threshold": 0.15,
        "training_source": "self_test",
        "calibration_method": "identity",
    }

    accept_bundle = {
        **common_bundle_data,
        "model": _SelfTestModel(
            0.10
        ),
    }

    retry_bundle = {
        **common_bundle_data,
        "model": _SelfTestModel(
            0.17
        ),
    }

    reject_bundle = {
        **common_bundle_data,
        "model": _SelfTestModel(
            0.90
        ),
    }

    accept_result = detect_gp_attack(
        features=features,
        deterministic_pass=True,
        bundle=accept_bundle,
        explicit_threshold=0.15,
        gray_zone_retry_upper=0.20,
    )

    retry_result = detect_gp_attack(
        features=features,
        deterministic_pass=True,
        bundle=retry_bundle,
        explicit_threshold=0.15,
        gray_zone_retry_upper=0.20,
    )

    reject_result = detect_gp_attack(
        features=features,
        deterministic_pass=True,
        bundle=reject_bundle,
        explicit_threshold=0.15,
        gray_zone_retry_upper=0.20,
    )

    deterministic_failure_result = (
        detect_gp_attack(
            features=features,
            deterministic_pass=False,
            bundle=accept_bundle,
            explicit_threshold=0.15,
            gray_zone_retry_upper=0.20,
        )
    )

    uncertainty_test = (
        gp_predictive_uncertainty(
            0.5
        )
    )

    success = all(
        (
            accept_result.evaluated,

            accept_result.recommended_decision
            == AuthenticationDecision.ACCEPT,

            retry_result.recommended_decision
            == AuthenticationDecision.RETRY,

            retry_result.within_retry_gray_zone,

            reject_result.recommended_decision
            == AuthenticationDecision.REJECT,

            not deterministic_failure_result.evaluated,

            deterministic_failure_result
            .recommended_decision
            == AuthenticationDecision.REJECT,

            math.isclose(
                uncertainty_test,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
        )
    )

    return {
        "success": success,

        "accept_probability": (
            accept_result.attack_probability
        ),

        "accept_decision": (
            accept_result
            .recommended_decision
            .value
        ),

        "retry_probability": (
            retry_result.attack_probability
        ),

        "retry_decision": (
            retry_result
            .recommended_decision
            .value
        ),

        "retry_gray_zone": (
            retry_result
            .within_retry_gray_zone
        ),

        "reject_probability": (
            reject_result.attack_probability
        ),

        "reject_decision": (
            reject_result
            .recommended_decision
            .value
        ),

        "deterministic_failure_evaluated": (
            deterministic_failure_result
            .evaluated
        ),

        "deterministic_failure_decision": (
            deterministic_failure_result
            .recommended_decision
            .value
        ),

        "maximum_entropy_at_half": (
            uncertainty_test
        ),
    }


__all__ = [
    "ATTACK_CLASS_LABEL",
    "DEFAULT_MODEL_NAME",
    "UNCERTAINTY_METHOD",
    "GPThresholdPolicy",
    "GPAttackDetectionResult",
    "validate_gp_bundle",
    "load_gp_bundle",
    "set_gp_bundle",
    "ensure_gp_bundle",
    "clear_gp_bundle_cache",
    "calculate_bayes_risk_threshold",
    "resolve_gp_threshold_policy",
    "build_gp_feature_row",
    "predict_raw_attack_probability",
    "calibrate_attack_probability",
    "predict_attack_probability",
    "gp_predictive_uncertainty",
    "apply_gp_policy",
    "detect_gp_attack",
    "get_gp_detector_information",
    "run_gp_attack_detector_self_test",
]