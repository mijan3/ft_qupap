"""
FT-QuPAP Gaussian Process Predictor

This module performs live attack-probability prediction using the
trained FT-QuPAP Gaussian Process model.

Notebook-compatible prediction flow:

    1. Validate the nine receiver-observable features.
    2. Preserve the exact FEATURE_COLUMNS order.
    3. Apply an external StandardScaler only when the model does not
       already contain its own scaler.
    4. Calculate the raw attack probability using predict_proba().
    5. Apply the optional isotonic probability calibrator.
    6. Clip the calibrated probability to [0, 1].
    7. Calculate normalized binary predictive entropy.

The returned uncertainty value is the entropy of the calibrated attack
probability. It is not a direct posterior variance from the Gaussian
Process classifier.

Security boundary:

Only Authentication Server observable features may be supplied. Hidden
simulator information such as Eve's configured interception fraction,
attack positions, scenario label, or ground-truth attack flag must not
enter the predictor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from .feature_preprocessor import (
    make_feature_frame,
    transform_feature_table,
)
from .feature_schema import (
    FEATURE_COLUMNS,
    validate_feature_order,
)
from .gp_model_loader import (
    DEFAULT_MODEL_DIRECTORY,
    GPModelBundle,
    load_gp_model_bundle,
)


MIN_ENTROPY_PROBABILITY = 1e-12
POSITIVE_ATTACK_CLASS = 1

_DEFAULT_BUNDLE: GPModelBundle | None = None
_DEFAULT_BUNDLE_LOCK = RLock()


class GPPredictorError(Exception):
    """Base exception for Gaussian Process prediction failures."""


class GPBundleUnavailableError(GPPredictorError):
    """Raised when no trained GP model bundle is available."""


class InvalidGPPredictionError(GPPredictorError):
    """Raised when a model returns malformed probability output."""


class PositiveClassNotFoundError(GPPredictorError):
    """Raised when the attack class cannot be identified."""


class ProbabilityCalibrationError(GPPredictorError):
    """Raised when probability calibration fails."""


class BatchPredictionError(GPPredictorError):
    """Raised when one or more batch feature records are invalid."""


def validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate a finite probability in the interval [0, 1].
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise InvalidGPPredictionError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise InvalidGPPredictionError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def clip_probability(
    value: Any,
    field_name: str = "probability",
) -> float:
    """
    Convert a finite numeric value and clip it into [0, 1].
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise InvalidGPPredictionError(
            f"{field_name} must be finite."
        )

    return float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
    )


def gp_predictive_uncertainty(
    attack_probability: float,
) -> float:
    """
    Calculate normalized binary entropy for P(attack).

    Formula:

        H(p) = -p log2(p) - (1-p) log2(1-p)

    Binary entropy has a maximum value of one bit at p = 0.5.
    Therefore, the returned value is already normalized to [0, 1].

    Low entropy:
        The model strongly favors either benign or attack.

    High entropy:
        The probability is near 0.5 and the prediction is ambiguous.
    """

    probability = clip_probability(
        attack_probability,
        "attack_probability",
    )

    clipped_probability = float(
        np.clip(
            probability,
            MIN_ENTROPY_PROBABILITY,
            1.0 - MIN_ENTROPY_PROBABILITY,
        )
    )

    entropy = -(
        clipped_probability
        * math.log2(clipped_probability)
        + (
            1.0 - clipped_probability
        )
        * math.log2(
            1.0 - clipped_probability
        )
    )

    return float(
        np.clip(
            entropy,
            0.0,
            1.0,
        )
    )


def normalize_gp_bundle(
    bundle: GPModelBundle | Mapping[str, Any],
) -> GPModelBundle:
    """
    Convert a GPModelBundle or notebook-style dictionary.

    Supported notebook-style keys include:

        model
        calibrator
        scaler
        feature_columns
        protocol_version
        seed
        gp_attack_threshold
        raw_calibration_gp_attack_threshold
        calibration_method
        training_source
    """

    if isinstance(
        bundle,
        GPModelBundle,
    ):
        return bundle

    if not isinstance(
        bundle,
        Mapping,
    ):
        raise TypeError(
            "gp_bundle must be GPModelBundle or Mapping."
        )

    if "model" not in bundle:
        raise GPBundleUnavailableError(
            "GP bundle does not contain a model."
        )

    feature_columns = bundle.get(
        "feature_columns",
        FEATURE_COLUMNS,
    )

    try:
        validate_feature_order(
            feature_columns
        )

    except Exception as error:
        raise InvalidGPPredictionError(
            "GP bundle feature order does not match "
            "the FT-QuPAP feature schema."
        ) from error

    known_keys = {
        "model",
        "calibrator",
        "scaler",
        "feature_columns",
        "protocol_version",
        "seed",
        "gp_attack_threshold",
        "raw_calibration_gp_attack_threshold",
        "calibration_method",
        "training_source",
    }

    metadata = {
        str(key): value
        for key, value in bundle.items()
        if key not in known_keys
    }

    return GPModelBundle(
        model=bundle["model"],
        feature_columns=tuple(
            feature_columns
        ),
        protocol_version=bundle.get(
            "protocol_version"
        ),
        seed=bundle.get(
            "seed"
        ),
        calibrator=bundle.get(
            "calibrator"
        ),
        scaler=bundle.get(
            "scaler"
        ),
        gp_attack_threshold=bundle.get(
            "gp_attack_threshold"
        ),
        raw_calibration_gp_attack_threshold=(
            bundle.get(
                "raw_calibration_gp_attack_threshold"
            )
        ),
        calibration_method=bundle.get(
            "calibration_method"
        ),
        training_source=bundle.get(
            "training_source"
        ),
        metadata=metadata,
    )


def set_default_gp_bundle(
    bundle: GPModelBundle | Mapping[str, Any],
) -> GPModelBundle:
    """
    Install the process-wide default GP model bundle.
    """

    normalized_bundle = normalize_gp_bundle(
        bundle
    )

    global _DEFAULT_BUNDLE

    with _DEFAULT_BUNDLE_LOCK:
        _DEFAULT_BUNDLE = normalized_bundle

    return normalized_bundle


def clear_default_gp_bundle() -> None:
    """
    Remove the cached default GP bundle.
    """

    global _DEFAULT_BUNDLE

    with _DEFAULT_BUNDLE_LOCK:
        _DEFAULT_BUNDLE = None


def get_default_gp_bundle(
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
    force_reload: bool = False,
) -> GPModelBundle:
    """
    Load or return the cached default GP model bundle.
    """

    global _DEFAULT_BUNDLE

    if not isinstance(
        force_reload,
        bool,
    ):
        raise TypeError(
            "force_reload must be boolean."
        )

    with _DEFAULT_BUNDLE_LOCK:
        if (
            _DEFAULT_BUNDLE is not None
            and not force_reload
        ):
            return _DEFAULT_BUNDLE

        try:
            loaded_bundle = load_gp_model_bundle(
                model_directory=model_directory,
            )

        except Exception as error:
            raise GPBundleUnavailableError(
                "The FT-QuPAP GP model bundle could not "
                "be loaded."
            ) from error

        _DEFAULT_BUNDLE = loaded_bundle

        return loaded_bundle


def resolve_gp_bundle(
    gp_bundle: (
        GPModelBundle
        | Mapping[str, Any]
        | None
    ),
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
) -> GPModelBundle:
    """
    Resolve an explicit or default GP bundle.
    """

    if gp_bundle is None:
        return get_default_gp_bundle(
            model_directory=model_directory,
        )

    return normalize_gp_bundle(
        gp_bundle
    )


def prepare_prediction_frame(
    features: Mapping[str, Any],
    bundle: GPModelBundle,
) -> pd.DataFrame:
    """
    Build the exact feature input expected by the loaded model.

    An external scaler is used only when:

        - the bundle contains a scaler, and
        - the model itself does not already contain a scaler.
    """

    if not isinstance(
        bundle,
        GPModelBundle,
    ):
        raise TypeError(
            "bundle must be GPModelBundle."
        )

    raw_frame = make_feature_frame(
        features
    )

    if bundle.external_scaler_required:
        if bundle.scaler is None:
            raise GPBundleUnavailableError(
                "The model requires an external feature scaler, "
                "but no scaler was loaded."
            )

        return transform_feature_table(
            raw_frame,
            bundle.scaler,
            allow_extra_columns=False,
        )

    return raw_frame


def resolve_positive_class_index(
    model: Any,
    probability_column_count: int,
    *,
    positive_class: int = (
        POSITIVE_ATTACK_CLASS
    ),
) -> int:
    """
    Resolve the probability column for the attack class.

    The FT-QuPAP model uses:

        benign = 0
        attack = 1
    """

    if (
        isinstance(
            probability_column_count,
            bool,
        )
        or not isinstance(
            probability_column_count,
            int,
        )
    ):
        raise TypeError(
            "probability_column_count must be an integer."
        )

    if probability_column_count < 1:
        raise InvalidGPPredictionError(
            "The model returned no probability columns."
        )

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is not None:
        normalized_classes = list(
            np.asarray(
                classes
            ).reshape(-1)
        )

        matching_indices = [
            index
            for index, class_value
            in enumerate(normalized_classes)
            if class_value == positive_class
        ]

        if len(matching_indices) != 1:
            raise PositiveClassNotFoundError(
                f"Could not uniquely identify attack class "
                f"{positive_class!r} in model classes "
                f"{normalized_classes!r}."
            )

        selected_index = matching_indices[0]

        if selected_index >= probability_column_count:
            raise InvalidGPPredictionError(
                "Model classes_ does not match the "
                "predict_proba output width."
            )

        return selected_index

    # Notebook-compatible fallback for a normal two-column binary model.
    if probability_column_count == 2:
        return 1

    raise PositiveClassNotFoundError(
        "The model does not expose classes_, and the attack "
        "probability column cannot be determined safely."
    )


def predict_raw_attack_probability(
    features: Mapping[str, Any],
    gp_bundle: (
        GPModelBundle
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
) -> float:
    """
    Predict the uncalibrated probability of attack.

    Notebook equivalent:

        raw_probability = float(
            gp_bundle["model"].predict_proba(feature_row)[0, 1]
        )
    """

    bundle = resolve_gp_bundle(
        gp_bundle,
        model_directory=model_directory,
    )

    feature_frame = prepare_prediction_frame(
        features,
        bundle,
    )

    try:
        probability_output = (
            bundle.model.predict_proba(
                feature_frame
            )
        )

    except Exception as error:
        raise InvalidGPPredictionError(
            "The GP model could not predict attack probability."
        ) from error

    try:
        probability_matrix = np.asarray(
            probability_output,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidGPPredictionError(
            "The model returned nonnumeric probabilities."
        ) from error

    if probability_matrix.ndim != 2:
        raise InvalidGPPredictionError(
            "predict_proba() must return a two-dimensional matrix."
        )

    if probability_matrix.shape[0] != 1:
        raise InvalidGPPredictionError(
            "Single-session prediction must return exactly one row."
        )

    if not np.all(
        np.isfinite(
            probability_matrix
        )
    ):
        raise InvalidGPPredictionError(
            "The model returned NaN or infinite probabilities."
        )

    positive_class_index = (
        resolve_positive_class_index(
            bundle.model,
            probability_matrix.shape[1],
        )
    )

    raw_probability = probability_matrix[
        0,
        positive_class_index,
    ]

    return clip_probability(
        raw_probability,
        "raw_attack_probability",
    )


def _extract_calibrated_probability(
    output: Any,
    calibrator: Any,
) -> float:
    """
    Extract one attack probability from calibrator output.
    """

    try:
        values = np.asarray(
            output,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ProbabilityCalibrationError(
            "The calibrator returned nonnumeric output."
        ) from error

    if values.size == 0:
        raise ProbabilityCalibrationError(
            "The calibrator returned no probability."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise ProbabilityCalibrationError(
            "The calibrator returned NaN or infinity."
        )

    if values.ndim == 1:
        if values.shape[0] != 1:
            raise ProbabilityCalibrationError(
                "Single-probability calibration returned "
                "an unexpected output length."
            )

        return clip_probability(
            values[0],
            "calibrated_attack_probability",
        )

    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ProbabilityCalibrationError(
                "Single-probability calibration must return "
                "exactly one row."
            )

        positive_class_index = (
            resolve_positive_class_index(
                calibrator,
                values.shape[1],
            )
        )

        return clip_probability(
            values[
                0,
                positive_class_index,
            ],
            "calibrated_attack_probability",
        )

    raise ProbabilityCalibrationError(
        "Unsupported calibrator output shape."
    )


def apply_probability_calibrator(
    raw_probability: float,
    calibrator: Any | None,
) -> float:
    """
    Apply an optional probability calibrator.

    IsotonicRegression uses:

        calibrator.predict([raw_probability])

    Other compatible calibrators may provide transform() or
    predict_proba().
    """

    raw_probability = validate_probability(
        raw_probability,
        "raw_probability",
    )

    if calibrator is None:
        return raw_probability

    one_dimensional_input = np.asarray(
        [
            raw_probability,
        ],
        dtype=float,
    )

    two_dimensional_input = (
        one_dimensional_input.reshape(
            -1,
            1,
        )
    )

    try:
        predict_method = getattr(
            calibrator,
            "predict",
            None,
        )

        if callable(
            predict_method
        ):
            output = predict_method(
                one_dimensional_input
            )

            return _extract_calibrated_probability(
                output,
                calibrator,
            )

        transform_method = getattr(
            calibrator,
            "transform",
            None,
        )

        if callable(
            transform_method
        ):
            output = transform_method(
                one_dimensional_input
            )

            return _extract_calibrated_probability(
                output,
                calibrator,
            )

        predict_proba_method = getattr(
            calibrator,
            "predict_proba",
            None,
        )

        if callable(
            predict_proba_method
        ):
            output = predict_proba_method(
                two_dimensional_input
            )

            return _extract_calibrated_probability(
                output,
                calibrator,
            )

    except ProbabilityCalibrationError:
        raise

    except Exception as error:
        raise ProbabilityCalibrationError(
            "The probability calibrator failed."
        ) from error

    raise ProbabilityCalibrationError(
        "The calibration model does not provide a supported "
        "prediction method."
    )


@dataclass(frozen=True)
class GPPrediction:
    """
    Result from one FT-QuPAP GP inference operation.
    """

    raw_attack_probability: float
    attack_probability: float
    uncertainty: float

    calibrated: bool
    model_name: str

    feature_columns: tuple[str, ...]
    positive_class: int = POSITIVE_ATTACK_CLASS

    operational_threshold: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_attack_probability",
            validate_probability(
                self.raw_attack_probability,
                "raw_attack_probability",
            ),
        )

        object.__setattr__(
            self,
            "attack_probability",
            validate_probability(
                self.attack_probability,
                "attack_probability",
            ),
        )

        object.__setattr__(
            self,
            "uncertainty",
            validate_probability(
                self.uncertainty,
                "uncertainty",
            ),
        )

        if not isinstance(
            self.calibrated,
            bool,
        ):
            raise TypeError(
                "calibrated must be boolean."
            )

        if not isinstance(
            self.model_name,
            str,
        ) or not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        try:
            validated_columns = (
                validate_feature_order(
                    self.feature_columns
                )
            )

        except Exception as error:
            raise InvalidGPPredictionError(
                "Prediction feature order is invalid."
            ) from error

        object.__setattr__(
            self,
            "feature_columns",
            tuple(
                validated_columns
            ),
        )

        if (
            isinstance(
                self.positive_class,
                bool,
            )
            or not isinstance(
                self.positive_class,
                int,
            )
        ):
            raise TypeError(
                "positive_class must be an integer."
            )

        if self.operational_threshold is not None:
            object.__setattr__(
                self,
                "operational_threshold",
                validate_probability(
                    self.operational_threshold,
                    "operational_threshold",
                ),
            )

    @property
    def p_attack(self) -> float:
        """Notebook-compatible attack-probability alias."""

        return self.attack_probability

    @property
    def predicted_attack(self) -> bool | None:
        """
        Return the threshold-based attack classification.

        None is returned when no operational threshold was loaded.
        """

        if self.operational_threshold is None:
            return None

        return bool(
            self.attack_probability
            >= self.operational_threshold
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe prediction record."""

        return {
            "raw_attack_probability":
                self.raw_attack_probability,

            "p_attack":
                self.attack_probability,

            "uncertainty":
                self.uncertainty,

            "calibrated":
                self.calibrated,

            "model_name":
                self.model_name,

            "feature_columns":
                list(
                    self.feature_columns
                ),

            "positive_class":
                self.positive_class,

            "operational_threshold":
                self.operational_threshold,

            "predicted_attack":
                self.predicted_attack,
        }


def predict_attack(
    features: Mapping[str, Any],
    gp_bundle: (
        GPModelBundle
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
) -> GPPrediction:
    """
    Perform complete FT-QuPAP GP prediction.

    Returns both raw and calibrated attack probability together with
    normalized binary entropy.
    """

    bundle = resolve_gp_bundle(
        gp_bundle,
        model_directory=model_directory,
    )

    raw_probability = (
        predict_raw_attack_probability(
            features,
            bundle,
        )
    )

    calibrated_probability = (
        apply_probability_calibrator(
            raw_probability,
            bundle.calibrator,
        )
    )

    uncertainty = gp_predictive_uncertainty(
        calibrated_probability
    )

    return GPPrediction(
        raw_attack_probability=(
            raw_probability
        ),
        attack_probability=(
            calibrated_probability
        ),
        uncertainty=uncertainty,
        calibrated=(
            bundle.calibrator is not None
        ),
        model_name=type(
            bundle.model
        ).__name__,
        feature_columns=(
            bundle.feature_columns
        ),
        positive_class=(
            POSITIVE_ATTACK_CLASS
        ),
        operational_threshold=(
            bundle.gp_attack_threshold
        ),
    )


def gp_predict_attack_probability(
    features: Mapping[str, Any],
    gp_bundle: (
        GPModelBundle
        | Mapping[str, Any]
        | None
    ) = None,
) -> float:
    """
    Return the calibrated FT-QuPAP attack probability.

    This preserves the notebook-compatible public function.

    Before calibration:
        The clipped raw model probability is returned.

    After calibration:
        The clipped calibrator output is returned.
    """

    return predict_attack(
        features,
        gp_bundle,
    ).attack_probability


def predict_attack_batch(
    feature_records: Sequence[
        Mapping[str, Any]
    ],
    gp_bundle: (
        GPModelBundle
        | Mapping[str, Any]
        | None
    ) = None,
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
) -> list[GPPrediction]:
    """
    Predict attack probabilities for multiple feature records.

    Each row is validated independently so hidden simulator fields
    cannot enter the prediction matrix.
    """

    if isinstance(
        feature_records,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "feature_records must be a sequence of mappings."
        )

    if not isinstance(
        feature_records,
        Sequence,
    ):
        raise TypeError(
            "feature_records must be a sequence."
        )

    if len(feature_records) == 0:
        return []

    bundle = resolve_gp_bundle(
        gp_bundle,
        model_directory=model_directory,
    )

    predictions: list[
        GPPrediction
    ] = []

    for index, features in enumerate(
        feature_records
    ):
        if not isinstance(
            features,
            Mapping,
        ):
            raise BatchPredictionError(
                f"Feature record {index} must be a mapping."
            )

        try:
            predictions.append(
                predict_attack(
                    features,
                    bundle,
                )
            )

        except Exception as error:
            raise BatchPredictionError(
                f"Prediction failed for feature record {index}."
            ) from error

    return predictions


class FTQuPAPGPPredictor:
    """
    Reusable FT-QuPAP GP attack-prediction service.
    """

    def __init__(
        self,
        bundle: (
            GPModelBundle
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        model_directory: str | Path = (
            DEFAULT_MODEL_DIRECTORY
        ),
        lazy_load: bool = True,
    ) -> None:
        if not isinstance(
            lazy_load,
            bool,
        ):
            raise TypeError(
                "lazy_load must be boolean."
            )

        self.model_directory = Path(
            model_directory
        )

        self._bundle: (
            GPModelBundle | None
        ) = (
            None
            if bundle is None
            else normalize_gp_bundle(
                bundle
            )
        )

        if (
            self._bundle is None
            and not lazy_load
        ):
            self.load()

    @property
    def loaded(self) -> bool:
        """Return whether the predictor currently has a model."""

        return self._bundle is not None

    @property
    def bundle(self) -> GPModelBundle:
        """Return the loaded bundle, loading it when necessary."""

        if self._bundle is None:
            self.load()

        if self._bundle is None:
            raise GPBundleUnavailableError(
                "GP model bundle is unavailable."
            )

        return self._bundle

    def load(
        self,
        *,
        force_reload: bool = False,
    ) -> GPModelBundle:
        """Load the GP artifacts from the configured model directory."""

        if (
            self._bundle is not None
            and not force_reload
        ):
            return self._bundle

        try:
            self._bundle = (
                load_gp_model_bundle(
                    model_directory=(
                        self.model_directory
                    )
                )
            )

        except Exception as error:
            raise GPBundleUnavailableError(
                "Could not load the FT-QuPAP GP model."
            ) from error

        return self._bundle

    def replace_bundle(
        self,
        bundle: GPModelBundle | Mapping[str, Any],
    ) -> None:
        """Replace the currently loaded model bundle."""

        self._bundle = normalize_gp_bundle(
            bundle
        )

    def predict(
        self,
        features: Mapping[str, Any],
    ) -> GPPrediction:
        """Predict attack evidence for one session."""

        return predict_attack(
            features,
            self.bundle,
        )

    def predict_probability(
        self,
        features: Mapping[str, Any],
    ) -> float:
        """Return only calibrated P(attack)."""

        return self.predict(
            features
        ).attack_probability

    def predict_many(
        self,
        feature_records: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[GPPrediction]:
        """Predict multiple session feature records."""

        return predict_attack_batch(
            feature_records,
            self.bundle,
        )

    def describe_model(self) -> dict[str, Any]:
        """Return safe loaded-model metadata."""

        return self.bundle.describe()


def run_self_test() -> None:
    """
    Verify raw prediction, calibration, entropy, and schema handling.
    """

    from sklearn.isotonic import (
        IsotonicRegression,
    )
    from sklearn.linear_model import (
        LogisticRegression,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        StandardScaler,
    )

    training_frame = pd.DataFrame(
        [
            [
                0.01,
                0.10,
                1.0,
                0.00,
                0.01,
                0.01,
                1.0,
                0.0,
                0.0,
            ],
            [
                0.03,
                0.20,
                1.0,
                0.01,
                0.02,
                0.02,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.04,
                0.30,
                2.0,
                0.02,
                0.03,
                0.02,
                0.0,
                0.0,
                1.0,
            ],
            [
                0.20,
                2.00,
                4.0,
                0.20,
                0.05,
                0.10,
                1.0,
                0.0,
                0.0,
            ],
            [
                0.30,
                3.00,
                5.0,
                0.30,
                0.08,
                0.15,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.40,
                4.00,
                6.0,
                0.40,
                0.10,
                0.20,
                0.0,
                0.0,
                1.0,
            ],
        ],
        columns=FEATURE_COLUMNS,
        dtype=float,
    )

    labels = np.asarray(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=int,
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=7,
                    max_iter=300,
                ),
            ),
        ]
    )

    model.fit(
        training_frame,
        labels,
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        np.asarray(
            [
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
            ],
            dtype=float,
        ),
        np.asarray(
            [
                0.0,
                0.10,
                0.50,
                0.90,
                1.0,
            ],
            dtype=float,
        ),
    )

    bundle = GPModelBundle(
        model=model,
        calibrator=calibrator,
        feature_columns=tuple(
            FEATURE_COLUMNS
        ),
        protocol_version=(
            "FT-QuPAP-predictor-test"
        ),
        seed=7,
        gp_attack_threshold=0.15,
        raw_calibration_gp_attack_threshold=0.10,
        calibration_method=(
            "isotonic_regression"
        ),
        training_source="self_test",
    )

    benign_features = {
        "qber_raw": 0.01,
        "mean_syndrome_weight": 0.10,
        "max_syndrome_weight": 1.0,
        "correction_failure_rate": 0.00,
        "loss_rate": 0.01,
        "noise_estimate": 0.01,
        "ctx_urban": 1.0,
        "ctx_suburban": 0.0,
        "ctx_rural": 0.0,
    }

    attack_features = {
        "qber_raw": 0.35,
        "mean_syndrome_weight": 3.50,
        "max_syndrome_weight": 6.0,
        "correction_failure_rate": 0.35,
        "loss_rate": 0.10,
        "noise_estimate": 0.18,
        "ctx_urban": 1.0,
        "ctx_suburban": 0.0,
        "ctx_rural": 0.0,
    }

    benign_prediction = predict_attack(
        benign_features,
        bundle,
    )

    attack_prediction = predict_attack(
        attack_features,
        bundle,
    )

    if not (
        0.0
        <= benign_prediction.attack_probability
        <= 1.0
    ):
        raise GPPredictorError(
            "Benign probability is invalid."
        )

    if not (
        0.0
        <= attack_prediction.attack_probability
        <= 1.0
    ):
        raise GPPredictorError(
            "Attack probability is invalid."
        )

    if (
        attack_prediction.attack_probability
        <= benign_prediction.attack_probability
    ):
        raise GPPredictorError(
            "Attack example did not receive greater risk."
        )

    if not benign_prediction.calibrated:
        raise GPPredictorError(
            "Calibration status is incorrect."
        )

    if (
        benign_prediction.operational_threshold
        != 0.15
    ):
        raise GPPredictorError(
            "Operational threshold was not preserved."
        )

    maximum_entropy = (
        gp_predictive_uncertainty(
            0.5
        )
    )

    if not math.isclose(
        maximum_entropy,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GPPredictorError(
            "Binary entropy at p=0.5 must equal 1."
        )

    batch_predictions = (
        predict_attack_batch(
            [
                benign_features,
                attack_features,
            ],
            bundle,
        )
    )

    if len(batch_predictions) != 2:
        raise GPPredictorError(
            "Batch prediction count is incorrect."
        )

    notebook_probability = (
        gp_predict_attack_probability(
            benign_features,
            bundle,
        )
    )

    if not math.isclose(
        notebook_probability,
        benign_prediction.attack_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GPPredictorError(
            "Notebook-compatible probability differs."
        )

    print(
        "GP predictor self-test completed successfully."
    )

    print(
        "Benign P(attack):",
        f"{benign_prediction.attack_probability:.6f}",
    )

    print(
        "Attack P(attack):",
        f"{attack_prediction.attack_probability:.6f}",
    )

    print(
        "Benign uncertainty:",
        f"{benign_prediction.uncertainty:.6f}",
    )

    print(
        "Calibration applied:",
        benign_prediction.calibrated,
    )


__all__ = [
    "MIN_ENTROPY_PROBABILITY",
    "POSITIVE_ATTACK_CLASS",
    "GPPredictorError",
    "GPBundleUnavailableError",
    "InvalidGPPredictionError",
    "PositiveClassNotFoundError",
    "ProbabilityCalibrationError",
    "BatchPredictionError",
    "GPPrediction",
    "FTQuPAPGPPredictor",
    "validate_probability",
    "clip_probability",
    "gp_predictive_uncertainty",
    "normalize_gp_bundle",
    "set_default_gp_bundle",
    "clear_default_gp_bundle",
    "get_default_gp_bundle",
    "resolve_gp_bundle",
    "prepare_prediction_frame",
    "resolve_positive_class_index",
    "predict_raw_attack_probability",
    "apply_probability_calibrator",
    "predict_attack",
    "gp_predict_attack_probability",
    "predict_attack_batch",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        GPPredictorError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "\n[GP PREDICTOR ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error