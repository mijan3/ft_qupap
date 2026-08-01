"""
FT-QuPAP Probability Calibration

This module calibrates raw Gaussian Process attack probabilities using
isotonic regression.

Notebook-compatible workflow:

    1. Train the GP model using only the training split.
    2. Generate raw P(attack) values for the disjoint calibration split.
    3. Fit IsotonicRegression using calibration probabilities and labels.
    4. Apply the fitted calibrator to calibration, test, and live outputs.
    5. Clip every calibrated probability into [0, 1].
    6. Evaluate calibration using adaptive probability bins and ECE.

Important:

The calibration split must remain separate from:

- GP training data
- held-out test data
- independent evaluation seeds

The calibrator changes probability interpretation. It does not replace
the Gaussian Process classifier and does not independently decide
whether authentication is accepted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


DEFAULT_CALIBRATION_BINS = 10
DEFAULT_MINIMUM_BIN_SIZE = 50

CALIBRATION_METHOD = "isotonic_regression"

SUPPORTED_ARTIFACT_EXTENSIONS = frozenset(
    {
        ".pkl",
        ".joblib",
    }
)


class ProbabilityCalibratorError(Exception):
    """Base exception for probability-calibration failures."""


class InvalidCalibrationDataError(
    ProbabilityCalibratorError
):
    """Raised when labels or probabilities are malformed."""


class CalibrationModelNotFittedError(
    ProbabilityCalibratorError
):
    """Raised when an unfitted calibrator is used."""


class InvalidCalibrationModelError(
    ProbabilityCalibratorError
):
    """Raised when a calibration model is unsupported."""


class CalibrationArtifactError(
    ProbabilityCalibratorError
):
    """Raised when a calibrator artifact cannot be saved or loaded."""


class CalibrationPredictionError(
    ProbabilityCalibratorError
):
    """Raised when the calibrator returns invalid probabilities."""


def create_isotonic_calibrator() -> IsotonicRegression:
    """
    Create the notebook-compatible isotonic calibrator.

    The configured bounds guarantee outputs within [0, 1], while
    out_of_bounds="clip" supports raw probabilities outside the fitted
    calibration range.
    """

    return IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )


def normalize_probability_array(
    probabilities: Any,
    field_name: str = "probabilities",
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    """
    Convert probability input into a finite one-dimensional array.
    """

    try:
        normalized = np.asarray(
            probabilities,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidCalibrationDataError(
            f"{field_name} must contain numeric values."
        ) from error

    if normalized.size == 0 and not allow_empty:
        raise InvalidCalibrationDataError(
            f"{field_name} cannot be empty."
        )

    if not np.all(
        np.isfinite(normalized)
    ):
        raise InvalidCalibrationDataError(
            f"{field_name} cannot contain NaN or infinity."
        )

    if np.any(
        normalized < 0.0
    ) or np.any(
        normalized > 1.0
    ):
        raise InvalidCalibrationDataError(
            f"{field_name} values must be between 0 and 1."
        )

    return normalized.astype(
        float,
        copy=True,
    )


def normalize_binary_labels(
    labels: Any,
    field_name: str = "labels",
    *,
    require_both_classes: bool = False,
) -> np.ndarray:
    """
    Convert labels into a one-dimensional integer binary array.
    """

    try:
        raw_labels = np.asarray(
            labels
        ).reshape(-1)

    except Exception as error:
        raise InvalidCalibrationDataError(
            f"{field_name} could not be converted."
        ) from error

    if raw_labels.size == 0:
        raise InvalidCalibrationDataError(
            f"{field_name} cannot be empty."
        )

    try:
        numeric_labels = raw_labels.astype(
            float
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidCalibrationDataError(
            f"{field_name} must contain binary numeric values."
        ) from error

    if not np.all(
        np.isfinite(numeric_labels)
    ):
        raise InvalidCalibrationDataError(
            f"{field_name} cannot contain NaN or infinity."
        )

    if not np.all(
        np.isin(
            numeric_labels,
            [
                0.0,
                1.0,
            ],
        )
    ):
        raise InvalidCalibrationDataError(
            f"{field_name} must contain only 0 and 1."
        )

    normalized = numeric_labels.astype(
        int
    )

    if (
        require_both_classes
        and set(
            normalized.tolist()
        )
        != {
            0,
            1,
        }
    ):
        raise InvalidCalibrationDataError(
            "Calibration labels must contain both "
            "benign class 0 and attack class 1."
        )

    return normalized


def validate_calibration_pairs(
    raw_probabilities: Any,
    labels: Any,
    *,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate paired raw probabilities and binary labels.
    """

    probabilities = normalize_probability_array(
        raw_probabilities,
        "raw_probabilities",
    )

    normalized_labels = normalize_binary_labels(
        labels,
        "labels",
        require_both_classes=require_both_classes,
    )

    if len(probabilities) != len(
        normalized_labels
    ):
        raise InvalidCalibrationDataError(
            "Raw probabilities and labels must have equal length."
        )

    return (
        probabilities,
        normalized_labels,
    )


def is_calibrator_fitted(
    calibrator: Any,
) -> bool:
    """
    Return whether an isotonic calibrator contains fitted thresholds.
    """

    if not isinstance(
        calibrator,
        IsotonicRegression,
    ):
        return False

    return bool(
        hasattr(
            calibrator,
            "X_thresholds_",
        )
        and hasattr(
            calibrator,
            "y_thresholds_",
        )
        and hasattr(
            calibrator,
            "f_",
        )
    )


def validate_calibrator(
    calibrator: Any,
    *,
    require_fitted: bool = True,
) -> IsotonicRegression:
    """
    Validate an FT-QuPAP isotonic probability calibrator.
    """

    if not isinstance(
        calibrator,
        IsotonicRegression,
    ):
        raise InvalidCalibrationModelError(
            "calibrator must be "
            "sklearn.isotonic.IsotonicRegression."
        )

    if (
        require_fitted
        and not is_calibrator_fitted(
            calibrator
        )
    ):
        raise CalibrationModelNotFittedError(
            "Probability calibrator has not been fitted."
        )

    return calibrator


def fit_probability_calibrator(
    raw_probabilities: Any,
    labels: Any,
) -> IsotonicRegression:
    """
    Fit isotonic calibration using a disjoint calibration split.

    Raw probabilities must come from the already trained GP model.
    """

    (
        probabilities,
        normalized_labels,
    ) = validate_calibration_pairs(
        raw_probabilities,
        labels,
        require_both_classes=True,
    )

    calibrator = create_isotonic_calibrator()

    try:
        calibrator.fit(
            probabilities,
            normalized_labels,
        )

    except Exception as error:
        raise ProbabilityCalibratorError(
            "Could not fit the isotonic probability calibrator."
        ) from error

    return validate_calibrator(
        calibrator,
        require_fitted=True,
    )


def apply_probability_calibrator(
    calibrator: IsotonicRegression,
    raw_probabilities: Any,
) -> np.ndarray:
    """
    Apply a fitted calibrator and clip outputs into [0, 1].
    """

    calibrator = validate_calibrator(
        calibrator,
        require_fitted=True,
    )

    probabilities = normalize_probability_array(
        raw_probabilities,
        "raw_probabilities",
    )

    try:
        calibrated = calibrator.predict(
            probabilities
        )

    except Exception as error:
        raise CalibrationPredictionError(
            "The isotonic calibrator could not predict probabilities."
        ) from error

    try:
        calibrated = np.asarray(
            calibrated,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise CalibrationPredictionError(
            "The calibrator returned nonnumeric output."
        ) from error

    if calibrated.shape != probabilities.shape:
        raise CalibrationPredictionError(
            "The calibrator returned an unexpected output shape."
        )

    if not np.all(
        np.isfinite(calibrated)
    ):
        raise CalibrationPredictionError(
            "The calibrator returned NaN or infinity."
        )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    ).astype(
        float,
        copy=False,
    )


def resolve_positive_class_index(
    model: Any,
    probability_column_count: int,
) -> int:
    """
    Resolve the predict_proba column representing attack class 1.
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
        raise CalibrationPredictionError(
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
            in enumerate(
                normalized_classes
            )
            if class_value == 1
        ]

        if len(matching_indices) != 1:
            raise CalibrationPredictionError(
                "Could not uniquely identify attack class 1."
            )

        selected_index = matching_indices[0]

        if selected_index >= probability_column_count:
            raise CalibrationPredictionError(
                "Model classes_ does not match predict_proba output."
            )

        return selected_index

    if probability_column_count == 2:
        return 1

    raise CalibrationPredictionError(
        "Attack probability column cannot be determined."
    )


def raw_model_probabilities(
    model: Any,
    features: Any,
) -> np.ndarray:
    """
    Generate raw attack probabilities from a fitted classifier.
    """

    predict_proba = getattr(
        model,
        "predict_proba",
        None,
    )

    if not callable(
        predict_proba
    ):
        raise TypeError(
            "model must provide predict_proba()."
        )

    try:
        probability_output = predict_proba(
            features
        )

    except Exception as error:
        raise CalibrationPredictionError(
            "The model could not generate raw probabilities."
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
        raise CalibrationPredictionError(
            "The model returned nonnumeric probabilities."
        ) from error

    if probability_matrix.ndim != 2:
        raise CalibrationPredictionError(
            "predict_proba() must return a two-dimensional matrix."
        )

    if probability_matrix.shape[0] == 0:
        raise CalibrationPredictionError(
            "The model returned no prediction rows."
        )

    if not np.all(
        np.isfinite(
            probability_matrix
        )
    ):
        raise CalibrationPredictionError(
            "The model returned NaN or infinite probabilities."
        )

    positive_class_index = (
        resolve_positive_class_index(
            model,
            probability_matrix.shape[1],
        )
    )

    return normalize_probability_array(
        probability_matrix[
            :,
            positive_class_index,
        ],
        "raw_model_probabilities",
    )


def calibrated_probabilities(
    model: Any,
    calibrator: IsotonicRegression | None,
    features: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return raw and calibrated attack probabilities.

    Notebook-compatible behavior:

        raw_probability = model.predict_proba(features)[:, 1]

        calibrated_probability = np.clip(
            calibrator.predict(raw_probability),
            0.0,
            1.0,
        )

    When calibrator is None, the clipped raw probabilities are returned
    as the final probabilities.
    """

    raw_probabilities = raw_model_probabilities(
        model,
        features,
    )

    if calibrator is None:
        return (
            raw_probabilities,
            raw_probabilities.copy(),
        )

    calibrated = apply_probability_calibrator(
        calibrator,
        raw_probabilities,
    )

    return (
        raw_probabilities,
        calibrated,
    )


def calibration_diagnostics(
    labels: Any,
    probabilities: Any,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    min_bin_size: int = DEFAULT_MINIMUM_BIN_SIZE,
) -> tuple[pd.DataFrame, float]:
    """
    Calculate adaptive calibration bins and expected calibration error.

    The implementation follows the notebook's equal-count binning
    procedure.

    ECE formula:

        sum_b (
            number_in_bin / total_rows
        ) * abs(
            mean_predicted_probability
            - observed_attack_rate
        )
    """

    normalized_labels = normalize_binary_labels(
        labels,
        "labels",
        require_both_classes=False,
    )

    normalized_probabilities = (
        normalize_probability_array(
            probabilities,
            "probabilities",
        )
    )

    if len(normalized_labels) != len(
        normalized_probabilities
    ):
        raise InvalidCalibrationDataError(
            "Labels and probabilities must have equal length."
        )

    if (
        isinstance(n_bins, bool)
        or not isinstance(
            n_bins,
            int,
        )
    ):
        raise TypeError(
            "n_bins must be an integer."
        )

    if n_bins < 1:
        raise ValueError(
            "n_bins must be at least 1."
        )

    if (
        isinstance(min_bin_size, bool)
        or not isinstance(
            min_bin_size,
            int,
        )
    ):
        raise TypeError(
            "min_bin_size must be an integer."
        )

    if min_bin_size < 1:
        raise ValueError(
            "min_bin_size must be at least 1."
        )

    order = np.argsort(
        normalized_probabilities
    )

    ordered_indices = order.tolist()

    target_bins = min(
        n_bins,
        max(
            1,
            len(normalized_labels)
            // min_bin_size,
        ),
    )

    split_indices = np.array_split(
        ordered_indices,
        target_bins,
    )

    rows: list[dict[str, Any]] = []

    expected_calibration_error = 0.0

    for bin_index, indices in enumerate(
        split_indices
    ):
        if len(indices) == 0:
            continue

        indices = np.asarray(
            indices,
            dtype=int,
        )

        bin_probabilities = (
            normalized_probabilities[
                indices
            ]
        )

        bin_labels = normalized_labels[
            indices
        ]

        sample_count = int(
            len(indices)
        )

        mean_predicted_probability = float(
            np.mean(
                bin_probabilities
            )
        )

        observed_attack_rate = float(
            np.mean(
                bin_labels
            )
        )

        calibration_gap = float(
            abs(
                mean_predicted_probability
                - observed_attack_rate
            )
        )

        expected_calibration_error += (
            sample_count
            / len(
                normalized_labels
            )
        ) * calibration_gap

        rows.append(
            {
                "bin_index":
                    int(bin_index),

                "bin_lower":
                    float(
                        np.min(
                            bin_probabilities
                        )
                    ),

                "bin_upper":
                    float(
                        np.max(
                            bin_probabilities
                        )
                    ),

                "sample_count":
                    sample_count,

                "mean_predicted_probability":
                    mean_predicted_probability,

                "observed_attack_rate":
                    observed_attack_rate,

                "absolute_calibration_gap":
                    calibration_gap,
            }
        )

    calibration_table = pd.DataFrame(
        rows,
        columns=[
            "bin_index",
            "bin_lower",
            "bin_upper",
            "sample_count",
            "mean_predicted_probability",
            "observed_attack_rate",
            "absolute_calibration_gap",
        ],
    )

    return (
        calibration_table,
        float(
            expected_calibration_error
        ),
    )


@dataclass(frozen=True)
class CalibrationResult:
    """
    Result of fitting and evaluating one probability calibrator.
    """

    calibrator: IsotonicRegression

    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    labels: np.ndarray

    calibration_curve: pd.DataFrame
    expected_calibration_error: float

    method: str = CALIBRATION_METHOD

    def __post_init__(self) -> None:
        validate_calibrator(
            self.calibrator,
            require_fitted=True,
        )

        raw_probabilities = (
            normalize_probability_array(
                self.raw_probabilities,
                "raw_probabilities",
            )
        )

        calibrated_probabilities = (
            normalize_probability_array(
                self.calibrated_probabilities,
                "calibrated_probabilities",
            )
        )

        labels = normalize_binary_labels(
            self.labels,
            "labels",
            require_both_classes=True,
        )

        if not (
            len(raw_probabilities)
            == len(calibrated_probabilities)
            == len(labels)
        ):
            raise InvalidCalibrationDataError(
                "CalibrationResult arrays must have equal length."
            )

        object.__setattr__(
            self,
            "raw_probabilities",
            raw_probabilities,
        )

        object.__setattr__(
            self,
            "calibrated_probabilities",
            calibrated_probabilities,
        )

        object.__setattr__(
            self,
            "labels",
            labels,
        )

        if not isinstance(
            self.calibration_curve,
            pd.DataFrame,
        ):
            raise TypeError(
                "calibration_curve must be a pandas DataFrame."
            )

        if (
            isinstance(
                self.expected_calibration_error,
                bool,
            )
            or not isinstance(
                self.expected_calibration_error,
                (int, float),
            )
        ):
            raise TypeError(
                "expected_calibration_error must be numeric."
            )

        normalized_ece = float(
            self.expected_calibration_error
        )

        if (
            not math.isfinite(
                normalized_ece
            )
            or not 0.0
            <= normalized_ece
            <= 1.0
        ):
            raise ValueError(
                "expected_calibration_error must be "
                "finite and between 0 and 1."
            )

        object.__setattr__(
            self,
            "expected_calibration_error",
            normalized_ece,
        )

        if (
            not isinstance(
                self.method,
                str,
            )
            or not self.method.strip()
        ):
            raise ValueError(
                "method cannot be empty."
            )

    @property
    def sample_count(self) -> int:
        """Return the number of calibration observations."""

        return len(
            self.labels
        )

    def summary(self) -> dict[str, Any]:
        """
        Return JSON-safe calibration metadata.
        """

        return {
            "calibration_method":
                self.method,

            "calibration_rows":
                self.sample_count,

            "expected_calibration_error":
                self.expected_calibration_error,

            "calibration_bins":
                int(
                    len(
                        self.calibration_curve
                    )
                ),

            "raw_probability_min":
                float(
                    np.min(
                        self.raw_probabilities
                    )
                ),

            "raw_probability_max":
                float(
                    np.max(
                        self.raw_probabilities
                    )
                ),

            "calibrated_probability_min":
                float(
                    np.min(
                        self.calibrated_probabilities
                    )
                ),

            "calibrated_probability_max":
                float(
                    np.max(
                        self.calibrated_probabilities
                    )
                ),
        }


def fit_and_evaluate_calibrator(
    raw_probabilities: Any,
    labels: Any,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    min_bin_size: int = DEFAULT_MINIMUM_BIN_SIZE,
) -> CalibrationResult:
    """
    Fit isotonic regression and evaluate it on the calibration split.
    """

    (
        normalized_probabilities,
        normalized_labels,
    ) = validate_calibration_pairs(
        raw_probabilities,
        labels,
        require_both_classes=True,
    )

    calibrator = fit_probability_calibrator(
        normalized_probabilities,
        normalized_labels,
    )

    calibrated = apply_probability_calibrator(
        calibrator,
        normalized_probabilities,
    )

    (
        calibration_curve,
        expected_calibration_error,
    ) = calibration_diagnostics(
        labels=normalized_labels,
        probabilities=calibrated,
        n_bins=n_bins,
        min_bin_size=min_bin_size,
    )

    return CalibrationResult(
        calibrator=calibrator,
        raw_probabilities=normalized_probabilities,
        calibrated_probabilities=calibrated,
        labels=normalized_labels,
        calibration_curve=calibration_curve,
        expected_calibration_error=(
            expected_calibration_error
        ),
    )


def save_probability_calibrator(
    calibrator: IsotonicRegression,
    destination: str | Path,
) -> Path:
    """
    Save the fitted calibrator as a trusted joblib/pickle artifact.
    """

    calibrator = validate_calibrator(
        calibrator,
        require_fitted=True,
    )

    path = Path(
        destination
    )

    if (
        path.suffix.lower()
        not in SUPPORTED_ARTIFACT_EXTENSIONS
    ):
        raise CalibrationArtifactError(
            "Calibrator path must end with .pkl or .joblib."
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            calibrator,
            path,
        )

    except Exception as error:
        raise CalibrationArtifactError(
            f"Could not save probability calibrator to {path}."
        ) from error

    return path


def load_probability_calibrator(
    source: str | Path,
) -> IsotonicRegression:
    """
    Load a trusted isotonic calibrator artifact.

    Pickle and joblib files must never be loaded from untrusted sources.
    """

    path = Path(
        source
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Probability calibrator does not exist: {path}"
        )

    if not path.is_file():
        raise CalibrationArtifactError(
            f"Calibrator path is not a file: {path}"
        )

    if (
        path.suffix.lower()
        not in SUPPORTED_ARTIFACT_EXTENSIONS
    ):
        raise CalibrationArtifactError(
            "Calibrator path must end with .pkl or .joblib."
        )

    try:
        calibrator = joblib.load(
            path
        )

    except Exception as error:
        raise CalibrationArtifactError(
            f"Could not load probability calibrator from {path}."
        ) from error

    return validate_calibrator(
        calibrator,
        require_fitted=True,
    )


class ProbabilityCalibrator:
    """
    Reusable FT-QuPAP isotonic calibration service.
    """

    def __init__(
        self,
        calibrator: IsotonicRegression | None = None,
    ) -> None:
        self.calibrator = (
            create_isotonic_calibrator()
            if calibrator is None
            else validate_calibrator(
                calibrator,
                require_fitted=False,
            )
        )

    @property
    def fitted(self) -> bool:
        """Return whether the calibrator is fitted."""

        return is_calibrator_fitted(
            self.calibrator
        )

    def fit(
        self,
        raw_probabilities: Any,
        labels: Any,
    ) -> "ProbabilityCalibrator":
        """Fit using a disjoint calibration split."""

        self.calibrator = (
            fit_probability_calibrator(
                raw_probabilities,
                labels,
            )
        )

        return self

    def transform(
        self,
        raw_probabilities: Any,
    ) -> np.ndarray:
        """Apply the fitted probability calibrator."""

        return apply_probability_calibrator(
            self.calibrator,
            raw_probabilities,
        )

    def fit_transform(
        self,
        raw_probabilities: Any,
        labels: Any,
    ) -> np.ndarray:
        """Fit and calibrate the supplied probabilities."""

        self.fit(
            raw_probabilities,
            labels,
        )

        return self.transform(
            raw_probabilities
        )

    def evaluate(
        self,
        labels: Any,
        raw_probabilities: Any,
        *,
        n_bins: int = DEFAULT_CALIBRATION_BINS,
        min_bin_size: int = DEFAULT_MINIMUM_BIN_SIZE,
    ) -> tuple[pd.DataFrame, float]:
        """Evaluate calibration using calibrated probabilities."""

        calibrated = self.transform(
            raw_probabilities
        )

        return calibration_diagnostics(
            labels=labels,
            probabilities=calibrated,
            n_bins=n_bins,
            min_bin_size=min_bin_size,
        )

    def save(
        self,
        destination: str | Path,
    ) -> Path:
        """Save the fitted calibrator."""

        return save_probability_calibrator(
            self.calibrator,
            destination,
        )

    @classmethod
    def load(
        cls,
        source: str | Path,
    ) -> "ProbabilityCalibrator":
        """Load a trusted calibrator artifact."""

        return cls(
            calibrator=(
                load_probability_calibrator(
                    source
                )
            )
        )


def run_self_test() -> None:
    """
    Verify fitting, monotonic calibration, diagnostics, and persistence.
    """

    import tempfile

    raw_probabilities = np.asarray(
        [
            0.02,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.55,
            0.65,
            0.75,
            0.90,
            0.95,
            0.99,
        ],
        dtype=float,
    )

    labels = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=int,
    )

    result = fit_and_evaluate_calibrator(
        raw_probabilities,
        labels,
        n_bins=4,
        min_bin_size=2,
    )

    if not is_calibrator_fitted(
        result.calibrator
    ):
        raise ProbabilityCalibratorError(
            "Calibrator was not fitted."
        )

    if result.sample_count != len(
        labels
    ):
        raise ProbabilityCalibratorError(
            "Calibration sample count is incorrect."
        )

    if not np.all(
        np.diff(
            result.calibrated_probabilities
        )
        >= -1e-12
    ):
        raise ProbabilityCalibratorError(
            "Isotonic outputs are not monotonic."
        )

    if not np.all(
        (
            result.calibrated_probabilities
            >= 0.0
        )
        & (
            result.calibrated_probabilities
            <= 1.0
        )
    ):
        raise ProbabilityCalibratorError(
            "Calibrated probabilities are outside [0, 1]."
        )

    if not (
        0.0
        <= result.expected_calibration_error
        <= 1.0
    ):
        raise ProbabilityCalibratorError(
            "Expected calibration error is invalid."
        )

    expected_columns = [
        "bin_index",
        "bin_lower",
        "bin_upper",
        "sample_count",
        "mean_predicted_probability",
        "observed_attack_rate",
        "absolute_calibration_gap",
    ]

    if list(
        result.calibration_curve.columns
    ) != expected_columns:
        raise ProbabilityCalibratorError(
            "Calibration diagnostic columns are incorrect."
        )

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = (
            Path(directory)
            / "calibration_model.pkl"
        )

        save_probability_calibrator(
            result.calibrator,
            artifact_path,
        )

        loaded_calibrator = (
            load_probability_calibrator(
                artifact_path
            )
        )

        loaded_probabilities = (
            apply_probability_calibrator(
                loaded_calibrator,
                raw_probabilities,
            )
        )

        if not np.allclose(
            loaded_probabilities,
            result.calibrated_probabilities,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ProbabilityCalibratorError(
                "Loaded calibrator changed the probabilities."
            )

    print(
        "Probability calibrator self-test "
        "completed successfully."
    )

    print(
        "Calibration method:",
        result.method,
    )

    print(
        "Calibration rows:",
        result.sample_count,
    )

    print(
        "Calibration bins:",
        len(
            result.calibration_curve
        ),
    )

    print(
        "Expected calibration error:",
        f"{result.expected_calibration_error:.6f}",
    )


__all__ = [
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_MINIMUM_BIN_SIZE",
    "CALIBRATION_METHOD",
    "SUPPORTED_ARTIFACT_EXTENSIONS",
    "ProbabilityCalibratorError",
    "InvalidCalibrationDataError",
    "CalibrationModelNotFittedError",
    "InvalidCalibrationModelError",
    "CalibrationArtifactError",
    "CalibrationPredictionError",
    "CalibrationResult",
    "ProbabilityCalibrator",
    "create_isotonic_calibrator",
    "normalize_probability_array",
    "normalize_binary_labels",
    "validate_calibration_pairs",
    "is_calibrator_fitted",
    "validate_calibrator",
    "fit_probability_calibrator",
    "apply_probability_calibrator",
    "resolve_positive_class_index",
    "raw_model_probabilities",
    "calibrated_probabilities",
    "calibration_diagnostics",
    "fit_and_evaluate_calibrator",
    "save_probability_calibrator",
    "load_probability_calibrator",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ProbabilityCalibratorError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "\n[PROBABILITY CALIBRATOR ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error