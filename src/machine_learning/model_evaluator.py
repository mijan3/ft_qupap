"""
FT-QuPAP Gaussian Process Model Evaluator

This module evaluates the final calibrated FT-QuPAP Gaussian Process
attack detector.

Evaluation scopes
=================

1. Held-out session-level test split

   - The split must not have been used for GP training.
   - The split must not have been used for isotonic calibration.
   - The split must not have been used for threshold selection.
   - The already selected operational threshold remains fixed.

2. Independent multi-seed protocol experiments

   - Evaluation seeds must be disjoint from training, calibration,
     and held-out test seeds.
   - The same fixed model, calibrator, feature order, and threshold
     are used without retraining.
   - Probability quality and end-to-end authentication behavior are
     reported separately.

Reported GP metrics
===================

- ROC-AUC
- PR-AUC
- Brier score
- expected calibration error
- selected operational threshold

Reported authentication metrics
================================

- attack detection rate
- attack acceptance / false-accept rate
- valid-user acceptance rate
- false-reject rate
- tag-recovery rate
- rejection-reason distribution

Decision-path separation
========================

The evaluator reports:

1. End-to-end FT-QuPAP outcomes over every P1 session.
2. GP-only outcomes among deterministic-pass sessions.

This prevents mandatory deterministic failures such as invalid
credentials, replay, malformed schedules, decoder failure, or tag
mismatch from being incorrectly credited to the adaptive GP detector.

Security boundary
=================

Only receiver-observable FEATURE_COLUMNS are supplied to the model.
Hidden simulator metadata may remain in offline result tables, but it
is never selected as model input.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .feature_preprocessor import (
    transform_feature_table,
    validate_feature_frame,
)
from .feature_schema import (
    FEATURE_COLUMNS,
    validate_feature_order,
)
from .gp_model_loader import GPModelBundle
from .gp_predictor import (
    gp_predictive_uncertainty,
    normalize_gp_bundle,
    resolve_positive_class_index,
)
from .metrics import (
    GPMetricsReport,
    authentication_decision_metrics,
    bootstrap_mean_interval,
    build_rejection_reason_distribution,
    evaluate_gp_metrics,
    probability_metrics,
)
from .model_trainer import (
    DEFAULT_LABEL_COLUMN,
    GPTrainingResult,
)
from .probability_calibrator import (
    apply_probability_calibrator,
)


DEFAULT_P1_PROTOCOL_NAME = "P1_FT_QuPAP_GP"

DEFAULT_PROTOCOL_COLUMN = "protocol"
DEFAULT_ACTUAL_ATTACK_COLUMN = "actual_attack"
DEFAULT_ACCEPTED_COLUMN = "accepted"
DEFAULT_DETERMINISTIC_PASS_COLUMN = "deterministic_pass"
DEFAULT_DETERMINISTIC_REASONS_COLUMN = "deterministic_reasons"
DEFAULT_REASON_COLUMN = "reason"
DEFAULT_PROBABILITY_COLUMN = "p_attack"
DEFAULT_SEED_COLUMN = "seed"
DEFAULT_TAG_RECOVERED_COLUMN = "tag_recovered"

DEFAULT_BOOTSTRAP_RESAMPLES = 5000
DEFAULT_BOOTSTRAP_RANDOM_STATE = 20260701

DEFAULT_CALIBRATION_BINS = 10
DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE = 30


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "results"
)

DEFAULT_GP_PERFORMANCE_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "gp_performance_metrics.csv"
)

DEFAULT_HELDOUT_PREDICTIONS_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "gp_heldout_predictions.csv"
)

DEFAULT_INDEPENDENT_PREDICTIONS_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "independent_p1_predictions.csv"
)

DEFAULT_CALIBRATION_CURVE_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "gp_calibration_curve.csv"
)

DEFAULT_SECURITY_METRICS_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "security_metrics.csv"
)

DEFAULT_GP_DECISION_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "gp_decision_path_metrics.csv"
)

DEFAULT_REJECTION_REASON_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "rejection_reason_distribution.csv"
)

DEFAULT_SEED_METRICS_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "paper_seed_metrics.csv"
)

DEFAULT_CONFIDENCE_INTERVAL_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "paper_confidence_intervals.csv"
)

DEFAULT_EVALUATION_METADATA_PATH = (
    DEFAULT_RESULTS_DIRECTORY
    / "model_evaluation_metadata.json"
)


class ModelEvaluatorError(Exception):
    """Base exception for FT-QuPAP model-evaluation failures."""


class InvalidEvaluationDatasetError(
    ModelEvaluatorError
):
    """Raised when an evaluation table is malformed."""


class EvaluationThresholdUnavailableError(
    ModelEvaluatorError
):
    """Raised when no operational GP threshold is available."""


class EvaluationPredictionError(
    ModelEvaluatorError
):
    """Raised when the GP or calibrator returns invalid output."""


class EvaluationSeedOverlapError(
    ModelEvaluatorError
):
    """Raised when independent seeds overlap model-development seeds."""


class EvaluationArtifactError(
    ModelEvaluatorError
):
    """Raised when evaluation artifacts cannot be exported."""


class MissingEvaluationClassError(
    ModelEvaluatorError
):
    """Raised when an evaluation set does not contain both classes."""


def validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate one finite probability in [0, 1].
    """

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def validate_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Validate an integer greater than zero.
    """

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (
                int,
                np.integer,
            ),
        )
    ):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    normalized = int(value)

    if normalized < 1:
        raise ValueError(
            f"{field_name} must be at least 1."
        )

    return normalized


def json_safe(
    value: Any,
) -> Any:
    """
    Convert common NumPy, pandas, and project objects to JSON-safe data.
    """

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return json_safe(
            value.item()
        )

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.DataFrame):
        return [
            json_safe(record)
            for record
            in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(value, pd.Series):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    return str(value)


def normalize_binary_labels(
    labels: Any,
    *,
    field_name: str,
    require_both_classes: bool = True,
) -> np.ndarray:
    """
    Normalize benign/attack labels.

    Class definitions:

        0 = benign
        1 = attack
    """

    try:
        normalized = np.asarray(
            labels,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidEvaluationDatasetError(
            f"{field_name} must contain numeric values."
        ) from error

    if normalized.size == 0:
        raise InvalidEvaluationDatasetError(
            f"{field_name} cannot be empty."
        )

    if not np.all(
        np.isfinite(normalized)
    ):
        raise InvalidEvaluationDatasetError(
            f"{field_name} cannot contain NaN or infinity."
        )

    if not np.all(
        np.isin(
            normalized,
            [
                0.0,
                1.0,
            ],
        )
    ):
        raise InvalidEvaluationDatasetError(
            f"{field_name} must contain only 0 and 1."
        )

    integer_labels = normalized.astype(
        int
    )

    if (
        require_both_classes
        and set(
            integer_labels.tolist()
        )
        != {
            0,
            1,
        }
    ):
        raise MissingEvaluationClassError(
            f"{field_name} must contain benign class 0 "
            "and attack class 1."
        )

    return integer_labels


def normalize_boolean_value(
    value: Any,
    field_name: str,
) -> bool:
    """
    Normalize one boolean or binary 0/1 value.
    """

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(value)

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric_value = float(value)

        if (
            math.isfinite(numeric_value)
            and numeric_value
            in {
                0.0,
                1.0,
            }
        ):
            return bool(
                int(numeric_value)
            )

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
        }:
            return False

    raise InvalidEvaluationDatasetError(
        f"{field_name} must be boolean or binary 0/1."
    )


def normalize_boolean_series(
    values: Sequence[Any],
    field_name: str,
) -> pd.Series:
    """
    Normalize a complete boolean result-table column.
    """

    normalized = [
        normalize_boolean_value(
            value,
            f"{field_name}[{index}]",
        )
        for index, value
        in enumerate(values)
    ]

    return pd.Series(
        normalized,
        dtype=bool,
    )


def normalize_reason_value(
    value: Any,
) -> str:
    """
    Normalize a decision or deterministic-reason field.
    """

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return ";".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    return str(value).strip()


def resolve_model_bundle(
    source: (
        GPTrainingResult
        | GPModelBundle
        | Mapping[str, Any]
    ),
) -> GPModelBundle:
    """
    Resolve training results or loaded artifacts into GPModelBundle.
    """

    if isinstance(source, GPTrainingResult):
        return GPModelBundle(
            model=source.model,
            feature_columns=tuple(
                FEATURE_COLUMNS
            ),
            protocol_version=(
                source.config.protocol_version
            ),
            seed=source.config.random_state,
            calibrator=source.calibrator,
            gp_attack_threshold=(
                source.operational_threshold
            ),
            raw_calibration_gp_attack_threshold=(
                source.raw_calibration_threshold
            ),
            calibration_method=(
                "isotonic_regression"
            ),
            training_source=(
                source.config.training_source
            ),
            metadata={
                "session_gp_data_mode":
                    source.config.session_gp_data_mode,

                "session_gp_split_sizes":
                    source.split_sizes,
            },
        )

    if isinstance(source, GPModelBundle):
        return source

    if isinstance(source, Mapping):
        return normalize_gp_bundle(
            source
        )

    raise TypeError(
        "source must be GPTrainingResult, GPModelBundle, "
        "or a notebook-style model bundle mapping."
    )


def require_operational_threshold(
    bundle: GPModelBundle,
) -> float:
    """
    Return the fixed operational threshold.
    """

    if not isinstance(
        bundle,
        GPModelBundle,
    ):
        raise TypeError(
            "bundle must be GPModelBundle."
        )

    if bundle.gp_attack_threshold is None:
        raise EvaluationThresholdUnavailableError(
            "The loaded model bundle does not contain "
            "gp_attack_threshold."
        )

    return validate_probability(
        bundle.gp_attack_threshold,
        "gp_attack_threshold",
    )


def prepare_evaluation_features(
    table: pd.DataFrame,
    bundle: GPModelBundle,
) -> pd.DataFrame:
    """
    Select and preprocess the nine model features.

    An external scaler is applied only when the model does not already
    contain a scaler.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in table.columns
    ]

    if missing_columns:
        raise InvalidEvaluationDatasetError(
            "Evaluation table is missing feature columns: "
            f"{missing_columns!r}"
        )

    raw_features = validate_feature_frame(
        table.loc[
            :,
            FEATURE_COLUMNS,
        ],
        allow_extra_columns=False,
        reject_hidden_columns=True,
    )

    validate_feature_order(
        list(
            raw_features.columns
        )
    )

    if bundle.external_scaler_required:
        if bundle.scaler is None:
            raise EvaluationPredictionError(
                "The GP model requires an external scaler, "
                "but no scaler was loaded."
            )

        return transform_feature_table(
            raw_features,
            bundle.scaler,
            allow_extra_columns=False,
        )

    return raw_features


def predict_raw_probabilities(
    table: pd.DataFrame,
    bundle: GPModelBundle,
) -> np.ndarray:
    """
    Generate uncalibrated attack probabilities for a table.
    """

    model_input = prepare_evaluation_features(
        table,
        bundle,
    )

    try:
        probability_output = np.asarray(
            bundle.model.predict_proba(
                model_input
            ),
            dtype=float,
        )

    except Exception as error:
        raise EvaluationPredictionError(
            "The GP model could not generate evaluation probabilities."
        ) from error

    if probability_output.ndim != 2:
        raise EvaluationPredictionError(
            "predict_proba() must return a two-dimensional matrix."
        )

    if probability_output.shape[0] != len(table):
        raise EvaluationPredictionError(
            "The model prediction row count does not match "
            "the evaluation table."
        )

    if not np.all(
        np.isfinite(
            probability_output
        )
    ):
        raise EvaluationPredictionError(
            "The model returned NaN or infinite probabilities."
        )

    positive_class_index = (
        resolve_positive_class_index(
            bundle.model,
            probability_output.shape[1],
        )
    )

    raw_probabilities = probability_output[
        :,
        positive_class_index,
    ]

    return np.clip(
        raw_probabilities,
        0.0,
        1.0,
    ).astype(
        float,
        copy=False,
    )


def predict_calibrated_probabilities(
    table: pd.DataFrame,
    bundle: GPModelBundle,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate raw and final calibrated P(attack) values.
    """

    raw_probabilities = predict_raw_probabilities(
        table,
        bundle,
    )

    if bundle.calibrator is None:
        return (
            raw_probabilities,
            raw_probabilities.copy(),
        )

    try:
        calibrated_probabilities = (
            apply_probability_calibrator(
                bundle.calibrator,
                raw_probabilities,
            )
        )

    except Exception as error:
        raise EvaluationPredictionError(
            "The GP probability calibrator failed."
        ) from error

    if (
        calibrated_probabilities.shape
        != raw_probabilities.shape
    ):
        raise EvaluationPredictionError(
            "Calibrated probability shape does not match "
            "the raw probability shape."
        )

    return (
        raw_probabilities,
        np.clip(
            calibrated_probabilities,
            0.0,
            1.0,
        ),
    )


def uncertainty_values(
    probabilities: Sequence[float],
) -> np.ndarray:
    """
    Calculate notebook-compatible binary-entropy uncertainty.
    """

    return np.asarray(
        [
            gp_predictive_uncertainty(
                float(probability)
            )
            for probability in probabilities
        ],
        dtype=float,
    )


def conditional_threshold_rate(
    table: pd.DataFrame,
    threshold: float,
    *,
    reject_when_above: bool,
) -> float:
    """
    Return a conditional GP threshold event rate.

    NaN is returned when the conditional table is empty.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    threshold = validate_probability(
        threshold,
        "threshold",
    )

    if table.empty:
        return float("nan")

    if DEFAULT_PROBABILITY_COLUMN not in table.columns:
        raise InvalidEvaluationDatasetError(
            f"Conditional table is missing "
            f"{DEFAULT_PROBABILITY_COLUMN!r}."
        )

    probabilities = pd.to_numeric(
        table[
            DEFAULT_PROBABILITY_COLUMN
        ],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise InvalidEvaluationDatasetError(
            "Conditional GP probabilities cannot contain missing values."
        )

    rejected = probabilities.to_numpy(
        dtype=float
    ) >= threshold

    if reject_when_above:
        return float(
            np.mean(rejected)
        )

    return float(
        np.mean(~rejected)
    )


@dataclass(frozen=True)
class HeldOutEvaluation:
    """
    Final evaluation on the untouched session-level test split.
    """

    predictions: pd.DataFrame
    report: GPMetricsReport
    metrics_row: dict[str, Any]

    model_protocol_version: str | None
    model_seed: int | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.predictions,
            pd.DataFrame,
        ):
            raise TypeError(
                "predictions must be a pandas DataFrame."
            )

        if self.predictions.empty:
            raise InvalidEvaluationDatasetError(
                "Held-out predictions cannot be empty."
            )

        if not isinstance(
            self.report,
            GPMetricsReport,
        ):
            raise TypeError(
                "report must be GPMetricsReport."
            )

        if not isinstance(
            self.metrics_row,
            dict,
        ):
            raise TypeError(
                "metrics_row must be a dictionary."
            )

    @property
    def threshold(self) -> float:
        """Return the fixed operational threshold."""

        return float(
            self.report.selected_threshold
        )

    @property
    def sample_count(self) -> int:
        """Return the number of held-out sessions."""

        return int(
            len(
                self.predictions
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe held-out evaluation metadata."""

        return {
            "evaluation_set":
                "heldout_session_level_test",

            "sample_count":
                self.sample_count,

            "model_protocol_version":
                self.model_protocol_version,

            "model_seed":
                self.model_seed,

            **copy.deepcopy(
                self.metrics_row
            ),
        }


@dataclass(frozen=True)
class IndependentEvaluation:
    """
    Evaluation using disjoint multi-seed protocol experiments.
    """

    normalized_results: pd.DataFrame
    p1_predictions: pd.DataFrame

    performance_row: dict[str, Any]

    decision_path_table: pd.DataFrame
    security_metrics_table: pd.DataFrame
    rejection_reason_table: pd.DataFrame

    seed_metrics_table: pd.DataFrame
    confidence_interval_table: pd.DataFrame

    seed_provenance: dict[str, Any]

    def __post_init__(self) -> None:
        dataframe_fields = (
            "normalized_results",
            "p1_predictions",
            "decision_path_table",
            "security_metrics_table",
            "rejection_reason_table",
            "seed_metrics_table",
            "confidence_interval_table",
        )

        for field_name in dataframe_fields:
            value = getattr(
                self,
                field_name,
            )

            if not isinstance(
                value,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"{field_name} must be a pandas DataFrame."
                )

        if self.p1_predictions.empty:
            raise InvalidEvaluationDatasetError(
                "Independent P1 predictions cannot be empty."
            )

        if not isinstance(
            self.performance_row,
            dict,
        ):
            raise TypeError(
                "performance_row must be a dictionary."
            )

        if not isinstance(
            self.seed_provenance,
            dict,
        ):
            raise TypeError(
                "seed_provenance must be a dictionary."
            )

    @property
    def session_count(self) -> int:
        """Return independent P1 session count."""

        return int(
            len(
                self.p1_predictions
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe independent-evaluation metadata."""

        return {
            "evaluation_set":
                "independent_paper_experiment",

            "session_count":
                self.session_count,

            "seed_provenance":
                copy.deepcopy(
                    self.seed_provenance
                ),

            **copy.deepcopy(
                self.performance_row
            ),
        }


@dataclass(frozen=True)
class ModelEvaluationResult:
    """
    Combined held-out and optional independent evaluation result.
    """

    heldout: HeldOutEvaluation

    independent: IndependentEvaluation | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.heldout,
            HeldOutEvaluation,
        ):
            raise TypeError(
                "heldout must be HeldOutEvaluation."
            )

        if (
            self.independent is not None
            and not isinstance(
                self.independent,
                IndependentEvaluation,
            )
        ):
            raise TypeError(
                "independent must be IndependentEvaluation or None."
            )

    @property
    def performance_table(self) -> pd.DataFrame:
        """
        Return notebook-compatible held-out and independent rows.
        """

        rows = [
            copy.deepcopy(
                self.heldout.metrics_row
            )
        ]

        if self.independent is not None:
            rows.append(
                copy.deepcopy(
                    self.independent
                    .performance_row
                )
            )

        return pd.DataFrame(
            rows
        )

    def metadata(self) -> dict[str, Any]:
        """
        Return complete evaluation metadata.
        """

        return {
            "heldout_test_evaluated":
                True,

            "independent_evaluation_completed":
                self.independent is not None,

            "heldout":
                self.heldout.as_dict(),

            "independent": (
                None
                if self.independent is None
                else self.independent.as_dict()
            ),

            "evaluated_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }


def evaluate_heldout_table(
    source: (
        GPTrainingResult
        | GPModelBundle
        | Mapping[str, Any]
    ),
    test_table: pd.DataFrame,
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    n_calibration_bins: int = (
        DEFAULT_CALIBRATION_BINS
    ),
    minimum_calibration_bin_size: int = (
        DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE
    ),
) -> HeldOutEvaluation:
    """
    Evaluate the fixed calibrated model on an untouched test table.
    """

    if not isinstance(test_table, pd.DataFrame):
        raise TypeError(
            "test_table must be a pandas DataFrame."
        )

    if test_table.empty:
        raise InvalidEvaluationDatasetError(
            "Held-out test table cannot be empty."
        )

    if label_column not in test_table.columns:
        raise InvalidEvaluationDatasetError(
            f"Held-out table is missing label column "
            f"{label_column!r}."
        )

    bundle = resolve_model_bundle(
        source
    )

    threshold = require_operational_threshold(
        bundle
    )

    labels = normalize_binary_labels(
        test_table[
            label_column
        ],
        field_name=label_column,
        require_both_classes=True,
    )

    (
        raw_probabilities,
        calibrated_probabilities,
    ) = predict_calibrated_probabilities(
        test_table,
        bundle,
    )

    report = evaluate_gp_metrics(
        labels=labels,
        probabilities=(
            calibrated_probabilities
        ),
        threshold=threshold,
        evaluation_scope=(
            "heldout_session_level_test"
        ),
        n_calibration_bins=(
            n_calibration_bins
        ),
        minimum_calibration_bin_size=(
            minimum_calibration_bin_size
        ),
    )

    predictions = test_table.copy()

    predictions[
        DEFAULT_ACTUAL_ATTACK_COLUMN
    ] = labels

    predictions[
        "raw_attack_probability"
    ] = raw_probabilities

    predictions[
        DEFAULT_PROBABILITY_COLUMN
    ] = calibrated_probabilities

    predictions[
        "uncertainty"
    ] = uncertainty_values(
        calibrated_probabilities
    )

    predictions[
        "predicted_attack"
    ] = (
        calibrated_probabilities
        >= threshold
    )

    predictions[
        DEFAULT_ACCEPTED_COLUMN
    ] = ~predictions[
        "predicted_attack"
    ]

    probability_result = probability_metrics(
        labels,
        calibrated_probabilities,
    )

    decision_metrics = (
        report.decision_metrics
    )

    metrics_row = {
        "evaluation_set":
            "heldout_session_level_test",

        "roc_auc":
            probability_result[
                "roc_auc"
            ],

        "pr_auc":
            probability_result[
                "pr_auc"
            ],

        "brier_score":
            probability_result[
                "brier_score"
            ],

        "expected_calibration_error":
            report.expected_calibration_error,

        "selected_attack_threshold":
            threshold,

        "test_session_count":
            int(
                len(
                    predictions
                )
            ),

        "attack_session_count":
            int(
                np.sum(
                    labels == 1
                )
            ),

        "benign_session_count":
            int(
                np.sum(
                    labels == 0
                )
            ),

        "attack_detection_rate":
            decision_metrics
            .attack_detection_rate,

        "attack_acceptance_rate":
            decision_metrics
            .attack_acceptance_rate,

        "valid_user_acceptance_rate":
            decision_metrics
            .valid_user_acceptance_rate,

        "false_reject_rate":
            decision_metrics
            .false_reject_rate,

        "gp_only_eligible_attack_sessions":
            int(
                np.sum(
                    labels == 1
                )
            ),

        "gp_only_eligible_benign_sessions":
            int(
                np.sum(
                    labels == 0
                )
            ),

        "gp_only_attack_detection_rate":
            decision_metrics
            .attack_detection_rate,

        "gp_only_false_accept_rate":
            decision_metrics
            .attack_acceptance_rate,

        "gp_only_false_reject_rate":
            decision_metrics
            .false_reject_rate,
    }

    return HeldOutEvaluation(
        predictions=predictions.reset_index(
            drop=True
        ),
        report=report,
        metrics_row=metrics_row,
        model_protocol_version=(
            bundle.protocol_version
        ),
        model_seed=bundle.seed,
    )


def evaluate_training_result_heldout(
    training_result: GPTrainingResult,
    *,
    n_calibration_bins: int = (
        DEFAULT_CALIBRATION_BINS
    ),
    minimum_calibration_bin_size: int = (
        DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE
    ),
) -> HeldOutEvaluation:
    """
    Evaluate the untouched test split from GPTrainingResult.
    """

    if not isinstance(
        training_result,
        GPTrainingResult,
    ):
        raise TypeError(
            "training_result must be GPTrainingResult."
        )

    return evaluate_heldout_table(
        source=training_result,
        test_table=(
            training_result.splits.test
        ),
        label_column=(
            training_result
            .config
            .label_column
        ),
        n_calibration_bins=(
            n_calibration_bins
        ),
        minimum_calibration_bin_size=(
            minimum_calibration_bin_size
        ),
    )


def normalize_independent_results(
    results: pd.DataFrame,
    *,
    p1_protocol_name: str = (
        DEFAULT_P1_PROTOCOL_NAME
    ),
) -> pd.DataFrame:
    """
    Validate and normalize independent experiment results.
    """

    if not isinstance(results, pd.DataFrame):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    if results.empty:
        raise InvalidEvaluationDatasetError(
            "Independent results cannot be empty."
        )

    required_columns = {
        DEFAULT_PROTOCOL_COLUMN,
        DEFAULT_ACTUAL_ATTACK_COLUMN,
        DEFAULT_ACCEPTED_COLUMN,
        DEFAULT_DETERMINISTIC_PASS_COLUMN,
        DEFAULT_PROBABILITY_COLUMN,
        DEFAULT_SEED_COLUMN,
    }

    missing_columns = sorted(
        required_columns.difference(
            results.columns
        )
    )

    if missing_columns:
        raise InvalidEvaluationDatasetError(
            "Independent results are missing columns: "
            f"{missing_columns!r}"
        )

    normalized = results.copy()

    normalized[
        DEFAULT_PROTOCOL_COLUMN
    ] = (
        normalized[
            DEFAULT_PROTOCOL_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    if (
        normalized[
            DEFAULT_PROTOCOL_COLUMN
        ]
        .eq("")
        .any()
    ):
        raise InvalidEvaluationDatasetError(
            "Protocol names cannot be empty."
        )

    normalized[
        DEFAULT_ACTUAL_ATTACK_COLUMN
    ] = normalize_binary_labels(
        normalized[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ],
        field_name=(
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ),
        require_both_classes=False,
    )

    normalized[
        DEFAULT_ACCEPTED_COLUMN
    ] = normalize_boolean_series(
        normalized[
            DEFAULT_ACCEPTED_COLUMN
        ].tolist(),
        DEFAULT_ACCEPTED_COLUMN,
    ).to_numpy()

    normalized[
        DEFAULT_DETERMINISTIC_PASS_COLUMN
    ] = normalize_boolean_series(
        normalized[
            DEFAULT_DETERMINISTIC_PASS_COLUMN
        ].tolist(),
        DEFAULT_DETERMINISTIC_PASS_COLUMN,
    ).to_numpy()

    normalized[
        DEFAULT_PROBABILITY_COLUMN
    ] = pd.to_numeric(
        normalized[
            DEFAULT_PROBABILITY_COLUMN
        ],
        errors="coerce",
    )

    finite_probability_mask = (
        normalized[
            DEFAULT_PROBABILITY_COLUMN
        ].notna()
    )

    finite_probabilities = normalized.loc[
        finite_probability_mask,
        DEFAULT_PROBABILITY_COLUMN,
    ].to_numpy(
        dtype=float
    )

    if (
        len(finite_probabilities)
        and (
            not np.all(
                np.isfinite(
                    finite_probabilities
                )
            )
            or np.any(
                finite_probabilities < 0.0
            )
            or np.any(
                finite_probabilities > 1.0
            )
        )
    ):
        raise InvalidEvaluationDatasetError(
            "Independent P(attack) values must be finite "
            "and between 0 and 1."
        )

    try:
        normalized[
            DEFAULT_SEED_COLUMN
        ] = pd.to_numeric(
            normalized[
                DEFAULT_SEED_COLUMN
            ],
            errors="raise",
        ).astype(int)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidEvaluationDatasetError(
            "Independent seed values must be integers."
        ) from error

    if DEFAULT_REASON_COLUMN not in normalized.columns:
        normalized[
            DEFAULT_REASON_COLUMN
        ] = ""

    normalized[
        DEFAULT_REASON_COLUMN
    ] = [
        normalize_reason_value(
            value
        )
        for value
        in normalized[
            DEFAULT_REASON_COLUMN
        ]
    ]

    if (
        DEFAULT_DETERMINISTIC_REASONS_COLUMN
        not in normalized.columns
    ):
        normalized[
            DEFAULT_DETERMINISTIC_REASONS_COLUMN
        ] = ""

    normalized[
        DEFAULT_DETERMINISTIC_REASONS_COLUMN
    ] = [
        normalize_reason_value(
            value
        )
        for value
        in normalized[
            DEFAULT_DETERMINISTIC_REASONS_COLUMN
        ]
    ]

    if (
        DEFAULT_TAG_RECOVERED_COLUMN
        not in normalized.columns
    ):
        normalized[
            DEFAULT_TAG_RECOVERED_COLUMN
        ] = np.nan

    else:
        tag_values: list[
            float
        ] = []

        for index, value in enumerate(
            normalized[
                DEFAULT_TAG_RECOVERED_COLUMN
            ]
        ):
            if value is None or (
                isinstance(value, float)
                and math.isnan(value)
            ):
                tag_values.append(
                    float("nan")
                )

            else:
                tag_values.append(
                    float(
                        normalize_boolean_value(
                            value,
                            (
                                f"{DEFAULT_TAG_RECOVERED_COLUMN}"
                                f"[{index}]"
                            ),
                        )
                    )
                )

        normalized[
            DEFAULT_TAG_RECOVERED_COLUMN
        ] = tag_values

    p1_rows = normalized[
        normalized[
            DEFAULT_PROTOCOL_COLUMN
        ]
        == p1_protocol_name
    ]

    if p1_rows.empty:
        raise InvalidEvaluationDatasetError(
            f"No rows were found for P1 protocol "
            f"{p1_protocol_name!r}."
        )

    p1_probability_rows = p1_rows.dropna(
        subset=[
            DEFAULT_PROBABILITY_COLUMN,
        ]
    )

    if p1_probability_rows.empty:
        raise InvalidEvaluationDatasetError(
            "P1 independent rows contain no GP probabilities."
        )

    normalize_binary_labels(
        p1_probability_rows[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ],
        field_name=(
            "P1 independent actual_attack"
        ),
        require_both_classes=True,
    )

    return normalized.reset_index(
        drop=True
    )


def extract_seed_set(
    table: pd.DataFrame,
    seed_column: str = DEFAULT_SEED_COLUMN,
) -> set[int]:
    """
    Extract integer seed values from a table.
    """

    if seed_column not in table.columns:
        return set()

    numeric = pd.to_numeric(
        table[
            seed_column
        ],
        errors="coerce",
    ).dropna()

    return {
        int(value)
        for value in numeric.tolist()
    }


def validate_evaluation_seed_disjointness(
    training_result: GPTrainingResult,
    independent_results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Confirm independent seeds do not overlap model-development seeds.
    """

    if not isinstance(
        training_result,
        GPTrainingResult,
    ):
        raise TypeError(
            "training_result must be GPTrainingResult."
        )

    if not isinstance(
        independent_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "independent_results must be a pandas DataFrame."
        )

    train_seeds = extract_seed_set(
        training_result.splits.train
    )

    calibration_seeds = extract_seed_set(
        training_result.splits.calibration
    )

    test_seeds = extract_seed_set(
        training_result.splits.test
    )

    independent_seeds = extract_seed_set(
        independent_results
    )

    development_sets = {
        "training":
            train_seeds,

        "calibration":
            calibration_seeds,

        "heldout_test":
            test_seeds,
    }

    overlaps: dict[
        str,
        list[int]
    ] = {}

    for split_name, split_seeds in development_sets.items():
        overlap = sorted(
            split_seeds.intersection(
                independent_seeds
            )
        )

        if overlap:
            overlaps[
                split_name
            ] = overlap

    if overlaps:
        raise EvaluationSeedOverlapError(
            "Independent evaluation seeds overlap "
            f"model-development seeds: {overlaps!r}"
        )

    return {
        "training_seeds":
            sorted(
                train_seeds
            ),

        "calibration_seeds":
            sorted(
                calibration_seeds
            ),

        "heldout_test_seeds":
            sorted(
                test_seeds
            ),

        "independent_evaluation_seeds":
            sorted(
                independent_seeds
            ),

        "seed_sets_disjoint":
            True,
    }


def protocol_security_metrics(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate end-to-end security and availability rates per protocol.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for protocol, group in results.groupby(
        DEFAULT_PROTOCOL_COLUMN,
        sort=True,
    ):
        attack_rows = group[
            group[
                DEFAULT_ACTUAL_ATTACK_COLUMN
            ]
            == 1
        ]

        benign_rows = group[
            group[
                DEFAULT_ACTUAL_ATTACK_COLUMN
            ]
            == 0
        ]

        if attack_rows.empty or benign_rows.empty:
            raise MissingEvaluationClassError(
                f"Protocol {protocol!r} must contain "
                "both benign and attack sessions."
            )

        attack_acceptance_rate = float(
            attack_rows[
                DEFAULT_ACCEPTED_COLUMN
            ].mean()
        )

        valid_user_acceptance_rate = float(
            benign_rows[
                DEFAULT_ACCEPTED_COLUMN
            ].mean()
        )

        row = {
            "protocol":
                protocol,

            "session_count":
                int(
                    len(group)
                ),

            "attack_session_count":
                int(
                    len(
                        attack_rows
                    )
                ),

            "benign_session_count":
                int(
                    len(
                        benign_rows
                    )
                ),

            "attack_acceptance_rate":
                attack_acceptance_rate,

            "attack_detection_rate":
                1.0
                - attack_acceptance_rate,

            "valid_user_acceptance_rate":
                valid_user_acceptance_rate,

            "false_reject_rate":
                1.0
                - valid_user_acceptance_rate,
        }

        if (
            group[
                DEFAULT_TAG_RECOVERED_COLUMN
            ]
            .notna()
            .any()
        ):
            row[
                "tag_recovery_success_rate"
            ] = float(
                group[
                    DEFAULT_TAG_RECOVERED_COLUMN
                ].mean()
            )

        else:
            row[
                "tag_recovery_success_rate"
            ] = float("nan")

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_gp_decision_path_table(
    p1_results: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Separate end-to-end outcomes from GP-only eligible outcomes.
    """

    threshold = validate_probability(
        threshold,
        "threshold",
    )

    p1_attack = p1_results[
        p1_results[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 1
    ]

    p1_benign = p1_results[
        p1_results[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 0
    ]

    eligible = p1_results[
        p1_results[
            DEFAULT_DETERMINISTIC_PASS_COLUMN
        ]
    ]

    eligible_attack = eligible[
        eligible[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 1
    ]

    eligible_benign = eligible[
        eligible[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 0
    ]

    gp_attack_detection_rate = (
        conditional_threshold_rate(
            eligible_attack,
            threshold,
            reject_when_above=True,
        )
    )

    gp_false_accept_rate = (
        conditional_threshold_rate(
            eligible_attack,
            threshold,
            reject_when_above=False,
        )
    )

    gp_false_reject_rate = (
        conditional_threshold_rate(
            eligible_benign,
            threshold,
            reject_when_above=True,
        )
    )

    gp_valid_user_acceptance_rate = (
        float(
            1.0
            - gp_false_reject_rate
        )
        if math.isfinite(
            gp_false_reject_rate
        )
        else float("nan")
    )

    return pd.DataFrame(
        [
            {
                "metric_scope":
                    "end_to_end_all_p1",

                "attack_session_count":
                    int(
                        len(
                            p1_attack
                        )
                    ),

                "benign_session_count":
                    int(
                        len(
                            p1_benign
                        )
                    ),

                "attack_detection_rate":
                    float(
                        np.mean(
                            ~p1_attack[
                                DEFAULT_ACCEPTED_COLUMN
                            ]
                        )
                    ),

                "attack_acceptance_rate":
                    float(
                        np.mean(
                            p1_attack[
                                DEFAULT_ACCEPTED_COLUMN
                            ]
                        )
                    ),

                "valid_user_acceptance_rate":
                    float(
                        np.mean(
                            p1_benign[
                                DEFAULT_ACCEPTED_COLUMN
                            ]
                        )
                    ),

                "false_reject_rate":
                    float(
                        np.mean(
                            ~p1_benign[
                                DEFAULT_ACCEPTED_COLUMN
                            ]
                        )
                    ),
            },
            {
                "metric_scope":
                    "gp_only_deterministic_pass",

                "attack_session_count":
                    int(
                        len(
                            eligible_attack
                        )
                    ),

                "benign_session_count":
                    int(
                        len(
                            eligible_benign
                        )
                    ),

                "attack_detection_rate":
                    gp_attack_detection_rate,

                "attack_acceptance_rate":
                    gp_false_accept_rate,

                "valid_user_acceptance_rate":
                    gp_valid_user_acceptance_rate,

                "false_reject_rate":
                    gp_false_reject_rate,
            },
        ]
    )


def build_seed_level_metrics(
    results: pd.DataFrame,
    *,
    p1_protocol_name: str = (
        DEFAULT_P1_PROTOCOL_NAME
    ),
) -> pd.DataFrame:
    """
    Build one paper-metric row per protocol and independent seed.
    """

    rows: list[
        dict[str, Any]
    ] = []

    grouped = results.groupby(
        [
            DEFAULT_PROTOCOL_COLUMN,
            DEFAULT_SEED_COLUMN,
        ],
        sort=True,
    )

    for (
        protocol,
        seed,
    ), group in grouped:
        attack_rows = group[
            group[
                DEFAULT_ACTUAL_ATTACK_COLUMN
            ]
            == 1
        ]

        benign_rows = group[
            group[
                DEFAULT_ACTUAL_ATTACK_COLUMN
            ]
            == 0
        ]

        attack_acceptance_rate = (
            float(
                attack_rows[
                    DEFAULT_ACCEPTED_COLUMN
                ].mean()
            )
            if not attack_rows.empty
            else float("nan")
        )

        valid_user_acceptance_rate = (
            float(
                benign_rows[
                    DEFAULT_ACCEPTED_COLUMN
                ].mean()
            )
            if not benign_rows.empty
            else float("nan")
        )

        row: dict[str, Any] = {
            "protocol":
                protocol,

            "seed":
                int(seed),

            "session_count":
                int(
                    len(group)
                ),

            "attack_session_count":
                int(
                    len(
                        attack_rows
                    )
                ),

            "benign_session_count":
                int(
                    len(
                        benign_rows
                    )
                ),

            "attack_acceptance_rate":
                attack_acceptance_rate,

            "attack_detection_rate": (
                1.0
                - attack_acceptance_rate
                if math.isfinite(
                    attack_acceptance_rate
                )
                else float("nan")
            ),

            "valid_user_acceptance_rate":
                valid_user_acceptance_rate,

            "false_reject_rate": (
                1.0
                - valid_user_acceptance_rate
                if math.isfinite(
                    valid_user_acceptance_rate
                )
                else float("nan")
            ),

            "benign_tag_recovery": (
                float(
                    benign_rows[
                        DEFAULT_TAG_RECOVERED_COLUMN
                    ].mean()
                )
                if (
                    not benign_rows.empty
                    and benign_rows[
                        DEFAULT_TAG_RECOVERED_COLUMN
                    ].notna().any()
                )
                else float("nan")
            ),

            "attack_tag_recovery": (
                float(
                    attack_rows[
                        DEFAULT_TAG_RECOVERED_COLUMN
                    ].mean()
                )
                if (
                    not attack_rows.empty
                    and attack_rows[
                        DEFAULT_TAG_RECOVERED_COLUMN
                    ].notna().any()
                )
                else float("nan")
            ),

            "roc_auc":
                float("nan"),

            "pr_auc":
                float("nan"),

            "brier_score":
                float("nan"),
        }

        if protocol == p1_protocol_name:
            gp_rows = group.dropna(
                subset=[
                    DEFAULT_PROBABILITY_COLUMN,
                ]
            )

            if (
                not gp_rows.empty
                and gp_rows[
                    DEFAULT_ACTUAL_ATTACK_COLUMN
                ].nunique()
                == 2
            ):
                seed_probability_metrics = (
                    probability_metrics(
                        gp_rows[
                            DEFAULT_ACTUAL_ATTACK_COLUMN
                        ],
                        gp_rows[
                            DEFAULT_PROBABILITY_COLUMN
                        ],
                    )
                )

                row.update(
                    seed_probability_metrics
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_confidence_interval_table(
    seed_metrics: pd.DataFrame,
    *,
    resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    ),
    random_state: int = (
        DEFAULT_BOOTSTRAP_RANDOM_STATE
    ),
) -> pd.DataFrame:
    """
    Calculate paper-reportable means and 95% bootstrap intervals.

    Confidence intervals use independent seed-level values rather than
    treating every repeated session as fully independent.
    """

    if not isinstance(
        seed_metrics,
        pd.DataFrame,
    ):
        raise TypeError(
            "seed_metrics must be a pandas DataFrame."
        )

    resamples = validate_positive_integer(
        resamples,
        "resamples",
    )

    if (
        isinstance(random_state, bool)
        or not isinstance(
            random_state,
            (
                int,
                np.integer,
            ),
        )
    ):
        raise TypeError(
            "random_state must be an integer."
        )

    metric_columns = (
        "valid_user_acceptance_rate",
        "false_reject_rate",
        "attack_detection_rate",
        "attack_acceptance_rate",
        "benign_tag_recovery",
        "attack_tag_recovery",
        "roc_auc",
        "pr_auc",
        "brier_score",
    )

    rows: list[
        dict[str, Any]
    ] = []

    for protocol, group in seed_metrics.groupby(
        DEFAULT_PROTOCOL_COLUMN,
        sort=True,
    ):
        for metric_index, metric_name in enumerate(
            metric_columns
        ):
            values = pd.to_numeric(
                group[
                    metric_name
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            finite_values = values[
                np.isfinite(
                    values
                )
            ]

            if len(finite_values) == 0:
                continue

            (
                mean_value,
                lower_bound,
                upper_bound,
            ) = bootstrap_mean_interval(
                finite_values,
                resamples=resamples,
                random_state=(
                    int(random_state)
                    + metric_index
                ),
            )

            rows.append(
                {
                    "protocol":
                        protocol,

                    "metric":
                        metric_name,

                    "independent_seed_count":
                        int(
                            len(
                                finite_values
                            )
                        ),

                    "mean":
                        mean_value,

                    "ci95_lower":
                        lower_bound,

                    "ci95_upper":
                        upper_bound,

                    "bootstrap_resamples":
                        resamples,
                }
            )

    return pd.DataFrame(
        rows
    )


def evaluate_independent_results(
    results: pd.DataFrame,
    source: (
        GPTrainingResult
        | GPModelBundle
        | Mapping[str, Any]
    ),
    *,
    p1_protocol_name: str = (
        DEFAULT_P1_PROTOCOL_NAME
    ),
    seed_provenance: (
        Mapping[str, Any]
        | None
    ) = None,
    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    ),
    bootstrap_random_state: int = (
        DEFAULT_BOOTSTRAP_RANDOM_STATE
    ),
) -> IndependentEvaluation:
    """
    Evaluate fixed GP policy on independent protocol experiments.
    """

    normalized = normalize_independent_results(
        results,
        p1_protocol_name=(
            p1_protocol_name
        ),
    )

    bundle = resolve_model_bundle(
        source
    )

    threshold = require_operational_threshold(
        bundle
    )

    p1_results = normalized[
        normalized[
            DEFAULT_PROTOCOL_COLUMN
        ]
        == p1_protocol_name
    ].dropna(
        subset=[
            DEFAULT_PROBABILITY_COLUMN,
        ]
    ).copy()

    labels = normalize_binary_labels(
        p1_results[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ],
        field_name=(
            "independent P1 actual_attack"
        ),
        require_both_classes=True,
    )

    probabilities = p1_results[
        DEFAULT_PROBABILITY_COLUMN
    ].to_numpy(
        dtype=float
    )

    gp_metrics = probability_metrics(
        labels,
        probabilities,
    )

    end_to_end_metrics = (
        authentication_decision_metrics(
            labels=labels,
            accepted=p1_results[
                DEFAULT_ACCEPTED_COLUMN
            ],
        )
    )

    eligible = p1_results[
        p1_results[
            DEFAULT_DETERMINISTIC_PASS_COLUMN
        ]
    ]

    eligible_attack = eligible[
        eligible[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 1
    ]

    eligible_benign = eligible[
        eligible[
            DEFAULT_ACTUAL_ATTACK_COLUMN
        ]
        == 0
    ]

    gp_only_attack_detection_rate = (
        conditional_threshold_rate(
            eligible_attack,
            threshold,
            reject_when_above=True,
        )
    )

    gp_only_false_accept_rate = (
        conditional_threshold_rate(
            eligible_attack,
            threshold,
            reject_when_above=False,
        )
    )

    gp_only_false_reject_rate = (
        conditional_threshold_rate(
            eligible_benign,
            threshold,
            reject_when_above=True,
        )
    )

    independent_report = evaluate_gp_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
        evaluation_scope=(
            "independent_paper_experiment"
        ),
        n_calibration_bins=(
            DEFAULT_CALIBRATION_BINS
        ),
        minimum_calibration_bin_size=(
            DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE
        ),
    )

    performance_row = {
        "evaluation_set":
            "independent_paper_experiment",

        "roc_auc":
            gp_metrics[
                "roc_auc"
            ],

        "pr_auc":
            gp_metrics[
                "pr_auc"
            ],

        "brier_score":
            gp_metrics[
                "brier_score"
            ],

        "expected_calibration_error":
            independent_report
            .expected_calibration_error,

        "selected_attack_threshold":
            threshold,

        "test_session_count":
            int(
                len(
                    p1_results
                )
            ),

        "attack_session_count":
            end_to_end_metrics
            .attack_session_count,

        "benign_session_count":
            end_to_end_metrics
            .benign_session_count,

        "attack_detection_rate":
            end_to_end_metrics
            .attack_detection_rate,

        "attack_acceptance_rate":
            end_to_end_metrics
            .attack_acceptance_rate,

        "valid_user_acceptance_rate":
            end_to_end_metrics
            .valid_user_acceptance_rate,

        "false_reject_rate":
            end_to_end_metrics
            .false_reject_rate,

        "gp_only_eligible_attack_sessions":
            int(
                len(
                    eligible_attack
                )
            ),

        "gp_only_eligible_benign_sessions":
            int(
                len(
                    eligible_benign
                )
            ),

        "gp_only_attack_detection_rate":
            gp_only_attack_detection_rate,

        "gp_only_false_accept_rate":
            gp_only_false_accept_rate,

        "gp_only_false_reject_rate":
            gp_only_false_reject_rate,
    }

    p1_predictions = p1_results.copy()

    p1_predictions[
        "predicted_attack_by_gp"
    ] = (
        p1_predictions[
            DEFAULT_PROBABILITY_COLUMN
        ]
        >= threshold
    )

    p1_predictions[
        "gp_policy_accept"
    ] = ~p1_predictions[
        "predicted_attack_by_gp"
    ]

    if "uncertainty" not in p1_predictions.columns:
        p1_predictions[
            "uncertainty"
        ] = uncertainty_values(
            p1_predictions[
                DEFAULT_PROBABILITY_COLUMN
            ]
        )

    decision_path_table = (
        build_gp_decision_path_table(
            p1_results,
            threshold,
        )
    )

    security_metrics_table = (
        protocol_security_metrics(
            normalized
        )
    )

    rejection_reason_table = (
        build_rejection_reason_distribution(
            normalized
        )
    )

    seed_metrics_table = (
        build_seed_level_metrics(
            normalized,
            p1_protocol_name=(
                p1_protocol_name
            ),
        )
    )

    confidence_interval_table = (
        build_confidence_interval_table(
            seed_metrics_table,
            resamples=(
                bootstrap_resamples
            ),
            random_state=(
                bootstrap_random_state
            ),
        )
    )

    provenance = (
        {}
        if seed_provenance is None
        else copy.deepcopy(
            dict(
                seed_provenance
            )
        )
    )

    if not provenance:
        provenance = {
            "independent_evaluation_seeds":
                sorted(
                    extract_seed_set(
                        normalized
                    )
                ),

            "seed_sets_disjoint":
                None,
        }

    return IndependentEvaluation(
        normalized_results=normalized,
        p1_predictions=(
            p1_predictions.reset_index(
                drop=True
            )
        ),
        performance_row=(
            performance_row
        ),
        decision_path_table=(
            decision_path_table
        ),
        security_metrics_table=(
            security_metrics_table
        ),
        rejection_reason_table=(
            rejection_reason_table
        ),
        seed_metrics_table=(
            seed_metrics_table
        ),
        confidence_interval_table=(
            confidence_interval_table
        ),
        seed_provenance=provenance,
    )


def evaluate_model(
    source: (
        GPTrainingResult
        | GPModelBundle
        | Mapping[str, Any]
    ),
    *,
    heldout_table: pd.DataFrame | None = None,
    heldout_label_column: str = (
        DEFAULT_LABEL_COLUMN
    ),
    independent_results: pd.DataFrame | None = None,
    p1_protocol_name: str = (
        DEFAULT_P1_PROTOCOL_NAME
    ),
    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    ),
    bootstrap_random_state: int = (
        DEFAULT_BOOTSTRAP_RANDOM_STATE
    ),
) -> ModelEvaluationResult:
    """
    Execute held-out and optional independent evaluation.
    """

    if heldout_table is None:
        if not isinstance(
            source,
            GPTrainingResult,
        ):
            raise InvalidEvaluationDatasetError(
                "heldout_table is required when source is not "
                "GPTrainingResult."
            )

        heldout_table = (
            source.splits.test
        )

        heldout_label_column = (
            source.config.label_column
        )

    heldout = evaluate_heldout_table(
        source=source,
        test_table=heldout_table,
        label_column=(
            heldout_label_column
        ),
    )

    independent: (
        IndependentEvaluation | None
    ) = None

    if independent_results is not None:
        seed_provenance: dict[
            str,
            Any
        ]

        if isinstance(
            source,
            GPTrainingResult,
        ):
            normalized_for_seed_check = (
                normalize_independent_results(
                    independent_results,
                    p1_protocol_name=(
                        p1_protocol_name
                    ),
                )
            )

            seed_provenance = (
                validate_evaluation_seed_disjointness(
                    source,
                    normalized_for_seed_check,
                )
            )

        else:
            seed_provenance = {
                "independent_evaluation_seeds":
                    sorted(
                        extract_seed_set(
                            independent_results
                        )
                    ),

                "seed_sets_disjoint":
                    "not_verified_without_training_split_tables",
            }

        independent = evaluate_independent_results(
            results=independent_results,
            source=source,
            p1_protocol_name=(
                p1_protocol_name
            ),
            seed_provenance=(
                seed_provenance
            ),
            bootstrap_resamples=(
                bootstrap_resamples
            ),
            bootstrap_random_state=(
                bootstrap_random_state
            ),
        )

    return ModelEvaluationResult(
        heldout=heldout,
        independent=independent,
    )


def write_json_artifact(
    destination: str | Path,
    value: Any,
) -> Path:
    """
    Write one JSON-safe evaluation artifact.
    """

    path = Path(
        destination
    )

    if path.suffix.lower() != ".json":
        raise EvaluationArtifactError(
            "JSON artifact path must end with .json."
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                json_safe(
                    value
                ),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise EvaluationArtifactError(
            f"Could not write evaluation JSON: {path}"
        ) from error

    return path


def update_model_metadata_with_evaluation(
    metadata_path: str | Path,
    evaluation: ModelEvaluationResult,
) -> Path:
    """
    Merge held-out evaluation evidence into model_metadata.json.
    """

    if not isinstance(
        evaluation,
        ModelEvaluationResult,
    ):
        raise TypeError(
            "evaluation must be ModelEvaluationResult."
        )

    path = Path(
        metadata_path
    )

    existing_metadata: dict[
        str,
        Any
    ] = {}

    if path.exists():
        if not path.is_file():
            raise EvaluationArtifactError(
                f"Metadata path is not a file: {path}"
            )

        try:
            loaded = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as error:
            raise EvaluationArtifactError(
                f"Could not read model metadata: {path}"
            ) from error

        if not isinstance(
            loaded,
            Mapping,
        ):
            raise EvaluationArtifactError(
                "Model metadata must contain a JSON object."
            )

        existing_metadata = dict(
            loaded
        )

    existing_metadata[
        "heldout_test_evaluated"
    ] = True

    existing_metadata[
        "model_evaluation"
    ] = evaluation.metadata()

    return write_json_artifact(
        path,
        existing_metadata,
    )


def export_model_evaluation(
    evaluation: ModelEvaluationResult,
    *,
    output_directory: str | Path = (
        DEFAULT_RESULTS_DIRECTORY
    ),
    model_metadata_path: str | Path | None = None,
) -> dict[str, Path]:
    """
    Export notebook-compatible GP evaluation artifacts.
    """

    if not isinstance(
        evaluation,
        ModelEvaluationResult,
    ):
        raise TypeError(
            "evaluation must be ModelEvaluationResult."
        )

    output_directory = Path(
        output_directory
    )

    try:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise EvaluationArtifactError(
            "Could not create evaluation output directory."
        ) from error

    paths: dict[
        str,
        Path
    ] = {
        "gp_performance_metrics":
            output_directory
            / "gp_performance_metrics.csv",

        "heldout_predictions":
            output_directory
            / "gp_heldout_predictions.csv",

        "heldout_calibration_curve":
            output_directory
            / "gp_calibration_curve.csv",

        "evaluation_metadata":
            output_directory
            / "model_evaluation_metadata.json",
    }

    try:
        evaluation.performance_table.to_csv(
            paths[
                "gp_performance_metrics"
            ],
            index=False,
        )

        evaluation.heldout.predictions.to_csv(
            paths[
                "heldout_predictions"
            ],
            index=False,
        )

        evaluation.heldout.report.calibration_curve.to_csv(
            paths[
                "heldout_calibration_curve"
            ],
            index=False,
        )

        write_json_artifact(
            paths[
                "evaluation_metadata"
            ],
            evaluation.metadata(),
        )

        if evaluation.independent is not None:
            independent_paths = {
                "independent_predictions":
                    output_directory
                    / "independent_p1_predictions.csv",

                "security_metrics":
                    output_directory
                    / "security_metrics.csv",

                "gp_decision_path":
                    output_directory
                    / "gp_decision_path_metrics.csv",

                "rejection_reason_distribution":
                    output_directory
                    / "rejection_reason_distribution.csv",

                "seed_metrics":
                    output_directory
                    / "paper_seed_metrics.csv",

                "confidence_intervals":
                    output_directory
                    / "paper_confidence_intervals.csv",

                "experiment_results":
                    output_directory
                    / "experiment_results.csv",
            }

            evaluation.independent.p1_predictions.to_csv(
                independent_paths[
                    "independent_predictions"
                ],
                index=False,
            )

            evaluation.independent.security_metrics_table.to_csv(
                independent_paths[
                    "security_metrics"
                ],
                index=False,
            )

            evaluation.independent.decision_path_table.to_csv(
                independent_paths[
                    "gp_decision_path"
                ],
                index=False,
            )

            evaluation.independent.rejection_reason_table.to_csv(
                independent_paths[
                    "rejection_reason_distribution"
                ],
                index=False,
            )

            evaluation.independent.seed_metrics_table.to_csv(
                independent_paths[
                    "seed_metrics"
                ],
                index=False,
            )

            evaluation.independent.confidence_interval_table.to_csv(
                independent_paths[
                    "confidence_intervals"
                ],
                index=False,
            )

            evaluation.independent.normalized_results.to_csv(
                independent_paths[
                    "experiment_results"
                ],
                index=False,
            )

            paths.update(
                independent_paths
            )

        if model_metadata_path is not None:
            paths[
                "updated_model_metadata"
            ] = update_model_metadata_with_evaluation(
                model_metadata_path,
                evaluation,
            )

    except ModelEvaluatorError:
        raise

    except Exception as error:
        raise EvaluationArtifactError(
            "Could not export one or more evaluation artifacts."
        ) from error

    return {
        name: path.resolve()
        for name, path
        in paths.items()
    }


class FTQuPAPModelEvaluator:
    """
    Reusable FT-QuPAP model-evaluation service.
    """

    def __init__(
        self,
        source: (
            GPTrainingResult
            | GPModelBundle
            | Mapping[str, Any]
        ),
        *,
        p1_protocol_name: str = (
            DEFAULT_P1_PROTOCOL_NAME
        ),
        bootstrap_resamples: int = (
            DEFAULT_BOOTSTRAP_RESAMPLES
        ),
        bootstrap_random_state: int = (
            DEFAULT_BOOTSTRAP_RANDOM_STATE
        ),
    ) -> None:
        self.source = source

        if (
            not isinstance(
                p1_protocol_name,
                str,
            )
            or not p1_protocol_name.strip()
        ):
            raise ValueError(
                "p1_protocol_name cannot be empty."
            )

        self.p1_protocol_name = (
            p1_protocol_name.strip()
        )

        self.bootstrap_resamples = (
            validate_positive_integer(
                bootstrap_resamples,
                "bootstrap_resamples",
            )
        )

        if (
            isinstance(
                bootstrap_random_state,
                bool,
            )
            or not isinstance(
                bootstrap_random_state,
                (
                    int,
                    np.integer,
                ),
            )
        ):
            raise TypeError(
                "bootstrap_random_state must be an integer."
            )

        self.bootstrap_random_state = int(
            bootstrap_random_state
        )

        self.result: (
            ModelEvaluationResult | None
        ) = None

    @property
    def evaluated(self) -> bool:
        """Return whether evaluation has completed."""

        return self.result is not None

    def evaluate_heldout(
        self,
        test_table: pd.DataFrame | None = None,
        *,
        label_column: str = (
            DEFAULT_LABEL_COLUMN
        ),
    ) -> HeldOutEvaluation:
        """
        Evaluate the held-out session-level test split.
        """

        if test_table is None:
            if not isinstance(
                self.source,
                GPTrainingResult,
            ):
                raise InvalidEvaluationDatasetError(
                    "test_table is required when source is not "
                    "GPTrainingResult."
                )

            return evaluate_training_result_heldout(
                self.source
            )

        return evaluate_heldout_table(
            source=self.source,
            test_table=test_table,
            label_column=label_column,
        )

    def evaluate(
        self,
        *,
        heldout_table: pd.DataFrame | None = None,
        heldout_label_column: str = (
            DEFAULT_LABEL_COLUMN
        ),
        independent_results: pd.DataFrame | None = None,
    ) -> ModelEvaluationResult:
        """
        Execute held-out and optional independent evaluation.
        """

        self.result = evaluate_model(
            source=self.source,
            heldout_table=heldout_table,
            heldout_label_column=(
                heldout_label_column
            ),
            independent_results=(
                independent_results
            ),
            p1_protocol_name=(
                self.p1_protocol_name
            ),
            bootstrap_resamples=(
                self.bootstrap_resamples
            ),
            bootstrap_random_state=(
                self.bootstrap_random_state
            ),
        )

        return self.result

    def export(
        self,
        *,
        output_directory: str | Path = (
            DEFAULT_RESULTS_DIRECTORY
        ),
        model_metadata_path: str | Path | None = None,
    ) -> dict[str, Path]:
        """
        Export artifacts from the latest evaluation.
        """

        if self.result is None:
            raise ModelEvaluatorError(
                "No completed model evaluation is available."
            )

        return export_model_evaluation(
            self.result,
            output_directory=(
                output_directory
            ),
            model_metadata_path=(
                model_metadata_path
            ),
        )


def build_self_test_independent_results(
    training_result: GPTrainingResult,
    heldout: HeldOutEvaluation,
) -> pd.DataFrame:
    """
    Build a small independent multi-seed result table for self-testing.
    """

    threshold = heldout.threshold

    rows: list[
        dict[str, Any]
    ] = []

    independent_seeds = (
        8001,
        8002,
        8003,
    )

    heldout_rows = (
        heldout.predictions.reset_index(
            drop=True
        )
    )

    for seed_index, seed in enumerate(
        independent_seeds
    ):
        for row_index, source_row in heldout_rows.iterrows():
            actual_attack = int(
                source_row[
                    DEFAULT_ACTUAL_ATTACK_COLUMN
                ]
            )

            base_probability = float(
                source_row[
                    DEFAULT_PROBABILITY_COLUMN
                ]
            )

            adjusted_probability = float(
                np.clip(
                    base_probability
                    + (
                        seed_index
                        - 1
                    )
                    * 0.005,
                    0.0,
                    1.0,
                )
            )

            deterministic_pass = not (
                actual_attack == 1
                and row_index % 9 == 0
            )

            gp_accept = bool(
                adjusted_probability
                < threshold
            )

            accepted = bool(
                deterministic_pass
                and gp_accept
            )

            if not deterministic_pass:
                reason = (
                    "deterministic_protocol_check_failed"
                )

                deterministic_reasons = (
                    "authentication_tag_mismatch"
                )

            elif accepted:
                reason = (
                    "accepted_by_calibrated_bayesian_policy"
                )

                deterministic_reasons = ""

            else:
                reason = (
                    "rejected_by_calibrated_bayesian_policy"
                )

                deterministic_reasons = ""

            rows.append(
                {
                    "protocol":
                        DEFAULT_P1_PROTOCOL_NAME,

                    "seed":
                        seed,

                    "actual_attack":
                        actual_attack,

                    "accepted":
                        accepted,

                    "reason":
                        reason,

                    "deterministic_pass":
                        deterministic_pass,

                    "deterministic_reasons":
                        deterministic_reasons,

                    "p_attack":
                        adjusted_probability,

                    "uncertainty":
                        gp_predictive_uncertainty(
                            adjusted_probability
                        ),

                    "tag_recovered":
                        deterministic_pass,

                    "qber_raw":
                        source_row[
                            "qber_raw"
                        ],
                }
            )

            baseline_accepted = bool(
                actual_attack == 0
                or (
                    actual_attack == 1
                    and row_index % 5 == 0
                )
            )

            rows.append(
                {
                    "protocol":
                        "B1_QuPAP_Like",

                    "seed":
                        seed,

                    "actual_attack":
                        actual_attack,

                    "accepted":
                        baseline_accepted,

                    "reason": (
                        "accepted"
                        if baseline_accepted
                        else "fixed_qber_threshold_exceeded"
                    ),

                    "deterministic_pass":
                        True,

                    "deterministic_reasons":
                        "",

                    "p_attack":
                        np.nan,

                    "uncertainty":
                        np.nan,

                    "tag_recovered":
                        baseline_accepted,

                    "qber_raw":
                        source_row[
                            "qber_raw"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


def run_self_test() -> None:
    """
    Verify held-out evaluation, independent metrics, seed checks,
    decision-path separation, confidence intervals, and export.
    """

    import tempfile

    from .model_trainer import (
        FTQuPAPModelTrainer,
        GPTrainingConfig,
        build_self_test_dataset,
    )

    dataset = build_self_test_dataset(
        random_state=7
    )

    trainer = FTQuPAPModelTrainer(
        GPTrainingConfig(
            random_state=7,
            max_exact_gp_train_rows=30,
            protocol_version=(
                "FT-QuPAP-evaluator-self-test"
            ),
            training_source=(
                "model_evaluator_self_test"
            ),
            session_gp_data_mode=(
                "self_test"
            ),
        )
    )

    training_result = trainer.fit(
        dataset
    )

    heldout = (
        evaluate_training_result_heldout(
            training_result,
            n_calibration_bins=4,
            minimum_calibration_bin_size=2,
        )
    )

    if heldout.sample_count != 16:
        raise ModelEvaluatorError(
            "Held-out sample count is incorrect."
        )

    if not (
        0.0
        <= heldout.report.roc_auc
        <= 1.0
    ):
        raise ModelEvaluatorError(
            "Held-out ROC-AUC is invalid."
        )

    if not (
        0.0
        <= heldout.report.pr_auc
        <= 1.0
    ):
        raise ModelEvaluatorError(
            "Held-out PR-AUC is invalid."
        )

    if not (
        0.0
        <= heldout.report.brier_score
        <= 1.0
    ):
        raise ModelEvaluatorError(
            "Held-out Brier score is invalid."
        )

    independent_results = (
        build_self_test_independent_results(
            training_result,
            heldout,
        )
    )

    evaluator = FTQuPAPModelEvaluator(
        training_result,
        bootstrap_resamples=1000,
        bootstrap_random_state=7,
    )

    evaluation = evaluator.evaluate(
        independent_results=(
            independent_results
        )
    )

    if not evaluator.evaluated:
        raise ModelEvaluatorError(
            "Evaluator was not marked as completed."
        )

    if evaluation.independent is None:
        raise ModelEvaluatorError(
            "Independent evaluation is missing."
        )

    if (
        evaluation.independent
        .seed_provenance
        .get(
            "seed_sets_disjoint"
        )
        is not True
    ):
        raise ModelEvaluatorError(
            "Independent seed separation was not verified."
        )

    if len(
        evaluation.performance_table
    ) != 2:
        raise ModelEvaluatorError(
            "Performance table must contain held-out "
            "and independent rows."
        )

    expected_scopes = {
        "end_to_end_all_p1",
        "gp_only_deterministic_pass",
    }

    actual_scopes = set(
        evaluation.independent
        .decision_path_table[
            "metric_scope"
        ]
    )

    if actual_scopes != expected_scopes:
        raise ModelEvaluatorError(
            "GP decision-path scopes are incorrect."
        )

    if (
        evaluation.independent
        .confidence_interval_table
        .empty
    ):
        raise ModelEvaluatorError(
            "Seed-level confidence intervals were not generated."
        )

    with tempfile.TemporaryDirectory() as directory:
        output_directory = (
            Path(directory)
            / "results"
        )

        metadata_path = (
            Path(directory)
            / "models"
            / "model_metadata.json"
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.write_text(
            json.dumps(
                training_result.metadata(),
                indent=2,
            ),
            encoding="utf-8",
        )

        artifact_paths = (
            evaluator.export(
                output_directory=(
                    output_directory
                ),
                model_metadata_path=(
                    metadata_path
                ),
            )
        )

        for artifact_path in artifact_paths.values():
            if not artifact_path.exists():
                raise ModelEvaluatorError(
                    "Evaluation artifact is missing: "
                    f"{artifact_path}"
                )

        updated_metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            updated_metadata.get(
                "heldout_test_evaluated"
            )
            is not True
        ):
            raise ModelEvaluatorError(
                "Model metadata was not updated."
            )

    print(
        "Model evaluator self-test completed successfully."
    )

    print(
        "Held-out rows:",
        heldout.sample_count,
    )

    print(
        "Held-out ROC-AUC:",
        f"{heldout.report.roc_auc:.6f}",
    )

    print(
        "Held-out PR-AUC:",
        f"{heldout.report.pr_auc:.6f}",
    )

    print(
        "Held-out Brier score:",
        f"{heldout.report.brier_score:.6f}",
    )

    print(
        "Independent P1 rows:",
        evaluation.independent.session_count,
    )

    print(
        "Independent seed sets disjoint:",
        evaluation.independent
        .seed_provenance[
            "seed_sets_disjoint"
        ],
    )

    print(
        "Performance table rows:",
        len(
            evaluation.performance_table
        ),
    )


__all__ = [
    "DEFAULT_P1_PROTOCOL_NAME",
    "DEFAULT_PROTOCOL_COLUMN",
    "DEFAULT_ACTUAL_ATTACK_COLUMN",
    "DEFAULT_ACCEPTED_COLUMN",
    "DEFAULT_DETERMINISTIC_PASS_COLUMN",
    "DEFAULT_DETERMINISTIC_REASONS_COLUMN",
    "DEFAULT_REASON_COLUMN",
    "DEFAULT_PROBABILITY_COLUMN",
    "DEFAULT_SEED_COLUMN",
    "DEFAULT_TAG_RECOVERED_COLUMN",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_RANDOM_STATE",
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE",
    "PROJECT_ROOT",
    "DEFAULT_RESULTS_DIRECTORY",
    "DEFAULT_GP_PERFORMANCE_PATH",
    "DEFAULT_HELDOUT_PREDICTIONS_PATH",
    "DEFAULT_INDEPENDENT_PREDICTIONS_PATH",
    "DEFAULT_CALIBRATION_CURVE_PATH",
    "DEFAULT_SECURITY_METRICS_PATH",
    "DEFAULT_GP_DECISION_PATH",
    "DEFAULT_REJECTION_REASON_PATH",
    "DEFAULT_SEED_METRICS_PATH",
    "DEFAULT_CONFIDENCE_INTERVAL_PATH",
    "DEFAULT_EVALUATION_METADATA_PATH",
    "ModelEvaluatorError",
    "InvalidEvaluationDatasetError",
    "EvaluationThresholdUnavailableError",
    "EvaluationPredictionError",
    "EvaluationSeedOverlapError",
    "EvaluationArtifactError",
    "MissingEvaluationClassError",
    "HeldOutEvaluation",
    "IndependentEvaluation",
    "ModelEvaluationResult",
    "FTQuPAPModelEvaluator",
    "validate_probability",
    "validate_positive_integer",
    "json_safe",
    "normalize_binary_labels",
    "normalize_boolean_value",
    "normalize_boolean_series",
    "normalize_reason_value",
    "resolve_model_bundle",
    "require_operational_threshold",
    "prepare_evaluation_features",
    "predict_raw_probabilities",
    "predict_calibrated_probabilities",
    "uncertainty_values",
    "conditional_threshold_rate",
    "evaluate_heldout_table",
    "evaluate_training_result_heldout",
    "normalize_independent_results",
    "extract_seed_set",
    "validate_evaluation_seed_disjointness",
    "protocol_security_metrics",
    "build_gp_decision_path_table",
    "build_seed_level_metrics",
    "build_confidence_interval_table",
    "evaluate_independent_results",
    "evaluate_model",
    "write_json_artifact",
    "update_model_metadata_with_evaluation",
    "export_model_evaluation",
    "build_self_test_independent_results",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ModelEvaluatorError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "\n[MODEL EVALUATOR ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error