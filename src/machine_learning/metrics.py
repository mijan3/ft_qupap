"""
FT-QuPAP Machine-Learning Metrics

This module calculates the probability-quality, decision-policy,
calibration, and confidence-interval metrics used by the FT-QuPAP
Gaussian Process evaluation.

Probability-quality metrics:

    - ROC-AUC
    - PR-AUC
    - Brier score

Decision-policy metrics:

    - attack detection rate
    - attack acceptance / false-accept rate
    - valid-user acceptance rate
    - false-reject rate
    - overall decision accuracy

Calibration metrics:

    - expected calibration error
    - adaptive calibration table

Statistical reporting:

    - mean across independent seeds
    - percentile-bootstrap 95% confidence interval

Class definitions:

    label 0 = benign session
    label 1 = attack session

Decision definitions:

    accepted = True:
        Authentication was granted.

    accepted = False:
        Authentication was rejected.

Threshold rule:

    predicted attack = P(attack) >= threshold

Therefore, a probability exactly equal to the threshold is classified
as attack and the session is rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from .probability_calibrator import calibration_diagnostics


DEFAULT_BOOTSTRAP_RESAMPLES = 5000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_STATE = 20260701
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE = 30


class MetricsError(Exception):
    """Base exception for FT-QuPAP metric failures."""


class InvalidMetricDataError(MetricsError):
    """Raised when labels, probabilities, or decisions are malformed."""


class MissingMetricClassError(MetricsError):
    """Raised when both benign and attack classes are required."""


class InvalidBootstrapConfigurationError(MetricsError):
    """Raised when confidence-interval settings are invalid."""


class MetricConsistencyError(MetricsError):
    """Raised when calculated metric values contradict each other."""


def normalize_binary_labels(
    labels: Any,
    *,
    require_both_classes: bool = False,
) -> np.ndarray:
    """
    Validate binary FT-QuPAP session labels.

    Returns:
        One-dimensional NumPy array containing integer values 0 or 1.
    """

    if isinstance(
        labels,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "labels must be a numeric sequence."
        )

    try:
        numeric_labels = np.asarray(
            labels,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidMetricDataError(
            "Labels could not be converted to numeric values."
        ) from error

    if numeric_labels.size == 0:
        raise InvalidMetricDataError(
            "Labels cannot be empty."
        )

    if not np.all(
        np.isfinite(
            numeric_labels
        )
    ):
        raise InvalidMetricDataError(
            "Labels cannot contain NaN or infinity."
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
        raise InvalidMetricDataError(
            "Labels must contain only 0 and 1."
        )

    normalized = numeric_labels.astype(
        int,
        copy=True,
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
        raise MissingMetricClassError(
            "Both benign class 0 and attack class 1 "
            "are required for this metric."
        )

    return normalized


def normalize_probabilities(
    probabilities: Any,
) -> np.ndarray:
    """
    Validate attack probabilities in the interval [0, 1].
    """

    if isinstance(
        probabilities,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "probabilities must be a numeric sequence."
        )

    try:
        normalized = np.asarray(
            probabilities,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidMetricDataError(
            "Probabilities could not be converted to numeric values."
        ) from error

    if normalized.size == 0:
        raise InvalidMetricDataError(
            "Probabilities cannot be empty."
        )

    if not np.all(
        np.isfinite(
            normalized
        )
    ):
        raise InvalidMetricDataError(
            "Probabilities cannot contain NaN or infinity."
        )

    if (
        np.any(
            normalized < 0.0
        )
        or np.any(
            normalized > 1.0
        )
    ):
        raise InvalidMetricDataError(
            "Probabilities must be between 0 and 1."
        )

    return normalized.astype(
        float,
        copy=True,
    )


def normalize_acceptance_decisions(
    accepted: Any,
) -> np.ndarray:
    """
    Validate session acceptance decisions.

    Accepted values may be booleans or numeric 0/1 values.
    """

    if isinstance(
        accepted,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "accepted must be a sequence."
        )

    try:
        raw_values = np.asarray(
            accepted
        ).reshape(-1)

    except Exception as error:
        raise InvalidMetricDataError(
            "Acceptance decisions could not be converted."
        ) from error

    if raw_values.size == 0:
        raise InvalidMetricDataError(
            "Acceptance decisions cannot be empty."
        )

    normalized: list[bool] = []

    for index, value in enumerate(
        raw_values.tolist()
    ):
        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ):
            normalized.append(
                bool(value)
            )

        elif isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):
            numeric_value = float(
                value
            )

            if (
                not math.isfinite(
                    numeric_value
                )
                or numeric_value
                not in {
                    0.0,
                    1.0,
                }
            ):
                raise InvalidMetricDataError(
                    f"accepted[{index}] must be boolean or 0/1."
                )

            normalized.append(
                bool(
                    int(
                        numeric_value
                    )
                )
            )

        else:
            raise InvalidMetricDataError(
                f"accepted[{index}] must be boolean or 0/1."
            )

    return np.asarray(
        normalized,
        dtype=bool,
    )


def validate_equal_lengths(
    **arrays: np.ndarray,
) -> int:
    """
    Require every supplied metric array to have equal length.
    """

    if not arrays:
        raise InvalidMetricDataError(
            "At least one metric array is required."
        )

    lengths = {
        name: len(value)
        for name, value in arrays.items()
    }

    unique_lengths = set(
        lengths.values()
    )

    if len(unique_lengths) != 1:
        raise InvalidMetricDataError(
            "Metric arrays must have equal length: "
            f"{lengths!r}"
        )

    length = next(
        iter(
            unique_lengths
        )
    )

    if length == 0:
        raise InvalidMetricDataError(
            "Metric arrays cannot be empty."
        )

    return int(
        length
    )


def validate_threshold(
    threshold: Any,
) -> float:
    """
    Validate an operational decision threshold.
    """

    if (
        isinstance(
            threshold,
            bool,
        )
        or not isinstance(
            threshold,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ):
        raise TypeError(
            "threshold must be numeric."
        )

    normalized = float(
        threshold
    )

    if not math.isfinite(
        normalized
    ):
        raise InvalidMetricDataError(
            "threshold must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise InvalidMetricDataError(
            "threshold must be between 0 and 1."
        )

    return normalized


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """
    Calculate notebook-compatible GP probability metrics.

    Both classes are required because ROC-AUC and PR-AUC are undefined
    for a single-class evaluation set.
    """

    normalized_labels = normalize_binary_labels(
        labels,
        require_both_classes=True,
    )

    normalized_probabilities = normalize_probabilities(
        probabilities
    )

    validate_equal_lengths(
        labels=normalized_labels,
        probabilities=normalized_probabilities,
    )

    return {
        "roc_auc":
            float(
                roc_auc_score(
                    normalized_labels,
                    normalized_probabilities,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    normalized_labels,
                    normalized_probabilities,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    normalized_labels,
                    normalized_probabilities,
                )
            ),
    }


def threshold_attack_predictions(
    probabilities: Sequence[float],
    threshold: float,
) -> np.ndarray:
    """
    Convert attack probabilities into attack predictions.

    Equality with the threshold produces an attack prediction.
    """

    normalized_probabilities = normalize_probabilities(
        probabilities
    )

    normalized_threshold = validate_threshold(
        threshold
    )

    return (
        normalized_probabilities
        >= normalized_threshold
    )


@dataclass(frozen=True)
class DecisionMetrics:
    """
    Authentication-policy metrics for labeled sessions.
    """

    session_count: int
    attack_session_count: int
    benign_session_count: int

    true_attack_rejections: int
    false_attack_acceptances: int
    true_benign_acceptances: int
    false_benign_rejections: int

    attack_detection_rate: float
    attack_acceptance_rate: float

    valid_user_acceptance_rate: float
    false_reject_rate: float

    overall_acceptance_rate: float
    overall_rejection_rate: float
    decision_accuracy: float

    def __post_init__(self) -> None:
        count_fields = (
            "session_count",
            "attack_session_count",
            "benign_session_count",
            "true_attack_rejections",
            "false_attack_acceptances",
            "true_benign_acceptances",
            "false_benign_rejections",
        )

        for field_name in count_fields:
            value = getattr(
                self,
                field_name,
            )

            if (
                isinstance(
                    value,
                    bool,
                )
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

            if int(value) < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        probability_fields = (
            "attack_detection_rate",
            "attack_acceptance_rate",
            "valid_user_acceptance_rate",
            "false_reject_rate",
            "overall_acceptance_rate",
            "overall_rejection_rate",
            "decision_accuracy",
        )

        for field_name in probability_fields:
            value = float(
                getattr(
                    self,
                    field_name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{field_name} must be finite and "
                    "between 0 and 1."
                )

        if (
            self.attack_session_count
            + self.benign_session_count
            != self.session_count
        ):
            raise MetricConsistencyError(
                "Attack and benign counts do not equal "
                "the total session count."
            )

        if (
            self.true_attack_rejections
            + self.false_attack_acceptances
            != self.attack_session_count
        ):
            raise MetricConsistencyError(
                "Attack outcome counts are inconsistent."
            )

        if (
            self.true_benign_acceptances
            + self.false_benign_rejections
            != self.benign_session_count
        ):
            raise MetricConsistencyError(
                "Benign outcome counts are inconsistent."
            )

        if not math.isclose(
            self.attack_detection_rate
            + self.attack_acceptance_rate,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MetricConsistencyError(
                "Attack detection and attack acceptance "
                "rates must sum to one."
            )

        if not math.isclose(
            self.valid_user_acceptance_rate
            + self.false_reject_rate,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MetricConsistencyError(
                "Valid-user acceptance and false-reject "
                "rates must sum to one."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy-metric record."""

        return {
            "session_count":
                self.session_count,

            "attack_session_count":
                self.attack_session_count,

            "benign_session_count":
                self.benign_session_count,

            "true_attack_rejections":
                self.true_attack_rejections,

            "false_attack_acceptances":
                self.false_attack_acceptances,

            "true_benign_acceptances":
                self.true_benign_acceptances,

            "false_benign_rejections":
                self.false_benign_rejections,

            "attack_detection_rate":
                self.attack_detection_rate,

            "attack_acceptance_rate":
                self.attack_acceptance_rate,

            "false_accept_rate":
                self.attack_acceptance_rate,

            "valid_user_acceptance_rate":
                self.valid_user_acceptance_rate,

            "false_reject_rate":
                self.false_reject_rate,

            "overall_acceptance_rate":
                self.overall_acceptance_rate,

            "overall_rejection_rate":
                self.overall_rejection_rate,

            "decision_accuracy":
                self.decision_accuracy,
        }


def authentication_decision_metrics(
    labels: Sequence[int],
    accepted: Sequence[bool],
) -> DecisionMetrics:
    """
    Calculate end-to-end authentication-policy metrics.

    Args:
        labels:
            Actual session classes, where 0 is benign and 1 is attack.

        accepted:
            Final authentication acceptance decisions.
    """

    normalized_labels = normalize_binary_labels(
        labels,
        require_both_classes=True,
    )

    normalized_accepted = (
        normalize_acceptance_decisions(
            accepted
        )
    )

    session_count = validate_equal_lengths(
        labels=normalized_labels,
        accepted=normalized_accepted,
    )

    attack_mask = (
        normalized_labels == 1
    )

    benign_mask = (
        normalized_labels == 0
    )

    rejected = ~normalized_accepted

    true_attack_rejections = int(
        np.sum(
            rejected
            & attack_mask
        )
    )

    false_attack_acceptances = int(
        np.sum(
            normalized_accepted
            & attack_mask
        )
    )

    true_benign_acceptances = int(
        np.sum(
            normalized_accepted
            & benign_mask
        )
    )

    false_benign_rejections = int(
        np.sum(
            rejected
            & benign_mask
        )
    )

    attack_session_count = int(
        np.sum(
            attack_mask
        )
    )

    benign_session_count = int(
        np.sum(
            benign_mask
        )
    )

    attack_detection_rate = float(
        true_attack_rejections
        / attack_session_count
    )

    attack_acceptance_rate = float(
        false_attack_acceptances
        / attack_session_count
    )

    valid_user_acceptance_rate = float(
        true_benign_acceptances
        / benign_session_count
    )

    false_reject_rate = float(
        false_benign_rejections
        / benign_session_count
    )

    overall_acceptance_rate = float(
        np.mean(
            normalized_accepted
        )
    )

    overall_rejection_rate = float(
        np.mean(
            rejected
        )
    )

    decision_accuracy = float(
        (
            true_attack_rejections
            + true_benign_acceptances
        )
        / session_count
    )

    return DecisionMetrics(
        session_count=session_count,
        attack_session_count=(
            attack_session_count
        ),
        benign_session_count=(
            benign_session_count
        ),
        true_attack_rejections=(
            true_attack_rejections
        ),
        false_attack_acceptances=(
            false_attack_acceptances
        ),
        true_benign_acceptances=(
            true_benign_acceptances
        ),
        false_benign_rejections=(
            false_benign_rejections
        ),
        attack_detection_rate=(
            attack_detection_rate
        ),
        attack_acceptance_rate=(
            attack_acceptance_rate
        ),
        valid_user_acceptance_rate=(
            valid_user_acceptance_rate
        ),
        false_reject_rate=(
            false_reject_rate
        ),
        overall_acceptance_rate=(
            overall_acceptance_rate
        ),
        overall_rejection_rate=(
            overall_rejection_rate
        ),
        decision_accuracy=(
            decision_accuracy
        ),
    )


def threshold_decision_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> DecisionMetrics:
    """
    Calculate authentication outcomes for a probability threshold.

    Attack prediction means the session is rejected.
    """

    predicted_attack = (
        threshold_attack_predictions(
            probabilities,
            threshold,
        )
    )

    accepted = ~predicted_attack

    return authentication_decision_metrics(
        labels=labels,
        accepted=accepted,
    )


def confusion_counts(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, int]:
    """
    Return conventional attack-class confusion counts.

    Positive class:
        attack

    Negative class:
        benign
    """

    normalized_labels = normalize_binary_labels(
        labels,
        require_both_classes=True,
    )

    predicted_attack = (
        threshold_attack_predictions(
            probabilities,
            threshold,
        )
    ).astype(
        int
    )

    validate_equal_lengths(
        labels=normalized_labels,
        predictions=predicted_attack,
    )

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            normalized_labels,
            predicted_attack,
            labels=[
                0,
                1,
            ],
        ).ravel()
    )

    return {
        "true_negative":
            int(
                true_negative
            ),

        "false_positive":
            int(
                false_positive
            ),

        "false_negative":
            int(
                false_negative
            ),

        "true_positive":
            int(
                true_positive
            ),
    }


def bootstrap_mean_interval(
    values: Sequence[float],
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Return the mean and percentile-bootstrap confidence interval.

    The notebook uses 5,000 bootstrap resamples and a 95% interval.

    Non-finite values are excluded.

    When no finite values exist:
        (NaN, NaN, NaN) is returned.

    When only one finite value exists:
        (mean, NaN, NaN) is returned.
    """

    if (
        isinstance(
            resamples,
            bool,
        )
        or not isinstance(
            resamples,
            int,
        )
    ):
        raise TypeError(
            "resamples must be an integer."
        )

    if resamples < 1:
        raise InvalidBootstrapConfigurationError(
            "resamples must be at least 1."
        )

    if (
        isinstance(
            random_state,
            bool,
        )
        or not isinstance(
            random_state,
            int,
        )
    ):
        raise TypeError(
            "random_state must be an integer."
        )

    if (
        isinstance(
            confidence_level,
            bool,
        )
        or not isinstance(
            confidence_level,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            "confidence_level must be numeric."
        )

    confidence_level = float(
        confidence_level
    )

    if not 0.0 < confidence_level < 1.0:
        raise InvalidBootstrapConfigurationError(
            "confidence_level must be between 0 and 1."
        )

    try:
        normalized_values = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidMetricDataError(
            "Bootstrap values must be numeric."
        ) from error

    finite_values = normalized_values[
        np.isfinite(
            normalized_values
        )
    ]

    if len(finite_values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    mean_value = float(
        np.mean(
            finite_values
        )
    )

    if len(finite_values) == 1:
        return (
            mean_value,
            float("nan"),
            float("nan"),
        )

    local_rng = np.random.default_rng(
        random_state
    )

    samples = local_rng.choice(
        finite_values,
        size=(
            resamples,
            len(
                finite_values
            ),
        ),
        replace=True,
    )

    sample_means = np.mean(
        samples,
        axis=1,
    )

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100.0
        * alpha
        / 2.0
    )

    upper_percentile = (
        100.0
        * (
            1.0
            - alpha
            / 2.0
        )
    )

    return (
        mean_value,

        float(
            np.percentile(
                sample_means,
                lower_percentile,
            )
        ),

        float(
            np.percentile(
                sample_means,
                upper_percentile,
            )
        ),
    )


@dataclass(frozen=True)
class GPMetricsReport:
    """
    Complete held-out or independent GP evaluation report.
    """

    roc_auc: float
    pr_auc: float
    brier_score: float

    expected_calibration_error: float

    selected_threshold: float
    decision_metrics: DecisionMetrics

    calibration_curve: pd.DataFrame

    evaluation_scope: str = "held_out_test"

    def __post_init__(self) -> None:
        probability_fields = (
            "roc_auc",
            "pr_auc",
            "brier_score",
            "expected_calibration_error",
            "selected_threshold",
        )

        for field_name in probability_fields:
            value = float(
                getattr(
                    self,
                    field_name,
                )
            )

            if (
                not math.isfinite(
                    value
                )
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{field_name} must be finite and "
                    "between 0 and 1."
                )

        if not isinstance(
            self.decision_metrics,
            DecisionMetrics,
        ):
            raise TypeError(
                "decision_metrics must be DecisionMetrics."
            )

        if not isinstance(
            self.calibration_curve,
            pd.DataFrame,
        ):
            raise TypeError(
                "calibration_curve must be a pandas DataFrame."
            )

        if (
            not isinstance(
                self.evaluation_scope,
                str,
            )
            or not self.evaluation_scope.strip()
        ):
            raise ValueError(
                "evaluation_scope cannot be empty."
            )

    def as_dict(self) -> dict[str, Any]:
        """
        Return a paper-ready GP metric record.
        """

        return {
            "evaluation_scope":
                self.evaluation_scope,

            "roc_auc":
                self.roc_auc,

            "pr_auc":
                self.pr_auc,

            "brier_score":
                self.brier_score,

            "expected_calibration_error":
                self.expected_calibration_error,

            "selected_attack_threshold":
                self.selected_threshold,

            **self.decision_metrics.as_dict(),
        }


def evaluate_gp_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    *,
    evaluation_scope: str = "held_out_test",
    n_calibration_bins: int = (
        DEFAULT_CALIBRATION_BINS
    ),
    minimum_calibration_bin_size: int = (
        DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE
    ),
) -> GPMetricsReport:
    """
    Calculate complete GP probability and threshold-policy metrics.
    """

    normalized_labels = normalize_binary_labels(
        labels,
        require_both_classes=True,
    )

    normalized_probabilities = normalize_probabilities(
        probabilities
    )

    validate_equal_lengths(
        labels=normalized_labels,
        probabilities=normalized_probabilities,
    )

    normalized_threshold = validate_threshold(
        threshold
    )

    probability_results = probability_metrics(
        normalized_labels,
        normalized_probabilities,
    )

    decision_results = threshold_decision_metrics(
        labels=normalized_labels,
        probabilities=normalized_probabilities,
        threshold=normalized_threshold,
    )

    (
        calibration_curve,
        expected_calibration_error,
    ) = calibration_diagnostics(
        labels=normalized_labels,
        probabilities=normalized_probabilities,
        n_bins=n_calibration_bins,
        min_bin_size=(
            minimum_calibration_bin_size
        ),
    )

    return GPMetricsReport(
        roc_auc=probability_results[
            "roc_auc"
        ],
        pr_auc=probability_results[
            "pr_auc"
        ],
        brier_score=probability_results[
            "brier_score"
        ],
        expected_calibration_error=(
            expected_calibration_error
        ),
        selected_threshold=(
            normalized_threshold
        ),
        decision_metrics=(
            decision_results
        ),
        calibration_curve=(
            calibration_curve
        ),
        evaluation_scope=(
            evaluation_scope
        ),
    )


def metrics_dataframe(
    reports: Sequence[
        GPMetricsReport
    ],
) -> pd.DataFrame:
    """
    Convert multiple GP metric reports into a table.
    """

    if isinstance(
        reports,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "reports must be a sequence of GPMetricsReport objects."
        )

    rows: list[
        dict[str, Any]
    ] = []

    for index, report in enumerate(
        reports
    ):
        if not isinstance(
            report,
            GPMetricsReport,
        ):
            raise TypeError(
                f"reports[{index}] must be GPMetricsReport."
            )

        rows.append(
            report.as_dict()
        )

    return pd.DataFrame(
        rows
    )


def summarize_seed_metrics(
    seed_metrics: pd.DataFrame,
    *,
    group_column: str = "protocol",
    seed_column: str = "seed",
    metric_columns: Sequence[str],
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Build paper-ready seed-level means and 95% confidence intervals.

    Confidence intervals are calculated across independent seed-level
    values, not across every individual repeated session.
    """

    if not isinstance(
        seed_metrics,
        pd.DataFrame,
    ):
        raise TypeError(
            "seed_metrics must be a pandas DataFrame."
        )

    required_columns = {
        group_column,
        seed_column,
        *metric_columns,
    }

    missing_columns = sorted(
        required_columns.difference(
            seed_metrics.columns
        )
    )

    if missing_columns:
        raise InvalidMetricDataError(
            "Missing seed-metric columns: "
            f"{missing_columns!r}"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for group_name, group in seed_metrics.groupby(
        group_column,
        dropna=False,
    ):
        row: dict[str, Any] = {
            group_column:
                group_name,

            "independent_seed_count":
                int(
                    group[
                        seed_column
                    ].nunique()
                ),
        }

        for metric_index, metric_name in enumerate(
            metric_columns
        ):
            (
                mean_value,
                lower_bound,
                upper_bound,
            ) = bootstrap_mean_interval(
                group[
                    metric_name
                ],
                resamples=resamples,
                random_state=(
                    random_state
                    + metric_index
                ),
            )

            row[
                f"{metric_name}_mean"
            ] = mean_value

            row[
                f"{metric_name}_ci95_lower"
            ] = lower_bound

            row[
                f"{metric_name}_ci95_upper"
            ] = upper_bound

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_rejection_reason_distribution(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize end-to-end and deterministic rejection paths.

    Expected columns:

        protocol
        actual_attack
        accepted
        reason
        deterministic_reasons
    """

    if not isinstance(
        results,
        pd.DataFrame,
    ):
        raise TypeError(
            "results must be a pandas DataFrame."
        )

    required_columns = {
        "protocol",
        "actual_attack",
        "accepted",
        "reason",
        "deterministic_reasons",
    }

    missing_columns = sorted(
        required_columns.difference(
            results.columns
        )
    )

    if missing_columns:
        raise InvalidMetricDataError(
            "Missing rejection-distribution columns: "
            f"{missing_columns!r}"
        )

    working = results.copy()

    working[
        "deterministic_reasons"
    ] = (
        working[
            "deterministic_reasons"
        ]
        .fillna("")
        .astype(str)
    )

    return (
        working.groupby(
            [
                "protocol",
                "actual_attack",
                "accepted",
                "reason",
                "deterministic_reasons",
            ],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "session_count",
            }
        )
        .sort_values(
            [
                "protocol",
                "actual_attack",
                "accepted",
                "session_count",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def run_self_test() -> None:
    """
    Verify probability, policy, calibration, and bootstrap metrics.
    """

    labels = np.asarray(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
        ],
        dtype=int,
    )

    probabilities = np.asarray(
        [
            0.01,
            0.03,
            0.06,
            0.10,
            0.20,
            0.40,
            0.80,
            0.95,
        ],
        dtype=float,
    )

    report = evaluate_gp_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=0.15,
        evaluation_scope="self_test",
        n_calibration_bins=4,
        minimum_calibration_bin_size=2,
    )

    if not math.isclose(
        report.roc_auc,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MetricsError(
            "Perfect ranking did not produce ROC-AUC 1.0."
        )

    if not math.isclose(
        report.pr_auc,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MetricsError(
            "Perfect ranking did not produce PR-AUC 1.0."
        )

    if not math.isclose(
        report.decision_metrics.attack_detection_rate,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MetricsError(
            "Attack detection rate is incorrect."
        )

    if not math.isclose(
        report.decision_metrics
        .valid_user_acceptance_rate,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MetricsError(
            "Valid-user acceptance rate is incorrect."
        )

    boundary_predictions = (
        threshold_attack_predictions(
            [
                0.149999,
                0.15,
            ],
            0.15,
        )
    )

    if boundary_predictions.tolist() != [
        False,
        True,
    ]:
        raise MetricsError(
            "Threshold boundary behavior is incorrect."
        )

    (
        mean_value,
        lower_bound,
        upper_bound,
    ) = bootstrap_mean_interval(
        [
            0.80,
            0.85,
            0.90,
            0.95,
        ],
        resamples=1000,
        random_state=7,
    )

    if not math.isclose(
        mean_value,
        0.875,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise MetricsError(
            "Bootstrap mean is incorrect."
        )

    if not (
        lower_bound
        <= mean_value
        <= upper_bound
    ):
        raise MetricsError(
            "Bootstrap confidence interval is invalid."
        )

    metric_table = metrics_dataframe(
        [
            report,
        ]
    )

    if len(metric_table) != 1:
        raise MetricsError(
            "Metric table row count is incorrect."
        )

    print(
        "Machine-learning metrics self-test "
        "completed successfully."
    )

    print(
        "ROC-AUC:",
        f"{report.roc_auc:.6f}",
    )

    print(
        "PR-AUC:",
        f"{report.pr_auc:.6f}",
    )

    print(
        "Brier score:",
        f"{report.brier_score:.6f}",
    )

    print(
        "Attack detection rate:",
        f"{report.decision_metrics.attack_detection_rate:.6f}",
    )

    print(
        "Valid-user acceptance rate:",
        (
            f"{report.decision_metrics.valid_user_acceptance_rate:.6f}"
        ),
    )


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_CALIBRATION_BINS",
    "DEFAULT_MINIMUM_CALIBRATION_BIN_SIZE",
    "MetricsError",
    "InvalidMetricDataError",
    "MissingMetricClassError",
    "InvalidBootstrapConfigurationError",
    "MetricConsistencyError",
    "DecisionMetrics",
    "GPMetricsReport",
    "normalize_binary_labels",
    "normalize_probabilities",
    "normalize_acceptance_decisions",
    "validate_equal_lengths",
    "validate_threshold",
    "probability_metrics",
    "threshold_attack_predictions",
    "authentication_decision_metrics",
    "threshold_decision_metrics",
    "confusion_counts",
    "bootstrap_mean_interval",
    "evaluate_gp_metrics",
    "metrics_dataframe",
    "summarize_seed_metrics",
    "build_rejection_reason_distribution",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        MetricsError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[METRICS ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error