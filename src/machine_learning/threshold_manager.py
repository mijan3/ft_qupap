"""
FT-QuPAP GP Threshold Manager

This module selects, stores, loads, and exposes the calibrated
Gaussian Process attack-probability threshold.

Notebook-compatible threshold selection
=======================================

The threshold must be selected using only the disjoint calibration
split.

For every candidate threshold:

    rejected = attack_probability >= threshold

    false_accept_rate =
        fraction of attack sessions that were not rejected

    false_reject_rate =
        fraction of benign sessions that were rejected

    Bayesian risk =
        false_accept_cost * false_accept_rate
        + false_reject_cost * false_reject_rate

The candidate with the minimum Bayesian risk is selected.

When multiple candidates have equal risk, the candidate closest to the
theoretical Bayesian threshold is selected:

    theoretical_threshold =
        false_reject_cost
        / (false_accept_cost + false_reject_cost)

The final deployed operational threshold is:

    operational_threshold = max(
        raw_calibration_threshold,
        minimum_operational_threshold,
    )

Default FT-QuPAP values:

    false_accept_cost = 10.0
    false_reject_cost = 1.0
    minimum_operational_threshold = 0.15
    GP gray-zone retry upper bound = 0.20

Important:

- Threshold selection must not use held-out test sessions.
- The selected threshold must be fixed before independent evaluation.
- A session is rejected when P(attack) >= threshold.
- A probability exactly equal to the threshold is rejected.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np


DEFAULT_FALSE_ACCEPT_COST = 10.0
DEFAULT_FALSE_REJECT_COST = 1.0

DEFAULT_MIN_OPERATIONAL_THRESHOLD = 0.15
DEFAULT_GP_GRAY_ZONE_RETRY_UPPER = 0.20

THRESHOLD_ARTIFACT_VERSION = (
    "FT-QuPAP-GP-Threshold-1.0"
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "threshold.json"
)


class ThresholdManagerError(Exception):
    """Base exception for GP threshold-management failures."""


class InvalidThresholdDataError(
    ThresholdManagerError
):
    """Raised when calibration labels or probabilities are invalid."""


class InvalidThresholdPolicyError(
    ThresholdManagerError
):
    """Raised when threshold-policy values are invalid."""


class ThresholdSelectionError(
    ThresholdManagerError
):
    """Raised when a threshold cannot be selected."""


class ThresholdArtifactError(
    ThresholdManagerError
):
    """Raised when a threshold artifact is malformed."""


class ThresholdNotAvailableError(
    ThresholdManagerError
):
    """Raised when no operational threshold is available."""


def validate_probability(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> float | None:
    """
    Validate a finite probability in the interval [0, 1].
    """

    if value is None:
        if allow_none:
            return None

        raise InvalidThresholdDataError(
            f"{field_name} cannot be None."
        )

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
        raise InvalidThresholdDataError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise InvalidThresholdDataError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def validate_positive_cost(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate a positive Bayesian decision cost.
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
        raise InvalidThresholdPolicyError(
            f"{field_name} must be finite."
        )

    if normalized <= 0.0:
        raise InvalidThresholdPolicyError(
            f"{field_name} must be greater than zero."
        )

    return normalized


def normalize_probability_array(
    probabilities: Any,
) -> np.ndarray:
    """
    Convert calibration probabilities to a one-dimensional array.
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
        raise InvalidThresholdDataError(
            "Probabilities could not be converted to numbers."
        ) from error

    if normalized.size == 0:
        raise InvalidThresholdDataError(
            "Probabilities cannot be empty."
        )

    if not np.all(
        np.isfinite(normalized)
    ):
        raise InvalidThresholdDataError(
            "Probabilities cannot contain NaN or infinity."
        )

    if (
        np.any(normalized < 0.0)
        or np.any(normalized > 1.0)
    ):
        raise InvalidThresholdDataError(
            "Probabilities must be between 0 and 1."
        )

    return normalized.astype(
        float,
        copy=True,
    )


def normalize_binary_labels(
    labels: Any,
    *,
    require_both_classes: bool = True,
) -> np.ndarray:
    """
    Convert calibration labels to binary integers.

    FT-QuPAP uses:

        benign = 0
        attack = 1
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
        raise InvalidThresholdDataError(
            "Labels could not be converted to numbers."
        ) from error

    if numeric_labels.size == 0:
        raise InvalidThresholdDataError(
            "Labels cannot be empty."
        )

    if not np.all(
        np.isfinite(numeric_labels)
    ):
        raise InvalidThresholdDataError(
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
        raise InvalidThresholdDataError(
            "Labels must contain only 0 and 1."
        )

    normalized = numeric_labels.astype(
        int,
        copy=True,
    )

    if (
        require_both_classes
        and set(normalized.tolist())
        != {
            0,
            1,
        }
    ):
        raise InvalidThresholdDataError(
            "Threshold selection requires attack "
            "and benign sessions."
        )

    return normalized


def validate_calibration_data(
    labels: Any,
    probabilities: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate paired calibration labels and probabilities.
    """

    normalized_labels = normalize_binary_labels(
        labels,
        require_both_classes=True,
    )

    normalized_probabilities = (
        normalize_probability_array(
            probabilities
        )
    )

    if len(normalized_labels) != len(
        normalized_probabilities
    ):
        raise InvalidThresholdDataError(
            "Labels and probabilities must have equal length."
        )

    return (
        normalized_labels,
        normalized_probabilities,
    )


def theoretical_bayes_threshold(
    false_accept_cost: float = (
        DEFAULT_FALSE_ACCEPT_COST
    ),
    false_reject_cost: float = (
        DEFAULT_FALSE_REJECT_COST
    ),
) -> float:
    """
    Calculate the theoretical Bayesian probability threshold.

    Formula:

        false_reject_cost
        ---------------------------------------------
        false_accept_cost + false_reject_cost
    """

    false_accept_cost = validate_positive_cost(
        false_accept_cost,
        "false_accept_cost",
    )

    false_reject_cost = validate_positive_cost(
        false_reject_cost,
        "false_reject_cost",
    )

    return float(
        false_reject_cost
        / (
            false_accept_cost
            + false_reject_cost
        )
    )


def build_threshold_candidates(
    probabilities: Any,
) -> np.ndarray:
    """
    Build notebook-compatible threshold candidates.

    Candidates include:

    - 0.0
    - midpoints between adjacent unique probabilities
    - 1.0

    When every probability is identical, only 0.0 and 1.0 are used.
    """

    normalized_probabilities = (
        normalize_probability_array(
            probabilities
        )
    )

    values = np.unique(
        normalized_probabilities
    )

    if len(values) == 1:
        return np.asarray(
            [
                0.0,
                1.0,
            ],
            dtype=float,
        )

    midpoints = (
        values[:-1]
        + values[1:]
    ) / 2.0

    candidates = np.unique(
        np.concatenate(
            (
                np.asarray(
                    [
                        0.0,
                    ],
                    dtype=float,
                ),
                midpoints,
                np.asarray(
                    [
                        1.0,
                    ],
                    dtype=float,
                ),
            )
        )
    )

    return candidates.astype(
        float,
        copy=False,
    )


@dataclass(frozen=True)
class ThresholdRates:
    """
    Error rates produced by one threshold.
    """

    threshold: float

    false_accept_rate: float
    false_reject_rate: float

    attack_detection_rate: float
    valid_user_acceptance_rate: float

    risk: float

    attack_count: int
    benign_count: int

    def __post_init__(self) -> None:
        probability_fields = {
            "threshold":
                self.threshold,
            "false_accept_rate":
                self.false_accept_rate,
            "false_reject_rate":
                self.false_reject_rate,
            "attack_detection_rate":
                self.attack_detection_rate,
            "valid_user_acceptance_rate":
                self.valid_user_acceptance_rate,
        }

        for field_name, value in probability_fields.items():
            object.__setattr__(
                self,
                field_name,
                validate_probability(
                    value,
                    field_name,
                ),
            )

        if (
            isinstance(self.risk, bool)
            or not isinstance(
                self.risk,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
        ):
            raise TypeError(
                "risk must be numeric."
            )

        normalized_risk = float(
            self.risk
        )

        if (
            not math.isfinite(
                normalized_risk
            )
            or normalized_risk < 0.0
        ):
            raise ValueError(
                "risk must be finite and nonnegative."
            )

        object.__setattr__(
            self,
            "risk",
            normalized_risk,
        )

        for field_name in (
            "attack_count",
            "benign_count",
        ):
            value = getattr(
                self,
                field_name,
            )

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

            if int(value) < 1:
                raise ValueError(
                    f"{field_name} must be at least 1."
                )

            object.__setattr__(
                self,
                field_name,
                int(value),
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable threshold-rate record."""

        return {
            "threshold":
                self.threshold,

            "false_accept_rate":
                self.false_accept_rate,

            "false_reject_rate":
                self.false_reject_rate,

            "attack_detection_rate":
                self.attack_detection_rate,

            "valid_user_acceptance_rate":
                self.valid_user_acceptance_rate,

            "risk":
                self.risk,

            "attack_count":
                self.attack_count,

            "benign_count":
                self.benign_count,
        }


def evaluate_threshold(
    labels: Any,
    probabilities: Any,
    threshold: float,
    false_accept_cost: float = (
        DEFAULT_FALSE_ACCEPT_COST
    ),
    false_reject_cost: float = (
        DEFAULT_FALSE_REJECT_COST
    ),
) -> ThresholdRates:
    """
    Evaluate one rejection threshold.

    Rejection rule:

        rejected = P(attack) >= threshold
    """

    (
        normalized_labels,
        normalized_probabilities,
    ) = validate_calibration_data(
        labels,
        probabilities,
    )

    threshold = validate_probability(
        threshold,
        "threshold",
    )

    false_accept_cost = validate_positive_cost(
        false_accept_cost,
        "false_accept_cost",
    )

    false_reject_cost = validate_positive_cost(
        false_reject_cost,
        "false_reject_cost",
    )

    attack_mask = (
        normalized_labels == 1
    )

    benign_mask = (
        normalized_labels == 0
    )

    rejected = (
        normalized_probabilities
        >= threshold
    )

    false_accept_rate = float(
        np.mean(
            ~rejected[
                attack_mask
            ]
        )
    )

    false_reject_rate = float(
        np.mean(
            rejected[
                benign_mask
            ]
        )
    )

    attack_detection_rate = float(
        np.mean(
            rejected[
                attack_mask
            ]
        )
    )

    valid_user_acceptance_rate = float(
        np.mean(
            ~rejected[
                benign_mask
            ]
        )
    )

    risk = float(
        false_accept_cost
        * false_accept_rate
        + false_reject_cost
        * false_reject_rate
    )

    return ThresholdRates(
        threshold=threshold,
        false_accept_rate=(
            false_accept_rate
        ),
        false_reject_rate=(
            false_reject_rate
        ),
        attack_detection_rate=(
            attack_detection_rate
        ),
        valid_user_acceptance_rate=(
            valid_user_acceptance_rate
        ),
        risk=risk,
        attack_count=int(
            np.sum(
                attack_mask
            )
        ),
        benign_count=int(
            np.sum(
                benign_mask
            )
        ),
    )


def select_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    false_accept_cost: float,
    false_reject_cost: float,
) -> tuple[float, float, float]:
    """
    Select the raw calibration threshold.

    This function preserves the notebook-compatible public interface.

    Returns:
        threshold
        false_accept_rate
        false_reject_rate
    """

    (
        normalized_labels,
        normalized_probabilities,
    ) = validate_calibration_data(
        labels,
        probabilities,
    )

    false_accept_cost = validate_positive_cost(
        false_accept_cost,
        "false_accept_cost",
    )

    false_reject_cost = validate_positive_cost(
        false_reject_cost,
        "false_reject_cost",
    )

    attack_mask = (
        normalized_labels == 1
    )

    benign_mask = (
        normalized_labels == 0
    )

    candidates = build_threshold_candidates(
        normalized_probabilities
    )

    theoretical_threshold = (
        theoretical_bayes_threshold(
            false_accept_cost,
            false_reject_cost,
        )
    )

    scored: list[
        tuple[
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []

    for threshold in candidates:
        rejected = (
            normalized_probabilities
            >= threshold
        )

        false_accept_rate = float(
            np.mean(
                ~rejected[
                    attack_mask
                ]
            )
        )

        false_reject_rate = float(
            np.mean(
                rejected[
                    benign_mask
                ]
            )
        )

        risk = float(
            false_accept_cost
            * false_accept_rate
            + false_reject_cost
            * false_reject_rate
        )

        scored.append(
            (
                risk,
                abs(
                    float(threshold)
                    - theoretical_threshold
                ),
                float(threshold),
                false_accept_rate,
                false_reject_rate,
            )
        )

    if not scored:
        raise ThresholdSelectionError(
            "No threshold candidates were generated."
        )

    (
        _,
        _,
        selected_threshold,
        false_accept_rate,
        false_reject_rate,
    ) = min(
        scored,
        key=lambda item: (
            item[0],
            item[1],
        ),
    )

    return (
        float(selected_threshold),
        float(false_accept_rate),
        float(false_reject_rate),
    )


@dataclass(frozen=True)
class ThresholdPolicy:
    """
    FT-QuPAP threshold-selection policy.
    """

    false_accept_cost: float = (
        DEFAULT_FALSE_ACCEPT_COST
    )

    false_reject_cost: float = (
        DEFAULT_FALSE_REJECT_COST
    )

    min_operational_threshold: float = (
        DEFAULT_MIN_OPERATIONAL_THRESHOLD
    )

    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_RETRY_UPPER
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "false_accept_cost",
            validate_positive_cost(
                self.false_accept_cost,
                "false_accept_cost",
            ),
        )

        object.__setattr__(
            self,
            "false_reject_cost",
            validate_positive_cost(
                self.false_reject_cost,
                "false_reject_cost",
            ),
        )

        object.__setattr__(
            self,
            "min_operational_threshold",
            validate_probability(
                self.min_operational_threshold,
                "min_operational_threshold",
            ),
        )

        object.__setattr__(
            self,
            "gp_gray_zone_retry_upper",
            validate_probability(
                self.gp_gray_zone_retry_upper,
                "gp_gray_zone_retry_upper",
            ),
        )

    @property
    def theoretical_threshold(self) -> float:
        """Return the policy's theoretical Bayesian threshold."""

        return theoretical_bayes_threshold(
            self.false_accept_cost,
            self.false_reject_cost,
        )

    def apply_operational_floor(
        self,
        raw_threshold: float,
    ) -> float:
        """
        Apply the configured minimum operational threshold.
        """

        raw_threshold = validate_probability(
            raw_threshold,
            "raw_threshold",
        )

        return float(
            max(
                raw_threshold,
                self.min_operational_threshold,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe policy record."""

        return {
            "false_accept_cost":
                self.false_accept_cost,

            "false_reject_cost":
                self.false_reject_cost,

            "theoretical_bayes_threshold":
                self.theoretical_threshold,

            "min_operational_gp_threshold":
                self.min_operational_threshold,

            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,
        }


DEFAULT_THRESHOLD_POLICY = ThresholdPolicy()


@dataclass(frozen=True)
class ThresholdSelection:
    """
    Complete result of calibration threshold selection.
    """

    raw_threshold: float
    operational_threshold: float

    raw_rates: ThresholdRates
    operational_rates: ThresholdRates

    theoretical_threshold: float
    minimum_operational_threshold: float

    false_accept_cost: float
    false_reject_cost: float

    gp_gray_zone_retry_upper: float

    candidate_count: int
    calibration_rows: int

    def __post_init__(self) -> None:
        probability_fields = (
            "raw_threshold",
            "operational_threshold",
            "theoretical_threshold",
            "minimum_operational_threshold",
            "gp_gray_zone_retry_upper",
        )

        for field_name in probability_fields:
            object.__setattr__(
                self,
                field_name,
                validate_probability(
                    getattr(
                        self,
                        field_name,
                    ),
                    field_name,
                ),
            )

        object.__setattr__(
            self,
            "false_accept_cost",
            validate_positive_cost(
                self.false_accept_cost,
                "false_accept_cost",
            ),
        )

        object.__setattr__(
            self,
            "false_reject_cost",
            validate_positive_cost(
                self.false_reject_cost,
                "false_reject_cost",
            ),
        )

        if not isinstance(
            self.raw_rates,
            ThresholdRates,
        ):
            raise TypeError(
                "raw_rates must be ThresholdRates."
            )

        if not isinstance(
            self.operational_rates,
            ThresholdRates,
        ):
            raise TypeError(
                "operational_rates must be ThresholdRates."
            )

        for field_name in (
            "candidate_count",
            "calibration_rows",
        ):
            value = getattr(
                self,
                field_name,
            )

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

            if int(value) < 1:
                raise ValueError(
                    f"{field_name} must be at least 1."
                )

            object.__setattr__(
                self,
                field_name,
                int(value),
            )

        if not math.isclose(
            self.raw_rates.threshold,
            self.raw_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ThresholdSelectionError(
                "raw_rates threshold does not match raw_threshold."
            )

        if not math.isclose(
            self.operational_rates.threshold,
            self.operational_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ThresholdSelectionError(
                "operational_rates threshold does not match "
                "operational_threshold."
            )

        if (
            self.operational_threshold
            < self.minimum_operational_threshold
        ):
            raise ThresholdSelectionError(
                "Operational threshold is below the configured floor."
            )

    @property
    def floor_applied(self) -> bool:
        """
        Return whether the minimum operational threshold changed
        the raw threshold.
        """

        return bool(
            self.operational_threshold
            > self.raw_threshold
        )

    @property
    def calibration_false_accept_rate(self) -> float:
        """Return FAR at the deployed operational threshold."""

        return (
            self.operational_rates
            .false_accept_rate
        )

    @property
    def calibration_false_reject_rate(self) -> float:
        """Return FRR at the deployed operational threshold."""

        return (
            self.operational_rates
            .false_reject_rate
        )

    def as_dict(self) -> dict[str, Any]:
        """Return complete threshold-selection evidence."""

        return {
            "raw_calibration_gp_attack_threshold":
                self.raw_threshold,

            "gp_attack_threshold":
                self.operational_threshold,

            "min_operational_gp_threshold":
                self.minimum_operational_threshold,

            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,

            "theoretical_bayes_threshold":
                self.theoretical_threshold,

            "bayes_cost_false_accept":
                self.false_accept_cost,

            "bayes_cost_false_reject":
                self.false_reject_cost,

            "calibration_false_accept_rate":
                self.calibration_false_accept_rate,

            "calibration_false_reject_rate":
                self.calibration_false_reject_rate,

            "calibration_attack_detection_rate":
                self.operational_rates
                .attack_detection_rate,

            "calibration_valid_user_acceptance_rate":
                self.operational_rates
                .valid_user_acceptance_rate,

            "raw_threshold_false_accept_rate":
                self.raw_rates
                .false_accept_rate,

            "raw_threshold_false_reject_rate":
                self.raw_rates
                .false_reject_rate,

            "raw_threshold_bayes_risk":
                self.raw_rates.risk,

            "operational_threshold_bayes_risk":
                self.operational_rates.risk,

            "operational_floor_applied":
                self.floor_applied,

            "candidate_count":
                self.candidate_count,

            "calibration_rows":
                self.calibration_rows,
        }


def select_operational_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
    policy: ThresholdPolicy | None = None,
) -> ThresholdSelection:
    """
    Select raw and operational FT-QuPAP GP thresholds.
    """

    active_policy = (
        DEFAULT_THRESHOLD_POLICY
        if policy is None
        else policy
    )

    if not isinstance(
        active_policy,
        ThresholdPolicy,
    ):
        raise TypeError(
            "policy must be ThresholdPolicy or None."
        )

    (
        normalized_labels,
        normalized_probabilities,
    ) = validate_calibration_data(
        labels,
        probabilities,
    )

    candidates = build_threshold_candidates(
        normalized_probabilities
    )

    (
        raw_threshold,
        _,
        _,
    ) = select_threshold(
        labels=normalized_labels,
        probabilities=(
            normalized_probabilities
        ),
        false_accept_cost=(
            active_policy
            .false_accept_cost
        ),
        false_reject_cost=(
            active_policy
            .false_reject_cost
        ),
    )

    operational_threshold = (
        active_policy.apply_operational_floor(
            raw_threshold
        )
    )

    raw_rates = evaluate_threshold(
        labels=normalized_labels,
        probabilities=(
            normalized_probabilities
        ),
        threshold=raw_threshold,
        false_accept_cost=(
            active_policy
            .false_accept_cost
        ),
        false_reject_cost=(
            active_policy
            .false_reject_cost
        ),
    )

    operational_rates = evaluate_threshold(
        labels=normalized_labels,
        probabilities=(
            normalized_probabilities
        ),
        threshold=(
            operational_threshold
        ),
        false_accept_cost=(
            active_policy
            .false_accept_cost
        ),
        false_reject_cost=(
            active_policy
            .false_reject_cost
        ),
    )

    return ThresholdSelection(
        raw_threshold=raw_threshold,
        operational_threshold=(
            operational_threshold
        ),
        raw_rates=raw_rates,
        operational_rates=(
            operational_rates
        ),
        theoretical_threshold=(
            active_policy
            .theoretical_threshold
        ),
        minimum_operational_threshold=(
            active_policy
            .min_operational_threshold
        ),
        false_accept_cost=(
            active_policy
            .false_accept_cost
        ),
        false_reject_cost=(
            active_policy
            .false_reject_cost
        ),
        gp_gray_zone_retry_upper=(
            active_policy
            .gp_gray_zone_retry_upper
        ),
        candidate_count=len(
            candidates
        ),
        calibration_rows=len(
            normalized_labels
        ),
    )


@dataclass
class ThresholdConfiguration:
    """
    Loaded or selected GP threshold configuration.
    """

    gp_attack_threshold: float

    raw_calibration_gp_attack_threshold: (
        float | None
    ) = None

    min_operational_gp_threshold: float = (
        DEFAULT_MIN_OPERATIONAL_THRESHOLD
    )

    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_RETRY_UPPER
    )

    bayes_cost_false_accept: float = (
        DEFAULT_FALSE_ACCEPT_COST
    )

    bayes_cost_false_reject: float = (
        DEFAULT_FALSE_REJECT_COST
    )

    calibration_false_accept_rate: (
        float | None
    ) = None

    calibration_false_reject_rate: (
        float | None
    ) = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.gp_attack_threshold = (
            validate_probability(
                self.gp_attack_threshold,
                "gp_attack_threshold",
            )
        )

        self.raw_calibration_gp_attack_threshold = (
            validate_probability(
                self.raw_calibration_gp_attack_threshold,
                (
                    "raw_calibration_"
                    "gp_attack_threshold"
                ),
                allow_none=True,
            )
        )

        self.min_operational_gp_threshold = (
            validate_probability(
                self.min_operational_gp_threshold,
                "min_operational_gp_threshold",
            )
        )

        self.gp_gray_zone_retry_upper = (
            validate_probability(
                self.gp_gray_zone_retry_upper,
                "gp_gray_zone_retry_upper",
            )
        )

        self.bayes_cost_false_accept = (
            validate_positive_cost(
                self.bayes_cost_false_accept,
                "bayes_cost_false_accept",
            )
        )

        self.bayes_cost_false_reject = (
            validate_positive_cost(
                self.bayes_cost_false_reject,
                "bayes_cost_false_reject",
            )
        )

        self.calibration_false_accept_rate = (
            validate_probability(
                self.calibration_false_accept_rate,
                "calibration_false_accept_rate",
                allow_none=True,
            )
        )

        self.calibration_false_reject_rate = (
            validate_probability(
                self.calibration_false_reject_rate,
                "calibration_false_reject_rate",
                allow_none=True,
            )
        )

        if (
            self.gp_attack_threshold
            < self.min_operational_gp_threshold
        ):
            raise InvalidThresholdPolicyError(
                "gp_attack_threshold cannot be below "
                "min_operational_gp_threshold."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        self.metadata = copy.deepcopy(
            self.metadata
        )

    @classmethod
    def from_selection(
        cls,
        selection: ThresholdSelection,
    ) -> "ThresholdConfiguration":
        """
        Build configuration from calibration selection evidence.
        """

        if not isinstance(
            selection,
            ThresholdSelection,
        ):
            raise TypeError(
                "selection must be ThresholdSelection."
            )

        return cls(
            gp_attack_threshold=(
                selection
                .operational_threshold
            ),
            raw_calibration_gp_attack_threshold=(
                selection.raw_threshold
            ),
            min_operational_gp_threshold=(
                selection
                .minimum_operational_threshold
            ),
            gp_gray_zone_retry_upper=(
                selection
                .gp_gray_zone_retry_upper
            ),
            bayes_cost_false_accept=(
                selection
                .false_accept_cost
            ),
            bayes_cost_false_reject=(
                selection
                .false_reject_cost
            ),
            calibration_false_accept_rate=(
                selection
                .calibration_false_accept_rate
            ),
            calibration_false_reject_rate=(
                selection
                .calibration_false_reject_rate
            ),
            metadata={
                "selection":
                    selection.as_dict(),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe threshold configuration."""

        record: dict[str, Any] = {
            "artifact_version":
                THRESHOLD_ARTIFACT_VERSION,

            "raw_calibration_gp_attack_threshold":
                self.raw_calibration_gp_attack_threshold,

            "gp_attack_threshold":
                self.gp_attack_threshold,

            "min_operational_gp_threshold":
                self.min_operational_gp_threshold,

            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,

            "bayes_cost_false_accept":
                self.bayes_cost_false_accept,

            "bayes_cost_false_reject":
                self.bayes_cost_false_reject,

            "calibration_false_accept_rate":
                self.calibration_false_accept_rate,

            "calibration_false_reject_rate":
                self.calibration_false_reject_rate,
        }

        if self.metadata:
            record[
                "metadata"
            ] = copy.deepcopy(
                self.metadata
            )

        return record


_CURRENT_CONFIGURATION: (
    ThresholdConfiguration | None
) = None

_THRESHOLD_LOCK = RLock()


def set_current_threshold(
    threshold: (
        float
        | ThresholdConfiguration
        | ThresholdSelection
    ),
) -> ThresholdConfiguration:
    """
    Install the process-wide active GP threshold.
    """

    if isinstance(
        threshold,
        ThresholdConfiguration,
    ):
        configuration = copy.deepcopy(
            threshold
        )

    elif isinstance(
        threshold,
        ThresholdSelection,
    ):
        configuration = (
            ThresholdConfiguration
            .from_selection(
                threshold
            )
        )

    else:
        configuration = (
            ThresholdConfiguration(
                gp_attack_threshold=(
                    validate_probability(
                        threshold,
                        "threshold",
                    )
                ),
                min_operational_gp_threshold=0.0,
            )
        )

    global _CURRENT_CONFIGURATION

    with _THRESHOLD_LOCK:
        _CURRENT_CONFIGURATION = (
            configuration
        )

    return copy.deepcopy(
        configuration
    )


def clear_current_threshold() -> None:
    """Remove the process-wide active threshold."""

    global _CURRENT_CONFIGURATION

    with _THRESHOLD_LOCK:
        _CURRENT_CONFIGURATION = None


def current_configuration() -> (
    ThresholdConfiguration | None
):
    """
    Return a detached active threshold configuration.
    """

    with _THRESHOLD_LOCK:
        if _CURRENT_CONFIGURATION is None:
            return None

        return copy.deepcopy(
            _CURRENT_CONFIGURATION
        )


def current_threshold() -> float | None:
    """
    Return the currently installed operational GP threshold.

    This preserves the original project-template function name.
    """

    configuration = current_configuration()

    if configuration is None:
        return None

    return float(
        configuration.gp_attack_threshold
    )


def require_current_threshold() -> float:
    """
    Return the active threshold or raise an explicit error.
    """

    threshold = current_threshold()

    if threshold is None:
        raise ThresholdNotAvailableError(
            "No operational GP threshold is currently installed."
        )

    return threshold


def save_threshold_configuration(
    configuration: ThresholdConfiguration,
    destination: str | Path = (
        DEFAULT_THRESHOLD_PATH
    ),
) -> Path:
    """
    Save a threshold configuration as JSON.
    """

    if not isinstance(
        configuration,
        ThresholdConfiguration,
    ):
        raise TypeError(
            "configuration must be ThresholdConfiguration."
        )

    path = Path(
        destination
    )

    if path.suffix.lower() != ".json":
        raise ThresholdArtifactError(
            "Threshold artifact path must end with .json."
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                configuration.as_dict(),
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
        raise ThresholdArtifactError(
            f"Could not save threshold artifact to {path}."
        ) from error

    return path


def load_threshold_configuration(
    source: str | Path = (
        DEFAULT_THRESHOLD_PATH
    ),
    *,
    install: bool = False,
) -> ThresholdConfiguration:
    """
    Load and validate a trusted threshold JSON artifact.

    Supported formats:

        {
            "gp_attack_threshold": 0.15
        }

    and the complete artifact produced by
    save_threshold_configuration().
    """

    if not isinstance(
        install,
        bool,
    ):
        raise TypeError(
            "install must be boolean."
        )

    path = Path(
        source
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Threshold artifact does not exist: {path}"
        )

    if not path.is_file():
        raise ThresholdArtifactError(
            f"Threshold artifact path is not a file: {path}"
        )

    if path.suffix.lower() != ".json":
        raise ThresholdArtifactError(
            "Threshold artifact path must end with .json."
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
        raise ThresholdArtifactError(
            f"Could not read threshold artifact: {path}"
        ) from error

    if isinstance(
        loaded,
        (
            int,
            float,
        ),
    ) and not isinstance(
        loaded,
        bool,
    ):
        configuration = (
            ThresholdConfiguration(
                gp_attack_threshold=float(
                    loaded
                ),
                min_operational_gp_threshold=0.0,
            )
        )

    elif isinstance(
        loaded,
        Mapping,
    ):
        artifact_version = loaded.get(
            "artifact_version"
        )

        if (
            artifact_version is not None
            and artifact_version
            != THRESHOLD_ARTIFACT_VERSION
        ):
            raise ThresholdArtifactError(
                "Unsupported threshold artifact version: "
                f"{artifact_version!r}"
            )

        operational_threshold = None

        for field_name in (
            "gp_attack_threshold",
            "operational_attack_threshold",
            "operational_threshold",
            "threshold",
        ):
            if field_name in loaded:
                operational_threshold = (
                    loaded[field_name]
                )
                break

        if operational_threshold is None:
            raise ThresholdArtifactError(
                "Threshold artifact does not contain "
                "gp_attack_threshold."
            )

        raw_threshold = None

        for field_name in (
            "raw_calibration_gp_attack_threshold",
            "raw_calibration_threshold",
            "raw_threshold",
        ):
            if field_name in loaded:
                raw_threshold = loaded[
                    field_name
                ]
                break

        known_keys = {
            "artifact_version",
            "gp_attack_threshold",
            "operational_attack_threshold",
            "operational_threshold",
            "threshold",
            "raw_calibration_gp_attack_threshold",
            "raw_calibration_threshold",
            "raw_threshold",
            "min_operational_gp_threshold",
            "gp_gray_zone_retry_upper",
            "bayes_cost_false_accept",
            "bayes_cost_false_reject",
            "calibration_false_accept_rate",
            "calibration_false_reject_rate",
            "metadata",
        }

        metadata = {
            str(key): copy.deepcopy(
                value
            )
            for key, value
            in loaded.items()
            if key not in known_keys
        }

        supplied_metadata = loaded.get(
            "metadata"
        )

        if supplied_metadata is not None:
            if not isinstance(
                supplied_metadata,
                Mapping,
            ):
                raise ThresholdArtifactError(
                    "Threshold metadata must be a JSON object."
                )

            metadata.update(
                copy.deepcopy(
                    dict(
                        supplied_metadata
                    )
                )
            )

        configuration = ThresholdConfiguration(
            gp_attack_threshold=(
                operational_threshold
            ),
            raw_calibration_gp_attack_threshold=(
                raw_threshold
            ),
            min_operational_gp_threshold=(
                loaded.get(
                    "min_operational_gp_threshold",
                    min(
                        DEFAULT_MIN_OPERATIONAL_THRESHOLD,
                        float(
                            operational_threshold
                        ),
                    ),
                )
            ),
            gp_gray_zone_retry_upper=(
                loaded.get(
                    "gp_gray_zone_retry_upper",
                    DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
                )
            ),
            bayes_cost_false_accept=(
                loaded.get(
                    "bayes_cost_false_accept",
                    DEFAULT_FALSE_ACCEPT_COST,
                )
            ),
            bayes_cost_false_reject=(
                loaded.get(
                    "bayes_cost_false_reject",
                    DEFAULT_FALSE_REJECT_COST,
                )
            ),
            calibration_false_accept_rate=(
                loaded.get(
                    "calibration_false_accept_rate"
                )
            ),
            calibration_false_reject_rate=(
                loaded.get(
                    "calibration_false_reject_rate"
                )
            ),
            metadata=metadata,
        )

    else:
        raise ThresholdArtifactError(
            "Threshold artifact must contain a number "
            "or JSON object."
        )

    if install:
        set_current_threshold(
            configuration
        )

    return configuration


class ThresholdManager:
    """
    Reusable FT-QuPAP threshold-selection service.
    """

    def __init__(
        self,
        policy: ThresholdPolicy | None = None,
    ) -> None:
        self.policy = (
            DEFAULT_THRESHOLD_POLICY
            if policy is None
            else policy
        )

        if not isinstance(
            self.policy,
            ThresholdPolicy,
        ):
            raise TypeError(
                "policy must be ThresholdPolicy or None."
            )

        self.selection: (
            ThresholdSelection | None
        ) = None

        self.configuration: (
            ThresholdConfiguration | None
        ) = None

    @property
    def fitted(self) -> bool:
        """Return whether a calibration threshold was selected."""

        return self.selection is not None

    @property
    def available(self) -> bool:
        """Return whether an operational threshold is available."""

        return self.configuration is not None

    @property
    def threshold(self) -> float | None:
        """Return the manager's operational threshold."""

        if self.configuration is None:
            return None

        return float(
            self.configuration
            .gp_attack_threshold
        )

    def select(
        self,
        labels: Sequence[int],
        probabilities: Sequence[float],
        *,
        install: bool = True,
    ) -> ThresholdSelection:
        """
        Select the calibration and operational thresholds.
        """

        if not isinstance(
            install,
            bool,
        ):
            raise TypeError(
                "install must be boolean."
            )

        self.selection = (
            select_operational_threshold(
                labels=labels,
                probabilities=probabilities,
                policy=self.policy,
            )
        )

        self.configuration = (
            ThresholdConfiguration
            .from_selection(
                self.selection
            )
        )

        if install:
            set_current_threshold(
                self.configuration
            )

        return self.selection

    def evaluate(
        self,
        labels: Sequence[int],
        probabilities: Sequence[float],
        threshold: float | None = None,
    ) -> ThresholdRates:
        """
        Evaluate a threshold against supplied labeled sessions.
        """

        selected_threshold = (
            self.threshold
            if threshold is None
            else threshold
        )

        if selected_threshold is None:
            raise ThresholdNotAvailableError(
                "No threshold is available for evaluation."
            )

        return evaluate_threshold(
            labels=labels,
            probabilities=probabilities,
            threshold=selected_threshold,
            false_accept_cost=(
                self.policy
                .false_accept_cost
            ),
            false_reject_cost=(
                self.policy
                .false_reject_cost
            ),
        )

    def save(
        self,
        destination: str | Path = (
            DEFAULT_THRESHOLD_PATH
        ),
    ) -> Path:
        """Save the manager's current threshold configuration."""

        if self.configuration is None:
            raise ThresholdNotAvailableError(
                "No threshold configuration is available to save."
            )

        return save_threshold_configuration(
            self.configuration,
            destination,
        )

    def load(
        self,
        source: str | Path = (
            DEFAULT_THRESHOLD_PATH
        ),
        *,
        install: bool = True,
    ) -> ThresholdConfiguration:
        """Load a trusted threshold artifact."""

        self.configuration = (
            load_threshold_configuration(
                source,
                install=install,
            )
        )

        self.selection = None

        return self.configuration

    def reject(
        self,
        attack_probability: float,
    ) -> bool:
        """
        Apply the active threshold.

        Equality with the threshold causes rejection.
        """

        if self.threshold is None:
            raise ThresholdNotAvailableError(
                "No operational threshold is available."
            )

        probability = validate_probability(
            attack_probability,
            "attack_probability",
        )

        return bool(
            probability
            >= self.threshold
        )

    def accept(
        self,
        attack_probability: float,
    ) -> bool:
        """
        Return whether attack probability is below the threshold.
        """

        return not self.reject(
            attack_probability
        )


def run_self_test() -> None:
    """
    Verify threshold selection, operational floor, persistence,
    and strict rejection behavior.
    """

    import tempfile

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
            0.04,
            0.06,
            0.08,
            0.14,
            0.30,
            0.70,
            0.95,
        ],
        dtype=float,
    )

    manager = ThresholdManager()

    selection = manager.select(
        labels,
        probabilities,
        install=True,
    )

    if not math.isclose(
        selection.raw_threshold,
        0.11,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Raw calibration threshold is incorrect."
        )

    if not math.isclose(
        selection.operational_threshold,
        DEFAULT_MIN_OPERATIONAL_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Operational threshold floor was not applied."
        )

    if not selection.floor_applied:
        raise ThresholdManagerError(
            "Operational-floor status is incorrect."
        )

    expected_theoretical_threshold = (
        DEFAULT_FALSE_REJECT_COST
        / (
            DEFAULT_FALSE_ACCEPT_COST
            + DEFAULT_FALSE_REJECT_COST
        )
    )

    if not math.isclose(
        selection.theoretical_threshold,
        expected_theoretical_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Theoretical Bayesian threshold is incorrect."
        )

    if current_threshold() != 0.15:
        raise ThresholdManagerError(
            "Current threshold was not installed."
        )

    if manager.accept(0.149999):
        pass

    else:
        raise ThresholdManagerError(
            "Probability below threshold was rejected."
        )

    if not manager.reject(0.15):
        raise ThresholdManagerError(
            "Probability equal to threshold was accepted."
        )

    if not manager.reject(0.80):
        raise ThresholdManagerError(
            "High attack probability was accepted."
        )

    raw_threshold, raw_far, raw_frr = (
        select_threshold(
            labels,
            probabilities,
            DEFAULT_FALSE_ACCEPT_COST,
            DEFAULT_FALSE_REJECT_COST,
        )
    )

    if not math.isclose(
        raw_threshold,
        selection.raw_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Notebook-compatible selection differs."
        )

    if not math.isclose(
        raw_far,
        selection.raw_rates.false_accept_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Raw false-accept rate differs."
        )

    if not math.isclose(
        raw_frr,
        selection.raw_rates.false_reject_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ThresholdManagerError(
            "Raw false-reject rate differs."
        )

    with tempfile.TemporaryDirectory() as directory:
        artifact_path = (
            Path(directory)
            / "threshold.json"
        )

        manager.save(
            artifact_path
        )

        clear_current_threshold()

        if current_threshold() is not None:
            raise ThresholdManagerError(
                "Current threshold was not cleared."
            )

        loaded_manager = ThresholdManager()

        loaded_configuration = (
            loaded_manager.load(
                artifact_path,
                install=True,
            )
        )

        if not math.isclose(
            loaded_configuration.gp_attack_threshold,
            selection.operational_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ThresholdManagerError(
                "Loaded operational threshold differs."
            )

        if not math.isclose(
            require_current_threshold(),
            selection.operational_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ThresholdManagerError(
                "Loaded threshold was not installed."
            )

    print(
        "Threshold manager self-test completed successfully."
    )

    print(
        "Theoretical Bayes threshold:",
        f"{selection.theoretical_threshold:.6f}",
    )

    print(
        "Raw calibration threshold:",
        f"{selection.raw_threshold:.6f}",
    )

    print(
        "Operational threshold:",
        f"{selection.operational_threshold:.6f}",
    )

    print(
        "Operational floor applied:",
        selection.floor_applied,
    )

    print(
        "Calibration FAR:",
        f"{selection.calibration_false_accept_rate:.6f}",
    )

    print(
        "Calibration FRR:",
        f"{selection.calibration_false_reject_rate:.6f}",
    )


__all__ = [
    "DEFAULT_FALSE_ACCEPT_COST",
    "DEFAULT_FALSE_REJECT_COST",
    "DEFAULT_MIN_OPERATIONAL_THRESHOLD",
    "DEFAULT_GP_GRAY_ZONE_RETRY_UPPER",
    "THRESHOLD_ARTIFACT_VERSION",
    "PROJECT_ROOT",
    "DEFAULT_THRESHOLD_PATH",
    "ThresholdManagerError",
    "InvalidThresholdDataError",
    "InvalidThresholdPolicyError",
    "ThresholdSelectionError",
    "ThresholdArtifactError",
    "ThresholdNotAvailableError",
    "ThresholdRates",
    "ThresholdPolicy",
    "ThresholdSelection",
    "ThresholdConfiguration",
    "ThresholdManager",
    "DEFAULT_THRESHOLD_POLICY",
    "validate_probability",
    "validate_positive_cost",
    "normalize_probability_array",
    "normalize_binary_labels",
    "validate_calibration_data",
    "theoretical_bayes_threshold",
    "build_threshold_candidates",
    "evaluate_threshold",
    "select_threshold",
    "select_operational_threshold",
    "set_current_threshold",
    "clear_current_threshold",
    "current_configuration",
    "current_threshold",
    "require_current_threshold",
    "save_threshold_configuration",
    "load_threshold_configuration",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ThresholdManagerError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "\n[THRESHOLD MANAGER ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error