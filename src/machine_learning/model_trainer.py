"""
FT-QuPAP Gaussian Process Model Trainer

This module trains the final session-level Gaussian Process attack
detector used by FT-QuPAP.

Training workflow
=================

1. Validate the session-level dataset.
2. Keep training, calibration, and test partitions disjoint.
3. Select a balanced stratified subset of the training partition.
4. Train the exact GaussianProcessClassifier on that subset.
5. Fit isotonic probability calibration using calibration rows only.
6. Select the raw Bayes-risk threshold from calibration data only.
7. Apply the minimum operational GP threshold.
8. Keep the held-out test partition untouched for model_evaluator.py.
9. Export all reproducible model artifacts.

Security boundary
=================

Only the nine receiver-observable FEATURE_COLUMNS may enter the model.

Simulator-only fields such as:

- eve_fraction
- eve_mode
- scenario
- scenario_severity
- attack positions
- ground-truth labels

may exist as offline metadata, but they must never become GP input
features.

Security warning
================

Pickle and joblib artifacts can execute code while loading. Only load
artifacts created by this trusted project.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_preprocessor import (
    save_feature_scaler,
    validate_feature_frame,
)
from .feature_schema import (
    FEATURE_COLUMNS,
    FEATURE_COUNT,
    validate_feature_order,
)
from .probability_calibrator import (
    CALIBRATION_METHOD,
    apply_probability_calibrator,
    fit_probability_calibrator,
    save_probability_calibrator,
)
from .threshold_manager import (
    DEFAULT_FALSE_ACCEPT_COST,
    DEFAULT_FALSE_REJECT_COST,
    DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
    DEFAULT_MIN_OPERATIONAL_THRESHOLD,
    ThresholdConfiguration,
    ThresholdPolicy,
    ThresholdSelection,
    save_threshold_configuration,
    select_operational_threshold,
)


DEFAULT_RANDOM_STATE = 20260701
DEFAULT_MAX_EXACT_GP_TRAIN_ROWS = 2500

DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_CALIBRATION_FRACTION = 0.20
DEFAULT_TEST_FRACTION = 0.20

DEFAULT_LABEL_COLUMN = "label_attack"
DEFAULT_SPLIT_COLUMN = "split"

DEFAULT_PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
    "-large-session-calibrated"
)

DEFAULT_TRAINING_SOURCE = (
    "large_session_level_ft_qupap_simulator"
)

DEFAULT_SESSION_GP_DATA_MODE = "large"

VALID_SPLIT_NAMES = (
    "train",
    "calibration",
    "test",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

DEFAULT_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "generated"
)

DEFAULT_MODEL_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "gp_model.pkl"
)

DEFAULT_FEATURE_SCALER_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "feature_scaler.pkl"
)

DEFAULT_CALIBRATION_MODEL_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "calibration_model.pkl"
)

DEFAULT_THRESHOLD_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "threshold.json"
)

DEFAULT_FEATURE_ORDER_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "feature_order.json"
)

DEFAULT_MODEL_METADATA_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "model_metadata.json"
)

DEFAULT_COMBINED_BUNDLE_PATH = (
    DEFAULT_MODEL_DIRECTORY
    / "ft_qupap_gp_detector.joblib"
)

DEFAULT_DATASET_PATH = (
    DEFAULT_DATA_DIRECTORY
    / "session_level_gp_dataset.csv"
)

DEFAULT_EXACT_TRAINING_SUBSET_PATH = (
    DEFAULT_DATA_DIRECTORY
    / "session_level_gp_exact_train_subset.csv"
)

DEFAULT_CALIBRATION_PREDICTIONS_PATH = (
    DEFAULT_DATA_DIRECTORY
    / "session_level_gp_calibration_predictions.csv"
)


class ModelTrainerError(Exception):
    """Base exception for FT-QuPAP model-training failures."""


class InvalidTrainingDatasetError(
    ModelTrainerError
):
    """Raised when the supplied dataset is malformed."""


class InvalidTrainingSplitError(
    ModelTrainerError
):
    """Raised when train/calibration/test partitions are invalid."""


class InsufficientTrainingClassError(
    ModelTrainerError
):
    """Raised when a required split does not contain both classes."""


class GPTrainingFailureError(
    ModelTrainerError
):
    """Raised when the GP model cannot be trained."""


class TrainingArtifactError(
    ModelTrainerError
):
    """Raised when model artifacts cannot be exported."""


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

    normalized = int(
        value
    )

    if normalized < 1:
        raise ValueError(
            f"{field_name} must be at least 1."
        )

    return normalized


def validate_fraction(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate a finite fraction in the interval (0, 1).
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

    normalized = float(
        value
    )

    if not math.isfinite(
        normalized
    ):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 < normalized < 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def normalize_binary_labels(
    labels: Any,
    *,
    field_name: str = DEFAULT_LABEL_COLUMN,
    require_both_classes: bool = True,
) -> np.ndarray:
    """
    Normalize FT-QuPAP labels.

    Class definitions:

        0 = benign session
        1 = attack session
    """

    try:
        normalized = np.asarray(
            labels,
            dtype=float,
        ).reshape(
            -1
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidTrainingDatasetError(
            f"{field_name} must contain numeric labels."
        ) from error

    if normalized.size == 0:
        raise InvalidTrainingDatasetError(
            f"{field_name} cannot be empty."
        )

    if not np.all(
        np.isfinite(
            normalized
        )
    ):
        raise InvalidTrainingDatasetError(
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
        raise InvalidTrainingDatasetError(
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
        raise InsufficientTrainingClassError(
            f"{field_name} must contain benign class 0 "
            "and attack class 1."
        )

    return integer_labels


def normalize_split_name(
    value: Any,
) -> str:
    """
    Validate one dataset split name.
    """

    if not isinstance(
        value,
        str,
    ):
        raise InvalidTrainingSplitError(
            "Dataset split values must be strings."
        )

    normalized = value.strip().lower()

    if normalized not in VALID_SPLIT_NAMES:
        raise InvalidTrainingSplitError(
            f"Unsupported split {value!r}. "
            f"Expected {VALID_SPLIT_NAMES!r}."
        )

    return normalized


def json_safe(
    value: Any,
) -> Any:
    """
    Convert common project values into JSON-safe representations.
    """

    if value is None:
        return None

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        np.generic,
    ):
        return json_safe(
            value.item()
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key):
                json_safe(item)
            for key, item
            in value.items()
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

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
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

    return str(
        value
    )


@dataclass(frozen=True)
class GPTrainingConfig:
    """
    Reproducible FT-QuPAP GP training configuration.
    """

    random_state: int = DEFAULT_RANDOM_STATE

    max_exact_gp_train_rows: int = (
        DEFAULT_MAX_EXACT_GP_TRAIN_ROWS
    )

    train_fraction: float = (
        DEFAULT_TRAIN_FRACTION
    )

    calibration_fraction: float = (
        DEFAULT_CALIBRATION_FRACTION
    )

    test_fraction: float = (
        DEFAULT_TEST_FRACTION
    )

    label_column: str = (
        DEFAULT_LABEL_COLUMN
    )

    split_column: str = (
        DEFAULT_SPLIT_COLUMN
    )

    max_iter_predict: int = 100

    n_restarts_optimizer: int = 0

    # The notebook disables kernel optimization for reproducibility and
    # practical exact-GP training time.
    optimizer: str | None = None

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

    protocol_version: str = (
        DEFAULT_PROTOCOL_VERSION
    )

    training_source: str = (
        DEFAULT_TRAINING_SOURCE
    )

    session_gp_data_mode: str = (
        DEFAULT_SESSION_GP_DATA_MODE
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "random_state",
            validate_positive_integer(
                self.random_state,
                "random_state",
            ),
        )

        object.__setattr__(
            self,
            "max_exact_gp_train_rows",
            validate_positive_integer(
                self.max_exact_gp_train_rows,
                "max_exact_gp_train_rows",
            ),
        )

        train_fraction = validate_fraction(
            self.train_fraction,
            "train_fraction",
        )

        calibration_fraction = (
            validate_fraction(
                self.calibration_fraction,
                "calibration_fraction",
            )
        )

        test_fraction = validate_fraction(
            self.test_fraction,
            "test_fraction",
        )

        if not math.isclose(
            train_fraction
            + calibration_fraction
            + test_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "train_fraction, calibration_fraction, "
                "and test_fraction must sum to 1.0."
            )

        object.__setattr__(
            self,
            "train_fraction",
            train_fraction,
        )

        object.__setattr__(
            self,
            "calibration_fraction",
            calibration_fraction,
        )

        object.__setattr__(
            self,
            "test_fraction",
            test_fraction,
        )

        object.__setattr__(
            self,
            "max_iter_predict",
            validate_positive_integer(
                self.max_iter_predict,
                "max_iter_predict",
            ),
        )

        if (
            isinstance(
                self.n_restarts_optimizer,
                bool,
            )
            or not isinstance(
                self.n_restarts_optimizer,
                (
                    int,
                    np.integer,
                ),
            )
        ):
            raise TypeError(
                "n_restarts_optimizer must be an integer."
            )

        if int(
            self.n_restarts_optimizer
        ) < 0:
            raise ValueError(
                "n_restarts_optimizer cannot be negative."
            )

        object.__setattr__(
            self,
            "n_restarts_optimizer",
            int(
                self.n_restarts_optimizer
            ),
        )

        for field_name in (
            "label_column",
            "split_column",
            "protocol_version",
            "training_source",
            "session_gp_data_mode",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(
                    value,
                    str,
                )
                or not value.strip()
            ):
                raise ValueError(
                    f"{field_name} cannot be empty."
                )

            object.__setattr__(
                self,
                field_name,
                value.strip(),
            )

        threshold_policy = ThresholdPolicy(
            false_accept_cost=(
                self.false_accept_cost
            ),
            false_reject_cost=(
                self.false_reject_cost
            ),
            min_operational_threshold=(
                self.min_operational_threshold
            ),
            gp_gray_zone_retry_upper=(
                self.gp_gray_zone_retry_upper
            ),
        )

        object.__setattr__(
            self,
            "false_accept_cost",
            threshold_policy.false_accept_cost,
        )

        object.__setattr__(
            self,
            "false_reject_cost",
            threshold_policy.false_reject_cost,
        )

        object.__setattr__(
            self,
            "min_operational_threshold",
            threshold_policy.min_operational_threshold,
        )

        object.__setattr__(
            self,
            "gp_gray_zone_retry_upper",
            threshold_policy.gp_gray_zone_retry_upper,
        )

    @property
    def threshold_policy(self) -> ThresholdPolicy:
        """
        Return the configured threshold-selection policy.
        """

        return ThresholdPolicy(
            false_accept_cost=(
                self.false_accept_cost
            ),
            false_reject_cost=(
                self.false_reject_cost
            ),
            min_operational_threshold=(
                self.min_operational_threshold
            ),
            gp_gray_zone_retry_upper=(
                self.gp_gray_zone_retry_upper
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return JSON-safe training configuration.
        """

        return {
            "random_state":
                self.random_state,

            "max_exact_gp_train_rows":
                self.max_exact_gp_train_rows,

            "train_fraction":
                self.train_fraction,

            "calibration_fraction":
                self.calibration_fraction,

            "test_fraction":
                self.test_fraction,

            "label_column":
                self.label_column,

            "split_column":
                self.split_column,

            "max_iter_predict":
                self.max_iter_predict,

            "n_restarts_optimizer":
                self.n_restarts_optimizer,

            "optimizer":
                self.optimizer,

            "false_accept_cost":
                self.false_accept_cost,

            "false_reject_cost":
                self.false_reject_cost,

            "min_operational_threshold":
                self.min_operational_threshold,

            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,

            "protocol_version":
                self.protocol_version,

            "training_source":
                self.training_source,

            "session_gp_data_mode":
                self.session_gp_data_mode,
        }


@dataclass(frozen=True)
class GPTrainingSplits:
    """
    Disjoint training, calibration, and held-out test partitions.
    """

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame

    label_column: str = (
        DEFAULT_LABEL_COLUMN
    )

    split_column: str = (
        DEFAULT_SPLIT_COLUMN
    )

    def __post_init__(self) -> None:
        for split_name in VALID_SPLIT_NAMES:
            table = getattr(
                self,
                split_name,
            )

            if not isinstance(
                table,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"{split_name} must be a pandas DataFrame."
                )

            if table.empty:
                raise InvalidTrainingSplitError(
                    f"{split_name} split cannot be empty."
                )

            if self.label_column not in table.columns:
                raise InvalidTrainingSplitError(
                    f"{split_name} split is missing "
                    f"{self.label_column!r}."
                )

            normalize_binary_labels(
                table[
                    self.label_column
                ],
                field_name=(
                    f"{split_name}."
                    f"{self.label_column}"
                ),
                require_both_classes=True,
            )

            validate_feature_frame(
                table.loc[
                    :,
                    FEATURE_COLUMNS,
                ],
                allow_extra_columns=False,
                reject_hidden_columns=True,
            )

    @property
    def sizes(self) -> dict[str, int]:
        """
        Return full partition sizes.
        """

        return {
            "train_full":
                int(
                    len(
                        self.train
                    )
                ),

            "calibration":
                int(
                    len(
                        self.calibration
                    )
                ),

            "test":
                int(
                    len(
                        self.test
                    )
                ),
        }

    def combined(self) -> pd.DataFrame:
        """
        Combine all partitions with explicit split labels.
        """

        frames: list[
            pd.DataFrame
        ] = []

        for split_name in VALID_SPLIT_NAMES:
            frame = getattr(
                self,
                split_name,
            ).copy()

            frame[
                self.split_column
            ] = split_name

            frames.append(
                frame
            )

        return pd.concat(
            frames,
            ignore_index=True,
        )


@dataclass(frozen=True)
class GPTrainingResult:
    """
    Complete GP training result before held-out evaluation.
    """

    model: Pipeline
    calibrator: Any

    threshold_selection: ThresholdSelection

    full_dataset: pd.DataFrame
    splits: GPTrainingSplits

    exact_training_subset: pd.DataFrame

    calibration_predictions: pd.DataFrame

    config: GPTrainingConfig

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            Pipeline,
        ):
            raise TypeError(
                "model must be a scikit-learn Pipeline."
            )

        if not callable(
            getattr(
                self.model,
                "predict_proba",
                None,
            )
        ):
            raise TypeError(
                "model must provide predict_proba()."
            )

        if not isinstance(
            self.threshold_selection,
            ThresholdSelection,
        ):
            raise TypeError(
                "threshold_selection must be ThresholdSelection."
            )

        for field_name in (
            "full_dataset",
            "exact_training_subset",
            "calibration_predictions",
        ):
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

        if not isinstance(
            self.splits,
            GPTrainingSplits,
        ):
            raise TypeError(
                "splits must be GPTrainingSplits."
            )

        if not isinstance(
            self.config,
            GPTrainingConfig,
        ):
            raise TypeError(
                "config must be GPTrainingConfig."
            )

    @property
    def raw_calibration_threshold(
        self,
    ) -> float:
        """
        Return the raw calibration-selected threshold.
        """

        return float(
            self.threshold_selection
            .raw_threshold
        )

    @property
    def operational_threshold(
        self,
    ) -> float:
        """
        Return the final deployed threshold.
        """

        return float(
            self.threshold_selection
            .operational_threshold
        )

    @property
    def split_sizes(
        self,
    ) -> dict[str, int]:
        """
        Return notebook-compatible split sizes.
        """

        return {
            "train_full":
                int(
                    len(
                        self.splits.train
                    )
                ),

            "train_exact_gp_subset":
                int(
                    len(
                        self.exact_training_subset
                    )
                ),

            "calibration":
                int(
                    len(
                        self.splits.calibration
                    )
                ),

            "test":
                int(
                    len(
                        self.splits.test
                    )
                ),
        }

    @property
    def scaler(
        self,
    ) -> StandardScaler:
        """
        Return the fitted scaler from the model pipeline.
        """

        scaler = self.model.named_steps.get(
            "scaler"
        )

        if not isinstance(
            scaler,
            StandardScaler,
        ):
            raise ModelTrainerError(
                "The GP pipeline does not contain a fitted "
                "StandardScaler named 'scaler'."
            )

        return scaler

    def combined_bundle(
        self,
    ) -> dict[str, Any]:
        """
        Return the notebook-compatible final model bundle.
        """

        return {
            "model":
                self.model,

            "calibrator":
                self.calibrator,

            "feature_columns":
                list(
                    FEATURE_COLUMNS
                ),

            "protocol_version":
                self.config.protocol_version,

            "seed":
                self.config.random_state,

            "raw_calibration_gp_attack_threshold":
                self.raw_calibration_threshold,

            "gp_attack_threshold":
                self.operational_threshold,

            "training_source":
                self.config.training_source,

            "calibration_method":
                CALIBRATION_METHOD,

            "session_gp_data_mode":
                self.config.session_gp_data_mode,

            "session_gp_split_sizes":
                self.split_sizes,
        }

    def metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata without estimator objects.
        """

        classifier = self.model.named_steps.get(
            "gp"
        )

        fitted_kernel = getattr(
            classifier,
            "kernel_",
            None,
        )

        configured_kernel = getattr(
            classifier,
            "kernel",
            None,
        )

        kernel_value = (
            fitted_kernel
            if fitted_kernel is not None
            else configured_kernel
        )

        return {
            "protocol_version":
                self.config.protocol_version,

            "seed":
                self.config.random_state,

            "training_source":
                self.config.training_source,

            "calibration_method":
                CALIBRATION_METHOD,

            "session_gp_data_mode":
                self.config.session_gp_data_mode,

            "feature_columns":
                list(
                    FEATURE_COLUMNS
                ),

            "feature_count":
                FEATURE_COUNT,

            "session_gp_split_sizes":
                self.split_sizes,

            "full_dataset_rows":
                int(
                    len(
                        self.full_dataset
                    )
                ),

            "max_exact_gp_train_rows":
                self.config.max_exact_gp_train_rows,

            "raw_calibration_gp_attack_threshold":
                self.raw_calibration_threshold,

            "gp_attack_threshold":
                self.operational_threshold,

            "threshold_selection":
                self.threshold_selection.as_dict(),

            "model_type": (
                None
                if classifier is None
                else type(
                    classifier
                ).__name__
            ),

            "pipeline_steps": [
                name
                for name, _
                in self.model.steps
            ],

            "kernel": (
                None
                if kernel_value is None
                else str(
                    kernel_value
                )
            ),

            "training_configuration":
                self.config.as_dict(),

            # model_evaluator.py changes this after held-out testing.
            "heldout_test_evaluated":
                False,

            "created_at_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }


def validate_training_table(
    table: pd.DataFrame,
    *,
    label_column: str = (
        DEFAULT_LABEL_COLUMN
    ),
    split_column: str = (
        DEFAULT_SPLIT_COLUMN
    ),
) -> pd.DataFrame:
    """
    Validate one FT-QuPAP session-level GP dataset.

    Extra columns are retained as offline metadata. Only FEATURE_COLUMNS
    are selected when fitting or predicting with the GP.
    """

    if not isinstance(
        table,
        pd.DataFrame,
    ):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    if table.empty:
        raise InvalidTrainingDatasetError(
            "Training dataset cannot be empty."
        )

    required_columns = [
        *FEATURE_COLUMNS,
        label_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing_columns:
        raise InvalidTrainingDatasetError(
            "Training dataset is missing columns: "
            f"{missing_columns!r}"
        )

    validate_feature_order(
        FEATURE_COLUMNS
    )

    validated_features = (
        validate_feature_frame(
            table.loc[
                :,
                FEATURE_COLUMNS,
            ],
            allow_extra_columns=False,
            reject_hidden_columns=True,
        )
    )

    normalized_labels = (
        normalize_binary_labels(
            table[
                label_column
            ],
            field_name=label_column,
            require_both_classes=True,
        )
    )

    validated = table.copy()

    for column in FEATURE_COLUMNS:
        validated[
            column
        ] = validated_features[
            column
        ]

    validated[
        label_column
    ] = normalized_labels

    if split_column in validated.columns:
        validated[
            split_column
        ] = [
            normalize_split_name(
                value
            )
            for value
            in validated[
                split_column
            ]
        ]

    return validated.reset_index(
        drop=True
    )


def split_training_dataset(
    table: pd.DataFrame,
    config: GPTrainingConfig | None = None,
) -> GPTrainingSplits:
    """
    Create disjoint training, calibration, and held-out test partitions.

    Existing explicit split values are preserved when all required
    partitions are present.

    Otherwise, a reproducible stratified 60/20/20 split is created.
    """

    active_config = (
        GPTrainingConfig()
        if config is None
        else config
    )

    if not isinstance(
        active_config,
        GPTrainingConfig,
    ):
        raise TypeError(
            "config must be GPTrainingConfig or None."
        )

    validated = validate_training_table(
        table,
        label_column=(
            active_config.label_column
        ),
        split_column=(
            active_config.split_column
        ),
    )

    split_column = (
        active_config.split_column
    )

    label_column = (
        active_config.label_column
    )

    explicit_split_available = (
        split_column in validated.columns
        and set(
            validated[
                split_column
            ].unique()
        )
        == set(
            VALID_SPLIT_NAMES
        )
    )

    if explicit_split_available:
        train = validated[
            validated[
                split_column
            ]
            == "train"
        ].copy()

        calibration = validated[
            validated[
                split_column
            ]
            == "calibration"
        ].copy()

        test = validated[
            validated[
                split_column
            ]
            == "test"
        ].copy()

    else:
        indexed = validated.copy()

        indexed[
            "_ft_qupap_source_row"
        ] = np.arange(
            len(
                indexed
            ),
            dtype=int,
        )

        temporary_fraction = (
            active_config.calibration_fraction
            + active_config.test_fraction
        )

        try:
            train, temporary = train_test_split(
                indexed,
                test_size=temporary_fraction,
                stratify=indexed[
                    label_column
                ],
                random_state=(
                    active_config.random_state
                ),
            )

            test_share_of_temporary = (
                active_config.test_fraction
                / temporary_fraction
            )

            calibration, test = (
                train_test_split(
                    temporary,
                    test_size=(
                        test_share_of_temporary
                    ),
                    stratify=temporary[
                        label_column
                    ],
                    random_state=(
                        active_config.random_state
                    ),
                )
            )

        except ValueError as error:
            raise InvalidTrainingSplitError(
                "The dataset is too small or imbalanced for "
                "the requested stratified split."
            ) from error

        train_indices = set(
            train[
                "_ft_qupap_source_row"
            ].tolist()
        )

        calibration_indices = set(
            calibration[
                "_ft_qupap_source_row"
            ].tolist()
        )

        test_indices = set(
            test[
                "_ft_qupap_source_row"
            ].tolist()
        )

        if train_indices & calibration_indices:
            raise InvalidTrainingSplitError(
                "Training and calibration rows overlap."
            )

        if train_indices & test_indices:
            raise InvalidTrainingSplitError(
                "Training and test rows overlap."
            )

        if calibration_indices & test_indices:
            raise InvalidTrainingSplitError(
                "Calibration and test rows overlap."
            )

        for frame, split_name in (
            (
                train,
                "train",
            ),
            (
                calibration,
                "calibration",
            ),
            (
                test,
                "test",
            ),
        ):
            frame[
                split_column
            ] = split_name

            frame.drop(
                columns=[
                    "_ft_qupap_source_row",
                ],
                inplace=True,
            )

    return GPTrainingSplits(
        train=train.reset_index(
            drop=True
        ),
        calibration=calibration.reset_index(
            drop=True
        ),
        test=test.reset_index(
            drop=True
        ),
        label_column=label_column,
        split_column=split_column,
    )


def stratified_gp_training_subset(
    training_table: pd.DataFrame,
    max_rows: int = (
        DEFAULT_MAX_EXACT_GP_TRAIN_ROWS
    ),
    *,
    label_column: str = (
        DEFAULT_LABEL_COLUMN
    ),
    random_state: int = (
        DEFAULT_RANDOM_STATE
    ),
) -> pd.DataFrame:
    """
    Select a balanced stratified exact-GP training subset.

    Exact Gaussian Process training has cubic complexity in the number
    of training rows. The full source training partition remains
    preserved, but only this balanced subset is used to fit the GP.
    """

    if not isinstance(
        training_table,
        pd.DataFrame,
    ):
        raise TypeError(
            "training_table must be a pandas DataFrame."
        )

    max_rows = validate_positive_integer(
        max_rows,
        "max_rows",
    )

    random_state = validate_positive_integer(
        random_state,
        "random_state",
    )

    validated = validate_training_table(
        training_table,
        label_column=label_column,
    )

    if len(
        validated
    ) <= max_rows:
        return validated.copy().reset_index(
            drop=True
        )

    classes = sorted(
        validated[
            label_column
        ].unique().tolist()
    )

    if set(
        classes
    ) != {
        0,
        1,
    }:
        raise InsufficientTrainingClassError(
            "Exact GP training requires classes 0 and 1."
        )

    rows_per_class = (
        max_rows
        // len(
            classes
        )
    )

    selected_indices: list[int] = []

    for label in classes:
        class_table = validated[
            validated[
                label_column
            ]
            == label
        ]

        take_n = min(
            rows_per_class,
            len(
                class_table
            ),
        )

        sampled_indices = (
            class_table.sample(
                n=take_n,
                random_state=(
                    random_state
                    + int(
                        label
                    )
                ),
            ).index.tolist()
        )

        selected_indices.extend(
            sampled_indices
        )

    if len(
        selected_indices
    ) < max_rows:
        remaining = validated.loc[
            ~validated.index.isin(
                selected_indices
            )
        ]

        if not remaining.empty:
            extra_n = min(
                max_rows
                - len(
                    selected_indices
                ),
                len(
                    remaining
                ),
            )

            additional_indices = (
                remaining.sample(
                    n=extra_n,
                    random_state=(
                        random_state
                        + 99
                    ),
                ).index.tolist()
            )

            selected_indices.extend(
                additional_indices
            )

    subset = validated.loc[
        selected_indices
    ].copy()

    return subset.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(
        drop=True
    )


def build_session_level_gp_pipeline(
    config: GPTrainingConfig | None = None,
) -> Pipeline:
    """
    Construct the final notebook-aligned GP classifier.
    """

    active_config = (
        GPTrainingConfig()
        if config is None
        else config
    )

    if not isinstance(
        active_config,
        GPTrainingConfig,
    ):
        raise TypeError(
            "config must be GPTrainingConfig or None."
        )

    gp_kernel = (
        ConstantKernel(
            1.0,
            (
                1e-2,
                1e2,
            ),
        )
        * RBF(
            length_scale=np.ones(
                FEATURE_COUNT
            ),
            length_scale_bounds=(
                1e-2,
                1e3,
            ),
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(
                1e-6,
                1.0,
            ),
        )
    )

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "gp",
                GaussianProcessClassifier(
                    kernel=gp_kernel,
                    random_state=(
                        active_config.random_state
                    ),
                    max_iter_predict=(
                        active_config.max_iter_predict
                    ),
                    n_restarts_optimizer=(
                        active_config
                        .n_restarts_optimizer
                    ),
                    optimizer=(
                        active_config.optimizer
                    ),
                ),
            ),
        ]
    )


def fit_gp_pipeline(
    exact_training_subset: pd.DataFrame,
    config: GPTrainingConfig | None = None,
) -> Pipeline:
    """
    Fit the GP using only the exact training subset.
    """

    active_config = (
        GPTrainingConfig()
        if config is None
        else config
    )

    validated = validate_training_table(
        exact_training_subset,
        label_column=(
            active_config.label_column
        ),
        split_column=(
            active_config.split_column
        ),
    )

    model = build_session_level_gp_pipeline(
        active_config
    )

    try:
        model.fit(
            validated.loc[
                :,
                FEATURE_COLUMNS,
            ],
            validated[
                active_config.label_column
            ],
        )

    except Exception as error:
        raise GPTrainingFailureError(
            "Could not fit the session-level GP classifier."
        ) from error

    if not callable(
        getattr(
            model,
            "predict_proba",
            None,
        )
    ):
        raise GPTrainingFailureError(
            "The fitted GP pipeline does not provide "
            "predict_proba()."
        )

    return model


def raw_attack_probabilities(
    model: Pipeline,
    feature_table: pd.DataFrame,
) -> np.ndarray:
    """
    Generate raw probabilities for attack class 1.
    """

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "model must be a scikit-learn Pipeline."
        )

    features = validate_feature_frame(
        feature_table.loc[
            :,
            FEATURE_COLUMNS,
        ],
        allow_extra_columns=False,
        reject_hidden_columns=True,
    )

    try:
        probability_output = np.asarray(
            model.predict_proba(
                features
            ),
            dtype=float,
        )

    except Exception as error:
        raise GPTrainingFailureError(
            "The GP model could not generate probabilities."
        ) from error

    if (
        probability_output.ndim != 2
        or probability_output.shape[0]
        != len(
            features
        )
    ):
        raise GPTrainingFailureError(
            "The GP model returned an invalid probability shape."
        )

    if not np.all(
        np.isfinite(
            probability_output
        )
    ):
        raise GPTrainingFailureError(
            "The GP model returned NaN or infinite probabilities."
        )

    model_classes = list(
        np.asarray(
            getattr(
                model,
                "classes_",
                [
                    0,
                    1,
                ],
            )
        ).reshape(
            -1
        )
    )

    if 1 not in model_classes:
        raise GPTrainingFailureError(
            "The GP model does not contain attack class 1."
        )

    attack_class_index = (
        model_classes.index(
            1
        )
    )

    if (
        attack_class_index
        >= probability_output.shape[1]
    ):
        raise GPTrainingFailureError(
            "Model classes do not match predict_proba output."
        )

    return np.clip(
        probability_output[
            :,
            attack_class_index,
        ],
        0.0,
        1.0,
    )


def fit_calibration_and_threshold(
    model: Pipeline,
    calibration_table: pd.DataFrame,
    config: GPTrainingConfig | None = None,
) -> tuple[
    Any,
    ThresholdSelection,
    pd.DataFrame,
]:
    """
    Fit isotonic calibration and select the operational threshold.

    This function never accesses held-out test rows.
    """

    active_config = (
        GPTrainingConfig()
        if config is None
        else config
    )

    validated = validate_training_table(
        calibration_table,
        label_column=(
            active_config.label_column
        ),
        split_column=(
            active_config.split_column
        ),
    )

    labels = normalize_binary_labels(
        validated[
            active_config.label_column
        ],
        field_name=(
            "calibration."
            f"{active_config.label_column}"
        ),
        require_both_classes=True,
    )

    raw_probabilities = (
        raw_attack_probabilities(
            model,
            validated,
        )
    )

    calibrator = fit_probability_calibrator(
        raw_probabilities,
        labels,
    )

    calibrated_probabilities = (
        apply_probability_calibrator(
            calibrator,
            raw_probabilities,
        )
    )

    threshold_selection = (
        select_operational_threshold(
            labels=labels,
            probabilities=(
                calibrated_probabilities
            ),
            policy=(
                active_config
                .threshold_policy
            ),
        )
    )

    prediction_table = validated.copy()

    prediction_table[
        "raw_attack_probability"
    ] = raw_probabilities

    prediction_table[
        "p_attack"
    ] = calibrated_probabilities

    prediction_table[
        "rejected"
    ] = (
        calibrated_probabilities
        >= threshold_selection
        .operational_threshold
    )

    return (
        calibrator,
        threshold_selection,
        prediction_table,
    )


def train_from_table(
    table: pd.DataFrame,
    *,
    config: GPTrainingConfig | None = None,
) -> GPTrainingResult:
    """
    Execute final FT-QuPAP GP training and calibration.

    The test partition is preserved but is not evaluated here.
    """

    active_config = (
        GPTrainingConfig()
        if config is None
        else config
    )

    if not isinstance(
        active_config,
        GPTrainingConfig,
    ):
        raise TypeError(
            "config must be GPTrainingConfig or None."
        )

    validated = validate_training_table(
        table,
        label_column=(
            active_config.label_column
        ),
        split_column=(
            active_config.split_column
        ),
    )

    splits = split_training_dataset(
        validated,
        active_config,
    )

    exact_training_subset = (
        stratified_gp_training_subset(
            splits.train,
            max_rows=(
                active_config
                .max_exact_gp_train_rows
            ),
            label_column=(
                active_config
                .label_column
            ),
            random_state=(
                active_config
                .random_state
            ),
        )
    )

    model = fit_gp_pipeline(
        exact_training_subset,
        active_config,
    )

    (
        calibrator,
        threshold_selection,
        calibration_predictions,
    ) = fit_calibration_and_threshold(
        model,
        splits.calibration,
        active_config,
    )

    return GPTrainingResult(
        model=model,
        calibrator=calibrator,
        threshold_selection=(
            threshold_selection
        ),
        full_dataset=validated,
        splits=splits,
        exact_training_subset=(
            exact_training_subset
        ),
        calibration_predictions=(
            calibration_predictions
        ),
        config=active_config,
    )


def train_from_csv(
    source: str | Path,
    *,
    config: GPTrainingConfig | None = None,
) -> GPTrainingResult:
    """
    Load a CSV dataset and execute the training workflow.
    """

    path = Path(
        source
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset does not exist: {path}"
        )

    if not path.is_file():
        raise InvalidTrainingDatasetError(
            f"Training dataset path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise InvalidTrainingDatasetError(
            "Training dataset path must end with .csv."
        )

    try:
        table = pd.read_csv(
            path
        )

    except Exception as error:
        raise InvalidTrainingDatasetError(
            f"Could not read training dataset: {path}"
        ) from error

    return train_from_table(
        table,
        config=config,
    )


def write_json_artifact(
    path: str | Path,
    value: Any,
) -> Path:
    """
    Write one JSON-safe project artifact.
    """

    destination = Path(
        path
    )

    if destination.suffix.lower() != ".json":
        raise TrainingArtifactError(
            "JSON artifact path must end with .json."
        )

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
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
        raise TrainingArtifactError(
            f"Could not write JSON artifact: {destination}"
        ) from error

    return destination


def export_training_artifacts(
    result: GPTrainingResult,
    *,
    model_directory: str | Path = (
        DEFAULT_MODEL_DIRECTORY
    ),
    data_directory: str | Path = (
        DEFAULT_DATA_DIRECTORY
    ),
) -> dict[str, Path]:
    """
    Export all final GP training artifacts.

    Held-out metrics are intentionally not exported here because
    model_evaluator.py owns the held-out evaluation stage.
    """

    if not isinstance(
        result,
        GPTrainingResult,
    ):
        raise TypeError(
            "result must be GPTrainingResult."
        )

    model_directory = Path(
        model_directory
    )

    data_directory = Path(
        data_directory
    )

    try:
        model_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError as error:
        raise TrainingArtifactError(
            "Could not create model or data directories."
        ) from error

    paths = {
        "model":
            model_directory
            / "gp_model.pkl",

        "feature_scaler":
            model_directory
            / "feature_scaler.pkl",

        "calibrator":
            model_directory
            / "calibration_model.pkl",

        "threshold":
            model_directory
            / "threshold.json",

        "feature_order":
            model_directory
            / "feature_order.json",

        "metadata":
            model_directory
            / "model_metadata.json",

        "combined_bundle":
            model_directory
            / "ft_qupap_gp_detector.joblib",

        "dataset":
            data_directory
            / "session_level_gp_dataset.csv",

        "exact_training_subset":
            data_directory
            / "session_level_gp_exact_train_subset.csv",

        "calibration_predictions":
            data_directory
            / "session_level_gp_calibration_predictions.csv",
    }

    try:
        joblib.dump(
            result.model,
            paths[
                "model"
            ],
        )

        save_feature_scaler(
            result.scaler,
            paths[
                "feature_scaler"
            ],
        )

        save_probability_calibrator(
            result.calibrator,
            paths[
                "calibrator"
            ],
        )

        threshold_configuration = (
            ThresholdConfiguration
            .from_selection(
                result.threshold_selection
            )
        )

        save_threshold_configuration(
            threshold_configuration,
            paths[
                "threshold"
            ],
        )

        write_json_artifact(
            paths[
                "feature_order"
            ],
            {
                "feature_columns":
                    list(
                        FEATURE_COLUMNS
                    ),

                "feature_count":
                    FEATURE_COUNT,
            },
        )

        write_json_artifact(
            paths[
                "metadata"
            ],
            result.metadata(),
        )

        joblib.dump(
            result.combined_bundle(),
            paths[
                "combined_bundle"
            ],
        )

        result.full_dataset.to_csv(
            paths[
                "dataset"
            ],
            index=False,
        )

        result.exact_training_subset.to_csv(
            paths[
                "exact_training_subset"
            ],
            index=False,
        )

        result.calibration_predictions.to_csv(
            paths[
                "calibration_predictions"
            ],
            index=False,
        )

    except ModelTrainerError:
        raise

    except Exception as error:
        raise TrainingArtifactError(
            "Could not export one or more training artifacts."
        ) from error

    return {
        name:
            path.resolve()
        for name, path
        in paths.items()
    }


class FTQuPAPModelTrainer:
    """
    Reusable FT-QuPAP GP training service.
    """

    def __init__(
        self,
        config: GPTrainingConfig | None = None,
    ) -> None:
        self.config = (
            GPTrainingConfig()
            if config is None
            else config
        )

        if not isinstance(
            self.config,
            GPTrainingConfig,
        ):
            raise TypeError(
                "config must be GPTrainingConfig or None."
            )

        self.result: (
            GPTrainingResult | None
        ) = None

    @property
    def trained(
        self,
    ) -> bool:
        """
        Return whether model training has completed.
        """

        return self.result is not None

    def fit(
        self,
        table: pd.DataFrame,
    ) -> GPTrainingResult:
        """
        Train from an in-memory session table.
        """

        self.result = train_from_table(
            table,
            config=self.config,
        )

        return self.result

    def fit_csv(
        self,
        source: str | Path,
    ) -> GPTrainingResult:
        """
        Train from a CSV dataset.
        """

        self.result = train_from_csv(
            source,
            config=self.config,
        )

        return self.result

    def export(
        self,
        *,
        model_directory: str | Path = (
            DEFAULT_MODEL_DIRECTORY
        ),
        data_directory: str | Path = (
            DEFAULT_DATA_DIRECTORY
        ),
    ) -> dict[str, Path]:
        """
        Export artifacts from the latest training result.
        """

        if self.result is None:
            raise ModelTrainerError(
                "No trained model is available to export."
            )

        return export_training_artifacts(
            self.result,
            model_directory=(
                model_directory
            ),
            data_directory=(
                data_directory
            ),
        )


def build_self_test_dataset(
    random_state: int = 7,
) -> pd.DataFrame:
    """
    Build a small reproducible session-level dataset.
    """

    rng = np.random.default_rng(
        random_state
    )

    split_sizes = {
        "train":
            40,

        "calibration":
            16,

        "test":
            16,
    }

    contexts = (
        "urban",
        "suburban",
        "rural",
    )

    rows: list[
        dict[str, Any]
    ] = []

    for split_name, split_size in split_sizes.items():
        for row_index in range(
            split_size
        ):
            label = row_index % 2

            if label == 0:
                qber_raw = float(
                    rng.uniform(
                        0.0,
                        0.07,
                    )
                )

                mean_syndrome_weight = float(
                    rng.uniform(
                        0.0,
                        0.8,
                    )
                )

                max_syndrome_weight = float(
                    rng.integers(
                        0,
                        3,
                    )
                )

                correction_failure_rate = float(
                    rng.uniform(
                        0.0,
                        0.04,
                    )
                )

                loss_rate = float(
                    rng.uniform(
                        0.0,
                        0.04,
                    )
                )

                noise_estimate = float(
                    rng.uniform(
                        0.0,
                        0.05,
                    )
                )

            else:
                qber_raw = float(
                    rng.uniform(
                        0.16,
                        0.45,
                    )
                )

                mean_syndrome_weight = float(
                    rng.uniform(
                        1.5,
                        4.5,
                    )
                )

                max_syndrome_weight = float(
                    rng.integers(
                        4,
                        7,
                    )
                )

                correction_failure_rate = float(
                    rng.uniform(
                        0.15,
                        0.60,
                    )
                )

                loss_rate = float(
                    rng.uniform(
                        0.02,
                        0.12,
                    )
                )

                noise_estimate = float(
                    rng.uniform(
                        0.08,
                        0.30,
                    )
                )

            context = contexts[
                (
                    row_index
                    + label
                )
                % len(
                    contexts
                )
            ]

            rows.append(
                {
                    "qber_raw":
                        qber_raw,

                    "mean_syndrome_weight":
                        mean_syndrome_weight,

                    "max_syndrome_weight":
                        max_syndrome_weight,

                    "correction_failure_rate":
                        correction_failure_rate,

                    "loss_rate":
                        loss_rate,

                    "noise_estimate":
                        noise_estimate,

                    "ctx_urban":
                        float(
                            context
                            == "urban"
                        ),

                    "ctx_suburban":
                        float(
                            context
                            == "suburban"
                        ),

                    "ctx_rural":
                        float(
                            context
                            == "rural"
                        ),

                    "label_attack":
                        label,

                    "split":
                        split_name,

                    "seed":
                        random_state
                        + len(
                            rows
                        ),

                    "scenario": (
                        "benign_self_test"
                        if label == 0
                        else "attack_self_test"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def run_self_test() -> None:
    """
    Verify training, calibration, thresholding, and artifact export.
    """

    import tempfile

    dataset = build_self_test_dataset()

    config = GPTrainingConfig(
        random_state=7,
        max_exact_gp_train_rows=30,
        protocol_version=(
            "FT-QuPAP-trainer-self-test"
        ),
        training_source=(
            "self_test_dataset"
        ),
        session_gp_data_mode=(
            "self_test"
        ),
    )

    trainer = FTQuPAPModelTrainer(
        config
    )

    result = trainer.fit(
        dataset
    )

    if not trainer.trained:
        raise ModelTrainerError(
            "Trainer was not marked as trained."
        )

    if len(
        result.exact_training_subset
    ) != 30:
        raise ModelTrainerError(
            "Exact GP training subset size is incorrect."
        )

    class_counts = (
        result.exact_training_subset[
            config.label_column
        ]
        .value_counts()
        .to_dict()
    )

    if class_counts != {
        0:
            15,

        1:
            15,
    }:
        raise ModelTrainerError(
            "Exact GP training subset is not balanced."
        )

    expected_split_sizes = {
        "train_full":
            40,

        "train_exact_gp_subset":
            30,

        "calibration":
            16,

        "test":
            16,
    }

    if (
        result.split_sizes
        != expected_split_sizes
    ):
        raise ModelTrainerError(
            "Training split sizes are incorrect."
        )

    if not (
        0.0
        <= result.raw_calibration_threshold
        <= 1.0
    ):
        raise ModelTrainerError(
            "Raw calibration threshold is invalid."
        )

    if (
        result.operational_threshold
        < config.min_operational_threshold
    ):
        raise ModelTrainerError(
            "Operational threshold floor was not applied."
        )

    calibrated_probabilities = (
        result.calibration_predictions[
            "p_attack"
        ]
    )

    if (
        calibrated_probabilities.min()
        < 0.0
        or calibrated_probabilities.max()
        > 1.0
    ):
        raise ModelTrainerError(
            "Calibrated probabilities are outside [0, 1]."
        )

    # The held-out test partition must remain untouched by the trainer.
    forbidden_test_columns = {
        "raw_attack_probability",
        "p_attack",
        "rejected",
    }

    if forbidden_test_columns.intersection(
        result.splits.test.columns
    ):
        raise ModelTrainerError(
            "The trainer evaluated the held-out test split."
        )

    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(
            directory
        )

        artifact_paths = trainer.export(
            model_directory=(
                temporary_root
                / "models"
            ),
            data_directory=(
                temporary_root
                / "data"
            ),
        )

        for path in artifact_paths.values():
            if not path.exists():
                raise ModelTrainerError(
                    f"Expected artifact is missing: {path}"
                )

        loaded_bundle = joblib.load(
            artifact_paths[
                "combined_bundle"
            ]
        )

        if (
            loaded_bundle[
                "feature_columns"
            ]
            != FEATURE_COLUMNS
        ):
            raise ModelTrainerError(
                "Exported GP feature order is incorrect."
            )

        if not math.isclose(
            loaded_bundle[
                "gp_attack_threshold"
            ],
            result.operational_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ModelTrainerError(
                "Exported operational threshold differs."
            )

    print(
        "Model trainer self-test completed successfully."
    )

    print(
        "Full dataset rows:",
        len(
            result.full_dataset
        ),
    )

    print(
        "Exact GP training rows:",
        len(
            result.exact_training_subset
        ),
    )

    print(
        "Calibration rows:",
        len(
            result.splits.calibration
        ),
    )

    print(
        "Held-out test rows:",
        len(
            result.splits.test
        ),
    )

    print(
        "Raw calibration threshold:",
        f"{result.raw_calibration_threshold:.6f}",
    )

    print(
        "Operational threshold:",
        f"{result.operational_threshold:.6f}",
    )

    print(
        "Held-out test evaluated:",
        False,
    )


__all__ = [
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_MAX_EXACT_GP_TRAIN_ROWS",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_CALIBRATION_FRACTION",
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_LABEL_COLUMN",
    "DEFAULT_SPLIT_COLUMN",
    "DEFAULT_PROTOCOL_VERSION",
    "DEFAULT_TRAINING_SOURCE",
    "DEFAULT_SESSION_GP_DATA_MODE",
    "VALID_SPLIT_NAMES",
    "PROJECT_ROOT",
    "DEFAULT_MODEL_DIRECTORY",
    "DEFAULT_DATA_DIRECTORY",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_FEATURE_SCALER_PATH",
    "DEFAULT_CALIBRATION_MODEL_PATH",
    "DEFAULT_THRESHOLD_PATH",
    "DEFAULT_FEATURE_ORDER_PATH",
    "DEFAULT_MODEL_METADATA_PATH",
    "DEFAULT_COMBINED_BUNDLE_PATH",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_EXACT_TRAINING_SUBSET_PATH",
    "DEFAULT_CALIBRATION_PREDICTIONS_PATH",
    "ModelTrainerError",
    "InvalidTrainingDatasetError",
    "InvalidTrainingSplitError",
    "InsufficientTrainingClassError",
    "GPTrainingFailureError",
    "TrainingArtifactError",
    "GPTrainingConfig",
    "GPTrainingSplits",
    "GPTrainingResult",
    "FTQuPAPModelTrainer",
    "validate_positive_integer",
    "validate_fraction",
    "normalize_binary_labels",
    "normalize_split_name",
    "json_safe",
    "validate_training_table",
    "split_training_dataset",
    "stratified_gp_training_subset",
    "build_session_level_gp_pipeline",
    "fit_gp_pipeline",
    "raw_attack_probabilities",
    "fit_calibration_and_threshold",
    "train_from_table",
    "train_from_csv",
    "write_json_artifact",
    "export_training_artifacts",
    "build_self_test_dataset",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ModelTrainerError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "\n[MODEL TRAINER ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error