"""
FT-QuPAP GP Feature Preprocessor

This module converts receiver-observable FT-QuPAP feature dictionaries
and tables into the exact format required by the Gaussian Process
attack detector.

Responsibilities:

- preserve the exact trained feature order
- validate receiver-observable feature values
- create pandas DataFrames for model prediction
- fit StandardScaler using training data only
- transform validation, test, and live-session features
- save and load the standalone feature scaler
- prevent labels and simulator-only values from becoming model inputs

Important:

The final Gaussian Process model may already be stored as a scikit-learn
Pipeline containing its own StandardScaler. In that case, pass the raw
ordered feature frame directly to the pipeline and do not scale it twice.

A standalone scaler is supported because the project structure contains:

    models/feature_scaler.pkl
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_schema import (
    CONTEXT_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_COUNT,
    HIDDEN_SIMULATOR_FIELDS,
    PROBABILITY_FEATURE_COLUMNS,
    validate_feature_mapping,
    validate_feature_order,
)


SCALER_ARTIFACT_VERSION = "FT-QuPAP-FeatureScaler-1.0"


class FeaturePreprocessorError(Exception):
    """Base exception for feature-preprocessing failures."""


class InvalidFeatureTableError(
    FeaturePreprocessorError
):
    """Raised when a feature table is malformed."""


class FeatureScalerNotFittedError(
    FeaturePreprocessorError
):
    """Raised when transformation is attempted before fitting."""


class FeatureScalerArtifactError(
    FeaturePreprocessorError
):
    """Raised when a saved scaler artifact is invalid."""


class FeatureScalingError(
    FeaturePreprocessorError
):
    """Raised when StandardScaler cannot process the features."""


def _is_sequence_like(
    value: Any,
) -> bool:
    """Return whether a value is a supported non-string sequence."""

    return isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    )


def _validate_finite_frame(
    frame: pd.DataFrame,
) -> None:
    """Ensure every selected feature value is finite."""

    matrix = frame.to_numpy(
        dtype=float,
    )

    if matrix.ndim != 2:
        raise InvalidFeatureTableError(
            "Feature data must be two-dimensional."
        )

    if matrix.shape[1] != FEATURE_COUNT:
        raise InvalidFeatureTableError(
            f"Feature data must contain {FEATURE_COUNT} columns."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise InvalidFeatureTableError(
            "Feature data cannot contain NaN or infinity."
        )


def _validate_context_columns(
    frame: pd.DataFrame,
) -> None:
    """
    Validate one-hot context columns for every feature row.
    """

    context_matrix = frame.loc[
        :,
        list(CONTEXT_FEATURE_COLUMNS),
    ].to_numpy(
        dtype=float
    )

    valid_binary_values = np.logical_or(
        context_matrix == 0.0,
        context_matrix == 1.0,
    )

    if not np.all(
        valid_binary_values
    ):
        raise InvalidFeatureTableError(
            "Context features must contain only 0.0 or 1.0."
        )

    active_context_counts = np.sum(
        context_matrix,
        axis=1,
    )

    if not np.all(
        active_context_counts == 1.0
    ):
        raise InvalidFeatureTableError(
            "Every feature row must have exactly one active context."
        )


def _validate_probability_columns(
    frame: pd.DataFrame,
) -> None:
    """
    Validate bounded probability/rate feature columns.
    """

    for column in PROBABILITY_FEATURE_COLUMNS:
        values = frame[
            column
        ].to_numpy(
            dtype=float
        )

        if np.any(
            values < 0.0
        ) or np.any(
            values > 1.0
        ):
            raise InvalidFeatureTableError(
                f"{column} must be between 0 and 1."
            )


def _find_hidden_columns(
    columns: Sequence[Any],
) -> list[str]:
    """Return simulator-only columns found in a table."""

    hidden: list[str] = []

    for column in columns:
        normalized = str(
            column
        ).strip().lower()

        if normalized in HIDDEN_SIMULATOR_FIELDS:
            hidden.append(
                str(column)
            )

    return hidden


def make_feature_frame(
    features: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Create one ordered GP feature row.

    This is the primary live-session preprocessing function.

    Args:
        features:
            Complete receiver-observable feature mapping.

    Returns:
        A one-row pandas DataFrame whose columns exactly match
        FEATURE_COLUMNS.
    """

    if not isinstance(
        features,
        Mapping,
    ):
        raise TypeError(
            "features must be a mapping."
        )

    normalized = validate_feature_mapping(
        features,
        require_exact_schema=True,
        reject_hidden_fields=True,
    )

    return pd.DataFrame(
        [
            [
                normalized[column]
                for column in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
        dtype=float,
    )


def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    allow_extra_columns: bool = False,
    reject_hidden_columns: bool = True,
) -> pd.DataFrame:
    """
    Validate and reorder a complete GP feature table.

    Extra columns can be allowed for offline training tables containing
    fields such as label_attack, scenario, seed, or split. Only the
    nine FEATURE_COLUMNS are returned.

    Hidden simulator columns may coexist as offline dataset metadata,
    but they are never included in the returned model input.
    """

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "frame must be a pandas DataFrame."
        )

    if frame.empty:
        raise InvalidFeatureTableError(
            "Feature table cannot be empty."
        )

    supplied_columns = [
        str(column)
        for column in frame.columns
    ]

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in supplied_columns
    ]

    if missing_columns:
        raise InvalidFeatureTableError(
            f"Missing feature columns: {missing_columns}"
        )

    extra_columns = [
        column
        for column in supplied_columns
        if column not in FEATURE_COLUMNS
    ]

    if (
        extra_columns
        and not allow_extra_columns
    ):
        raise InvalidFeatureTableError(
            f"Unexpected feature columns: {extra_columns}"
        )

    if reject_hidden_columns and not allow_extra_columns:
        hidden_columns = _find_hidden_columns(
            supplied_columns
        )

        if hidden_columns:
            raise InvalidFeatureTableError(
                "Simulator-only columns cannot be used as live "
                f"GP input: {hidden_columns}"
            )

    ordered = frame.loc[
        :,
        FEATURE_COLUMNS,
    ].copy()

    for column in FEATURE_COLUMNS:
        try:
            ordered[column] = pd.to_numeric(
                ordered[column],
                errors="raise",
            ).astype(float)

        except (
            TypeError,
            ValueError,
        ) as error:
            raise InvalidFeatureTableError(
                f"Feature column {column!r} must be numeric."
            ) from error

    _validate_finite_frame(
        ordered
    )

    _validate_probability_columns(
        ordered
    )

    if np.any(
        ordered["mean_syndrome_weight"].to_numpy(
            dtype=float
        )
        < 0.0
    ):
        raise InvalidFeatureTableError(
            "mean_syndrome_weight cannot be negative."
        )

    if np.any(
        ordered["max_syndrome_weight"].to_numpy(
            dtype=float
        )
        < 0.0
    ):
        raise InvalidFeatureTableError(
            "max_syndrome_weight cannot be negative."
        )

    _validate_context_columns(
        ordered
    )

    return ordered


def make_feature_table(
    data: (
        Mapping[str, Any]
        | Sequence[Mapping[str, Any]]
        | Sequence[Sequence[float]]
        | np.ndarray
        | pd.DataFrame
    ),
    *,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Convert supported feature data into an ordered DataFrame.

    Supported inputs:

    - one feature mapping
    - sequence of feature mappings
    - two-dimensional numeric sequence
    - NumPy matrix
    - pandas DataFrame
    """

    if isinstance(
        data,
        pd.DataFrame,
    ):
        return validate_feature_frame(
            data,
            allow_extra_columns=allow_extra_columns,
            reject_hidden_columns=(
                not allow_extra_columns
            ),
        )

    if isinstance(
        data,
        Mapping,
    ):
        return make_feature_frame(
            data
        )

    if isinstance(
        data,
        np.ndarray,
    ):
        matrix = np.asarray(
            data,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(
                1,
                -1,
            )

        if matrix.ndim != 2:
            raise InvalidFeatureTableError(
                "NumPy feature input must be one- or "
                "two-dimensional."
            )

        if matrix.shape[1] != FEATURE_COUNT:
            raise InvalidFeatureTableError(
                f"NumPy feature input must contain "
                f"{FEATURE_COUNT} columns."
            )

        return validate_feature_frame(
            pd.DataFrame(
                matrix,
                columns=FEATURE_COLUMNS,
            ),
            allow_extra_columns=False,
        )

    if not _is_sequence_like(
        data
    ):
        raise TypeError(
            "Unsupported feature-data type."
        )

    if len(data) == 0:
        raise InvalidFeatureTableError(
            "Feature sequence cannot be empty."
        )

    first_item = data[0]

    if isinstance(
        first_item,
        Mapping,
    ):
        rows: list[dict[str, float]] = []

        for index, row in enumerate(
            data
        ):
            if not isinstance(
                row,
                Mapping,
            ):
                raise InvalidFeatureTableError(
                    "All feature rows must use the same format."
                )

            try:
                normalized = validate_feature_mapping(
                    row,
                    require_exact_schema=True,
                    reject_hidden_fields=True,
                )

            except (
                TypeError,
                ValueError,
            ) as error:
                raise InvalidFeatureTableError(
                    f"Invalid feature mapping at row {index}."
                ) from error

            rows.append(
                normalized
            )

        return validate_feature_frame(
            pd.DataFrame(
                rows,
                columns=FEATURE_COLUMNS,
            ),
            allow_extra_columns=False,
        )

    try:
        matrix = np.asarray(
            data,
            dtype=float,
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise InvalidFeatureTableError(
            "Numeric feature sequence could not be converted."
        ) from error

    if matrix.ndim == 1:
        matrix = matrix.reshape(
            1,
            -1,
        )

    if matrix.ndim != 2:
        raise InvalidFeatureTableError(
            "Numeric feature data must be two-dimensional."
        )

    if matrix.shape[1] != FEATURE_COUNT:
        raise InvalidFeatureTableError(
            f"Numeric feature data must contain "
            f"{FEATURE_COUNT} columns."
        )

    return validate_feature_frame(
        pd.DataFrame(
            matrix,
            columns=FEATURE_COLUMNS,
        ),
        allow_extra_columns=False,
    )


def feature_matrix(
    data: Any,
    *,
    allow_extra_columns: bool = True,
) -> np.ndarray:
    """
    Return an ordered floating-point NumPy feature matrix.
    """

    frame = make_feature_table(
        data,
        allow_extra_columns=allow_extra_columns,
    )

    return frame.to_numpy(
        dtype=float,
        copy=True,
    )


def is_scaler_fitted(
    scaler: StandardScaler,
) -> bool:
    """Return whether a StandardScaler contains fitted parameters."""

    if not isinstance(
        scaler,
        StandardScaler,
    ):
        return False

    return bool(
        hasattr(
            scaler,
            "mean_",
        )
        and hasattr(
            scaler,
            "scale_",
        )
        and hasattr(
            scaler,
            "n_features_in_",
        )
    )


def validate_scaler(
    scaler: StandardScaler,
    *,
    require_fitted: bool = True,
) -> StandardScaler:
    """
    Validate a StandardScaler for the FT-QuPAP feature schema.
    """

    if not isinstance(
        scaler,
        StandardScaler,
    ):
        raise TypeError(
            "scaler must be sklearn.preprocessing.StandardScaler."
        )

    fitted = is_scaler_fitted(
        scaler
    )

    if require_fitted and not fitted:
        raise FeatureScalerNotFittedError(
            "Feature scaler has not been fitted."
        )

    if fitted:
        feature_count = int(
            scaler.n_features_in_
        )

        if feature_count != FEATURE_COUNT:
            raise FeatureScalerArtifactError(
                "Feature scaler expects "
                f"{feature_count} features instead of "
                f"{FEATURE_COUNT}."
            )

        feature_names = getattr(
            scaler,
            "feature_names_in_",
            None,
        )

        if feature_names is not None:
            validate_feature_order(
                [
                    str(name)
                    for name in feature_names
                ]
            )

    return scaler


def fit_feature_scaler(
    training_features: Any,
) -> StandardScaler:
    """
    Fit StandardScaler using the training split only.

    Validation, calibration, test, and live-session data must never be
    used to fit this scaler.
    """

    training_frame = make_feature_table(
        training_features,
        allow_extra_columns=True,
    )

    scaler = StandardScaler()

    try:
        scaler.fit(
            training_frame
        )

    except Exception as error:
        raise FeatureScalingError(
            "Could not fit the feature scaler."
        ) from error

    return validate_scaler(
        scaler,
        require_fitted=True,
    )


def transform_feature_table(
    features: Any,
    scaler: StandardScaler,
    *,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Transform features using an already fitted scaler.
    """

    scaler = validate_scaler(
        scaler,
        require_fitted=True,
    )

    frame = make_feature_table(
        features,
        allow_extra_columns=allow_extra_columns,
    )

    try:
        scaled_matrix = scaler.transform(
            frame
        )

    except Exception as error:
        raise FeatureScalingError(
            "Could not transform feature data."
        ) from error

    return pd.DataFrame(
        scaled_matrix,
        columns=FEATURE_COLUMNS,
        index=frame.index,
        dtype=float,
    )


def fit_transform_feature_table(
    training_features: Any,
) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Fit a scaler and transform the same training feature table.
    """

    training_frame = make_feature_table(
        training_features,
        allow_extra_columns=True,
    )

    scaler = fit_feature_scaler(
        training_frame
    )

    transformed = transform_feature_table(
        training_frame,
        scaler,
        allow_extra_columns=False,
    )

    return (
        scaler,
        transformed,
    )


def inverse_transform_feature_table(
    scaled_features: Any,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Convert scaled features back to their original feature space.
    """

    scaler = validate_scaler(
        scaler,
        require_fitted=True,
    )

    if isinstance(
        scaled_features,
        pd.DataFrame,
    ):
        validate_feature_order(
            [
                str(column)
                for column in scaled_features.columns
            ]
        )

        matrix = scaled_features.to_numpy(
            dtype=float
        )

        index = scaled_features.index

    else:
        matrix = np.asarray(
            scaled_features,
            dtype=float,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(
                1,
                -1,
            )

        if (
            matrix.ndim != 2
            or matrix.shape[1] != FEATURE_COUNT
        ):
            raise InvalidFeatureTableError(
                "Scaled features must have shape "
                f"(n_samples, {FEATURE_COUNT})."
            )

        index = None

    if not np.all(
        np.isfinite(matrix)
    ):
        raise InvalidFeatureTableError(
            "Scaled features cannot contain NaN or infinity."
        )

    try:
        original_matrix = scaler.inverse_transform(
            matrix
        )

    except Exception as error:
        raise FeatureScalingError(
            "Could not inverse-transform feature data."
        ) from error

    return pd.DataFrame(
        original_matrix,
        columns=FEATURE_COLUMNS,
        index=index,
        dtype=float,
    )


def save_feature_scaler(
    scaler: StandardScaler,
    destination: str | Path,
) -> Path:
    """
    Save a trusted FT-QuPAP feature-scaler artifact.

    Serialized model artifacts must only be loaded from trusted project
    output because pickle/joblib files can execute code while loading.
    """

    scaler = validate_scaler(
        scaler,
        require_fitted=True,
    )

    path = Path(
        destination
    )

    if path.suffix.lower() not in {
        ".pkl",
        ".joblib",
    }:
        raise FeatureScalerArtifactError(
            "Scaler artifact path must end with .pkl or .joblib."
        )

    artifact = {
        "artifact_version":
            SCALER_ARTIFACT_VERSION,
        "feature_columns":
            list(FEATURE_COLUMNS),
        "feature_count":
            FEATURE_COUNT,
        "scaler":
            scaler,
    }

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            artifact,
            path,
        )

    except Exception as error:
        raise FeatureScalerArtifactError(
            f"Could not save feature scaler to {path}."
        ) from error

    return path


def load_feature_scaler(
    source: str | Path,
) -> StandardScaler:
    """
    Load and validate a trusted feature-scaler artifact.

    Both formats are accepted:

    1. The FT-QuPAP metadata bundle produced by save_feature_scaler().
    2. A directly serialized StandardScaler object.
    """

    path = Path(
        source
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature scaler does not exist: {path}"
        )

    if not path.is_file():
        raise FeatureScalerArtifactError(
            f"Feature scaler path is not a file: {path}"
        )

    try:
        loaded = joblib.load(
            path
        )

    except Exception as error:
        raise FeatureScalerArtifactError(
            f"Could not load feature scaler from {path}."
        ) from error

    if isinstance(
        loaded,
        StandardScaler,
    ):
        scaler = loaded

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
            != SCALER_ARTIFACT_VERSION
        ):
            raise FeatureScalerArtifactError(
                "Unsupported feature-scaler artifact version: "
                f"{artifact_version!r}"
            )

        feature_columns = loaded.get(
            "feature_columns"
        )

        if feature_columns is not None:
            validate_feature_order(
                feature_columns
            )

        scaler = loaded.get(
            "scaler"
        )

    else:
        raise FeatureScalerArtifactError(
            "Unsupported feature-scaler artifact type."
        )

    return validate_scaler(
        scaler,
        require_fitted=True,
    )


@dataclass
class FeaturePreprocessor:
    """
    Reusable FT-QuPAP preprocessing service.
    """

    scaler: StandardScaler = field(
        default_factory=StandardScaler
    )

    def __post_init__(self) -> None:
        validate_scaler(
            self.scaler,
            require_fitted=False,
        )

    @property
    def fitted(self) -> bool:
        """Return whether the scaler has training statistics."""

        return is_scaler_fitted(
            self.scaler
        )

    def make_frame(
        self,
        features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Create one raw, ordered prediction row."""

        return make_feature_frame(
            features
        )

    def make_table(
        self,
        data: Any,
        *,
        allow_extra_columns: bool = True,
    ) -> pd.DataFrame:
        """Create an ordered raw feature table."""

        return make_feature_table(
            data,
            allow_extra_columns=allow_extra_columns,
        )

    def fit(
        self,
        training_features: Any,
    ) -> "FeaturePreprocessor":
        """Fit the scaler using training features only."""

        self.scaler = fit_feature_scaler(
            training_features
        )

        return self

    def transform(
        self,
        features: Any,
        *,
        allow_extra_columns: bool = True,
    ) -> pd.DataFrame:
        """Transform feature data using the fitted scaler."""

        return transform_feature_table(
            features,
            self.scaler,
            allow_extra_columns=allow_extra_columns,
        )

    def fit_transform(
        self,
        training_features: Any,
    ) -> pd.DataFrame:
        """Fit and transform the training feature table."""

        self.scaler, transformed = (
            fit_transform_feature_table(
                training_features
            )
        )

        return transformed

    def transform_one(
        self,
        features: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Transform one live FT-QuPAP feature record."""

        return self.transform(
            make_feature_frame(
                features
            ),
            allow_extra_columns=False,
        )

    def inverse_transform(
        self,
        scaled_features: Any,
    ) -> pd.DataFrame:
        """Restore scaled data to the original feature space."""

        return inverse_transform_feature_table(
            scaled_features,
            self.scaler,
        )

    def save(
        self,
        destination: str | Path,
    ) -> Path:
        """Save the fitted standalone scaler."""

        return save_feature_scaler(
            self.scaler,
            destination,
        )

    @classmethod
    def load(
        cls,
        source: str | Path,
    ) -> "FeaturePreprocessor":
        """Create a preprocessor from a trusted scaler artifact."""

        return cls(
            scaler=load_feature_scaler(
                source
            )
        )


def run_self_test() -> None:
    """
    Verify feature ordering, scaling, restoration, and artifact loading.
    """

    training_rows = [
        {
            "qber_raw": 0.01,
            "mean_syndrome_weight": 0.10,
            "max_syndrome_weight": 1.0,
            "correction_failure_rate": 0.00,
            "loss_rate": 0.01,
            "noise_estimate": 0.01,
            "ctx_urban": 1.0,
            "ctx_suburban": 0.0,
            "ctx_rural": 0.0,
        },
        {
            "qber_raw": 0.05,
            "mean_syndrome_weight": 0.50,
            "max_syndrome_weight": 2.0,
            "correction_failure_rate": 0.02,
            "loss_rate": 0.03,
            "noise_estimate": 0.04,
            "ctx_urban": 0.0,
            "ctx_suburban": 1.0,
            "ctx_rural": 0.0,
        },
        {
            "qber_raw": 0.18,
            "mean_syndrome_weight": 2.00,
            "max_syndrome_weight": 4.0,
            "correction_failure_rate": 0.20,
            "loss_rate": 0.08,
            "noise_estimate": 0.06,
            "ctx_urban": 0.0,
            "ctx_suburban": 0.0,
            "ctx_rural": 1.0,
        },
    ]

    preprocessor = FeaturePreprocessor()

    scaled_training = preprocessor.fit_transform(
        training_rows
    )

    if not preprocessor.fitted:
        raise FeaturePreprocessorError(
            "Preprocessor was not marked as fitted."
        )

    if list(
        scaled_training.columns
    ) != FEATURE_COLUMNS:
        raise FeaturePreprocessorError(
            "Scaled feature order is incorrect."
        )

    if scaled_training.shape != (
        3,
        FEATURE_COUNT,
    ):
        raise FeaturePreprocessorError(
            "Scaled training shape is incorrect."
        )

    live_features = {
        "qber_raw": 0.03,
        "mean_syndrome_weight": 0.30,
        "max_syndrome_weight": 1.0,
        "correction_failure_rate": 0.01,
        "loss_rate": 0.02,
        "noise_estimate": 0.02,
        "ctx_urban": 1.0,
        "ctx_suburban": 0.0,
        "ctx_rural": 0.0,
    }

    live_frame = make_feature_frame(
        live_features
    )

    scaled_live = preprocessor.transform_one(
        live_features
    )

    restored_live = preprocessor.inverse_transform(
        scaled_live
    )

    if not np.allclose(
        restored_live.to_numpy(
            dtype=float
        ),
        live_frame.to_numpy(
            dtype=float
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise FeaturePreprocessorError(
            "Inverse transformation did not restore features."
        )

    metadata_table = pd.DataFrame(
        [
            {
                **training_rows[0],
                "label_attack": 0,
                "scenario": "benign_clean",
            },
            {
                **training_rows[2],
                "label_attack": 1,
                "scenario": "attack_intercept_resend",
            },
        ]
    )

    selected_features = make_feature_table(
        metadata_table,
        allow_extra_columns=True,
    )

    if list(
        selected_features.columns
    ) != FEATURE_COLUMNS:
        raise FeaturePreprocessorError(
            "Training metadata entered the feature matrix."
        )

    if not np.allclose(
        scaled_training.mean(
            axis=0
        ).to_numpy(),
        np.zeros(
            FEATURE_COUNT
        ),
        atol=1e-12,
    ):
        raise FeaturePreprocessorError(
            "StandardScaler training means are incorrect."
        )

    print(
        "Feature preprocessor self-test completed successfully."
    )

    print(
        "Training rows:",
        len(training_rows),
    )

    print(
        "Feature count:",
        FEATURE_COUNT,
    )

    print(
        "Scaler fitted:",
        preprocessor.fitted,
    )

    print(
        "Live feature shape:",
        live_frame.shape,
    )


__all__ = [
    "SCALER_ARTIFACT_VERSION",
    "FeaturePreprocessorError",
    "InvalidFeatureTableError",
    "FeatureScalerNotFittedError",
    "FeatureScalerArtifactError",
    "FeatureScalingError",
    "FeaturePreprocessor",
    "make_feature_frame",
    "validate_feature_frame",
    "make_feature_table",
    "feature_matrix",
    "is_scaler_fitted",
    "validate_scaler",
    "fit_feature_scaler",
    "transform_feature_table",
    "fit_transform_feature_table",
    "inverse_transform_feature_table",
    "save_feature_scaler",
    "load_feature_scaler",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        FeaturePreprocessorError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[FEATURE PREPROCESSOR ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error