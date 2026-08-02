#!/usr/bin/env python3
"""
Train, calibrate, validate, and export the FT-QuPAP Gaussian Process detector.

Flowchart/notebook alignment:
- Flowchart Steps 17-18:
    Extract observable GP features and estimate calibrated P(attack).
- Flowchart Step 19:
    Apply a cost-sensitive threshold and retry gray-zone policy.
- Notebook 08:
    Fit StandardScaler and an exact GaussianProcessClassifier using only
    the training split.
- Notebook 09:
    Fit IsotonicRegression using only the disjoint calibration/validation
    split, select the Bayesian-risk threshold from that split, apply the
    operational lower bound, and evaluate once on the independent test split.

Input files:
    data/processed/training_features.csv
    data/processed/validation_features.csv
    data/processed/independent_test_features.csv

Exported model files:
    models/gp_model.pkl
    models/feature_scaler.pkl
    models/calibration_model.pkl
    models/threshold.json
    models/feature_order.json
    models/model_metadata.json

Generated result files:
    data/results/performance_metrics.csv
    data/results/confusion_matrix.csv
    data/results/calibration_results.csv
    data/results/threshold_analysis.csv

Important research boundary:
The preceding generate_demo_data.py script marks its synthetic fixtures as
research_eligible=false. A model trained from those fixtures is suitable for
dashboard development, integration testing, and capstone demonstration, but
its metrics must not be reported as final paper evidence. Run this exporter
again with full protocol-engine session traces for final research artifacts.

Run from the project root:
    python scripts/export_gp_model.py

Useful options:
    python scripts/export_gp_model.py --max-train-rows 300
    python scripts/export_gp_model.py --max-train-rows 2500 --force
    python scripts/export_gp_model.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

TRAINING_FILE = PROCESSED_DIR / "training_features.csv"
CALIBRATION_FILE = PROCESSED_DIR / "validation_features.csv"
INDEPENDENT_TEST_FILE = PROCESSED_DIR / "independent_test_features.csv"

GP_MODEL_FILE = MODEL_DIR / "gp_model.pkl"
FEATURE_SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"
CALIBRATION_MODEL_FILE = MODEL_DIR / "calibration_model.pkl"
THRESHOLD_FILE = MODEL_DIR / "threshold.json"
FEATURE_ORDER_FILE = MODEL_DIR / "feature_order.json"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

PERFORMANCE_METRICS_FILE = RESULTS_DIR / "performance_metrics.csv"
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix.csv"
CALIBRATION_RESULTS_FILE = RESULTS_DIR / "calibration_results.csv"
THRESHOLD_ANALYSIS_FILE = RESULTS_DIR / "threshold_analysis.csv"

LOG_FILE = LOG_DIR / "export_gp_model.log"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = "research-simulator-v5-1-large-ml-operational-threshold"
MODEL_SCHEMA_VERSION = 1
MASTER_SEED = 20260701

FEATURE_ORDER = [
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

LABEL_COLUMN = "label_attack"
SESSION_ID_COLUMN = "session_id"

DEFAULT_MAX_TRAIN_ROWS = 300
DEFAULT_FALSE_ACCEPT_COST = 10.0
DEFAULT_FALSE_REJECT_COST = 1.0
DEFAULT_MINIMUM_OPERATIONAL_THRESHOLD = 0.15
DEFAULT_RETRY_UPPER_PROBABILITY = 0.20
DEFAULT_FIXED_QBER_THRESHOLD = 0.11
DEFAULT_MAXIMUM_LOSS_RATE = 0.15
DEFAULT_MINIMUM_OBSERVED_CHECK_BLOCKS = 24
DEFAULT_MAXIMUM_AUTHENTICATION_ATTEMPTS = 3

MANAGED_OUTPUTS = (
    GP_MODEL_FILE,
    FEATURE_SCALER_FILE,
    CALIBRATION_MODEL_FILE,
    THRESHOLD_FILE,
    FEATURE_ORDER_FILE,
    MODEL_METADATA_FILE,
    PERFORMANCE_METRICS_FILE,
    CONFUSION_MATRIX_FILE,
    CALIBRATION_RESULTS_FILE,
    THRESHOLD_ANALYSIS_FILE,
)


class ModelExportError(RuntimeError):
    """Raised when the FT-QuPAP GP detector cannot be exported safely."""


@dataclass(frozen=True)
class SplitSummary:
    """Validated summary of one offline dataset split."""

    name: str
    path: str
    rows: int
    benign_rows: int
    attack_rows: int
    sha256: str
    research_scope: str


@dataclass(frozen=True)
class DecisionMetrics:
    """Held-out deterministic decision metrics at one selected threshold."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    accuracy: float
    balanced_accuracy: float
    precision_attack: float
    recall_attack: float
    f1_attack: float
    false_accept_rate: float
    false_reject_rate: float
    attack_detection_rate: float
    benign_acceptance_rate: float


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and file logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.export_gp_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train, calibrate, evaluate, and export the FT-QuPAP "
            "session-level Gaussian Process attack detector."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Reproducibility seed (default: {MASTER_SEED}).",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=DEFAULT_MAX_TRAIN_ROWS,
        help=(
            "Maximum balanced rows used by the exact GP. "
            f"Default {DEFAULT_MAX_TRAIN_ROWS}; notebook large profile 2500."
        ),
    )
    parser.add_argument(
        "--false-accept-cost",
        type=float,
        default=DEFAULT_FALSE_ACCEPT_COST,
        help=(
            "Bayesian cost of accepting an attack "
            f"(default: {DEFAULT_FALSE_ACCEPT_COST})."
        ),
    )
    parser.add_argument(
        "--false-reject-cost",
        type=float,
        default=DEFAULT_FALSE_REJECT_COST,
        help=(
            "Bayesian cost of rejecting a benign user "
            f"(default: {DEFAULT_FALSE_REJECT_COST})."
        ),
    )
    parser.add_argument(
        "--minimum-operational-threshold",
        type=float,
        default=DEFAULT_MINIMUM_OPERATIONAL_THRESHOLD,
        help=(
            "Availability-constrained lower bound applied after raw "
            "calibration-only threshold selection "
            f"(default: {DEFAULT_MINIMUM_OPERATIONAL_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--retry-upper-probability",
        type=float,
        default=DEFAULT_RETRY_UPPER_PROBABILITY,
        help=(
            "Upper P(attack) boundary for the low-risk retry gray zone "
            f"(default: {DEFAULT_RETRY_UPPER_PROBABILITY})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing managed model and result files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing model artifacts without retraining.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject unsafe or internally inconsistent command-line settings."""

    if args.max_train_rows < 20:
        raise ModelExportError("--max-train-rows must be at least 20.")

    if args.false_accept_cost <= 0.0:
        raise ModelExportError("--false-accept-cost must be positive.")

    if args.false_reject_cost <= 0.0:
        raise ModelExportError("--false-reject-cost must be positive.")

    for name, value in (
        (
            "minimum operational threshold",
            args.minimum_operational_threshold,
        ),
        ("retry upper probability", args.retry_upper_probability),
    ):
        if not 0.0 <= value <= 1.0:
            raise ModelExportError(f"{name} must be between 0 and 1.")

    if (
        args.retry_upper_probability
        < args.minimum_operational_threshold
    ):
        raise ModelExportError(
            "--retry-upper-probability must be greater than or equal to "
            "--minimum-operational-threshold."
        )


def ensure_directories() -> None:
    """Create all model, result, and log directories."""

    for directory in (MODEL_DIR, RESULTS_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic, human-readable JSON."""

    atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )


def atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    """Atomically save a CSV table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        table.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_joblib_dump(path: Path, value: Any) -> None:
    """Atomically serialize one Python/scikit-learn artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        joblib.dump(value, temporary_path, compress=3)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def require_input_files() -> None:
    """Ensure the three disjoint processed splits are available."""

    missing = [
        path
        for path in (
            TRAINING_FILE,
            CALIBRATION_FILE,
            INDEPENDENT_TEST_FILE,
        )
        if not path.is_file()
    ]

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise ModelExportError(
            "Required processed feature files are missing:\n"
            f"{formatted}\n"
            "Run: python scripts/generate_demo_data.py"
        )


def enforce_overwrite_policy(force: bool) -> None:
    """Avoid silently replacing prior model evidence."""

    existing = [path for path in MANAGED_OUTPUTS if path.exists()]

    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise ModelExportError(
            "Managed outputs already exist:\n"
            f"{formatted}\n"
            "Use --force only after preserving any final research artifacts."
        )


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Normalize common CSV Boolean representations."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "false": False,
                "0": False,
                "no": False,
            }
        )
    )

    if normalized.isna().any():
        raise ModelExportError(
            "research_eligible contains an unsupported Boolean value."
        )

    return normalized.astype(bool)


def research_scope(table: pd.DataFrame) -> str:
    """Classify the evidence boundary recorded in a split."""

    if "research_eligible" not in table.columns:
        return "unspecified"

    flags = parse_boolean_series(table["research_eligible"])

    if bool(flags.all()):
        return "research_eligible"
    if bool((~flags).all()):
        return "development_only"
    return "mixed"


def validate_feature_table(
    table: pd.DataFrame,
    split_name: str,
    path: Path,
) -> SplitSummary:
    """Validate labels, observable features, context encoding, and scope."""

    required_columns = set(FEATURE_ORDER + [LABEL_COLUMN])
    missing = sorted(required_columns - set(table.columns))

    if missing:
        raise ModelExportError(
            f"{split_name} is missing required columns: {missing}"
        )

    if table.empty:
        raise ModelExportError(f"{split_name} is empty.")

    labels = pd.to_numeric(
        table[LABEL_COLUMN],
        errors="coerce",
    )

    if labels.isna().any():
        raise ModelExportError(
            f"{split_name}.{LABEL_COLUMN} contains non-numeric values."
        )

    integer_labels = labels.astype(int)
    if not np.array_equal(labels.to_numpy(), integer_labels.to_numpy()):
        raise ModelExportError(
            f"{split_name}.{LABEL_COLUMN} must contain integers."
        )

    classes = set(integer_labels.unique())
    if classes != {0, 1}:
        raise ModelExportError(
            f"{split_name} must contain both labels 0 and 1; found {classes}."
        )

    feature_table = table[FEATURE_ORDER].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if feature_table.isna().any().any():
        columns = feature_table.columns[
            feature_table.isna().any()
        ].tolist()
        raise ModelExportError(
            f"{split_name} has missing/non-numeric features in: {columns}"
        )

    values = feature_table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ModelExportError(
            f"{split_name} contains NaN or infinite feature values."
        )

    probability_columns = (
        "qber_raw",
        "correction_failure_rate",
        "loss_rate",
        "noise_estimate",
    )
    for column in probability_columns:
        invalid = ~feature_table[column].between(0.0, 1.0)
        if invalid.any():
            raise ModelExportError(
                f"{split_name}.{column} must be between 0 and 1."
            )

    context_columns = ["ctx_urban", "ctx_suburban", "ctx_rural"]
    context_values = feature_table[context_columns].to_numpy(dtype=float)

    if not np.isin(context_values, [0.0, 1.0]).all():
        raise ModelExportError(
            f"{split_name} context columns must be one-hot 0/1 values."
        )

    if not np.allclose(context_values.sum(axis=1), 1.0):
        raise ModelExportError(
            f"{split_name} must activate exactly one context per row."
        )

    return SplitSummary(
        name=split_name,
        path=str(path.relative_to(PROJECT_ROOT)),
        rows=int(len(table)),
        benign_rows=int((integer_labels == 0).sum()),
        attack_rows=int((integer_labels == 1).sum()),
        sha256=sha256_file(path),
        research_scope=research_scope(table),
    )


def validate_split_independence(
    training: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Reject session-ID overlap between train, calibration, and test."""

    tables = {
        "training": training,
        "calibration": calibration,
        "independent_test": test,
    }

    if not all(SESSION_ID_COLUMN in table.columns for table in tables.values()):
        return

    identifiers = {
        name: set(table[SESSION_ID_COLUMN].astype(str))
        for name, table in tables.items()
    }

    comparisons = (
        ("training", "calibration"),
        ("training", "independent_test"),
        ("calibration", "independent_test"),
    )

    for left, right in comparisons:
        overlap = identifiers[left] & identifiers[right]
        if overlap:
            preview = sorted(overlap)[:5]
            raise ModelExportError(
                f"{left} and {right} share session IDs: {preview}"
            )


def load_and_validate_splits(
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[SplitSummary],
]:
    """Load all three offline datasets and perform leakage checks."""

    require_input_files()

    training = pd.read_csv(TRAINING_FILE)
    calibration = pd.read_csv(CALIBRATION_FILE)
    test = pd.read_csv(INDEPENDENT_TEST_FILE)

    summaries = [
        validate_feature_table(training, "training", TRAINING_FILE),
        validate_feature_table(
            calibration,
            "calibration",
            CALIBRATION_FILE,
        ),
        validate_feature_table(
            test,
            "independent_test",
            INDEPENDENT_TEST_FILE,
        ),
    ]

    validate_split_independence(training, calibration, test)

    for summary in summaries:
        logger.info(
            "%s rows=%d benign=%d attack=%d scope=%s",
            summary.name,
            summary.rows,
            summary.benign_rows,
            summary.attack_rows,
            summary.research_scope,
        )

    return training, calibration, test, summaries


def balanced_training_subset(
    training: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select a reproducible class-balanced exact-GP subset.

    Exact Gaussian Process training grows cubically with row count. Full
    calibration and independent-test splits are retained without subsampling.
    """

    if len(training) <= max_rows:
        return training.sample(
            frac=1.0,
            random_state=seed,
        ).reset_index(drop=True)

    labels = sorted(training[LABEL_COLUMN].astype(int).unique())
    rows_per_class = max_rows // len(labels)
    selected_indices: list[int] = []

    for label in labels:
        class_indices = training.index[
            training[LABEL_COLUMN].astype(int) == label
        ]
        take_n = min(rows_per_class, len(class_indices))
        sampled = (
            pd.Series(class_indices)
            .sample(
                n=take_n,
                random_state=seed + int(label),
                replace=False,
            )
            .astype(int)
            .tolist()
        )
        selected_indices.extend(sampled)

    remaining_capacity = max_rows - len(selected_indices)
    if remaining_capacity > 0:
        remaining_indices = training.index.difference(selected_indices)
        if len(remaining_indices) > 0:
            extra = (
                pd.Series(remaining_indices)
                .sample(
                    n=min(remaining_capacity, len(remaining_indices)),
                    random_state=seed + 99,
                    replace=False,
                )
                .astype(int)
                .tolist()
            )
            selected_indices.extend(extra)

    subset = training.loc[selected_indices].copy()

    return subset.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)


def build_gp_model(seed: int) -> GaussianProcessClassifier:
    """Construct the notebook-aligned exact Gaussian Process classifier."""

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(
            length_scale=np.ones(len(FEATURE_ORDER)),
            length_scale_bounds=(1e-2, 1e3),
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1.0),
        )
    )

    return GaussianProcessClassifier(
        kernel=kernel,
        random_state=seed,
        max_iter_predict=100,
        n_restarts_optimizer=0,
        optimizer=None,
    )


def calibrated_probabilities(
    model: GaussianProcessClassifier,
    scaler: StandardScaler,
    calibrator: IsotonicRegression,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw GP and isotonic-calibrated attack probabilities."""

    transformed = scaler.transform(features[FEATURE_ORDER])
    raw = np.asarray(
        model.predict_proba(transformed)[:, 1],
        dtype=float,
    )
    calibrated = np.asarray(
        calibrator.predict(raw),
        dtype=float,
    )

    return (
        np.clip(raw, 0.0, 1.0),
        np.clip(calibrated, 0.0, 1.0),
    )


def threshold_candidates(probabilities: Sequence[float]) -> np.ndarray:
    """Create complete decision boundaries from unique probabilities."""

    values = np.unique(np.asarray(probabilities, dtype=float))

    if len(values) == 0:
        raise ModelExportError(
            "Threshold selection received no probabilities."
        )

    if len(values) == 1:
        return np.array([0.0, 1.0], dtype=float)

    midpoints = (values[:-1] + values[1:]) / 2.0

    return np.unique(
        np.concatenate(
            (
                np.array([0.0]),
                midpoints,
                np.array([1.0]),
            )
        )
    )


def build_threshold_analysis(
    labels: Sequence[int],
    probabilities: Sequence[float],
    false_accept_cost: float,
    false_reject_cost: float,
) -> tuple[pd.DataFrame, float]:
    """
    Score thresholds using the calibration split only.

    Protocol interpretation:
    - p_attack < threshold: GP accepts.
    - p_attack >= threshold: GP rejects.
    - False accept: an attack remains below threshold.
    - False reject: a benign session reaches/exceeds threshold.
    """

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    attack_mask = y == 1
    benign_mask = y == 0

    if not attack_mask.any() or not benign_mask.any():
        raise ModelExportError(
            "Threshold selection requires benign and attack calibration rows."
        )

    theoretical_threshold = (
        false_reject_cost
        / (false_accept_cost + false_reject_cost)
    )

    rows: list[dict[str, Any]] = []

    for threshold in threshold_candidates(p):
        rejected = p >= threshold
        false_accept_rate = float(np.mean(~rejected[attack_mask]))
        false_reject_rate = float(np.mean(rejected[benign_mask]))
        risk = (
            false_accept_cost * false_accept_rate
            + false_reject_cost * false_reject_rate
        )

        rows.append(
            {
                "threshold": float(threshold),
                "false_accept_rate": false_accept_rate,
                "false_reject_rate": false_reject_rate,
                "attack_detection_rate": float(
                    np.mean(rejected[attack_mask])
                ),
                "benign_acceptance_rate": float(
                    np.mean(~rejected[benign_mask])
                ),
                "bayes_risk": float(risk),
                "distance_from_theoretical_threshold": float(
                    abs(threshold - theoretical_threshold)
                ),
                "boundary_source": "calibration_probability_midpoint",
            }
        )

    analysis = pd.DataFrame(rows).sort_values(
        [
            "bayes_risk",
            "distance_from_theoretical_threshold",
            "threshold",
        ],
        kind="stable",
    )

    selected = analysis.iloc[0]
    raw_threshold = float(selected["threshold"])

    analysis = analysis.sort_values("threshold").reset_index(drop=True)
    analysis["selected_raw_threshold"] = np.isclose(
        analysis["threshold"],
        raw_threshold,
        atol=1e-15,
    )

    return analysis, raw_threshold


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Build quantile calibration bins and calculate held-out ECE."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if len(y) != len(p) or len(y) == 0:
        raise ModelExportError(
            "Calibration diagnostics require equal, non-empty arrays."
        )

    target_bins = min(n_bins, len(y))
    ordered_indices = np.argsort(p)
    groups = np.array_split(ordered_indices, target_bins)

    rows: list[dict[str, Any]] = []
    ece = 0.0

    for bin_index, indices in enumerate(groups):
        if len(indices) == 0:
            continue

        mean_probability = float(np.mean(p[indices]))
        observed_attack_rate = float(np.mean(y[indices]))
        count = int(len(indices))
        absolute_gap = abs(mean_probability - observed_attack_rate)

        ece += (count / len(y)) * absolute_gap

        rows.append(
            {
                "bin_index": bin_index,
                "bin_lower": float(np.min(p[indices])),
                "bin_upper": float(np.max(p[indices])),
                "sample_count": count,
                "mean_predicted_probability": mean_probability,
                "observed_attack_rate": observed_attack_rate,
                "absolute_calibration_gap": float(absolute_gap),
            }
        )

    return pd.DataFrame(rows), float(ece)


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, float]:
    """Calculate held-out discrimination and probability-quality metrics."""

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if set(np.unique(y)) != {0, 1}:
        raise ModelExportError(
            "Probability metrics require both labels 0 and 1."
        )

    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
    }


def decision_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> DecisionMetrics:
    """Calculate held-out policy metrics at the operational threshold."""

    y = np.asarray(labels, dtype=int)
    predicted_attack = (
        np.asarray(probabilities, dtype=float) >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted_attack,
        labels=[0, 1],
    ).ravel()

    false_accept_rate = float(
        fn / (tp + fn)
        if (tp + fn)
        else 0.0
    )
    false_reject_rate = float(
        fp / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    return DecisionMetrics(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        accuracy=float(accuracy_score(y, predicted_attack)),
        balanced_accuracy=float(
            balanced_accuracy_score(y, predicted_attack)
        ),
        precision_attack=float(
            precision_score(
                y,
                predicted_attack,
                zero_division=0,
            )
        ),
        recall_attack=float(
            recall_score(
                y,
                predicted_attack,
                zero_division=0,
            )
        ),
        f1_attack=float(
            f1_score(
                y,
                predicted_attack,
                zero_division=0,
            )
        ),
        false_accept_rate=false_accept_rate,
        false_reject_rate=false_reject_rate,
        attack_detection_rate=float(
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        ),
        benign_acceptance_rate=float(
            tn / (tn + fp)
            if (tn + fp)
            else 0.0
        ),
    )


def make_performance_table(
    probability_result: Mapping[str, float],
    policy_result: DecisionMetrics,
    ece: float,
    raw_threshold: float,
    operational_threshold: float,
    split_summaries: Sequence[SplitSummary],
    exact_gp_training_rows: int,
) -> pd.DataFrame:
    """Create a long-form metrics table for dashboards and reports."""

    metrics: dict[str, float | int] = {
        **probability_result,
        "expected_calibration_error": ece,
        "accuracy": policy_result.accuracy,
        "balanced_accuracy": policy_result.balanced_accuracy,
        "precision_attack": policy_result.precision_attack,
        "recall_attack": policy_result.recall_attack,
        "f1_attack": policy_result.f1_attack,
        "false_accept_rate": policy_result.false_accept_rate,
        "false_reject_rate": policy_result.false_reject_rate,
        "attack_detection_rate": policy_result.attack_detection_rate,
        "benign_acceptance_rate": (
            policy_result.benign_acceptance_rate
        ),
        "raw_calibrated_threshold": raw_threshold,
        "operational_threshold": operational_threshold,
        "exact_gp_training_rows": exact_gp_training_rows,
    }

    for summary in split_summaries:
        metrics[f"{summary.name}_rows"] = summary.rows

    return pd.DataFrame(
        [
            {
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "evaluation_split": "independent_test",
                "metric": name,
                "value": value,
            }
            for name, value in metrics.items()
        ]
    )


def make_confusion_table(
    result: DecisionMetrics,
    threshold: float,
) -> pd.DataFrame:
    """Create an explicit 2x2 held-out confusion matrix table."""

    return pd.DataFrame(
        [
            {
                "actual_label": "benign",
                "predicted_label": "benign",
                "count": result.true_negative,
                "operational_threshold": threshold,
            },
            {
                "actual_label": "benign",
                "predicted_label": "attack",
                "count": result.false_positive,
                "operational_threshold": threshold,
            },
            {
                "actual_label": "attack",
                "predicted_label": "benign",
                "count": result.false_negative,
                "operational_threshold": threshold,
            },
            {
                "actual_label": "attack",
                "predicted_label": "attack",
                "count": result.true_positive,
                "operational_threshold": threshold,
            },
        ]
    )


def installed_version(distribution: str) -> str | None:
    """Read one installed package version without failing the export."""

    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def derive_evidence_scope(
    summaries: Sequence[SplitSummary],
) -> str:
    """Combine split-level research eligibility into one model scope."""

    scopes = {summary.research_scope for summary in summaries}

    if scopes == {"development_only"}:
        return "synthetic_demo_development_only"
    if scopes == {"research_eligible"}:
        return "research_eligible_session_traces"
    if scopes == {"unspecified"}:
        return "source_scope_unspecified"
    return "mixed_or_partially_specified"


def build_model_metadata(
    *,
    args: argparse.Namespace,
    model: GaussianProcessClassifier,
    raw_threshold: float,
    operational_threshold: float,
    split_summaries: Sequence[SplitSummary],
    training_rows_before_subset: int,
    exact_gp_training_rows: int,
    probability_result: Mapping[str, float],
    policy_result: DecisionMetrics,
    ece: float,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Build the reproducibility and evidence-boundary manifest."""

    model_identity_material = json.dumps(
        {
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "feature_order": FEATURE_ORDER,
            "operational_threshold": operational_threshold,
            "protocol_version": PROTOCOL_VERSION,
        },
        sort_keys=True,
    ).encode("utf-8")

    model_id = hashlib.sha256(model_identity_material).hexdigest()[:24]

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_id": model_id,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": utc_now_iso(),
        "model_type": "GaussianProcessClassifier",
        "calibration_type": "IsotonicRegression",
        "probability_semantics": "P(attack)",
        "decision_rule": {
            "accept": "p_attack < operational_threshold",
            "reject": "p_attack >= operational_threshold",
            "retry_gray_zone_enabled": bool(
                operational_threshold < args.retry_upper_probability
            ),
            "retry_gray_zone": (
                "operational_threshold <= p_attack < "
                "retry_upper_probability, subject to deterministic "
                "low-risk checks and a fresh nonce/new session. The "
                "interval is empty when threshold >= upper boundary."
            ),
        },
        "feature_order": FEATURE_ORDER,
        "label_column": LABEL_COLUMN,
        "seed": int(args.seed),
        "kernel": str(model.kernel_),
        "optimizer": "disabled_for_reproducible_exact_gp",
        "max_iter_predict": int(model.max_iter_predict),
        "threshold_policy": {
            "raw_calibrated_threshold": raw_threshold,
            "minimum_operational_threshold": float(
                args.minimum_operational_threshold
            ),
            "operational_threshold": operational_threshold,
            "retry_upper_probability": float(
                args.retry_upper_probability
            ),
            "fixed_qber_threshold": (
                DEFAULT_FIXED_QBER_THRESHOLD
            ),
            "maximum_loss_rate": DEFAULT_MAXIMUM_LOSS_RATE,
            "minimum_observed_check_blocks": (
                DEFAULT_MINIMUM_OBSERVED_CHECK_BLOCKS
            ),
            "maximum_authentication_attempts": (
                DEFAULT_MAXIMUM_AUTHENTICATION_ATTEMPTS
            ),
            "cost_false_accept": float(args.false_accept_cost),
            "cost_false_reject": float(args.false_reject_cost),
        },
        "training": {
            "full_training_rows": training_rows_before_subset,
            "exact_gp_training_rows": exact_gp_training_rows,
            "maximum_requested_rows": int(args.max_train_rows),
            "balanced_subset": True,
            "scaler_fit_scope": "exact_gp_training_subset_only",
            "gp_fit_scope": "exact_gp_training_subset_only",
            "calibrator_fit_scope": "disjoint_calibration_split_only",
            "threshold_selection_scope": (
                "disjoint_calibration_split_only"
            ),
            "heldout_test_used_for_tuning": False,
        },
        "dataset_splits": [
            asdict(summary)
            for summary in split_summaries
        ],
        "evidence_scope": derive_evidence_scope(split_summaries),
        "heldout_metrics": {
            **dict(probability_result),
            **asdict(policy_result),
            "expected_calibration_error": ece,
        },
        "artifacts": dict(artifact_hashes),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": installed_version("numpy"),
            "pandas": installed_version("pandas"),
            "scikit_learn": installed_version("scikit-learn"),
            "joblib": installed_version("joblib"),
        },
    }


def train_and_export(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Execute the complete training, calibration, testing, and export flow."""

    ensure_directories()
    enforce_overwrite_policy(args.force)

    (
        training,
        calibration,
        independent_test,
        split_summaries,
    ) = load_and_validate_splits(logger)

    exact_training = balanced_training_subset(
        training,
        max_rows=args.max_train_rows,
        seed=args.seed,
    )

    logger.info(
        "Exact GP training subset: %d of %d rows.",
        len(exact_training),
        len(training),
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(
        exact_training[FEATURE_ORDER]
    )
    y_train = exact_training[LABEL_COLUMN].astype(int).to_numpy()

    model = build_gp_model(args.seed)
    logger.info("Training exact Gaussian Process classifier...")
    model.fit(x_train_scaled, y_train)
    logger.info("GP training complete.")

    x_calibration_scaled = scaler.transform(
        calibration[FEATURE_ORDER]
    )
    y_calibration = (
        calibration[LABEL_COLUMN].astype(int).to_numpy()
    )

    raw_calibration_probability = np.asarray(
        model.predict_proba(x_calibration_scaled)[:, 1],
        dtype=float,
    )

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    calibrator.fit(
        raw_calibration_probability,
        y_calibration,
    )

    calibrated_calibration_probability = np.clip(
        np.asarray(
            calibrator.predict(raw_calibration_probability),
            dtype=float,
        ),
        0.0,
        1.0,
    )

    threshold_analysis, raw_threshold = build_threshold_analysis(
        labels=y_calibration,
        probabilities=calibrated_calibration_probability,
        false_accept_cost=args.false_accept_cost,
        false_reject_cost=args.false_reject_cost,
    )

    operational_threshold = float(
        np.clip(
            max(
                raw_threshold,
                args.minimum_operational_threshold,
            ),
            0.0,
            1.0,
        )
    )

    # The operational lower bound may not coincide with a calibration
    # probability midpoint. Add it explicitly so threshold_analysis.csv
    # always contains the deployed boundary and its measured calibration risk.
    if not np.isclose(
        threshold_analysis["threshold"].to_numpy(dtype=float),
        operational_threshold,
        atol=1e-15,
    ).any():
        calibration_rejected = (
            calibrated_calibration_probability >= operational_threshold
        )
        calibration_attack_mask = y_calibration == 1
        calibration_benign_mask = y_calibration == 0
        operational_far = float(
            np.mean(~calibration_rejected[calibration_attack_mask])
        )
        operational_frr = float(
            np.mean(calibration_rejected[calibration_benign_mask])
        )
        theoretical_threshold = (
            args.false_reject_cost
            / (args.false_accept_cost + args.false_reject_cost)
        )
        operational_row = pd.DataFrame(
            [
                {
                    "threshold": operational_threshold,
                    "false_accept_rate": operational_far,
                    "false_reject_rate": operational_frr,
                    "attack_detection_rate": 1.0 - operational_far,
                    "benign_acceptance_rate": 1.0 - operational_frr,
                    "bayes_risk": (
                        args.false_accept_cost * operational_far
                        + args.false_reject_cost * operational_frr
                    ),
                    "distance_from_theoretical_threshold": abs(
                        operational_threshold - theoretical_threshold
                    ),
                    "boundary_source": "operational_policy_boundary",
                    "selected_raw_threshold": False,
                }
            ]
        )
        threshold_analysis = pd.concat(
            [threshold_analysis, operational_row],
            ignore_index=True,
        ).sort_values("threshold").reset_index(drop=True)

    threshold_analysis["operational_threshold"] = (
        operational_threshold
    )
    threshold_analysis["selected_operational_threshold"] = np.isclose(
        threshold_analysis["threshold"],
        operational_threshold,
        atol=1e-15,
    )

    gray_zone_enabled = bool(
        operational_threshold < args.retry_upper_probability
    )

    raw_test_probability, calibrated_test_probability = (
        calibrated_probabilities(
            model,
            scaler,
            calibrator,
            independent_test,
        )
    )
    y_test = (
        independent_test[LABEL_COLUMN].astype(int).to_numpy()
    )

    probability_result = probability_metrics(
        y_test,
        calibrated_test_probability,
    )
    calibration_table, ece = expected_calibration_error(
        y_test,
        calibrated_test_probability,
        n_bins=10,
    )
    policy_result = decision_metrics(
        y_test,
        calibrated_test_probability,
        operational_threshold,
    )

    calibration_table["evaluation_split"] = "independent_test"
    calibration_table["calibration_method"] = (
        "isotonic_regression"
    )
    calibration_table["operational_threshold"] = (
        operational_threshold
    )

    threshold_data = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "probability_semantics": "P(attack)",
        "raw_calibrated_threshold": raw_threshold,
        "minimum_operational_threshold": float(
            args.minimum_operational_threshold
        ),
        "operational_threshold": operational_threshold,
        "retry_upper_probability": float(
            args.retry_upper_probability
        ),
        "fixed_qber_threshold": DEFAULT_FIXED_QBER_THRESHOLD,
        "maximum_loss_rate": DEFAULT_MAXIMUM_LOSS_RATE,
        "minimum_observed_check_blocks": (
            DEFAULT_MINIMUM_OBSERVED_CHECK_BLOCKS
        ),
        "maximum_authentication_attempts": (
            DEFAULT_MAXIMUM_AUTHENTICATION_ATTEMPTS
        ),
        "cost_false_accept": float(args.false_accept_cost),
        "cost_false_reject": float(args.false_reject_cost),
        "accept_rule": "p_attack < operational_threshold",
        "reject_rule": "p_attack >= operational_threshold",
        "gp_gray_zone_enabled": gray_zone_enabled,
        "retry_rule": (
            "operational_threshold <= p_attack < "
            "retry_upper_probability; deterministic low-risk checks "
            "must also pass. The interval is disabled when the "
            "operational threshold is not below the upper boundary."
        ),
    }

    performance_table = make_performance_table(
        probability_result=probability_result,
        policy_result=policy_result,
        ece=ece,
        raw_threshold=raw_threshold,
        operational_threshold=operational_threshold,
        split_summaries=split_summaries,
        exact_gp_training_rows=len(exact_training),
    )
    confusion_table = make_confusion_table(
        policy_result,
        operational_threshold,
    )

    atomic_joblib_dump(FEATURE_SCALER_FILE, scaler)
    atomic_joblib_dump(GP_MODEL_FILE, model)
    atomic_joblib_dump(CALIBRATION_MODEL_FILE, calibrator)
    atomic_write_json(FEATURE_ORDER_FILE, FEATURE_ORDER)
    atomic_write_json(THRESHOLD_FILE, threshold_data)

    atomic_write_csv(
        PERFORMANCE_METRICS_FILE,
        performance_table,
    )
    atomic_write_csv(
        CONFUSION_MATRIX_FILE,
        confusion_table,
    )
    atomic_write_csv(
        CALIBRATION_RESULTS_FILE,
        calibration_table,
    )
    atomic_write_csv(
        THRESHOLD_ANALYSIS_FILE,
        threshold_analysis,
    )

    artifact_hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in (
            GP_MODEL_FILE,
            FEATURE_SCALER_FILE,
            CALIBRATION_MODEL_FILE,
            THRESHOLD_FILE,
            FEATURE_ORDER_FILE,
        )
    }

    metadata = build_model_metadata(
        args=args,
        model=model,
        raw_threshold=raw_threshold,
        operational_threshold=operational_threshold,
        split_summaries=split_summaries,
        training_rows_before_subset=len(training),
        exact_gp_training_rows=len(exact_training),
        probability_result=probability_result,
        policy_result=policy_result,
        ece=ece,
        artifact_hashes=artifact_hashes,
    )
    atomic_write_json(MODEL_METADATA_FILE, metadata)

    logger.info(
        "Raw calibration-only threshold: %.6f",
        raw_threshold,
    )
    logger.info(
        "Operational threshold: %.6f",
        operational_threshold,
    )
    logger.info(
        "Held-out ROC-AUC=%.6f PR-AUC=%.6f Brier=%.6f",
        probability_result["roc_auc"],
        probability_result["pr_auc"],
        probability_result["brier_score"],
    )
    logger.info(
        "Held-out attack detection=%.6f benign acceptance=%.6f",
        policy_result.attack_detection_rate,
        policy_result.benign_acceptance_rate,
    )
    logger.info("Model ID: %s", metadata["model_id"])

    # Avoid leaving large arrays in the returned summary.
    return {
        "model_id": metadata["model_id"],
        "evidence_scope": metadata["evidence_scope"],
        "training_rows": len(training),
        "exact_gp_training_rows": len(exact_training),
        "calibration_rows": len(calibration),
        "independent_test_rows": len(independent_test),
        "raw_calibrated_threshold": raw_threshold,
        "operational_threshold": operational_threshold,
        "retry_upper_probability": args.retry_upper_probability,
        "heldout_metrics": {
            **probability_result,
            "expected_calibration_error": ece,
            "attack_detection_rate": (
                policy_result.attack_detection_rate
            ),
            "benign_acceptance_rate": (
                policy_result.benign_acceptance_rate
            ),
            "false_accept_rate": policy_result.false_accept_rate,
            "false_reject_rate": policy_result.false_reject_rate,
        },
        "exported_files": [
            str(path.relative_to(PROJECT_ROOT))
            for path in MANAGED_OUTPUTS
        ],
        "raw_test_probability_range": [
            float(np.min(raw_test_probability)),
            float(np.max(raw_test_probability)),
        ],
        "calibrated_test_probability_range": [
            float(np.min(calibrated_test_probability)),
            float(np.max(calibrated_test_probability)),
        ],
    }


def validate_json_file(path: Path) -> Any:
    """Read a required JSON file with a useful error."""

    if not path.is_file():
        raise ModelExportError(f"Missing artifact: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelExportError(
            f"Invalid JSON artifact: {path}"
        ) from exc


def validate_existing_artifacts(
    logger: logging.Logger,
) -> dict[str, Any]:
    """Load and smoke-test all exported GP deployment artifacts."""

    require_input_files()

    missing = [path for path in MANAGED_OUTPUTS if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise ModelExportError(
            "Model export is incomplete; missing files:\n"
            f"{formatted}"
        )

    feature_order = validate_json_file(FEATURE_ORDER_FILE)
    threshold_data = validate_json_file(THRESHOLD_FILE)
    metadata = validate_json_file(MODEL_METADATA_FILE)

    if feature_order != FEATURE_ORDER:
        raise ModelExportError(
            "feature_order.json does not match the protocol feature schema."
        )

    if metadata.get("feature_order") != FEATURE_ORDER:
        raise ModelExportError(
            "model_metadata.json feature order is inconsistent."
        )

    raw_threshold = float(
        threshold_data["raw_calibrated_threshold"]
    )
    operational_threshold = float(
        threshold_data["operational_threshold"]
    )
    minimum_threshold = float(
        threshold_data["minimum_operational_threshold"]
    )
    retry_upper = float(
        threshold_data["retry_upper_probability"]
    )

    for name, value in (
        ("raw threshold", raw_threshold),
        ("operational threshold", operational_threshold),
        ("minimum operational threshold", minimum_threshold),
        ("retry upper probability", retry_upper),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ModelExportError(
                f"Invalid {name} in threshold.json: {value}"
            )

    expected_operational = max(raw_threshold, minimum_threshold)
    if not math.isclose(
        operational_threshold,
        expected_operational,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ModelExportError(
            "Operational threshold is not max(raw, minimum)."
        )

    scaler = joblib.load(FEATURE_SCALER_FILE)
    model = joblib.load(GP_MODEL_FILE)
    calibrator = joblib.load(CALIBRATION_MODEL_FILE)

    if not isinstance(scaler, StandardScaler):
        raise ModelExportError(
            "feature_scaler.pkl is not a StandardScaler."
        )

    if not isinstance(model, GaussianProcessClassifier):
        raise ModelExportError(
            "gp_model.pkl is not a GaussianProcessClassifier."
        )

    if not isinstance(calibrator, IsotonicRegression):
        raise ModelExportError(
            "calibration_model.pkl is not an IsotonicRegression."
        )

    test_table = pd.read_csv(INDEPENDENT_TEST_FILE)
    validate_feature_table(
        test_table,
        "independent_test",
        INDEPENDENT_TEST_FILE,
    )

    smoke_features = test_table.iloc[[0]][FEATURE_ORDER]
    raw_probability = float(
        model.predict_proba(
            scaler.transform(smoke_features)
        )[0, 1]
    )
    calibrated_probability = float(
        calibrator.predict([raw_probability])[0]
    )

    if not 0.0 <= raw_probability <= 1.0:
        raise ModelExportError(
            "GP smoke-test probability is outside [0, 1]."
        )

    if not 0.0 <= calibrated_probability <= 1.0:
        raise ModelExportError(
            "Calibrated smoke-test probability is outside [0, 1]."
        )

    recorded_artifacts = metadata.get("artifacts", {})
    for relative_name, expected_hash in recorded_artifacts.items():
        artifact_path = PROJECT_ROOT / relative_name
        if not artifact_path.is_file():
            raise ModelExportError(
                f"Metadata references a missing artifact: {relative_name}"
            )
        actual_hash = sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise ModelExportError(
                f"Artifact checksum mismatch: {relative_name}"
            )

    logger.info(
        "Validated model %s with smoke P(attack)=%.6f.",
        metadata.get("model_id", "unknown"),
        calibrated_probability,
    )

    return {
        "valid": True,
        "model_id": metadata.get("model_id"),
        "protocol_version": metadata.get("protocol_version"),
        "evidence_scope": metadata.get("evidence_scope"),
        "feature_count": len(feature_order),
        "raw_calibrated_threshold": raw_threshold,
        "operational_threshold": operational_threshold,
        "retry_upper_probability": retry_upper,
        "smoke_raw_attack_probability": raw_probability,
        "smoke_calibrated_attack_probability": (
            calibrated_probability
        ),
        "validated_files": [
            str(path.relative_to(PROJECT_ROOT))
            for path in MANAGED_OUTPUTS
        ],
    }


def main() -> int:
    """Command-line entry point."""

    args = parse_arguments()
    logger = configure_logging()

    try:
        validate_arguments(args)
        ensure_directories()

        if args.validate_only:
            summary = validate_existing_artifacts(logger)
        else:
            summary = train_and_export(args, logger)

        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    except (
        ModelExportError,
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        OSError,
    ) as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    except Exception:
        logger.exception("Unexpected GP model export failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
