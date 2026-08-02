#!/usr/bin/env python3
"""
Generate FT-QuPAP result figures from executed model and protocol artifacts.

Notebook alignment
------------------
This script reproduces the result-visualization responsibilities of the final
FT-QuPAP notebook:

    Cell 92:
        Raw QBER versus channel noise / Eve activity.
    Cell 93:
        Calibrated GP attack probability and the deployed decision boundary.
    Cell 94:
        ROC and precision-recall curves.
    Cell 94A:
        Held-out calibration curve.
    Result reporting:
        Confusion matrix and retry-policy behavior.

Expected output files
---------------------
    outputs/figures/roc_curve.png
    outputs/figures/pr_curve.png
    outputs/figures/calibration_curve.png
    outputs/figures/confusion_matrix.png
    outputs/figures/qber_comparison.png
    outputs/figures/attack_probability.png
    outputs/figures/retry_analysis.png
    outputs/figures/figure_manifest.json

Primary inputs
--------------
Model evaluation:
    models/gp_model.pkl
    models/feature_scaler.pkl
    models/calibration_model.pkl
    models/threshold.json
    models/feature_order.json
    models/model_metadata.json
    data/processed/independent_test_features.csv

Previously exported result tables:
    data/results/performance_metrics.csv
    data/results/confusion_matrix.csv
    data/results/calibration_results.csv
    data/results/threshold_analysis.csv

Executed protocol/demo results:
    data/demo/demo_session_logs.csv
    data/demo/dashboard_results.csv
    data/results/retry_results.csv
    data/results/baseline_comparison.csv

Important evidence rule
-----------------------
The script never invents or substitutes numerical results. A figure is created
only when a suitable executed table or a valid model-plus-held-out-test bundle
is available. Synthetic development fixtures remain marked as development
evidence in the figure manifest and must not be presented as final paper
results.

Usage
-----
    python scripts/generate_result_graphs.py
    python scripts/generate_result_graphs.py --force
    python scripts/generate_result_graphs.py --strict
    python scripts/generate_result_graphs.py --dpi 300
    python scripts/generate_result_graphs.py --validate-only
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import matplotlib

# Required for servers, CI, notebooks, and headless capstone laptops.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

GP_MODEL_FILE = MODEL_DIR / "gp_model.pkl"
FEATURE_SCALER_FILE = MODEL_DIR / "feature_scaler.pkl"
CALIBRATION_MODEL_FILE = MODEL_DIR / "calibration_model.pkl"
THRESHOLD_FILE = MODEL_DIR / "threshold.json"
FEATURE_ORDER_FILE = MODEL_DIR / "feature_order.json"
MODEL_METADATA_FILE = MODEL_DIR / "model_metadata.json"

INDEPENDENT_TEST_FILE = PROCESSED_DIR / "independent_test_features.csv"

PERFORMANCE_METRICS_FILE = RESULTS_DIR / "performance_metrics.csv"
CONFUSION_MATRIX_TABLE = RESULTS_DIR / "confusion_matrix.csv"
CALIBRATION_RESULTS_FILE = RESULTS_DIR / "calibration_results.csv"
THRESHOLD_ANALYSIS_FILE = RESULTS_DIR / "threshold_analysis.csv"
RETRY_RESULTS_FILE = RESULTS_DIR / "retry_results.csv"
BASELINE_COMPARISON_FILE = RESULTS_DIR / "baseline_comparison.csv"

DEMO_SESSION_LOGS_FILE = DEMO_DIR / "demo_session_logs.csv"
DASHBOARD_RESULTS_FILE = DEMO_DIR / "dashboard_results.csv"

ROC_FIGURE = OUTPUT_FIGURE_DIR / "roc_curve.png"
PR_FIGURE = OUTPUT_FIGURE_DIR / "pr_curve.png"
CALIBRATION_FIGURE = OUTPUT_FIGURE_DIR / "calibration_curve.png"
CONFUSION_FIGURE = OUTPUT_FIGURE_DIR / "confusion_matrix.png"
QBER_FIGURE = OUTPUT_FIGURE_DIR / "qber_comparison.png"
ATTACK_PROBABILITY_FIGURE = (
    OUTPUT_FIGURE_DIR / "attack_probability.png"
)
RETRY_FIGURE = OUTPUT_FIGURE_DIR / "retry_analysis.png"
MANIFEST_FILE = OUTPUT_FIGURE_DIR / "figure_manifest.json"
LOG_FILE = OUTPUT_LOG_DIR / "generate_result_graphs.log"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
SCHEMA_VERSION = 1
MASTER_SEED = 20260701

EXPECTED_FEATURE_ORDER = [
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

LABEL_CANDIDATES = (
    "label_attack",
    "actual_attack",
    "is_attack",
    "attack_label",
)
PROBABILITY_CANDIDATES = (
    "p_attack",
    "attack_probability",
    "calibrated_attack_probability",
    "calibrated_probability",
)
QBER_CANDIDATES = (
    "qber_raw",
    "raw_qber",
    "qber",
    "qber_raw_fixture",
)
NOISE_CANDIDATES = (
    "noise",
    "noise_probability",
    "configured_noise",
    "noise_estimate",
    "noise_estimate_fixture",
)
EVE_CANDIDATES = (
    "eve_fraction",
    "interception_fraction",
    "attack_fraction",
)
SCENARIO_CANDIDATES = (
    "scenario_id",
    "scenario",
    "display_name",
    "category",
)
OUTCOME_CANDIDATES = (
    "actual_outcome",
    "outcome",
    "decision",
    "reason",
)
RETRY_ATTEMPT_CANDIDATES = (
    "retry_attempts",
    "attempts",
    "attempt_count",
)
RETRY_USED_CANDIDATES = (
    "retry_used",
    "used_retry",
)

MANAGED_FIGURES = (
    ROC_FIGURE,
    PR_FIGURE,
    CALIBRATION_FIGURE,
    CONFUSION_FIGURE,
    QBER_FIGURE,
    ATTACK_PROBABILITY_FIGURE,
    RETRY_FIGURE,
)


class FigureGenerationError(RuntimeError):
    """Raised when figure generation cannot proceed safely."""


@dataclass(frozen=True)
class FigureRecord:
    """Manifest record for one expected figure."""

    name: str
    path: str
    status: str
    source_files: list[str]
    source_rows: int | None
    evidence_scope: str
    sha256: str | None = None
    bytes: int | None = None
    note: str = ""


@dataclass
class FigureRunManifest:
    """Reproducibility manifest for one graph-generation run."""

    schema_version: int = SCHEMA_VERSION
    protocol: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION
    generated_at_utc: str = ""
    project_root: str = ""
    status: str = "RUNNING"
    strict_mode: bool = False
    dpi: int = 200
    evidence_scope: str = "unknown"
    model_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    figures: list[FigureRecord] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        created = sum(item.status == "CREATED" for item in self.figures)
        skipped = sum(item.status == "SKIPPED" for item in self.figures)
        failed = sum(item.status == "FAILED" for item in self.figures)

        if failed:
            self.status = "FAILED"
        elif self.strict_mode and skipped:
            self.status = "FAILED_STRICT"
        elif created == 0:
            self.status = "FAILED_NO_FIGURES"
        elif skipped:
            self.status = "CREATED_WITH_SKIPS"
        else:
            self.status = "CREATED"

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionBundle:
    """Held-out labels and calibrated GP probabilities."""

    labels: np.ndarray
    raw_probabilities: np.ndarray
    calibrated_probabilities: np.ndarray
    table: pd.DataFrame
    source_files: tuple[Path, ...]
    evidence_scope: str
    model_id: str | None


def utc_now_iso() -> str:
    """Return a stable timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and persistent logging."""

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.generate_result_graphs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate notebook-aligned FT-QuPAP ROC, PR, calibration, "
            "confusion-matrix, QBER, attack-probability, and retry figures."
        )
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dots per inch (default: 200).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing managed figures.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail when any expected figure cannot be generated from "
            "available evidence."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Inspect input availability and existing figures without "
            "creating or overwriting images."
        ),
    )
    parser.add_argument(
        "--show-source-summary",
        action="store_true",
        help="Print discovered source tables and usable columns.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject invalid output settings."""

    if args.dpi < 72 or args.dpi > 1200:
        raise FigureGenerationError("--dpi must be between 72 and 1200.")


def ensure_directories() -> None:
    """Create output directories."""

    OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def relative_path(path: Path) -> str:
    """Return a project-relative path where possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 digest."""

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
    """Atomically write deterministic JSON."""

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


def installed_version(distribution: str) -> str | None:
    """Read an installed package version without failing."""

    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def first_existing_column(
    table: pd.DataFrame,
    candidates: Sequence[str],
) -> str | None:
    """Return the first candidate column present in a table."""

    for column in candidates:
        if column in table.columns:
            return column
    return None


def load_csv_if_available(
    path: Path,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Load a non-empty CSV or return None with a warning."""

    if not path.is_file():
        return None

    try:
        table = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", relative_path(path), exc)
        return None

    if table.empty:
        logger.warning("Source table is empty: %s", relative_path(path))
        return None

    return table


def parse_boolean_series(series: pd.Series) -> pd.Series | None:
    """Normalize common Boolean CSV values."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        return None

    return normalized.astype(bool)


def table_evidence_scope(table: pd.DataFrame) -> str:
    """Infer whether a source table is research or development evidence."""

    if "research_eligible" not in table.columns:
        if "data_origin" in table.columns:
            origins = " ".join(
                table["data_origin"].astype(str).str.lower().unique()
            )
            if "synthetic" in origins or "fixture" in origins:
                return "development_only"
        if "execution_status" in table.columns:
            statuses = " ".join(
                table["execution_status"].astype(str).str.lower().unique()
            )
            if "synthetic_fixture_not_executed" in statuses:
                return "development_only"
        return "unspecified"

    flags = parse_boolean_series(table["research_eligible"])
    if flags is None:
        return "unspecified"
    if bool(flags.all()):
        return "research_eligible"
    if bool((~flags).all()):
        return "development_only"
    return "mixed"


def metadata_evidence_scope() -> tuple[str, str | None]:
    """Read evidence scope and model ID from model metadata."""

    if not MODEL_METADATA_FILE.is_file():
        return "unspecified", None

    try:
        metadata = json.loads(
            MODEL_METADATA_FILE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return "unspecified", None

    scope = str(metadata.get("evidence_scope", "unspecified"))
    model_id = metadata.get("model_id")
    return scope, str(model_id) if model_id is not None else None


def combine_evidence_scopes(scopes: Iterable[str]) -> str:
    """Combine source evidence labels conservatively."""

    normalized = {
        str(scope)
        for scope in scopes
        if scope not in {"", "unknown", "unspecified", None}
    }

    if not normalized:
        return "unspecified"

    development_markers = {
        "development_only",
        "synthetic_demo_development_only",
    }
    research_markers = {
        "research_eligible",
        "research_eligible_session_traces",
    }

    if normalized.issubset(development_markers):
        return "development_only"
    if normalized.issubset(research_markers):
        return "research_eligible"
    return "mixed_or_partially_specified"


def evidence_footer(scope: str) -> str:
    """Return a concise figure annotation for non-final evidence."""

    if scope in {
        "development_only",
        "synthetic_demo_development_only",
    }:
        return "Development/demo evidence — not final paper results"
    if scope in {
        "mixed",
        "mixed_or_partially_specified",
        "unspecified",
    }:
        return "Evidence scope not fully established"
    return ""


def add_evidence_footer(axis: plt.Axes, scope: str) -> None:
    """Annotate figures that are not confirmed research evidence."""

    footer = evidence_footer(scope)
    if footer:
        axis.figure.text(
            0.5,
            0.01,
            footer,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_figure(
    figure: plt.Figure,
    path: Path,
    dpi: int,
    force: bool,
) -> None:
    """Save one PNG safely and close its figure."""

    if path.exists() and not force:
        plt.close(figure)
        raise FigureGenerationError(
            f"Output already exists: {relative_path(path)}. Use --force."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".png",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        figure.savefig(
            temporary_path,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": path.stem,
                "Author": PROTOCOL_NAME,
                "Software": "scripts/generate_result_graphs.py",
            },
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        plt.close(figure)


def validate_probability_array(
    values: np.ndarray,
    name: str,
) -> None:
    """Validate a probability vector."""

    if values.ndim != 1 or len(values) == 0:
        raise FigureGenerationError(
            f"{name} must be a non-empty one-dimensional array."
        )
    if not np.isfinite(values).all():
        raise FigureGenerationError(f"{name} contains NaN or infinity.")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise FigureGenerationError(
            f"{name} contains values outside [0, 1]."
        )


def normalize_labels(series: pd.Series) -> np.ndarray:
    """Convert common attack-label representations to 0/1."""

    if pd.api.types.is_bool_dtype(series):
        labels = series.astype(int).to_numpy()
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise FigureGenerationError("Attack labels contain NaN.")
        labels = numeric.astype(int).to_numpy()
    else:
        mapping = {
            "attack": 1,
            "attacker": 1,
            "malicious": 1,
            "true": 1,
            "1": 1,
            "benign": 0,
            "normal": 0,
            "valid": 0,
            "false": 0,
            "0": 0,
        }
        normalized = (
            series.astype(str).str.strip().str.lower().map(mapping)
        )
        if normalized.isna().any():
            raise FigureGenerationError(
                "Attack-label text contains unsupported values."
            )
        labels = normalized.astype(int).to_numpy()

    if set(np.unique(labels)) != {0, 1}:
        raise FigureGenerationError(
            "ROC/PR figures require both benign and attack labels."
        )
    return labels


def load_prediction_bundle(
    logger: logging.Logger,
) -> PredictionBundle | None:
    """
    Load labels/probabilities from an executed table, or recompute them from
    the exported model and independent test split.
    """

    candidate_tables = (
        DASHBOARD_RESULTS_FILE,
        DEMO_SESSION_LOGS_FILE,
    )

    for path in candidate_tables:
        table = load_csv_if_available(path, logger)
        if table is None:
            continue

        label_column = first_existing_column(table, LABEL_CANDIDATES)
        probability_column = first_existing_column(
            table,
            PROBABILITY_CANDIDATES,
        )
        if label_column is None or probability_column is None:
            continue

        candidate = table[[label_column, probability_column]].copy()
        candidate[probability_column] = pd.to_numeric(
            candidate[probability_column],
            errors="coerce",
        )
        candidate = candidate.dropna()

        if candidate.empty:
            continue

        try:
            labels = normalize_labels(candidate[label_column])
            probabilities = candidate[probability_column].to_numpy(
                dtype=float
            )
            validate_probability_array(
                probabilities,
                "table attack probabilities",
            )
        except FigureGenerationError:
            continue

        scope = table_evidence_scope(table)
        return PredictionBundle(
            labels=labels,
            raw_probabilities=probabilities.copy(),
            calibrated_probabilities=probabilities,
            table=table.loc[candidate.index].reset_index(drop=True),
            source_files=(path,),
            evidence_scope=scope,
            model_id=None,
        )

    required = (
        GP_MODEL_FILE,
        FEATURE_SCALER_FILE,
        CALIBRATION_MODEL_FILE,
        FEATURE_ORDER_FILE,
        INDEPENDENT_TEST_FILE,
    )
    if not all(path.is_file() for path in required):
        return None

    try:
        feature_order = json.loads(
            FEATURE_ORDER_FILE.read_text(encoding="utf-8")
        )
        if feature_order != EXPECTED_FEATURE_ORDER:
            raise FigureGenerationError(
                "feature_order.json does not match the FT-QuPAP schema."
            )

        table = pd.read_csv(INDEPENDENT_TEST_FILE)
        missing = set(feature_order + ["label_attack"]) - set(
            table.columns
        )
        if missing:
            raise FigureGenerationError(
                "Independent test data is missing columns: "
                f"{sorted(missing)}"
            )

        features = table[feature_order].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if features.isna().any().any():
            raise FigureGenerationError(
                "Independent test features contain missing values."
            )

        labels = normalize_labels(table["label_attack"])
        scaler = joblib.load(FEATURE_SCALER_FILE)
        model = joblib.load(GP_MODEL_FILE)
        calibrator = joblib.load(CALIBRATION_MODEL_FILE)

        if not isinstance(scaler, StandardScaler):
            raise FigureGenerationError(
                "feature_scaler.pkl is not a StandardScaler."
            )
        if not isinstance(model, GaussianProcessClassifier):
            raise FigureGenerationError(
                "gp_model.pkl is not a GaussianProcessClassifier."
            )
        if not isinstance(calibrator, IsotonicRegression):
            raise FigureGenerationError(
                "calibration_model.pkl is not an IsotonicRegression."
            )

        transformed = scaler.transform(features)
        raw = np.asarray(
            model.predict_proba(transformed)[:, 1],
            dtype=float,
        )
        calibrated = np.asarray(
            calibrator.predict(raw),
            dtype=float,
        )

        validate_probability_array(raw, "raw GP probabilities")
        validate_probability_array(
            calibrated,
            "calibrated GP probabilities",
        )

        scope, model_id = metadata_evidence_scope()

        enriched = table.copy()
        enriched["raw_p_attack"] = raw
        enriched["p_attack"] = calibrated

        return PredictionBundle(
            labels=labels,
            raw_probabilities=raw,
            calibrated_probabilities=calibrated,
            table=enriched,
            source_files=required,
            evidence_scope=scope,
            model_id=model_id,
        )

    except Exception as exc:
        logger.warning(
            "Could not create held-out prediction bundle: %s",
            exc,
        )
        return None


def load_threshold() -> float | None:
    """Read the deployed GP operational threshold."""

    if not THRESHOLD_FILE.is_file():
        return None

    try:
        payload = json.loads(
            THRESHOLD_FILE.read_text(encoding="utf-8")
        )
        value = float(payload["operational_threshold"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


def metric_from_long_table(metric_name: str) -> float | None:
    """Read one value from performance_metrics.csv."""

    table = load_csv_if_available(
        PERFORMANCE_METRICS_FILE,
        logging.getLogger("ft_qupap.generate_result_graphs"),
    )
    if table is None or not {"metric", "value"}.issubset(table.columns):
        return None

    selected = table[table["metric"].astype(str) == metric_name]
    if selected.empty:
        return None

    try:
        value = float(selected.iloc[-1]["value"])
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def make_record(
    *,
    name: str,
    path: Path,
    status: str,
    source_files: Sequence[Path],
    source_rows: int | None,
    evidence_scope: str,
    note: str = "",
) -> FigureRecord:
    """Create a figure manifest record."""

    return FigureRecord(
        name=name,
        path=relative_path(path),
        status=status,
        source_files=[relative_path(item) for item in source_files],
        source_rows=source_rows,
        evidence_scope=evidence_scope,
        sha256=sha256_file(path) if path.is_file() else None,
        bytes=path.stat().st_size if path.is_file() else None,
        note=note,
    )


def generate_roc_figure(
    bundle: PredictionBundle,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate the held-out GP ROC curve."""

    fpr, tpr, _ = roc_curve(
        bundle.labels,
        bundle.calibrated_probabilities,
    )
    auc = roc_auc_score(
        bundle.labels,
        bundle.calibrated_probabilities,
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Random")
    axis.set_title("FT-QuPAP GP ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True)
    axis.legend()
    add_evidence_footer(axis, bundle.evidence_scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(figure, ROC_FIGURE, args.dpi, args.force)

    return make_record(
        name="roc_curve",
        path=ROC_FIGURE,
        status="CREATED",
        source_files=bundle.source_files,
        source_rows=len(bundle.labels),
        evidence_scope=bundle.evidence_scope,
        note=f"Held-out calibrated ROC-AUC={auc:.9f}",
    )


def generate_pr_figure(
    bundle: PredictionBundle,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate the held-out precision-recall curve."""

    precision, recall, _ = precision_recall_curve(
        bundle.labels,
        bundle.calibrated_probabilities,
    )
    pr_auc = average_precision_score(
        bundle.labels,
        bundle.calibrated_probabilities,
    )
    prevalence = float(np.mean(bundle.labels))

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    axis.axhline(
        prevalence,
        linestyle="--",
        label=f"Attack prevalence = {prevalence:.3f}",
    )
    axis.set_title("FT-QuPAP GP Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(True)
    axis.legend()
    add_evidence_footer(axis, bundle.evidence_scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(figure, PR_FIGURE, args.dpi, args.force)

    return make_record(
        name="pr_curve",
        path=PR_FIGURE,
        status="CREATED",
        source_files=bundle.source_files,
        source_rows=len(bundle.labels),
        evidence_scope=bundle.evidence_scope,
        note=f"Held-out calibrated PR-AUC={pr_auc:.9f}",
    )


def quantile_calibration_table(
    labels: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    """Create equal-count held-out calibration bins."""

    ordered = np.argsort(probabilities)
    groups = np.array_split(ordered, min(bins, len(ordered)))
    rows: list[dict[str, Any]] = []

    for index, indices in enumerate(groups):
        if len(indices) == 0:
            continue
        rows.append(
            {
                "bin_index": index,
                "sample_count": int(len(indices)),
                "mean_predicted_probability": float(
                    np.mean(probabilities[indices])
                ),
                "observed_attack_rate": float(
                    np.mean(labels[indices])
                ),
            }
        )

    return pd.DataFrame(rows)


def calibration_source(
    bundle: PredictionBundle | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, tuple[Path, ...], str] | None:
    """Load or calculate calibration-curve points."""

    table = load_csv_if_available(CALIBRATION_RESULTS_FILE, logger)
    required = {
        "mean_predicted_probability",
        "observed_attack_rate",
    }

    if table is not None and required.issubset(table.columns):
        valid = table.copy()
        for column in required:
            valid[column] = pd.to_numeric(
                valid[column],
                errors="coerce",
            )
        valid = valid.dropna(subset=list(required))
        if not valid.empty:
            scope = (
                bundle.evidence_scope
                if bundle is not None
                else metadata_evidence_scope()[0]
            )
            return valid, (CALIBRATION_RESULTS_FILE,), scope

    if bundle is None:
        return None

    calculated = quantile_calibration_table(
        bundle.labels,
        bundle.calibrated_probabilities,
    )
    return calculated, bundle.source_files, bundle.evidence_scope


def generate_calibration_figure(
    table: pd.DataFrame,
    source_files: Sequence[Path],
    scope: str,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate held-out probability calibration plot."""

    x = table["mean_predicted_probability"].to_numpy(dtype=float)
    y = table["observed_attack_rate"].to_numpy(dtype=float)

    weights = (
        table["sample_count"].to_numpy(dtype=float)
        if "sample_count" in table.columns
        else np.ones(len(table), dtype=float)
    )
    weights = weights / np.sum(weights)
    ece = float(np.sum(weights * np.abs(x - y)))

    exported_ece = metric_from_long_table(
        "expected_calibration_error"
    )
    displayed_ece = exported_ece if exported_ece is not None else ece

    order = np.argsort(x)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        x[order],
        y[order],
        marker="o",
        label=f"Held-out calibrated GP (ECE={displayed_ece:.3f})",
    )
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect calibration",
    )
    axis.set_title("FT-QuPAP Held-Out GP Calibration Curve")
    axis.set_xlabel("Mean predicted attack probability")
    axis.set_ylabel("Observed attack frequency")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True)
    axis.legend()
    add_evidence_footer(axis, scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(
        figure,
        CALIBRATION_FIGURE,
        args.dpi,
        args.force,
    )

    return make_record(
        name="calibration_curve",
        path=CALIBRATION_FIGURE,
        status="CREATED",
        source_files=source_files,
        source_rows=len(table),
        evidence_scope=scope,
        note=f"Expected calibration error={displayed_ece:.9f}",
    )


def matrix_from_exported_table(
    logger: logging.Logger,
) -> tuple[np.ndarray, tuple[Path, ...]] | None:
    """Load the explicit 2x2 confusion matrix table."""

    table = load_csv_if_available(CONFUSION_MATRIX_TABLE, logger)
    if table is None:
        return None

    required = {"actual_label", "predicted_label", "count"}
    if not required.issubset(table.columns):
        return None

    label_order = ["benign", "attack"]
    matrix = np.zeros((2, 2), dtype=int)

    try:
        for row in table.itertuples(index=False):
            actual = str(getattr(row, "actual_label")).strip().lower()
            predicted = str(
                getattr(row, "predicted_label")
            ).strip().lower()
            count = int(getattr(row, "count"))

            if actual not in label_order or predicted not in label_order:
                return None

            matrix[
                label_order.index(actual),
                label_order.index(predicted),
            ] += count
    except (TypeError, ValueError):
        return None

    return matrix, (CONFUSION_MATRIX_TABLE,)


def confusion_source(
    bundle: PredictionBundle | None,
    threshold: float | None,
    logger: logging.Logger,
) -> tuple[np.ndarray, tuple[Path, ...], str] | None:
    """Load or calculate held-out confusion counts."""

    exported = matrix_from_exported_table(logger)
    if exported is not None:
        matrix, files = exported
        scope = (
            bundle.evidence_scope
            if bundle is not None
            else metadata_evidence_scope()[0]
        )
        return matrix, files, scope

    if bundle is None or threshold is None:
        return None

    predictions = (
        bundle.calibrated_probabilities >= threshold
    ).astype(int)
    matrix = confusion_matrix(
        bundle.labels,
        predictions,
        labels=[0, 1],
    )
    return matrix, bundle.source_files, bundle.evidence_scope


def generate_confusion_figure(
    matrix: np.ndarray,
    source_files: Sequence[Path],
    scope: str,
    threshold: float | None,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate a held-out confusion-matrix image."""

    if matrix.shape != (2, 2):
        raise FigureGenerationError(
            "Confusion matrix must have shape (2, 2)."
        )

    figure, axis = plt.subplots(figsize=(5.5, 5))
    image = axis.imshow(matrix)

    axis.set_xticks([0, 1], labels=["Benign", "Attack"])
    axis.set_yticks([0, 1], labels=["Benign", "Attack"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")

    title = "FT-QuPAP Held-Out Confusion Matrix"
    if threshold is not None:
        title += f"\nOperational threshold = {threshold:.3f}"
    axis.set_title(title)

    maximum = float(np.max(matrix)) if matrix.size else 0.0
    midpoint = maximum / 2.0

    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
                fontsize=12,
                fontweight="bold",
            )

    figure.colorbar(image, ax=axis, label="Session count")
    add_evidence_footer(axis, scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(
        figure,
        CONFUSION_FIGURE,
        args.dpi,
        args.force,
    )

    return make_record(
        name="confusion_matrix",
        path=CONFUSION_FIGURE,
        status="CREATED",
        source_files=source_files,
        source_rows=int(np.sum(matrix)),
        evidence_scope=scope,
        note=(
            "Matrix order: actual [benign, attack] x "
            "predicted [benign, attack]"
        ),
    )


def usable_numeric_table(
    path: Path,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Load one table while dropping unexecuted placeholder rows where clear."""

    table = load_csv_if_available(path, logger)
    if table is None:
        return None

    if "execution_status" in table.columns:
        status = table["execution_status"].astype(str).str.lower()
        executed_mask = ~status.str.contains(
            "synthetic_fixture_not_executed",
            na=False,
        )
        if executed_mask.any():
            table = table.loc[executed_mask].copy()

    return table if not table.empty else None


def qber_source(
    bundle: PredictionBundle | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, Path, str, str] | None:
    """
    Select a QBER table and plot mode.

    Mode 'noise_eve' reproduces notebook Cell 92.
    Mode 'scenario' provides a safe fallback when executed demo data does not
    contain the notebook experiment grid.
    """

    candidates = (
        DEMO_SESSION_LOGS_FILE,
        DASHBOARD_RESULTS_FILE,
        INDEPENDENT_TEST_FILE,
    )

    for path in candidates:
        table = usable_numeric_table(path, logger)
        if table is None:
            continue

        qber_column = first_existing_column(table, QBER_CANDIDATES)
        if qber_column is None:
            continue

        table = table.copy()
        table[qber_column] = pd.to_numeric(
            table[qber_column],
            errors="coerce",
        )
        table = table.dropna(subset=[qber_column])
        if table.empty:
            continue

        noise_column = first_existing_column(table, NOISE_CANDIDATES)
        eve_column = first_existing_column(table, EVE_CANDIDATES)

        if noise_column is not None and eve_column is not None:
            table[noise_column] = pd.to_numeric(
                table[noise_column],
                errors="coerce",
            )
            table[eve_column] = pd.to_numeric(
                table[eve_column],
                errors="coerce",
            )
            valid = table.dropna(
                subset=[noise_column, eve_column, qber_column]
            )
            if (
                not valid.empty
                and valid[noise_column].nunique() >= 2
            ):
                valid.attrs["qber_column"] = qber_column
                valid.attrs["noise_column"] = noise_column
                valid.attrs["eve_column"] = eve_column
                return (
                    valid,
                    path,
                    table_evidence_scope(valid),
                    "noise_eve",
                )

        scenario_column = first_existing_column(
            table,
            SCENARIO_CANDIDATES,
        )
        if scenario_column is not None:
            table.attrs["qber_column"] = qber_column
            table.attrs["scenario_column"] = scenario_column
            return (
                table,
                path,
                table_evidence_scope(table),
                "scenario",
            )

    if bundle is not None and "qber_raw" in bundle.table.columns:
        table = bundle.table.copy()
        scenario_column = first_existing_column(
            table,
            SCENARIO_CANDIDATES,
        )
        if scenario_column is not None:
            table.attrs["qber_column"] = "qber_raw"
            table.attrs["scenario_column"] = scenario_column
            return (
                table,
                bundle.source_files[-1],
                bundle.evidence_scope,
                "scenario",
            )

    return None


def generate_qber_figure(
    table: pd.DataFrame,
    source_file: Path,
    scope: str,
    mode: str,
    threshold: float | None,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate the raw-QBER result figure."""

    figure, axis = plt.subplots(figsize=(8, 5))

    if mode == "noise_eve":
        qber_column = str(table.attrs["qber_column"])
        noise_column = str(table.attrs["noise_column"])
        eve_column = str(table.attrs["eve_column"])

        grouped = (
            table.groupby(
                [noise_column, eve_column],
                as_index=False,
            )[qber_column]
            .mean()
            .sort_values([eve_column, noise_column])
        )

        for eve_fraction in sorted(grouped[eve_column].unique()):
            subset = grouped[
                grouped[eve_column] == eve_fraction
            ]
            axis.plot(
                subset[noise_column],
                subset[qber_column],
                marker="o",
                label=f"Eve fraction = {eve_fraction:.2f}",
            )

        axis.set_title(
            "FT-QuPAP Raw QBER versus Noise and Eve Fraction"
        )
        axis.set_xlabel("Configured / observed noise probability")
        axis.set_ylabel("Mean raw QBER")
        source_rows = len(table)

    else:
        qber_column = str(table.attrs["qber_column"])
        scenario_column = str(table.attrs["scenario_column"])

        grouped = (
            table.groupby(scenario_column, as_index=False)
            .agg(
                mean_qber=(qber_column, "mean"),
                session_count=(qber_column, "count"),
            )
            .sort_values("mean_qber")
        )

        axis.barh(
            grouped[scenario_column].astype(str),
            grouped["mean_qber"],
        )
        axis.set_title("FT-QuPAP Mean Raw QBER by Scenario")
        axis.set_xlabel("Mean raw QBER")
        axis.set_ylabel("Scenario")
        source_rows = int(grouped["session_count"].sum())

    fixed_qber = 0.11
    if THRESHOLD_FILE.is_file():
        try:
            threshold_payload = json.loads(
                THRESHOLD_FILE.read_text(encoding="utf-8")
            )
            fixed_qber = float(
                threshold_payload.get(
                    "fixed_qber_threshold",
                    fixed_qber,
                )
            )
        except Exception:
            pass

    axis.axhline(
        fixed_qber,
        linestyle="--",
        label=f"Fixed-QBER policy = {fixed_qber:.3f}",
    ) if mode == "noise_eve" else axis.axvline(
        fixed_qber,
        linestyle="--",
        label=f"Fixed-QBER policy = {fixed_qber:.3f}",
    )

    axis.grid(True)
    axis.legend()
    add_evidence_footer(axis, scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(
        figure,
        QBER_FIGURE,
        args.dpi,
        args.force,
    )

    return make_record(
        name="qber_comparison",
        path=QBER_FIGURE,
        status="CREATED",
        source_files=[source_file],
        source_rows=source_rows,
        evidence_scope=scope,
        note=(
            "Notebook-style noise/Eve grid"
            if mode == "noise_eve"
            else "Scenario-grouped QBER fallback"
        ),
    )


def attack_probability_source(
    bundle: PredictionBundle | None,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, tuple[Path, ...], str, str] | None:
    """Select a table for GP attack-probability visualization."""

    for path in (
        DEMO_SESSION_LOGS_FILE,
        DASHBOARD_RESULTS_FILE,
    ):
        table = usable_numeric_table(path, logger)
        if table is None:
            continue

        probability_column = first_existing_column(
            table,
            PROBABILITY_CANDIDATES,
        )
        if probability_column is None:
            continue

        table = table.copy()
        table[probability_column] = pd.to_numeric(
            table[probability_column],
            errors="coerce",
        )
        table = table.dropna(subset=[probability_column])
        if table.empty:
            continue

        noise_column = first_existing_column(table, NOISE_CANDIDATES)
        eve_column = first_existing_column(table, EVE_CANDIDATES)

        if noise_column is not None and eve_column is not None:
            table[noise_column] = pd.to_numeric(
                table[noise_column],
                errors="coerce",
            )
            table[eve_column] = pd.to_numeric(
                table[eve_column],
                errors="coerce",
            )
            valid = table.dropna(
                subset=[
                    probability_column,
                    noise_column,
                    eve_column,
                ]
            )
            if (
                not valid.empty
                and valid[noise_column].nunique() >= 2
            ):
                valid.attrs["probability_column"] = (
                    probability_column
                )
                valid.attrs["noise_column"] = noise_column
                valid.attrs["eve_column"] = eve_column
                return (
                    valid,
                    (path,),
                    table_evidence_scope(valid),
                    "noise_eve",
                )

        scenario_column = first_existing_column(
            table,
            SCENARIO_CANDIDATES,
        )
        if scenario_column is not None:
            table.attrs["probability_column"] = probability_column
            table.attrs["scenario_column"] = scenario_column
            return (
                table,
                (path,),
                table_evidence_scope(table),
                "scenario",
            )

    if bundle is not None:
        table = bundle.table.copy()
        table["p_attack"] = bundle.calibrated_probabilities

        scenario_column = first_existing_column(
            table,
            SCENARIO_CANDIDATES,
        )
        if scenario_column is not None:
            table.attrs["probability_column"] = "p_attack"
            table.attrs["scenario_column"] = scenario_column
            return (
                table,
                bundle.source_files,
                bundle.evidence_scope,
                "scenario",
            )

        table["actual_class"] = np.where(
            bundle.labels == 1,
            "Attack",
            "Benign",
        )
        table.attrs["probability_column"] = "p_attack"
        table.attrs["scenario_column"] = "actual_class"
        return (
            table,
            bundle.source_files,
            bundle.evidence_scope,
            "scenario",
        )

    return None


def generate_attack_probability_figure(
    table: pd.DataFrame,
    source_files: Sequence[Path],
    scope: str,
    mode: str,
    threshold: float | None,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate calibrated GP P(attack) visualization."""

    figure, axis = plt.subplots(figsize=(8, 5))

    if mode == "noise_eve":
        probability_column = str(
            table.attrs["probability_column"]
        )
        noise_column = str(table.attrs["noise_column"])
        eve_column = str(table.attrs["eve_column"])

        grouped = (
            table.groupby(
                [noise_column, eve_column],
                as_index=False,
            )[probability_column]
            .mean()
            .sort_values([eve_column, noise_column])
        )

        for eve_fraction in sorted(grouped[eve_column].unique()):
            subset = grouped[
                grouped[eve_column] == eve_fraction
            ]
            axis.plot(
                subset[noise_column],
                subset[probability_column],
                marker="o",
                label=f"P(attack), Eve = {eve_fraction:.2f}",
            )

        axis.set_xlabel("Configured / observed noise probability")
        axis.set_ylabel("Mean predicted attack probability")
        axis.set_title(
            "Calibrated GP Attack Probability across Conditions"
        )
        source_rows = len(table)

    else:
        probability_column = str(
            table.attrs["probability_column"]
        )
        scenario_column = str(table.attrs["scenario_column"])

        grouped = (
            table.groupby(scenario_column, as_index=False)
            .agg(
                mean_p_attack=(probability_column, "mean"),
                session_count=(probability_column, "count"),
            )
            .sort_values("mean_p_attack")
        )

        axis.barh(
            grouped[scenario_column].astype(str),
            grouped["mean_p_attack"],
        )
        axis.set_xlabel("Mean calibrated P(attack)")
        axis.set_ylabel("Scenario / actual class")
        axis.set_title("FT-QuPAP Calibrated Attack Probability")
        source_rows = int(grouped["session_count"].sum())

    if threshold is not None:
        if mode == "noise_eve":
            axis.axhline(
                threshold,
                linestyle="--",
                label=f"Operational threshold = {threshold:.3f}",
            )
        else:
            axis.axvline(
                threshold,
                linestyle="--",
                label=f"Operational threshold = {threshold:.3f}",
            )

    axis.set_xlim(
        left=0.0,
        right=1.0,
    ) if mode == "scenario" else axis.set_ylim(
        bottom=0.0,
        top=1.0,
    )
    axis.grid(True)
    axis.legend()
    add_evidence_footer(axis, scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(
        figure,
        ATTACK_PROBABILITY_FIGURE,
        args.dpi,
        args.force,
    )

    return make_record(
        name="attack_probability",
        path=ATTACK_PROBABILITY_FIGURE,
        status="CREATED",
        source_files=source_files,
        source_rows=source_rows,
        evidence_scope=scope,
        note=(
            "Notebook-style noise/Eve grid"
            if mode == "noise_eve"
            else "Scenario/class-grouped probability fallback"
        ),
    )


def retry_source(
    logger: logging.Logger,
) -> tuple[pd.DataFrame, Path, str] | None:
    """Load real retry results or executed demo rows containing retry data."""

    for path in (
        RETRY_RESULTS_FILE,
        DEMO_SESSION_LOGS_FILE,
        DASHBOARD_RESULTS_FILE,
    ):
        table = usable_numeric_table(path, logger)
        if table is None:
            continue

        attempts_column = first_existing_column(
            table,
            RETRY_ATTEMPT_CANDIDATES,
        )
        retry_used_column = first_existing_column(
            table,
            RETRY_USED_CANDIDATES,
        )

        if attempts_column is None and retry_used_column is None:
            continue

        table = table.copy()

        if attempts_column is not None:
            table[attempts_column] = pd.to_numeric(
                table[attempts_column],
                errors="coerce",
            )

        if retry_used_column is not None:
            parsed = parse_boolean_series(table[retry_used_column])
            if parsed is not None:
                table["_retry_used_normalized"] = parsed
            else:
                table["_retry_used_normalized"] = False
        elif attempts_column is not None:
            table["_retry_used_normalized"] = (
                table[attempts_column].fillna(1) > 1
            )

        if table["_retry_used_normalized"].any() or path == RETRY_RESULTS_FILE:
            table.attrs["attempts_column"] = attempts_column
            return table, path, table_evidence_scope(table)

    return None


def generate_retry_figure(
    table: pd.DataFrame,
    source_file: Path,
    scope: str,
    args: argparse.Namespace,
) -> FigureRecord:
    """Generate retry-policy outcome counts."""

    outcome_column = first_existing_column(
        table,
        OUTCOME_CANDIDATES,
    )

    if outcome_column is None:
        if "accepted" in table.columns:
            accepted = parse_boolean_series(table["accepted"])
            if accepted is not None:
                table = table.copy()
                table["_retry_outcome"] = np.where(
                    accepted,
                    "accepted_after_retry",
                    "rejected_after_retry",
                )
                outcome_column = "_retry_outcome"

    if outcome_column is None:
        table = table.copy()
        table["_retry_outcome"] = "retry_executed"
        outcome_column = "_retry_outcome"

    outcome_text = (
        table[outcome_column]
        .fillna("unspecified")
        .astype(str)
        .str.strip()
        .replace("", "unspecified")
    )
    counts = outcome_text.value_counts().sort_values()

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(counts.index.astype(str), counts.to_numpy())
    axis.set_title("FT-QuPAP Retry Policy Outcomes")
    axis.set_xlabel("Session count")
    axis.set_ylabel("Final outcome / reason")
    axis.grid(True)

    attempts_column = table.attrs.get("attempts_column")
    note_parts = []
    if attempts_column is not None:
        attempts = pd.to_numeric(
            table[str(attempts_column)],
            errors="coerce",
        ).dropna()
        if not attempts.empty:
            mean_attempts = float(attempts.mean())
            maximum_attempts = int(attempts.max())
            note_parts.append(
                f"mean attempts={mean_attempts:.3f}"
            )
            note_parts.append(
                f"maximum attempts={maximum_attempts}"
            )
            axis.text(
                0.98,
                0.02,
                (
                    f"Mean attempts: {mean_attempts:.2f}\n"
                    f"Maximum attempts: {maximum_attempts}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="bottom",
            )

    add_evidence_footer(axis, scope)
    figure.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_figure(
        figure,
        RETRY_FIGURE,
        args.dpi,
        args.force,
    )

    return make_record(
        name="retry_analysis",
        path=RETRY_FIGURE,
        status="CREATED",
        source_files=[source_file],
        source_rows=len(table),
        evidence_scope=scope,
        note="; ".join(note_parts) or "Retry outcome counts",
    )


def record_skipped(
    manifest: FigureRunManifest,
    name: str,
    path: Path,
    note: str,
) -> None:
    """Add a skipped figure to the manifest."""

    manifest.figures.append(
        make_record(
            name=name,
            path=path,
            status="SKIPPED",
            source_files=[],
            source_rows=None,
            evidence_scope="unspecified",
            note=note,
        )
    )
    manifest.warnings.append(f"{name}: {note}")


def record_failed(
    manifest: FigureRunManifest,
    name: str,
    path: Path,
    source_files: Sequence[Path],
    scope: str,
    exc: Exception,
) -> None:
    """Add a failed figure record."""

    message = f"{type(exc).__name__}: {exc}"
    manifest.figures.append(
        make_record(
            name=name,
            path=path,
            status="FAILED",
            source_files=source_files,
            source_rows=None,
            evidence_scope=scope,
            note=message,
        )
    )
    manifest.warnings.append(f"{name}: {message}")


def show_source_summary(logger: logging.Logger) -> None:
    """Print all recognized source files and their columns."""

    paths = (
        INDEPENDENT_TEST_FILE,
        PERFORMANCE_METRICS_FILE,
        CONFUSION_MATRIX_TABLE,
        CALIBRATION_RESULTS_FILE,
        THRESHOLD_ANALYSIS_FILE,
        RETRY_RESULTS_FILE,
        BASELINE_COMPARISON_FILE,
        DEMO_SESSION_LOGS_FILE,
        DASHBOARD_RESULTS_FILE,
    )

    print("FT-QuPAP result-source summary")
    print("=" * 72)

    for path in paths:
        if not path.is_file():
            print(f"MISSING  {relative_path(path)}")
            continue

        try:
            table = pd.read_csv(path, nrows=5)
            print(
                f"FOUND    {relative_path(path)} | "
                f"columns={list(table.columns)}"
            )
        except Exception as exc:
            print(
                f"INVALID  {relative_path(path)} | {exc}"
            )

    for path in MODEL_BUNDLE_FILES_FOR_SUMMARY:
        print(
            f"{'FOUND' if path.is_file() else 'MISSING':8} "
            f"{relative_path(path)}"
        )

    logger.info("Displayed source summary.")


MODEL_BUNDLE_FILES_FOR_SUMMARY = (
    GP_MODEL_FILE,
    FEATURE_SCALER_FILE,
    CALIBRATION_MODEL_FILE,
    THRESHOLD_FILE,
    FEATURE_ORDER_FILE,
    MODEL_METADATA_FILE,
)


def validate_existing_figures(
    manifest: FigureRunManifest,
) -> None:
    """Validate existing expected figures without generating them."""

    for path in MANAGED_FIGURES:
        if not path.is_file():
            manifest.figures.append(
                make_record(
                    name=path.stem,
                    path=path,
                    status="SKIPPED",
                    source_files=[],
                    source_rows=None,
                    evidence_scope="unspecified",
                    note="Figure does not exist.",
                )
            )
            continue

        if path.stat().st_size <= 0:
            manifest.figures.append(
                make_record(
                    name=path.stem,
                    path=path,
                    status="FAILED",
                    source_files=[],
                    source_rows=None,
                    evidence_scope="unspecified",
                    note="Figure file is empty.",
                )
            )
            continue

        manifest.figures.append(
            make_record(
                name=path.stem,
                path=path,
                status="CREATED",
                source_files=[],
                source_rows=None,
                evidence_scope="unspecified",
                note="Existing figure validated by size and SHA-256.",
            )
        )


def generate_all_figures(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> FigureRunManifest:
    """Generate every supported figure without inventing missing evidence."""

    ensure_directories()

    metadata_scope, metadata_model_id = metadata_evidence_scope()

    manifest = FigureRunManifest(
        generated_at_utc=utc_now_iso(),
        project_root=str(PROJECT_ROOT),
        strict_mode=args.strict,
        dpi=args.dpi,
        evidence_scope=metadata_scope,
        model_id=metadata_model_id,
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": installed_version("numpy"),
            "pandas": installed_version("pandas"),
            "matplotlib": installed_version("matplotlib"),
            "scikit_learn": installed_version("scikit-learn"),
            "joblib": installed_version("joblib"),
        },
    )

    if args.validate_only:
        validate_existing_figures(manifest)
        manifest.finalize()
        return manifest

    bundle = load_prediction_bundle(logger)
    if bundle is not None:
        manifest.model_id = bundle.model_id or manifest.model_id

        for name, path, generator in (
            ("roc_curve", ROC_FIGURE, generate_roc_figure),
            ("pr_curve", PR_FIGURE, generate_pr_figure),
        ):
            try:
                record = generator(bundle, args)
                manifest.figures.append(record)
                logger.info("Created %s", relative_path(path))
            except Exception as exc:
                record_failed(
                    manifest,
                    name,
                    path,
                    bundle.source_files,
                    bundle.evidence_scope,
                    exc,
                )
                logger.error("%s generation failed: %s", name, exc)
    else:
        record_skipped(
            manifest,
            "roc_curve",
            ROC_FIGURE,
            "No labels plus calibrated attack probabilities were available.",
        )
        record_skipped(
            manifest,
            "pr_curve",
            PR_FIGURE,
            "No labels plus calibrated attack probabilities were available.",
        )

    calibration = calibration_source(bundle, logger)
    if calibration is None:
        record_skipped(
            manifest,
            "calibration_curve",
            CALIBRATION_FIGURE,
            "No calibration bins or held-out predictions were available.",
        )
    else:
        table, source_files, scope = calibration
        try:
            manifest.figures.append(
                generate_calibration_figure(
                    table,
                    source_files,
                    scope,
                    args,
                )
            )
            logger.info(
                "Created %s",
                relative_path(CALIBRATION_FIGURE),
            )
        except Exception as exc:
            record_failed(
                manifest,
                "calibration_curve",
                CALIBRATION_FIGURE,
                source_files,
                scope,
                exc,
            )

    threshold = load_threshold()

    confusion = confusion_source(bundle, threshold, logger)
    if confusion is None:
        record_skipped(
            manifest,
            "confusion_matrix",
            CONFUSION_FIGURE,
            "No exported confusion table or thresholded held-out predictions.",
        )
    else:
        matrix, source_files, scope = confusion
        try:
            manifest.figures.append(
                generate_confusion_figure(
                    matrix,
                    source_files,
                    scope,
                    threshold,
                    args,
                )
            )
            logger.info(
                "Created %s",
                relative_path(CONFUSION_FIGURE),
            )
        except Exception as exc:
            record_failed(
                manifest,
                "confusion_matrix",
                CONFUSION_FIGURE,
                source_files,
                scope,
                exc,
            )

    qber = qber_source(bundle, logger)
    if qber is None:
        record_skipped(
            manifest,
            "qber_comparison",
            QBER_FIGURE,
            "No executed QBER table or held-out QBER scenario data.",
        )
    else:
        table, source_file, scope, mode = qber
        try:
            manifest.figures.append(
                generate_qber_figure(
                    table,
                    source_file,
                    scope,
                    mode,
                    threshold,
                    args,
                )
            )
            logger.info(
                "Created %s",
                relative_path(QBER_FIGURE),
            )
        except Exception as exc:
            record_failed(
                manifest,
                "qber_comparison",
                QBER_FIGURE,
                [source_file],
                scope,
                exc,
            )

    attack_probability = attack_probability_source(bundle, logger)
    if attack_probability is None:
        record_skipped(
            manifest,
            "attack_probability",
            ATTACK_PROBABILITY_FIGURE,
            "No calibrated P(attack) values were available.",
        )
    else:
        table, source_files, scope, mode = attack_probability
        try:
            manifest.figures.append(
                generate_attack_probability_figure(
                    table,
                    source_files,
                    scope,
                    mode,
                    threshold,
                    args,
                )
            )
            logger.info(
                "Created %s",
                relative_path(ATTACK_PROBABILITY_FIGURE),
            )
        except Exception as exc:
            record_failed(
                manifest,
                "attack_probability",
                ATTACK_PROBABILITY_FIGURE,
                source_files,
                scope,
                exc,
            )

    retry = retry_source(logger)
    if retry is None:
        record_skipped(
            manifest,
            "retry_analysis",
            RETRY_FIGURE,
            "No executed retry-policy rows were available.",
        )
    else:
        table, source_file, scope = retry
        try:
            manifest.figures.append(
                generate_retry_figure(
                    table,
                    source_file,
                    scope,
                    args,
                )
            )
            logger.info(
                "Created %s",
                relative_path(RETRY_FIGURE),
            )
        except Exception as exc:
            record_failed(
                manifest,
                "retry_analysis",
                RETRY_FIGURE,
                [source_file],
                scope,
                exc,
            )

    manifest.evidence_scope = combine_evidence_scopes(
        item.evidence_scope
        for item in manifest.figures
        if item.status == "CREATED"
    )
    manifest.finalize()
    return manifest


def print_summary(manifest: FigureRunManifest) -> None:
    """Print a compact terminal summary."""

    created = [
        item for item in manifest.figures if item.status == "CREATED"
    ]
    skipped = [
        item for item in manifest.figures if item.status == "SKIPPED"
    ]
    failed = [
        item for item in manifest.figures if item.status == "FAILED"
    ]

    print("\n" + "=" * 76)
    print("FT-QuPAP RESULT FIGURE SUMMARY")
    print("=" * 76)
    print(f"Status:          {manifest.status}")
    print(f"Evidence scope:  {manifest.evidence_scope}")
    print(f"Created:         {len(created)}")
    print(f"Skipped:         {len(skipped)}")
    print(f"Failed:          {len(failed)}")

    for item in manifest.figures:
        print(
            f"{item.status:8} {item.path}"
            + (f" | {item.note}" if item.note else "")
        )

    print(f"Manifest:        {relative_path(MANIFEST_FILE)}")
    print("=" * 76)


def main() -> int:
    """Command-line entry point."""

    logger = configure_logging()

    try:
        args = parse_arguments()
        validate_arguments(args)
        ensure_directories()

        if args.show_source_summary:
            show_source_summary(logger)

        manifest = generate_all_figures(args, logger)
        atomic_write_json(
            MANIFEST_FILE,
            manifest.to_dictionary(),
        )
        print_summary(manifest)

        return 0 if manifest.status in {
            "CREATED",
            "CREATED_WITH_SKIPS",
        } else 1

    except FigureGenerationError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        logger.error("Figure generation interrupted by user.")
        print("\nFigure generation interrupted.", file=sys.stderr)
        return 130

    except Exception:
        logger.exception("Unexpected result-graph generation failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
