#!/usr/bin/env python3
"""
Generate reproducible FT-QuPAP development and capstone-demo datasets.

Notebook/flowchart alignment:
- Cells 12-16: observable GP feature schema and labelled benign/attack traces.
- Cell 17: stratified 65% / 17.5% / 17.5% train-validation-test split.
- Cells 69-76 and the project scenario modules: controlled demo cases for
  ideal, benign-noisy, retry, Eve, replay, credential, ciphertext, tag,
  excessive-loss, and uncorrectable-error paths.

Created files:
    data/raw/normal_sessions.csv
    data/raw/noisy_sessions.csv
    data/raw/attack_sessions.csv
    data/processed/training_features.csv
    data/processed/validation_features.csv
    data/processed/independent_test_features.csv
    data/demo/controlled_scenarios.csv
    data/demo/demo_session_logs.csv
    data/demo/dashboard_results.csv
    data/demo/demo_data_manifest.json

Important research boundary:
The generated rows are deterministic synthetic fixtures for software
integration, GP pipeline development, dashboard testing, and capstone-day
rehearsal. They are NOT final experimental evidence and must not be used to
replace the session-level traces and independent multi-seed results produced
by the complete FT-QuPAP notebook/protocol engine.

Run from the project root:
    python scripts/generate_demo_data.py

Useful options:
    python scripts/generate_demo_data.py --samples-per-scenario 250
    python scripts/generate_demo_data.py --seed 20260701 --force
    python scripts/generate_demo_data.py --validate-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
DATABASE_DIR = PROJECT_ROOT / "database"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

NORMAL_SESSIONS_FILE = RAW_DIR / "normal_sessions.csv"
NOISY_SESSIONS_FILE = RAW_DIR / "noisy_sessions.csv"
ATTACK_SESSIONS_FILE = RAW_DIR / "attack_sessions.csv"
TRAINING_FEATURES_FILE = PROCESSED_DIR / "training_features.csv"
VALIDATION_FEATURES_FILE = PROCESSED_DIR / "validation_features.csv"
INDEPENDENT_TEST_FEATURES_FILE = (
    PROCESSED_DIR / "independent_test_features.csv"
)
CONTROLLED_SCENARIOS_FILE = DEMO_DIR / "controlled_scenarios.csv"
DEMO_SESSION_LOGS_FILE = DEMO_DIR / "demo_session_logs.csv"
DASHBOARD_RESULTS_FILE = DEMO_DIR / "dashboard_results.csv"
MANIFEST_FILE = DEMO_DIR / "demo_data_manifest.json"
SUBSCRIBERS_FILE = DATABASE_DIR / "subscribers.json"
LOG_FILE = LOG_DIR / "generate_demo_data.log"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = "research-simulator-v5-1-large-ml-operational-threshold"
DEFAULT_SEED = 20260701
DEFAULT_SAMPLES_PER_SCENARIO = 100
DEFAULT_PSEUDONYM_ID = "PID-6G-UE-0001"
SCHEMA_VERSION = 1

CONTEXT_CATEGORIES = ("urban", "suburban", "rural")
BENIGN_SCENARIOS = (
    "benign_clean",
    "benign_noisy",
    "benign_lossy",
)
ATTACK_SCENARIOS = (
    "attack_intercept_resend",
    "attack_burst_injection",
    "attack_selective_loss",
)
ALL_GP_SCENARIOS = BENIGN_SCENARIOS + ATTACK_SCENARIOS

FEATURE_COLUMNS = [
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

METADATA_COLUMNS = [
    "session_id",
    "split",
    "scenario",
    "scenario_severity",
    "context",
    "data_origin",
    "research_eligible",
]

MANAGED_FILES = (
    NORMAL_SESSIONS_FILE,
    NOISY_SESSIONS_FILE,
    ATTACK_SESSIONS_FILE,
    TRAINING_FEATURES_FILE,
    VALIDATION_FEATURES_FILE,
    INDEPENDENT_TEST_FEATURES_FILE,
    CONTROLLED_SCENARIOS_FILE,
    DEMO_SESSION_LOGS_FILE,
    DASHBOARD_RESULTS_FILE,
    MANIFEST_FILE,
)


class DemoDataError(RuntimeError):
    """Raised when FT-QuPAP demo data cannot be generated or validated."""


@dataclass(frozen=True)
class DemoScenario:
    """One controlled capstone demonstration case."""

    scenario_id: str
    scenario_module: str
    display_name: str
    category: str
    context: str
    bit_flip_prob: float
    phase_flip_prob: float
    depolarizing_prob: float
    loss_prob: float
    eve_fraction: float
    eve_mode: str
    tamper_type: str
    allow_retry: bool
    expected_outcome: str
    expected_primary_reason: str
    description: str


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in stable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and file logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.generate_demo_data")
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
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic FT-QuPAP raw traces, processed GP splits, "
            "and controlled capstone-demo fixtures."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=DEFAULT_SAMPLES_PER_SCENARIO,
        help=(
            "Rows generated for each of the six GP scenarios "
            f"(default: {DEFAULT_SAMPLES_PER_SCENARIO})."
        ),
    )
    parser.add_argument(
        "--pseudonym-id",
        default=None,
        help=(
            "Pseudonymous demo subscriber. By default, the first active "
            "subscriber is read from database/subscribers.json."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite managed data files if they already exist.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing generated files without modifying them.",
    )
    return parser.parse_args()


def clipped(value: float, lower: float, upper: float) -> float:
    """Clip a floating-point value to a closed interval."""

    return float(np.clip(value, lower, upper))


def encode_context(context: str) -> dict[str, float]:
    """Return the notebook-aligned one-hot context encoding."""

    if context not in CONTEXT_CATEGORIES:
        raise DemoDataError(f"Unknown channel context: {context}")

    return {
        f"ctx_{name}": float(context == name)
        for name in CONTEXT_CATEGORIES
    }


def context_base_noise(context: str) -> float:
    """Return the notebook's trusted nominal receiver-noise estimate."""

    base_noise = {
        "urban": 0.012,
        "suburban": 0.008,
        "rural": 0.005,
    }

    try:
        return base_noise[context]
    except KeyError as exc:
        raise DemoDataError(
            f"Unknown channel context: {context}"
        ) from exc


def deterministic_session_id(
    seed: int,
    scenario: str,
    row_index: int,
) -> str:
    """Create a stable non-secret session fixture identifier."""

    material = f"{PROTOCOL_NAME}|{seed}|{scenario}|{row_index}".encode(
        "utf-8"
    )
    suffix = hashlib.sha256(material).hexdigest()[:16].upper()
    return f"DEMO-{suffix}"


def generate_rich_gp_trace(
    rng: np.random.Generator,
    scenario: str,
    context: str,
) -> dict[str, Any]:
    """
    Create one labelled synthetic FT-QuPAP observable trace.

    This function follows the distributions used by notebook Cells 13-15.
    Scenario and severity are simulator metadata only and are never included
    in FEATURE_COLUMNS.
    """

    if scenario not in ALL_GP_SCENARIOS:
        raise DemoDataError(f"Unsupported GP scenario: {scenario}")

    base_noise = context_base_noise(context)
    severity = 0.0
    label_attack = 0

    if scenario == "benign_clean":
        noise_estimate = clipped(
            rng.normal(base_noise, 0.002), 0.000, 0.030
        )
        qber_raw = clipped(
            rng.normal(0.20 * noise_estimate, 0.004),
            0.000,
            0.035,
        )
        mean_syndrome_weight = clipped(
            rng.normal(7.0 * noise_estimate, 0.08),
            0.000,
            0.70,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.0, 0.8)),
            0.0,
            2.0,
        )
        correction_failure_rate = clipped(
            rng.normal(0.002, 0.003), 0.000, 0.020
        )
        loss_rate = clipped(
            rng.normal(0.005, 0.003), 0.000, 0.025
        )

    elif scenario == "benign_noisy":
        noise_estimate = clipped(
            base_noise + rng.uniform(0.020, 0.075),
            0.010,
            0.100,
        )
        qber_raw = clipped(
            0.30 * noise_estimate + rng.normal(0.015, 0.012),
            0.005,
            0.115,
        )
        mean_syndrome_weight = clipped(
            8.0 * noise_estimate + rng.normal(0.25, 0.15),
            0.050,
            1.60,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.3, 1.0)),
            1.0,
            3.0,
        )
        correction_failure_rate = clipped(
            0.60 * noise_estimate + rng.normal(0.010, 0.010),
            0.000,
            0.100,
        )
        loss_rate = clipped(
            rng.normal(0.012, 0.007), 0.000, 0.050
        )

    elif scenario == "benign_lossy":
        noise_estimate = clipped(
            base_noise + rng.uniform(0.005, 0.030),
            0.005,
            0.060,
        )
        qber_raw = clipped(
            0.25 * noise_estimate + rng.normal(0.012, 0.008),
            0.002,
            0.075,
        )
        mean_syndrome_weight = clipped(
            6.0 * noise_estimate + rng.normal(0.20, 0.12),
            0.000,
            1.30,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.2, 0.9)),
            0.0,
            3.0,
        )
        correction_failure_rate = clipped(
            rng.normal(0.020, 0.015), 0.000, 0.090
        )
        loss_rate = clipped(
            rng.normal(0.090, 0.030), 0.030, 0.145
        )

    elif scenario == "attack_intercept_resend":
        label_attack = 1
        severity = float(rng.uniform(0.20, 1.00))
        noise_estimate = clipped(
            rng.normal(base_noise + 0.010, 0.008),
            0.000,
            0.080,
        )
        qber_raw = clipped(
            0.045
            + 0.220 * severity
            + 0.20 * noise_estimate
            + rng.normal(0.0, 0.018),
            0.025,
            0.350,
        )
        mean_syndrome_weight = clipped(
            0.60
            + 2.60 * severity
            + 5.0 * noise_estimate
            + rng.normal(0.0, 0.25),
            0.300,
            5.50,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.5, 1.8)),
            1.0,
            6.0,
        )
        correction_failure_rate = clipped(
            0.040
            + 0.550 * severity
            + rng.normal(0.0, 0.035),
            0.020,
            1.000,
        )
        loss_rate = clipped(
            rng.normal(0.015 + 0.040 * severity, 0.015),
            0.000,
            0.180,
        )

    elif scenario == "attack_burst_injection":
        label_attack = 1
        severity = float(rng.uniform(0.20, 1.00))
        noise_estimate = clipped(
            rng.normal(base_noise + 0.020, 0.012),
            0.000,
            0.100,
        )
        qber_raw = clipped(
            0.030
            + 0.170 * severity
            + 0.35 * noise_estimate
            + rng.normal(0.0, 0.020),
            0.020,
            0.300,
        )
        mean_syndrome_weight = clipped(
            0.50
            + 3.20 * severity
            + 7.0 * noise_estimate
            + rng.normal(0.0, 0.30),
            0.300,
            6.00,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.6, 2.0)),
            1.0,
            6.0,
        )
        correction_failure_rate = clipped(
            0.050
            + 0.620 * severity
            + 0.80 * noise_estimate
            + rng.normal(0.0, 0.040),
            0.020,
            1.000,
        )
        loss_rate = clipped(
            rng.normal(0.020 + 0.020 * severity, 0.012),
            0.000,
            0.120,
        )

    else:  # attack_selective_loss
        label_attack = 1
        severity = float(rng.uniform(0.20, 1.00))
        noise_estimate = clipped(
            rng.normal(base_noise + 0.006, 0.005),
            0.000,
            0.060,
        )
        qber_raw = clipped(
            0.010
            + 0.080 * severity
            + 0.20 * noise_estimate
            + rng.normal(0.0, 0.012),
            0.005,
            0.180,
        )
        mean_syndrome_weight = clipped(
            0.25
            + 1.20 * severity
            + 4.0 * noise_estimate
            + rng.normal(0.0, 0.20),
            0.100,
            3.50,
        )
        max_syndrome_weight = clipped(
            np.ceil(mean_syndrome_weight + rng.uniform(0.3, 1.4)),
            1.0,
            5.0,
        )
        correction_failure_rate = clipped(
            0.030
            + 0.350 * severity
            + rng.normal(0.0, 0.030),
            0.010,
            0.800,
        )
        loss_rate = clipped(
            0.080
            + 0.420 * severity
            + rng.normal(0.0, 0.040),
            0.060,
            0.650,
        )

    row: dict[str, Any] = {
        "qber_raw": qber_raw,
        "mean_syndrome_weight": mean_syndrome_weight,
        "max_syndrome_weight": max_syndrome_weight,
        "correction_failure_rate": correction_failure_rate,
        "loss_rate": loss_rate,
        "noise_estimate": noise_estimate,
        "label_attack": label_attack,
        "scenario": scenario,
        "scenario_severity": severity,
        "context": context,
    }
    row.update(encode_context(context))
    return row


def generate_gp_dataset(
    seed: int,
    samples_per_scenario: int,
) -> pd.DataFrame:
    """Generate the notebook-aligned six-scenario synthetic dataset."""

    if samples_per_scenario < 4:
        raise DemoDataError(
            "--samples-per-scenario must be at least 4 so stratified "
            "train/validation/test splitting remains valid."
        )

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for scenario in ALL_GP_SCENARIOS:
        for row_index in range(samples_per_scenario):
            context = str(rng.choice(CONTEXT_CATEGORIES))
            row = generate_rich_gp_trace(rng, scenario, context)
            row.update(
                {
                    "session_id": deterministic_session_id(
                        seed, scenario, row_index
                    ),
                    "split": "unassigned",
                    "data_origin": "synthetic_notebook_aligned_fixture",
                    "research_eligible": False,
                    "source_seed": seed,
                }
            )
            rows.append(row)

    dataset = pd.DataFrame(rows)
    dataset = dataset.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    validate_gp_dataframe(dataset, require_split=False)
    return dataset


def split_gp_dataset(
    dataset: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match notebook Cell 17's 65%/17.5%/17.5% stratified split."""

    train, temporary = train_test_split(
        dataset,
        test_size=0.35,
        stratify=dataset["label_attack"],
        random_state=seed,
    )
    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label_attack"],
        random_state=seed,
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "independent_test"

    ordered_columns = (
        METADATA_COLUMNS
        + ["source_seed"]
        + FEATURE_COLUMNS
        + ["label_attack"]
    )

    train = train[ordered_columns].reset_index(drop=True)
    validation = validation[ordered_columns].reset_index(drop=True)
    test = test[ordered_columns].reset_index(drop=True)

    validate_split_disjointness(train, validation, test)
    validate_gp_dataframe(train, require_split=True)
    validate_gp_dataframe(validation, require_split=True)
    validate_gp_dataframe(test, require_split=True)
    return train, validation, test


def validate_gp_dataframe(
    dataframe: pd.DataFrame,
    *,
    require_split: bool,
) -> None:
    """Validate feature completeness, bounds, labels, and one-hot context."""

    required_columns = set(FEATURE_COLUMNS) | {
        "session_id",
        "scenario",
        "scenario_severity",
        "context",
        "label_attack",
        "data_origin",
        "research_eligible",
        "source_seed",
    }
    if require_split:
        required_columns.add("split")

    missing = required_columns - set(dataframe.columns)
    if missing:
        raise DemoDataError(
            f"Generated dataset is missing columns: {sorted(missing)}"
        )

    if dataframe.empty:
        raise DemoDataError("Generated GP dataset is empty.")

    if dataframe["session_id"].duplicated().any():
        raise DemoDataError("Duplicate synthetic session identifiers found.")

    feature_matrix = dataframe[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(feature_matrix).all():
        raise DemoDataError("Feature dataset contains NaN or infinite values.")

    if not dataframe["label_attack"].isin([0, 1]).all():
        raise DemoDataError("label_attack must contain only 0 or 1.")

    if not dataframe["scenario"].isin(ALL_GP_SCENARIOS).all():
        raise DemoDataError("Unknown scenario found in GP dataset.")

    if not dataframe["context"].isin(CONTEXT_CATEGORIES).all():
        raise DemoDataError("Unknown context found in GP dataset.")

    one_hot_sum = dataframe[
        ["ctx_urban", "ctx_suburban", "ctx_rural"]
    ].sum(axis=1)
    if not np.allclose(one_hot_sum.to_numpy(dtype=float), 1.0):
        raise DemoDataError("Context columns are not valid one-hot values.")

    bounded_zero_one = (
        "qber_raw",
        "correction_failure_rate",
        "loss_rate",
        "noise_estimate",
    )
    for column in bounded_zero_one:
        if not dataframe[column].between(0.0, 1.0).all():
            raise DemoDataError(f"{column} is outside [0, 1].")

    if (dataframe["mean_syndrome_weight"] < 0.0).any():
        raise DemoDataError("mean_syndrome_weight cannot be negative.")
    if (dataframe["max_syndrome_weight"] < 0.0).any():
        raise DemoDataError("max_syndrome_weight cannot be negative.")

    leakage_columns = {
        "scenario",
        "scenario_severity",
        "label_attack",
        "eve_fraction",
    } & set(FEATURE_COLUMNS)
    if leakage_columns:
        raise DemoDataError(
            f"Feature leakage detected: {sorted(leakage_columns)}"
        )


def validate_split_disjointness(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure no session fixture appears in more than one split."""

    train_ids = set(train["session_id"])
    validation_ids = set(validation["session_id"])
    test_ids = set(test["session_id"])

    if train_ids & validation_ids:
        raise DemoDataError("Training and validation splits overlap.")
    if train_ids & test_ids:
        raise DemoDataError("Training and independent-test splits overlap.")
    if validation_ids & test_ids:
        raise DemoDataError(
            "Validation and independent-test splits overlap."
        )


def build_controlled_scenarios() -> list[DemoScenario]:
    """Define the project-tree scenario modules and flowchart outcomes."""

    return [
        DemoScenario(
            "normal_session",
            "scenarios.normal_session",
            "Normal Authentication",
            "benign",
            "urban",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "none",
            "none",
            True,
            "accept",
            "authentication_successful",
            "Valid credentials, fresh request, recoverable payload, matching tag, and low attack risk.",
        ),
        DemoScenario(
            "benign_noisy_session",
            "scenarios.benign_noisy_session",
            "Benign Noisy Channel",
            "benign",
            "urban",
            0.001,
            0.001,
            0.0005,
            0.001,
            0.0,
            "none",
            "none",
            True,
            "accept",
            "bounded_noise_corrected",
            "Steane CSS corrects bounded ordinary channel disturbance without treating every QBER rise as an attack.",
        ),
        DemoScenario(
            "accept_after_retry",
            "scenarios.accept_after_retry",
            "Accept After Retry",
            "benign_retry",
            "rural",
            0.003,
            0.003,
            0.0015,
            0.010,
            0.0,
            "none",
            "transient_payload_recovery_failure",
            True,
            "accept_after_retry",
            "fresh_nonce_new_session",
            "A low-risk gray-zone or recoverable channel failure triggers one fresh-session retry.",
        ),
        DemoScenario(
            "intercept_resend_attack",
            "scenarios.intercept_resend_attack",
            "Intercept-Measure-Resend Attack",
            "quantum_attack",
            "urban",
            0.010,
            0.010,
            0.005,
            0.005,
            0.50,
            "intercept_resend",
            "eve_interception",
            False,
            "reject",
            "qber_or_gp_attack_risk",
            "Eve measures and resends a substantial fraction of CSS blocks, increasing disturbance evidence.",
        ),
        DemoScenario(
            "partial_eavesdropping",
            "scenarios.partial_eavesdropping",
            "Partial Eavesdropping",
            "quantum_attack",
            "suburban",
            0.005,
            0.005,
            0.0025,
            0.005,
            0.35,
            "intercept_resend",
            "partial_eve",
            False,
            "reject",
            "calibrated_gp_or_deterministic_check",
            "Partial Eve activity is evaluated through raw QBER, syndromes, correction failures, loss, and calibrated GP risk.",
        ),
        DemoScenario(
            "full_eavesdropping",
            "scenarios.full_eavesdropping",
            "Full Eavesdropping",
            "quantum_attack",
            "urban",
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            "intercept_resend",
            "full_eve",
            False,
            "reject",
            "excessive_disturbance",
            "Eve intercepts all transmitted blocks; high disturbance should force rejection.",
        ),
        DemoScenario(
            "replay_attack",
            "scenarios.replay_attack",
            "Replay Attack",
            "classical_attack",
            "urban",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "none",
            "reused_nonce",
            False,
            "reject",
            "nonce_replay_detected",
            "The same pseudonym, timestamp, and nonce request is submitted again and rejected before quantum processing.",
        ),
        DemoScenario(
            "forged_server_signature",
            "scenarios.forged_server_signature",
            "Forged Server Signature",
            "credential_attack",
            "urban",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "none",
            "forged_mldsa_signature",
            False,
            "reject",
            "invalid_server_credential",
            "The mobile station rejects a server package that fails ML-DSA verification.",
        ),
        DemoScenario(
            "modified_authentication_request",
            "scenarios.modified_authentication_request",
            "Modified Authentication Request",
            "classical_attack",
            "urban",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "none",
            "modified_request_transcript",
            False,
            "reject",
            "freshness_or_transcript_binding_failure",
            "Modification of IDp, timestamp, nonce, or context breaks request validation or transcript binding.",
        ),
        DemoScenario(
            "tampered_mlkem_ciphertext",
            "scenarios.tampered_mlkem_ciphertext",
            "Tampered ML-KEM Ciphertext",
            "pqc_attack",
            "suburban",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            "none",
            "tampered_mlkem_ciphertext",
            False,
            "reject",
            "ciphertext_validation_or_key_mismatch",
            "Malformed or modified ML-KEM ciphertext is rejected or leads to keys that cannot validate the protected session.",
        ),
        DemoScenario(
            "forged_kmac_tag",
            "scenarios.forged_kmac_tag",
            "Forged KMAC Tag",
            "authentication_attack",
            "urban",
            0.001,
            0.001,
            0.0005,
            0.001,
            0.0,
            "none",
            "forged_authentication_tag",
            False,
            "reject",
            "kmac_tag_mismatch",
            "The decoded 128-bit tag does not match the AS-computed transcript-bound KMAC value.",
        ),
        DemoScenario(
            "excessive_loss",
            "scenarios.excessive_loss",
            "Excessive Quantum Loss",
            "channel_failure",
            "rural",
            0.005,
            0.005,
            0.0025,
            0.250,
            0.0,
            "none",
            "excessive_erasure",
            False,
            "reject",
            "loss_policy_exceeded",
            "Loss exceeds the deterministic 0.15 acceptance policy or leaves insufficient declared check evidence.",
        ),
        DemoScenario(
            "uncorrectable_quantum_error",
            "scenarios.uncorrectable_quantum_error",
            "Uncorrectable Quantum Error",
            "channel_failure",
            "urban",
            0.040,
            0.040,
            0.020,
            0.010,
            0.0,
            "none",
            "multi_error_steane_block",
            False,
            "reject",
            "required_payload_unrecoverable",
            "A required Steane block contains errors outside the simulator's bounded correction model.",
        ),
    ]


def load_pseudonym_id(requested: str | None, logger: logging.Logger) -> str:
    """Resolve the pseudonymous demo subscriber without exposing raw IMSI."""

    if requested:
        return requested.strip()

    if not SUBSCRIBERS_FILE.exists():
        logger.warning(
            "%s was not found; using default pseudonym %s. Run "
            "scripts/initialize_database.py first for registered data.",
            SUBSCRIBERS_FILE.relative_to(PROJECT_ROOT),
            DEFAULT_PSEUDONYM_ID,
        )
        return DEFAULT_PSEUDONYM_ID

    try:
        payload = json.loads(SUBSCRIBERS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDataError(
            f"Could not read {SUBSCRIBERS_FILE}: {exc}"
        ) from exc

    candidates: list[Mapping[str, Any]] = []
    if isinstance(payload, list):
        candidates = [row for row in payload if isinstance(row, Mapping)]
    elif isinstance(payload, Mapping):
        for key in ("subscribers", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = [
                    row for row in value if isinstance(row, Mapping)
                ]
                break
            if isinstance(value, Mapping):
                candidates = [
                    row for row in value.values()
                    if isinstance(row, Mapping)
                ]
                break
        if not candidates and any(
            key in payload
            for key in ("pseudonym_id", "pseudonymous_id", "idp")
        ):
            candidates = [payload]
        if not candidates:
            candidates = [
                row for row in payload.values()
                if isinstance(row, Mapping)
            ]

    for subscriber in candidates:
        status = str(
            subscriber.get(
                "subscriber_status",
                subscriber.get("status", "active"),
            )
        ).lower()
        if status != "active":
            continue
        for key in ("pseudonym_id", "pseudonymous_id", "idp"):
            value = subscriber.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    logger.warning(
        "No active pseudonym found in %s; using %s.",
        SUBSCRIBERS_FILE.relative_to(PROJECT_ROOT),
        DEFAULT_PSEUDONYM_ID,
    )
    return DEFAULT_PSEUDONYM_ID


def scenario_fixture_profile(scenario_id: str) -> str:
    """Map a demo case to a notebook-aligned observable profile."""

    mapping = {
        "normal_session": "benign_clean",
        "benign_noisy_session": "benign_noisy",
        "accept_after_retry": "benign_noisy",
        "intercept_resend_attack": "attack_intercept_resend",
        "partial_eavesdropping": "attack_intercept_resend",
        "full_eavesdropping": "attack_intercept_resend",
        "replay_attack": "benign_clean",
        "forged_server_signature": "benign_clean",
        "modified_authentication_request": "benign_clean",
        "tampered_mlkem_ciphertext": "benign_clean",
        "forged_kmac_tag": "benign_clean",
        "excessive_loss": "attack_selective_loss",
        "uncorrectable_quantum_error": "attack_burst_injection",
    }
    return mapping[scenario_id]


def build_demo_frames(
    scenarios: Sequence[DemoScenario],
    seed: int,
    pseudonym_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create controlled-scenario, synthetic-log, and dashboard fixtures."""

    generated_at = utc_now_iso()
    controlled_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    dashboard_rows: list[dict[str, Any]] = []

    for display_order, scenario in enumerate(scenarios, start=1):
        controlled_row = asdict(scenario)
        controlled_row.update(
            {
                "display_order": display_order,
                "data_origin": "controlled_demo_configuration",
                "research_eligible": False,
            }
        )
        controlled_rows.append(controlled_row)

        fixture_rng = np.random.default_rng(
            np.random.SeedSequence([seed, display_order])
        )
        profile = scenario_fixture_profile(scenario.scenario_id)
        trace = generate_rich_gp_trace(
            fixture_rng,
            profile,
            scenario.context,
        )

        # Make the purpose of selected deterministic demo paths explicit.
        if scenario.scenario_id == "accept_after_retry":
            trace["qber_raw"] = min(float(trace["qber_raw"]), 0.10)
            trace["loss_rate"] = min(float(trace["loss_rate"]), 0.10)
        elif scenario.scenario_id == "full_eavesdropping":
            trace["qber_raw"] = max(float(trace["qber_raw"]), 0.24)
            trace["correction_failure_rate"] = max(
                float(trace["correction_failure_rate"]), 0.55
            )
        elif scenario.scenario_id == "excessive_loss":
            trace["loss_rate"] = max(float(trace["loss_rate"]), 0.25)
        elif scenario.scenario_id == "uncorrectable_quantum_error":
            trace["correction_failure_rate"] = max(
                float(trace["correction_failure_rate"]), 0.60
            )

        session_id = deterministic_session_id(
            seed,
            f"controlled:{scenario.scenario_id}",
            display_order,
        )
        retry_expected = scenario.expected_outcome == "accept_after_retry"

        log_rows.append(
            {
                "demo_case_id": f"CASE-{display_order:02d}",
                "session_id": session_id,
                "scenario_id": scenario.scenario_id,
                "pseudonym_id": pseudonym_id,
                "context": scenario.context,
                "category": scenario.category,
                "fixture_profile": profile,
                "qber_raw_fixture": trace["qber_raw"],
                "mean_syndrome_weight_fixture": trace[
                    "mean_syndrome_weight"
                ],
                "max_syndrome_weight_fixture": trace[
                    "max_syndrome_weight"
                ],
                "correction_failure_rate_fixture": trace[
                    "correction_failure_rate"
                ],
                "loss_rate_fixture": trace["loss_rate"],
                "noise_estimate_fixture": trace["noise_estimate"],
                "expected_outcome": scenario.expected_outcome,
                "expected_reason": scenario.expected_primary_reason,
                "retry_expected": retry_expected,
                "execution_status": "synthetic_fixture_not_executed",
                "actual_outcome": "",
                "actual_reason": "",
                "p_attack": np.nan,
                "uncertainty": np.nan,
                "research_eligible": False,
                "generated_at_utc": generated_at,
            }
        )

        dashboard_rows.append(
            {
                "display_order": display_order,
                "scenario_id": scenario.scenario_id,
                "display_name": scenario.display_name,
                "category": scenario.category,
                "context": scenario.context,
                "expected_status": scenario.expected_outcome,
                "expected_reason": scenario.expected_primary_reason,
                "live_status": "not_run",
                "latest_session_id": session_id,
                "qber_raw_fixture": trace["qber_raw"],
                "loss_rate_fixture": trace["loss_rate"],
                "retry_expected": retry_expected,
                "research_eligible": False,
                "data_origin": "synthetic_dashboard_fixture",
            }
        )

    controlled = pd.DataFrame(controlled_rows)
    logs = pd.DataFrame(log_rows)
    dashboard = pd.DataFrame(dashboard_rows)
    validate_demo_frames(controlled, logs, dashboard)
    return controlled, logs, dashboard


def validate_demo_frames(
    controlled: pd.DataFrame,
    logs: pd.DataFrame,
    dashboard: pd.DataFrame,
) -> None:
    """Validate one-to-one scenario coverage and fixture safety labels."""

    expected_ids = {scenario.scenario_id for scenario in build_controlled_scenarios()}

    for name, dataframe in (
        ("controlled_scenarios", controlled),
        ("demo_session_logs", logs),
        ("dashboard_results", dashboard),
    ):
        if dataframe.empty:
            raise DemoDataError(f"{name} is empty.")
        if set(dataframe["scenario_id"]) != expected_ids:
            raise DemoDataError(
                f"{name} does not cover exactly the controlled scenarios."
            )
        if dataframe["scenario_id"].duplicated().any():
            raise DemoDataError(f"{name} contains duplicate scenario IDs.")
        if not (dataframe["research_eligible"] == False).all():  # noqa: E712
            raise DemoDataError(
                f"{name} contains a fixture incorrectly marked as research eligible."
            )

    allowed_outcomes = {"accept", "accept_after_retry", "reject"}
    if not controlled["expected_outcome"].isin(allowed_outcomes).all():
        raise DemoDataError("Unsupported controlled-scenario outcome.")


def ensure_overwrite_allowed(force: bool) -> None:
    """Protect existing generated evidence from accidental replacement."""

    existing = [path for path in MANAGED_FILES if path.exists()]
    if existing and not force:
        rendered = "\n  - ".join(
            str(path.relative_to(PROJECT_ROOT)) for path in existing
        )
        raise DemoDataError(
            "Managed output files already exist. Use --force to overwrite:\n"
            f"  - {rendered}"
        )


def atomic_write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    """Atomically write a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            dataframe.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_generated_files(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    controlled: pd.DataFrame,
    logs: pd.DataFrame,
    dashboard: pd.DataFrame,
    *,
    seed: int,
    samples_per_scenario: int,
    pseudonym_id: str,
    logger: logging.Logger,
) -> None:
    """Write raw, processed, demo, and manifest artifacts."""

    raw_order = (
        [
            "session_id",
            "scenario",
            "scenario_severity",
            "context",
            "data_origin",
            "research_eligible",
            "source_seed",
        ]
        + FEATURE_COLUMNS
        + ["label_attack"]
    )

    normal = dataset[
        dataset["scenario"] == "benign_clean"
    ][raw_order].reset_index(drop=True)
    noisy = dataset[
        dataset["scenario"].isin(("benign_noisy", "benign_lossy"))
    ][raw_order].reset_index(drop=True)
    attacks = dataset[
        dataset["scenario"].isin(ATTACK_SCENARIOS)
    ][raw_order].reset_index(drop=True)

    file_frames = {
        NORMAL_SESSIONS_FILE: normal,
        NOISY_SESSIONS_FILE: noisy,
        ATTACK_SESSIONS_FILE: attacks,
        TRAINING_FEATURES_FILE: train,
        VALIDATION_FEATURES_FILE: validation,
        INDEPENDENT_TEST_FEATURES_FILE: test,
        CONTROLLED_SCENARIOS_FILE: controlled,
        DEMO_SESSION_LOGS_FILE: logs,
        DASHBOARD_RESULTS_FILE: dashboard,
    }

    for path, dataframe in file_frames.items():
        atomic_write_csv(path, dataframe)
        logger.info(
            "Wrote %s (%d rows).",
            path.relative_to(PROJECT_ROOT),
            len(dataframe),
        )

    checksums = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in file_frames
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": utc_now_iso(),
        "seed": seed,
        "samples_per_scenario": samples_per_scenario,
        "pseudonym_id": pseudonym_id,
        "data_origin": "synthetic_notebook_aligned_fixture",
        "research_eligible": False,
        "research_boundary": (
            "These files support development, testing, and capstone demo "
            "rehearsal. Final paper metrics must come from the clean, "
            "executed complete protocol notebook and independent evaluation "
            "seeds."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "hidden_simulator_fields_excluded_from_features": [
            "scenario",
            "scenario_severity",
            "label_attack",
            "eve_fraction",
        ],
        "contexts": list(CONTEXT_CATEGORIES),
        "gp_scenarios": list(ALL_GP_SCENARIOS),
        "split_strategy": {
            "train_fraction": 0.65,
            "validation_fraction": 0.175,
            "independent_test_fraction": 0.175,
            "stratified_by": "label_attack",
        },
        "row_counts": {
            str(path.relative_to(PROJECT_ROOT)): int(len(dataframe))
            for path, dataframe in file_frames.items()
        },
        "class_counts": {
            "all": {
                str(key): int(value)
                for key, value in dataset["label_attack"]
                .value_counts()
                .sort_index()
                .items()
            },
            "train": {
                str(key): int(value)
                for key, value in train["label_attack"]
                .value_counts()
                .sort_index()
                .items()
            },
            "validation": {
                str(key): int(value)
                for key, value in validation["label_attack"]
                .value_counts()
                .sort_index()
                .items()
            },
            "independent_test": {
                str(key): int(value)
                for key, value in test["label_attack"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "artifact_sha256": checksums,
    }
    atomic_write_json(MANIFEST_FILE, manifest)
    logger.info("Wrote %s.", MANIFEST_FILE.relative_to(PROJECT_ROOT))


def read_existing_csv(path: Path) -> pd.DataFrame:
    """Read one generated CSV with a clear error."""

    if not path.exists():
        raise DemoDataError(
            f"Required generated file is missing: {path.relative_to(PROJECT_ROOT)}"
        )
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DemoDataError(
            f"Could not read {path.relative_to(PROJECT_ROOT)}: {exc}"
        ) from exc


def validate_existing_files(logger: logging.Logger) -> None:
    """Validate all current generated artifacts and checksum manifest."""

    normal = read_existing_csv(NORMAL_SESSIONS_FILE)
    noisy = read_existing_csv(NOISY_SESSIONS_FILE)
    attacks = read_existing_csv(ATTACK_SESSIONS_FILE)
    train = read_existing_csv(TRAINING_FEATURES_FILE)
    validation = read_existing_csv(VALIDATION_FEATURES_FILE)
    test = read_existing_csv(INDEPENDENT_TEST_FEATURES_FILE)
    controlled = read_existing_csv(CONTROLLED_SCENARIOS_FILE)
    logs = read_existing_csv(DEMO_SESSION_LOGS_FILE)
    dashboard = read_existing_csv(DASHBOARD_RESULTS_FILE)

    raw_combined = pd.concat([normal, noisy, attacks], ignore_index=True)
    validate_gp_dataframe(raw_combined, require_split=False)
    validate_gp_dataframe(train, require_split=True)
    validate_gp_dataframe(validation, require_split=True)
    validate_gp_dataframe(test, require_split=True)
    validate_split_disjointness(train, validation, test)
    validate_demo_frames(controlled, logs, dashboard)

    if not MANIFEST_FILE.exists():
        raise DemoDataError(
            f"Missing manifest: {MANIFEST_FILE.relative_to(PROJECT_ROOT)}"
        )

    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoDataError(f"Invalid manifest: {exc}") from exc

    if manifest.get("research_eligible") is not False:
        raise DemoDataError(
            "Manifest must mark synthetic demo data as research_eligible=false."
        )

    checksums = manifest.get("artifact_sha256", {})
    if not isinstance(checksums, Mapping):
        raise DemoDataError("Manifest artifact_sha256 must be an object.")

    for relative_name, expected_checksum in checksums.items():
        path = PROJECT_ROOT / relative_name
        if not path.exists():
            raise DemoDataError(f"Manifest artifact is missing: {relative_name}")
        actual_checksum = sha256_file(path)
        if actual_checksum != expected_checksum:
            raise DemoDataError(
                f"Checksum mismatch for {relative_name}: generated file was modified."
            )

    logger.info("All FT-QuPAP demo-data artifacts are valid.")


def print_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print the generated split sizes and next workflow step."""

    print("\nFT-QuPAP demo data is ready.")
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Independent-test rows: {len(test)}")
    print("Controlled demo cases: 13")
    print("Next: python scripts/export_gp_model.py")


def main() -> int:
    args = parse_arguments()
    logger = configure_logging()

    try:
        logger.info("FT-QuPAP demo-data generation started.")
        logger.info("Project root: %s", PROJECT_ROOT)

        if args.validate_only:
            validate_existing_files(logger)
            print("FT-QuPAP generated data validation passed.")
            return 0

        ensure_overwrite_allowed(args.force)

        pseudonym_id = load_pseudonym_id(args.pseudonym_id, logger)
        dataset = generate_gp_dataset(
            seed=args.seed,
            samples_per_scenario=args.samples_per_scenario,
        )
        train, validation, test = split_gp_dataset(dataset, args.seed)

        scenarios = build_controlled_scenarios()
        controlled, logs, dashboard = build_demo_frames(
            scenarios,
            seed=args.seed,
            pseudonym_id=pseudonym_id,
        )

        write_generated_files(
            dataset,
            train,
            validation,
            test,
            controlled,
            logs,
            dashboard,
            seed=args.seed,
            samples_per_scenario=args.samples_per_scenario,
            pseudonym_id=pseudonym_id,
            logger=logger,
        )

        validate_existing_files(logger)
        logger.info("FT-QuPAP demo-data generation completed successfully.")
        print_summary(train, validation, test)
        return 0

    except DemoDataError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Demo-data generation cancelled by user.")
        return 130
    except Exception:
        logger.exception("Unexpected demo-data generation failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
