#!/usr/bin/env python3
"""
Execute the controlled FT-QuPAP capstone demonstration scenarios.

This script is an orchestration layer. It does not reimplement ML-DSA,
ML-KEM, KDF, KMAC, Steane CSS, QBER calculation, deterministic verification,
Gaussian Process inference, or retry policy. Those operations remain inside
src/ and the scenario modules.

Flowchart/notebook alignment
----------------------------
The runner demonstrates the complete online path:

    request
    -> freshness and replay verification
    -> ML-DSA-authenticated server package
    -> ML-KEM encapsulation/decapsulation
    -> transcript-bound K_auth and K_ctrl
    -> 128-bit KMAC authentication tag
    -> 128 payload blocks + 32 independent check blocks
    -> Steane [[7,1,3]] encoding
    -> noisy/untrusted quantum channel
    -> raw QBER and syndrome processing
    -> tag verification and deterministic checks
    -> calibrated GP P(attack)
    -> ACCEPT / ACCEPT AFTER RETRY / REJECT

Scenario module contract
------------------------
Each module listed in data/demo/controlled_scenarios.csv should expose one of:

    def run_scenario(
        protocol_engine,
        scenario_config,
        seed,
        run_id,
        persist=False,
    ) -> Mapping[str, Any]:
        ...

or:

    class Scenario:
        def __init__(self, protocol_engine, scenario_config, seed, ...):
            ...
        def run(self) -> Mapping[str, Any]:
            ...

The preferred return format is the protocol-engine session dictionary:

    {
        "decision": {
            "accepted": bool,
            "reason": str,
            "deterministic_pass": bool,
            "deterministic_reasons": list[str],
            "p_attack": float | None,
            "uncertainty": float | None,
        },
        "qber_raw": float,
        "loss_rate": float,
        "observed_check_blocks": int,
        "features": {...},
        "retry_attempts": int,
        "retry_used": bool,
        "attempt_history": [...],
        "timings": {...},
        ...
    }

The runner stores only non-secret diagnostics. Private keys, shared secrets,
K_auth, K_ctrl, signatures, ciphertexts, and raw authentication tags are
redacted before JSON persistence.

Default command
---------------
    python scripts/run_demo_scenarios.py

Useful commands
---------------
    python scripts/run_demo_scenarios.py --list-scenarios
    python scripts/run_demo_scenarios.py --scenario normal_session
    python scripts/run_demo_scenarios.py --scenario normal_session,replay_attack
    python scripts/run_demo_scenarios.py --repeat 3
    python scripts/run_demo_scenarios.py --dry-run
    python scripts/run_demo_scenarios.py --hardware auto
    python scripts/run_demo_scenarios.py --strict

Generated files
---------------
    data/demo/demo_session_logs.csv
    data/demo/dashboard_results.csv
    data/results/retry_results.csv
    database/demo_sessions.json
    outputs/reports/demo_scenario_run.json
    outputs/logs/run_demo_scenarios.log

Research boundary
-----------------
Controlled capstone scenarios are deterministic demonstration evidence. They
are not a substitute for the notebook's disjoint-seed training, calibration,
held-out testing, and independent multi-seed experimental evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import hashlib
import hmac
import importlib
import inspect
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DATABASE_DIR = PROJECT_ROOT / "database"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

CONTROLLED_SCENARIOS_FILE = DATA_DEMO_DIR / "controlled_scenarios.csv"
DEMO_SESSION_LOGS_FILE = DATA_DEMO_DIR / "demo_session_logs.csv"
DASHBOARD_RESULTS_FILE = DATA_DEMO_DIR / "dashboard_results.csv"
RETRY_RESULTS_FILE = DATA_RESULTS_DIR / "retry_results.csv"
DEMO_SESSIONS_DATABASE = DATABASE_DIR / "demo_sessions.json"
RUN_REPORT_FILE = OUTPUT_REPORT_DIR / "demo_scenario_run.json"
LOG_FILE = OUTPUT_LOG_DIR / "run_demo_scenarios.log"

MODEL_VALIDATOR = PROJECT_ROOT / "scripts" / "validate_model_files.py"

MODEL_BUNDLE_FILES = (
    MODEL_DIR / "gp_model.pkl",
    MODEL_DIR / "feature_scaler.pkl",
    MODEL_DIR / "calibration_model.pkl",
    MODEL_DIR / "threshold.json",
    MODEL_DIR / "feature_order.json",
    MODEL_DIR / "model_metadata.json",
)

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
SCHEMA_VERSION = 1
MASTER_SEED = 20260701
DEFAULT_HISTORY_LIMIT = 500

OUTCOME_ACCEPT = "accept"
OUTCOME_ACCEPT_AFTER_RETRY = "accept_after_retry"
OUTCOME_REJECT = "reject"
VALID_EXPECTED_OUTCOMES = {
    OUTCOME_ACCEPT,
    OUTCOME_ACCEPT_AFTER_RETRY,
    OUTCOME_REJECT,
}

SENSITIVE_KEY_PARTS = (
    "private_key",
    "secret_key",
    "shared_secret",
    "session_secret",
    "key_material",
    "k_auth",
    "k_ctrl",
    "ciphertext",
    "signature",
    "tag_ms",
    "received_tag",
    "expected_tag",
    "authentication_tag",
    "raw_identity",
    "subscriber_identity",
    "imsi",
)

SAFE_SENSITIVE_NAME_EXCEPTIONS = {
    "tag_recovered",
    "tag_match",
    "tag_verified",
    "signature_valid",
    "credential_valid",
    "ciphertext_valid",
    "shared_secret_match",
    "session_secret_match",
}

CSV_FIELDS = (
    "run_id",
    "record_id",
    "executed_at_utc",
    "protocol",
    "protocol_version",
    "scenario_id",
    "scenario_module",
    "display_name",
    "category",
    "context",
    "scenario_seed",
    "repetition",
    "expected_outcome",
    "actual_outcome",
    "expectation_met",
    "accepted",
    "reason",
    "deterministic_pass",
    "deterministic_reasons",
    "qber_raw",
    "qber_mismatches",
    "qber_observed",
    "observed_check_blocks",
    "required_check_blocks",
    "loss_rate",
    "p_attack",
    "uncertainty",
    "gp_attack_threshold",
    "tag_recovered",
    "retry_attempts",
    "retry_used",
    "physical_qubits",
    "logical_payload_blocks",
    "logical_check_blocks",
    "end_to_end_seconds",
    "total_retry_seconds",
    "execution_status",
    "execution_error",
    "hardware_status",
    "data_origin",
    "research_eligible",
)

DASHBOARD_FIELDS = (
    "run_id",
    "executed_at_utc",
    "scenario_id",
    "display_name",
    "category",
    "context",
    "expected_outcome",
    "actual_outcome",
    "expectation_met",
    "accepted",
    "reason",
    "qber_raw",
    "loss_rate",
    "p_attack",
    "uncertainty",
    "deterministic_pass",
    "tag_recovered",
    "retry_attempts",
    "retry_used",
    "physical_qubits",
    "end_to_end_seconds",
    "hardware_status",
    "execution_status",
)

RETRY_FIELDS = (
    "run_id",
    "record_id",
    "executed_at_utc",
    "scenario_id",
    "scenario_seed",
    "accepted",
    "actual_outcome",
    "reason",
    "retry_attempts",
    "retry_used",
    "qber_raw",
    "loss_rate",
    "p_attack",
    "tag_recovered",
    "total_retry_seconds",
    "expectation_met",
)


class DemoRunnerError(RuntimeError):
    """Raised when controlled demonstration execution cannot continue."""


@dataclass(frozen=True)
class ScenarioDefinition:
    """One flowchart-aligned controlled demonstration scenario."""

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
    display_order: int = 0
    enabled: bool = True

    def to_config(self) -> dict[str, Any]:
        """Return the scenario configuration passed to its module."""

        return asdict(self)


@dataclass
class ScenarioExecution:
    """Normalized non-secret result of one scenario invocation."""

    run_id: str
    record_id: str
    executed_at_utc: str
    protocol: str
    protocol_version: str
    scenario_id: str
    scenario_module: str
    display_name: str
    category: str
    context: str
    scenario_seed: int
    repetition: int
    expected_outcome: str
    actual_outcome: str = "error"
    expectation_met: bool = False
    accepted: bool = False
    reason: str = "scenario_not_executed"
    deterministic_pass: bool = False
    deterministic_reasons: list[str] = field(default_factory=list)
    qber_raw: float | None = None
    qber_mismatches: int | None = None
    qber_observed: int | None = None
    observed_check_blocks: int | None = None
    required_check_blocks: int | None = None
    loss_rate: float | None = None
    p_attack: float | None = None
    uncertainty: float | None = None
    gp_attack_threshold: float | None = None
    tag_recovered: bool | None = None
    retry_attempts: int = 1
    retry_used: bool = False
    physical_qubits: int | None = None
    logical_payload_blocks: int | None = None
    logical_check_blocks: int | None = None
    end_to_end_seconds: float | None = None
    total_retry_seconds: float | None = None
    execution_status: str = "ERROR"
    execution_error: str = ""
    hardware_status: str = "OFF"
    data_origin: str = "executed_protocol_scenario"
    research_eligible: bool = False
    safe_details: dict[str, Any] = field(default_factory=dict)

    def csv_row(self) -> dict[str, Any]:
        """Return a flat row for CSV storage."""

        payload = asdict(self)
        payload.pop("safe_details", None)
        payload["deterministic_reasons"] = "|".join(
            self.deterministic_reasons
        )
        return payload


@dataclass
class DemoRunReport:
    """Complete report for one invocation of the scenario runner."""

    schema_version: int
    protocol: str
    protocol_version: str
    run_id: str
    started_at_utc: str
    finished_at_utc: str = ""
    master_seed: int = MASTER_SEED
    selected_scenarios: list[str] = field(default_factory=list)
    repeat: int = 1
    dry_run: bool = False
    strict: bool = False
    status: str = "RUNNING"
    warnings: list[str] = field(default_factory=list)
    executions: list[ScenarioExecution] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Calculate run-level outcome counts and status."""

        self.finished_at_utc = utc_now_iso()

        total = len(self.executions)
        errors = sum(
            row.execution_status == "ERROR"
            for row in self.executions
        )
        mismatches = sum(
            row.execution_status == "COMPLETED"
            and not row.expectation_met
            for row in self.executions
        )
        accepted = sum(row.accepted for row in self.executions)
        retried = sum(row.retry_used for row in self.executions)
        rejected = sum(
            row.execution_status == "COMPLETED" and not row.accepted
            for row in self.executions
        )

        self.summary = {
            "total_executions": total,
            "completed": total - errors,
            "errors": errors,
            "expectation_mismatches": mismatches,
            "accepted": accepted,
            "rejected": rejected,
            "retry_used": retried,
            "expectations_met": (
                total > 0 and errors == 0 and mismatches == 0
            ),
        }

        if errors:
            self.status = "FAILED"
        elif mismatches:
            self.status = "FAILED_EXPECTATION"
        elif self.strict and self.warnings:
            self.status = "FAILED_STRICT"
        elif self.warnings:
            self.status = "PASSED_WITH_WARNINGS"
        else:
            self.status = "PASSED"

    def to_dictionary(self) -> dict[str, Any]:
        """Return JSON-safe report data."""

        return json_safe(asdict(self))


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and persistent logging."""

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.run_demo_scenarios")
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
            "Run flowchart-aligned FT-QuPAP capstone scenarios and "
            "export non-secret dashboard evidence."
        )
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help=(
            "Comma-separated scenario IDs. Default: all enabled scenarios."
        ),
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List configured scenarios and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Master deterministic seed (default: {MASTER_SEED}).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of executions per selected scenario (default: 1).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional human-readable run identifier.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate module contracts and the execution plan without "
            "running protocol sessions."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat warnings as failure and require every controlled "
            "scenario to match its expected outcome."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first scenario execution error.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Skip the exported GP model-bundle validation step.",
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help=(
            "Permit execution without a complete exported model bundle. "
            "Only use for scenarios that stop before GP inference."
        ),
    )
    parser.add_argument(
        "--hardware",
        choices=("off", "auto", "required"),
        default="off",
        help=(
            "LED status integration: off, auto fallback, or required "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help=(
            "Maximum compact records retained in database/demo_sessions.json "
            f"(default: {DEFAULT_HISTORY_LIMIT})."
        ),
    )
    parser.add_argument(
        "--append-csv",
        action="store_true",
        help=(
            "Append executed rows to demo_session_logs.csv. Dashboard output "
            "is always replaced with the current run."
        ),
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Run scenarios without writing CSV/JSON output artifacts.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line settings."""

    if args.seed < 0:
        raise DemoRunnerError("--seed must be non-negative.")
    if args.repeat < 1:
        raise DemoRunnerError("--repeat must be at least 1.")
    if args.history_limit < 1:
        raise DemoRunnerError("--history-limit must be at least 1.")


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse common CSV Boolean values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    return default


def parse_float(value: Any, default: float = 0.0) -> float:
    """Parse a finite floating-point configuration value."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if math.isfinite(number) else default


def parse_int(value: Any, default: int = 0) -> int:
    """Parse an integer configuration value."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def builtin_scenarios() -> list[ScenarioDefinition]:
    """Return the project-tree scenario catalogue."""

    rows = [
        (
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
            "Valid credentials, fresh request, recoverable payload, "
            "matching KMAC tag, and low calibrated attack risk.",
        ),
        (
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
            "Steane CSS corrects bounded benign disturbance.",
        ),
        (
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
            "A low-risk failure triggers a fresh nonce, fresh ML-KEM "
            "session, and bounded re-authentication.",
        ),
        (
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
            "Eve measures and resends a substantial fraction of blocks.",
        ),
        (
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
            "Partial Eve activity is evaluated using observable evidence.",
        ),
        (
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
            "Eve intercepts every transmitted block.",
        ),
        (
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
            "The same pseudonym, timestamp, and nonce are reused.",
        ),
        (
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
            "The mobile station rejects a modified ML-DSA signature.",
        ),
        (
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
            "Modification breaks request validation or transcript binding.",
        ),
        (
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
            "Malformed or modified ML-KEM ciphertext fails closed.",
        ),
        (
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
            "The recovered tag differs from the transcript-bound tag.",
        ),
        (
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
            "Loss exceeds the deterministic acceptance policy.",
        ),
        (
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
            "A required block exceeds the bounded correction model.",
        ),
    ]

    return [
        ScenarioDefinition(*row, display_order=index)
        for index, row in enumerate(rows, start=1)
    ]


def scenario_from_csv(row: Mapping[str, Any]) -> ScenarioDefinition:
    """Parse one controlled_scenarios.csv row."""

    expected = str(row.get("expected_outcome", "")).strip().lower()
    if expected not in VALID_EXPECTED_OUTCOMES:
        raise DemoRunnerError(
            "Unsupported expected_outcome for "
            f"{row.get('scenario_id')!r}: {expected!r}"
        )

    scenario_id = str(row.get("scenario_id", "")).strip()
    scenario_module = str(row.get("scenario_module", "")).strip()

    if not scenario_id or not scenario_module:
        raise DemoRunnerError(
            "Every controlled scenario requires scenario_id and "
            "scenario_module."
        )

    return ScenarioDefinition(
        scenario_id=scenario_id,
        scenario_module=scenario_module,
        display_name=str(
            row.get("display_name", scenario_id)
        ).strip(),
        category=str(row.get("category", "unspecified")).strip(),
        context=str(row.get("context", "urban")).strip(),
        bit_flip_prob=parse_float(row.get("bit_flip_prob")),
        phase_flip_prob=parse_float(row.get("phase_flip_prob")),
        depolarizing_prob=parse_float(row.get("depolarizing_prob")),
        loss_prob=parse_float(row.get("loss_prob")),
        eve_fraction=parse_float(row.get("eve_fraction")),
        eve_mode=str(row.get("eve_mode", "none")).strip(),
        tamper_type=str(row.get("tamper_type", "none")).strip(),
        allow_retry=parse_bool(row.get("allow_retry")),
        expected_outcome=expected,
        expected_primary_reason=str(
            row.get("expected_primary_reason", "")
        ).strip(),
        description=str(row.get("description", "")).strip(),
        display_order=parse_int(row.get("display_order"), 0),
        enabled=parse_bool(row.get("enabled"), True),
    )


def load_scenarios(logger: logging.Logger) -> list[ScenarioDefinition]:
    """Load generated controlled settings or use the built-in catalogue."""

    if not CONTROLLED_SCENARIOS_FILE.is_file():
        logger.warning(
            "%s not found; using the built-in scenario catalogue.",
            relative_path(CONTROLLED_SCENARIOS_FILE),
        )
        return builtin_scenarios()

    with CONTROLLED_SCENARIOS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise DemoRunnerError(
            f"{relative_path(CONTROLLED_SCENARIOS_FILE)} is empty."
        )

    scenarios = [scenario_from_csv(row) for row in rows]
    identifiers = [row.scenario_id for row in scenarios]

    if len(identifiers) != len(set(identifiers)):
        raise DemoRunnerError(
            "controlled_scenarios.csv contains duplicate scenario IDs."
        )

    scenarios.sort(
        key=lambda item: (
            item.display_order if item.display_order > 0 else 10_000,
            item.scenario_id,
        )
    )
    return scenarios


def select_scenarios(
    requested: str,
    scenarios: Sequence[ScenarioDefinition],
) -> list[ScenarioDefinition]:
    """Resolve CLI scenario IDs while preserving display order."""

    available = {item.scenario_id: item for item in scenarios}

    if requested.strip().lower() == "all":
        selected = [item for item in scenarios if item.enabled]
    else:
        requested_ids = [
            item.strip()
            for item in requested.split(",")
            if item.strip()
        ]
        unknown = sorted(set(requested_ids) - set(available))
        if unknown:
            raise DemoRunnerError(
                f"Unknown scenario IDs: {unknown}. "
                "Use --list-scenarios."
            )
        selected = [available[item] for item in requested_ids]

    if not selected:
        raise DemoRunnerError("No scenarios were selected.")

    return selected


def print_scenarios(scenarios: Sequence[ScenarioDefinition]) -> None:
    """Display the configured scenario catalogue."""

    print(
        f"{'ID':34} {'EXPECTED':19} {'CONTEXT':10} MODULE"
    )
    print("-" * 110)

    for item in scenarios:
        enabled = "" if item.enabled else " [disabled]"
        print(
            f"{item.scenario_id:34} "
            f"{item.expected_outcome:19} "
            f"{item.context:10} "
            f"{item.scenario_module}{enabled}"
        )


def derive_scenario_seed(
    master_seed: int,
    scenario_id: str,
    repetition: int,
) -> int:
    """Derive a stable 32-bit seed without Python's randomized hash()."""

    material = (
        f"{PROTOCOL_VERSION}|{master_seed}|"
        f"{scenario_id}|{repetition}"
    ).encode("utf-8")

    return int.from_bytes(
        hashlib.sha256(material).digest()[:4],
        byteorder="big",
        signed=False,
    )


def make_run_id(requested: str | None, seed: int) -> str:
    """Create a non-secret run identifier."""

    if requested:
        cleaned = "".join(
            character
            for character in requested.strip()
            if character.isalnum() or character in "-_."
        )
        if not cleaned:
            raise DemoRunnerError(
                "--run-id must contain an alphanumeric character."
            )
        return cleaned

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = hashlib.sha256(
        f"{seed}|{stamp}|{uuid.uuid4().hex}".encode("utf-8")
    ).hexdigest()[:8].upper()
    return f"DEMO-{stamp}-{suffix}"


def relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_output_directories() -> None:
    """Create runtime data, result, database, log, and report folders."""

    for directory in (
        DATA_DEMO_DIR,
        DATA_RESULTS_DIR,
        DATABASE_DIR,
        OUTPUT_LOG_DIR,
        OUTPUT_REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def complete_model_bundle() -> tuple[bool, list[Path]]:
    """Return whether all GP deployment files exist."""

    missing = [
        path for path in MODEL_BUNDLE_FILES if not path.is_file()
    ]
    return not missing, missing


def validate_model_bundle(
    args: argparse.Namespace,
    logger: logging.Logger,
    warnings: list[str],
) -> None:
    """Validate model files before scenarios can reach GP inference."""

    if args.skip_model_validation:
        warnings.append("GP model validation was skipped.")
        return

    complete, missing = complete_model_bundle()
    if not complete:
        message = (
            "The GP model bundle is incomplete: "
            + ", ".join(relative_path(path) for path in missing)
        )
        if args.allow_missing_model or args.dry_run:
            warnings.append(message)
            logger.warning(message)
            return
        raise DemoRunnerError(
            message + ". Run scripts/export_gp_model.py first."
        )

    if not MODEL_VALIDATOR.is_file():
        warnings.append(
            "validate_model_files.py is unavailable; only file presence "
            "was checked."
        )
        return

    command = [
        sys.executable,
        str(MODEL_VALIDATOR),
        "--no-report",
        "--skip-dataset-check",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        tail = (
            completed.stdout + "\n" + completed.stderr
        ).strip().splitlines()[-8:]
        raise DemoRunnerError(
            "GP model validation failed: " + " | ".join(tail)
        )

    logger.info("Exported GP model bundle passed validation.")


def import_scenario_module(
    definition: ScenarioDefinition,
) -> Any:
    """Import one configured scenario module."""

    try:
        return importlib.import_module(definition.scenario_module)
    except Exception as exc:
        raise DemoRunnerError(
            f"Could not import {definition.scenario_module}: {exc}"
        ) from exc


def import_protocol_engine_module() -> Any | None:
    """Import the protocol-engine module when available."""

    try:
        return importlib.import_module(
            "src.protocol.protocol_engine"
        )
    except ModuleNotFoundError:
        return None


def callable_accepts_parameter(
    function: Callable[..., Any],
    parameter_name: str,
) -> bool:
    """Return whether a callable explicitly or variadically accepts a name."""

    signature = inspect.signature(function)
    return (
        parameter_name in signature.parameters
        or any(
            parameter.kind
            == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )


def invoke_supported(
    function: Callable[..., Any],
    available: Mapping[str, Any],
) -> Any:
    """
    Invoke a project callable using only supported keyword parameters.

    Missing required parameters produce an explicit module-contract error.
    """

    signature = inspect.signature(function)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    kwargs: dict[str, Any] = {}
    missing: list[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if parameter.default is inspect.Parameter.empty:
                missing.append(name)
            continue

        if name in available:
            kwargs[name] = available[name]
        elif parameter.default is inspect.Parameter.empty:
            missing.append(name)

    if missing and not accepts_kwargs:
        raise DemoRunnerError(
            f"{function!r} requires unsupported parameters: {missing}"
        )

    if accepts_kwargs:
        for name, value in available.items():
            kwargs.setdefault(name, value)

    result = function(**kwargs)

    if inspect.isawaitable(result):
        return asyncio.run(result)

    return result


def create_protocol_engine(
    seed: int,
    scenario_config: Mapping[str, Any],
) -> Any | None:
    """Create a fresh engine so scenario state does not leak across cases."""

    module = import_protocol_engine_module()
    if module is None:
        return None

    available = {
        "project_root": PROJECT_ROOT,
        "seed": seed,
        "master_seed": seed,
        "scenario_seed": seed,
        "scenario_config": scenario_config,
        "config": scenario_config,
        "demo_mode": True,
        "persist": False,
    }

    for factory_name in (
        "create_protocol_engine",
        "build_protocol_engine",
        "create_default_engine",
    ):
        factory = getattr(module, factory_name, None)
        if callable(factory):
            return invoke_supported(factory, available)

    engine_class = getattr(module, "ProtocolEngine", None)
    if engine_class is None:
        return None

    for constructor_name in (
        "from_project_root",
        "from_config",
        "for_demo",
    ):
        constructor = getattr(engine_class, constructor_name, None)
        if callable(constructor):
            return invoke_supported(constructor, available)

    return invoke_supported(engine_class, available)


def resolve_scenario_callable(module: Any) -> tuple[str, Any]:
    """Resolve the preferred scenario execution interface."""

    for name in ("run_scenario", "execute_scenario", "run"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return "function", candidate

    scenario_class = getattr(module, "Scenario", None)
    if scenario_class is None:
        scenario_class = getattr(module, "SCENARIO_CLASS", None)

    if inspect.isclass(scenario_class):
        return "class", scenario_class

    build_scenario = getattr(module, "build_scenario", None)
    if callable(build_scenario):
        return "builder", build_scenario

    raise DemoRunnerError(
        f"{module.__name__} must expose run_scenario(), Scenario, "
        "SCENARIO_CLASS, or build_scenario()."
    )


def execute_scenario_module(
    definition: ScenarioDefinition,
    seed: int,
    run_id: str,
) -> Any:
    """Execute one scenario using a fresh protocol engine."""

    module = import_scenario_module(definition)
    scenario_config = definition.to_config()
    engine = create_protocol_engine(seed, scenario_config)

    available = {
        "protocol_engine": engine,
        "engine": engine,
        "scenario_config": scenario_config,
        "config": scenario_config,
        "definition": definition,
        "scenario_definition": definition,
        "scenario_id": definition.scenario_id,
        "seed": seed,
        "scenario_seed": seed,
        "master_seed": seed,
        "run_id": run_id,
        "project_root": PROJECT_ROOT,
        "persist": False,
        "save_result": False,
        "demo_mode": True,
    }

    interface_type, target = resolve_scenario_callable(module)

    if interface_type == "function":
        return invoke_supported(target, available)

    if interface_type == "class":
        instance = invoke_supported(target, available)
        for method_name in ("run", "execute", "run_scenario"):
            method = getattr(instance, method_name, None)
            if callable(method):
                return invoke_supported(method, available)
        raise DemoRunnerError(
            f"{module.__name__}.Scenario has no run/execute method."
        )

    scenario_object = invoke_supported(target, available)

    if isinstance(scenario_object, Mapping):
        if engine is None:
            raise DemoRunnerError(
                f"{module.__name__}.build_scenario returned configuration "
                "but ProtocolEngine is unavailable."
            )

        for method_name in (
            "run_scenario",
            "execute_scenario",
            "run",
        ):
            method = getattr(engine, method_name, None)
            if callable(method):
                engine_available = dict(available)
                engine_available.update(
                    {
                        "scenario": scenario_object,
                        "scenario_config": scenario_object,
                        "config": scenario_object,
                    }
                )
                return invoke_supported(method, engine_available)

    for method_name in ("run", "execute"):
        method = getattr(scenario_object, method_name, None)
        if callable(method):
            return invoke_supported(method, available)

    raise DemoRunnerError(
        f"{module.__name__}.build_scenario produced an unsupported object."
    )


def mapping_from_result(result: Any) -> Mapping[str, Any]:
    """Convert common project result objects into a mapping."""

    if isinstance(result, Mapping):
        return result

    if dataclasses.is_dataclass(result):
        return asdict(result)

    for method_name in (
        "to_dictionary",
        "to_dict",
        "as_dict",
        "model_dump",
    ):
        method = getattr(result, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, Mapping):
                return converted

    raise DemoRunnerError(
        "Scenario output must be a mapping, dataclass, or object with "
        "to_dictionary()/to_dict()/model_dump()."
    )


def unwrap_final_session(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find the final protocol-session object in wrapper results."""

    if (
        "decision" in result
        or "accepted" in result
        or "execution_status" in result
    ):
        return result

    for key in (
        "final_session",
        "final_result",
        "replay_attempt",
        "attack_result",
        "session",
        "result",
    ):
        candidate = result.get(key)
        if isinstance(candidate, Mapping):
            return candidate

    return result


def optional_float(value: Any) -> float | None:
    """Return a finite float or None."""

    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def optional_int(value: Any) -> int | None:
    """Return an integer or None."""

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_reason_list(value: Any) -> list[str]:
    """Normalize deterministic-reason storage."""

    if value is None:
        return []

    if isinstance(value, str):
        return [
            item.strip()
            for item in value.replace(",", "|").split("|")
            if item.strip()
        ]

    if isinstance(value, Iterable) and not isinstance(
        value,
        (bytes, bytearray, Mapping),
    ):
        return [str(item) for item in value if str(item)]

    return [str(value)]


def compare_tags(session: Mapping[str, Any]) -> bool | None:
    """Compare in-memory tags without persisting their values."""

    explicit = session.get("tag_recovered")
    if explicit is None:
        explicit = session.get("tag_match")
    if explicit is not None:
        return bool(explicit)

    received = session.get("received_tag")
    expected = session.get("expected_tag")

    if isinstance(received, (bytes, bytearray)) and isinstance(
        expected,
        (bytes, bytearray),
    ):
        return hmac.compare_digest(bytes(received), bytes(expected))

    return None


def determine_actual_outcome(
    accepted: bool,
    retry_used: bool,
    reason: str,
) -> str:
    """Map the final protocol result to one flowchart outcome."""

    if not accepted:
        return OUTCOME_REJECT

    if retry_used or reason == "accepted_after_retry":
        return OUTCOME_ACCEPT_AFTER_RETRY

    return OUTCOME_ACCEPT


def expectation_matches(
    expected: str,
    actual: str,
) -> bool:
    """Apply controlled-scenario expected-outcome logic."""

    if expected == OUTCOME_ACCEPT:
        return actual == OUTCOME_ACCEPT

    if expected == OUTCOME_ACCEPT_AFTER_RETRY:
        return actual == OUTCOME_ACCEPT_AFTER_RETRY

    if expected == OUTCOME_REJECT:
        return actual == OUTCOME_REJECT

    return False


def is_sensitive_key(key: str) -> bool:
    """Return whether a field must be excluded from persisted details."""

    normalized = key.strip().lower()

    if normalized in SAFE_SENSITIVE_NAME_EXCEPTIONS:
        return False

    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def json_safe(value: Any, *, key_name: str = "") -> Any:
    """Convert values to JSON while redacting cryptographic material."""

    if key_name and is_sensitive_key(key_name):
        if isinstance(value, (bytes, bytearray)):
            return f"<redacted:{len(value)} bytes>"
        return "<redacted>"

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<redacted:{len(value)} bytes>"

    if dataclasses.is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item, key_name=str(key))
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except Exception:
            pass

    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return json_safe(list_method())
        except Exception:
            pass

    return str(value)


def compact_safe_details(
    result: Mapping[str, Any],
    session: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep useful non-secret diagnostics for the JSON run report."""

    allowed_session_fields = (
        "decision",
        "features",
        "timings",
        "channel",
        "attempt_history",
        "payload_failures",
        "qber_raw",
        "qber_mismatches",
        "qber_observed",
        "observed_check_blocks",
        "required_check_blocks",
        "loss_rate",
        "physical_qubits",
        "logical_payload_blocks",
        "logical_check_blocks",
        "retry_attempts",
        "retry_used",
        "attempt_index",
        "schedule_reason",
        "bootstrap_mode",
        "decision_mode",
        "use_css",
    )

    details = {
        key: session.get(key)
        for key in allowed_session_fields
        if key in session
    }

    wrapper_metadata = {}
    for key in (
        "scenario_notes",
        "original_attempt_accepted",
        "attack_stage",
        "tamper_applied",
        "expected_reason",
    ):
        if key in result:
            wrapper_metadata[key] = result[key]

    if wrapper_metadata:
        details["scenario_wrapper"] = wrapper_metadata

    return json_safe(details)


def normalize_execution(
    definition: ScenarioDefinition,
    raw_result: Any,
    *,
    run_id: str,
    seed: int,
    repetition: int,
    started: float,
) -> ScenarioExecution:
    """Normalize one real protocol result into dashboard-safe fields."""

    result = mapping_from_result(raw_result)
    session = unwrap_final_session(result)

    decision_value = session.get("decision", {})
    decision = (
        decision_value
        if isinstance(decision_value, Mapping)
        else mapping_from_result(decision_value)
    )

    accepted = bool(
        decision.get(
            "accepted",
            session.get("accepted", False),
        )
    )
    reason = str(
        decision.get(
            "reason",
            session.get("reason", "unspecified"),
        )
    )

    retry_attempts = optional_int(
        session.get(
            "retry_attempts",
            result.get("retry_attempts", 1),
        )
    )
    retry_attempts = max(1, retry_attempts or 1)

    retry_used = bool(
        session.get(
            "retry_used",
            result.get("retry_used", retry_attempts > 1),
        )
    )

    actual_outcome = determine_actual_outcome(
        accepted,
        retry_used,
        reason,
    )

    timings = session.get("timings", {})
    if not isinstance(timings, Mapping):
        timings = {}

    p_attack = optional_float(
        decision.get(
            "p_attack",
            session.get("p_attack"),
        )
    )
    uncertainty = optional_float(
        decision.get(
            "uncertainty",
            session.get("uncertainty"),
        )
    )
    gp_threshold = optional_float(
        decision.get(
            "gp_attack_threshold",
            session.get("gp_attack_threshold"),
        )
    )

    deterministic_reasons = normalize_reason_list(
        decision.get(
            "deterministic_reasons",
            session.get("deterministic_reasons"),
        )
    )
    deterministic_pass = bool(
        decision.get(
            "deterministic_pass",
            session.get(
                "deterministic_pass",
                len(deterministic_reasons) == 0,
            ),
        )
    )

    record_id = (
        f"{run_id}:{definition.scenario_id}:"
        f"{repetition}:{seed:08X}"
    )

    end_to_end = optional_float(
        timings.get(
            "end_to_end_s",
            session.get("end_to_end_seconds"),
        )
    )
    if end_to_end is None:
        end_to_end = time.perf_counter() - started

    total_retry = optional_float(
        timings.get(
            "total_retry_end_to_end_s",
            session.get("total_retry_seconds"),
        )
    )
    if total_retry is None and retry_used:
        total_retry = end_to_end

    return ScenarioExecution(
        run_id=run_id,
        record_id=record_id,
        executed_at_utc=utc_now_iso(),
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        scenario_id=definition.scenario_id,
        scenario_module=definition.scenario_module,
        display_name=definition.display_name,
        category=definition.category,
        context=definition.context,
        scenario_seed=seed,
        repetition=repetition,
        expected_outcome=definition.expected_outcome,
        actual_outcome=actual_outcome,
        expectation_met=expectation_matches(
            definition.expected_outcome,
            actual_outcome,
        ),
        accepted=accepted,
        reason=reason,
        deterministic_pass=deterministic_pass,
        deterministic_reasons=deterministic_reasons,
        qber_raw=optional_float(session.get("qber_raw")),
        qber_mismatches=optional_int(
            session.get("qber_mismatches")
        ),
        qber_observed=optional_int(session.get("qber_observed")),
        observed_check_blocks=optional_int(
            session.get("observed_check_blocks")
        ),
        required_check_blocks=optional_int(
            session.get("required_check_blocks")
        ),
        loss_rate=optional_float(session.get("loss_rate")),
        p_attack=p_attack,
        uncertainty=uncertainty,
        gp_attack_threshold=gp_threshold,
        tag_recovered=compare_tags(session),
        retry_attempts=retry_attempts,
        retry_used=retry_used,
        physical_qubits=optional_int(
            session.get("physical_qubits")
        ),
        logical_payload_blocks=optional_int(
            session.get("logical_payload_blocks")
        ),
        logical_check_blocks=optional_int(
            session.get("logical_check_blocks")
        ),
        end_to_end_seconds=end_to_end,
        total_retry_seconds=total_retry,
        execution_status="COMPLETED",
        execution_error="",
        safe_details=compact_safe_details(result, session),
    )


def error_execution(
    definition: ScenarioDefinition,
    *,
    run_id: str,
    seed: int,
    repetition: int,
    exc: Exception,
    started: float,
) -> ScenarioExecution:
    """Build a non-secret error record."""

    message = f"{type(exc).__name__}: {exc}"

    return ScenarioExecution(
        run_id=run_id,
        record_id=(
            f"{run_id}:{definition.scenario_id}:"
            f"{repetition}:{seed:08X}"
        ),
        executed_at_utc=utc_now_iso(),
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        scenario_id=definition.scenario_id,
        scenario_module=definition.scenario_module,
        display_name=definition.display_name,
        category=definition.category,
        context=definition.context,
        scenario_seed=seed,
        repetition=repetition,
        expected_outcome=definition.expected_outcome,
        actual_outcome="error",
        expectation_met=False,
        accepted=False,
        reason="scenario_execution_error",
        deterministic_pass=False,
        deterministic_reasons=["scenario_execution_error"],
        retry_attempts=1,
        retry_used=False,
        end_to_end_seconds=time.perf_counter() - started,
        execution_status="ERROR",
        execution_error=message,
        safe_details={
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    )


class HardwareIndicator:
    """Optional adapter for the project's LED controller."""

    def __init__(
        self,
        mode: str,
        logger: logging.Logger,
        warnings: list[str],
    ) -> None:
        self.mode = mode
        self.logger = logger
        self.warnings = warnings
        self.controller: Any | None = None

        if mode != "off":
            self.controller = self._load_controller()

    def _load_controller(self) -> Any | None:
        """Load a project LED controller or fallback adapter."""

        candidates = (
            "src.hardware.led_controller",
            "src.hardware.hardware_fallback",
        )

        last_error: Exception | None = None

        for module_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                last_error = exc
                continue

            for factory_name in (
                "create_led_controller",
                "build_led_controller",
            ):
                factory = getattr(module, factory_name, None)
                if callable(factory):
                    try:
                        return invoke_supported(
                            factory,
                            {
                                "demo_mode": True,
                                "project_root": PROJECT_ROOT,
                            },
                        )
                    except Exception as exc:
                        last_error = exc

            controller_class = getattr(module, "LEDController", None)
            if inspect.isclass(controller_class):
                try:
                    return invoke_supported(
                        controller_class,
                        {
                            "demo_mode": True,
                            "project_root": PROJECT_ROOT,
                        },
                    )
                except Exception as exc:
                    last_error = exc

            for object_name in (
                "controller",
                "led_controller",
                "HARDWARE_FALLBACK",
            ):
                controller = getattr(module, object_name, None)
                if controller is not None:
                    return controller

        message = (
            "LED controller unavailable"
            + (f": {last_error}" if last_error else ".")
        )

        if self.mode == "required":
            raise DemoRunnerError(message)

        self.warnings.append(message)
        self.logger.warning(message)
        return None

    def indicate(self, execution: ScenarioExecution) -> str:
        """Display GREEN, YELLOW, or RED for the final outcome."""

        if self.mode == "off":
            return "OFF"

        status = (
            "YELLOW"
            if execution.retry_used
            else ("GREEN" if execution.accepted else "RED")
        )

        if self.controller is None:
            return f"FALLBACK_{status}"

        available = {
            "status": status,
            "state": status,
            "color": status,
            "result": execution.csv_row(),
            "accepted": execution.accepted,
            "retry_used": execution.retry_used,
            "reason": execution.reason,
        }

        for method_name in (
            "set_status",
            "display_status",
            "send_status",
            "send",
            "update",
        ):
            method = getattr(self.controller, method_name, None)
            if callable(method):
                try:
                    invoke_supported(method, available)
                    return status
                except Exception as exc:
                    message = (
                        f"LED status update failed for "
                        f"{execution.scenario_id}: {exc}"
                    )
                    if self.mode == "required":
                        raise DemoRunnerError(message) from exc
                    self.warnings.append(message)
                    self.logger.warning(message)
                    return f"FALLBACK_{status}"

        message = "Loaded LED controller has no supported status method."
        if self.mode == "required":
            raise DemoRunnerError(message)

        self.warnings.append(message)
        return f"FALLBACK_{status}"

    def close(self) -> None:
        """Close a serial/device controller when supported."""

        if self.controller is None:
            return

        for method_name in ("close", "disconnect", "shutdown"):
            method = getattr(self.controller, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break


def validate_scenario_contracts(
    selected: Sequence[ScenarioDefinition],
) -> list[str]:
    """Import selected modules and verify their supported interfaces."""

    validated: list[str] = []

    for definition in selected:
        module = import_scenario_module(definition)
        interface_type, target = resolve_scenario_callable(module)
        validated.append(
            f"{definition.scenario_id}:{interface_type}:"
            f"{getattr(target, '__name__', type(target).__name__)}"
        )

    return validated


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
            json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read existing CSV rows when appending."""

    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Atomically write a CSV table with stable columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                normalized = {
                    name: json_safe(row.get(name))
                    for name in fieldnames
                }
                writer.writerow(normalized)

        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_demo_database() -> dict[str, Any]:
    """Load existing compact demo-session history."""

    if not DEMO_SESSIONS_DATABASE.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "updated_at_utc": None,
            "last_run_id": None,
            "sessions": [],
        }

    try:
        payload = json.loads(
            DEMO_SESSIONS_DATABASE.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoRunnerError(
            f"Could not read {relative_path(DEMO_SESSIONS_DATABASE)}: "
            f"{exc}"
        ) from exc

    if not isinstance(payload, MutableMapping):
        raise DemoRunnerError(
            "database/demo_sessions.json must contain a JSON object."
        )

    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        payload["sessions"] = []

    return dict(payload)


def persist_results(
    report: DemoRunReport,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """Persist current-run CSVs, compact history, and run report."""

    rows = [execution.csv_row() for execution in report.executions]

    if args.append_csv:
        existing = read_csv_rows(DEMO_SESSION_LOGS_FILE)
        log_rows: list[Mapping[str, Any]] = [*existing, *rows]
    else:
        log_rows = rows

    atomic_write_csv(
        DEMO_SESSION_LOGS_FILE,
        log_rows,
        CSV_FIELDS,
    )

    dashboard_rows = [
        {name: row.get(name) for name in DASHBOARD_FIELDS}
        for row in rows
    ]
    atomic_write_csv(
        DASHBOARD_RESULTS_FILE,
        dashboard_rows,
        DASHBOARD_FIELDS,
    )

    retry_rows = [
        {name: row.get(name) for name in RETRY_FIELDS}
        for row in rows
        if bool(row.get("retry_used"))
        or row.get("scenario_id") == "accept_after_retry"
    ]
    atomic_write_csv(
        RETRY_RESULTS_FILE,
        retry_rows,
        RETRY_FIELDS,
    )

    database = load_demo_database()
    existing_sessions = database.get("sessions", [])
    compact_records = [
        {
            **execution.csv_row(),
            "safe_details": execution.safe_details,
        }
        for execution in report.executions
    ]
    combined = [*existing_sessions, *compact_records]
    database.update(
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "updated_at_utc": utc_now_iso(),
            "last_run_id": report.run_id,
            "sessions": combined[-args.history_limit :],
        }
    )
    atomic_write_json(DEMO_SESSIONS_DATABASE, database)
    atomic_write_json(RUN_REPORT_FILE, report.to_dictionary())

    logger.info(
        "Saved demo logs: %s",
        relative_path(DEMO_SESSION_LOGS_FILE),
    )
    logger.info(
        "Saved dashboard results: %s",
        relative_path(DASHBOARD_RESULTS_FILE),
    )
    logger.info(
        "Saved run report: %s",
        relative_path(RUN_REPORT_FILE),
    )


def set_process_seed(seed: int) -> None:
    """Seed Python and compatible numerical libraries."""

    random.seed(seed)
    os.environ["FT_QUPAP_MASTER_SEED"] = str(seed)
    os.environ["FT_QUPAP_SCENARIO_SEED"] = str(seed)
    os.environ["FT_QUPAP_DEMO_MODE"] = "1"

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def log_execution(
    execution: ScenarioExecution,
    logger: logging.Logger,
) -> None:
    """Print one compact non-secret scenario result."""

    status = (
        "PASS"
        if execution.execution_status == "COMPLETED"
        and execution.expectation_met
        else "FAIL"
    )

    logger.info(
        (
            "%s | %-32s | expected=%s actual=%s "
            "reason=%s qber=%s p_attack=%s retry=%d"
        ),
        status,
        execution.scenario_id,
        execution.expected_outcome,
        execution.actual_outcome,
        execution.reason,
        (
            "n/a"
            if execution.qber_raw is None
            else f"{execution.qber_raw:.6f}"
        ),
        (
            "n/a"
            if execution.p_attack is None
            else f"{execution.p_attack:.6f}"
        ),
        execution.retry_attempts,
    )


def execute_plan(
    selected: Sequence[ScenarioDefinition],
    args: argparse.Namespace,
    run_id: str,
    report: DemoRunReport,
    logger: logging.Logger,
) -> None:
    """Execute selected scenarios in deterministic order."""

    indicator = HardwareIndicator(
        args.hardware,
        logger,
        report.warnings,
    )

    try:
        for definition in selected:
            for repetition in range(1, args.repeat + 1):
                scenario_seed = derive_scenario_seed(
                    args.seed,
                    definition.scenario_id,
                    repetition,
                )
                set_process_seed(scenario_seed)
                started = time.perf_counter()

                logger.info(
                    "Starting %s repetition=%d seed=%d",
                    definition.scenario_id,
                    repetition,
                    scenario_seed,
                )

                try:
                    raw_result = execute_scenario_module(
                        definition,
                        scenario_seed,
                        run_id,
                    )
                    execution = normalize_execution(
                        definition,
                        raw_result,
                        run_id=run_id,
                        seed=scenario_seed,
                        repetition=repetition,
                        started=started,
                    )
                    execution.hardware_status = indicator.indicate(
                        execution
                    )

                except Exception as exc:
                    execution = error_execution(
                        definition,
                        run_id=run_id,
                        seed=scenario_seed,
                        repetition=repetition,
                        exc=exc,
                        started=started,
                    )
                    execution.hardware_status = indicator.indicate(
                        execution
                    )
                    logger.error(
                        "Scenario %s failed: %s",
                        definition.scenario_id,
                        execution.execution_error,
                    )
                    logger.debug(traceback.format_exc())

                report.executions.append(execution)
                log_execution(execution, logger)

                if (
                    args.fail_fast
                    and execution.execution_status == "ERROR"
                ):
                    return
    finally:
        indicator.close()


def print_summary(report: DemoRunReport) -> None:
    """Display a compact final capstone-run summary."""

    summary = report.summary

    print("\n" + "=" * 78)
    print("FT-QuPAP CONTROLLED DEMONSTRATION SUMMARY")
    print("=" * 78)
    print(f"Run ID:                 {report.run_id}")
    print(f"Status:                 {report.status}")
    print(
        f"Total executions:       "
        f"{summary.get('total_executions', 0)}"
    )
    print(f"Completed:              {summary.get('completed', 0)}")
    print(f"Accepted:               {summary.get('accepted', 0)}")
    print(f"Rejected:               {summary.get('rejected', 0)}")
    print(f"Retry used:             {summary.get('retry_used', 0)}")
    print(
        f"Expectation mismatches: "
        f"{summary.get('expectation_mismatches', 0)}"
    )
    print(f"Execution errors:       {summary.get('errors', 0)}")
    print(f"Warnings:               {len(report.warnings)}")
    print("=" * 78)


def main() -> int:
    """Command-line entry point."""

    logger = configure_logging()

    try:
        args = parse_arguments()
        validate_arguments(args)
        ensure_output_directories()

        scenarios = load_scenarios(logger)

        if args.list_scenarios:
            print_scenarios(scenarios)
            return 0

        selected = select_scenarios(args.scenario, scenarios)
        run_id = make_run_id(args.run_id, args.seed)

        report = DemoRunReport(
            schema_version=SCHEMA_VERSION,
            protocol=PROTOCOL_NAME,
            protocol_version=PROTOCOL_VERSION,
            run_id=run_id,
            started_at_utc=utc_now_iso(),
            master_seed=args.seed,
            selected_scenarios=[
                item.scenario_id for item in selected
            ],
            repeat=args.repeat,
            dry_run=args.dry_run,
            strict=args.strict,
        )

        validate_model_bundle(
            args,
            logger,
            report.warnings,
        )

        contracts = validate_scenario_contracts(selected)
        logger.info(
            "Validated scenario contracts: %s",
            ", ".join(contracts),
        )

        if args.dry_run:
            report.summary = {
                "selected_scenarios": len(selected),
                "planned_executions": len(selected) * args.repeat,
                "validated_contracts": contracts,
            }
            report.finished_at_utc = utc_now_iso()
            report.status = (
                "FAILED_STRICT"
                if args.strict and report.warnings
                else (
                    "PASSED_WITH_WARNINGS"
                    if report.warnings
                    else "PASSED"
                )
            )

            if not args.no_persist:
                atomic_write_json(
                    RUN_REPORT_FILE,
                    report.to_dictionary(),
                )

            print_summary(report)
            return 0 if report.status in {
                "PASSED",
                "PASSED_WITH_WARNINGS",
            } else 1

        execute_plan(
            selected,
            args,
            run_id,
            report,
            logger,
        )

        report.finalize()

        if not args.no_persist:
            persist_results(report, args, logger)

        print_summary(report)

        return 0 if report.status in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        } else 1

    except DemoRunnerError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        logger.error("Demonstration interrupted by user.")
        print("\nDemonstration interrupted.", file=sys.stderr)
        return 130

    except Exception:
        logger.exception("Unexpected demonstration-runner failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
