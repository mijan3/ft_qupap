#!/usr/bin/env python3
"""
Safely clear FT-QuPAP capstone demonstration logs and runtime evidence.

Purpose
-------
This script resets only the generated demonstration/runtime artifacts created
by scripts/run_demo_scenarios.py. It intentionally preserves the protocol
implementation, subscriber registration, ML-DSA trust anchors, ML-KEM/GP
model files, processed datasets, held-out metrics, and final research
evidence.

Default cleanup targets
-----------------------
    data/demo/demo_session_logs.csv
    data/demo/dashboard_results.csv
    data/results/retry_results.csv
    database/demo_sessions.json
    outputs/reports/demo_scenario_run.json
    outputs/logs/run_demo_scenarios.log

Optional targets
----------------
    --reset-replay-cache
        database/used_nonces.json

    --include-service-logs
        outputs/logs/protocol.log
        outputs/logs/authentication.log
        outputs/logs/attack_detection.log
        outputs/logs/hardware.log

    --clear-derived-figures
        outputs/figures/qber_comparison.png
        outputs/figures/attack_probability.png
        outputs/figures/retry_analysis.png
        outputs/figures/figure_manifest.json

Safety model
------------
1. Running without --apply performs a preview only.
2. Applying cleanup creates a local rollback ZIP by default.
3. CSV files are reset to header-only form instead of being deleted.
4. demo_sessions.json is reset to a compatible empty structure.
5. The replay cache is never cleared unless --reset-replay-cache is explicit.
6. Research datasets, models, metrics, trust anchors, registrations, and
   private-key files are outside the managed target list.
7. Every action is validated and written to a cleanup report.

Examples
--------
Preview the default cleanup:
    python scripts/clear_demo_logs.py

Apply the default cleanup with a backup:
    python scripts/clear_demo_logs.py --apply

Prepare a fully fresh capstone demonstration:
    python scripts/clear_demo_logs.py \
        --apply \
        --reset-replay-cache \
        --include-service-logs

Also remove figures derived from the old demo run:
    python scripts/clear_demo_logs.py \
        --apply \
        --clear-derived-figures

Skip rollback archive creation:
    python scripts/clear_demo_logs.py --apply --no-backup

Validate that managed runtime artifacts are empty:
    python scripts/clear_demo_logs.py --validate-only

Research boundary
-----------------
The FT-QuPAP notebook and research proposal require preservation of seeds,
splits, GP artifacts, figures, logs, result tables, metadata, and checksums
before numerical claims are finalized. Therefore, this script does not clear
paper-ready result tables or model artifacts. Use scripts/create_backup.py
before any broader project cleanup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DEMO_DIR = PROJECT_ROOT / "data" / "demo"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DATABASE_DIR = PROJECT_ROOT / "database"
OUTPUT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_BACKUP_DIR = PROJECT_ROOT / "outputs" / "backups"

DEMO_SESSION_LOGS_FILE = DATA_DEMO_DIR / "demo_session_logs.csv"
DASHBOARD_RESULTS_FILE = DATA_DEMO_DIR / "dashboard_results.csv"
RETRY_RESULTS_FILE = DATA_RESULTS_DIR / "retry_results.csv"

DEMO_SESSIONS_FILE = DATABASE_DIR / "demo_sessions.json"
USED_NONCES_FILE = DATABASE_DIR / "used_nonces.json"

DEMO_RUN_REPORT_FILE = OUTPUT_REPORT_DIR / "demo_scenario_run.json"
CLEANUP_REPORT_FILE = OUTPUT_REPORT_DIR / "demo_cleanup_report.json"

RUN_DEMO_LOG_FILE = OUTPUT_LOG_DIR / "run_demo_scenarios.log"
PROTOCOL_LOG_FILE = OUTPUT_LOG_DIR / "protocol.log"
AUTHENTICATION_LOG_FILE = OUTPUT_LOG_DIR / "authentication.log"
ATTACK_DETECTION_LOG_FILE = OUTPUT_LOG_DIR / "attack_detection.log"
HARDWARE_LOG_FILE = OUTPUT_LOG_DIR / "hardware.log"
CLEANUP_LOG_FILE = OUTPUT_LOG_DIR / "clear_demo_logs.log"
LOCK_FILE = OUTPUT_LOG_DIR / ".clear_demo_logs.lock"

QBER_FIGURE_FILE = OUTPUT_FIGURE_DIR / "qber_comparison.png"
ATTACK_PROBABILITY_FIGURE_FILE = (
    OUTPUT_FIGURE_DIR / "attack_probability.png"
)
RETRY_ANALYSIS_FIGURE_FILE = OUTPUT_FIGURE_DIR / "retry_analysis.png"
FIGURE_MANIFEST_FILE = OUTPUT_FIGURE_DIR / "figure_manifest.json"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
SCHEMA_VERSION = 1

DEMO_CSV_FIELDS = (
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

DASHBOARD_CSV_FIELDS = (
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

RETRY_CSV_FIELDS = (
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

PROTECTED_PATHS = (
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "processed",
    PROJECT_ROOT / "data" / "results" / "performance_metrics.csv",
    PROJECT_ROOT / "data" / "results" / "baseline_comparison.csv",
    PROJECT_ROOT / "data" / "results" / "confusion_matrix.csv",
    PROJECT_ROOT / "data" / "results" / "calibration_results.csv",
    PROJECT_ROOT / "data" / "results" / "threshold_analysis.csv",
    DATABASE_DIR / "subscribers.json",
    DATABASE_DIR / "registration_records.json",
    DATABASE_DIR / "trusted_server_keys.json",
    DATABASE_DIR / ".secrets",
)


class CleanupError(RuntimeError):
    """Raised when cleanup cannot be completed safely."""


@dataclass(frozen=True)
class TargetSpec:
    """One explicitly managed cleanup artifact."""

    name: str
    path: Path
    action: str
    group: str
    description: str
    csv_fields: tuple[str, ...] = ()
    backup_safe: bool = True


@dataclass
class TargetResult:
    """Before/after status for one cleanup target."""

    name: str
    path: str
    group: str
    action: str
    description: str
    existed_before: bool
    bytes_before: int
    records_before: int | None
    sha256_before: str | None
    operation_status: str = "PLANNED"
    existed_after: bool | None = None
    bytes_after: int | None = None
    records_after: int | None = None
    sha256_after: str | None = None
    backup_included: bool = False
    message: str = ""


@dataclass
class CleanupReport:
    """Machine-readable report for one cleanup invocation."""

    schema_version: int
    protocol: str
    protocol_version: str
    mode: str
    started_at_utc: str
    finished_at_utc: str = ""
    project_root: str = ""
    status: str = "RUNNING"
    backup_file: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    targets: list[TargetResult] = field(default_factory=list)
    protected_paths: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        """Calculate final counts and status."""

        self.finished_at_utc = utc_now_iso()

        failed = sum(
            item.operation_status == "FAILED"
            for item in self.targets
        )
        changed = sum(
            item.operation_status == "CLEARED"
            for item in self.targets
        )
        missing = sum(
            item.operation_status == "ALREADY_MISSING"
            for item in self.targets
        )
        validated = sum(
            item.operation_status == "VALIDATED"
            for item in self.targets
        )
        planned = sum(
            item.operation_status == "PLANNED"
            for item in self.targets
        )

        self.summary = {
            "target_count": len(self.targets),
            "changed_count": changed,
            "missing_count": missing,
            "validated_count": validated,
            "planned_count": planned,
            "failed_count": failed,
            "bytes_before": sum(
                item.bytes_before for item in self.targets
            ),
            "bytes_after": sum(
                item.bytes_after or 0
                for item in self.targets
            ),
        }

        if failed:
            self.status = "FAILED"
        elif self.mode == "preview":
            self.status = "PREVIEW"
        elif self.mode == "validate":
            self.status = "VALID"
        elif self.warnings:
            self.status = "CLEARED_WITH_WARNINGS"
        else:
            self.status = "CLEARED"

    def to_dictionary(self) -> dict[str, Any]:
        """Return JSON-serializable report content."""

        return asdict(self)


def utc_now_iso() -> str:
    """Return a stable timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and persistent cleanup logging."""

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.clear_demo_logs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        CLEANUP_LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(
        description=(
            "Safely preview, clear, or validate FT-QuPAP demo logs "
            "without deleting model, registration, or research artifacts."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Perform the cleanup. Without this option the command only "
            "shows the planned actions."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Verify that selected targets are already empty or absent. "
            "No cleanup is performed."
        ),
    )
    parser.add_argument(
        "--reset-replay-cache",
        action="store_true",
        help=(
            "Also reset database/used_nonces.json. This is intended only "
            "for a fresh isolated demonstration environment."
        ),
    )
    parser.add_argument(
        "--include-service-logs",
        action="store_true",
        help=(
            "Also truncate protocol, authentication, attack-detection, "
            "and hardware logs."
        ),
    )
    parser.add_argument(
        "--clear-derived-figures",
        action="store_true",
        help=(
            "Also remove QBER, attack-probability, retry-analysis, and "
            "figure-manifest files derived from the previous demo run."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Do not create a rollback ZIP before applying cleanup."
        ),
    )
    parser.add_argument(
        "--backup-directory",
        type=Path,
        default=OUTPUT_BACKUP_DIR,
        help=(
            "Rollback ZIP directory. Relative paths are resolved from "
            "the project root."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat missing targets and backup exclusions as failures."
        ),
    )
    parser.add_argument(
        "--break-stale-lock",
        action="store_true",
        help=(
            "Remove a cleanup lock older than --stale-lock-seconds."
        ),
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=int,
        default=3600,
        help="Age after which a cleanup lock is considered stale.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=CLEANUP_REPORT_FILE,
        help=(
            "Cleanup report path. Relative paths are resolved from the "
            "project root."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write the JSON cleanup report.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject contradictory or unsafe options."""

    if args.apply and args.validate_only:
        raise CleanupError(
            "--apply and --validate-only cannot be used together."
        )

    if args.stale_lock_seconds < 60:
        raise CleanupError(
            "--stale-lock-seconds must be at least 60."
        )

    if args.no_backup and not args.apply:
        raise CleanupError(
            "--no-backup is meaningful only with --apply."
        )


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def assert_safe_project_root() -> None:
    """Prevent cleanup from operating from an invalid filesystem root."""

    root = PROJECT_ROOT.resolve()

    if root == Path(root.anchor):
        raise CleanupError(
            "Refusing to use the filesystem root as PROJECT_ROOT."
        )

    if not (root / "scripts").is_dir():
        raise CleanupError(
            f"Project scripts directory is missing: {root / 'scripts'}"
        )

    if Path(__file__).resolve().parent != (root / "scripts").resolve():
        raise CleanupError(
            "clear_demo_logs.py must be executed from the project's "
            "scripts directory."
        )


def ensure_runtime_directories() -> None:
    """Create only the directories required for cleanup reporting."""

    for directory in (
        DATA_DEMO_DIR,
        DATA_RESULTS_DIR,
        DATABASE_DIR,
        OUTPUT_LOG_DIR,
        OUTPUT_REPORT_DIR,
        OUTPUT_BACKUP_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""

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
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
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


def atomic_write_csv_header(
    path: Path,
    fields: Sequence[str],
) -> None:
    """Atomically reset a CSV to its stable header."""

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
            writer = csv.writer(stream)
            writer.writerow(fields)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def count_csv_records(path: Path) -> int | None:
    """Count CSV data rows while excluding the header."""

    if not path.is_file():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.reader(stream)
            next(reader, None)
            return sum(1 for row in reader if any(cell for cell in row))
    except (OSError, UnicodeError, csv.Error):
        return None


def count_json_records(path: Path) -> int | None:
    """Count recognized JSON collection records."""

    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None

    if isinstance(payload, list):
        return len(payload)

    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return len(sessions)
        return len(payload)

    return None


def inspect_target(target: TargetSpec) -> TargetResult:
    """Inspect a target before or after cleanup."""

    exists = target.path.is_file()
    size = target.path.stat().st_size if exists else 0

    if target.action == "reset_csv":
        records = count_csv_records(target.path)
    elif target.action in {
        "reset_demo_sessions",
        "reset_empty_object",
    }:
        records = count_json_records(target.path)
    else:
        records = None

    digest = sha256_file(target.path) if exists else None

    return TargetResult(
        name=target.name,
        path=relative_path(target.path),
        group=target.group,
        action=target.action,
        description=target.description,
        existed_before=exists,
        bytes_before=size,
        records_before=records,
        sha256_before=digest,
    )


def relative_path(path: Path) -> str:
    """Return a project-relative path when possible."""

    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def base_targets() -> list[TargetSpec]:
    """Return the default demonstration cleanup targets."""

    return [
        TargetSpec(
            name="demo_session_logs",
            path=DEMO_SESSION_LOGS_FILE,
            action="reset_csv",
            group="demo",
            description=(
                "Reset executed per-session demonstration rows while "
                "preserving the CSV schema."
            ),
            csv_fields=DEMO_CSV_FIELDS,
        ),
        TargetSpec(
            name="dashboard_results",
            path=DASHBOARD_RESULTS_FILE,
            action="reset_csv",
            group="demo",
            description=(
                "Reset current dashboard scenario results while preserving "
                "the CSV schema."
            ),
            csv_fields=DASHBOARD_CSV_FIELDS,
        ),
        TargetSpec(
            name="retry_results",
            path=RETRY_RESULTS_FILE,
            action="reset_csv",
            group="demo",
            description=(
                "Reset executed retry-policy rows while preserving the "
                "CSV schema."
            ),
            csv_fields=RETRY_CSV_FIELDS,
        ),
        TargetSpec(
            name="demo_sessions_database",
            path=DEMO_SESSIONS_FILE,
            action="reset_demo_sessions",
            group="runtime",
            description=(
                "Clear compact non-secret session history used by the "
                "dashboard."
            ),
        ),
        TargetSpec(
            name="demo_run_report",
            path=DEMO_RUN_REPORT_FILE,
            action="remove",
            group="demo",
            description=(
                "Remove the previous controlled-scenario run report."
            ),
        ),
        TargetSpec(
            name="run_demo_log",
            path=RUN_DEMO_LOG_FILE,
            action="truncate",
            group="logs",
            description=(
                "Truncate the controlled demonstration runner log."
            ),
        ),
    ]


def selected_targets(args: argparse.Namespace) -> list[TargetSpec]:
    """Build the exact cleanup plan from explicit options."""

    targets = base_targets()

    if args.reset_replay_cache:
        targets.append(
            TargetSpec(
                name="used_nonce_replay_cache",
                path=USED_NONCES_FILE,
                action="reset_empty_object",
                group="runtime",
                description=(
                    "Reset the replay cache for a fresh isolated demo. "
                    "Never use this option against a production cache."
                ),
                backup_safe=False,
            )
        )

    if args.include_service_logs:
        service_logs = (
            (
                "protocol_log",
                PROTOCOL_LOG_FILE,
                "Protocol-engine operational log.",
            ),
            (
                "authentication_log",
                AUTHENTICATION_LOG_FILE,
                "Authentication decision log.",
            ),
            (
                "attack_detection_log",
                ATTACK_DETECTION_LOG_FILE,
                "GP and deterministic attack-detection log.",
            ),
            (
                "hardware_log",
                HARDWARE_LOG_FILE,
                "ESP32/Arduino indicator log.",
            ),
        )

        for name, path, description in service_logs:
            targets.append(
                TargetSpec(
                    name=name,
                    path=path,
                    action="truncate",
                    group="service_logs",
                    description=description,
                    backup_safe=False,
                )
            )

    if args.clear_derived_figures:
        figures = (
            (
                "qber_comparison_figure",
                QBER_FIGURE_FILE,
                "Remove QBER figure derived from prior demo sessions.",
            ),
            (
                "attack_probability_figure",
                ATTACK_PROBABILITY_FIGURE_FILE,
                "Remove P(attack) figure derived from prior demo sessions.",
            ),
            (
                "retry_analysis_figure",
                RETRY_ANALYSIS_FIGURE_FILE,
                "Remove retry figure derived from prior demo sessions.",
            ),
            (
                "figure_manifest",
                FIGURE_MANIFEST_FILE,
                "Remove the stale figure-generation manifest.",
            ),
        )

        for name, path, description in figures:
            targets.append(
                TargetSpec(
                    name=name,
                    path=path,
                    action="remove",
                    group="derived_figures",
                    description=description,
                )
            )

    validate_target_boundaries(targets)
    return targets


def is_path_within(path: Path, directory: Path) -> bool:
    """Return whether a resolved path is inside a resolved directory."""

    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def validate_target_boundaries(targets: Sequence[TargetSpec]) -> None:
    """Ensure every target is inside the project and outside protected paths."""

    seen: set[Path] = set()

    for target in targets:
        resolved = target.path.resolve()

        if resolved in seen:
            raise CleanupError(
                f"Duplicate cleanup target: {relative_path(resolved)}"
            )
        seen.add(resolved)

        if not is_path_within(resolved, PROJECT_ROOT):
            raise CleanupError(
                f"Target escapes the project root: {resolved}"
            )

        for protected in PROTECTED_PATHS:
            protected_resolved = protected.resolve()

            if protected_resolved.is_dir():
                if is_path_within(resolved, protected_resolved):
                    raise CleanupError(
                        "Cleanup target overlaps protected path: "
                        f"{relative_path(resolved)}"
                    )
            elif resolved == protected_resolved:
                raise CleanupError(
                    "Cleanup target is protected: "
                    f"{relative_path(resolved)}"
                )


def acquire_lock(
    *,
    break_stale: bool,
    stale_seconds: int,
) -> int:
    """Create an exclusive cleanup lock and return its file descriptor."""

    OUTPUT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime

        if break_stale and age >= stale_seconds:
            LOCK_FILE.unlink(missing_ok=True)
        else:
            raise CleanupError(
                "Another cleanup may be active, or a stale lock exists: "
                f"{relative_path(LOCK_FILE)}. Use --break-stale-lock "
                "only after confirming no cleanup is running."
            )

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(LOCK_FILE, flags, 0o600)

    lock_payload = {
        "pid": os.getpid(),
        "created_at_utc": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
    }
    os.write(
        descriptor,
        (
            json.dumps(lock_payload, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    os.fsync(descriptor)
    return descriptor


def release_lock(descriptor: int | None) -> None:
    """Release the cleanup lock."""

    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass

    LOCK_FILE.unlink(missing_ok=True)


def safe_backup_targets(
    targets: Sequence[TargetSpec],
) -> list[TargetSpec]:
    """Select non-secret existing files for rollback backup."""

    return [
        target
        for target in targets
        if target.backup_safe
        and target.path.is_file()
        and target.path.stat().st_size > 0
    ]


def create_backup_archive(
    targets: Sequence[TargetSpec],
    backup_directory: Path,
    logger: logging.Logger,
) -> Path | None:
    """Create a local rollback ZIP of non-secret managed artifacts."""

    selected = safe_backup_targets(targets)

    if not selected:
        logger.info("No non-empty backup-safe targets require archiving.")
        return None

    backup_directory.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = (
        backup_directory
        / f"ft_qupap_demo_cleanup_{stamp}_{os.getpid()}.zip"
    )

    manifest_rows = []

    with zipfile.ZipFile(
        archive_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for target in selected:
            relative_name = relative_path(target.path)
            archive.write(
                target.path,
                arcname=f"artifacts/{relative_name}",
            )
            manifest_rows.append(
                {
                    "name": target.name,
                    "path": relative_name,
                    "bytes": target.path.stat().st_size,
                    "sha256": sha256_file(target.path),
                }
            )

        archive.writestr(
            "backup_manifest.json",
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL_NAME,
                    "protocol_version": PROTOCOL_VERSION,
                    "created_at_utc": utc_now_iso(),
                    "purpose": (
                        "Local rollback archive created before "
                        "clear_demo_logs.py cleanup."
                    ),
                    "warning": (
                        "This is an operational rollback archive, not the "
                        "paper reproducibility archive."
                    ),
                    "files": manifest_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    try:
        archive_path.chmod(0o600)
    except OSError:
        pass

    logger.info(
        "Created rollback backup: %s",
        relative_path(archive_path),
    )
    return archive_path


def reset_demo_sessions(path: Path) -> None:
    """Reset demo session history while preserving compatible JSON shape."""

    existing: Any = None

    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None

    if isinstance(existing, list):
        payload: Any = []
    elif isinstance(existing, dict):
        payload = {
            "schema_version": existing.get(
                "schema_version",
                SCHEMA_VERSION,
            ),
            "protocol": existing.get("protocol", PROTOCOL_NAME),
            "protocol_version": existing.get(
                "protocol_version",
                PROTOCOL_VERSION,
            ),
            "updated_at_utc": utc_now_iso(),
            "last_run_id": None,
            "sessions": [],
        }

        if "database_version" in existing:
            payload["database_version"] = existing["database_version"]

        if "created_at_utc" in existing:
            payload["created_at_utc"] = existing["created_at_utc"]
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "updated_at_utc": utc_now_iso(),
            "last_run_id": None,
            "sessions": [],
        }

    atomic_write_json(path, payload)


def apply_target(target: TargetSpec) -> None:
    """Perform one cleanup action."""

    if target.action == "reset_csv":
        atomic_write_csv_header(target.path, target.csv_fields)
        return

    if target.action == "reset_demo_sessions":
        reset_demo_sessions(target.path)
        return

    if target.action == "reset_empty_object":
        atomic_write_json(target.path, {})
        return

    if target.action == "truncate":
        atomic_write_text(target.path, "")
        return

    if target.action == "remove":
        target.path.unlink(missing_ok=True)
        return

    raise CleanupError(
        f"Unsupported cleanup action: {target.action}"
    )


def csv_is_empty_with_schema(
    path: Path,
    expected_fields: Sequence[str],
) -> tuple[bool, str]:
    """Validate a header-only CSV with an exact stable schema."""

    if not path.is_file():
        return False, "CSV file is missing."

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            rows = [
                row for row in reader
                if any(cell for cell in row)
            ]
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, f"CSV could not be read: {exc}"

    if header != list(expected_fields):
        return False, (
            "CSV header mismatch. Expected "
            f"{list(expected_fields)}, found {header}."
        )

    if rows:
        return False, f"CSV still contains {len(rows)} data rows."

    return True, "CSV contains only the expected header."


def demo_sessions_is_empty(path: Path) -> tuple[bool, str]:
    """Validate an empty list or object-with-empty-sessions database."""

    if not path.is_file():
        return False, "demo_sessions.json is missing."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"Invalid demo_sessions.json: {exc}"

    if isinstance(payload, list):
        return (
            (len(payload) == 0),
            (
                "Empty legacy session list."
                if len(payload) == 0
                else f"Legacy session list contains {len(payload)} rows."
            ),
        )

    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list) and len(sessions) == 0:
            return True, "Session database has an empty sessions array."
        return False, "Session database sessions array is missing or non-empty."

    return False, "demo_sessions.json must contain a list or object."


def empty_object_is_valid(path: Path) -> tuple[bool, str]:
    """Validate an empty JSON object."""

    if not path.is_file():
        return False, "JSON file is missing."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"Invalid JSON: {exc}"

    if payload == {}:
        return True, "JSON object is empty."

    return False, "JSON object is not empty."


def validate_target_empty(
    target: TargetSpec,
) -> tuple[bool, str]:
    """Validate the expected post-cleanup state of one target."""

    if target.action == "reset_csv":
        return csv_is_empty_with_schema(
            target.path,
            target.csv_fields,
        )

    if target.action == "reset_demo_sessions":
        return demo_sessions_is_empty(target.path)

    if target.action == "reset_empty_object":
        return empty_object_is_valid(target.path)

    if target.action == "truncate":
        if not target.path.is_file():
            return False, "Log file is missing."
        if target.path.stat().st_size == 0:
            return True, "Log file is empty."
        return False, (
            f"Log file still contains {target.path.stat().st_size} bytes."
        )

    if target.action == "remove":
        if not target.path.exists():
            return True, "File is absent."
        return False, "File still exists."

    return False, f"Unsupported validation action: {target.action}"


def update_after_state(
    result: TargetResult,
    target: TargetSpec,
) -> None:
    """Populate post-operation file status."""

    exists = target.path.is_file()
    result.existed_after = exists
    result.bytes_after = target.path.stat().st_size if exists else 0
    result.sha256_after = sha256_file(target.path) if exists else None

    if target.action == "reset_csv":
        result.records_after = count_csv_records(target.path)
    elif target.action in {
        "reset_demo_sessions",
        "reset_empty_object",
    }:
        result.records_after = count_json_records(target.path)
    else:
        result.records_after = None


def preview_plan(
    targets: Sequence[TargetSpec],
    report: CleanupReport,
) -> None:
    """Populate a non-destructive cleanup preview."""

    for target in targets:
        result = inspect_target(target)
        result.operation_status = "PLANNED"
        result.message = (
            "Would clear existing artifact."
            if result.existed_before
            else "Target does not currently exist; schema files may be created."
        )
        report.targets.append(result)


def validate_plan(
    targets: Sequence[TargetSpec],
    report: CleanupReport,
    strict: bool,
) -> None:
    """Validate that selected runtime artifacts are empty."""

    for target in targets:
        result = inspect_target(target)
        valid, message = validate_target_empty(target)
        result.message = message
        update_after_state(result, target)

        if valid:
            result.operation_status = "VALIDATED"
        elif not target.path.exists() and not strict:
            result.operation_status = "ALREADY_MISSING"
            report.warnings.append(
                f"{relative_path(target.path)} is missing."
            )
        else:
            result.operation_status = "FAILED"

        report.targets.append(result)


def apply_plan(
    targets: Sequence[TargetSpec],
    report: CleanupReport,
    *,
    strict: bool,
    logger: logging.Logger,
) -> None:
    """Apply every cleanup target and validate its final state."""

    for target in targets:
        result = inspect_target(target)

        try:
            apply_target(target)
            valid, message = validate_target_empty(target)
            update_after_state(result, target)
            result.message = message

            if valid:
                result.operation_status = "CLEARED"
                logger.info(
                    "CLEARED | %s | %s",
                    relative_path(target.path),
                    message,
                )
            else:
                result.operation_status = "FAILED"
                logger.error(
                    "FAILED | %s | %s",
                    relative_path(target.path),
                    message,
                )

        except Exception as exc:
            update_after_state(result, target)
            result.operation_status = "FAILED"
            result.message = f"{type(exc).__name__}: {exc}"
            logger.error(
                "FAILED | %s | %s",
                relative_path(target.path),
                result.message,
            )

        if (
            not result.existed_before
            and target.action == "remove"
            and result.operation_status == "CLEARED"
        ):
            result.operation_status = "ALREADY_MISSING"
            result.message = "Target was already absent."
            if strict:
                result.operation_status = "FAILED"
            else:
                report.warnings.append(
                    f"{relative_path(target.path)} was already absent."
                )

        report.targets.append(result)


def mark_backup_inclusions(
    report: CleanupReport,
    targets: Sequence[TargetSpec],
) -> None:
    """Mark which before-state files were included in the rollback ZIP."""

    safe_names = {
        target.name
        for target in safe_backup_targets(targets)
    }

    for result in report.targets:
        result.backup_included = result.name in safe_names


def protected_path_summary() -> list[str]:
    """Return the explicit preservation boundary shown in reports."""

    return [relative_path(path) for path in PROTECTED_PATHS]


def write_report(
    report: CleanupReport,
    report_path: Path,
) -> None:
    """Persist the cleanup report atomically."""

    atomic_write_json(
        report_path,
        report.to_dictionary(),
    )


def print_plan(
    report: CleanupReport,
) -> None:
    """Print a compact human-readable cleanup summary."""

    print("\n" + "=" * 88)
    print("FT-QuPAP DEMO CLEANUP")
    print("=" * 88)
    print(f"Mode:       {report.mode}")
    print(f"Status:     {report.status}")
    print(f"Project:    {report.project_root}")
    print(f"Backup:     {report.backup_file or 'not created'}")
    print("-" * 88)

    for item in report.targets:
        records = (
            "n/a"
            if item.records_before is None
            else str(item.records_before)
        )
        print(
            f"{item.operation_status:16} "
            f"{item.path:48} "
            f"bytes={item.bytes_before:<8} rows={records}"
        )
        if item.message:
            print(f"{'':18}{item.message}")

    print("-" * 88)
    print(
        "Protected: models, training/validation/test data, paper metrics, "
        "subscriber registration, trust anchors, and private keys."
    )
    print("=" * 88)


def report_mode(args: argparse.Namespace) -> str:
    """Return the execution mode."""

    if args.validate_only:
        return "validate"
    if args.apply:
        return "apply"
    return "preview"


def main() -> int:
    """Command-line entry point."""

    logger = configure_logging()
    lock_descriptor: int | None = None

    try:
        args = parse_arguments()
        validate_arguments(args)
        assert_safe_project_root()
        ensure_runtime_directories()

        targets = selected_targets(args)
        mode = report_mode(args)

        report = CleanupReport(
            schema_version=SCHEMA_VERSION,
            protocol=PROTOCOL_NAME,
            protocol_version=PROTOCOL_VERSION,
            mode=mode,
            started_at_utc=utc_now_iso(),
            project_root=str(PROJECT_ROOT),
            options={
                "apply": bool(args.apply),
                "validate_only": bool(args.validate_only),
                "reset_replay_cache": bool(
                    args.reset_replay_cache
                ),
                "include_service_logs": bool(
                    args.include_service_logs
                ),
                "clear_derived_figures": bool(
                    args.clear_derived_figures
                ),
                "backup_enabled": bool(
                    args.apply and not args.no_backup
                ),
                "strict": bool(args.strict),
            },
            protected_paths=protected_path_summary(),
        )

        if mode == "preview":
            preview_plan(targets, report)
            report.finalize()

        elif mode == "validate":
            validate_plan(
                targets,
                report,
                strict=args.strict,
            )
            report.finalize()

        else:
            lock_descriptor = acquire_lock(
                break_stale=args.break_stale_lock,
                stale_seconds=args.stale_lock_seconds,
            )

            if not args.no_backup:
                backup_directory = resolve_project_path(
                    args.backup_directory
                )
                backup_path = create_backup_archive(
                    targets,
                    backup_directory,
                    logger,
                )
                report.backup_file = (
                    relative_path(backup_path)
                    if backup_path is not None
                    else None
                )

                excluded_existing = [
                    target
                    for target in targets
                    if not target.backup_safe
                    and target.path.is_file()
                    and target.path.stat().st_size > 0
                ]
                for target in excluded_existing:
                    message = (
                        f"{relative_path(target.path)} was intentionally "
                        "excluded from the rollback ZIP because it may "
                        "contain runtime-sensitive information."
                    )
                    report.warnings.append(message)

                    if args.strict:
                        raise CleanupError(message)

            apply_plan(
                targets,
                report,
                strict=args.strict,
                logger=logger,
            )
            mark_backup_inclusions(report, targets)
            report.finalize()

        if not args.no_report:
            report_path = resolve_project_path(args.report)
            write_report(report, report_path)
            logger.info(
                "Saved cleanup report: %s",
                relative_path(report_path),
            )

        print_plan(report)

        if report.status in {
            "PREVIEW",
            "VALID",
            "CLEARED",
            "CLEARED_WITH_WARNINGS",
        }:
            return 0

        return 1

    except CleanupError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        logger.error("Demo cleanup interrupted by user.")
        print("\nDemo cleanup interrupted.", file=sys.stderr)
        return 130

    except Exception:
        logger.exception("Unexpected FT-QuPAP demo cleanup failure.")
        return 1

    finally:
        release_lock(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
