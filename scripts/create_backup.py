#!/usr/bin/env python3
"""
Create and validate a secret-safe FT-QuPAP reproducibility archive.

The script implements the project-level equivalent of the notebook's final
portable export. It preserves the executed notebook, implementation, model
bundle, processed datasets, result tables, figures, reports, documentation,
package versions, Git metadata, and SHA-256 checksums.

It intentionally excludes:
- .env and database/.secrets/
- ML-DSA and ML-KEM private/secret keys
- K_ss, K_auth, K_ctrl and shared-session-secret dumps
- raw subscriber identities
- reusable raw KMAC tags
- ciphertext dumps
- used nonce/replay-cache data
- virtual environments, caches, Git internals, and existing archives

Default use:
    python scripts/create_backup.py

Preview:
    python scripts/create_backup.py --dry-run

Strict paper-evidence archive:
    python scripts/create_backup.py \
        --strict \
        --executed-notebook notebooks/11_complete_protocol_experiment.ipynb

Validate an archive:
    python scripts/create_backup.py \
        --validate-only outputs/backups/<archive-name>.zip
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
BACKUP_DIR = OUTPUT_DIR / "backups"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"

LOG_FILE = LOG_DIR / "create_backup.log"
REPORT_FILE = REPORT_DIR / "backup_report.json"
LOCK_FILE = BACKUP_DIR / ".create_backup.lock"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
SCHEMA_VERSION = 1

ROOT_FILES = (
    "requirements.txt",
    "README.md",
    "config.py",
    ".env.example",
    ".gitignore",
    "run_demo.bat",
    "run_demo.sh",
)

SOURCE_DIRECTORIES = (
    "src",
    "scenarios",
    "dashboard",
    "tests",
    "scripts",
    "hardware",
)

EVIDENCE_DIRECTORIES = (
    "notebooks",
    "models",
    "data/processed",
    "data/results",
    "outputs/figures",
    "outputs/reports",
    "docs",
)

OPTIONAL_DEMO_DIRECTORIES = ("data/demo",)
OPTIONAL_RAW_DIRECTORIES = ("data/raw",)

SINGLE_FILES = (
    "assets/images/protocol_flowchart.png",
    "assets/images/system_architecture.png",
    "assets/images/project_logo.png",
)

APPROVED_LOGS = (
    "outputs/logs/protocol.log",
    "outputs/logs/authentication.log",
    "outputs/logs/attack_detection.log",
    "outputs/logs/hardware.log",
    "outputs/logs/export_gp_model.log",
    "outputs/logs/validate_model_files.log",
    "outputs/logs/run_all_tests.log",
    "outputs/logs/run_demo_scenarios.log",
    "outputs/logs/generate_result_graphs.log",
)

MODEL_FILES = (
    "models/gp_model.pkl",
    "models/feature_scaler.pkl",
    "models/calibration_model.pkl",
    "models/threshold.json",
    "models/feature_order.json",
    "models/model_metadata.json",
)

RESULT_FILES = (
    "data/results/performance_metrics.csv",
    "data/results/baseline_comparison.csv",
    "data/results/retry_results.csv",
    "data/results/confusion_matrix.csv",
    "data/results/calibration_results.csv",
    "data/results/threshold_analysis.csv",
)

FIGURE_FILES = (
    "outputs/figures/roc_curve.png",
    "outputs/figures/pr_curve.png",
    "outputs/figures/calibration_curve.png",
    "outputs/figures/confusion_matrix.png",
    "outputs/figures/qber_comparison.png",
    "outputs/figures/attack_probability.png",
    "outputs/figures/retry_analysis.png",
)

SKIP_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "node_modules",
    "backups",
}

SKIP_FILE_PATTERNS = (
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.temp",
    "*.swp",
    "*.swo",
    "*.pid",
    "*.lock",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.7z",
    "*.rar",
    ".DS_Store",
    "Thumbs.db",
)

SENSITIVE_PATH_PATTERNS = (
    ".env",
    ".env.*",
    "**/.secrets/**",
    "**/secrets/**",
    "**/*private_key*",
    "**/*secret_key*",
    "**/*shared_secret*",
    "**/*session_secret*",
    "**/*session_key*",
    "**/*k_auth*",
    "**/*k_ctrl*",
    "**/*k_ss*",
    "**/*.pem",
    "**/*.key",
    "**/*.p12",
    "**/*.pfx",
    "**/*.der",
    "**/subscribers.json",
    "**/used_nonces.json",
    "**/registration_records.json",
    "**/*ciphertext_dump*",
    "**/*raw_tag*",
)

SENSITIVE_PATH_EXCEPTIONS = {".env.example"}

DYNAMIC_SCAN_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".log",
    ".txt",
    ".yaml",
    ".yml",
}

DYNAMIC_SCAN_ROOTS = {"data", "database", "outputs"}

SENSITIVE_CONTENT_PATTERNS = (
    re.compile(
        r"(?i)\b(?:private[_ -]?key|secret[_ -]?key)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{24,}"
    ),
    re.compile(
        r"(?i)\b(?:k_auth|k_ctrl|k_ss|shared_secret|session_secret)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{16,}"
    ),
    re.compile(
        r"(?i)\b(?:raw_subscriber_identity|imsi)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9-]{6,}"
    ),
    re.compile(
        r"(?i)\b(?:raw_tag|authentication_tag)\b"
        r"\s*[:=]\s*[\"']?[A-Fa-f0-9]{16,}"
    ),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |ML-DSA |ML-KEM )?PRIVATE KEY-----"
    ),
)

DEFAULT_MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_SCAN_BYTES = 16 * 1024 * 1024


class BackupError(RuntimeError):
    """Raised when an archive cannot be created or validated safely."""


@dataclass(frozen=True)
class Candidate:
    source: Path
    archive_path: str
    category: str
    external: bool = False


@dataclass(frozen=True)
class Artifact:
    archive_path: str
    source_path: str
    category: str
    bytes: int
    sha256: str
    modified_at_utc: str
    external: bool


@dataclass(frozen=True)
class Exclusion:
    source_path: str
    reason: str
    severity: str


@dataclass
class BackupReport:
    schema_version: int
    protocol: str
    protocol_version: str
    mode: str
    started_at_utc: str
    finished_at_utc: str = ""
    project_root: str = ""
    status: str = "RUNNING"
    archive_path: str | None = None
    archive_sha256: str | None = None
    archive_bytes: int | None = None
    archive_id: str | None = None
    evidence_scope: str = "unspecified"
    model_id: str | None = None
    strict: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    included: list[Artifact] = field(default_factory=list)
    excluded: list[Exclusion] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def finalize(self) -> None:
        self.finished_at_utc = utc_now_iso()

        rejected = sum(
            item.severity == "REJECTED"
            for item in self.excluded
        )

        self.summary = {
            "included_file_count": len(self.included),
            "included_bytes": sum(item.bytes for item in self.included),
            "excluded_count": len(self.excluded),
            "rejected_sensitive_count": rejected,
            "missing_required_count": len(self.missing_required),
            "warning_count": len(self.warnings),
        }

        has_issues = bool(
            self.missing_required
            or rejected
            or self.warnings
        )

        if self.mode == "validate":
            self.status = "INVALID" if has_issues else "VALID"
        elif self.mode == "dry_run":
            self.status = (
                "DRY_RUN_FAILED_STRICT"
                if self.strict and has_issues
                else "DRY_RUN"
            )
        elif self.archive_path is None:
            self.status = "FAILED"
        elif self.strict and has_issues:
            self.status = "CREATED_FAILED_STRICT"
        elif has_issues:
            self.status = "CREATED_WITH_WARNINGS"
        else:
            self.status = "CREATED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.create_backup")
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
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate a secret-safe FT-QuPAP reproducibility "
            "archive with per-file SHA-256 checksums."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=BACKUP_DIR,
        help="Archive directory; relative paths use the project root.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Archive base name without .zip.",
    )
    parser.add_argument(
        "--label",
        default="evidence",
        help="Short run label used in the manifest and filename.",
    )
    parser.add_argument(
        "--executed-notebook",
        type=Path,
        default=None,
        help=(
            "Exact executed .ipynb file. External files are stored under "
            "artifacts/notebooks/executed/."
        ),
    )
    parser.add_argument(
        "--include-demo",
        action="store_true",
        help="Include data/demo controlled scenario artifacts.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include approved logs after sensitive-content scanning.",
    )
    parser.add_argument(
        "--include-raw-data",
        action="store_true",
        help="Include data/raw; disabled by default.",
    )
    parser.add_argument(
        "--no-source",
        action="store_true",
        help="Exclude source, tests, scenarios, dashboard, and hardware.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail on missing required artifacts, rejected candidates, "
            "or warnings."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without creating an archive.",
    )
    parser.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        metavar="ARCHIVE",
        help="Validate an existing archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing archive with the same name.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=9,
        help="ZIP deflate level from 0 to 9.",
    )
    parser.add_argument(
        "--max-file-mib",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES // (1024 * 1024),
        help="Maximum individual file size in MiB.",
    )
    parser.add_argument(
        "--break-stale-lock",
        action="store_true",
        help="Remove an archive lock older than the stale-lock limit.",
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=int,
        default=3600,
        help="Lock age considered stale.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write outputs/reports/backup_report.json.",
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.dry_run and args.validate_only is not None:
        raise BackupError(
            "--dry-run and --validate-only cannot be combined."
        )

    if not 0 <= args.compression_level <= 9:
        raise BackupError("--compression-level must be from 0 to 9.")

    if args.max_file_mib < 1:
        raise BackupError("--max-file-mib must be at least 1.")

    if args.stale_lock_seconds < 60:
        raise BackupError(
            "--stale-lock-seconds must be at least 60."
        )

    validate_safe_name(args.label, "--label")

    if args.name is not None:
        validate_safe_name(args.name, "--name")


def validate_safe_name(value: str, field: str) -> None:
    if not value or len(value) > 80:
        raise BackupError(f"{field} must contain 1 to 80 characters.")

    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise BackupError(
            f"{field} may contain only letters, digits, '.', '_', and '-'."
        )


def assert_project_root() -> None:
    root = PROJECT_ROOT.resolve()

    if root == Path(root.anchor):
        raise BackupError("Refusing to use the filesystem root.")

    if Path(__file__).resolve().parent != (root / "scripts").resolve():
        raise BackupError(
            "create_backup.py must be placed in the project's scripts folder."
        )


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def ensure_directories() -> None:
    for directory in (BACKUP_DIR, LOG_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return str(path.resolve())


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
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


def installed_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": installed_version("numpy"),
        "pandas": installed_version("pandas"),
        "scikit_learn": installed_version("scikit-learn"),
        "joblib": installed_version("joblib"),
        "cryptography": installed_version("cryptography"),
        "pycryptodome": installed_version("pycryptodome"),
        "pqcrypto": installed_version("pqcrypto"),
        "qiskit": installed_version("qiskit"),
        "qiskit_aer": installed_version("qiskit-aer"),
    }


def collect_git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if completed.returncode != 0:
            return None

        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    status = git("status", "--porcelain")

    return {
        "commit": commit,
        "branch": branch,
        "working_tree_dirty": (
            None if status is None else bool(status)
        ),
    }


def read_model_metadata() -> tuple[str, str | None]:
    path = PROJECT_ROOT / "models" / "model_metadata.json"

    if not path.is_file():
        return "unspecified", None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unspecified", None

    scope = str(payload.get("evidence_scope", "unspecified"))
    model_id_value = payload.get("model_id")
    model_id = (
        str(model_id_value)
        if model_id_value is not None
        else None
    )

    return scope, model_id


def skip_generated(path: Path) -> bool:
    if any(part in SKIP_DIRECTORY_NAMES for part in path.parts):
        return True

    return any(
        fnmatch.fnmatch(path.name, pattern)
        for pattern in SKIP_FILE_PATTERNS
    )


def sensitive_path(path: Path) -> bool:
    normalized = project_relative(path).lower()
    basename = path.name.lower()

    if normalized in SENSITIVE_PATH_EXCEPTIONS:
        return False

    for pattern in SENSITIVE_PATH_PATTERNS:
        pattern_lower = pattern.lower()

        if fnmatch.fnmatch(normalized, pattern_lower):
            return True

        basename_pattern = pattern_lower.replace("**/", "")
        if fnmatch.fnmatch(basename, basename_pattern):
            return True

    parts = set(PurePosixPath(normalized).parts)

    return ".secrets" in parts or "secrets" in parts


def scan_dynamic_text(path: Path) -> str | None:
    if path.suffix.lower() not in DYNAMIC_SCAN_EXTENSIONS:
        return None

    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None

    if not relative.parts or relative.parts[0] not in DYNAMIC_SCAN_ROOTS:
        return None

    if path.stat().st_size > MAX_SCAN_BYTES:
        return "Dynamic text file is too large for secret scanning."

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return f"Unable to scan text file: {exc}"

    for pattern in SENSITIVE_CONTENT_PATTERNS:
        if pattern.search(content):
            return (
                "Generated text appears to contain private/session keys, "
                "raw identity, reusable tag, or private-key material."
            )

    return None


def classify(path: Path, external: bool = False) -> str:
    if external:
        return "executed_notebook"

    parts = PurePosixPath(project_relative(path)).parts

    if not parts:
        return "root"

    first = parts[0]

    if first == "models":
        return "model_bundle"
    if first == "notebooks":
        return "notebook"
    if first == "data":
        return (
            f"data_{parts[1]}"
            if len(parts) > 1
            else "data"
        )
    if first == "outputs":
        return (
            f"output_{parts[1]}"
            if len(parts) > 1
            else "output"
        )
    if first in {"src", "scenarios", "dashboard"}:
        return "implementation"
    if first == "tests":
        return "tests"
    if first == "scripts":
        return "scripts"
    if first == "hardware":
        return "hardware"
    if first == "docs":
        return "documentation"
    if first == "assets":
        return "assets"

    return "root"


def archive_name_for(path: Path, external: bool) -> str:
    if external:
        return f"artifacts/notebooks/executed/{path.name}"

    return f"artifacts/{project_relative(path)}"


def walk_files(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return

    for root, directories, files in os.walk(
        directory,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)

        directories[:] = sorted(
            name
            for name in directories
            if name not in SKIP_DIRECTORY_NAMES
            and not (root_path / name).is_symlink()
        )

        for name in sorted(files):
            path = root_path / name

            if path.is_file() and not path.is_symlink():
                yield path


def add_candidate(
    selected: dict[str, Candidate],
    exclusions: list[Exclusion],
    path: Path,
    *,
    external: bool = False,
) -> None:
    if not path.exists():
        return

    if path.is_symlink():
        exclusions.append(
            Exclusion(
                source_path=project_relative(path),
                reason="Symbolic links are not archived.",
                severity="SKIPPED",
            )
        )
        return

    if not path.is_file():
        return

    if skip_generated(path):
        exclusions.append(
            Exclusion(
                source_path=project_relative(path),
                reason=(
                    "Cache, temporary file, lock, or existing archive."
                ),
                severity="SKIPPED",
            )
        )
        return

    if sensitive_path(path):
        exclusions.append(
            Exclusion(
                source_path=project_relative(path),
                reason="Path matches the secret-material deny list.",
                severity="REJECTED",
            )
        )
        return

    archive_path = archive_name_for(path, external)
    key = archive_path.casefold()

    if key in selected:
        if selected[key].source.resolve() != path.resolve():
            exclusions.append(
                Exclusion(
                    source_path=project_relative(path),
                    reason=(
                        "Case-insensitive archive-path collision with "
                        f"{project_relative(selected[key].source)}."
                    ),
                    severity="REJECTED",
                )
            )
        return

    selected[key] = Candidate(
        source=path.resolve(),
        archive_path=archive_path,
        category=classify(path, external),
        external=external,
    )


def collect_candidates(
    args: argparse.Namespace,
) -> tuple[list[Candidate], list[Exclusion]]:
    selected: dict[str, Candidate] = {}
    exclusions: list[Exclusion] = []

    for relative in ROOT_FILES:
        add_candidate(
            selected,
            exclusions,
            PROJECT_ROOT / relative,
        )

    directories = list(EVIDENCE_DIRECTORIES)

    if not args.no_source:
        directories.extend(SOURCE_DIRECTORIES)

    if args.include_demo:
        directories.extend(OPTIONAL_DEMO_DIRECTORIES)

    if args.include_raw_data:
        directories.extend(OPTIONAL_RAW_DIRECTORIES)

    for relative in directories:
        directory = PROJECT_ROOT / relative

        for path in walk_files(directory):
            if path.resolve() in {
                LOG_FILE.resolve(),
                REPORT_FILE.resolve(),
                LOCK_FILE.resolve(),
            }:
                continue

            add_candidate(selected, exclusions, path)

    for relative in SINGLE_FILES:
        add_candidate(
            selected,
            exclusions,
            PROJECT_ROOT / relative,
        )

    if args.include_logs:
        for relative in APPROVED_LOGS:
            add_candidate(
                selected,
                exclusions,
                PROJECT_ROOT / relative,
            )

    if args.executed_notebook is not None:
        notebook = resolve_project_path(args.executed_notebook)

        if not notebook.is_file():
            raise BackupError(
                f"Executed notebook does not exist: {notebook}"
            )

        if notebook.suffix.lower() != ".ipynb":
            raise BackupError(
                "--executed-notebook must be an .ipynb file."
            )

        add_candidate(
            selected,
            exclusions,
            notebook,
            external=not is_within(notebook, PROJECT_ROOT),
        )

    return (
        sorted(
            selected.values(),
            key=lambda item: item.archive_path,
        ),
        exclusions,
    )


def evaluate_candidates(
    candidates: Sequence[Candidate],
    exclusions: Sequence[Exclusion],
    max_file_bytes: int,
) -> tuple[list[Artifact], list[Exclusion]]:
    included: list[Artifact] = []
    final_exclusions = list(exclusions)

    for candidate in candidates:
        path = candidate.source

        try:
            size = path.stat().st_size
        except OSError as exc:
            final_exclusions.append(
                Exclusion(
                    source_path=project_relative(path),
                    reason=f"Unable to inspect file: {exc}",
                    severity="REJECTED",
                )
            )
            continue

        if size > max_file_bytes:
            final_exclusions.append(
                Exclusion(
                    source_path=project_relative(path),
                    reason=(
                        f"File exceeds configured limit: "
                        f"{size} > {max_file_bytes} bytes."
                    ),
                    severity="REJECTED",
                )
            )
            continue

        scan_reason = scan_dynamic_text(path)

        if scan_reason is not None:
            final_exclusions.append(
                Exclusion(
                    source_path=project_relative(path),
                    reason=scan_reason,
                    severity="REJECTED",
                )
            )
            continue

        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(timespec="seconds")

        included.append(
            Artifact(
                archive_path=candidate.archive_path,
                source_path=project_relative(path),
                category=candidate.category,
                bytes=size,
                sha256=sha256_file(path),
                modified_at_utc=modified,
                external=candidate.external,
            )
        )

    return included, final_exclusions


def preferred_notebook(args: argparse.Namespace) -> Path:
    if args.executed_notebook is not None:
        return resolve_project_path(args.executed_notebook)

    preferred = (
        PROJECT_ROOT
        / "notebooks"
        / "11_complete_protocol_experiment.ipynb"
    )

    if preferred.is_file():
        return preferred

    notebooks = sorted(
        (PROJECT_ROOT / "notebooks").glob("*.ipynb")
    )

    return notebooks[-1] if notebooks else preferred


def required_files(args: argparse.Namespace) -> list[Path]:
    paths = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "README.md",
        preferred_notebook(args),
    ]

    paths.extend(PROJECT_ROOT / item for item in MODEL_FILES)
    paths.extend(PROJECT_ROOT / item for item in RESULT_FILES)
    paths.extend(PROJECT_ROOT / item for item in FIGURE_FILES)

    return paths


def missing_required(
    args: argparse.Namespace,
    artifacts: Sequence[Artifact],
) -> list[str]:
    included_sources = {
        str(
            Path(item.source_path).resolve()
            if Path(item.source_path).is_absolute()
            else (PROJECT_ROOT / item.source_path).resolve()
        )
        for item in artifacts
    }

    missing = []

    for path in required_files(args):
        if (
            not path.is_file()
            or str(path.resolve()) not in included_sources
        ):
            missing.append(project_relative(path))

    return sorted(set(missing))


def build_archive_id(
    artifacts: Sequence[Artifact],
    *,
    label: str,
    evidence_scope: str,
    model_id: str | None,
) -> str:
    payload = json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "label": label,
            "evidence_scope": evidence_scope,
            "model_id": model_id,
            "artifacts": [
                {
                    "path": item.archive_path,
                    "sha256": item.sha256,
                }
                for item in artifacts
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()[:24]


def default_archive_name(
    label: str,
    evidence_scope: str,
    model_id: str | None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    scope = re.sub(
        r"[^A-Za-z0-9_-]+",
        "-",
        evidence_scope,
    ).strip("-")[:30] or "unspecified"
    model = (model_id or "no-model")[:12]

    return f"FT_QuPAP_{label}_{timestamp}_{scope}_{model}"


def build_manifest(report: BackupReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "protocol": report.protocol,
        "protocol_version": report.protocol_version,
        "archive_id": report.archive_id,
        "created_at_utc": report.started_at_utc,
        "label": report.options.get("label"),
        "evidence_scope": report.evidence_scope,
        "model_id": report.model_id,
        "command_line": [sys.executable, *sys.argv],
        "options": report.options,
        "runtime": report.runtime,
        "git": report.git,
        "missing_required": report.missing_required,
        "warnings": report.warnings,
        "excluded": [asdict(item) for item in report.excluded],
        "artifacts": [asdict(item) for item in report.included],
        "security_boundary": {
            "excluded": [
                "ML-KEM secret keys",
                "ML-DSA secret keys",
                "K_ss",
                "K_auth",
                "K_ctrl",
                "shared/session secrets",
                "raw subscriber identities",
                "reusable raw authentication tags",
                "ciphertext dumps",
                "used nonce/replay-cache data",
                ".env files",
            ],
            "database_included_by_default": False,
            "logs_included": bool(
                report.options.get("include_logs")
            ),
        },
    }


def build_checksum_file(artifacts: Sequence[Artifact]) -> str:
    return "".join(
        f"{item.sha256}  {item.archive_path}\n"
        for item in artifacts
    )


def build_archive_readme(report: BackupReport) -> str:
    completeness = (
        "complete"
        if not report.missing_required
        else "incomplete"
    )

    return f"""# FT-QuPAP Reproducibility Archive

Archive ID: `{report.archive_id}`  
Protocol version: `{report.protocol_version}`  
Created: `{report.started_at_utc}`  
Evidence scope: `{report.evidence_scope}`  
Model ID: `{report.model_id or "not available"}`  
Required-artifact status: **{completeness}**

## Contents

- `MANIFEST.json`: provenance, versions, security exclusions, and file hashes.
- `CHECKSUMS.sha256`: SHA-256 digest of every file below `artifacts/`.
- `artifacts/`: selected project-relative evidence and implementation files.

## Recommended execution order

1. Install `requirements.txt` in a clean virtual environment.
2. Run `python scripts/initialize_database.py`.
3. Generate protocol traces or development demo data.
4. Run `python scripts/export_gp_model.py`.
5. Run `python scripts/validate_model_files.py --strict`.
6. Run `python scripts/run_all_tests.py --strict`.
7. Run `python scripts/run_demo_scenarios.py --strict`.
8. Run `python scripts/generate_result_graphs.py --strict --force`.
9. Run `python scripts/create_backup.py --strict`.

## Interpretation boundary

The full-session path is a syndrome-level Steane CSS simulation. Qiskit/Aer
supports representative encoded-block validation; it is not a physical
1,120-qubit hardware execution.

The calibrated GP estimates attack probability under its training
distribution. It supplements, but does not replace, credential, freshness,
replay, transcript, decoder, KMAC, and loss checks.

## Sensitive data

This archive is designed to exclude private/secret keys, session keys, raw
identities, reusable tags, ciphertext dumps, replay caches, and `.env` files.

## Validate

```bash
python scripts/create_backup.py --validate-only <archive.zip>
```
"""


def candidate_map(
    candidates: Sequence[Candidate],
) -> dict[str, Path]:
    return {
        item.archive_path: item.source
        for item in candidates
    }


def create_archive(
    archive_path: Path,
    candidates: Sequence[Candidate],
    report: BackupReport,
    compression_level: int,
    force: bool,
) -> None:
    if archive_path.exists() and not force:
        raise BackupError(
            f"Archive exists: {archive_path}. Use --force."
        )

    archive_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.stem}.",
        suffix=".zip.tmp",
        dir=archive_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    sources = candidate_map(candidates)

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compression_level,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                "MANIFEST.json",
                json.dumps(
                    build_manifest(report),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
            )
            archive.writestr(
                "CHECKSUMS.sha256",
                build_checksum_file(report.included),
            )
            archive.writestr(
                "ARCHIVE_README.md",
                build_archive_readme(report),
            )

            for artifact in report.included:
                source = sources.get(artifact.archive_path)

                if source is None:
                    raise BackupError(
                        "Internal source mapping failed for "
                        f"{artifact.archive_path}."
                    )

                if sha256_file(source) != artifact.sha256:
                    raise BackupError(
                        "Source changed during archive creation: "
                        f"{project_relative(source)}"
                    )

                archive.write(
                    source,
                    arcname=artifact.archive_path,
                )

        validate_archive(temporary_path)
        os.replace(temporary_path, archive_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def safe_zip_name(name: str) -> bool:
    path = PurePosixPath(name)

    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
    )


def parse_checksums(content: str) -> dict[str, str]:
    checksums: dict[str, str] = {}

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        match = re.fullmatch(
            r"([0-9A-Fa-f]{64})  (.+)",
            line,
        )

        if match is None:
            raise BackupError(
                f"Invalid checksum line {line_number}."
            )

        digest, name = match.groups()
        checksums[name] = digest.lower()

    return checksums


def validate_archive(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BackupError(f"Archive not found: {path}")

    try:
        archive_context = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise BackupError(f"Invalid ZIP: {exc}") from exc

    with archive_context as archive:
        bad_file = archive.testzip()

        if bad_file is not None:
            raise BackupError(
                f"ZIP CRC check failed for {bad_file}."
            )

        names = archive.namelist()

        if len(names) != len(set(names)):
            raise BackupError(
                "Archive contains duplicate member names."
            )

        unsafe_names = [
            name for name in names if not safe_zip_name(name)
        ]

        if unsafe_names:
            raise BackupError(
                f"Unsafe ZIP member paths: {unsafe_names[:5]}"
            )

        internal = {
            "MANIFEST.json",
            "CHECKSUMS.sha256",
            "ARCHIVE_README.md",
        }

        missing_internal = sorted(internal - set(names))

        if missing_internal:
            raise BackupError(
                "Missing internal archive files: "
                f"{missing_internal}"
            )

        try:
            manifest = json.loads(
                archive.read("MANIFEST.json").decode("utf-8")
            )
            checksum_map = parse_checksums(
                archive.read("CHECKSUMS.sha256").decode(
                    "utf-8"
                )
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError(
                f"Invalid archive metadata: {exc}"
            ) from exc

        artifacts = manifest.get("artifacts")

        if not isinstance(artifacts, list):
            raise BackupError(
                "MANIFEST.json artifacts must be a list."
            )

        manifest_paths: set[str] = set()

        for entry in artifacts:
            if not isinstance(entry, Mapping):
                raise BackupError(
                    "Manifest artifact entries must be objects."
                )

            archive_path = str(
                entry.get("archive_path", "")
            )
            expected_hash = str(
                entry.get("sha256", "")
            ).lower()
            expected_size = int(entry.get("bytes", -1))

            if archive_path in manifest_paths:
                raise BackupError(
                    f"Duplicate manifest path: {archive_path}"
                )

            manifest_paths.add(archive_path)

            if archive_path not in names:
                raise BackupError(
                    f"Missing artifact: {archive_path}"
                )

            if checksum_map.get(archive_path) != expected_hash:
                raise BackupError(
                    f"Checksum manifest mismatch: {archive_path}"
                )

            content = archive.read(archive_path)

            if sha256_bytes(content) != expected_hash:
                raise BackupError(
                    f"Artifact SHA-256 mismatch: {archive_path}"
                )

            if len(content) != expected_size:
                raise BackupError(
                    f"Artifact size mismatch: {archive_path}"
                )

        if set(checksum_map) != manifest_paths:
            raise BackupError(
                "CHECKSUMS.sha256 and MANIFEST.json disagree."
            )

        unmanifested = [
            name
            for name in names
            if name.startswith("artifacts/")
            and name not in manifest_paths
        ]

        if unmanifested:
            raise BackupError(
                f"Unmanifested artifacts: {unmanifested[:5]}"
            )

        return {
            "valid": True,
            "archive_id": manifest.get("archive_id"),
            "protocol": manifest.get("protocol"),
            "protocol_version": manifest.get(
                "protocol_version"
            ),
            "evidence_scope": manifest.get(
                "evidence_scope",
                "unspecified",
            ),
            "model_id": manifest.get("model_id"),
            "artifact_count": len(artifacts),
            "artifact_bytes": sum(
                int(item.get("bytes", 0))
                for item in artifacts
            ),
            "missing_required": manifest.get(
                "missing_required",
                [],
            ),
            "zip_bytes": path.stat().st_size,
            "zip_sha256": sha256_file(path),
        }


def acquire_lock(
    break_stale: bool,
    stale_seconds: int,
) -> int:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime

        if break_stale and age >= stale_seconds:
            LOCK_FILE.unlink(missing_ok=True)
        else:
            raise BackupError(
                "Another backup may be running, or a stale lock "
                f"exists: {LOCK_FILE}"
            )

    descriptor = os.open(
        LOCK_FILE,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    os.write(
        descriptor,
        (
            json.dumps(
                {
                    "pid": os.getpid(),
                    "created_at_utc": utc_now_iso(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    os.fsync(descriptor)

    return descriptor


def release_lock(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass

    LOCK_FILE.unlink(missing_ok=True)


def initialize_report(
    args: argparse.Namespace,
    mode: str,
) -> BackupReport:
    scope, model_id = read_model_metadata()

    return BackupReport(
        schema_version=SCHEMA_VERSION,
        protocol=PROTOCOL_NAME,
        protocol_version=PROTOCOL_VERSION,
        mode=mode,
        started_at_utc=utc_now_iso(),
        project_root=str(PROJECT_ROOT),
        evidence_scope=scope,
        model_id=model_id,
        strict=bool(args.strict),
        options={
            "label": args.label,
            "include_demo": bool(args.include_demo),
            "include_logs": bool(args.include_logs),
            "include_raw_data": bool(args.include_raw_data),
            "include_source": not bool(args.no_source),
            "executed_notebook": (
                str(resolve_project_path(args.executed_notebook))
                if args.executed_notebook is not None
                else None
            ),
            "compression_level": args.compression_level,
            "max_file_mib": args.max_file_mib,
        },
        runtime=collect_runtime(),
        git=collect_git_metadata(),
    )


def create_or_preview(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> BackupReport:
    mode = "dry_run" if args.dry_run else "create"
    report = initialize_report(args, mode)

    candidates, exclusions = collect_candidates(args)
    included, exclusions = evaluate_candidates(
        candidates,
        exclusions,
        args.max_file_mib * 1024 * 1024,
    )

    report.included = included
    report.excluded = exclusions
    report.missing_required = missing_required(
        args,
        included,
    )

    if report.evidence_scope in {
        "development_only",
        "synthetic_demo_development_only",
    }:
        report.warnings.append(
            "Evidence is marked development-only and must not be "
            "presented as final paper evidence."
        )
    elif report.evidence_scope in {
        "unspecified",
        "source_scope_unspecified",
        "mixed_or_partially_specified",
    }:
        report.warnings.append(
            "Evidence scope is not fully established."
        )

    if report.git.get("working_tree_dirty") is True:
        report.warnings.append(
            "Git working tree contains uncommitted changes."
        )

    rejected_logs = [
        item
        for item in report.excluded
        if item.severity == "REJECTED"
        and item.source_path.startswith("outputs/logs/")
    ]

    if args.include_logs and rejected_logs:
        report.warnings.append(
            "One or more requested logs were rejected by the "
            "sensitive-content scanner."
        )

    report.archive_id = build_archive_id(
        report.included,
        label=args.label,
        evidence_scope=report.evidence_scope,
        model_id=report.model_id,
    )

    if args.dry_run:
        report.finalize()
        return report

    output_directory = resolve_project_path(
        args.output_directory
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    base_name = (
        args.name
        if args.name is not None
        else default_archive_name(
            args.label,
            report.evidence_scope,
            report.model_id,
        )
    )
    archive_path = output_directory / f"{base_name}.zip"

    create_archive(
        archive_path,
        candidates,
        report,
        args.compression_level,
        args.force,
    )

    validation = validate_archive(archive_path)

    report.archive_path = project_relative(archive_path)
    report.archive_sha256 = validation["zip_sha256"]
    report.archive_bytes = validation["zip_bytes"]
    report.finalize()

    logger.info(
        "Created %s with %d artifacts.",
        archive_path,
        len(report.included),
    )

    return report


def validate_existing(
    args: argparse.Namespace,
    archive_path: Path,
) -> BackupReport:
    report = initialize_report(args, "validate")
    resolved = resolve_project_path(archive_path)

    try:
        result = validate_archive(resolved)
        report.archive_path = project_relative(resolved)
        report.archive_sha256 = result["zip_sha256"]
        report.archive_bytes = result["zip_bytes"]
        report.archive_id = result.get("archive_id")
        report.evidence_scope = str(
            result.get("evidence_scope", "unspecified")
        )
        model_id = result.get("model_id")
        report.model_id = (
            str(model_id) if model_id is not None else None
        )
        report.missing_required = list(
            result.get("missing_required", [])
        )
        report.summary = {
            "artifact_count": result["artifact_count"],
            "artifact_bytes": result["artifact_bytes"],
        }
    except BackupError as exc:
        report.excluded.append(
            Exclusion(
                source_path=project_relative(resolved),
                reason=str(exc),
                severity="REJECTED",
            )
        )

    report.finalize()
    return report


def print_summary(report: BackupReport) -> None:
    rejected = sum(
        item.severity == "REJECTED"
        for item in report.excluded
    )

    print("\n" + "=" * 86)
    print("FT-QuPAP REPRODUCIBILITY ARCHIVE")
    print("=" * 86)
    print(f"Mode:             {report.mode}")
    print(f"Status:           {report.status}")
    print(f"Archive ID:       {report.archive_id or 'not created'}")
    print(f"Evidence scope:   {report.evidence_scope}")
    print(f"Model ID:         {report.model_id or 'not available'}")
    print(f"Included files:   {len(report.included)}")
    print(
        f"Included bytes:   "
        f"{sum(item.bytes for item in report.included):,}"
    )
    print(f"Missing required: {len(report.missing_required)}")
    print(f"Rejected:         {rejected}")
    print(f"Warnings:         {len(report.warnings)}")
    print(f"Archive:          {report.archive_path or 'not created'}")

    if report.missing_required:
        print("\nMissing required artifacts:")
        for item in report.missing_required:
            print(f"  - {item}")

    rejected_items = [
        item
        for item in report.excluded
        if item.severity == "REJECTED"
    ]

    if rejected_items:
        print("\nRejected candidates:")
        for item in rejected_items[:20]:
            print(f"  - {item.source_path}: {item.reason}")

    print("=" * 86)


def main() -> int:
    logger = configure_logging()
    lock_descriptor: int | None = None

    try:
        args = parse_arguments()
        validate_arguments(args)
        assert_project_root()
        ensure_directories()

        if args.validate_only is not None:
            report = validate_existing(
                args,
                args.validate_only,
            )
        else:
            if not args.dry_run:
                lock_descriptor = acquire_lock(
                    args.break_stale_lock,
                    args.stale_lock_seconds,
                )

            report = create_or_preview(args, logger)

        if not args.no_report:
            atomic_write_json(
                REPORT_FILE,
                report.to_dict(),
            )

        print_summary(report)

        return 0 if report.status in {
            "CREATED",
            "CREATED_WITH_WARNINGS",
            "DRY_RUN",
            "VALID",
        } else 1

    except BackupError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        logger.error("Archive creation interrupted.")
        print("\nArchive creation interrupted.", file=sys.stderr)
        return 130

    except Exception:
        logger.exception("Unexpected backup failure.")
        return 1

    finally:
        release_lock(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
