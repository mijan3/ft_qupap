#!/usr/bin/env python3
"""
Run the complete FT-QuPAP automated test pipeline.

This runner follows the project test structure and the notebook's verification
discipline:

    tests/unit/
        Cryptographic primitives, transcript binding, nonce handling,
        QBER, Steane CSS, Gaussian Process inference, and retry policy.

    tests/integration/
        Registration, authenticated ML-KEM bootstrapping, quantum
        transmission, deterministic verification, GP detection, and
        complete authentication.

    tests/end_to_end/
        Normal, noisy-retry, eavesdropping, replay, and forged-tag sessions.

The runner also provides deployment checks for the exported GP detector and
creates machine-readable and human-readable reports without requiring an
external test-reporting plugin.

Default command:
    python scripts/run_all_tests.py

Common commands:
    python scripts/run_all_tests.py --suite unit
    python scripts/run_all_tests.py --suite integration
    python scripts/run_all_tests.py --suite end-to-end
    python scripts/run_all_tests.py --suite all --with-coverage
    python scripts/run_all_tests.py --suite all --parallel auto
    python scripts/run_all_tests.py --keyword replay
    python scripts/run_all_tests.py --list-tests
    python scripts/run_all_tests.py --strict

Generated artifacts:
    outputs/logs/run_all_tests.log
    outputs/reports/test_report.json
    outputs/reports/test_report.html
    outputs/reports/junit/*.xml

Exit codes:
    0   All requested checks and tests passed.
    1   One or more checks/tests failed.
    2   Invalid command-line configuration.
    130 Interrupted by the user.

Security note:
The test runner never prints private keys, ML-KEM shared secrets, K_auth,
K_ctrl, raw subscriber identities, or reusable authentication tags.
"""

from __future__ import annotations

import argparse
import compileall
import contextlib
import html
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

UNIT_TEST_DIR = TESTS_DIR / "unit"
INTEGRATION_TEST_DIR = TESTS_DIR / "integration"
END_TO_END_TEST_DIR = TESTS_DIR / "end_to_end"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_VALIDATOR = SCRIPTS_DIR / "validate_model_files.py"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"
JUNIT_DIR = REPORT_DIR / "junit"

LOG_FILE = LOG_DIR / "run_all_tests.log"
JSON_REPORT_FILE = REPORT_DIR / "test_report.json"
HTML_REPORT_FILE = REPORT_DIR / "test_report.html"

PROTOCOL_NAME = "FT-QuPAP"
PROTOCOL_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
MASTER_SEED = 20260701
MINIMUM_PYTHON = (3, 10)

MODEL_BUNDLE_FILES = (
    MODEL_DIR / "gp_model.pkl",
    MODEL_DIR / "feature_scaler.pkl",
    MODEL_DIR / "calibration_model.pkl",
    MODEL_DIR / "threshold.json",
    MODEL_DIR / "feature_order.json",
    MODEL_DIR / "model_metadata.json",
)

SUITE_DIRECTORIES: dict[str, Path] = {
    "unit": UNIT_TEST_DIR,
    "integration": INTEGRATION_TEST_DIR,
    "end-to-end": END_TO_END_TEST_DIR,
}

PYTEST_EXIT_MEANINGS = {
    0: "all tests passed",
    1: "one or more tests failed",
    2: "test execution was interrupted",
    3: "pytest internal error",
    4: "pytest command-line usage error",
    5: "no tests were collected",
}


class TestRunnerError(RuntimeError):
    """Raised when the test runner cannot start safely."""


@dataclass(frozen=True)
class CheckResult:
    """Result of one non-pytest validation stage."""

    name: str
    status: str
    duration_seconds: float
    message: str
    return_code: int = 0


@dataclass(frozen=True)
class TestCounts:
    """Counts parsed from one JUnit XML file."""

    tests: int = 0
    passed: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class SuiteResult:
    """Result of one pytest suite execution."""

    suite: str
    path: str
    status: str
    return_code: int
    return_code_meaning: str
    duration_seconds: float
    command: list[str]
    junit_file: str
    counts: TestCounts
    output_tail: list[str]


@dataclass
class TestRunReport:
    """Complete report for one invocation of this script."""

    protocol: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION
    started_at_utc: str = ""
    finished_at_utc: str = ""
    status: str = "RUNNING"
    project_root: str = ""
    seed: int = MASTER_SEED
    requested_suite: str = "all"
    strict_mode: bool = False
    coverage_enabled: bool = False
    parallel_mode: str = "disabled"
    command_line: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    suites: list[SuiteResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    totals: TestCounts = field(default_factory=TestCounts)

    def finalize(self) -> None:
        """Calculate totals and final status."""

        self.finished_at_utc = utc_now_iso()

        totals = TestCounts(
            tests=sum(item.counts.tests for item in self.suites),
            passed=sum(item.counts.passed for item in self.suites),
            failures=sum(item.counts.failures for item in self.suites),
            errors=sum(item.counts.errors for item in self.suites),
            skipped=sum(item.counts.skipped for item in self.suites),
            xfailed=sum(item.counts.xfailed for item in self.suites),
            duration_seconds=sum(
                item.duration_seconds for item in self.suites
            ),
        )
        self.totals = totals

        check_failed = any(
            item.status == "FAILED"
            for item in self.checks
        )
        suite_failed = any(
            item.status != "PASSED"
            for item in self.suites
        )

        if check_failed or suite_failed:
            self.status = "FAILED"
        elif self.strict_mode and self.warnings:
            self.status = "FAILED_STRICT"
        elif self.warnings:
            self.status = "PASSED_WITH_WARNINGS"
        else:
            self.status = "PASSED"

    def to_dictionary(self) -> dict[str, Any]:
        """Convert nested dataclasses into JSON-compatible data."""

        return asdict(self)


def utc_now_iso() -> str:
    """Return a timezone-aware ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging() -> logging.Logger:
    """Configure console and persistent runner logging."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ft_qupap.run_all_tests")
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
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run FT-QuPAP unit, integration, and end-to-end tests "
            "with reproducible environment settings and reports."
        )
    )
    parser.add_argument(
        "--suite",
        choices=("all", "unit", "integration", "end-to-end"),
        default="all",
        help="Test category to run (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MASTER_SEED,
        help=f"Deterministic test seed (default: {MASTER_SEED}).",
    )
    parser.add_argument(
        "--keyword",
        default=None,
        help="Pass a pytest -k expression.",
    )
    parser.add_argument(
        "--marker",
        default=None,
        help="Pass a pytest -m marker expression.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Use verbose pytest output.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop each suite after the first failure.",
    )
    parser.add_argument(
        "--stop-after-suite-failure",
        action="store_true",
        help="Do not start later suites after a suite fails.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat Python warnings as errors and runner warnings as "
            "a failed final status."
        ),
    )
    parser.add_argument(
        "--with-coverage",
        action="store_true",
        help=(
            "Enable pytest-cov for src, scenarios, and dashboard. "
            "Requires pytest-cov."
        ),
    )
    parser.add_argument(
        "--coverage-fail-under",
        type=float,
        default=80.0,
        help=(
            "Minimum total coverage percentage when --with-coverage "
            "is used (default: 80)."
        ),
    )
    parser.add_argument(
        "--parallel",
        default="disabled",
        help=(
            "pytest-xdist worker count, 'auto', or 'disabled' "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Per-test timeout in seconds. Requires pytest-timeout."
        ),
    )
    parser.add_argument(
        "--skip-compile-check",
        action="store_true",
        help="Skip Python bytecode compilation checks.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Do not run scripts/validate_model_files.py.",
    )
    parser.add_argument(
        "--require-model-bundle",
        action="store_true",
        help=(
            "Fail when the six exported GP model files are unavailable."
        ),
    )
    parser.add_argument(
        "--skip-dataset-model-check",
        action="store_true",
        help=(
            "Ask validate_model_files.py to skip independent-test "
            "metric comparisons."
        ),
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="Collect matching tests without executing them.",
    )
    parser.add_argument(
        "--keep-junit-history",
        action="store_true",
        help=(
            "Store timestamped JUnit files instead of replacing each "
            "suite's latest XML."
        ),
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not create JSON and HTML summary reports.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional pytest arguments after '--', for example: "
            "-- --maxfail=2 -s"
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line values before changing the environment."""

    if args.seed < 0:
        raise TestRunnerError("--seed must be non-negative.")

    if not 0.0 <= args.coverage_fail_under <= 100.0:
        raise TestRunnerError(
            "--coverage-fail-under must be between 0 and 100."
        )

    if args.timeout is not None and args.timeout < 1:
        raise TestRunnerError("--timeout must be at least 1 second.")

    parallel = str(args.parallel).strip().lower()
    if parallel not in {"disabled", "auto"}:
        try:
            workers = int(parallel)
        except ValueError as exc:
            raise TestRunnerError(
                "--parallel must be 'disabled', 'auto', or a "
                "positive integer."
            ) from exc
        if workers < 1:
            raise TestRunnerError(
                "--parallel worker count must be at least 1."
            )

    if args.pytest_args and args.pytest_args[0] == "--":
        args.pytest_args = args.pytest_args[1:]


def ensure_runtime_directories() -> None:
    """Create report and log directories."""

    for directory in (LOG_DIR, REPORT_DIR, JUNIT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def module_available(module_name: str) -> bool:
    """Return whether an importable module is installed."""

    return importlib.util.find_spec(module_name) is not None


def pytest_plugin_available(module_name: str) -> bool:
    """Check availability of a pytest plugin module."""

    return module_available(module_name)


def selected_suites(requested_suite: str) -> list[str]:
    """Resolve one suite selection into an ordered execution plan."""

    if requested_suite == "all":
        return ["unit", "integration", "end-to-end"]
    return [requested_suite]


def relative_or_absolute(path: Path) -> str:
    """Return a project-relative path when possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def command_for_report(command: Sequence[str]) -> list[str]:
    """Remove environment-specific executable prefixes from reports."""

    return [
        relative_or_absolute(Path(item))
        if item.startswith(str(PROJECT_ROOT))
        else item
        for item in command
    ]


def run_subprocess_streamed(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    logger: logging.Logger,
) -> tuple[int, float, list[str]]:
    """
    Run a command while streaming and retaining a bounded output tail.

    The child receives no sensitive data through command-line arguments.
    """

    start = time.perf_counter()
    output_tail: list[str] = []

    logger.info("Running: %s", " ".join(command))

    process = subprocess.Popen(
        list(command),
        cwd=str(PROJECT_ROOT),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None

    try:
        for line in process.stdout:
            clean_line = line.rstrip("\n")
            print(clean_line)
            logger.info("child | %s", clean_line)
            output_tail.append(clean_line)
            if len(output_tail) > 80:
                output_tail.pop(0)
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                process.kill()
        raise

    return_code = process.wait()
    duration = time.perf_counter() - start
    return return_code, duration, output_tail


def build_test_environment(args: argparse.Namespace) -> dict[str, str]:
    """Build deterministic environment variables for every child test."""

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONHASHSEED": str(args.seed),
            "FT_QUPAP_TEST_MODE": "1",
            "FT_QUPAP_MASTER_SEED": str(args.seed),
            "FT_QUPAP_PROTOCOL_VERSION": PROTOCOL_VERSION,
            "MPLBACKEND": "Agg",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    current_python_path = environment.get("PYTHONPATH", "")
    project_path = str(PROJECT_ROOT)
    environment["PYTHONPATH"] = (
        project_path
        if not current_python_path
        else project_path + os.pathsep + current_python_path
    )

    return environment


def preflight_check(
    args: argparse.Namespace,
    report: TestRunReport,
    logger: logging.Logger,
) -> bool:
    """Validate Python, pytest, project paths, and optional plugins."""

    started = time.perf_counter()
    messages: list[str] = []
    failed = False

    if sys.version_info < MINIMUM_PYTHON:
        messages.append(
            "Python "
            f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}+ is required; "
            f"found {platform.python_version()}."
        )
        failed = True
    else:
        messages.append(
            f"Python {platform.python_version()} is supported."
        )

    if not module_available("pytest"):
        messages.append(
            "pytest is not installed. Run setup_environment.py or "
            "install the project requirements."
        )
        failed = True
    else:
        messages.append("pytest is available.")

    for suite in selected_suites(args.suite):
        path = SUITE_DIRECTORIES[suite]
        if not path.is_dir():
            messages.append(
                f"Requested suite directory is missing: "
                f"{relative_or_absolute(path)}"
            )
            failed = True
            continue

        test_files = sorted(path.glob("test_*.py"))
        if not test_files:
            messages.append(
                f"No test_*.py files found in "
                f"{relative_or_absolute(path)}."
            )
            failed = True
        else:
            messages.append(
                f"{suite}: discovered {len(test_files)} test module(s)."
            )

    if args.with_coverage and not pytest_plugin_available("pytest_cov"):
        messages.append(
            "--with-coverage requires pytest-cov, but pytest_cov "
            "is not importable."
        )
        failed = True

    parallel = str(args.parallel).strip().lower()
    if parallel != "disabled" and not pytest_plugin_available("xdist"):
        messages.append(
            "--parallel requires pytest-xdist, but xdist "
            "is not importable."
        )
        failed = True

    if args.timeout is not None and not pytest_plugin_available(
        "pytest_timeout"
    ):
        messages.append(
            "--timeout requires pytest-timeout, but pytest_timeout "
            "is not importable."
        )
        failed = True

    if not SRC_DIR.is_dir():
        messages.append(
            "src/ is unavailable; the implementation is incomplete."
        )
        failed = True

    duration = time.perf_counter() - started
    status = "FAILED" if failed else "PASSED"
    message = " ".join(messages)

    report.checks.append(
        CheckResult(
            name="preflight",
            status=status,
            duration_seconds=duration,
            message=message,
            return_code=1 if failed else 0,
        )
    )

    log_method = logger.error if failed else logger.info
    log_method("Preflight %s: %s", status, message)
    return not failed


def compile_check(
    args: argparse.Namespace,
    report: TestRunReport,
    logger: logging.Logger,
) -> bool:
    """Compile project Python files to detect syntax errors early."""

    if args.skip_compile_check:
        report.warnings.append(
            "Python compile check was skipped by command-line option."
        )
        return True

    started = time.perf_counter()
    locations = [
        path
        for path in (
            SRC_DIR,
            SCENARIOS_DIR,
            DASHBOARD_DIR,
            SCRIPTS_DIR,
            TESTS_DIR,
        )
        if path.exists()
    ]

    if not locations:
        report.checks.append(
            CheckResult(
                name="compile",
                status="FAILED",
                duration_seconds=0.0,
                message="No project Python directories were found.",
                return_code=1,
            )
        )
        return False

    all_ok = True
    checked: list[str] = []

    for location in locations:
        checked.append(relative_or_absolute(location))
        success = compileall.compile_dir(
            str(location),
            quiet=1,
            force=False,
            legacy=False,
        )
        all_ok = all_ok and bool(success)

    duration = time.perf_counter() - started
    status = "PASSED" if all_ok else "FAILED"
    message = (
        f"Compiled Python modules under: {', '.join(checked)}."
        if all_ok
        else "One or more Python files contain syntax errors."
    )

    report.checks.append(
        CheckResult(
            name="compile",
            status=status,
            duration_seconds=duration,
            message=message,
            return_code=0 if all_ok else 1,
        )
    )

    log_method = logger.info if all_ok else logger.error
    log_method("Compile check %s: %s", status, message)
    return all_ok


def model_bundle_state() -> tuple[bool, list[Path]]:
    """Return whether all six GP deployment artifacts exist."""

    missing = [
        path for path in MODEL_BUNDLE_FILES if not path.is_file()
    ]
    return not missing, missing


def validate_model_bundle(
    args: argparse.Namespace,
    report: TestRunReport,
    logger: logging.Logger,
    environment: Mapping[str, str],
) -> bool:
    """Run the independent GP bundle validator when appropriate."""

    if args.skip_model_validation:
        report.warnings.append(
            "Exported GP model validation was skipped."
        )
        return True

    if not MODEL_VALIDATOR.is_file():
        message = (
            "scripts/validate_model_files.py is unavailable."
        )
        if args.require_model_bundle:
            report.checks.append(
                CheckResult(
                    name="model_bundle",
                    status="FAILED",
                    duration_seconds=0.0,
                    message=message,
                    return_code=1,
                )
            )
            return False

        report.warnings.append(message)
        return True

    bundle_complete, missing = model_bundle_state()
    if not bundle_complete:
        missing_names = ", ".join(
            relative_or_absolute(path) for path in missing
        )
        message = (
            "GP model validation not run because the model bundle is "
            f"incomplete: {missing_names}. Run export_gp_model.py first."
        )

        if args.require_model_bundle:
            report.checks.append(
                CheckResult(
                    name="model_bundle",
                    status="FAILED",
                    duration_seconds=0.0,
                    message=message,
                    return_code=1,
                )
            )
            logger.error(message)
            return False

        report.warnings.append(message)
        logger.warning(message)
        return True

    command = [
        sys.executable,
        str(MODEL_VALIDATOR),
        "--no-report",
    ]
    if args.skip_dataset_model_check:
        command.append("--skip-dataset-check")
    if args.strict:
        command.append("--strict")

    return_code, duration, output_tail = run_subprocess_streamed(
        command,
        environment=environment,
        logger=logger,
    )

    status = "PASSED" if return_code == 0 else "FAILED"
    message = (
        "Exported GP model bundle passed independent validation."
        if return_code == 0
        else (
            "Exported GP model bundle validation failed. Last output: "
            + " | ".join(output_tail[-5:])
        )
    )

    report.checks.append(
        CheckResult(
            name="model_bundle",
            status=status,
            duration_seconds=duration,
            message=message,
            return_code=return_code,
        )
    )

    return return_code == 0


def junit_path_for_suite(
    suite: str,
    keep_history: bool,
    run_stamp: str,
) -> Path:
    """Return the JUnit XML output path for one suite."""

    safe_suite = suite.replace("-", "_")
    if keep_history:
        return JUNIT_DIR / f"{safe_suite}_{run_stamp}.xml"
    return JUNIT_DIR / f"{safe_suite}.xml"


def build_pytest_command(
    *,
    args: argparse.Namespace,
    suite: str,
    junit_path: Path,
) -> list[str]:
    """Build one safe pytest command."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        str(SUITE_DIRECTORIES[suite]),
        f"--junitxml={junit_path}",
        "--disable-warnings" if not args.strict else "-W",
    ]

    if args.strict:
        command.append("error")
    else:
        command.extend(["--tb=short"])

    command.append("-vv" if args.verbose else "-q")

    if args.fail_fast:
        command.append("-x")

    if args.keyword:
        command.extend(["-k", args.keyword])

    if args.marker:
        command.extend(["-m", args.marker])

    if args.list_tests:
        command.append("--collect-only")

    parallel = str(args.parallel).strip().lower()
    if parallel != "disabled":
        command.extend(["-n", parallel])

    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])

    if args.with_coverage:
        command.extend(
            [
                "--cov=src",
                "--cov=scenarios",
                "--cov=dashboard",
                "--cov-branch",
                f"--cov-report=term-missing",
                f"--cov-report=xml:{REPORT_DIR / 'coverage.xml'}",
                f"--cov-report=html:{REPORT_DIR / 'coverage_html'}",
                f"--cov-fail-under={args.coverage_fail_under}",
            ]
        )

    command.extend(args.pytest_args)
    return command


def numeric_xml_attribute(
    element: ET.Element,
    name: str,
) -> int:
    """Read an integer JUnit attribute safely."""

    try:
        return int(float(element.attrib.get(name, "0")))
    except (TypeError, ValueError):
        return 0


def float_xml_attribute(
    element: ET.Element,
    name: str,
) -> float:
    """Read a floating-point JUnit attribute safely."""

    try:
        return float(element.attrib.get(name, "0"))
    except (TypeError, ValueError):
        return 0.0


def parse_junit_counts(path: Path) -> TestCounts:
    """Parse pytest JUnit XML across testsuite/testsuites root formats."""

    if not path.is_file():
        return TestCounts()

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return TestCounts()

    suites: list[ET.Element]
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        suites = list(root.iter("testsuite"))

    tests = sum(numeric_xml_attribute(item, "tests") for item in suites)
    failures = sum(
        numeric_xml_attribute(item, "failures") for item in suites
    )
    errors = sum(numeric_xml_attribute(item, "errors") for item in suites)
    skipped = sum(
        numeric_xml_attribute(item, "skipped") for item in suites
    )
    duration = sum(
        float_xml_attribute(item, "time") for item in suites
    )

    # Pytest represents xfail as skipped in standard JUnit XML. Preserve a
    # separate best-effort count by examining testcase properties/elements.
    xfailed = 0
    for testcase in root.iter("testcase"):
        skipped_element = testcase.find("skipped")
        if skipped_element is None:
            continue
        message = (
            skipped_element.attrib.get("message", "")
            + " "
            + (skipped_element.text or "")
        ).lower()
        if "xfail" in message or "expected failure" in message:
            xfailed += 1

    passed = max(0, tests - failures - errors - skipped)

    return TestCounts(
        tests=tests,
        passed=passed,
        failures=failures,
        errors=errors,
        skipped=skipped,
        xfailed=xfailed,
        duration_seconds=duration,
    )


def execute_suite(
    *,
    args: argparse.Namespace,
    suite: str,
    environment: Mapping[str, str],
    logger: logging.Logger,
    run_stamp: str,
) -> SuiteResult:
    """Execute one pytest category and parse its JUnit results."""

    junit_path = junit_path_for_suite(
        suite,
        args.keep_junit_history,
        run_stamp,
    )
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.unlink(missing_ok=True)

    command = build_pytest_command(
        args=args,
        suite=suite,
        junit_path=junit_path,
    )

    return_code, duration, output_tail = run_subprocess_streamed(
        command,
        environment=environment,
        logger=logger,
    )

    counts = parse_junit_counts(junit_path)
    passed = return_code == 0
    status = "PASSED" if passed else "FAILED"

    return SuiteResult(
        suite=suite,
        path=relative_or_absolute(SUITE_DIRECTORIES[suite]),
        status=status,
        return_code=return_code,
        return_code_meaning=PYTEST_EXIT_MEANINGS.get(
            return_code,
            "unknown pytest exit status",
        ),
        duration_seconds=duration,
        command=command_for_report(command),
        junit_file=relative_or_absolute(junit_path),
        counts=counts,
        output_tail=output_tail,
    )


def aggregate_phase_output(
    report: TestRunReport,
    logger: logging.Logger,
) -> None:
    """Log a compact per-suite summary."""

    for suite in report.suites:
        logger.info(
            (
                "%s | %s | tests=%d passed=%d failed=%d errors=%d "
                "skipped=%d duration=%.3fs"
            ),
            suite.status,
            suite.suite,
            suite.counts.tests,
            suite.counts.passed,
            suite.counts.failures,
            suite.counts.errors,
            suite.counts.skipped,
            suite.duration_seconds,
        )


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text file."""

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


def status_css_class(status: str) -> str:
    """Map a status string to a safe CSS class."""

    if status in {"PASSED", "PASSED_WITH_WARNINGS"}:
        return "pass"
    if status == "SKIPPED":
        return "skip"
    return "fail"


def render_html_report(report: TestRunReport) -> str:
    """Create a standalone test-report HTML document."""

    check_rows = []
    for item in report.checks:
        check_rows.append(
            "<tr>"
            f"<td>{html.escape(item.name)}</td>"
            f"<td class='{status_css_class(item.status)}'>"
            f"{html.escape(item.status)}</td>"
            f"<td>{item.duration_seconds:.3f}</td>"
            f"<td>{html.escape(item.message)}</td>"
            "</tr>"
        )

    suite_rows = []
    for item in report.suites:
        suite_rows.append(
            "<tr>"
            f"<td>{html.escape(item.suite)}</td>"
            f"<td class='{status_css_class(item.status)}'>"
            f"{html.escape(item.status)}</td>"
            f"<td>{item.counts.tests}</td>"
            f"<td>{item.counts.passed}</td>"
            f"<td>{item.counts.failures}</td>"
            f"<td>{item.counts.errors}</td>"
            f"<td>{item.counts.skipped}</td>"
            f"<td>{item.duration_seconds:.3f}</td>"
            f"<td><code>{html.escape(item.junit_file)}</code></td>"
            "</tr>"
        )

    warnings_html = (
        "<p>None.</p>"
        if not report.warnings
        else "<ul>"
        + "".join(
            f"<li>{html.escape(message)}</li>"
            for message in report.warnings
        )
        + "</ul>"
    )

    totals = report.totals
    status_class = status_css_class(report.status)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FT-QuPAP Test Report</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 2rem;
    color: #1f2937;
    background: #f8fafc;
}}
main {{
    max-width: 1200px;
    margin: auto;
    background: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
}}
h1, h2 {{ color: #0f172a; }}
.badge {{
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-weight: 700;
}}
.pass {{ color: #166534; font-weight: 700; }}
.fail {{ color: #991b1b; font-weight: 700; }}
.skip {{ color: #854d0e; font-weight: 700; }}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0 2rem;
}}
th, td {{
    border: 1px solid #dbe3ec;
    padding: 0.65rem;
    text-align: left;
    vertical-align: top;
}}
th {{ background: #eef2f7; }}
code {{
    white-space: pre-wrap;
    word-break: break-word;
}}
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 2rem;
}}
.card {{
    border: 1px solid #dbe3ec;
    border-radius: 8px;
    padding: 1rem;
    background: #f8fafc;
}}
.card strong {{
    display: block;
    font-size: 1.5rem;
    margin-top: 0.25rem;
}}
small {{ color: #64748b; }}
</style>
</head>
<body>
<main>
<h1>FT-QuPAP Automated Test Report</h1>
<p>
Status:
<span class="badge {status_class}">{html.escape(report.status)}</span>
</p>
<p>
<small>
Started: {html.escape(report.started_at_utc)}<br>
Finished: {html.escape(report.finished_at_utc)}<br>
Protocol version: {html.escape(report.protocol_version)}<br>
Requested suite: {html.escape(report.requested_suite)}<br>
Seed: {report.seed}
</small>
</p>

<div class="summary-grid">
<div class="card">Tests<strong>{totals.tests}</strong></div>
<div class="card">Passed<strong>{totals.passed}</strong></div>
<div class="card">Failures<strong>{totals.failures}</strong></div>
<div class="card">Errors<strong>{totals.errors}</strong></div>
<div class="card">Skipped<strong>{totals.skipped}</strong></div>
<div class="card">Duration<strong>{totals.duration_seconds:.2f}s</strong></div>
</div>

<h2>Pre-test checks</h2>
<table>
<thead>
<tr><th>Check</th><th>Status</th><th>Seconds</th><th>Details</th></tr>
</thead>
<tbody>
{''.join(check_rows) if check_rows else '<tr><td colspan="4">No checks recorded.</td></tr>'}
</tbody>
</table>

<h2>Pytest suites</h2>
<table>
<thead>
<tr>
<th>Suite</th><th>Status</th><th>Tests</th><th>Passed</th>
<th>Failures</th><th>Errors</th><th>Skipped</th>
<th>Seconds</th><th>JUnit file</th>
</tr>
</thead>
<tbody>
{''.join(suite_rows) if suite_rows else '<tr><td colspan="9">No suites executed.</td></tr>'}
</tbody>
</table>

<h2>Warnings</h2>
{warnings_html}

<h2>Environment</h2>
<pre><code>{html.escape(json.dumps(report.environment, indent=2, sort_keys=True))}</code></pre>
</main>
</body>
</html>
"""


def write_reports(
    report: TestRunReport,
    logger: logging.Logger,
) -> None:
    """Write the latest JSON and standalone HTML test reports."""

    payload = report.to_dictionary()
    atomic_write_json(JSON_REPORT_FILE, payload)
    atomic_write_text(
        HTML_REPORT_FILE,
        render_html_report(report),
    )

    logger.info(
        "Saved JSON report: %s",
        relative_or_absolute(JSON_REPORT_FILE),
    )
    logger.info(
        "Saved HTML report: %s",
        relative_or_absolute(HTML_REPORT_FILE),
    )


def build_environment_report(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collect non-sensitive runtime details for reproducibility."""

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytest_available": module_available("pytest"),
        "pytest_cov_available": pytest_plugin_available("pytest_cov"),
        "pytest_xdist_available": pytest_plugin_available("xdist"),
        "pytest_timeout_available": pytest_plugin_available(
            "pytest_timeout"
        ),
        "cpu_count": os.cpu_count(),
        "python_hash_seed": str(args.seed),
        "ft_qupap_master_seed": str(args.seed),
        "matplotlib_backend": "Agg",
    }


def print_final_summary(report: TestRunReport) -> None:
    """Print a compact terminal summary."""

    totals = report.totals
    print("\n" + "=" * 72)
    print("FT-QuPAP TEST SUMMARY")
    print("=" * 72)
    print(f"Status:   {report.status}")
    print(f"Suites:   {len(report.suites)}")
    print(f"Tests:    {totals.tests}")
    print(f"Passed:   {totals.passed}")
    print(f"Failures: {totals.failures}")
    print(f"Errors:   {totals.errors}")
    print(f"Skipped:  {totals.skipped}")
    print(f"Duration: {totals.duration_seconds:.2f} seconds")

    if report.warnings:
        print(f"Warnings: {len(report.warnings)}")

    print(f"JSON:     {relative_or_absolute(JSON_REPORT_FILE)}")
    print(f"HTML:     {relative_or_absolute(HTML_REPORT_FILE)}")
    print("=" * 72)


def run(args: argparse.Namespace, logger: logging.Logger) -> TestRunReport:
    """Execute the complete requested test workflow."""

    ensure_runtime_directories()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    report = TestRunReport(
        started_at_utc=utc_now_iso(),
        project_root=str(PROJECT_ROOT),
        seed=args.seed,
        requested_suite=args.suite,
        strict_mode=args.strict,
        coverage_enabled=args.with_coverage,
        parallel_mode=str(args.parallel),
        command_line=[sys.executable, *sys.argv],
        environment=build_environment_report(args),
    )

    environment = build_test_environment(args)

    preflight_ok = preflight_check(
        args,
        report,
        logger,
    )
    if not preflight_ok:
        report.finalize()
        return report

    compile_ok = compile_check(
        args,
        report,
        logger,
    )
    if not compile_ok:
        report.finalize()
        return report

    model_ok = validate_model_bundle(
        args,
        report,
        logger,
        environment,
    )
    if not model_ok:
        report.finalize()
        return report

    for suite in selected_suites(args.suite):
        result = execute_suite(
            args=args,
            suite=suite,
            environment=environment,
            logger=logger,
            run_stamp=run_stamp,
        )
        report.suites.append(result)

        if (
            result.status != "PASSED"
            and args.stop_after_suite_failure
        ):
            report.warnings.append(
                "Remaining suites were not executed because "
                "--stop-after-suite-failure was active."
            )
            break

    aggregate_phase_output(report, logger)
    report.finalize()
    return report


def main() -> int:
    """Command-line entry point."""

    logger = configure_logging()

    try:
        args = parse_arguments()
        validate_arguments(args)

        report = run(args, logger)

        if not args.no_report:
            write_reports(report, logger)

        print_final_summary(report)

        return 0 if report.status in {
            "PASSED",
            "PASSED_WITH_WARNINGS",
        } else 1

    except TestRunnerError as exc:
        logger.error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        logger.error("Test run interrupted by user.")
        print("\nTest run interrupted.", file=sys.stderr)
        return 130

    except Exception:
        logger.exception("Unexpected FT-QuPAP test-runner failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
