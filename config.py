"""
FT-QuPAP Application Configuration
===================================

Central configuration module for the FT-QuPAP v5.1 capstone system.

This module manages:

- Project directories
- Protocol parameters
- Post-quantum cryptographic settings
- Quantum payload and Steane-code settings
- QBER and loss thresholds
- Gaussian Process model configuration
- Retry-policy parameters
- Storage paths
- Dashboard settings
- Optional ESP32/Arduino hardware settings
- Logging configuration

Configuration values may be overridden through environment variables or
a project-level ``.env`` file.

No private keys, shared secrets, authentication tags, or subscriber
identities should be placed directly in this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final


# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------

PROJECT_NAME: Final[str] = "FT-QuPAP-Capstone"
PROTOCOL_NAME: Final[str] = "FT-QuPAP"
PROTOCOL_VERSION: Final[str] = "5.1"
PACKAGE_VERSION: Final[str] = "5.1.0"

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Optional .env loading
# ---------------------------------------------------------------------------

def load_environment_file(
    environment_file: Path | None = None,
) -> bool:
    """
    Load environment variables from a ``.env`` file.

    Existing operating-system environment variables are not overwritten.

    Args:
        environment_file:
            Optional path to the environment file. The default is
            ``PROJECT_ROOT/.env``.

    Returns:
        True when a file was found and processed.
    """

    resolved_file = (
        environment_file
        if environment_file is not None
        else PROJECT_ROOT / ".env"
    )

    if not resolved_file.exists():
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(
            dotenv_path=resolved_file,
            override=False,
        )

        return True

    except ImportError:
        return _load_environment_file_without_dependency(
            resolved_file
        )


def _load_environment_file_without_dependency(
    environment_file: Path,
) -> bool:
    """
    Load a simple .env file without python-dotenv.

    This fallback supports standard ``KEY=value`` lines.
    """

    try:
        lines = environment_file.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return False

    for line in lines:
        stripped_line = line.strip()

        if (
            not stripped_line
            or stripped_line.startswith("#")
            or "=" not in stripped_line
        ):
            continue

        key, value = stripped_line.split("=", 1)

        normalized_key = key.strip()
        normalized_value = value.strip()

        if not normalized_key:
            continue

        if (
            len(normalized_value) >= 2
            and normalized_value[0]
            == normalized_value[-1]
            and normalized_value[0] in {'"', "'"}
        ):
            normalized_value = normalized_value[1:-1]

        os.environ.setdefault(
            normalized_key,
            normalized_value,
        )

    return True


load_environment_file()


# ---------------------------------------------------------------------------
# Environment conversion helpers
# ---------------------------------------------------------------------------

def environment_string(
    name: str,
    default: str,
) -> str:
    """Read a non-empty string environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized_value = value.strip()

    return normalized_value or default


def environment_optional_string(
    name: str,
    default: str | None = None,
) -> str | None:
    """Read an optional string environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized_value = value.strip()

    return normalized_value or None


def environment_integer(
    name: str,
    default: int,
) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Environment variable {name} must be an integer."
        ) from error


def environment_float(
    name: str,
    default: float,
) -> float:
    """Read a floating-point environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        converted_value = float(value.strip())
    except ValueError as error:
        raise ValueError(
            f"Environment variable {name} must be numeric."
        ) from error

    if converted_value != converted_value:
        raise ValueError(
            f"Environment variable {name} cannot be NaN."
        )

    if converted_value in {
        float("inf"),
        float("-inf"),
    }:
        raise ValueError(
            f"Environment variable {name} must be finite."
        )

    return converted_value


def environment_boolean(
    name: str,
    default: bool,
) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized_value = value.strip().lower()

    if normalized_value in {
        "1",
        "true",
        "yes",
        "on",
        "enabled",
    }:
        return True

    if normalized_value in {
        "0",
        "false",
        "no",
        "off",
        "disabled",
    }:
        return False

    raise ValueError(
        f"Environment variable {name} must be boolean."
    )


def environment_path(
    name: str,
    default: Path,
) -> Path:
    """Read and resolve a path environment variable."""

    value = os.getenv(name)

    if value is None or not value.strip():
        return default.resolve()

    candidate = Path(value.strip()).expanduser()

    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate

    return candidate.resolve()


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathSettings:
    """Project directory and file locations."""

    project_root: Path = PROJECT_ROOT

    dashboard_directory: Path = (
        PROJECT_ROOT / "dashboard"
    )
    source_directory: Path = PROJECT_ROOT / "src"
    scenarios_directory: Path = (
        PROJECT_ROOT / "scenarios"
    )
    models_directory: Path = PROJECT_ROOT / "models"
    data_directory: Path = PROJECT_ROOT / "data"
    database_directory: Path = (
        PROJECT_ROOT / "database"
    )
    notebooks_directory: Path = (
        PROJECT_ROOT / "notebooks"
    )
    tests_directory: Path = PROJECT_ROOT / "tests"
    hardware_directory: Path = (
        PROJECT_ROOT / "hardware"
    )
    assets_directory: Path = PROJECT_ROOT / "assets"
    outputs_directory: Path = (
        PROJECT_ROOT / "outputs"
    )
    docs_directory: Path = PROJECT_ROOT / "docs"
    scripts_directory: Path = (
        PROJECT_ROOT / "scripts"
    )

    raw_data_directory: Path = (
        PROJECT_ROOT / "data" / "raw"
    )
    processed_data_directory: Path = (
        PROJECT_ROOT / "data" / "processed"
    )
    demo_data_directory: Path = (
        PROJECT_ROOT / "data" / "demo"
    )
    result_data_directory: Path = (
        PROJECT_ROOT / "data" / "results"
    )

    figure_output_directory: Path = (
        PROJECT_ROOT / "outputs" / "figures"
    )
    report_output_directory: Path = (
        PROJECT_ROOT / "outputs" / "reports"
    )
    log_output_directory: Path = (
        PROJECT_ROOT / "outputs" / "logs"
    )

    subscriber_database_file: Path = (
        PROJECT_ROOT
        / "database"
        / "subscribers.json"
    )
    nonce_database_file: Path = (
        PROJECT_ROOT
        / "database"
        / "used_nonces.json"
    )
    registration_database_file: Path = (
        PROJECT_ROOT
        / "database"
        / "registration_records.json"
    )
    trusted_server_keys_file: Path = (
        PROJECT_ROOT
        / "database"
        / "trusted_server_keys.json"
    )
    session_database_file: Path = (
        PROJECT_ROOT
        / "database"
        / "demo_sessions.json"
    )

    dashboard_results_file: Path = (
        PROJECT_ROOT
        / "data"
        / "demo"
        / "dashboard_results.csv"
    )
    demo_session_logs_file: Path = (
        PROJECT_ROOT
        / "data"
        / "demo"
        / "demo_session_logs.csv"
    )


# ---------------------------------------------------------------------------
# Protocol configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProtocolSettings:
    """Core FT-QuPAP protocol settings."""

    protocol_name: str = PROTOCOL_NAME
    protocol_version: str = PROTOCOL_VERSION

    network_id: str = field(
        default_factory=lambda: environment_string(
            "FT_QUPAP_NETWORK_ID",
            "FT-QUPAP-DEMO-NETWORK",
        )
    )

    freshness_window_seconds: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_FRESHNESS_WINDOW_SECONDS",
            60,
        )
    )

    maximum_authentication_attempts: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_MAXIMUM_ATTEMPTS",
            3,
        )
    )

    maximum_retries: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_MAXIMUM_RETRIES",
            2,
        )
    )

    supported_contexts: tuple[str, ...] = (
        "urban",
        "suburban",
        "rural",
    )

    default_context: str = field(
        default_factory=lambda: environment_string(
            "FT_QUPAP_DEFAULT_CONTEXT",
            "urban",
        ).lower()
    )

    deterministic_verification_required: bool = True
    transcript_binding_required: bool = True
    freshness_check_required: bool = True
    replay_check_required: bool = True


# ---------------------------------------------------------------------------
# Cryptographic configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CryptographySettings:
    """Post-quantum and symmetric cryptographic settings."""

    ml_dsa_parameter_set: str = "ML-DSA-65"
    ml_kem_parameter_set: str = "ML-KEM-768"

    kmac_algorithm: str = "KMAC256"
    kmac_tag_bytes: int = 16
    kmac_customization_string: str = (
        "FT-QuPAP-v5.1-Authentication"
    )

    kdf_algorithm: str = "KMAC256"
    transcript_hash_algorithm: str = "SHA3-256"
    pseudonymous_identity_algorithm: str = "SHA3-256"

    nonce_bytes: int = 32
    session_identifier_bytes: int = 16

    authentication_key_bytes: int = 32
    control_key_bytes: int = 32

    enforce_secure_comparison: bool = True
    require_server_signature: bool = True
    require_mlkem_ciphertext_binding: bool = True

    @property
    def kmac_tag_bits(self) -> int:
        """Return the KMAC tag size in bits."""

        return self.kmac_tag_bytes * 8


# ---------------------------------------------------------------------------
# Quantum configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantumSettings:
    """Quantum payload, Steane code, channel, and QBER settings."""

    logical_payload_blocks: int = 128
    independent_check_blocks: int = 32

    steane_physical_qubits_per_block: int = 7
    steane_logical_qubits_per_block: int = 1
    steane_code_distance: int = 3

    fixed_qber_threshold: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_FIXED_QBER_THRESHOLD",
            0.11,
        )
    )

    maximum_loss_rate: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_MAXIMUM_LOSS_RATE",
            0.15,
        )
    )

    minimum_observed_check_blocks: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_MINIMUM_OBSERVED_CHECK_BLOCKS",
            24,
        )
    )

    enable_error_correction: bool = True
    reject_uncorrectable_errors: bool = True
    randomize_check_bases: bool = True
    interleave_payload_and_check_blocks: bool = True

    @property
    def total_logical_blocks(self) -> int:
        """Return payload blocks plus independent check blocks."""

        return (
            self.logical_payload_blocks
            + self.independent_check_blocks
        )

    @property
    def total_physical_qubits(self) -> int:
        """Return physical qubits after Steane encoding."""

        return (
            self.total_logical_blocks
            * self.steane_physical_qubits_per_block
        )


# ---------------------------------------------------------------------------
# Machine-learning configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MachineLearningSettings:
    """Gaussian Process attack-detection settings."""

    enabled: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_GP_ENABLED",
            True,
        )
    )

    model_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_GP_MODEL_FILE",
            PROJECT_ROOT / "models" / "gp_model.pkl",
        )
    )

    feature_scaler_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_FEATURE_SCALER_FILE",
            PROJECT_ROOT
            / "models"
            / "feature_scaler.pkl",
        )
    )

    calibration_model_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_CALIBRATION_MODEL_FILE",
            PROJECT_ROOT
            / "models"
            / "calibration_model.pkl",
        )
    )

    threshold_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_THRESHOLD_FILE",
            PROJECT_ROOT
            / "models"
            / "threshold.json",
        )
    )

    feature_order_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_FEATURE_ORDER_FILE",
            PROJECT_ROOT
            / "models"
            / "feature_order.json",
        )
    )

    model_metadata_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_MODEL_METADATA_FILE",
            PROJECT_ROOT
            / "models"
            / "model_metadata.json",
        )
    )

    raw_calibrated_threshold: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_RAW_GP_THRESHOLD",
            0.25,
        )
    )

    minimum_operational_threshold: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_MINIMUM_GP_THRESHOLD",
            0.15,
        )
    )

    retry_upper_probability: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_GP_RETRY_UPPER_BOUND",
            0.20,
        )
    )

    calibration_required: bool = True
    feature_scaling_required: bool = True

    @property
    def operational_threshold(self) -> float:
        """
        Return the protected operational GP threshold.

        The operational threshold is never lower than the configured
        minimum threshold.
        """

        return max(
            self.raw_calibrated_threshold,
            self.minimum_operational_threshold,
        )


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageSettings:
    """JSON and CSV persistence settings."""

    subscriber_database_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_SUBSCRIBER_DATABASE",
            PROJECT_ROOT
            / "database"
            / "subscribers.json",
        )
    )

    nonce_database_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_NONCE_DATABASE",
            PROJECT_ROOT
            / "database"
            / "used_nonces.json",
        )
    )

    registration_database_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_REGISTRATION_DATABASE",
            PROJECT_ROOT
            / "database"
            / "registration_records.json",
        )
    )

    session_database_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_SESSION_DATABASE",
            PROJECT_ROOT
            / "database"
            / "demo_sessions.json",
        )
    )

    result_csv_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_RESULT_CSV",
            PROJECT_ROOT
            / "data"
            / "demo"
            / "dashboard_results.csv",
        )
    )

    protocol_csv_file: Path = field(
        default_factory=lambda: environment_path(
            "FT_QUPAP_PROTOCOL_CSV",
            PROJECT_ROOT
            / "data"
            / "demo"
            / "demo_session_logs.csv",
        )
    )

    pretty_print_json: bool = True
    atomic_writes: bool = True
    retain_nonce_digests: bool = True

    maximum_nonce_records: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_MAXIMUM_NONCE_RECORDS",
            100_000,
        )
    )

    maximum_session_records: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_MAXIMUM_SESSION_RECORDS",
            50_000,
        )
    )


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareSettings:
    """Optional ESP32/Arduino decision-indicator settings."""

    enabled: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_HARDWARE_ENABLED",
            False,
        )
    )

    serial_port: str | None = field(
        default_factory=lambda: environment_optional_string(
            "FT_QUPAP_SERIAL_PORT"
        )
    )

    baudrate: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_SERIAL_BAUDRATE",
            115_200,
        )
    )

    timeout_seconds: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_SERIAL_TIMEOUT",
            1.0,
        )
    )

    write_timeout_seconds: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_SERIAL_WRITE_TIMEOUT",
            1.0,
        )
    )

    reset_delay_seconds: float = field(
        default_factory=lambda: environment_float(
            "FT_QUPAP_HARDWARE_RESET_DELAY",
            2.0,
        )
    )

    use_fallback: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_HARDWARE_FALLBACK",
            True,
        )
    )

    fallback_echo: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_HARDWARE_FALLBACK_ECHO",
            False,
        )
    )

    wait_for_acknowledgement: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_HARDWARE_WAIT_FOR_ACK",
            False,
        )
    )

    accepted_command: str = "GREEN"
    retry_command: str = "YELLOW"
    rejected_command: str = "RED"


# ---------------------------------------------------------------------------
# Dashboard configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DashboardSettings:
    """Streamlit dashboard settings."""

    page_title: str = "FT-QuPAP Capstone Demo"
    page_icon: str = "🔐"
    layout: str = "wide"

    show_sidebar: bool = True
    show_protocol_details: bool = True
    show_advanced_metrics: bool = True
    show_session_history: bool = True

    refresh_interval_seconds: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_DASHBOARD_REFRESH_SECONDS",
            2,
        )
    )

    maximum_visible_sessions: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_DASHBOARD_MAX_SESSIONS",
            100,
        )
    )

    default_scenario: str = "normal_session"


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggingSettings:
    """Application logging settings."""

    level: str = field(
        default_factory=lambda: environment_string(
            "FT_QUPAP_LOG_LEVEL",
            "INFO",
        ).upper()
    )

    console_enabled: bool = True
    file_logging_enabled: bool = True

    protocol_log_file: Path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "protocol.log"
    )

    authentication_log_file: Path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "authentication.log"
    )

    attack_detection_log_file: Path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "attack_detection.log"
    )

    hardware_log_file: Path = (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / "hardware.log"
    )

    maximum_file_size_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5


# ---------------------------------------------------------------------------
# Complete application configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApplicationConfig:
    """Complete FT-QuPAP application configuration."""

    project_name: str = PROJECT_NAME
    package_version: str = PACKAGE_VERSION

    environment: str = field(
        default_factory=lambda: environment_string(
            "FT_QUPAP_ENVIRONMENT",
            "development",
        ).lower()
    )

    debug: bool = field(
        default_factory=lambda: environment_boolean(
            "FT_QUPAP_DEBUG",
            False,
        )
    )

    random_seed: int = field(
        default_factory=lambda: environment_integer(
            "FT_QUPAP_RANDOM_SEED",
            9102,
        )
    )

    paths: PathSettings = field(
        default_factory=PathSettings
    )

    protocol: ProtocolSettings = field(
        default_factory=ProtocolSettings
    )

    cryptography: CryptographySettings = field(
        default_factory=CryptographySettings
    )

    quantum: QuantumSettings = field(
        default_factory=QuantumSettings
    )

    machine_learning: MachineLearningSettings = field(
        default_factory=MachineLearningSettings
    )

    storage: StorageSettings = field(
        default_factory=StorageSettings
    )

    hardware: HardwareSettings = field(
        default_factory=HardwareSettings
    )

    dashboard: DashboardSettings = field(
        default_factory=DashboardSettings
    )

    logging: LoggingSettings = field(
        default_factory=LoggingSettings
    )

    def validate(self) -> None:
        """Validate all protocol and application settings."""

        validate_application_config(self)

    def create_required_directories(self) -> None:
        """Create runtime directories required by the application."""

        create_required_directories(self)

    def to_dictionary(
        self,
        *,
        convert_paths: bool = True,
    ) -> dict[str, Any]:
        """Return the configuration as a dictionary."""

        data = asdict(self)

        if convert_paths:
            return convert_paths_to_strings(data)

        return data

    def to_public_dictionary(self) -> dict[str, Any]:
        """
        Return dashboard-safe configuration information.

        Serial device information and future secret configuration
        fields are intentionally excluded.
        """

        return {
            "project_name": self.project_name,
            "package_version": self.package_version,
            "environment": self.environment,
            "debug": self.debug,
            "protocol": {
                "name": self.protocol.protocol_name,
                "version": self.protocol.protocol_version,
                "network_id": self.protocol.network_id,
                "freshness_window_seconds": (
                    self.protocol.freshness_window_seconds
                ),
                "maximum_authentication_attempts": (
                    self.protocol
                    .maximum_authentication_attempts
                ),
                "maximum_retries": (
                    self.protocol.maximum_retries
                ),
                "default_context": (
                    self.protocol.default_context
                ),
            },
            "cryptography": {
                "ml_dsa": (
                    self.cryptography
                    .ml_dsa_parameter_set
                ),
                "ml_kem": (
                    self.cryptography
                    .ml_kem_parameter_set
                ),
                "kmac": (
                    self.cryptography.kmac_algorithm
                ),
                "kmac_tag_bits": (
                    self.cryptography.kmac_tag_bits
                ),
            },
            "quantum": {
                "steane_code": "[[7,1,3]]",
                "logical_payload_blocks": (
                    self.quantum.logical_payload_blocks
                ),
                "independent_check_blocks": (
                    self.quantum
                    .independent_check_blocks
                ),
                "total_logical_blocks": (
                    self.quantum.total_logical_blocks
                ),
                "total_physical_qubits": (
                    self.quantum.total_physical_qubits
                ),
                "fixed_qber_threshold": (
                    self.quantum.fixed_qber_threshold
                ),
                "maximum_loss_rate": (
                    self.quantum.maximum_loss_rate
                ),
                "minimum_observed_check_blocks": (
                    self.quantum
                    .minimum_observed_check_blocks
                ),
            },
            "machine_learning": {
                "enabled": (
                    self.machine_learning.enabled
                ),
                "raw_calibrated_threshold": (
                    self.machine_learning
                    .raw_calibrated_threshold
                ),
                "minimum_operational_threshold": (
                    self.machine_learning
                    .minimum_operational_threshold
                ),
                "operational_threshold": (
                    self.machine_learning
                    .operational_threshold
                ),
                "retry_upper_probability": (
                    self.machine_learning
                    .retry_upper_probability
                ),
            },
            "hardware": {
                "enabled": self.hardware.enabled,
                "use_fallback": (
                    self.hardware.use_fallback
                ),
                "baudrate": self.hardware.baudrate,
            },
        }


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_application_config(
    config: ApplicationConfig,
) -> None:
    """Validate FT-QuPAP configuration consistency."""

    if config.protocol.default_context not in (
        config.protocol.supported_contexts
    ):
        raise ValueError(
            "The default context must be one of: "
            + ", ".join(
                config.protocol.supported_contexts
            )
        )

    if config.protocol.freshness_window_seconds <= 0:
        raise ValueError(
            "Freshness window must be greater than zero."
        )

    if (
        config.protocol
        .maximum_authentication_attempts
        < 1
    ):
        raise ValueError(
            "At least one authentication attempt is required."
        )

    if config.protocol.maximum_retries < 0:
        raise ValueError(
            "Maximum retries cannot be negative."
        )

    expected_attempts = (
        config.protocol.maximum_retries + 1
    )

    if (
        config.protocol
        .maximum_authentication_attempts
        != expected_attempts
    ):
        raise ValueError(
            "maximum_authentication_attempts must equal "
            "maximum_retries + 1."
        )

    if config.cryptography.kmac_tag_bytes < 16:
        raise ValueError(
            "The KMAC authentication tag must be at least "
            "16 bytes for this project."
        )

    if (
        config.quantum.logical_payload_blocks
        != config.cryptography.kmac_tag_bits
    ):
        raise ValueError(
            "Logical payload block count must match the "
            "KMAC authentication tag bit length."
        )

    if (
        config.quantum
        .minimum_observed_check_blocks
        > config.quantum.independent_check_blocks
    ):
        raise ValueError(
            "Minimum observed check blocks cannot exceed "
            "the generated independent check blocks."
        )

    validate_probability(
        "fixed_qber_threshold",
        config.quantum.fixed_qber_threshold,
    )

    validate_probability(
        "maximum_loss_rate",
        config.quantum.maximum_loss_rate,
    )

    validate_probability(
        "raw_calibrated_threshold",
        config.machine_learning.raw_calibrated_threshold,
    )

    validate_probability(
        "minimum_operational_threshold",
        config.machine_learning
        .minimum_operational_threshold,
    )

    validate_probability(
        "retry_upper_probability",
        config.machine_learning
        .retry_upper_probability,
    )

    if (
        config.machine_learning
        .minimum_operational_threshold
        >= config.machine_learning
        .retry_upper_probability
    ):
        raise ValueError(
            "The minimum operational GP threshold must be "
            "lower than the retry upper probability."
        )

    if config.hardware.baudrate <= 0:
        raise ValueError(
            "Hardware baud rate must be positive."
        )

    if config.dashboard.refresh_interval_seconds < 1:
        raise ValueError(
            "Dashboard refresh interval must be positive."
        )

    supported_log_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }

    if config.logging.level not in supported_log_levels:
        raise ValueError(
            "Unsupported logging level: "
            f"{config.logging.level}"
        )


def validate_probability(
    name: str,
    value: float,
) -> None:
    """Validate a probability in the inclusive range [0, 1]."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0."
        )


# ---------------------------------------------------------------------------
# Directory initialization
# ---------------------------------------------------------------------------

def create_required_directories(
    config: ApplicationConfig,
) -> None:
    """Create runtime directories used by FT-QuPAP."""

    required_directories = {
        config.paths.database_directory,
        config.paths.models_directory,
        config.paths.raw_data_directory,
        config.paths.processed_data_directory,
        config.paths.demo_data_directory,
        config.paths.result_data_directory,
        config.paths.figure_output_directory,
        config.paths.report_output_directory,
        config.paths.log_output_directory,
    }

    for directory in required_directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def initialize_empty_storage_files(
    config: ApplicationConfig,
) -> None:
    """
    Create initial JSON storage files when they do not exist.

    Existing files are never overwritten.
    """

    json_files_and_defaults: dict[
        Path,
        dict[str, Any],
    ] = {
        config.storage.subscriber_database_file: {
            "version": PROTOCOL_VERSION,
            "subscribers": [],
        },
        config.storage.nonce_database_file: {
            "version": PROTOCOL_VERSION,
            "nonce_records": [],
        },
        config.storage.registration_database_file: {
            "version": PROTOCOL_VERSION,
            "registration_records": [],
        },
        config.storage.session_database_file: {
            "version": PROTOCOL_VERSION,
            "sessions": [],
        },
    }

    for file_path, initial_data in (
        json_files_and_defaults.items()
    ):
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if file_path.exists():
            continue

        file_path.write_text(
            json.dumps(
                initial_data,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def convert_paths_to_strings(
    value: Any,
) -> Any:
    """Recursively convert Path objects into strings."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): convert_paths_to_strings(
                nested_value
            )
            for key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            convert_paths_to_strings(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_paths_to_strings(item)
            for item in value
        ]

    return value


# ---------------------------------------------------------------------------
# Public configuration access
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> ApplicationConfig:
    """
    Return the validated application configuration.

    The configuration is cached so every component uses the same
    immutable settings object.
    """

    config = ApplicationConfig()
    config.validate()

    return config


def reload_config() -> ApplicationConfig:
    """
    Clear the configuration cache and read environment values again.
    """

    get_config.cache_clear()
    load_environment_file()

    return get_config()


def initialize_application_environment(
) -> ApplicationConfig:
    """
    Validate configuration and prepare runtime directories/files.
    """

    config = get_config()

    config.create_required_directories()
    initialize_empty_storage_files(config)

    return config


# Default shared configuration instance.
settings = get_config()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_self_test() -> None:
    """Run basic configuration consistency checks."""

    config = get_config()

    assert config.project_name == PROJECT_NAME
    assert config.protocol.protocol_version == "5.1"

    assert (
        config.cryptography.ml_dsa_parameter_set
        == "ML-DSA-65"
    )

    assert (
        config.cryptography.ml_kem_parameter_set
        == "ML-KEM-768"
    )

    assert config.cryptography.kmac_tag_bits == 128

    assert config.quantum.logical_payload_blocks == 128
    assert config.quantum.independent_check_blocks == 32
    assert config.quantum.total_logical_blocks == 160
    assert config.quantum.total_physical_qubits == 1120

    assert config.quantum.fixed_qber_threshold == 0.11
    assert config.quantum.maximum_loss_rate == 0.15

    assert (
        config.protocol.maximum_authentication_attempts
        == 3
    )

    assert config.protocol.maximum_retries == 2

    public_config = config.to_public_dictionary()

    assert "cryptography" in public_config
    assert "quantum" in public_config
    assert "machine_learning" in public_config

    print("FT-QuPAP configuration self-test passed.")


if __name__ == "__main__":
    run_self_test()