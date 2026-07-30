"""
Central configuration for the FT-QuPAP v5.1 capstone project.

This module stores:

- Project paths
- ML-DSA and ML-KEM algorithm names
- KMAC tag size
- Payload and check-qubit counts
- Steane CSS parameters
- Freshness and replay settings
- QBER and Gaussian Process policy settings
- Retry configuration
- Streamlit and hardware settings

The trained GP attack threshold is not permanently hard-coded here.
It is loaded from models/threshold.json after model calibration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


# ---------------------------------------------------------------------
# Environment helper functions
# ---------------------------------------------------------------------

def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Read a Boolean value from an environment variable.

    Accepted true values:
        true, 1, yes, y, on

    Accepted false values:
        false, 0, no, n, off
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"true", "1", "yes", "y", "on"}:
        return True

    if normalized in {"false", "0", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must contain a Boolean value."
    )


def get_env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must contain an integer."
        ) from exc


def get_env_float(name: str, default: float) -> float:
    """Read a floating-point value from an environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must contain a number."
        ) from exc


def resolve_project_path(
    environment_name: str,
    default_relative_path: str,
) -> Path:
    """
    Read a project-relative path from the environment.

    An absolute path remains unchanged. A relative path is resolved from
    the FT-QuPAP project root.
    """
    configured_value = os.getenv(
        environment_name,
        default_relative_path,
    )

    configured_path = Path(configured_value)

    if configured_path.is_absolute():
        return configured_path

    return PROJECT_ROOT / configured_path


# ---------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------

PROJECT_NAME = "FT-QuPAP"
PROJECT_FULL_NAME = (
    "Fault-Tolerant Quantum Authentication Protocol with "
    "Post-Quantum Bootstrapping and Adaptive Eavesdropping Detection"
)

PROJECT_VERSION = "5.1"
SIMULATOR_VERSION = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)

SERVER_ID = "AS-6G-001"
DEFAULT_PSEUDONYM_ID = "PID-6G-UE-0001"

SUPPORTED_CONTEXTS = (
    "urban",
    "suburban",
    "rural",
)


# ---------------------------------------------------------------------
# Cryptographic configuration
# ---------------------------------------------------------------------

ML_DSA_ALGORITHM = "ML-DSA-65"
ML_KEM_ALGORITHM = "ML-KEM-768"

KMAC_ALGORITHM = "KMAC256"
KMAC_TAG_BYTES = 16
KMAC_TAG_BITS = KMAC_TAG_BYTES * 8

SESSION_KEY_DERIVATION_LABEL = b"FT-QuPAP"
AUTHENTICATION_KEY_LABEL = b"K_auth"
CONTROL_KEY_LABEL = b"K_ctrl"

TRANSCRIPT_HASH_ALGORITHM = "SHA3-256"


# ---------------------------------------------------------------------
# Quantum payload configuration
# ---------------------------------------------------------------------

PAYLOAD_LOGICAL_QUBITS = KMAC_TAG_BITS
CHECK_LOGICAL_QUBITS = 32

TOTAL_LOGICAL_QUBITS = (
    PAYLOAD_LOGICAL_QUBITS
    + CHECK_LOGICAL_QUBITS
)

STEANE_CODE_NAME = "[[7,1,3]]"
STEANE_PHYSICAL_QUBITS_PER_LOGICAL = 7

TOTAL_PHYSICAL_QUBITS = (
    TOTAL_LOGICAL_QUBITS
    * STEANE_PHYSICAL_QUBITS_PER_LOGICAL
)

MINIMUM_OBSERVED_CHECK_BLOCKS = get_env_int(
    "FT_QUPAP_MINIMUM_CHECK_BLOCKS",
    24,
)


# ---------------------------------------------------------------------
# Authentication freshness and replay configuration
# ---------------------------------------------------------------------

FRESHNESS_WINDOW_SECONDS = 60

NONCE_SIZE_BYTES = 16

REQUEST_TYPE = "FT-QuPAP-Authentication"


# ---------------------------------------------------------------------
# Fixed QBER policy
# ---------------------------------------------------------------------

FIXED_QBER_THRESHOLD = 0.11

MAXIMUM_ACCEPTABLE_LOSS_RATE = 0.15


# ---------------------------------------------------------------------
# Gaussian Process policy
# ---------------------------------------------------------------------

FEATURE_COLUMNS = (
    "qber_raw",
    "mean_syndrome_weight",
    "max_syndrome_weight",
    "correction_failure_rate",
    "loss_rate",
    "noise_estimate",
    "ctx_urban",
    "ctx_suburban",
    "ctx_rural",
)

BAYES_COST_FALSE_ACCEPT = 10.0
BAYES_COST_FALSE_REJECT = 1.0

# The final calibrated threshold is generated during model training and
# stored inside models/threshold.json.
DEFAULT_GP_ATTACK_THRESHOLD: Optional[float] = None

# Prevents the generated threshold from becoming too permissive.
MINIMUM_OPERATIONAL_GP_THRESHOLD = 0.15

# A low-risk rejected session may enter the retry gray zone only when its
# attack probability remains below this upper boundary.
GP_GRAY_ZONE_RETRY_UPPER = 0.20

# None means no separate predictive-uncertainty rejection boundary is
# enforced unless later configured by the research team.
GP_MAXIMUM_UNCERTAINTY: Optional[float] = None


# ---------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------

MAXIMUM_RETRIES = get_env_int(
    "FT_QUPAP_MAX_RETRIES",
    2,
)

MAXIMUM_AUTHENTICATION_ATTEMPTS = 1 + MAXIMUM_RETRIES

RETRY_ATTACK_PROBABILITY_LIMIT = 0.20
RETRY_QBER_LIMIT = 0.11


# Channel-context assumptions

CONTEXT_CHANNEL_PROFILES = {
    "urban": {
        "noise_scale": 1.20,
        "loss_scale": 0.90,
    },
    "suburban": {
        "noise_scale": 1.00,
        "loss_scale": 1.00,
    },
    "rural": {
        "noise_scale": 0.85,
        "loss_scale": 1.40,
    },
}

CONTEXT_BASE_NOISE = {
    "urban": 0.012,
    "suburban": 0.008,
    "rural": 0.005,
}


# Reproducibility configuration

RANDOM_SEED = get_env_int(
    "FT_QUPAP_RANDOM_SEED",
    42,
)

DEMO_MODE = get_env_bool(
    "FT_QUPAP_DEMO_MODE",
    True,
)


# Streamlit server configuration

STREAMLIT_HOST = os.getenv(
    "FT_QUPAP_HOST",
    "0.0.0.0",
)

STREAMLIT_PORT = get_env_int(
    "FT_QUPAP_PORT",
    8501,
)


# Hardware configuration

HARDWARE_ENABLED = get_env_bool(
    "FT_QUPAP_ENABLE_HARDWARE",
    False,
)

SERIAL_PORT = os.getenv(
    "FT_QUPAP_SERIAL_PORT",
    "COM3",
)

SERIAL_BAUDRATE = get_env_int(
    "FT_QUPAP_SERIAL_BAUDRATE",
    9600,
)


# Project directories

SOURCE_DIRECTORY = PROJECT_ROOT / "src"
DASHBOARD_DIRECTORY = PROJECT_ROOT / "dashboard"
SCENARIO_DIRECTORY = PROJECT_ROOT / "scenarios"
SCRIPT_DIRECTORY = PROJECT_ROOT / "scripts"
TEST_DIRECTORY = PROJECT_ROOT / "tests"
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"
ASSET_DIRECTORY = PROJECT_ROOT / "assets"

MODEL_DIRECTORY = PROJECT_ROOT / "models"
DATABASE_DIRECTORY = PROJECT_ROOT / "database"

DATA_DIRECTORY = PROJECT_ROOT / "data"
GENERATED_DATA_DIRECTORY = resolve_project_path(
    "FT_QUPAP_GENERATED_DATA_DIRECTORY",
    "data/generated",
)

OUTPUT_DIRECTORY = resolve_project_path(
    "FT_QUPAP_OUTPUT_DIRECTORY",
    "outputs",
)

FIGURE_DIRECTORY = OUTPUT_DIRECTORY / "figures"
TABLE_DIRECTORY = OUTPUT_DIRECTORY / "tables"
LOG_DIRECTORY = OUTPUT_DIRECTORY / "logs"


# Model files

GP_MODEL_PATH = resolve_project_path(
    "FT_QUPAP_GP_MODEL_PATH",
    "models/ft_qupap_gp_detector.joblib",
)

THRESHOLD_PATH = resolve_project_path(
    "FT_QUPAP_THRESHOLD_PATH",
    "models/threshold.json",
)

MODEL_METADATA_PATH = resolve_project_path(
    "FT_QUPAP_MODEL_METADATA_PATH",
    "models/model_metadata.json",
)

# Local database files

SUBSCRIBER_DATABASE_PATH = resolve_project_path(
    "FT_QUPAP_SUBSCRIBER_DATABASE",
    "database/subscribers.json",
)

SERVER_SIGNING_KEY_PATH = resolve_project_path(
    "FT_QUPAP_SERVER_KEY_FILE",
    "database/server_signing_keys.json",
)

SESSION_LOG_PATH = resolve_project_path(
    "FT_QUPAP_SESSION_LOG_FILE",
    "database/demo_sessions.json",
)

USED_NONCE_DATABASE_PATH = DATABASE_DIRECTORY / "used_nonces.json"

# Asset paths

PROTOCOL_FLOWCHART_PATH = (
    ASSET_DIRECTORY
    / "images"
    / "protocol_flowchart.png"
)


# Structured configuration classes

@dataclass(frozen=True)
class ProtocolConfiguration:
    """Immutable protocol-level configuration."""

    protocol_name: str = PROJECT_NAME
    protocol_version: str = PROJECT_VERSION
    server_id: str = SERVER_ID

    ml_dsa_algorithm: str = ML_DSA_ALGORITHM
    ml_kem_algorithm: str = ML_KEM_ALGORITHM

    kmac_tag_bytes: int = KMAC_TAG_BYTES

    payload_logical_qubits: int = PAYLOAD_LOGICAL_QUBITS
    check_logical_qubits: int = CHECK_LOGICAL_QUBITS
    steane_block_size: int = STEANE_PHYSICAL_QUBITS_PER_LOGICAL

    minimum_observed_check_blocks: int = (
        MINIMUM_OBSERVED_CHECK_BLOCKS
    )

    freshness_window_seconds: int = FRESHNESS_WINDOW_SECONDS
    fixed_qber_threshold: float = FIXED_QBER_THRESHOLD

    maximum_acceptable_loss_rate: float = (
        MAXIMUM_ACCEPTABLE_LOSS_RATE
    )

    minimum_operational_gp_threshold: float = (
        MINIMUM_OPERATIONAL_GP_THRESHOLD
    )

    gp_gray_zone_retry_upper: float = (
        GP_GRAY_ZONE_RETRY_UPPER
    )

    maximum_retries: int = MAXIMUM_RETRIES
    maximum_authentication_attempts: int = (
        MAXIMUM_AUTHENTICATION_ATTEMPTS
    )

    retry_attack_probability_limit: float = (
        RETRY_ATTACK_PROBABILITY_LIMIT
    )

    retry_qber_limit: float = RETRY_QBER_LIMIT


@dataclass(frozen=True)
class ApplicationConfiguration:
    """Immutable dashboard and runtime configuration."""

    demo_mode: bool = DEMO_MODE
    random_seed: int = RANDOM_SEED

    streamlit_host: str = STREAMLIT_HOST
    streamlit_port: int = STREAMLIT_PORT

    hardware_enabled: bool = HARDWARE_ENABLED
    serial_port: str = SERIAL_PORT
    serial_baudrate: int = SERIAL_BAUDRATE


PROTOCOL_CONFIG = ProtocolConfiguration()
APPLICATION_CONFIG = ApplicationConfiguration()


# Directory initialization

def create_required_directories() -> None:
    """
    Create directories that hold generated runtime artifacts.

    This function does not generate:

    - ML-DSA keys
    - GP models
    - subscriber records
    - result tables

    Those artifacts are generated by their dedicated scripts.
    """
    required_directories = (
        MODEL_DIRECTORY,
        DATABASE_DIRECTORY,
        GENERATED_DATA_DIRECTORY,
        OUTPUT_DIRECTORY,
        FIGURE_DIRECTORY,
        TABLE_DIRECTORY,
        LOG_DIRECTORY,
    )

    for directory in required_directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

# Configuration validation

def validate_configuration() -> None:
    """Validate important FT-QuPAP protocol invariants."""

    if KMAC_TAG_BITS != 128:
        raise ValueError(
            "The current FT-QuPAP flow requires a 128-bit KMAC tag."
        )

    if PAYLOAD_LOGICAL_QUBITS != 128:
        raise ValueError(
            "Payload logical-qubit count must equal the 128-bit tag."
        )

    if CHECK_LOGICAL_QUBITS != 32:
        raise ValueError(
            "The current protocol requires 32 independent check blocks."
        )

    if TOTAL_LOGICAL_QUBITS != 160:
        raise ValueError(
            "Total logical blocks must be 128 payload plus 32 checks."
        )

    if TOTAL_PHYSICAL_QUBITS != 1120:
        raise ValueError(
            "Steane encoding must produce 160 × 7 = 1120 physical qubits."
        )

    if not (
        1
        <= MINIMUM_OBSERVED_CHECK_BLOCKS
        <= CHECK_LOGICAL_QUBITS
    ):
        raise ValueError(
            "Minimum observed check blocks must be between 1 and 32."
        )

    if not 0.0 <= FIXED_QBER_THRESHOLD <= 1.0:
        raise ValueError(
            "The fixed QBER threshold must be between 0 and 1."
        )

    if not 0.0 <= MAXIMUM_ACCEPTABLE_LOSS_RATE <= 1.0:
        raise ValueError(
            "The maximum acceptable loss rate must be between 0 and 1."
        )

    if MAXIMUM_RETRIES < 0:
        raise ValueError(
            "Maximum retries cannot be negative."
        )

    if MAXIMUM_AUTHENTICATION_ATTEMPTS != 1 + MAXIMUM_RETRIES:
        raise ValueError(
            "Total attempts must equal one initial attempt plus retries."
        )

    if GP_GRAY_ZONE_RETRY_UPPER < MINIMUM_OPERATIONAL_GP_THRESHOLD:
        raise ValueError(
            "The GP retry upper boundary cannot be lower than "
            "the minimum operational threshold."
        )

    if set(SUPPORTED_CONTEXTS) != set(CONTEXT_CHANNEL_PROFILES):
        raise ValueError(
            "Every supported context requires a channel profile."
        )

    if set(SUPPORTED_CONTEXTS) != set(CONTEXT_BASE_NOISE):
        raise ValueError(
            "Every supported context requires a base-noise value."
        )


create_required_directories()
validate_configuration()