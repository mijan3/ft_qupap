"""
Shared constants for the FT-QuPAP v5.1 project.

This module re-exports the central values defined in config.py so that
all internal source modules can import protocol constants from one place.

Example:

    from src.common.constants import (
        KMAC_TAG_BITS,
        TOTAL_PHYSICAL_QUBITS,
        FEATURE_COLUMNS,
    )
"""

from config import (
    # Project information
    PROJECT_NAME,
    PROJECT_FULL_NAME,
    PROJECT_VERSION,
    SIMULATOR_VERSION,
    SERVER_ID,
    DEFAULT_PSEUDONYM_ID,
    SUPPORTED_CONTEXTS,

    # Cryptographic algorithms
    ML_DSA_ALGORITHM,
    ML_KEM_ALGORITHM,
    KMAC_ALGORITHM,
    KMAC_TAG_BYTES,
    KMAC_TAG_BITS,
    SESSION_KEY_DERIVATION_LABEL,
    AUTHENTICATION_KEY_LABEL,
    CONTROL_KEY_LABEL,
    TRANSCRIPT_HASH_ALGORITHM,

    # Quantum configuration
    PAYLOAD_LOGICAL_QUBITS,
    CHECK_LOGICAL_QUBITS,
    TOTAL_LOGICAL_QUBITS,
    STEANE_CODE_NAME,
    STEANE_PHYSICAL_QUBITS_PER_LOGICAL,
    TOTAL_PHYSICAL_QUBITS,
    MINIMUM_OBSERVED_CHECK_BLOCKS,

    # Freshness and replay
    FRESHNESS_WINDOW_SECONDS,
    NONCE_SIZE_BYTES,
    REQUEST_TYPE,

    # QBER policy
    FIXED_QBER_THRESHOLD,
    MAXIMUM_ACCEPTABLE_LOSS_RATE,

    # Gaussian Process configuration
    FEATURE_COLUMNS,
    BAYES_COST_FALSE_ACCEPT,
    BAYES_COST_FALSE_REJECT,
    DEFAULT_GP_ATTACK_THRESHOLD,
    MINIMUM_OPERATIONAL_GP_THRESHOLD,
    GP_GRAY_ZONE_RETRY_UPPER,
    GP_MAXIMUM_UNCERTAINTY,

    # Retry policy
    MAXIMUM_RETRIES,
    MAXIMUM_AUTHENTICATION_ATTEMPTS,
    RETRY_ATTACK_PROBABILITY_LIMIT,
    RETRY_QBER_LIMIT,

    # Context information
    CONTEXT_CHANNEL_PROFILES,
    CONTEXT_BASE_NOISE,

    # Application settings
    RANDOM_SEED,
    DEMO_MODE,
    STREAMLIT_HOST,
    STREAMLIT_PORT,

    # Hardware settings
    HARDWARE_ENABLED,
    SERIAL_PORT,
    SERIAL_BAUDRATE,

    # Paths
    PROJECT_ROOT,
    SOURCE_DIRECTORY,
    DASHBOARD_DIRECTORY,
    SCENARIO_DIRECTORY,
    SCRIPT_DIRECTORY,
    TEST_DIRECTORY,
    NOTEBOOK_DIRECTORY,
    ASSET_DIRECTORY,
    MODEL_DIRECTORY,
    DATABASE_DIRECTORY,
    DATA_DIRECTORY,
    GENERATED_DATA_DIRECTORY,
    OUTPUT_DIRECTORY,
    FIGURE_DIRECTORY,
    TABLE_DIRECTORY,
    LOG_DIRECTORY,
    GP_MODEL_PATH,
    THRESHOLD_PATH,
    MODEL_METADATA_PATH,
    SUBSCRIBER_DATABASE_PATH,
    SERVER_SIGNING_KEY_PATH,
    SESSION_LOG_PATH,
    USED_NONCE_DATABASE_PATH,
    PROTOCOL_FLOWCHART_PATH,

    # Structured configurations
    ProtocolConfiguration,
    ApplicationConfiguration,
    PROTOCOL_CONFIG,
    APPLICATION_CONFIG,

    # Helpers
    create_required_directories,
    validate_configuration,
)


# ---------------------------------------------------------------------
# Protocol labels
# ---------------------------------------------------------------------

PROTOCOL_DOMAIN_LABEL = b"FT-QuPAP-v5.1"

SERVER_PACKAGE_LABEL = b"FT-QuPAP-Server-Package"

AUTHENTICATION_REQUEST_LABEL = b"FT-QuPAP-M1"

MLKEM_CIPHERTEXT_LABEL = b"FT-QuPAP-M3"

CONTROL_SCHEDULE_LABEL = b"FT-QuPAP-Control-Schedule"

AUTHENTICATION_TAG_LABEL = b"FT-QuPAP-Authentication-Tag"


# ---------------------------------------------------------------------
# Protocol message names
# ---------------------------------------------------------------------

MESSAGE_M1 = "M1_AUTHENTICATION_REQUEST"

MESSAGE_M2 = "M2_SIGNED_SERVER_PACKAGE"

MESSAGE_M3 = "M3_MLKEM_CIPHERTEXT"

MESSAGE_QUANTUM_FRAME = "QUANTUM_AUTHENTICATION_FRAME"


# ---------------------------------------------------------------------
# Authentication decisions
# ---------------------------------------------------------------------

DECISION_ACCEPT = "ACCEPT"

DECISION_RETRY = "RETRY"

DECISION_REJECT = "REJECT"


# ---------------------------------------------------------------------
# Protocol stage names
# ---------------------------------------------------------------------

STAGE_REGISTRATION = "registration"

STAGE_REQUEST_CREATION = "request_creation"

STAGE_SUBSCRIBER_VERIFICATION = "subscriber_verification"

STAGE_FRESHNESS_VERIFICATION = "freshness_verification"

STAGE_REPLAY_VERIFICATION = "replay_verification"

STAGE_MLDSA_SIGNING = "mldsa_signing"

STAGE_MLDSA_VERIFICATION = "mldsa_verification"

STAGE_MLKEM_KEY_GENERATION = "mlkem_key_generation"

STAGE_MLKEM_ENCAPSULATION = "mlkem_encapsulation"

STAGE_MLKEM_DECAPSULATION = "mlkem_decapsulation"

STAGE_TRANSCRIPT_HASHING = "transcript_hashing"

STAGE_SESSION_KEY_DERIVATION = "session_key_derivation"

STAGE_KMAC_GENERATION = "kmac_generation"

STAGE_PAYLOAD_PREPARATION = "payload_preparation"

STAGE_CHECK_PREPARATION = "check_preparation"

STAGE_BLOCK_INTERLEAVING = "block_interleaving"

STAGE_STEANE_ENCODING = "steane_encoding"

STAGE_QUANTUM_TRANSMISSION = "quantum_transmission"

STAGE_CHECK_MEASUREMENT = "check_measurement"

STAGE_QBER_CALCULATION = "qber_calculation"

STAGE_SYNDROME_EXTRACTION = "syndrome_extraction"

STAGE_ERROR_CORRECTION = "error_correction"

STAGE_PAYLOAD_DECODING = "payload_decoding"

STAGE_TAG_VERIFICATION = "tag_verification"

STAGE_DETERMINISTIC_VERIFICATION = "deterministic_verification"

STAGE_GP_FEATURE_EXTRACTION = "gp_feature_extraction"

STAGE_GP_ATTACK_DETECTION = "gp_attack_detection"

STAGE_DECISION = "decision"

STAGE_RETRY = "retry"


# ---------------------------------------------------------------------
# Channel scenario names
# ---------------------------------------------------------------------

SCENARIO_NORMAL = "normal"

SCENARIO_BENIGN_NOISY = "benign_noisy"

SCENARIO_LOSSY = "lossy"

SCENARIO_PARTIAL_EVE = "partial_eve"

SCENARIO_FULL_EVE = "full_eve"

SCENARIO_REPLAY_ATTACK = "replay_attack"

SCENARIO_FORGED_SIGNATURE = "forged_signature"

SCENARIO_TAMPERED_CIPHERTEXT = "tampered_ciphertext"

SCENARIO_FORGED_TAG = "forged_tag"


# ---------------------------------------------------------------------
# Quantum block types
# ---------------------------------------------------------------------

BLOCK_TYPE_PAYLOAD = "payload"

BLOCK_TYPE_CHECK = "check"


# ---------------------------------------------------------------------
# Logical states and bases
# ---------------------------------------------------------------------

LOGICAL_ZERO = 0

LOGICAL_ONE = 1

BASIS_Z = "Z"

BASIS_X = "X"

SUPPORTED_BASES = (
    BASIS_Z,
    BASIS_X,
)


# ---------------------------------------------------------------------
# Runtime status values
# ---------------------------------------------------------------------

STATUS_WAITING = "WAITING"

STATUS_PROCESSING = "PROCESSING"

STATUS_PASSED = "PASSED"

STATUS_CORRECTED = "CORRECTED"

STATUS_RETRY = "RETRY"

STATUS_FAILED = "FAILED"

STATUS_SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------
# Required protocol invariants
# ---------------------------------------------------------------------

EXPECTED_KMAC_TAG_BITS = 128

EXPECTED_PAYLOAD_LOGICAL_QUBITS = 128

EXPECTED_CHECK_LOGICAL_QUBITS = 32

EXPECTED_TOTAL_LOGICAL_QUBITS = 160

EXPECTED_TOTAL_PHYSICAL_QUBITS = 1120


def validate_constant_invariants() -> None:
    """
    Validate constants imported from config.py.

    This provides an additional safety check before protocol execution.
    """

    if KMAC_TAG_BITS != EXPECTED_KMAC_TAG_BITS:
        raise ValueError(
            "FT-QuPAP requires a 128-bit KMAC authentication tag."
        )

    if PAYLOAD_LOGICAL_QUBITS != EXPECTED_PAYLOAD_LOGICAL_QUBITS:
        raise ValueError(
            "FT-QuPAP requires 128 payload logical qubits."
        )

    if CHECK_LOGICAL_QUBITS != EXPECTED_CHECK_LOGICAL_QUBITS:
        raise ValueError(
            "FT-QuPAP requires 32 independent check logical qubits."
        )

    if TOTAL_LOGICAL_QUBITS != EXPECTED_TOTAL_LOGICAL_QUBITS:
        raise ValueError(
            "FT-QuPAP requires 160 total logical qubits."
        )

    if TOTAL_PHYSICAL_QUBITS != EXPECTED_TOTAL_PHYSICAL_QUBITS:
        raise ValueError(
            "Steane encoding must produce 1120 physical qubits."
        )

    if len(FEATURE_COLUMNS) != 9:
        raise ValueError(
            "The Gaussian Process feature schema must contain 9 features."
        )


validate_constant_invariants()


__all__ = [
    # Configuration objects
    "PROTOCOL_CONFIG",
    "APPLICATION_CONFIG",
    "ProtocolConfiguration",
    "ApplicationConfiguration",

    # Project metadata
    "PROJECT_NAME",
    "PROJECT_FULL_NAME",
    "PROJECT_VERSION",
    "SIMULATOR_VERSION",
    "SERVER_ID",
    "DEFAULT_PSEUDONYM_ID",
    "SUPPORTED_CONTEXTS",

    # Cryptographic configuration
    "ML_DSA_ALGORITHM",
    "ML_KEM_ALGORITHM",
    "KMAC_ALGORITHM",
    "KMAC_TAG_BYTES",
    "KMAC_TAG_BITS",
    "SESSION_KEY_DERIVATION_LABEL",
    "AUTHENTICATION_KEY_LABEL",
    "CONTROL_KEY_LABEL",
    "TRANSCRIPT_HASH_ALGORITHM",

    # Quantum configuration
    "PAYLOAD_LOGICAL_QUBITS",
    "CHECK_LOGICAL_QUBITS",
    "TOTAL_LOGICAL_QUBITS",
    "STEANE_CODE_NAME",
    "STEANE_PHYSICAL_QUBITS_PER_LOGICAL",
    "TOTAL_PHYSICAL_QUBITS",
    "MINIMUM_OBSERVED_CHECK_BLOCKS",

    # Security policy
    "FRESHNESS_WINDOW_SECONDS",
    "NONCE_SIZE_BYTES",
    "REQUEST_TYPE",
    "FIXED_QBER_THRESHOLD",
    "MAXIMUM_ACCEPTABLE_LOSS_RATE",
    "FEATURE_COLUMNS",
    "BAYES_COST_FALSE_ACCEPT",
    "BAYES_COST_FALSE_REJECT",
    "DEFAULT_GP_ATTACK_THRESHOLD",
    "MINIMUM_OPERATIONAL_GP_THRESHOLD",
    "GP_GRAY_ZONE_RETRY_UPPER",
    "GP_MAXIMUM_UNCERTAINTY",

    # Retry configuration
    "MAXIMUM_RETRIES",
    "MAXIMUM_AUTHENTICATION_ATTEMPTS",
    "RETRY_ATTACK_PROBABILITY_LIMIT",
    "RETRY_QBER_LIMIT",

    # Context configuration
    "CONTEXT_CHANNEL_PROFILES",
    "CONTEXT_BASE_NOISE",

    # Runtime configuration
    "RANDOM_SEED",
    "DEMO_MODE",
    "STREAMLIT_HOST",
    "STREAMLIT_PORT",
    "HARDWARE_ENABLED",
    "SERIAL_PORT",
    "SERIAL_BAUDRATE",

    # Paths
    "PROJECT_ROOT",
    "MODEL_DIRECTORY",
    "DATABASE_DIRECTORY",
    "GENERATED_DATA_DIRECTORY",
    "OUTPUT_DIRECTORY",
    "FIGURE_DIRECTORY",
    "TABLE_DIRECTORY",
    "LOG_DIRECTORY",
    "GP_MODEL_PATH",
    "THRESHOLD_PATH",
    "MODEL_METADATA_PATH",
    "SUBSCRIBER_DATABASE_PATH",
    "SERVER_SIGNING_KEY_PATH",
    "SESSION_LOG_PATH",
    "USED_NONCE_DATABASE_PATH",
    "PROTOCOL_FLOWCHART_PATH",

    # Labels
    "PROTOCOL_DOMAIN_LABEL",
    "SERVER_PACKAGE_LABEL",
    "AUTHENTICATION_REQUEST_LABEL",
    "MLKEM_CIPHERTEXT_LABEL",
    "CONTROL_SCHEDULE_LABEL",
    "AUTHENTICATION_TAG_LABEL",

    # Messages
    "MESSAGE_M1",
    "MESSAGE_M2",
    "MESSAGE_M3",
    "MESSAGE_QUANTUM_FRAME",

    # Decisions
    "DECISION_ACCEPT",
    "DECISION_RETRY",
    "DECISION_REJECT",

    # Block and basis values
    "BLOCK_TYPE_PAYLOAD",
    "BLOCK_TYPE_CHECK",
    "LOGICAL_ZERO",
    "LOGICAL_ONE",
    "BASIS_Z",
    "BASIS_X",
    "SUPPORTED_BASES",

    # Status values
    "STATUS_WAITING",
    "STATUS_PROCESSING",
    "STATUS_PASSED",
    "STATUS_CORRECTED",
    "STATUS_RETRY",
    "STATUS_FAILED",
    "STATUS_SKIPPED",

    # Helpers
    "create_required_directories",
    "validate_configuration",
    "validate_constant_invariants",
]