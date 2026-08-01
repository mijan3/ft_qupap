"""
FT-QuPAP Core Source Package
============================

Core implementation package for:

    FT-QuPAP v5.1
    Fault-Tolerant Quantum Authentication Protocol with
    Post-Quantum Cryptographic Bootstrapping and
    Adaptive Eavesdropping Detection.

Main source packages:

    common
        Shared constants, enumerations, validation, serialization,
        timing, randomness, and project-specific exceptions.

    cryptography
        ML-DSA-65, ML-KEM-768, KMAC256, session-key derivation,
        transcript hashing, pseudonymous identity generation,
        nonce handling, and secure comparison.

    mobile_station
        Registration, server-package verification, ML-KEM
        encapsulation, session-key derivation, KMAC tag generation,
        Steane encoding, and quantum transmission.

    authentication_server
        Subscriber verification, freshness and replay checking,
        ML-KEM decapsulation, quantum evidence analysis,
        deterministic verification, GP attack detection,
        and retry-policy processing.

    quantum
        Classical-to-qubit conversion, Steane [[7,1,3]] CSS coding,
        channel noise, qubit loss, Eve attacks, measurement,
        syndrome processing, error correction, and QBER calculation.

    protocol
        End-to-end protocol orchestration, session state,
        verification, decision processing, retry management,
        transcript construction, and protocol logging.

    machine_learning
        Feature processing, Gaussian Process model loading,
        calibrated attack-probability prediction, threshold
        management, training, and evaluation.

    storage
        Subscriber, nonce, registration, session, authentication
        result, and CSV-based experimental storage.

    hardware
        Optional ESP32/Arduino decision indicators:

            GREEN  -> accepted
            YELLOW -> retry
            RED    -> rejected

Security note:
    This package initializer intentionally avoids importing every
    subpackage automatically. Keeping initialization lightweight helps
    prevent circular imports and avoids loading optional dependencies
    such as Qiskit, scikit-learn, PySerial, and PQC libraries before
    they are required.
"""

from __future__ import annotations

from typing import Final


PACKAGE_NAME: Final[str] = "FT-QuPAP-Capstone"
PROTOCOL_NAME: Final[str] = "FT-QuPAP"
PROTOCOL_VERSION: Final[str] = "5.1"
PACKAGE_VERSION: Final[str] = "5.1.0"

FULL_PROTOCOL_NAME: Final[str] = (
    "Fault-Tolerant Quantum Authentication Protocol "
    "with Post-Quantum Cryptographic Bootstrapping and "
    "Adaptive Eavesdropping Detection"
)

ML_DSA_PARAMETER_SET: Final[str] = "ML-DSA-65"
ML_KEM_PARAMETER_SET: Final[str] = "ML-KEM-768"
AUTHENTICATION_FUNCTION: Final[str] = "KMAC256"

STEANE_CODE: Final[str] = "[[7,1,3]]"
LOGICAL_PAYLOAD_BLOCKS: Final[int] = 128
INDEPENDENT_CHECK_BLOCKS: Final[int] = 32
TOTAL_LOGICAL_BLOCKS: Final[int] = 160
TOTAL_PHYSICAL_QUBITS: Final[int] = 1120

KMAC_TAG_BYTES: Final[int] = 16
KMAC_TAG_BITS: Final[int] = KMAC_TAG_BYTES * 8

FRESHNESS_WINDOW_SECONDS: Final[int] = 60
MAX_AUTHENTICATION_ATTEMPTS: Final[int] = 3
MAX_RETRIES: Final[int] = 2

FIXED_QBER_THRESHOLD: Final[float] = 0.11
MINIMUM_GP_THRESHOLD: Final[float] = 0.15
GP_RETRY_UPPER_BOUND: Final[float] = 0.20

MAXIMUM_LOSS_RATE: Final[float] = 0.15
MINIMUM_OBSERVED_CHECK_BLOCKS: Final[int] = 24


def get_package_information() -> dict[str, object]:
    """
    Return public package and protocol metadata.

    The returned dictionary contains only non-secret configuration
    information and may safely be displayed in the dashboard.
    """

    return {
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_version": PROTOCOL_VERSION,
        "full_protocol_name": FULL_PROTOCOL_NAME,
        "ml_dsa_parameter_set": ML_DSA_PARAMETER_SET,
        "ml_kem_parameter_set": ML_KEM_PARAMETER_SET,
        "authentication_function": AUTHENTICATION_FUNCTION,
        "kmac_tag_bits": KMAC_TAG_BITS,
        "steane_code": STEANE_CODE,
        "logical_payload_blocks": LOGICAL_PAYLOAD_BLOCKS,
        "independent_check_blocks": INDEPENDENT_CHECK_BLOCKS,
        "total_logical_blocks": TOTAL_LOGICAL_BLOCKS,
        "total_physical_qubits": TOTAL_PHYSICAL_QUBITS,
        "freshness_window_seconds": FRESHNESS_WINDOW_SECONDS,
        "maximum_authentication_attempts": (
            MAX_AUTHENTICATION_ATTEMPTS
        ),
        "maximum_retries": MAX_RETRIES,
        "fixed_qber_threshold": FIXED_QBER_THRESHOLD,
        "minimum_gp_threshold": MINIMUM_GP_THRESHOLD,
        "gp_retry_upper_bound": GP_RETRY_UPPER_BOUND,
        "maximum_loss_rate": MAXIMUM_LOSS_RATE,
        "minimum_observed_check_blocks": (
            MINIMUM_OBSERVED_CHECK_BLOCKS
        ),
    }


def get_version() -> str:
    """Return the FT-QuPAP package version."""

    return PACKAGE_VERSION


__version__ = PACKAGE_VERSION


__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "FULL_PROTOCOL_NAME",
    "ML_DSA_PARAMETER_SET",
    "ML_KEM_PARAMETER_SET",
    "AUTHENTICATION_FUNCTION",
    "STEANE_CODE",
    "LOGICAL_PAYLOAD_BLOCKS",
    "INDEPENDENT_CHECK_BLOCKS",
    "TOTAL_LOGICAL_BLOCKS",
    "TOTAL_PHYSICAL_QUBITS",
    "KMAC_TAG_BYTES",
    "KMAC_TAG_BITS",
    "FRESHNESS_WINDOW_SECONDS",
    "MAX_AUTHENTICATION_ATTEMPTS",
    "MAX_RETRIES",
    "FIXED_QBER_THRESHOLD",
    "MINIMUM_GP_THRESHOLD",
    "GP_RETRY_UPPER_BOUND",
    "MAXIMUM_LOSS_RATE",
    "MINIMUM_OBSERVED_CHECK_BLOCKS",
    "get_package_information",
    "get_version",
    "__version__",
]