"""
Enumeration classes used throughout the FT-QuPAP v5.1 project.

Enums provide controlled values for:

- Authentication decisions
- Mobile-network contexts
- Protocol stage statuses
- Quantum block types
- Quantum measurement bases
- Attack scenarios
- Authentication failure reasons
"""

from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """
    Base enum whose members behave like strings.

    Example:

        decision = AuthenticationDecision.ACCEPT
        print(decision.value)   # ACCEPT
        print(str(decision))    # ACCEPT
    """

    def __str__(self) -> str:
        return self.value


class AuthenticationDecision(StringEnum):
    """Possible final FT-QuPAP authentication decisions."""

    ACCEPT = "ACCEPT"
    RETRY = "RETRY"
    REJECT = "REJECT"


class ChannelContext(StringEnum):
    """Supported mobile-network operating contexts."""

    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"


class ProtocolStageStatus(StringEnum):
    """Execution status of an individual protocol stage."""

    WAITING = "WAITING"
    PROCESSING = "PROCESSING"
    PASSED = "PASSED"
    CORRECTED = "CORRECTED"
    RETRY = "RETRY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ProtocolMessageType(StringEnum):
    """Classical and quantum message types used by FT-QuPAP."""

    AUTHENTICATION_REQUEST = "M1_AUTHENTICATION_REQUEST"
    SIGNED_SERVER_PACKAGE = "M2_SIGNED_SERVER_PACKAGE"
    MLKEM_CIPHERTEXT = "M3_MLKEM_CIPHERTEXT"
    QUANTUM_FRAME = "QUANTUM_AUTHENTICATION_FRAME"


class QuantumBlockType(StringEnum):
    """Types of logical blocks transmitted in the quantum frame."""

    PAYLOAD = "payload"
    CHECK = "check"


class QuantumBasis(StringEnum):
    """Measurement and preparation bases used by the simulator."""

    Z = "Z"
    X = "X"


class LogicalBit(int, Enum):
    """Classical logical values mapped to logical quantum states."""

    ZERO = 0
    ONE = 1

    def __str__(self) -> str:
        return str(self.value)


class ChannelScenario(StringEnum):
    """Supported capstone demonstration scenarios."""

    NORMAL = "normal"
    BENIGN_NOISY = "benign_noisy"
    LOSSY = "lossy"
    PARTIAL_EVE = "partial_eve"
    FULL_EVE = "full_eve"
    REPLAY_ATTACK = "replay_attack"
    FORGED_SIGNATURE = "forged_signature"
    TAMPERED_CIPHERTEXT = "tampered_ciphertext"
    FORGED_TAG = "forged_tag"
    UNCORRECTABLE_ERROR = "uncorrectable_error"


class EveAttackMode(StringEnum):
    """Supported simulated eavesdropping behaviours."""

    NONE = "none"
    INTERCEPT_RESEND = "intercept_resend"
    PARTIAL_INTERCEPT_RESEND = "partial_intercept_resend"
    FULL_INTERCEPT_RESEND = "full_intercept_resend"


class ErrorType(StringEnum):
    """Physical-qubit errors represented by the simulator."""

    NONE = "none"
    BIT_FLIP = "bit_flip"
    PHASE_FLIP = "phase_flip"
    BIT_PHASE_FLIP = "bit_phase_flip"
    DEPOLARIZING = "depolarizing"
    LOSS = "loss"


class VerificationResult(StringEnum):
    """Result of a deterministic verification operation."""

    VALID = "valid"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class FailureReason(StringEnum):
    """Standardized reasons for authentication failure."""

    NONE = "none"

    UNKNOWN_SUBSCRIBER = "unknown_subscriber"
    INACTIVE_SUBSCRIBER = "inactive_subscriber"
    INVALID_CONTEXT = "invalid_context"

    STALE_TIMESTAMP = "stale_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    NONCE_REPLAY = "nonce_replay"
    INVALID_NONCE = "invalid_nonce"

    SERVER_SIGNATURE_INVALID = "server_signature_invalid"
    SERVER_PACKAGE_INVALID = "server_package_invalid"
    SERVER_CREDENTIAL_EXPIRED = "server_credential_expired"

    MLKEM_CIPHERTEXT_INVALID = "mlkem_ciphertext_invalid"
    MLKEM_DECAPSULATION_FAILED = "mlkem_decapsulation_failed"
    SHARED_SECRET_MISMATCH = "shared_secret_mismatch"

    TRANSCRIPT_MISMATCH = "transcript_mismatch"
    SESSION_KEY_DERIVATION_FAILED = "session_key_derivation_failed"

    CONTROL_SCHEDULE_INVALID = "control_schedule_invalid"
    CONTROL_SCHEDULE_DECRYPTION_FAILED = (
        "control_schedule_decryption_failed"
    )

    INSUFFICIENT_CHECK_BLOCKS = "insufficient_check_blocks"
    EXCESSIVE_QBER = "excessive_qber"
    EXCESSIVE_LOSS = "excessive_loss"

    SYNDROME_EXTRACTION_FAILED = "syndrome_extraction_failed"
    UNCORRECTABLE_QUANTUM_ERROR = "uncorrectable_quantum_error"
    PAYLOAD_RECOVERY_FAILED = "payload_recovery_failed"

    KMAC_TAG_MISMATCH = "kmac_tag_mismatch"

    GP_MODEL_UNAVAILABLE = "gp_model_unavailable"
    GP_ATTACK_DETECTED = "gp_attack_detected"
    GP_UNCERTAINTY_TOO_HIGH = "gp_uncertainty_too_high"

    MAXIMUM_RETRIES_EXCEEDED = "maximum_retries_exceeded"
    INTERNAL_PROTOCOL_ERROR = "internal_protocol_error"


class RetryReason(StringEnum):
    """Reasons for starting a fresh authentication retry."""

    NONE = "none"
    RECOVERABLE_NOISE = "recoverable_noise"
    PAYLOAD_RECOVERY_FAILURE = "payload_recovery_failure"
    LOW_RISK_GP_GRAY_ZONE = "low_risk_gp_gray_zone"
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"


class AuthenticationPhase(StringEnum):
    """High-level phases shown in the capstone dashboard."""

    REGISTRATION = "registration"
    CLASSICAL_BOOTSTRAP = "classical_bootstrap"
    QUANTUM_PREPARATION = "quantum_preparation"
    QUANTUM_TRANSMISSION = "quantum_transmission"
    DETERMINISTIC_VERIFICATION = "deterministic_verification"
    ADAPTIVE_ATTACK_DETECTION = "adaptive_attack_detection"
    FINAL_DECISION = "final_decision"


def enum_values(enum_class: type[Enum]) -> tuple[object, ...]:
    """
    Return all raw values contained in an enum.

    Example:

        values = enum_values(ChannelContext)
        # ("urban", "suburban", "rural")
    """

    return tuple(member.value for member in enum_class)


def parse_channel_context(value: str) -> ChannelContext:
    """
    Convert user input into a validated ChannelContext value.

    The comparison is case-insensitive and ignores surrounding spaces.
    """

    normalized = value.strip().lower()

    try:
        return ChannelContext(normalized)
    except ValueError as exc:
        supported = ", ".join(
            str(item.value)
            for item in ChannelContext
        )

        raise ValueError(
            f"Unsupported channel context '{value}'. "
            f"Supported contexts: {supported}."
        ) from exc


def parse_authentication_decision(
    value: str,
) -> AuthenticationDecision:
    """
    Convert a text value into AuthenticationDecision.
    """

    normalized = value.strip().upper()

    try:
        return AuthenticationDecision(normalized)
    except ValueError as exc:
        supported = ", ".join(
            str(item.value)
            for item in AuthenticationDecision
        )

        raise ValueError(
            f"Unsupported authentication decision '{value}'. "
            f"Supported decisions: {supported}."
        ) from exc


__all__ = [
    "StringEnum",
    "AuthenticationDecision",
    "ChannelContext",
    "ProtocolStageStatus",
    "ProtocolMessageType",
    "QuantumBlockType",
    "QuantumBasis",
    "LogicalBit",
    "ChannelScenario",
    "EveAttackMode",
    "ErrorType",
    "VerificationResult",
    "FailureReason",
    "RetryReason",
    "AuthenticationPhase",
    "enum_values",
    "parse_channel_context",
    "parse_authentication_decision",
]