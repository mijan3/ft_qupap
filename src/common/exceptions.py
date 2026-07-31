"""
Custom exception classes for the FT-QuPAP v5.1 project.

These exceptions make protocol failures easier to identify, log,
display in the dashboard, and test.
"""

from __future__ import annotations

from typing import Any, Optional


class FTQuPAPError(Exception):
    """
    Base exception for all FT-QuPAP project errors.

    Every custom project exception should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "FT_QUPAP_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the exception into a JSON-compatible dictionary.
        """

        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class ConfigurationError(FTQuPAPError):
    """
    Raised when project or protocol configuration is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="CONFIGURATION_ERROR",
            details=details,
        )


class CryptographicError(FTQuPAPError):
    """
    Base exception for cryptographic failures.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "CRYPTOGRAPHIC_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
        )


class MLDSAError(CryptographicError):
    """
    Raised when ML-DSA key generation, signing, or verification fails.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="MLDSA_ERROR",
            details=details,
        )


class MLDSAVerificationError(MLDSAError):
    """
    Raised when a server ML-DSA signature is invalid.
    """

    def __init__(
        self,
        message: str = "ML-DSA server signature verification failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            details=details,
        )

        self.code = "MLDSA_VERIFICATION_ERROR"


class MLKEMError(CryptographicError):
    """
    Raised when ML-KEM operations fail.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="MLKEM_ERROR",
            details=details,
        )


class MLKEMCiphertextError(MLKEMError):
    """
    Raised when an ML-KEM ciphertext is malformed or invalid.
    """

    def __init__(
        self,
        message: str = "The ML-KEM ciphertext is invalid.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            details=details,
        )

        self.code = "MLKEM_CIPHERTEXT_ERROR"


class MLKEMDecapsulationError(MLKEMError):
    """
    Raised when ML-KEM decapsulation cannot recover session material.
    """

    def __init__(
        self,
        message: str = "ML-KEM decapsulation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            details=details,
        )

        self.code = "MLKEM_DECAPSULATION_ERROR"


class KMACError(CryptographicError):
    """
    Raised when KMAC generation or verification fails.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="KMAC_ERROR",
            details=details,
        )


class KMACTagMismatchError(KMACError):
    """
    Raised when the reconstructed tag does not match the expected tag.
    """

    def __init__(
        self,
        message: str = "The received KMAC authentication tag is invalid.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            details=details,
        )

        self.code = "KMAC_TAG_MISMATCH"


class KeyDerivationError(CryptographicError):
    """
    Raised when transcript-bound session key derivation fails.
    """

    def __init__(
        self,
        message: str = "Session key derivation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="KEY_DERIVATION_ERROR",
            details=details,
        )


class ProtocolValidationError(FTQuPAPError):
    """
    Raised when a protocol message or invariant is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "PROTOCOL_VALIDATION_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
        )


class RegistrationError(ProtocolValidationError):
    """
    Raised when subscriber registration is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="REGISTRATION_ERROR",
            details=details,
        )


class UnknownSubscriberError(ProtocolValidationError):
    """
    Raised when a pseudonymous subscriber identity is unknown.
    """

    def __init__(
        self,
        pseudonym_id: str,
    ) -> None:
        super().__init__(
            f"Unknown subscriber pseudonym: {pseudonym_id}",
            code="UNKNOWN_SUBSCRIBER",
            details={
                "pseudonym_id": pseudonym_id,
            },
        )


class InactiveSubscriberError(ProtocolValidationError):
    """
    Raised when a registered subscriber is inactive.
    """

    def __init__(
        self,
        pseudonym_id: str,
    ) -> None:
        super().__init__(
            f"Subscriber {pseudonym_id} is inactive.",
            code="INACTIVE_SUBSCRIBER",
            details={
                "pseudonym_id": pseudonym_id,
            },
        )


class FreshnessError(ProtocolValidationError):
    """
    Raised when a protocol timestamp is stale or invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        timestamp: Optional[int] = None,
        current_time: Optional[int] = None,
    ) -> None:
        details: dict[str, Any] = {}

        if timestamp is not None:
            details["timestamp"] = timestamp

        if current_time is not None:
            details["current_time"] = current_time

        super().__init__(
            message,
            code="FRESHNESS_ERROR",
            details=details,
        )


class ReplayAttackError(ProtocolValidationError):
    """
    Raised when a nonce has already been used.
    """

    def __init__(
        self,
        nonce: str,
    ) -> None:
        super().__init__(
            "Replay attack detected: nonce has already been used.",
            code="REPLAY_ATTACK_DETECTED",
            details={
                "nonce": nonce,
            },
        )


class TranscriptMismatchError(ProtocolValidationError):
    """
    Raised when Mobile Station and Authentication Server transcripts differ.
    """

    def __init__(
        self,
        message: str = "Protocol transcript verification failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="TRANSCRIPT_MISMATCH",
            details=details,
        )


class ControlScheduleError(ProtocolValidationError):
    """
    Raised when the encrypted block-control schedule is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="CONTROL_SCHEDULE_ERROR",
            details=details,
        )


class QuantumSimulationError(FTQuPAPError):
    """
    Base exception for quantum preparation, transmission, and decoding.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "QUANTUM_SIMULATION_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
        )


class SteaneEncodingError(QuantumSimulationError):
    """
    Raised when Steane CSS encoding fails.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="STEANE_ENCODING_ERROR",
            details=details,
        )


class SyndromeExtractionError(QuantumSimulationError):
    """
    Raised when syndrome extraction fails.
    """

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code="SYNDROME_EXTRACTION_ERROR",
            details=details,
        )


class UncorrectableQuantumError(QuantumSimulationError):
    """
    Raised when the simulated CSS layer cannot correct a block.
    """

    def __init__(
        self,
        block_index: int,
        *,
        syndrome: Optional[str] = None,
    ) -> None:
        details: dict[str, Any] = {
            "block_index": block_index,
        }

        if syndrome is not None:
            details["syndrome"] = syndrome

        super().__init__(
            f"Quantum block {block_index} contains an uncorrectable error.",
            code="UNCORRECTABLE_QUANTUM_ERROR",
            details=details,
        )


class InsufficientCheckBlocksError(QuantumSimulationError):
    """
    Raised when too few check blocks are available for reliable QBER.
    """

    def __init__(
        self,
        observed: int,
        required: int,
    ) -> None:
        super().__init__(
            (
                "Insufficient observed check blocks: "
                f"received {observed}, required {required}."
            ),
            code="INSUFFICIENT_CHECK_BLOCKS",
            details={
                "observed_check_blocks": observed,
                "required_check_blocks": required,
            },
        )


class ExcessiveQBERError(QuantumSimulationError):
    """
    Raised when raw QBER exceeds the configured security threshold.
    """

    def __init__(
        self,
        qber: float,
        threshold: float,
    ) -> None:
        super().__init__(
            (
                f"Raw QBER {qber:.6f} exceeds "
                f"the threshold {threshold:.6f}."
            ),
            code="EXCESSIVE_QBER",
            details={
                "qber": qber,
                "threshold": threshold,
            },
        )


class ExcessiveLossError(QuantumSimulationError):
    """
    Raised when quantum-channel loss exceeds the accepted policy.
    """

    def __init__(
        self,
        loss_rate: float,
        maximum_loss_rate: float,
    ) -> None:
        super().__init__(
            (
                f"Loss rate {loss_rate:.6f} exceeds "
                f"the limit {maximum_loss_rate:.6f}."
            ),
            code="EXCESSIVE_LOSS",
            details={
                "loss_rate": loss_rate,
                "maximum_loss_rate": maximum_loss_rate,
            },
        )


class MachineLearningError(FTQuPAPError):
    """
    Base exception for GP model operations.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "MACHINE_LEARNING_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
        )


class ModelNotGeneratedError(MachineLearningError):
    """
    Raised when the trained GP model file does not exist.
    """

    def __init__(
        self,
        model_path: str,
    ) -> None:
        super().__init__(
            (
                "The Gaussian Process model has not been generated. "
                "Run the model-training script first."
            ),
            code="MODEL_NOT_GENERATED",
            details={
                "model_path": model_path,
                "generation_command": (
                    "python scripts/train_gp_model.py --mode quick"
                ),
            },
        )


class ModelLoadingError(MachineLearningError):
    """
    Raised when a generated model file cannot be loaded.
    """

    def __init__(
        self,
        model_path: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Unable to load GP model from {model_path}: {reason}",
            code="MODEL_LOADING_ERROR",
            details={
                "model_path": model_path,
                "reason": reason,
            },
        )


class FeatureSchemaError(MachineLearningError):
    """
    Raised when GP features do not match the trained model schema.
    """

    def __init__(
        self,
        missing_features: list[str],
        unexpected_features: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            "The GP feature schema is invalid.",
            code="FEATURE_SCHEMA_ERROR",
            details={
                "missing_features": missing_features,
                "unexpected_features": unexpected_features or [],
            },
        )


class AttackDetectedError(MachineLearningError):
    """
    Raised when the GP detector classifies the session as an attack.
    """

    def __init__(
        self,
        probability: float,
        threshold: float,
    ) -> None:
        super().__init__(
            (
                f"Attack detected with probability {probability:.6f}; "
                f"threshold is {threshold:.6f}."
            ),
            code="GP_ATTACK_DETECTED",
            details={
                "attack_probability": probability,
                "threshold": threshold,
            },
        )


class RetryPolicyError(FTQuPAPError):
    """
    Raised when retry-policy execution is invalid.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "RETRY_POLICY_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
        )


class MaximumRetriesExceededError(RetryPolicyError):
    """
    Raised after all allowed authentication attempts have failed.
    """

    def __init__(
        self,
        attempts: int,
    ) -> None:
        super().__init__(
            (
                "Maximum authentication attempts exceeded. "
                f"Total attempts: {attempts}."
            ),
            code="MAXIMUM_RETRIES_EXCEEDED",
            details={
                "attempts": attempts,
            },
        )


class StorageError(FTQuPAPError):
    """
    Raised when reading or writing local project data fails.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Optional[str] = None,
    ) -> None:
        details = {}

        if path is not None:
            details["path"] = path

        super().__init__(
            message,
            code="STORAGE_ERROR",
            details=details,
        )


class HardwareConnectionError(FTQuPAPError):
    """
    Raised when the ESP32 or Arduino serial connection fails.
    """

    def __init__(
        self,
        port: str,
        reason: str,
    ) -> None:
        super().__init__(
            f"Unable to connect to the hardware controller on {port}: {reason}",
            code="HARDWARE_CONNECTION_ERROR",
            details={
                "port": port,
                "reason": reason,
            },
        )


def exception_to_dict(error: Exception) -> dict[str, Any]:
    """
    Convert any exception into a dictionary suitable for logging.

    FTQuPAPError objects preserve their custom code and details.
    Standard Python exceptions receive a generic representation.
    """

    if isinstance(error, FTQuPAPError):
        return error.to_dict()

    return {
        "error_type": error.__class__.__name__,
        "code": "UNEXPECTED_ERROR",
        "message": str(error),
        "details": {},
    }


__all__ = [
    "FTQuPAPError",
    "ConfigurationError",
    "CryptographicError",
    "MLDSAError",
    "MLDSAVerificationError",
    "MLKEMError",
    "MLKEMCiphertextError",
    "MLKEMDecapsulationError",
    "KMACError",
    "KMACTagMismatchError",
    "KeyDerivationError",
    "ProtocolValidationError",
    "RegistrationError",
    "UnknownSubscriberError",
    "InactiveSubscriberError",
    "FreshnessError",
    "ReplayAttackError",
    "TranscriptMismatchError",
    "ControlScheduleError",
    "QuantumSimulationError",
    "SteaneEncodingError",
    "SyndromeExtractionError",
    "UncorrectableQuantumError",
    "InsufficientCheckBlocksError",
    "ExcessiveQBERError",
    "ExcessiveLossError",
    "MachineLearningError",
    "ModelNotGeneratedError",
    "ModelLoadingError",
    "FeatureSchemaError",
    "AttackDetectedError",
    "RetryPolicyError",
    "MaximumRetriesExceededError",
    "StorageError",
    "HardwareConnectionError",
    "exception_to_dict",
]