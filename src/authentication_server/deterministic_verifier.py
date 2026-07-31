"""
Deterministic authentication verification for FT-QuPAP v5.1.

Before the Gaussian Process attack detector is consulted, the
Authentication Server performs mandatory deterministic checks.

The deterministic verification layer checks:

1. Enough check blocks were observed.
2. Raw QBER is within the fixed security threshold.
3. Quantum-channel loss is within the allowed limit.
4. Syndrome processing and CSS correction succeeded.
5. The complete 128-bit payload was recovered.
6. The reconstructed KMAC authentication tag is valid.

The Gaussian Process model cannot override a failed deterministic
security condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.authentication_server.check_block_analyzer import (
    CheckBlockAnalysisResult,
)

from src.common.constants import (
    FIXED_QBER_THRESHOLD,
    KMAC_TAG_BITS,
    MAXIMUM_ACCEPTABLE_LOSS_RATE,
    MINIMUM_OBSERVED_CHECK_BLOCKS,
    PAYLOAD_LOGICAL_QUBITS,
)

from src.common.enums import (
    FailureReason,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.validators import (
    validate_integer,
    validate_probability,
    validate_qber,
)


# ---------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class DeterministicVerificationResult:
    """
    Complete deterministic verification result.

    The `passed` field is True only when every mandatory condition
    succeeds.
    """

    passed: bool

    check_evidence_pass: bool
    qber_pass: bool
    loss_pass: bool
    syndrome_pass: bool
    payload_pass: bool
    tag_pass: bool

    qber_raw: float
    loss_rate: float

    observed_check_blocks: int
    required_check_blocks: int

    recovered_payload_bits: int
    expected_payload_bits: int

    correction_failures: int

    qber_threshold: float
    maximum_loss_rate: float

    failure_reason: FailureReason
    message: str

    details: dict[str, Any]

    def __post_init__(self) -> None:
        boolean_fields = (
            self.passed,
            self.check_evidence_pass,
            self.qber_pass,
            self.loss_pass,
            self.syndrome_pass,
            self.payload_pass,
            self.tag_pass,
        )

        if any(
            not isinstance(value, bool)
            for value in boolean_fields
        ):
            raise ProtocolValidationError(
                "Deterministic verification flags must be Boolean."
            )

        validate_qber(
            self.qber_raw
        )

        validate_probability(
            self.loss_rate,
            field_name="loss_rate",
        )

        validate_integer(
            self.observed_check_blocks,
            field_name="observed_check_blocks",
            minimum=0,
        )

        validate_integer(
            self.required_check_blocks,
            field_name="required_check_blocks",
            minimum=1,
        )

        validate_integer(
            self.recovered_payload_bits,
            field_name="recovered_payload_bits",
            minimum=0,
        )

        validate_integer(
            self.expected_payload_bits,
            field_name="expected_payload_bits",
            minimum=1,
        )

        validate_integer(
            self.correction_failures,
            field_name="correction_failures",
            minimum=0,
        )

        validate_probability(
            self.qber_threshold,
            field_name="qber_threshold",
        )

        validate_probability(
            self.maximum_loss_rate,
            field_name="maximum_loss_rate",
        )

        if not isinstance(
            self.failure_reason,
            FailureReason,
        ):
            raise ProtocolValidationError(
                "failure_reason must be a FailureReason value."
            )

        if not isinstance(
            self.message,
            str,
        ):
            raise ProtocolValidationError(
                "Verification message must be a string."
            )

        if not isinstance(
            self.details,
            dict,
        ):
            raise ProtocolValidationError(
                "Verification details must be a dictionary."
            )

        expected_pass = all(
            (
                self.check_evidence_pass,
                self.qber_pass,
                self.loss_pass,
                self.syndrome_pass,
                self.payload_pass,
                self.tag_pass,
            )
        )

        if self.passed != expected_pass:
            raise ProtocolValidationError(
                (
                    "The deterministic passed value does not match "
                    "the individual verification results."
                )
            )

        if (
            self.passed
            and self.failure_reason
            != FailureReason.NONE
        ):
            raise ProtocolValidationError(
                (
                    "A successful deterministic result cannot "
                    "contain a failure reason."
                )
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible result dictionary.
        """

        result = asdict(self)

        result["failure_reason"] = (
            self.failure_reason.value
        )

        return result


# ---------------------------------------------------------------------
# Check-analysis normalization
# ---------------------------------------------------------------------

def _normalize_check_analysis(
    check_analysis: CheckBlockAnalysisResult
    | Mapping[str, Any],
) -> dict[str, Any]:
    """
    Convert check-block analysis into a common dictionary form.
    """

    if isinstance(
        check_analysis,
        CheckBlockAnalysisResult,
    ):
        return check_analysis.to_dict()

    if not isinstance(
        check_analysis,
        Mapping,
    ):
        raise ProtocolValidationError(
            (
                "check_analysis must be a "
                "CheckBlockAnalysisResult or mapping."
            ),
            details={
                "received_type": type(
                    check_analysis
                ).__name__,
            },
        )

    required_fields = (
        "qber_raw",
        "observed_check_blocks",
    )

    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in check_analysis
    ]

    if missing_fields:
        raise ProtocolValidationError(
            "Check-block analysis data is incomplete.",
            details={
                "missing_fields": missing_fields,
            },
        )

    return dict(
        check_analysis
    )


def _validate_boolean(
    value: Any,
    *,
    field_name: str,
) -> bool:
    """
    Validate a strict Boolean value.
    """

    if not isinstance(
        value,
        bool,
    ):
        raise ProtocolValidationError(
            f"{field_name} must be Boolean.",
            details={
                "field_name": field_name,
                "received_type": type(
                    value
                ).__name__,
            },
        )

    return value


# ---------------------------------------------------------------------
# Failure-reason selection
# ---------------------------------------------------------------------

def select_deterministic_failure_reason(
    *,
    check_evidence_pass: bool,
    qber_pass: bool,
    loss_pass: bool,
    syndrome_pass: bool,
    payload_pass: bool,
    tag_pass: bool,
) -> FailureReason:
    """
    Select the first mandatory deterministic failure.

    Evaluation order follows protocol dependency order.
    """

    if not check_evidence_pass:
        return (
            FailureReason
            .INSUFFICIENT_CHECK_BLOCKS
        )

    if not qber_pass:
        return FailureReason.EXCESSIVE_QBER

    if not loss_pass:
        return FailureReason.EXCESSIVE_LOSS

    if not syndrome_pass:
        return (
            FailureReason
            .UNCORRECTABLE_QUANTUM_ERROR
        )

    if not payload_pass:
        return (
            FailureReason
            .PAYLOAD_RECOVERY_FAILED
        )

    if not tag_pass:
        return FailureReason.KMAC_TAG_MISMATCH

    return FailureReason.NONE


def build_deterministic_message(
    failure_reason: FailureReason,
) -> str:
    """
    Create a readable deterministic-verification message.
    """

    messages = {
        FailureReason.NONE: (
            "All deterministic authentication checks passed."
        ),

        FailureReason.INSUFFICIENT_CHECK_BLOCKS: (
            "Authentication failed because insufficient "
            "check-block evidence was available."
        ),

        FailureReason.EXCESSIVE_QBER: (
            "Authentication failed because raw QBER exceeded "
            "the fixed security threshold."
        ),

        FailureReason.EXCESSIVE_LOSS: (
            "Authentication failed because quantum-channel loss "
            "exceeded the allowed limit."
        ),

        FailureReason.UNCORRECTABLE_QUANTUM_ERROR: (
            "Authentication failed because one or more quantum "
            "blocks contained uncorrectable errors."
        ),

        FailureReason.PAYLOAD_RECOVERY_FAILED: (
            "Authentication failed because the complete KMAC "
            "payload could not be recovered."
        ),

        FailureReason.KMAC_TAG_MISMATCH: (
            "Authentication failed because the reconstructed "
            "KMAC tag was invalid."
        ),
    }

    return messages.get(
        failure_reason,
        "Deterministic authentication verification failed.",
    )


# ---------------------------------------------------------------------
# Main verification function
# ---------------------------------------------------------------------

def verify_deterministic_conditions(
    *,
    check_analysis: CheckBlockAnalysisResult
    | Mapping[str, Any],
    loss_rate: float,
    syndrome_processing_pass: bool,
    payload_recovery_pass: bool,
    recovered_payload_bits: int,
    tag_valid: bool,
    correction_failures: int = 0,
    required_check_blocks: int = (
        MINIMUM_OBSERVED_CHECK_BLOCKS
    ),
    qber_threshold: float = (
        FIXED_QBER_THRESHOLD
    ),
    maximum_loss_rate: float = (
        MAXIMUM_ACCEPTABLE_LOSS_RATE
    ),
    expected_payload_bits: int = (
        PAYLOAD_LOGICAL_QUBITS
    ),
    details: Mapping[str, Any] | None = None,
) -> DeterministicVerificationResult:
    """
    Evaluate all mandatory deterministic authentication conditions.

    This function does not invoke the Gaussian Process detector.
    """

    analysis = _normalize_check_analysis(
        check_analysis
    )

    qber_raw = validate_qber(
        analysis["qber_raw"]
    )

    observed_check_blocks = validate_integer(
        analysis[
            "observed_check_blocks"
        ],
        field_name="observed_check_blocks",
        minimum=0,
    )

    validated_loss_rate = (
        validate_probability(
            loss_rate,
            field_name="loss_rate",
        )
    )

    validated_syndrome_pass = (
        _validate_boolean(
            syndrome_processing_pass,
            field_name=(
                "syndrome_processing_pass"
            ),
        )
    )

    validated_payload_pass = (
        _validate_boolean(
            payload_recovery_pass,
            field_name=(
                "payload_recovery_pass"
            ),
        )
    )

    validated_tag_pass = _validate_boolean(
        tag_valid,
        field_name="tag_valid",
    )

    validated_recovered_bits = (
        validate_integer(
            recovered_payload_bits,
            field_name=(
                "recovered_payload_bits"
            ),
            minimum=0,
        )
    )

    validated_correction_failures = (
        validate_integer(
            correction_failures,
            field_name=(
                "correction_failures"
            ),
            minimum=0,
        )
    )

    validated_required_checks = (
        validate_integer(
            required_check_blocks,
            field_name=(
                "required_check_blocks"
            ),
            minimum=1,
        )
    )

    validated_qber_threshold = (
        validate_probability(
            qber_threshold,
            field_name="qber_threshold",
        )
    )

    validated_maximum_loss = (
        validate_probability(
            maximum_loss_rate,
            field_name=(
                "maximum_loss_rate"
            ),
        )
    )

    validated_expected_payload_bits = (
        validate_integer(
            expected_payload_bits,
            field_name=(
                "expected_payload_bits"
            ),
            minimum=1,
        )
    )

    check_evidence_pass = (
        observed_check_blocks
        >= validated_required_checks
    )

    qber_pass = (
        qber_raw
        <= validated_qber_threshold
    )

    loss_pass = (
        validated_loss_rate
        <= validated_maximum_loss
    )

    syndrome_pass = (
        validated_syndrome_pass
        and validated_correction_failures == 0
    )

    payload_pass = (
        validated_payload_pass
        and validated_recovered_bits
        == validated_expected_payload_bits
    )

    tag_pass = validated_tag_pass

    passed = all(
        (
            check_evidence_pass,
            qber_pass,
            loss_pass,
            syndrome_pass,
            payload_pass,
            tag_pass,
        )
    )

    failure_reason = (
        select_deterministic_failure_reason(
            check_evidence_pass=(
                check_evidence_pass
            ),
            qber_pass=qber_pass,
            loss_pass=loss_pass,
            syndrome_pass=syndrome_pass,
            payload_pass=payload_pass,
            tag_pass=tag_pass,
        )
    )

    normalized_details = (
        {}
        if details is None
        else dict(details)
    )

    normalized_details.update(
        {
            "kmac_tag_bits": KMAC_TAG_BITS,
            "check_analysis": analysis,
        }
    )

    return DeterministicVerificationResult(
        passed=passed,

        check_evidence_pass=(
            check_evidence_pass
        ),

        qber_pass=qber_pass,
        loss_pass=loss_pass,
        syndrome_pass=syndrome_pass,
        payload_pass=payload_pass,
        tag_pass=tag_pass,

        qber_raw=qber_raw,
        loss_rate=validated_loss_rate,

        observed_check_blocks=(
            observed_check_blocks
        ),

        required_check_blocks=(
            validated_required_checks
        ),

        recovered_payload_bits=(
            validated_recovered_bits
        ),

        expected_payload_bits=(
            validated_expected_payload_bits
        ),

        correction_failures=(
            validated_correction_failures
        ),

        qber_threshold=(
            validated_qber_threshold
        ),

        maximum_loss_rate=(
            validated_maximum_loss
        ),

        failure_reason=failure_reason,

        message=build_deterministic_message(
            failure_reason
        ),

        details=normalized_details,
    )


def require_deterministic_pass(
    result: DeterministicVerificationResult,
) -> None:
    """
    Raise ProtocolValidationError when deterministic verification fails.
    """

    if not isinstance(
        result,
        DeterministicVerificationResult,
    ):
        raise ProtocolValidationError(
            (
                "result must be a "
                "DeterministicVerificationResult object."
            )
        )

    if not result.passed:
        raise ProtocolValidationError(
            result.message,
            code=(
                result.failure_reason.value.upper()
            ),
            details=result.to_dict(),
        )


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_deterministic_verifier_self_test() -> dict[str, Any]:
    """
    Run successful and failed deterministic-verification tests.
    """

    valid_check_analysis = {
        "qber_raw": 0.035,
        "observed_check_blocks": 30,
        "minimum_evidence_pass": True,
        "fixed_qber_pass": True,
    }

    valid_result = (
        verify_deterministic_conditions(
            check_analysis=(
                valid_check_analysis
            ),
            loss_rate=0.05,
            syndrome_processing_pass=True,
            payload_recovery_pass=True,
            recovered_payload_bits=128,
            tag_valid=True,
            correction_failures=0,
        )
    )

    excessive_qber_result = (
        verify_deterministic_conditions(
            check_analysis={
                "qber_raw": 0.25,
                "observed_check_blocks": 32,
            },
            loss_rate=0.02,
            syndrome_processing_pass=True,
            payload_recovery_pass=True,
            recovered_payload_bits=128,
            tag_valid=True,
            correction_failures=0,
        )
    )

    invalid_tag_result = (
        verify_deterministic_conditions(
            check_analysis=(
                valid_check_analysis
            ),
            loss_rate=0.05,
            syndrome_processing_pass=True,
            payload_recovery_pass=True,
            recovered_payload_bits=128,
            tag_valid=False,
            correction_failures=0,
        )
    )

    incomplete_payload_result = (
        verify_deterministic_conditions(
            check_analysis=(
                valid_check_analysis
            ),
            loss_rate=0.05,
            syndrome_processing_pass=True,
            payload_recovery_pass=True,
            recovered_payload_bits=127,
            tag_valid=True,
            correction_failures=0,
        )
    )

    success = all(
        (
            valid_result.passed,

            not excessive_qber_result.passed,

            excessive_qber_result.failure_reason
            == FailureReason.EXCESSIVE_QBER,

            not invalid_tag_result.passed,

            invalid_tag_result.failure_reason
            == FailureReason.KMAC_TAG_MISMATCH,

            not incomplete_payload_result.passed,

            incomplete_payload_result.failure_reason
            == FailureReason.PAYLOAD_RECOVERY_FAILED,
        )
    )

    return {
        "success": success,

        "valid_result_passed": (
            valid_result.passed
        ),

        "valid_failure_reason": (
            valid_result.failure_reason.value
        ),

        "excessive_qber_rejected": (
            not excessive_qber_result.passed
        ),

        "excessive_qber_reason": (
            excessive_qber_result
            .failure_reason
            .value
        ),

        "invalid_tag_rejected": (
            not invalid_tag_result.passed
        ),

        "invalid_tag_reason": (
            invalid_tag_result
            .failure_reason
            .value
        ),

        "incomplete_payload_rejected": (
            not incomplete_payload_result.passed
        ),

        "incomplete_payload_reason": (
            incomplete_payload_result
            .failure_reason
            .value
        ),
    }


__all__ = [
    "DeterministicVerificationResult",
    "select_deterministic_failure_reason",
    "build_deterministic_message",
    "verify_deterministic_conditions",
    "require_deterministic_pass",
    "run_deterministic_verifier_self_test",
]