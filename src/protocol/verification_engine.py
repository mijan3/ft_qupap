"""
FT-QuPAP Deterministic Verification Engine

This module performs the mandatory deterministic verification stage of
the FT-QuPAP protocol.

The Gaussian Process detector must never replace these checks.
A session is eligible for the adaptive GP decision only when every
mandatory deterministic condition passes.

Notebook-compatible checks:

    1. Authentication Server credential is valid.
    2. Authentication request is fresh and replay-safe.
    3. AES-GCM control schedule was decrypted and validated.
    4. Received blocks match the authenticated schedule.
    5. Every required payload block is recoverable.
    6. At least 24 declared check blocks were observed.
    7. Physical-qubit loss does not exceed 15 percent.
    8. Recovered KMAC tag equals the independently computed tag.

Notebook-compatible failure reasons:

    invalid_server_credential
    freshness_or_replay_failure
    invalid_control_schedule
    schedule_block_mismatch
    payload_block_unrecoverable
    insufficient_check_block_evidence
    loss_rate_exceeds_policy
    authentication_tag_mismatch
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .result_models import VerificationResult


DEFAULT_MIN_OBSERVED_CHECK_BLOCKS = 24
DEFAULT_MAX_ACCEPTABLE_LOSS_RATE = 0.15
DEFAULT_STEANE_BLOCK_SIZE = 7


REASON_INVALID_SERVER_CREDENTIAL = (
    "invalid_server_credential"
)

REASON_FRESHNESS_OR_REPLAY_FAILURE = (
    "freshness_or_replay_failure"
)

REASON_INVALID_CONTROL_SCHEDULE = (
    "invalid_control_schedule"
)

REASON_SCHEDULE_BLOCK_MISMATCH = (
    "schedule_block_mismatch"
)

REASON_PAYLOAD_BLOCK_UNRECOVERABLE = (
    "payload_block_unrecoverable"
)

REASON_INSUFFICIENT_CHECK_EVIDENCE = (
    "insufficient_check_block_evidence"
)

REASON_LOSS_RATE_EXCEEDS_POLICY = (
    "loss_rate_exceeds_policy"
)

REASON_AUTHENTICATION_TAG_MISMATCH = (
    "authentication_tag_mismatch"
)


class VerificationEngineError(Exception):
    """Base exception for deterministic-verification failures."""


class InvalidVerificationInputError(VerificationEngineError):
    """Raised when deterministic evidence is malformed."""


class InvalidDecoderRecordError(VerificationEngineError):
    """Raised when a decoder record is invalid."""


class InvalidVerificationPolicyError(VerificationEngineError):
    """Raised when deterministic policy values are invalid."""


@dataclass(frozen=True)
class VerificationPolicy:
    """
    Deterministic FT-QuPAP verification policy.

    The default values match the final operational-threshold notebook.
    """

    min_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    )

    max_acceptable_loss_rate: float = (
        DEFAULT_MAX_ACCEPTABLE_LOSS_RATE
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.min_observed_check_blocks,
                bool,
            )
            or not isinstance(
                self.min_observed_check_blocks,
                int,
            )
        ):
            raise TypeError(
                "min_observed_check_blocks must be an integer."
            )

        if self.min_observed_check_blocks < 1:
            raise InvalidVerificationPolicyError(
                "min_observed_check_blocks must be at least 1."
            )

        if (
            isinstance(
                self.max_acceptable_loss_rate,
                bool,
            )
            or not isinstance(
                self.max_acceptable_loss_rate,
                (int, float),
            )
        ):
            raise TypeError(
                "max_acceptable_loss_rate must be numeric."
            )

        normalized_loss_limit = float(
            self.max_acceptable_loss_rate
        )

        if not 0.0 <= normalized_loss_limit <= 1.0:
            raise InvalidVerificationPolicyError(
                "max_acceptable_loss_rate must be "
                "between 0 and 1."
            )

        object.__setattr__(
            self,
            "max_acceptable_loss_rate",
            normalized_loss_limit,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable policy dictionary."""

        return {
            "min_observed_check_blocks":
                self.min_observed_check_blocks,
            "max_acceptable_loss_rate":
                self.max_acceptable_loss_rate,
        }


@dataclass(frozen=True)
class DeterministicVerificationEvidence:
    """
    Detailed non-secret deterministic verification evidence.

    This object contains no secret keys or raw authentication tags.
    """

    credential_valid: bool
    freshness_valid: bool
    schedule_valid: bool
    schedule_blocks_valid: bool
    payload_blocks_recoverable: bool
    check_evidence_sufficient: bool
    loss_policy_valid: bool
    tag_valid: bool

    qber_observed: int
    physical_qubits_per_check_block: int
    observed_check_blocks: int
    required_check_blocks: int

    loss_rate: float
    maximum_loss_rate: float

    schedule_mismatch_count: int
    uncorrectable_payload_count: int
    payload_failure_count: int

    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        boolean_fields = {
            "credential_valid":
                self.credential_valid,
            "freshness_valid":
                self.freshness_valid,
            "schedule_valid":
                self.schedule_valid,
            "schedule_blocks_valid":
                self.schedule_blocks_valid,
            "payload_blocks_recoverable":
                self.payload_blocks_recoverable,
            "check_evidence_sufficient":
                self.check_evidence_sufficient,
            "loss_policy_valid":
                self.loss_policy_valid,
            "tag_valid":
                self.tag_valid,
        }

        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        integer_fields = {
            "qber_observed":
                self.qber_observed,
            "physical_qubits_per_check_block":
                self.physical_qubits_per_check_block,
            "observed_check_blocks":
                self.observed_check_blocks,
            "required_check_blocks":
                self.required_check_blocks,
            "schedule_mismatch_count":
                self.schedule_mismatch_count,
            "uncorrectable_payload_count":
                self.uncorrectable_payload_count,
            "payload_failure_count":
                self.payload_failure_count,
        }

        for field_name, value in integer_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if self.physical_qubits_per_check_block < 1:
            raise ValueError(
                "physical_qubits_per_check_block "
                "must be at least 1."
            )

        for field_name, value in {
            "loss_rate":
                self.loss_rate,
            "maximum_loss_rate":
                self.maximum_loss_rate,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"{field_name} must be numeric."
                )

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if not isinstance(self.reasons, tuple):
            raise TypeError(
                "reasons must be a tuple."
            )

        for reason in self.reasons:
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    "Every verification reason must "
                    "be a nonempty string."
                )

    @property
    def deterministic_pass(self) -> bool:
        """Return whether every deterministic invariant passed."""

        return all(
            (
                self.credential_valid,
                self.freshness_valid,
                self.schedule_valid,
                self.schedule_blocks_valid,
                self.payload_blocks_recoverable,
                self.check_evidence_sufficient,
                self.loss_policy_valid,
                self.tag_valid,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return non-secret verification evidence."""

        return {
            "credential_valid":
                self.credential_valid,
            "freshness_valid":
                self.freshness_valid,
            "schedule_valid":
                self.schedule_valid,
            "schedule_blocks_valid":
                self.schedule_blocks_valid,
            "payload_blocks_recoverable":
                self.payload_blocks_recoverable,
            "check_evidence_sufficient":
                self.check_evidence_sufficient,
            "loss_policy_valid":
                self.loss_policy_valid,
            "tag_valid":
                self.tag_valid,
            "deterministic_pass":
                self.deterministic_pass,
            "qber_observed":
                self.qber_observed,
            "physical_qubits_per_check_block":
                self.physical_qubits_per_check_block,
            "observed_check_blocks":
                self.observed_check_blocks,
            "required_check_blocks":
                self.required_check_blocks,
            "loss_rate":
                self.loss_rate,
            "maximum_loss_rate":
                self.maximum_loss_rate,
            "schedule_mismatch_count":
                self.schedule_mismatch_count,
            "uncorrectable_payload_count":
                self.uncorrectable_payload_count,
            "payload_failure_count":
                self.payload_failure_count,
            "deterministic_reasons":
                list(self.reasons),
        }


@dataclass(frozen=True)
class VerificationEngineOutput:
    """
    Combined result-model output and detailed verification evidence.
    """

    verification: VerificationResult
    evidence: DeterministicVerificationEvidence

    def __post_init__(self) -> None:
        if not isinstance(
            self.verification,
            VerificationResult,
        ):
            raise TypeError(
                "verification must be VerificationResult."
            )

        if not isinstance(
            self.evidence,
            DeterministicVerificationEvidence,
        ):
            raise TypeError(
                "evidence must be "
                "DeterministicVerificationEvidence."
            )

        if (
            self.verification.deterministic_pass
            != self.evidence.deterministic_pass
        ):
            raise VerificationEngineError(
                "VerificationResult and evidence disagree."
            )

        if (
            tuple(self.verification.reasons)
            != tuple(self.evidence.reasons)
        ):
            raise VerificationEngineError(
                "Verification reasons are inconsistent."
            )

    @property
    def deterministic_pass(self) -> bool:
        """Return the final deterministic verification status."""

        return self.verification.deterministic_pass

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return ordered deterministic failure reasons."""

        return tuple(
            self.verification.reasons
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the complete non-secret verification output."""

        return {
            "verification":
                self.verification.as_dict(),
            "evidence":
                self.evidence.as_dict(),
        }


def validate_boolean(
    value: Any,
    field_name: str,
) -> bool:
    """Validate a boolean protocol result."""

    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be boolean."
        )

    return value


def validate_nonnegative_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a nonnegative integer."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return value


def validate_loss_rate(
    loss_rate: Any,
) -> float:
    """Validate a physical-qubit loss rate."""

    if (
        isinstance(loss_rate, bool)
        or not isinstance(
            loss_rate,
            (int, float),
        )
    ):
        raise TypeError(
            "loss_rate must be numeric."
        )

    normalized = float(
        loss_rate
    )

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            "loss_rate must be between 0 and 1."
        )

    return normalized


def normalize_decoder_records(
    decoder_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Validate and detach syndrome-decoder records.
    """

    if isinstance(
        decoder_records,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "decoder_records must be a sequence of mappings."
        )

    if not isinstance(
        decoder_records,
        Sequence,
    ):
        raise TypeError(
            "decoder_records must be a sequence."
        )

    normalized: list[dict[str, Any]] = []

    for index, record in enumerate(
        decoder_records
    ):
        if not isinstance(
            record,
            Mapping,
        ):
            raise InvalidDecoderRecordError(
                f"Decoder record {index} must be a mapping."
            )

        normalized.append(
            dict(record)
        )

    return normalized


def normalize_payload_failures(
    payload_failures: Sequence[str] | None,
) -> list[str]:
    """
    Validate payload-decoding failure identifiers.
    """

    if payload_failures is None:
        return []

    if isinstance(
        payload_failures,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "payload_failures must be a sequence of strings."
        )

    if not isinstance(
        payload_failures,
        Sequence,
    ):
        raise TypeError(
            "payload_failures must be a sequence."
        )

    normalized: list[str] = []

    for index, failure in enumerate(
        payload_failures
    ):
        if not isinstance(
            failure,
            str,
        ):
            raise TypeError(
                f"payload_failures[{index}] must be a string."
            )

        failure = failure.strip()

        if not failure:
            raise ValueError(
                f"payload_failures[{index}] cannot be empty."
            )

        if failure not in normalized:
            normalized.append(
                failure
            )

    return normalized


def validate_tag(
    tag: Any,
    field_name: str,
    allow_none: bool,
) -> bytes | None:
    """
    Validate one KMAC tag without exposing it in result objects.
    """

    if tag is None:
        if allow_none:
            return None

        raise InvalidVerificationInputError(
            f"{field_name} cannot be None."
        )

    if not isinstance(
        tag,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        raise TypeError(
            f"{field_name} must be bytes-like."
        )

    normalized = bytes(
        tag
    )

    if not normalized:
        raise InvalidVerificationInputError(
            f"{field_name} cannot be empty."
        )

    return normalized


def calculate_observed_check_blocks(
    qber_observed: int,
    physical_qubits_per_check_block: int,
) -> int:
    """
    Convert observed physical check bits to observed check blocks.

    Notebook operation:

        observed_check_blocks =
            qber_observed // physical_qubits_per_check_block
    """

    observed_bits = validate_nonnegative_integer(
        qber_observed,
        "qber_observed",
    )

    block_width = validate_nonnegative_integer(
        physical_qubits_per_check_block,
        "physical_qubits_per_check_block",
    )

    block_width = max(
        block_width,
        1,
    )

    return int(
        observed_bits
        // block_width
    )


def find_schedule_mismatches(
    decoder_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return blocks whose identity or schedule role is invalid.

    Notebook-compatible conditions:

        role == "unknown"

        or

        status == "schedule_block_mismatch"
    """

    normalized_records = (
        normalize_decoder_records(
            decoder_records
        )
    )

    return [
        record
        for record in normalized_records
        if record.get("role") == "unknown"
        or record.get("status")
        == REASON_SCHEDULE_BLOCK_MISMATCH
    ]


def find_uncorrectable_payload_blocks(
    decoder_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    Return required payload blocks that were not recoverable.
    """

    normalized_records = (
        normalize_decoder_records(
            decoder_records
        )
    )

    return [
        record
        for record in normalized_records
        if record.get("role") == "payload"
        and not bool(
            record.get(
                "correctable",
                False,
            )
        )
    ]


def verify_authentication_tag(
    received_tag: bytes | None,
    expected_tag: bytes,
) -> bool:
    """
    Compare the recovered and expected KMAC tags in constant time.
    """

    normalized_expected = validate_tag(
        expected_tag,
        "expected_tag",
        allow_none=False,
    )

    normalized_received = validate_tag(
        received_tag,
        "received_tag",
        allow_none=True,
    )

    if normalized_received is None:
        return False

    return hmac.compare_digest(
        normalized_received,
        normalized_expected,
    )


def unique_reasons(
    reasons: Sequence[str],
) -> tuple[str, ...]:
    """
    Preserve reason ordering while removing duplicates.
    """

    ordered: list[str] = []

    for reason in reasons:
        if reason not in ordered:
            ordered.append(
                reason
            )

    return tuple(
        ordered
    )


def evaluate_deterministic_verification(
    *,
    credential_valid: bool,
    freshness_valid: bool,
    schedule_valid: bool,
    decoder_records: Sequence[Mapping[str, Any]],
    payload_failures: Sequence[str] | None,
    qber_observed: int,
    physical_qubits_per_check_block: int,
    received_tag: bytes | None,
    expected_tag: bytes,
    loss_rate: float,
    policy: VerificationPolicy | None = None,
) -> VerificationEngineOutput:
    """
    Evaluate every mandatory FT-QuPAP deterministic invariant.

    Raw QBER must already have been calculated from declared check
    blocks before this function is called.
    """

    active_policy = (
        policy
        if policy is not None
        else VerificationPolicy()
    )

    if not isinstance(
        active_policy,
        VerificationPolicy,
    ):
        raise TypeError(
            "policy must be VerificationPolicy or None."
        )

    credential_valid = validate_boolean(
        credential_valid,
        "credential_valid",
    )

    freshness_valid = validate_boolean(
        freshness_valid,
        "freshness_valid",
    )

    schedule_valid = validate_boolean(
        schedule_valid,
        "schedule_valid",
    )

    normalized_records = (
        normalize_decoder_records(
            decoder_records
        )
    )

    normalized_payload_failures = (
        normalize_payload_failures(
            payload_failures
        )
    )

    qber_observed = validate_nonnegative_integer(
        qber_observed,
        "qber_observed",
    )

    physical_qubits_per_check_block = (
        validate_nonnegative_integer(
            physical_qubits_per_check_block,
            "physical_qubits_per_check_block",
        )
    )

    physical_qubits_per_check_block = max(
        physical_qubits_per_check_block,
        1,
    )

    loss_rate = validate_loss_rate(
        loss_rate
    )

    schedule_mismatches = (
        find_schedule_mismatches(
            normalized_records
        )
    )

    bad_payload_blocks = (
        find_uncorrectable_payload_blocks(
            normalized_records
        )
    )

    observed_check_blocks = (
        calculate_observed_check_blocks(
            qber_observed=(
                qber_observed
            ),
            physical_qubits_per_check_block=(
                physical_qubits_per_check_block
            ),
        )
    )

    schedule_blocks_valid = (
        len(schedule_mismatches) == 0
    )

    payload_blocks_recoverable = (
        len(bad_payload_blocks) == 0
        and len(normalized_payload_failures) == 0
    )

    check_evidence_sufficient = (
        observed_check_blocks
        >= active_policy.min_observed_check_blocks
    )

    loss_policy_valid = (
        loss_rate
        <= active_policy.max_acceptable_loss_rate
    )

    tag_valid = verify_authentication_tag(
        received_tag=received_tag,
        expected_tag=expected_tag,
    )

    reasons: list[str] = []

    if not credential_valid:
        reasons.append(
            REASON_INVALID_SERVER_CREDENTIAL
        )

    if not freshness_valid:
        reasons.append(
            REASON_FRESHNESS_OR_REPLAY_FAILURE
        )

    if not schedule_valid:
        reasons.append(
            REASON_INVALID_CONTROL_SCHEDULE
        )

    if not schedule_blocks_valid:
        reasons.append(
            REASON_SCHEDULE_BLOCK_MISMATCH
        )

    if not payload_blocks_recoverable:
        reasons.append(
            REASON_PAYLOAD_BLOCK_UNRECOVERABLE
        )

    if not check_evidence_sufficient:
        reasons.append(
            REASON_INSUFFICIENT_CHECK_EVIDENCE
        )

    if not loss_policy_valid:
        reasons.append(
            REASON_LOSS_RATE_EXCEEDS_POLICY
        )

    if not tag_valid:
        reasons.append(
            REASON_AUTHENTICATION_TAG_MISMATCH
        )

    normalized_reasons = unique_reasons(
        reasons
    )

    evidence = DeterministicVerificationEvidence(
        credential_valid=credential_valid,
        freshness_valid=freshness_valid,
        schedule_valid=schedule_valid,
        schedule_blocks_valid=(
            schedule_blocks_valid
        ),
        payload_blocks_recoverable=(
            payload_blocks_recoverable
        ),
        check_evidence_sufficient=(
            check_evidence_sufficient
        ),
        loss_policy_valid=(
            loss_policy_valid
        ),
        tag_valid=tag_valid,
        qber_observed=qber_observed,
        physical_qubits_per_check_block=(
            physical_qubits_per_check_block
        ),
        observed_check_blocks=(
            observed_check_blocks
        ),
        required_check_blocks=(
            active_policy.min_observed_check_blocks
        ),
        loss_rate=loss_rate,
        maximum_loss_rate=(
            active_policy.max_acceptable_loss_rate
        ),
        schedule_mismatch_count=len(
            schedule_mismatches
        ),
        uncorrectable_payload_count=len(
            bad_payload_blocks
        ),
        payload_failure_count=len(
            normalized_payload_failures
        ),
        reasons=normalized_reasons,
    )

    verification = VerificationResult(
        credential_valid=(
            credential_valid
        ),

        # The notebook returns one combined
        # freshness/replay verification value.
        request_fresh=(
            freshness_valid
        ),
        replay_safe=(
            freshness_valid
        ),

        # A schedule is valid only when it decrypts correctly and
        # all received blocks match authenticated schedule entries.
        schedule_valid=(
            schedule_valid
            and schedule_blocks_valid
        ),

        check_evidence_sufficient=(
            check_evidence_sufficient
        ),

        required_blocks_correctable=(
            payload_blocks_recoverable
        ),

        tag_valid=tag_valid,

        loss_policy_valid=(
            loss_policy_valid
        ),

        reasons=normalized_reasons,
    )

    return VerificationEngineOutput(
        verification=verification,
        evidence=evidence,
    )


def deterministic_protocol_checks(
    credential_valid: bool,
    freshness_valid: bool,
    schedule_valid: bool,
    decoder_records: Sequence[Mapping[str, Any]],
    payload_failures: Sequence[str] | None,
    qber_observed: int,
    physical_qubits_per_check_block: int,
    received_tag: bytes | None,
    expected_tag: bytes,
    loss_rate: float,
    *,
    min_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    ),
    max_acceptable_loss_rate: float = (
        DEFAULT_MAX_ACCEPTABLE_LOSS_RATE
    ),
) -> tuple[bool, list[str]]:
    """
    Notebook-compatible deterministic-check function.

    Returns:
        deterministic_ok:
            True only when every mandatory invariant passes.

        deterministic_reasons:
            Ordered list of deterministic rejection reasons.
    """

    output = evaluate_deterministic_verification(
        credential_valid=credential_valid,
        freshness_valid=freshness_valid,
        schedule_valid=schedule_valid,
        decoder_records=decoder_records,
        payload_failures=payload_failures,
        qber_observed=qber_observed,
        physical_qubits_per_check_block=(
            physical_qubits_per_check_block
        ),
        received_tag=received_tag,
        expected_tag=expected_tag,
        loss_rate=loss_rate,
        policy=VerificationPolicy(
            min_observed_check_blocks=(
                min_observed_check_blocks
            ),
            max_acceptable_loss_rate=(
                max_acceptable_loss_rate
            ),
        ),
    )

    return (
        output.deterministic_pass,
        list(
            output.reasons
        ),
    )


class FTQuPAPVerificationEngine:
    """
    Reusable deterministic verification service.
    """

    def __init__(
        self,
        policy: VerificationPolicy | None = None,
    ) -> None:
        self.policy = (
            policy
            if policy is not None
            else VerificationPolicy()
        )

        if not isinstance(
            self.policy,
            VerificationPolicy,
        ):
            raise TypeError(
                "policy must be VerificationPolicy."
            )

    def verify(
        self,
        *,
        credential_valid: bool,
        freshness_valid: bool,
        schedule_valid: bool,
        decoder_records: Sequence[
            Mapping[str, Any]
        ],
        payload_failures: Sequence[str] | None,
        qber_observed: int,
        physical_qubits_per_check_block: int,
        received_tag: bytes | None,
        expected_tag: bytes,
        loss_rate: float,
    ) -> VerificationEngineOutput:
        """
        Execute deterministic FT-QuPAP verification.
        """

        return evaluate_deterministic_verification(
            credential_valid=credential_valid,
            freshness_valid=freshness_valid,
            schedule_valid=schedule_valid,
            decoder_records=decoder_records,
            payload_failures=payload_failures,
            qber_observed=qber_observed,
            physical_qubits_per_check_block=(
                physical_qubits_per_check_block
            ),
            received_tag=received_tag,
            expected_tag=expected_tag,
            loss_rate=loss_rate,
            policy=self.policy,
        )


def run_self_test() -> None:
    """
    Verify successful and rejected deterministic sessions.
    """

    expected_tag = bytes.fromhex(
        "00112233445566778899aabbccddeeff"
    )

    valid_decoder_records = [
        {
            "block_id": f"P{index:04d}",
            "role": "payload",
            "status": "corrected",
            "correctable": True,
            "syndrome_weight": 0,
        }
        for index in range(128)
    ]

    valid_output = (
        evaluate_deterministic_verification(
            credential_valid=True,
            freshness_valid=True,
            schedule_valid=True,
            decoder_records=(
                valid_decoder_records
            ),
            payload_failures=[],
            qber_observed=(
                32
                * DEFAULT_STEANE_BLOCK_SIZE
            ),
            physical_qubits_per_check_block=(
                DEFAULT_STEANE_BLOCK_SIZE
            ),
            received_tag=expected_tag,
            expected_tag=expected_tag,
            loss_rate=0.01,
        )
    )

    if not valid_output.deterministic_pass:
        raise VerificationEngineError(
            "Valid deterministic session was rejected."
        )

    invalid_decoder_records = [
        *valid_decoder_records,
        {
            "block_id": "P0128",
            "role": "payload",
            "status": "uncorrectable",
            "correctable": False,
            "syndrome_weight": 3,
        },
        {
            "block_id": "UNKNOWN",
            "role": "unknown",
            "status": (
                REASON_SCHEDULE_BLOCK_MISMATCH
            ),
            "correctable": False,
        },
    ]

    rejected_output = (
        evaluate_deterministic_verification(
            credential_valid=False,
            freshness_valid=False,
            schedule_valid=False,
            decoder_records=(
                invalid_decoder_records
            ),
            payload_failures=[
                "P0128",
            ],
            qber_observed=(
                23
                * DEFAULT_STEANE_BLOCK_SIZE
            ),
            physical_qubits_per_check_block=(
                DEFAULT_STEANE_BLOCK_SIZE
            ),
            received_tag=None,
            expected_tag=expected_tag,
            loss_rate=0.20,
        )
    )

    expected_reasons = (
        REASON_INVALID_SERVER_CREDENTIAL,
        REASON_FRESHNESS_OR_REPLAY_FAILURE,
        REASON_INVALID_CONTROL_SCHEDULE,
        REASON_SCHEDULE_BLOCK_MISMATCH,
        REASON_PAYLOAD_BLOCK_UNRECOVERABLE,
        REASON_INSUFFICIENT_CHECK_EVIDENCE,
        REASON_LOSS_RATE_EXCEEDS_POLICY,
        REASON_AUTHENTICATION_TAG_MISMATCH,
    )

    if rejected_output.deterministic_pass:
        raise VerificationEngineError(
            "Invalid deterministic session was accepted."
        )

    if rejected_output.reasons != expected_reasons:
        raise VerificationEngineError(
            "Deterministic reason ordering is incorrect."
        )

    compatible_result = (
        deterministic_protocol_checks(
            credential_valid=True,
            freshness_valid=True,
            schedule_valid=True,
            decoder_records=(
                valid_decoder_records
            ),
            payload_failures=[],
            qber_observed=224,
            physical_qubits_per_check_block=7,
            received_tag=expected_tag,
            expected_tag=expected_tag,
            loss_rate=0.01,
        )
    )

    if compatible_result != (
        True,
        [],
    ):
        raise VerificationEngineError(
            "Notebook-compatible output failed."
        )

    print(
        "Verification engine self-test "
        "completed successfully."
    )


__all__ = [
    "DEFAULT_MIN_OBSERVED_CHECK_BLOCKS",
    "DEFAULT_MAX_ACCEPTABLE_LOSS_RATE",
    "DEFAULT_STEANE_BLOCK_SIZE",
    "REASON_INVALID_SERVER_CREDENTIAL",
    "REASON_FRESHNESS_OR_REPLAY_FAILURE",
    "REASON_INVALID_CONTROL_SCHEDULE",
    "REASON_SCHEDULE_BLOCK_MISMATCH",
    "REASON_PAYLOAD_BLOCK_UNRECOVERABLE",
    "REASON_INSUFFICIENT_CHECK_EVIDENCE",
    "REASON_LOSS_RATE_EXCEEDS_POLICY",
    "REASON_AUTHENTICATION_TAG_MISMATCH",
    "VerificationEngineError",
    "InvalidVerificationInputError",
    "InvalidDecoderRecordError",
    "InvalidVerificationPolicyError",
    "VerificationPolicy",
    "DeterministicVerificationEvidence",
    "VerificationEngineOutput",
    "FTQuPAPVerificationEngine",
    "validate_boolean",
    "validate_nonnegative_integer",
    "validate_loss_rate",
    "normalize_decoder_records",
    "normalize_payload_failures",
    "validate_tag",
    "calculate_observed_check_blocks",
    "find_schedule_mismatches",
    "find_uncorrectable_payload_blocks",
    "verify_authentication_tag",
    "unique_reasons",
    "evaluate_deterministic_verification",
    "deterministic_protocol_checks",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        VerificationEngineError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[VERIFICATION ENGINE ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error