"""
Controlled authentication retry policy for FT-QuPAP v5.1.

FT-QuPAP permits a limited retry when authentication failure may have
been caused by temporary quantum-channel noise, packet loss, insufficient
check-block evidence, or a Gaussian Process gray-zone result.

A retry never reuses the previous attempt's security material. The next
attempt must use:

- A new attempt number
- A fresh Mobile Station nonce
- A fresh Authentication Server nonce
- A new ephemeral ML-KEM key pair
- A new encrypted control schedule
- A newly prepared quantum frame
- A new transcript hash
- A newly generated KMAC authentication tag

Cryptographic failures, replay evidence, stale requests, invalid
signatures, transcript mismatches, and invalid KMAC tags are not
retryable.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from src.common.enums import (
    AuthenticationDecision,
)

from src.common.exceptions import (
    ProtocolValidationError,
)

from src.common.time_utils import (
    current_timestamp,
)

from src.common.validators import (
    validate_integer,
    validate_non_empty_string,
    validate_probability,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MAXIMUM_RETRIES = 1

MAXIMUM_SUPPORTED_RETRIES = 3


RETRY_REASON_GP_GRAY_ZONE = (
    "gp_probability_in_retry_gray_zone"
)

RETRY_REASON_GP_UNCERTAINTY = (
    "gp_uncertainty_high"
)

RETRY_REASON_INSUFFICIENT_CHECKS = (
    "insufficient_check_blocks"
)

RETRY_REASON_EXCESSIVE_LOSS = (
    "excessive_loss"
)

RETRY_REASON_EXCESSIVE_QBER = (
    "excessive_qber"
)

RETRY_REASON_TRANSIENT_NOISE = (
    "transient_quantum_noise"
)

RETRY_REASON_CORRECTION_FAILURE = (
    "uncorrectable_quantum_error"
)

RETRY_REASON_PAYLOAD_RECOVERY = (
    "payload_recovery_failed"
)

RETRY_REASON_CHANNEL_TIMEOUT = (
    "channel_timeout"
)


RESULT_REASON_ACCEPTED_INITIAL = (
    "accepted_without_retry"
)

RESULT_REASON_ACCEPTED_AFTER_RETRY = (
    "accepted_after_retry"
)

RESULT_REASON_RETRY_AUTHORIZED = (
    "retry_authorized"
)

RESULT_REASON_RETRY_LIMIT_REACHED = (
    "retry_limit_reached"
)

RESULT_REASON_NON_RETRYABLE_FAILURE = (
    "non_retryable_failure"
)

RESULT_REASON_REJECTED_BY_POLICY = (
    "rejected_by_retry_policy"
)


RETRYABLE_REASONS = frozenset(
    {
        RETRY_REASON_GP_GRAY_ZONE,
        RETRY_REASON_GP_UNCERTAINTY,
        RETRY_REASON_INSUFFICIENT_CHECKS,
        RETRY_REASON_EXCESSIVE_LOSS,
        RETRY_REASON_EXCESSIVE_QBER,
        RETRY_REASON_TRANSIENT_NOISE,
        RETRY_REASON_CORRECTION_FAILURE,
        RETRY_REASON_PAYLOAD_RECOVERY,
        RETRY_REASON_CHANNEL_TIMEOUT,
    }
)


NON_RETRYABLE_REASONS = frozenset(
    {
        "replay_detected",
        "nonce_reuse",
        "session_attempt_reuse",
        "message_reuse",
        "stale_timestamp",
        "future_timestamp",
        "unknown_pseudonym",
        "historical_pseudonym",
        "subscriber_inactive",
        "mldsa_signature_invalid",
        "server_signature_invalid",
        "transcript_mismatch",
        "kmac_tag_mismatch",
        "ciphertext_binding_invalid",
        "control_schedule_authentication_failed",
        "control_schedule_invalid",
        "mlkem_decapsulation_failed",
        "malformed_request",
        "invalid_frame_structure",
        "protocol_version_mismatch",
    }
)


RETRY_SECURITY_DIRECTIVES = (
    "increment_attempt_number",
    "generate_fresh_mobile_nonce",
    "generate_fresh_server_nonce",
    "generate_fresh_mlkem_keypair",
    "generate_new_control_schedule",
    "generate_new_check_blocks",
    "prepare_new_quantum_frame",
    "derive_new_session_keys",
    "generate_new_kmac_tag",
    "build_new_transcript",
)


# ---------------------------------------------------------------------
# Local exception
# ---------------------------------------------------------------------

class RetryPolicyError(RuntimeError):
    """Raised when retry-policy state is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.details = (
            {}
            if details is None
            else dict(details)
        )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RetryPolicyResult:
    """
    Result of the FT-QuPAP retry-policy evaluation.

    retry_attempts:
        Number of retries already used. Attempt one means zero retries;
        attempt two means one retry has been used.

    retry_used:
        True when the current result came from a retry attempt.

    retry_allowed:
        True when the protocol should start another fresh attempt.
    """

    session_id: str

    current_attempt_number: int
    next_attempt_number: int | None

    maximum_retries: int
    retry_attempts: int

    retry_used: bool
    retry_allowed: bool

    received_decision: AuthenticationDecision
    final_decision: AuthenticationDecision

    trigger_reason: str
    result_reason: str

    retryable_failure: bool
    retry_limit_reached: bool

    evaluated_at: int

    security_directives: tuple[str, ...]

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_non_empty_string(
            self.session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=256,
        )

        validate_integer(
            self.current_attempt_number,
            field_name="current_attempt_number",
            minimum=1,
            maximum=100,
        )

        if self.next_attempt_number is not None:
            validate_integer(
                self.next_attempt_number,
                field_name="next_attempt_number",
                minimum=2,
                maximum=101,
            )

        validate_integer(
            self.maximum_retries,
            field_name="maximum_retries",
            minimum=0,
            maximum=MAXIMUM_SUPPORTED_RETRIES,
        )

        validate_integer(
            self.retry_attempts,
            field_name="retry_attempts",
            minimum=0,
            maximum=MAXIMUM_SUPPORTED_RETRIES,
        )

        for field_name in (
            "retry_used",
            "retry_allowed",
            "retryable_failure",
            "retry_limit_reached",
        ):
            if not isinstance(
                getattr(
                    self,
                    field_name,
                ),
                bool,
            ):
                raise ProtocolValidationError(
                    f"{field_name} must be Boolean."
                )

        if not isinstance(
            self.received_decision,
            AuthenticationDecision,
        ):
            raise ProtocolValidationError(
                (
                    "received_decision must be an "
                    "AuthenticationDecision value."
                )
            )

        if not isinstance(
            self.final_decision,
            AuthenticationDecision,
        ):
            raise ProtocolValidationError(
                (
                    "final_decision must be an "
                    "AuthenticationDecision value."
                )
            )

        validate_non_empty_string(
            self.trigger_reason,
            field_name="trigger_reason",
            minimum_length=1,
            maximum_length=256,
        )

        validate_non_empty_string(
            self.result_reason,
            field_name="result_reason",
            minimum_length=1,
            maximum_length=256,
        )

        validate_integer(
            self.evaluated_at,
            field_name="evaluated_at",
            minimum=0,
        )

        if not isinstance(
            self.security_directives,
            tuple,
        ):
            raise ProtocolValidationError(
                "security_directives must be a tuple."
            )

        if not isinstance(
            self.details,
            dict,
        ):
            raise ProtocolValidationError(
                "details must be a dictionary."
            )

        expected_retry_attempts = (
            self.current_attempt_number - 1
        )

        if (
            self.retry_attempts
            != expected_retry_attempts
        ):
            raise ProtocolValidationError(
                (
                    "retry_attempts must equal "
                    "current_attempt_number - 1."
                )
            )

        if (
            self.retry_used
            != (
                self.current_attempt_number > 1
            )
        ):
            raise ProtocolValidationError(
                (
                    "retry_used does not match the "
                    "current attempt number."
                )
            )

        if self.retry_allowed:
            if (
                self.final_decision
                != AuthenticationDecision.RETRY
            ):
                raise ProtocolValidationError(
                    (
                        "An authorized retry must return "
                        "the RETRY decision."
                    )
                )

            if self.next_attempt_number is None:
                raise ProtocolValidationError(
                    (
                        "An authorized retry must provide "
                        "the next attempt number."
                    )
                )

            if (
                self.next_attempt_number
                != self.current_attempt_number + 1
            ):
                raise ProtocolValidationError(
                    (
                        "next_attempt_number must increment "
                        "the current attempt by one."
                    )
                )

            if not self.security_directives:
                raise ProtocolValidationError(
                    (
                        "An authorized retry must include "
                        "security directives."
                    )
                )

        else:
            if self.next_attempt_number is not None:
                raise ProtocolValidationError(
                    (
                        "A final ACCEPT or REJECT result cannot "
                        "contain a next attempt number."
                    )
                )

            if self.security_directives:
                raise ProtocolValidationError(
                    (
                        "Security directives are allowed only "
                        "when retry is authorized."
                    )
                )

    @property
    def accepted(self) -> bool:
        """Return True when authentication is finally accepted."""

        return (
            self.final_decision
            == AuthenticationDecision.ACCEPT
        )

    @property
    def rejected(self) -> bool:
        """Return True when authentication is finally rejected."""

        return (
            self.final_decision
            == AuthenticationDecision.REJECT
        )

    @property
    def retry_required(self) -> bool:
        """Return True when another fresh attempt must begin."""

        return (
            self.final_decision
            == AuthenticationDecision.RETRY
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        result = asdict(
            self
        )

        result["received_decision"] = (
            self.received_decision.value
        )

        result["final_decision"] = (
            self.final_decision.value
        )

        result["accepted"] = self.accepted
        result["rejected"] = self.rejected

        result["retry_required"] = (
            self.retry_required
        )

        result["security_directives"] = list(
            self.security_directives
        )

        return result


@dataclass
class RetrySessionState:
    """
    Stateful retry information for one authentication session.
    """

    session_id: str
    current_attempt_number: int = 1

    retry_attempts: int = 0
    retry_used: bool = False

    completed: bool = False
    final_decision: AuthenticationDecision | None = None

    last_result_reason: str | None = None

    created_at: int = 0
    updated_at: int = 0

    def __post_init__(self) -> None:
        self.session_id = (
            validate_non_empty_string(
                self.session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=256,
            )
        )

        self.current_attempt_number = (
            validate_integer(
                self.current_attempt_number,
                field_name="current_attempt_number",
                minimum=1,
                maximum=100,
            )
        )

        self.retry_attempts = validate_integer(
            self.retry_attempts,
            field_name="retry_attempts",
            minimum=0,
            maximum=MAXIMUM_SUPPORTED_RETRIES,
        )

        if not isinstance(
            self.retry_used,
            bool,
        ):
            raise ProtocolValidationError(
                "retry_used must be Boolean."
            )

        if not isinstance(
            self.completed,
            bool,
        ):
            raise ProtocolValidationError(
                "completed must be Boolean."
            )

        if (
            self.final_decision is not None
            and not isinstance(
                self.final_decision,
                AuthenticationDecision,
            )
        ):
            raise ProtocolValidationError(
                (
                    "final_decision must be an "
                    "AuthenticationDecision or None."
                )
            )

        self.created_at = validate_integer(
            self.created_at,
            field_name="created_at",
            minimum=0,
        )

        self.updated_at = validate_integer(
            self.updated_at,
            field_name="updated_at",
            minimum=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible session-state dictionary."""

        return {
            "session_id": self.session_id,
            "current_attempt_number": (
                self.current_attempt_number
            ),
            "retry_attempts": (
                self.retry_attempts
            ),
            "retry_used": self.retry_used,
            "completed": self.completed,
            "final_decision": (
                None
                if self.final_decision is None
                else self.final_decision.value
            ),
            "last_result_reason": (
                self.last_result_reason
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def normalize_authentication_decision(
    decision: AuthenticationDecision | str,
) -> AuthenticationDecision:
    """
    Normalize an authentication decision.
    """

    if isinstance(
        decision,
        AuthenticationDecision,
    ):
        return decision

    validated = validate_non_empty_string(
        decision,
        field_name="decision",
        minimum_length=1,
        maximum_length=32,
    ).strip().upper()

    for candidate in AuthenticationDecision:
        if (
            candidate.name.upper()
            == validated
            or str(candidate.value).upper()
            == validated
        ):
            return candidate

    raise ProtocolValidationError(
        f"Unsupported authentication decision: {decision}"
    )


def normalize_retry_reason(
    reason: str,
) -> str:
    """
    Normalize a retry trigger or failure reason.
    """

    validated = validate_non_empty_string(
        reason,
        field_name="reason",
        minimum_length=1,
        maximum_length=256,
    )

    return (
        validated
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def is_retryable_reason(
    reason: str,
) -> bool:
    """
    Return True for an approved transient failure reason.
    """

    normalized = normalize_retry_reason(
        reason
    )

    if normalized in NON_RETRYABLE_REASONS:
        return False

    return normalized in RETRYABLE_REASONS


def is_non_retryable_reason(
    reason: str,
) -> bool:
    """
    Return True for a security-critical final rejection reason.
    """

    return (
        normalize_retry_reason(
            reason
        )
        in NON_RETRYABLE_REASONS
    )


# ---------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------

def evaluate_retry_policy(
    *,
    session_id: str,
    current_attempt_number: int,
    decision: AuthenticationDecision | str,
    reason: str,
    maximum_retries: int = DEFAULT_MAXIMUM_RETRIES,
    evaluated_at: int | None = None,
    attack_probability: float | None = None,
    uncertainty: float | None = None,
    qber_raw: float | None = None,
    loss_rate: float | None = None,
    details: Mapping[str, Any] | None = None,
) -> RetryPolicyResult:
    """
    Evaluate the FT-QuPAP controlled retry policy.

    ACCEPT:
        Final acceptance. Attempt number determines whether this was
        accepted initially or accepted after retry.

    RETRY:
        Another attempt is authorized only when the trigger reason is
        retryable and retry capacity remains.

    REJECT:
        Retry may still be authorized for a specifically approved
        transient reason. Security-critical reasons remain final.
    """

    validated_session_id = (
        validate_non_empty_string(
            session_id,
            field_name="session_id",
            minimum_length=3,
            maximum_length=256,
        )
    )

    validated_attempt = validate_integer(
        current_attempt_number,
        field_name="current_attempt_number",
        minimum=1,
        maximum=100,
    )

    validated_maximum_retries = (
        validate_integer(
            maximum_retries,
            field_name="maximum_retries",
            minimum=0,
            maximum=MAXIMUM_SUPPORTED_RETRIES,
        )
    )

    selected_decision = (
        normalize_authentication_decision(
            decision
        )
    )

    normalized_reason = normalize_retry_reason(
        reason
    )

    selected_timestamp = (
        current_timestamp()
        if evaluated_at is None
        else validate_integer(
            evaluated_at,
            field_name="evaluated_at",
            minimum=0,
        )
    )

    normalized_details = (
        {}
        if details is None
        else dict(details)
    )

    optional_metrics = {
        "attack_probability": attack_probability,
        "uncertainty": uncertainty,
        "qber_raw": qber_raw,
        "loss_rate": loss_rate,
    }

    for field_name, value in optional_metrics.items():
        if value is not None:
            normalized_details[
                field_name
            ] = validate_probability(
                value,
                field_name=field_name,
            )

    retry_attempts = (
        validated_attempt - 1
    )

    retry_used = (
        retry_attempts > 0
    )

    retry_capacity_remaining = (
        retry_attempts
        < validated_maximum_retries
    )

    retryable_failure = (
        is_retryable_reason(
            normalized_reason
        )
    )

    non_retryable_failure = (
        is_non_retryable_reason(
            normalized_reason
        )
    )

    normalized_details.update(
        {
            "retry_capacity_remaining": (
                retry_capacity_remaining
            ),
            "non_retryable_failure": (
                non_retryable_failure
            ),
        }
    )

    if (
        selected_decision
        == AuthenticationDecision.ACCEPT
    ):
        result_reason = (
            RESULT_REASON_ACCEPTED_AFTER_RETRY
            if retry_used
            else RESULT_REASON_ACCEPTED_INITIAL
        )

        return RetryPolicyResult(
            session_id=validated_session_id,
            current_attempt_number=(
                validated_attempt
            ),
            next_attempt_number=None,
            maximum_retries=(
                validated_maximum_retries
            ),
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            retry_allowed=False,
            received_decision=(
                selected_decision
            ),
            final_decision=(
                AuthenticationDecision.ACCEPT
            ),
            trigger_reason=normalized_reason,
            result_reason=result_reason,
            retryable_failure=False,
            retry_limit_reached=False,
            evaluated_at=selected_timestamp,
            security_directives=(),
            details=normalized_details,
        )

    if non_retryable_failure:
        return RetryPolicyResult(
            session_id=validated_session_id,
            current_attempt_number=(
                validated_attempt
            ),
            next_attempt_number=None,
            maximum_retries=(
                validated_maximum_retries
            ),
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            retry_allowed=False,
            received_decision=(
                selected_decision
            ),
            final_decision=(
                AuthenticationDecision.REJECT
            ),
            trigger_reason=normalized_reason,
            result_reason=(
                RESULT_REASON_NON_RETRYABLE_FAILURE
            ),
            retryable_failure=False,
            retry_limit_reached=False,
            evaluated_at=selected_timestamp,
            security_directives=(),
            details=normalized_details,
        )

    if (
        retryable_failure
        and retry_capacity_remaining
    ):
        return RetryPolicyResult(
            session_id=validated_session_id,
            current_attempt_number=(
                validated_attempt
            ),
            next_attempt_number=(
                validated_attempt + 1
            ),
            maximum_retries=(
                validated_maximum_retries
            ),
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            retry_allowed=True,
            received_decision=(
                selected_decision
            ),
            final_decision=(
                AuthenticationDecision.RETRY
            ),
            trigger_reason=normalized_reason,
            result_reason=(
                RESULT_REASON_RETRY_AUTHORIZED
            ),
            retryable_failure=True,
            retry_limit_reached=False,
            evaluated_at=selected_timestamp,
            security_directives=(
                RETRY_SECURITY_DIRECTIVES
            ),
            details=normalized_details,
        )

    if (
        retryable_failure
        and not retry_capacity_remaining
    ):
        return RetryPolicyResult(
            session_id=validated_session_id,
            current_attempt_number=(
                validated_attempt
            ),
            next_attempt_number=None,
            maximum_retries=(
                validated_maximum_retries
            ),
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            retry_allowed=False,
            received_decision=(
                selected_decision
            ),
            final_decision=(
                AuthenticationDecision.REJECT
            ),
            trigger_reason=normalized_reason,
            result_reason=(
                RESULT_REASON_RETRY_LIMIT_REACHED
            ),
            retryable_failure=True,
            retry_limit_reached=True,
            evaluated_at=selected_timestamp,
            security_directives=(),
            details=normalized_details,
        )

    return RetryPolicyResult(
        session_id=validated_session_id,
        current_attempt_number=(
            validated_attempt
        ),
        next_attempt_number=None,
        maximum_retries=(
            validated_maximum_retries
        ),
        retry_attempts=retry_attempts,
        retry_used=retry_used,
        retry_allowed=False,
        received_decision=selected_decision,
        final_decision=(
            AuthenticationDecision.REJECT
        ),
        trigger_reason=normalized_reason,
        result_reason=(
            RESULT_REASON_REJECTED_BY_POLICY
        ),
        retryable_failure=False,
        retry_limit_reached=False,
        evaluated_at=selected_timestamp,
        security_directives=(),
        details=normalized_details,
    )


def require_retry_authorized(
    result: RetryPolicyResult,
) -> int:
    """
    Require an authorized retry and return its next attempt number.
    """

    if not isinstance(
        result,
        RetryPolicyResult,
    ):
        raise ProtocolValidationError(
            (
                "result must be a "
                "RetryPolicyResult object."
            )
        )

    if not result.retry_allowed:
        raise RetryPolicyError(
            "Another authentication attempt is not authorized.",
            details=result.to_dict(),
        )

    assert result.next_attempt_number is not None

    return result.next_attempt_number


# ---------------------------------------------------------------------
# Stateful retry tracker
# ---------------------------------------------------------------------

class RetryPolicyManager:
    """
    Thread-safe session retry-state manager.
    """

    def __init__(
        self,
        *,
        maximum_retries: int = (
            DEFAULT_MAXIMUM_RETRIES
        ),
    ) -> None:
        self.maximum_retries = (
            validate_integer(
                maximum_retries,
                field_name="maximum_retries",
                minimum=0,
                maximum=MAXIMUM_SUPPORTED_RETRIES,
            )
        )

        self._sessions: dict[
            str,
            RetrySessionState,
        ] = {}

        self._lock = threading.RLock()

    def begin_session(
        self,
        session_id: str,
        *,
        created_at: int | None = None,
    ) -> RetrySessionState:
        """
        Create retry state for a new authentication session.
        """

        validated_session_id = (
            validate_non_empty_string(
                session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=256,
            )
        )

        selected_timestamp = (
            current_timestamp()
            if created_at is None
            else validate_integer(
                created_at,
                field_name="created_at",
                minimum=0,
            )
        )

        with self._lock:
            if validated_session_id in self._sessions:
                raise RetryPolicyError(
                    (
                        "Retry state already exists "
                        "for this session."
                    ),
                    details={
                        "session_id": (
                            validated_session_id
                        ),
                    },
                )

            state = RetrySessionState(
                session_id=validated_session_id,
                current_attempt_number=1,
                retry_attempts=0,
                retry_used=False,
                completed=False,
                final_decision=None,
                created_at=selected_timestamp,
                updated_at=selected_timestamp,
            )

            self._sessions[
                validated_session_id
            ] = state

            return state

    def get_session(
        self,
        session_id: str,
    ) -> RetrySessionState:
        """Return retry state for one session."""

        validated_session_id = (
            validate_non_empty_string(
                session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=256,
            )
        )

        with self._lock:
            state = self._sessions.get(
                validated_session_id
            )

            if state is None:
                raise RetryPolicyError(
                    "Retry session state was not found.",
                    details={
                        "session_id": (
                            validated_session_id
                        ),
                    },
                )

            return state

    def evaluate(
        self,
        *,
        session_id: str,
        decision: AuthenticationDecision | str,
        reason: str,
        evaluated_at: int | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> RetryPolicyResult:
        """
        Evaluate and update one session's retry state.
        """

        selected_timestamp = (
            current_timestamp()
            if evaluated_at is None
            else validate_integer(
                evaluated_at,
                field_name="evaluated_at",
                minimum=0,
            )
        )

        with self._lock:
            state = self.get_session(
                session_id
            )

            if state.completed:
                raise RetryPolicyError(
                    (
                        "Retry policy cannot evaluate "
                        "a completed session."
                    ),
                    details=state.to_dict(),
                )

            result = evaluate_retry_policy(
                session_id=state.session_id,
                current_attempt_number=(
                    state.current_attempt_number
                ),
                decision=decision,
                reason=reason,
                maximum_retries=(
                    self.maximum_retries
                ),
                evaluated_at=selected_timestamp,
                details=details,
            )

            state.last_result_reason = (
                result.result_reason
            )

            state.updated_at = (
                selected_timestamp
            )

            if result.retry_allowed:
                assert (
                    result.next_attempt_number
                    is not None
                )

                state.current_attempt_number = (
                    result.next_attempt_number
                )

                state.retry_attempts = (
                    result.retry_attempts + 1
                )

                state.retry_used = True

            else:
                state.completed = True

                state.final_decision = (
                    result.final_decision
                )

                state.retry_attempts = (
                    result.retry_attempts
                )

                state.retry_used = (
                    result.retry_used
                )

            return result

    def remove_session(
        self,
        session_id: str,
    ) -> bool:
        """Delete one session's retry state."""

        validated_session_id = (
            validate_non_empty_string(
                session_id,
                field_name="session_id",
                minimum_length=3,
                maximum_length=256,
            )
        )

        with self._lock:
            return (
                self._sessions.pop(
                    validated_session_id,
                    None,
                )
                is not None
            )

    def list_sessions(
        self,
    ) -> list[dict[str, Any]]:
        """Return all retry-session states."""

        with self._lock:
            return [
                state.to_dict()
                for state in sorted(
                    self._sessions.values(),
                    key=lambda value: (
                        value.session_id
                    ),
                )
            ]


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

def run_retry_policy_self_test() -> dict[str, Any]:
    """
    Test initial acceptance, retry authorization, retry success,
    retry-limit rejection, and non-retryable failure rejection.
    """

    initial_accept = evaluate_retry_policy(
        session_id="FTQ-RETRY-INITIAL",
        current_attempt_number=1,
        decision=AuthenticationDecision.ACCEPT,
        reason="authentication_success",
        maximum_retries=1,
        evaluated_at=1_700_000_000,
    )

    gray_zone_retry = evaluate_retry_policy(
        session_id="FTQ-RETRY-GRAY",
        current_attempt_number=1,
        decision=AuthenticationDecision.RETRY,
        reason=RETRY_REASON_GP_GRAY_ZONE,
        maximum_retries=1,
        evaluated_at=1_700_000_001,
        attack_probability=0.17,
        uncertainty=0.65,
    )

    accepted_after_retry = (
        evaluate_retry_policy(
            session_id="FTQ-RETRY-GRAY",
            current_attempt_number=2,
            decision=AuthenticationDecision.ACCEPT,
            reason="authentication_success",
            maximum_retries=1,
            evaluated_at=1_700_000_002,
        )
    )

    retry_limit_result = (
        evaluate_retry_policy(
            session_id="FTQ-RETRY-LIMIT",
            current_attempt_number=2,
            decision=AuthenticationDecision.RETRY,
            reason=RETRY_REASON_EXCESSIVE_LOSS,
            maximum_retries=1,
            evaluated_at=1_700_000_003,
            loss_rate=0.30,
        )
    )

    replay_result = evaluate_retry_policy(
        session_id="FTQ-RETRY-REPLAY",
        current_attempt_number=1,
        decision=AuthenticationDecision.REJECT,
        reason="replay_detected",
        maximum_retries=1,
        evaluated_at=1_700_000_004,
    )

    manager = RetryPolicyManager(
        maximum_retries=1
    )

    manager.begin_session(
        "FTQ-RETRY-MANAGER",
        created_at=1_700_000_010,
    )

    managed_retry = manager.evaluate(
        session_id="FTQ-RETRY-MANAGER",
        decision=AuthenticationDecision.RETRY,
        reason=RETRY_REASON_TRANSIENT_NOISE,
        evaluated_at=1_700_000_011,
    )

    managed_accept = manager.evaluate(
        session_id="FTQ-RETRY-MANAGER",
        decision=AuthenticationDecision.ACCEPT,
        reason="authentication_success",
        evaluated_at=1_700_000_012,
    )

    managed_state = manager.get_session(
        "FTQ-RETRY-MANAGER"
    )

    success = all(
        (
            initial_accept.accepted,
            not initial_accept.retry_used,
            initial_accept.retry_attempts == 0,
            initial_accept.result_reason
            == RESULT_REASON_ACCEPTED_INITIAL,

            gray_zone_retry.retry_allowed,
            gray_zone_retry.retry_required,
            gray_zone_retry.next_attempt_number == 2,

            accepted_after_retry.accepted,
            accepted_after_retry.retry_used,
            accepted_after_retry.retry_attempts == 1,
            accepted_after_retry.result_reason
            == RESULT_REASON_ACCEPTED_AFTER_RETRY,

            retry_limit_result.rejected,
            retry_limit_result.retry_limit_reached,

            replay_result.rejected,
            not replay_result.retry_allowed,
            replay_result.result_reason
            == RESULT_REASON_NON_RETRYABLE_FAILURE,

            managed_retry.retry_allowed,
            managed_accept.accepted,
            managed_state.completed,
            managed_state.retry_used,
            managed_state.retry_attempts == 1,
        )
    )

    return {
        "success": success,

        "initial_accepted": (
            initial_accept.accepted
        ),

        "initial_retry_used": (
            initial_accept.retry_used
        ),

        "retry_authorized": (
            gray_zone_retry.retry_allowed
        ),

        "next_attempt_number": (
            gray_zone_retry
            .next_attempt_number
        ),

        "accepted_after_retry": (
            accepted_after_retry.accepted
        ),

        "retry_used": (
            accepted_after_retry.retry_used
        ),

        "retry_attempts": (
            accepted_after_retry.retry_attempts
        ),

        "accepted_after_retry_reason": (
            accepted_after_retry
            .result_reason
        ),

        "retry_limit_rejected": (
            retry_limit_result.rejected
        ),

        "replay_retry_rejected": (
            replay_result.rejected
        ),

        "managed_session_completed": (
            managed_state.completed
        ),

        "managed_final_decision": (
            None
            if managed_state.final_decision
            is None
            else managed_state
            .final_decision
            .value
        ),
    }


__all__ = [
    "DEFAULT_MAXIMUM_RETRIES",
    "MAXIMUM_SUPPORTED_RETRIES",
    "RETRYABLE_REASONS",
    "NON_RETRYABLE_REASONS",
    "RETRY_SECURITY_DIRECTIVES",
    "RETRY_REASON_GP_GRAY_ZONE",
    "RETRY_REASON_GP_UNCERTAINTY",
    "RETRY_REASON_INSUFFICIENT_CHECKS",
    "RETRY_REASON_EXCESSIVE_LOSS",
    "RETRY_REASON_EXCESSIVE_QBER",
    "RETRY_REASON_TRANSIENT_NOISE",
    "RETRY_REASON_CORRECTION_FAILURE",
    "RETRY_REASON_PAYLOAD_RECOVERY",
    "RETRY_REASON_CHANNEL_TIMEOUT",
    "RESULT_REASON_ACCEPTED_INITIAL",
    "RESULT_REASON_ACCEPTED_AFTER_RETRY",
    "RESULT_REASON_RETRY_AUTHORIZED",
    "RESULT_REASON_RETRY_LIMIT_REACHED",
    "RESULT_REASON_NON_RETRYABLE_FAILURE",
    "RetryPolicyError",
    "RetryPolicyResult",
    "RetrySessionState",
    "normalize_authentication_decision",
    "normalize_retry_reason",
    "is_retryable_reason",
    "is_non_retryable_reason",
    "evaluate_retry_policy",
    "require_retry_authorized",
    "RetryPolicyManager",
    "run_retry_policy_self_test",
]