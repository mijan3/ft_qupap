"""
FT-QuPAP Protocol Engine

This module coordinates complete FT-QuPAP authentication sessions.

The protocol engine does not directly implement cryptographic,
quantum, or machine-learning algorithms. Those operations remain in
their dedicated packages:

    src/cryptography/
    src/mobile_station/
    src/authentication_server/
    src/quantum/
    src/machine_learning/

The engine coordinates these operations through an injected attempt
executor.

One authentication attempt must perform the following protocol flow:

    1. Create or receive a fresh authentication request.
    2. Validate pseudonymous subscriber identity.
    3. Validate timestamp freshness and nonce replay status.
    4. Generate an ephemeral ML-KEM key pair.
    5. Build and ML-DSA-sign the server package.
    6. Verify the server signature at the mobile station.
    7. Perform ML-KEM encapsulation and decapsulation.
    8. Build and hash the authenticated transcript.
    9. Derive separated K_auth and K_ctrl session keys.
    10. Generate the transcript-bound KMAC authentication tag.
    11. Generate payload and independent check blocks.
    12. Interleave and Steane-encode all logical blocks.
    13. Encrypt the control schedule using K_ctrl.
    14. Transmit the frame through the configured quantum channel.
    15. Decrypt and validate the control schedule.
    16. Measure declared check blocks and calculate raw QBER.
    17. Extract syndrome, correct recoverable errors, and decode.
    18. Recover and verify the received authentication tag.
    19. Execute deterministic verification.
    20. Extract receiver-observable GP features.
    21. Calculate calibrated attack probability and uncertainty.
    22. Produce the final authentication decision.

Retry requirements:

    - Retries are bounded.
    - A retry is allowed only when the retry policy approves it.
    - Every retry must use a fresh nonce.
    - Every retry must generate a fresh ephemeral ML-KEM key pair.
    - Every retry must derive new K_auth and K_ctrl keys.
    - Strong deterministic or attack evidence remains fail-closed.
"""

from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from .protocol_state import ProtocolState
from .result_models import (
    AuthenticationResult,
    DecisionResult,
    RetryAttemptResult,
    build_rejected_result,
)


DEFAULT_MAX_AUTHENTICATION_ATTEMPTS = 3

ENGINE_STAGE_CREATED = "created"
ENGINE_STAGE_RUNNING = "running"
ENGINE_STAGE_ATTEMPT_COMPLETE = "attempt_complete"
ENGINE_STAGE_RETRY_APPROVED = "retry_approved"
ENGINE_STAGE_ACCEPTED = "accepted"
ENGINE_STAGE_REJECTED = "rejected"
ENGINE_STAGE_FAILED = "failed"

ACCEPTED_AFTER_RETRY_REASON = "accepted_after_retry"
PROTOCOL_ENGINE_FAILURE_REASON = "protocol_engine_failure"


class ProtocolEngineError(Exception):
    """Base exception for protocol-engine failures."""


class InvalidProtocolExecutorError(ProtocolEngineError):
    """Raised when the configured attempt executor is invalid."""


class InvalidRetryConfigurationError(ProtocolEngineError):
    """Raised when retry configuration is invalid."""


class InvalidProtocolResultError(ProtocolEngineError):
    """Raised when an attempt returns an invalid result."""


class ProtocolAttemptExecutor(Protocol):
    """
    Interface implemented by the complete one-attempt workflow.

    The executor must generate fresh session material whenever
    context.fresh_session_required is True.
    """

    def __call__(
        self,
        context: "ProtocolExecutionContext",
    ) -> AuthenticationResult:
        """Execute one complete FT-QuPAP authentication attempt."""


class RetryEvaluator(Protocol):
    """Interface used to decide whether another attempt is allowed."""

    def __call__(
        self,
        result: AuthenticationResult,
        attempt_number: int,
        maximum_attempts: int,
    ) -> bool:
        """Return True when a fresh-session retry is permitted."""


class ResultRecorder(Protocol):
    """Interface for optional session-result persistence or logging."""

    def __call__(
        self,
        result: AuthenticationResult,
        state: ProtocolState,
    ) -> None:
        """Record a non-secret protocol result."""


RequestFactory = Callable[[int], Any]


@dataclass
class ProtocolEngineConfig:
    """
    Configuration for the high-level protocol engine.
    """

    max_authentication_attempts: int = (
        DEFAULT_MAX_AUTHENTICATION_ATTEMPTS
    )

    retries_enabled: bool = True
    fail_closed_on_exception: bool = True
    accepted_after_retry_reason: str = (
        ACCEPTED_AFTER_RETRY_REASON
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.max_authentication_attempts,
                bool,
            )
            or not isinstance(
                self.max_authentication_attempts,
                int,
            )
        ):
            raise TypeError(
                "max_authentication_attempts must be an integer."
            )

        if self.max_authentication_attempts < 1:
            raise InvalidRetryConfigurationError(
                "max_authentication_attempts must be at least 1."
            )

        if not isinstance(
            self.retries_enabled,
            bool,
        ):
            raise TypeError(
                "retries_enabled must be boolean."
            )

        if not isinstance(
            self.fail_closed_on_exception,
            bool,
        ):
            raise TypeError(
                "fail_closed_on_exception must be boolean."
            )

        if not isinstance(
            self.accepted_after_retry_reason,
            str,
        ) or not self.accepted_after_retry_reason.strip():
            raise ValueError(
                "accepted_after_retry_reason cannot be empty."
            )

        self.accepted_after_retry_reason = (
            self.accepted_after_retry_reason.strip()
        )


@dataclass
class ProtocolExecutionContext:
    """
    Input context for one FT-QuPAP authentication attempt.

    Secrets must not be placed in metadata.
    """

    channel: Any

    request: Any | None = None
    attempt_number: int = 1

    use_css: bool = True
    decision_mode: str = "gp"
    bootstrap_mode: str = "mlkem"

    fresh_session_required: bool = True

    tamper_server_signature: bool = False
    tamper_authentication_request: bool = False
    tamper_mlkem_ciphertext: bool = False
    forge_kmac_tag: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.attempt_number,
                bool,
            )
            or not isinstance(
                self.attempt_number,
                int,
            )
        ):
            raise TypeError(
                "attempt_number must be an integer."
            )

        if self.attempt_number < 1:
            raise ValueError(
                "attempt_number must be at least 1."
            )

        boolean_fields = {
            "use_css":
                self.use_css,
            "fresh_session_required":
                self.fresh_session_required,
            "tamper_server_signature":
                self.tamper_server_signature,
            "tamper_authentication_request":
                self.tamper_authentication_request,
            "tamper_mlkem_ciphertext":
                self.tamper_mlkem_ciphertext,
            "forge_kmac_tag":
                self.forge_kmac_tag,
        }

        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        if self.decision_mode not in {
            "gp",
            "fixed_qber",
        }:
            raise ValueError(
                "decision_mode must be 'gp' or 'fixed_qber'."
            )

        if self.bootstrap_mode not in {
            "mlkem",
            "pre_shared_key",
        }:
            raise ValueError(
                "bootstrap_mode must be 'mlkem' or "
                "'pre_shared_key'."
            )

        if not isinstance(
            self.metadata,
            dict,
        ):
            raise TypeError(
                "metadata must be a dictionary."
            )

        self.metadata = copy.deepcopy(
            self.metadata
        )

    def for_retry(
        self,
        attempt_number: int,
        request: Any | None,
    ) -> "ProtocolExecutionContext":
        """
        Build a detached context for a fresh retry.

        The retry context explicitly requires new session material.
        """

        retry_metadata = copy.deepcopy(
            self.metadata
        )

        retry_metadata.update(
            {
                "retry": True,
                "previous_attempt_number":
                    self.attempt_number,
                "fresh_nonce_required": True,
                "fresh_mlkem_keypair_required": True,
                "fresh_session_keys_required": True,
            }
        )

        return ProtocolExecutionContext(
            channel=self.channel,
            request=request,
            attempt_number=attempt_number,
            use_css=self.use_css,
            decision_mode=self.decision_mode,
            bootstrap_mode=self.bootstrap_mode,
            fresh_session_required=True,
            tamper_server_signature=(
                self.tamper_server_signature
            ),
            tamper_authentication_request=(
                self.tamper_authentication_request
            ),
            tamper_mlkem_ciphertext=(
                self.tamper_mlkem_ciphertext
            ),
            forge_kmac_tag=self.forge_kmac_tag,
            metadata=retry_metadata,
        )


def default_retry_evaluator(
    result: AuthenticationResult,
    attempt_number: int,
    maximum_attempts: int,
) -> bool:
    """
    Apply the default protocol-level retry decision.

    Detailed retry safety checks should be performed by retry_engine.py
    or authentication_server/retry_policy.py. This default evaluator
    follows the retry_recommended field in the final decision.
    """

    if not isinstance(
        result,
        AuthenticationResult,
    ):
        raise TypeError(
            "result must be AuthenticationResult."
        )

    if result.accepted:
        return False

    if attempt_number >= maximum_attempts:
        return False

    return bool(
        result.decision.retry_recommended
    )


def extract_pseudonym_id(
    request: Any | None,
) -> str:
    """
    Extract a pseudonym from a request without accessing raw identity.
    """

    if request is None:
        return "unknown-pseudonym"

    if isinstance(
        request,
        Mapping,
    ):
        value = request.get(
            "pseudonym_id"
        )

    else:
        value = getattr(
            request,
            "pseudonym_id",
            None,
        )

    if isinstance(value, str) and value.strip():
        return value.strip()

    return "unknown-pseudonym"


def extract_channel_property(
    channel: Any,
    property_name: str,
    default: str,
) -> str:
    """
    Extract a public channel property from an object or mapping.
    """

    if isinstance(
        channel,
        Mapping,
    ):
        value = channel.get(
            property_name,
            default,
        )

    else:
        value = getattr(
            channel,
            property_name,
            default,
        )

    if value is None:
        return default

    normalized = str(value).strip()

    return normalized or default


def make_engine_failure_result(
    context: ProtocolExecutionContext,
    error: Exception,
) -> AuthenticationResult:
    """
    Build a fail-closed result after an unexpected execution failure.
    """

    session_id = (
        f"FAILED-{uuid.uuid4().hex}"
    )

    result = build_rejected_result(
        session_id=session_id,
        pseudonym_id=extract_pseudonym_id(
            context.request
        ),
        reason=PROTOCOL_ENGINE_FAILURE_REASON,
        deterministic_reasons=(
            PROTOCOL_ENGINE_FAILURE_REASON,
        ),
        channel_name=extract_channel_property(
            context.channel,
            "name",
            "unknown",
        ),
        channel_context=extract_channel_property(
            context.channel,
            "context",
            "unknown",
        ),
    )

    result.diagnostics.update(
        {
            "engine_failure": True,
            "error_type":
                type(error).__name__,
            "attempt_number":
                context.attempt_number,
        }
    )

    return result


def build_attempt_history_record(
    result: AuthenticationResult,
    attempt_number: int,
    retryable: bool,
    retry_reason: str | None,
) -> RetryAttemptResult:
    """
    Build the compact non-secret history entry for one attempt.
    """

    return RetryAttemptResult(
        attempt_number=attempt_number,
        session_id=result.session_id,
        accepted=result.accepted,
        reason=result.reason,
        deterministic_pass=(
            result.deterministic_pass
        ),
        qber_raw=result.qber_raw,
        p_attack=result.p_attack,
        loss_rate=result.loss_rate,
        tag_recovered=result.tag_recovered,
        retryable=retryable,
        retry_reason=(
            retry_reason
            if retryable
            else None
        ),
    )


class FTQuPAPProtocolEngine:
    """
    High-level FT-QuPAP authentication coordinator.

    The injected attempt_executor performs one full authentication
    attempt. This engine provides:

        - attempt lifecycle tracking
        - fail-closed exception handling
        - bounded retries
        - fresh request enforcement
        - attempt-history generation
        - total retry timing
        - optional non-secret result recording
    """

    def __init__(
        self,
        attempt_executor: ProtocolAttemptExecutor,
        retry_evaluator: RetryEvaluator | None = None,
        result_recorder: ResultRecorder | None = None,
        config: ProtocolEngineConfig | None = None,
    ) -> None:
        if not callable(
            attempt_executor
        ):
            raise InvalidProtocolExecutorError(
                "attempt_executor must be callable."
            )

        if (
            retry_evaluator is not None
            and not callable(retry_evaluator)
        ):
            raise TypeError(
                "retry_evaluator must be callable or None."
            )

        if (
            result_recorder is not None
            and not callable(result_recorder)
        ):
            raise TypeError(
                "result_recorder must be callable or None."
            )

        self.attempt_executor = (
            attempt_executor
        )

        self.retry_evaluator = (
            retry_evaluator
            or default_retry_evaluator
        )

        self.result_recorder = (
            result_recorder
        )

        self.config = (
            config
            or ProtocolEngineConfig()
        )

        self.state = ProtocolState()

        self.state.set_stage(
            "engine",
            ENGINE_STAGE_CREATED,
        )

    def _execute_attempt(
        self,
        context: ProtocolExecutionContext,
    ) -> AuthenticationResult:
        """
        Execute one attempt and validate its returned result.
        """

        self.state.set_stage(
            "engine",
            ENGINE_STAGE_RUNNING,
        )

        self.state.set_stage(
            f"attempt_{context.attempt_number}",
            ENGINE_STAGE_RUNNING,
        )

        attempt_start = time.perf_counter()

        try:
            result = self.attempt_executor(
                context
            )

        except Exception as error:
            self.state.set_stage(
                f"attempt_{context.attempt_number}",
                ENGINE_STAGE_FAILED,
            )

            if (
                not self.config
                .fail_closed_on_exception
            ):
                raise

            result = make_engine_failure_result(
                context,
                error,
            )

        if not isinstance(
            result,
            AuthenticationResult,
        ):
            raise InvalidProtocolResultError(
                "attempt_executor must return "
                "AuthenticationResult."
            )

        elapsed = (
            time.perf_counter()
            - attempt_start
        )

        result.timings.setdefault(
            "end_to_end_s",
            float(elapsed),
        )

        self.state.set_stage(
            f"attempt_{context.attempt_number}",
            ENGINE_STAGE_ATTEMPT_COMPLETE,
        )

        self.state.evidence.update(
            {
                "current_attempt":
                    context.attempt_number,
                "last_session_id":
                    result.session_id,
                "last_accepted":
                    result.accepted,
                "last_reason":
                    result.reason,
                "last_qber_raw":
                    result.qber_raw,
                "last_p_attack":
                    result.p_attack,
            }
        )

        return result

    def _record_result(
        self,
        result: AuthenticationResult,
    ) -> None:
        """
        Record a result without changing the authentication decision.
        """

        if self.result_recorder is None:
            return

        try:
            self.result_recorder(
                result,
                self.state,
            )

        except Exception as error:
            result.diagnostics[
                "result_recording_failed"
            ] = True

            result.diagnostics[
                "result_recording_error_type"
            ] = type(error).__name__

    def run_session(
        self,
        *,
        channel: Any,
        request: Any | None = None,
        use_css: bool = True,
        decision_mode: str = "gp",
        bootstrap_mode: str = "mlkem",
        tamper_server_signature: bool = False,
        tamper_authentication_request: bool = False,
        tamper_mlkem_ciphertext: bool = False,
        forge_kmac_tag: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuthenticationResult:
        """
        Execute exactly one FT-QuPAP authentication attempt.
        """

        context = ProtocolExecutionContext(
            channel=channel,
            request=copy.deepcopy(
                request
            ),
            attempt_number=1,
            use_css=use_css,
            decision_mode=decision_mode,
            bootstrap_mode=bootstrap_mode,
            fresh_session_required=True,
            tamper_server_signature=(
                tamper_server_signature
            ),
            tamper_authentication_request=(
                tamper_authentication_request
            ),
            tamper_mlkem_ciphertext=(
                tamper_mlkem_ciphertext
            ),
            forge_kmac_tag=forge_kmac_tag,
            metadata=(
                {}
                if metadata is None
                else metadata
            ),
        )

        result = self._execute_attempt(
            context
        )

        result.retry_attempts = 1
        result.attempt_history = [
            build_attempt_history_record(
                result=result,
                attempt_number=1,
                retryable=False,
                retry_reason=None,
            )
        ]

        result.timings.setdefault(
            "total_retry_end_to_end_s",
            result.timings.get(
                "end_to_end_s",
                0.0,
            ),
        )

        final_stage = (
            ENGINE_STAGE_ACCEPTED
            if result.accepted
            else ENGINE_STAGE_REJECTED
        )

        self.state.set_stage(
            "engine",
            final_stage,
        )

        self._record_result(
            result
        )

        return result

    def run_with_retries(
        self,
        *,
        channel: Any,
        request: Any | None = None,
        request_factory: RequestFactory | None = None,
        maximum_attempts: int | None = None,
        use_css: bool = True,
        decision_mode: str = "gp",
        bootstrap_mode: str = "mlkem",
        tamper_server_signature: bool = False,
        tamper_authentication_request: bool = False,
        tamper_mlkem_ciphertext: bool = False,
        forge_kmac_tag: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> AuthenticationResult:
        """
        Execute FT-QuPAP with bounded fresh-session retries.

        For retries, the original request is never reused. A fresh
        request is obtained from request_factory. When no request factory
        is supplied, request=None is passed to the attempt executor so
        that it must generate a new request and nonce.
        """

        max_attempts = (
            self.config.max_authentication_attempts
            if maximum_attempts is None
            else maximum_attempts
        )

        if (
            isinstance(
                max_attempts,
                bool,
            )
            or not isinstance(
                max_attempts,
                int,
            )
        ):
            raise TypeError(
                "maximum_attempts must be an integer."
            )

        if max_attempts < 1:
            raise InvalidRetryConfigurationError(
                "maximum_attempts must be at least 1."
            )

        if (
            request_factory is not None
            and not callable(request_factory)
        ):
            raise TypeError(
                "request_factory must be callable or None."
            )

        initial_context = ProtocolExecutionContext(
            channel=channel,
            request=copy.deepcopy(
                request
            ),
            attempt_number=1,
            use_css=use_css,
            decision_mode=decision_mode,
            bootstrap_mode=bootstrap_mode,
            fresh_session_required=True,
            tamper_server_signature=(
                tamper_server_signature
            ),
            tamper_authentication_request=(
                tamper_authentication_request
            ),
            tamper_mlkem_ciphertext=(
                tamper_mlkem_ciphertext
            ),
            forge_kmac_tag=forge_kmac_tag,
            metadata=(
                {}
                if metadata is None
                else metadata
            ),
        )

        attempts: list[
            AuthenticationResult
        ] = []

        history: list[
            RetryAttemptResult
        ] = []

        current_context = initial_context
        final_result: AuthenticationResult | None = None

        for attempt_number in range(
            1,
            max_attempts + 1,
        ):
            result = self._execute_attempt(
                current_context
            )

            attempts.append(
                result
            )

            retry_allowed = False
            retry_reason: str | None = None

            if (
                self.config.retries_enabled
                and not result.accepted
                and attempt_number < max_attempts
            ):
                retry_allowed = bool(
                    self.retry_evaluator(
                        result,
                        attempt_number,
                        max_attempts,
                    )
                )

                if retry_allowed:
                    retry_reason = (
                        result.reason
                        or "retry_policy_approved"
                    )

            history.append(
                build_attempt_history_record(
                    result=result,
                    attempt_number=attempt_number,
                    retryable=retry_allowed,
                    retry_reason=retry_reason,
                )
            )

            if result.accepted:
                final_result = result
                break

            if not retry_allowed:
                final_result = result
                break

            self.state.set_stage(
                "engine",
                ENGINE_STAGE_RETRY_APPROVED,
            )

            next_attempt_number = (
                attempt_number + 1
            )

            # A retry must never reuse the previous request nonce.
            next_request = (
                request_factory(
                    next_attempt_number
                )
                if request_factory is not None
                else None
            )

            current_context = (
                current_context.for_retry(
                    attempt_number=(
                        next_attempt_number
                    ),
                    request=next_request,
                )
            )

        if final_result is None:
            raise ProtocolEngineError(
                "Protocol execution ended without a final result."
            )

        final_result.retry_attempts = len(
            attempts
        )

        final_result.attempt_history = (
            history
        )

        final_result.timings[
            "total_retry_end_to_end_s"
        ] = float(
            sum(
                attempt.timings.get(
                    "end_to_end_s",
                    0.0,
                )
                for attempt in attempts
            )
        )

        if (
            final_result.accepted
            and len(attempts) > 1
        ):
            updated_decision: DecisionResult = replace(
                final_result.decision,
                reason=(
                    self.config
                    .accepted_after_retry_reason
                ),
            )

            final_result.decision = (
                updated_decision
            )

        final_stage = (
            ENGINE_STAGE_ACCEPTED
            if final_result.accepted
            else ENGINE_STAGE_REJECTED
        )

        self.state.set_stage(
            "engine",
            final_stage,
        )

        self.state.evidence.update(
            {
                "retry_attempts":
                    final_result.retry_attempts,
                "retry_used":
                    final_result.retry_used,
                "final_session_id":
                    final_result.session_id,
                "final_accepted":
                    final_result.accepted,
                "final_reason":
                    final_result.reason,
            }
        )

        self._record_result(
            final_result
        )

        return final_result


def run_ft_qupap_session(
    engine: FTQuPAPProtocolEngine,
    *,
    channel: Any,
    request: Any | None = None,
    **options: Any,
) -> AuthenticationResult:
    """
    Functional wrapper for one FT-QuPAP attempt.
    """

    if not isinstance(
        engine,
        FTQuPAPProtocolEngine,
    ):
        raise TypeError(
            "engine must be FTQuPAPProtocolEngine."
        )

    return engine.run_session(
        channel=channel,
        request=request,
        **options,
    )


def run_ft_qupap_with_retries(
    engine: FTQuPAPProtocolEngine,
    *,
    channel: Any,
    request: Any | None = None,
    request_factory: RequestFactory | None = None,
    maximum_attempts: int | None = None,
    **options: Any,
) -> AuthenticationResult:
    """
    Functional wrapper for bounded FT-QuPAP retries.
    """

    if not isinstance(
        engine,
        FTQuPAPProtocolEngine,
    ):
        raise TypeError(
            "engine must be FTQuPAPProtocolEngine."
        )

    return engine.run_with_retries(
        channel=channel,
        request=request,
        request_factory=request_factory,
        maximum_attempts=maximum_attempts,
        **options,
    )


__all__ = [
    "DEFAULT_MAX_AUTHENTICATION_ATTEMPTS",
    "ENGINE_STAGE_CREATED",
    "ENGINE_STAGE_RUNNING",
    "ENGINE_STAGE_ATTEMPT_COMPLETE",
    "ENGINE_STAGE_RETRY_APPROVED",
    "ENGINE_STAGE_ACCEPTED",
    "ENGINE_STAGE_REJECTED",
    "ENGINE_STAGE_FAILED",
    "ACCEPTED_AFTER_RETRY_REASON",
    "PROTOCOL_ENGINE_FAILURE_REASON",
    "ProtocolEngineError",
    "InvalidProtocolExecutorError",
    "InvalidRetryConfigurationError",
    "InvalidProtocolResultError",
    "ProtocolAttemptExecutor",
    "RetryEvaluator",
    "ResultRecorder",
    "RequestFactory",
    "ProtocolEngineConfig",
    "ProtocolExecutionContext",
    "FTQuPAPProtocolEngine",
    "default_retry_evaluator",
    "extract_pseudonym_id",
    "extract_channel_property",
    "make_engine_failure_result",
    "build_attempt_history_record",
    "run_ft_qupap_session",
    "run_ft_qupap_with_retries",
]