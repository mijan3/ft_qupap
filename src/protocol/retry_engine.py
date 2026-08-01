"""
FT-QuPAP Retry Engine

This module evaluates whether a rejected authentication attempt may be
retried using a completely fresh FT-QuPAP session.

Retry is permitted only for two fail-closed but low-risk situations:

1. Payload or KMAC-tag recovery failure where:
   - deterministic failure reasons contain only:
       * payload_block_unrecoverable
       * authentication_tag_mismatch
   - raw QBER is within the retry limit
   - loss is within the deterministic loss limit
   - sufficient check-block evidence was observed
   - GP attack probability is absent or below the retry limit

2. A deterministic-pass, tag-recovered session rejected by the
   calibrated GP policy, where:
   - the rejection reason is:
       rejected_by_calibrated_bayesian_policy
   - channel observables remain low-risk
   - P(attack) is below the configured GP gray-zone upper bound

The following failures are never retryable:

- invalid ML-DSA server credential
- unknown or inactive subscriber
- timestamp freshness failure
- nonce replay
- invalid control schedule
- schedule/block mismatch
- insufficient check-block evidence
- excessive loss
- high raw QBER
- strong GP attack probability
- maximum authentication-attempt limit reached

Security requirement:

A retry is not a continuation of the previous cryptographic session.
The protocol engine must generate:

- a fresh request nonce
- a fresh timestamp
- a fresh ephemeral ML-KEM key pair
- a fresh ML-KEM shared secret
- a fresh transcript hash
- fresh K_auth and K_ctrl keys
- a fresh KMAC authentication tag
- a fresh check schedule
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from .result_models import (
    AttackDetectionResult,
    AuthenticationResult,
    DecisionResult,
    VerificationResult,
)


DEFAULT_MAX_AUTHENTICATION_ATTEMPTS = 3
DEFAULT_RETRY_PROBABILITY_LIMIT = 0.20
DEFAULT_RETRY_QBER_LIMIT = 0.11
DEFAULT_MAX_ACCEPTABLE_LOSS_RATE = 0.15
DEFAULT_MIN_OBSERVED_CHECK_BLOCKS = 24
DEFAULT_GP_GRAY_ZONE_RETRY_UPPER = 0.20


PAYLOAD_BLOCK_UNRECOVERABLE = (
    "payload_block_unrecoverable"
)

AUTHENTICATION_TAG_MISMATCH = (
    "authentication_tag_mismatch"
)

GP_POLICY_REJECTION_REASON = (
    "rejected_by_calibrated_bayesian_policy"
)


RETRY_KIND_NONE = "none"

RETRY_KIND_PAYLOAD_RECOVERY = (
    "payload_recovery_failure"
)

RETRY_KIND_GP_GRAY_ZONE = (
    "gp_gray_zone_rejection"
)


RETRY_REASON_PAYLOAD_RECOVERY = (
    "retryable_payload_recovery_failure"
)

RETRY_REASON_GP_GRAY_ZONE = (
    "retryable_low_risk_gp_rejection"
)

RETRY_REASON_ALREADY_ACCEPTED = (
    "session_already_accepted"
)

RETRY_REASON_ATTEMPT_LIMIT = (
    "maximum_authentication_attempts_reached"
)

RETRY_REASON_NONRETRYABLE_EVIDENCE = (
    "strong_or_nonretryable_evidence"
)


RETRYABLE_PAYLOAD_FAILURE_REASONS = frozenset(
    {
        PAYLOAD_BLOCK_UNRECOVERABLE,
        AUTHENTICATION_TAG_MISMATCH,
    }
)


class RetryEngineError(Exception):
    """Base exception for FT-QuPAP retry-engine failures."""


class InvalidRetryPolicyError(RetryEngineError):
    """Raised when retry-policy parameters are invalid."""


class InvalidRetrySessionError(RetryEngineError):
    """Raised when session evidence cannot be evaluated."""


class InvalidRetryAttemptError(RetryEngineError):
    """Raised when retry-attempt information is invalid."""


SessionLike: TypeAlias = (
    AuthenticationResult
    | Mapping[str, Any]
)


def validate_probability(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate a finite probability in the interval [0, 1].
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric."
        )

    normalized = float(value)

    if not math.isfinite(normalized):
        raise ValueError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def validate_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """
    Validate an integer greater than zero.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value < 1:
        raise ValueError(
            f"{field_name} must be at least 1."
        )

    return value


def optional_probability(
    value: Any,
) -> float | None:
    """
    Normalize an optional finite probability.

    None and NaN are treated as unavailable GP evidence.
    """

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            "Probability evidence must be numeric or None."
        )

    normalized = float(value)

    if math.isnan(normalized):
        return None

    if not math.isfinite(normalized):
        raise ValueError(
            "Probability evidence must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            "Probability evidence must be between 0 and 1."
        )

    return normalized


@dataclass(frozen=True)
class RetryPolicy:
    """
    Bounded FT-QuPAP reauthentication policy.

    Defaults match the final operational-threshold notebook.
    """

    max_authentication_attempts: int = (
        DEFAULT_MAX_AUTHENTICATION_ATTEMPTS
    )

    retry_probability_limit: float = (
        DEFAULT_RETRY_PROBABILITY_LIMIT
    )

    retry_qber_limit: float = (
        DEFAULT_RETRY_QBER_LIMIT
    )

    max_acceptable_loss_rate: float = (
        DEFAULT_MAX_ACCEPTABLE_LOSS_RATE
    )

    min_observed_check_blocks: int = (
        DEFAULT_MIN_OBSERVED_CHECK_BLOCKS
    )

    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_RETRY_UPPER
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_authentication_attempts",
            validate_positive_integer(
                self.max_authentication_attempts,
                "max_authentication_attempts",
            ),
        )

        object.__setattr__(
            self,
            "retry_probability_limit",
            validate_probability(
                self.retry_probability_limit,
                "retry_probability_limit",
            ),
        )

        object.__setattr__(
            self,
            "retry_qber_limit",
            validate_probability(
                self.retry_qber_limit,
                "retry_qber_limit",
            ),
        )

        object.__setattr__(
            self,
            "max_acceptable_loss_rate",
            validate_probability(
                self.max_acceptable_loss_rate,
                "max_acceptable_loss_rate",
            ),
        )

        object.__setattr__(
            self,
            "min_observed_check_blocks",
            validate_positive_integer(
                self.min_observed_check_blocks,
                "min_observed_check_blocks",
            ),
        )

        object.__setattr__(
            self,
            "gp_gray_zone_retry_upper",
            validate_probability(
                self.gp_gray_zone_retry_upper,
                "gp_gray_zone_retry_upper",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable retry policy."""

        return {
            "max_authentication_attempts":
                self.max_authentication_attempts,
            "retry_probability_limit":
                self.retry_probability_limit,
            "retry_qber_limit":
                self.retry_qber_limit,
            "max_acceptable_loss_rate":
                self.max_acceptable_loss_rate,
            "min_observed_check_blocks":
                self.min_observed_check_blocks,
            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,
        }


DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True)
class RetryAssessment:
    """
    Auditable retry assessment for one failed authentication attempt.
    """

    retry_allowed: bool
    reason: str
    retry_kind: str

    attempt_number: int
    maximum_attempts: int

    accepted: bool
    deterministic_pass: bool
    deterministic_reasons: tuple[str, ...]

    qber_raw: float | None
    loss_rate: float | None
    observed_check_blocks: int

    p_attack: float | None
    tag_recovered: bool
    low_risk_observables: bool

    fresh_session_required: bool

    def __post_init__(self) -> None:
        boolean_fields = {
            "retry_allowed":
                self.retry_allowed,
            "accepted":
                self.accepted,
            "deterministic_pass":
                self.deterministic_pass,
            "tag_recovered":
                self.tag_recovered,
            "low_risk_observables":
                self.low_risk_observables,
            "fresh_session_required":
                self.fresh_session_required,
        }

        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        validate_positive_integer(
            self.attempt_number,
            "attempt_number",
        )

        validate_positive_integer(
            self.maximum_attempts,
            "maximum_attempts",
        )

        if self.attempt_number > self.maximum_attempts:
            raise InvalidRetryAttemptError(
                "attempt_number cannot exceed maximum_attempts."
            )

        if self.retry_kind not in {
            RETRY_KIND_NONE,
            RETRY_KIND_PAYLOAD_RECOVERY,
            RETRY_KIND_GP_GRAY_ZONE,
        }:
            raise ValueError(
                "Unsupported retry_kind."
            )

        if not isinstance(
            self.reason,
            str,
        ) or not self.reason.strip():
            raise ValueError(
                "reason cannot be empty."
            )

        if not isinstance(
            self.deterministic_reasons,
            tuple,
        ):
            raise TypeError(
                "deterministic_reasons must be a tuple."
            )

        if (
            isinstance(
                self.observed_check_blocks,
                bool,
            )
            or not isinstance(
                self.observed_check_blocks,
                int,
            )
        ):
            raise TypeError(
                "observed_check_blocks must be an integer."
            )

        if self.observed_check_blocks < 0:
            raise ValueError(
                "observed_check_blocks cannot be negative."
            )

        if self.qber_raw is not None:
            validate_probability(
                self.qber_raw,
                "qber_raw",
            )

        if self.loss_rate is not None:
            validate_probability(
                self.loss_rate,
                "loss_rate",
            )

        if self.p_attack is not None:
            validate_probability(
                self.p_attack,
                "p_attack",
            )

        if self.retry_allowed:
            if self.accepted:
                raise InvalidRetrySessionError(
                    "An accepted session cannot be retried."
                )

            if (
                self.attempt_number
                >= self.maximum_attempts
            ):
                raise InvalidRetryAttemptError(
                    "Retry cannot be allowed after the "
                    "maximum attempt."
                )

            if self.retry_kind == RETRY_KIND_NONE:
                raise InvalidRetrySessionError(
                    "A permitted retry must include a retry kind."
                )

            if not self.fresh_session_required:
                raise InvalidRetrySessionError(
                    "Every permitted retry must require "
                    "a fresh session."
                )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe retry assessment."""

        return {
            "retry_allowed":
                self.retry_allowed,
            "reason":
                self.reason,
            "retry_kind":
                self.retry_kind,
            "attempt_number":
                self.attempt_number,
            "maximum_attempts":
                self.maximum_attempts,
            "accepted":
                self.accepted,
            "deterministic_pass":
                self.deterministic_pass,
            "deterministic_reasons":
                list(
                    self.deterministic_reasons
                ),
            "qber_raw":
                self.qber_raw,
            "loss_rate":
                self.loss_rate,
            "observed_check_blocks":
                self.observed_check_blocks,
            "p_attack":
                self.p_attack,
            "tag_recovered":
                self.tag_recovered,
            "low_risk_observables":
                self.low_risk_observables,
            "fresh_session_required":
                self.fresh_session_required,
        }


def normalize_retry_policy(
    policy: RetryPolicy | Mapping[str, Any] | None,
) -> RetryPolicy:
    """
    Normalize a RetryPolicy or notebook-style policy mapping.
    """

    if policy is None:
        return DEFAULT_RETRY_POLICY

    if isinstance(
        policy,
        RetryPolicy,
    ):
        return policy

    if not isinstance(
        policy,
        Mapping,
    ):
        raise TypeError(
            "policy must be RetryPolicy, Mapping, or None."
        )

    return RetryPolicy(
        max_authentication_attempts=policy.get(
            "max_authentication_attempts",
            DEFAULT_MAX_AUTHENTICATION_ATTEMPTS,
        ),
        retry_probability_limit=policy.get(
            "retry_probability_limit",
            DEFAULT_RETRY_PROBABILITY_LIMIT,
        ),
        retry_qber_limit=policy.get(
            "retry_qber_limit",
            DEFAULT_RETRY_QBER_LIMIT,
        ),
        max_acceptable_loss_rate=policy.get(
            "max_acceptable_loss_rate",
            DEFAULT_MAX_ACCEPTABLE_LOSS_RATE,
        ),
        min_observed_check_blocks=policy.get(
            "min_observed_check_blocks",
            DEFAULT_MIN_OBSERVED_CHECK_BLOCKS,
        ),
        gp_gray_zone_retry_upper=policy.get(
            "gp_gray_zone_retry_upper",
            DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
        ),
    )


def session_decision(
    session: SessionLike,
) -> dict[str, Any]:
    """
    Return a detached decision dictionary.
    """

    if isinstance(
        session,
        AuthenticationResult,
    ):
        return session.decision.as_dict()

    if not isinstance(
        session,
        Mapping,
    ):
        raise InvalidRetrySessionError(
            "session must be AuthenticationResult or Mapping."
        )

    decision = session.get(
        "decision",
        {},
    )

    if isinstance(
        decision,
        DecisionResult,
    ):
        return decision.as_dict()

    if not isinstance(
        decision,
        Mapping,
    ):
        raise InvalidRetrySessionError(
            "session decision must be a mapping."
        )

    return dict(
        decision
    )


def session_value(
    session: SessionLike,
    field_name: str,
    default: Any = None,
) -> Any:
    """
    Read a field from an AuthenticationResult or dictionary session.
    """

    if isinstance(
        session,
        AuthenticationResult,
    ):
        return getattr(
            session,
            field_name,
            default,
        )

    if isinstance(
        session,
        Mapping,
    ):
        return session.get(
            field_name,
            default,
        )

    raise InvalidRetrySessionError(
        "session must be AuthenticationResult or Mapping."
    )


def session_accepted(
    session: SessionLike,
) -> bool:
    """Return the final acceptance status."""

    decision = session_decision(
        session
    )

    return bool(
        decision.get(
            "accepted",
            False,
        )
    )


def session_deterministic_pass(
    session: SessionLike,
) -> bool:
    """Return deterministic verification status."""

    decision = session_decision(
        session
    )

    return bool(
        decision.get(
            "deterministic_pass",
            False,
        )
    )


def session_deterministic_reasons(
    session: SessionLike,
) -> tuple[str, ...]:
    """
    Return normalized deterministic failure reasons.
    """

    decision = session_decision(
        session
    )

    reasons = decision.get(
        "deterministic_reasons",
        (),
    )

    if reasons is None:
        return ()

    if isinstance(
        reasons,
        str,
    ):
        reasons = (
            reasons,
        )

    if not isinstance(
        reasons,
        Sequence,
    ):
        raise InvalidRetrySessionError(
            "deterministic_reasons must be a sequence."
        )

    normalized: list[str] = []

    for reason in reasons:
        if not isinstance(
            reason,
            str,
        ):
            raise InvalidRetrySessionError(
                "Every deterministic reason must be a string."
            )

        reason = reason.strip()

        if reason and reason not in normalized:
            normalized.append(
                reason
            )

    return tuple(
        normalized
    )


def session_attack_probability(
    session: SessionLike,
) -> float | None:
    """
    Return P(attack) from the final decision record.
    """

    decision = session_decision(
        session
    )

    return optional_probability(
        decision.get(
            "p_attack"
        )
    )


def session_qber(
    session: SessionLike,
) -> float | None:
    """Return raw QBER when available."""

    value = session_value(
        session,
        "qber_raw",
    )

    return optional_probability(
        value
    )


def session_loss_rate(
    session: SessionLike,
) -> float | None:
    """Return physical-qubit loss rate when available."""

    value = session_value(
        session,
        "loss_rate",
    )

    return optional_probability(
        value
    )


def session_observed_check_blocks(
    session: SessionLike,
) -> int:
    """Return the number of observed declared check blocks."""

    value = session_value(
        session,
        "observed_check_blocks",
        0,
    )

    if isinstance(value, bool):
        raise InvalidRetrySessionError(
            "observed_check_blocks cannot be boolean."
        )

    try:
        normalized = int(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ) as error:
        raise InvalidRetrySessionError(
            "observed_check_blocks must be an integer."
        ) from error

    if normalized < 0:
        raise InvalidRetrySessionError(
            "observed_check_blocks cannot be negative."
        )

    return normalized


def tag_matches(
    session: SessionLike,
) -> bool:
    """
    Return whether the authentication tag was recovered and verified.

    The retry engine uses the safe tag_recovered result and does not
    require access to either raw KMAC tag.
    """

    value = session_value(
        session,
        "tag_recovered",
        False,
    )

    return bool(
        value
    )


def low_risk_observables(
    session: SessionLike,
    policy: RetryPolicy | Mapping[str, Any] | None = None,
) -> bool:
    """
    Check notebook-compatible low-risk channel evidence.

    Requirements:

        qber_raw <= retry_qber_limit
        loss_rate <= max_acceptable_loss_rate
        observed_check_blocks >= min_observed_check_blocks
    """

    active_policy = normalize_retry_policy(
        policy
    )

    qber_raw = session_qber(
        session
    )

    loss_rate = session_loss_rate(
        session
    )

    observed_check_blocks = (
        session_observed_check_blocks(
            session
        )
    )

    if qber_raw is None:
        return False

    if loss_rate is None:
        return False

    return bool(
        qber_raw
        <= active_policy.retry_qber_limit
        and loss_rate
        <= active_policy.max_acceptable_loss_rate
        and observed_check_blocks
        >= active_policy.min_observed_check_blocks
    )


def retryable_payload_recovery_failure(
    session: SessionLike,
    policy: RetryPolicy | Mapping[str, Any] | None = None,
) -> bool:
    """
    Return whether a payload/tag recovery failure may be retried.

    Only payload recovery and tag mismatch reasons are allowed.
    Credential, freshness, replay, schedule, evidence, and loss failures
    remain immediate final rejections.
    """

    active_policy = normalize_retry_policy(
        policy
    )

    if session_accepted(
        session
    ):
        return False

    deterministic_reasons = set(
        session_deterministic_reasons(
            session
        )
    )

    if not deterministic_reasons:
        return False

    if not deterministic_reasons.issubset(
        RETRYABLE_PAYLOAD_FAILURE_REASONS
    ):
        return False

    if not low_risk_observables(
        session,
        active_policy,
    ):
        return False

    p_attack = session_attack_probability(
        session
    )

    if (
        p_attack is not None
        and p_attack
        >= active_policy.retry_probability_limit
    ):
        return False

    return True


def retryable_low_risk_gp_rejection(
    session: SessionLike,
    policy: RetryPolicy | Mapping[str, Any] | None = None,
) -> bool:
    """
    Return whether a deterministic-pass GP gray-zone rejection may retry.
    """

    active_policy = normalize_retry_policy(
        policy
    )

    if session_accepted(
        session
    ):
        return False

    decision = session_decision(
        session
    )

    if decision.get(
        "reason"
    ) != GP_POLICY_REJECTION_REASON:
        return False

    if not session_deterministic_pass(
        session
    ):
        return False

    if not tag_matches(
        session
    ):
        return False

    if not low_risk_observables(
        session,
        active_policy,
    ):
        return False

    p_attack = session_attack_probability(
        session
    )

    if p_attack is None:
        return False

    return bool(
        p_attack
        < active_policy.gp_gray_zone_retry_upper
    )


def retryable_reauthentication_condition(
    session: SessionLike,
    policy: RetryPolicy | Mapping[str, Any] | None = None,
) -> bool:
    """
    Return whether a fresh FT-QuPAP reauthentication is permitted.
    """

    active_policy = normalize_retry_policy(
        policy
    )

    return bool(
        retryable_payload_recovery_failure(
            session,
            active_policy,
        )
        or retryable_low_risk_gp_rejection(
            session,
            active_policy,
        )
    )


def evaluate_retry(
    session: SessionLike,
    attempt_number: int,
    maximum_attempts: int | None = None,
    policy: RetryPolicy | Mapping[str, Any] | None = None,
) -> RetryAssessment:
    """
    Produce a complete retry assessment for one attempt.
    """

    active_policy = normalize_retry_policy(
        policy
    )

    attempt_number = validate_positive_integer(
        attempt_number,
        "attempt_number",
    )

    if maximum_attempts is None:
        maximum_attempts = (
            active_policy.max_authentication_attempts
        )

    maximum_attempts = validate_positive_integer(
        maximum_attempts,
        "maximum_attempts",
    )

    if attempt_number > maximum_attempts:
        raise InvalidRetryAttemptError(
            "attempt_number cannot exceed maximum_attempts."
        )

    accepted = session_accepted(
        session
    )

    deterministic_pass = (
        session_deterministic_pass(
            session
        )
    )

    deterministic_reasons = (
        session_deterministic_reasons(
            session
        )
    )

    qber_raw = session_qber(
        session
    )

    loss_rate = session_loss_rate(
        session
    )

    observed_check_blocks = (
        session_observed_check_blocks(
            session
        )
    )

    p_attack = session_attack_probability(
        session
    )

    tag_recovered = tag_matches(
        session
    )

    low_risk = low_risk_observables(
        session,
        active_policy,
    )

    retry_allowed = False
    retry_kind = RETRY_KIND_NONE

    if accepted:
        reason = (
            RETRY_REASON_ALREADY_ACCEPTED
        )

    elif attempt_number >= maximum_attempts:
        reason = (
            RETRY_REASON_ATTEMPT_LIMIT
        )

    elif retryable_payload_recovery_failure(
        session,
        active_policy,
    ):
        retry_allowed = True
        retry_kind = (
            RETRY_KIND_PAYLOAD_RECOVERY
        )
        reason = (
            RETRY_REASON_PAYLOAD_RECOVERY
        )

    elif retryable_low_risk_gp_rejection(
        session,
        active_policy,
    ):
        retry_allowed = True
        retry_kind = (
            RETRY_KIND_GP_GRAY_ZONE
        )
        reason = (
            RETRY_REASON_GP_GRAY_ZONE
        )

    else:
        reason = (
            RETRY_REASON_NONRETRYABLE_EVIDENCE
        )

    return RetryAssessment(
        retry_allowed=retry_allowed,
        reason=reason,
        retry_kind=retry_kind,
        attempt_number=attempt_number,
        maximum_attempts=maximum_attempts,
        accepted=accepted,
        deterministic_pass=(
            deterministic_pass
        ),
        deterministic_reasons=(
            deterministic_reasons
        ),
        qber_raw=qber_raw,
        loss_rate=loss_rate,
        observed_check_blocks=(
            observed_check_blocks
        ),
        p_attack=p_attack,
        tag_recovered=tag_recovered,
        low_risk_observables=low_risk,
        fresh_session_required=(
            retry_allowed
        ),
    )


def annotate_retry_recommendation(
    result: AuthenticationResult,
    assessment: RetryAssessment,
) -> AuthenticationResult:
    """
    Attach a retry recommendation to an AuthenticationResult.

    The authentication decision itself is not changed.
    """

    if not isinstance(
        result,
        AuthenticationResult,
    ):
        raise TypeError(
            "result must be AuthenticationResult."
        )

    if not isinstance(
        assessment,
        RetryAssessment,
    ):
        raise TypeError(
            "assessment must be RetryAssessment."
        )

    result.decision = replace(
        result.decision,
        retry_recommended=(
            assessment.retry_allowed
        ),
    )

    result.diagnostics[
        "retry_assessment"
    ] = assessment.as_dict()

    return result


def summarize_attempt_history(
    attempts: Sequence[SessionLike],
) -> list[dict[str, Any]]:
    """
    Keep only compact, non-secret fields for retry history.
    """

    if isinstance(
        attempts,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "attempts must be a sequence of sessions."
        )

    history: list[dict[str, Any]] = []

    for index, attempt in enumerate(
        attempts,
        start=1,
    ):
        decision = session_decision(
            attempt
        )

        history.append(
            {
                "attempt":
                    index,
                "accepted":
                    bool(
                        decision.get(
                            "accepted",
                            False,
                        )
                    ),
                "reason":
                    decision.get(
                        "reason"
                    ),
                "deterministic_reasons":
                    list(
                        session_deterministic_reasons(
                            attempt
                        )
                    ),
                "qber_raw":
                    session_qber(
                        attempt
                    ),
                "p_attack":
                    session_attack_probability(
                        attempt
                    ),
                "uncertainty":
                    decision.get(
                        "uncertainty"
                    ),
                "loss_rate":
                    session_loss_rate(
                        attempt
                    ),
                "observed_check_blocks":
                    session_observed_check_blocks(
                        attempt
                    ),
                "tag_recovered":
                    tag_matches(
                        attempt
                    ),
            }
        )

    return history


class FTQuPAPRetryEngine:
    """
    Reusable retry-policy service.

    The object is callable and therefore can be passed directly as the
    retry_evaluator argument of FTQuPAPProtocolEngine.
    """

    def __init__(
        self,
        policy: RetryPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = normalize_retry_policy(
            policy
        )

    def assess(
        self,
        result: AuthenticationResult,
        attempt_number: int,
        maximum_attempts: int | None = None,
    ) -> RetryAssessment:
        """
        Evaluate one authentication result.
        """

        return evaluate_retry(
            session=result,
            attempt_number=attempt_number,
            maximum_attempts=maximum_attempts,
            policy=self.policy,
        )

    def should_retry(
        self,
        result: AuthenticationResult,
        attempt_number: int,
        maximum_attempts: int | None = None,
    ) -> bool:
        """
        Return only the retry authorization decision.
        """

        assessment = self.assess(
            result=result,
            attempt_number=attempt_number,
            maximum_attempts=maximum_attempts,
        )

        return assessment.retry_allowed

    def annotate(
        self,
        result: AuthenticationResult,
        attempt_number: int,
        maximum_attempts: int | None = None,
    ) -> AuthenticationResult:
        """
        Add the retry recommendation and assessment to a result.
        """

        assessment = self.assess(
            result=result,
            attempt_number=attempt_number,
            maximum_attempts=maximum_attempts,
        )

        return annotate_retry_recommendation(
            result,
            assessment,
        )

    def __call__(
        self,
        result: AuthenticationResult,
        attempt_number: int,
        maximum_attempts: int,
    ) -> bool:
        """
        ProtocolEngine-compatible retry evaluator.
        """

        return self.should_retry(
            result=result,
            attempt_number=attempt_number,
            maximum_attempts=maximum_attempts,
        )


def run_self_test() -> None:
    """
    Verify payload-recovery, GP-gray-zone, and fail-closed behavior.
    """

    passing_verification = VerificationResult(
        credential_valid=True,
        request_fresh=True,
        replay_safe=True,
        schedule_valid=True,
        check_evidence_sufficient=True,
        required_blocks_correctable=True,
        tag_valid=True,
        loss_policy_valid=True,
        reasons=(),
    )

    payload_failure_reasons = (
        PAYLOAD_BLOCK_UNRECOVERABLE,
        AUTHENTICATION_TAG_MISMATCH,
    )

    payload_failure_verification = VerificationResult(
        credential_valid=True,
        request_fresh=True,
        replay_safe=True,
        schedule_valid=True,
        check_evidence_sufficient=True,
        required_blocks_correctable=False,
        tag_valid=False,
        loss_policy_valid=True,
        reasons=payload_failure_reasons,
    )

    payload_failure_decision = DecisionResult(
        accepted=False,
        reason="deterministic_protocol_check_failed",
        deterministic_pass=False,
        deterministic_reasons=(
            payload_failure_reasons
        ),
        p_attack=None,
        uncertainty=None,
        gp_attack_threshold=0.15,
        retry_recommended=False,
    )

    payload_failure_result = AuthenticationResult(
        session_id="SESSION-PAYLOAD-FAILURE",
        pseudonym_id="PID-TEST-001",
        decision=payload_failure_decision,
        verification=(
            payload_failure_verification
        ),
        attack_detection=None,
        qber_raw=0.05,
        qber_mismatches=11,
        qber_observed=224,
        observed_check_blocks=32,
        loss_rate=0.02,
        tag_recovered=False,
        physical_qubits=1120,
        channel_name="benign-noisy",
        channel_context="urban",
    )

    gray_zone_attack_detection = AttackDetectionResult(
        p_attack=0.18,
        uncertainty=0.25,
        threshold=0.15,
        model_available=True,
        calibrated=True,
    )

    gray_zone_decision = DecisionResult(
        accepted=False,
        reason=GP_POLICY_REJECTION_REASON,
        deterministic_pass=True,
        deterministic_reasons=(),
        p_attack=0.18,
        uncertainty=0.25,
        gp_attack_threshold=0.15,
        retry_recommended=False,
    )

    gray_zone_result = AuthenticationResult(
        session_id="SESSION-GP-GRAY-ZONE",
        pseudonym_id="PID-TEST-001",
        decision=gray_zone_decision,
        verification=passing_verification,
        attack_detection=(
            gray_zone_attack_detection
        ),
        qber_raw=0.01,
        qber_mismatches=2,
        qber_observed=224,
        observed_check_blocks=32,
        loss_rate=0.01,
        tag_recovered=True,
        physical_qubits=1120,
        channel_name="benign-noisy",
        channel_context="urban",
    )

    strong_attack_detection = AttackDetectionResult(
        p_attack=0.80,
        uncertainty=0.10,
        threshold=0.15,
        model_available=True,
        calibrated=True,
    )

    strong_attack_decision = DecisionResult(
        accepted=False,
        reason=GP_POLICY_REJECTION_REASON,
        deterministic_pass=True,
        deterministic_reasons=(),
        p_attack=0.80,
        uncertainty=0.10,
        gp_attack_threshold=0.15,
        retry_recommended=False,
    )

    strong_attack_result = AuthenticationResult(
        session_id="SESSION-STRONG-ATTACK",
        pseudonym_id="PID-TEST-001",
        decision=strong_attack_decision,
        verification=passing_verification,
        attack_detection=(
            strong_attack_detection
        ),
        qber_raw=0.30,
        qber_mismatches=67,
        qber_observed=224,
        observed_check_blocks=32,
        loss_rate=0.01,
        tag_recovered=True,
        physical_qubits=1120,
        channel_name="full-eve",
        channel_context="urban",
    )

    replay_reasons = (
        "freshness_or_replay_failure",
    )

    replay_verification = VerificationResult(
        credential_valid=True,
        request_fresh=False,
        replay_safe=False,
        schedule_valid=True,
        check_evidence_sufficient=True,
        required_blocks_correctable=True,
        tag_valid=True,
        loss_policy_valid=True,
        reasons=replay_reasons,
    )

    replay_decision = DecisionResult(
        accepted=False,
        reason="deterministic_protocol_check_failed",
        deterministic_pass=False,
        deterministic_reasons=replay_reasons,
        p_attack=None,
        uncertainty=None,
        gp_attack_threshold=0.15,
        retry_recommended=False,
    )

    replay_result = AuthenticationResult(
        session_id="SESSION-REPLAY",
        pseudonym_id="PID-TEST-001",
        decision=replay_decision,
        verification=replay_verification,
        attack_detection=None,
        qber_raw=0.0,
        qber_mismatches=0,
        qber_observed=224,
        observed_check_blocks=32,
        loss_rate=0.0,
        tag_recovered=True,
        physical_qubits=1120,
        channel_name="replay",
        channel_context="urban",
    )

    accepted_attack_detection = AttackDetectionResult(
        p_attack=0.05,
        uncertainty=0.10,
        threshold=0.15,
        model_available=True,
        calibrated=True,
    )

    accepted_decision = DecisionResult(
        accepted=True,
        reason="accepted_by_calibrated_bayesian_policy",
        deterministic_pass=True,
        deterministic_reasons=(),
        p_attack=0.05,
        uncertainty=0.10,
        gp_attack_threshold=0.15,
        retry_recommended=False,
    )

    accepted_result = AuthenticationResult(
        session_id="SESSION-ACCEPTED",
        pseudonym_id="PID-TEST-001",
        decision=accepted_decision,
        verification=passing_verification,
        attack_detection=(
            accepted_attack_detection
        ),
        qber_raw=0.0,
        qber_mismatches=0,
        qber_observed=224,
        observed_check_blocks=32,
        loss_rate=0.0,
        tag_recovered=True,
        physical_qubits=1120,
        channel_name="ideal",
        channel_context="urban",
    )

    engine = FTQuPAPRetryEngine()

    payload_assessment = engine.assess(
        payload_failure_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    gray_zone_assessment = engine.assess(
        gray_zone_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    strong_attack_assessment = engine.assess(
        strong_attack_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    replay_assessment = engine.assess(
        replay_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    limit_assessment = engine.assess(
        payload_failure_result,
        attempt_number=3,
        maximum_attempts=3,
    )

    accepted_assessment = engine.assess(
        accepted_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    if not payload_assessment.retry_allowed:
        raise RetryEngineError(
            "Low-risk payload recovery failure was not retryable."
        )

    if (
        payload_assessment.retry_kind
        != RETRY_KIND_PAYLOAD_RECOVERY
    ):
        raise RetryEngineError(
            "Payload retry kind is incorrect."
        )

    if not gray_zone_assessment.retry_allowed:
        raise RetryEngineError(
            "Low-risk GP gray-zone rejection was not retryable."
        )

    if (
        gray_zone_assessment.retry_kind
        != RETRY_KIND_GP_GRAY_ZONE
    ):
        raise RetryEngineError(
            "GP gray-zone retry kind is incorrect."
        )

    if strong_attack_assessment.retry_allowed:
        raise RetryEngineError(
            "Strong attack evidence was incorrectly retryable."
        )

    if replay_assessment.retry_allowed:
        raise RetryEngineError(
            "Replay failure was incorrectly retryable."
        )

    if limit_assessment.retry_allowed:
        raise RetryEngineError(
            "Retry was allowed after the maximum attempt."
        )

    if accepted_assessment.retry_allowed:
        raise RetryEngineError(
            "Accepted session was incorrectly retryable."
        )

    annotated_result = engine.annotate(
        payload_failure_result,
        attempt_number=1,
        maximum_attempts=3,
    )

    if not annotated_result.decision.retry_recommended:
        raise RetryEngineError(
            "Retry recommendation was not attached."
        )

    history = summarize_attempt_history(
        [
            payload_failure_result,
            accepted_result,
        ]
    )

    if len(history) != 2:
        raise RetryEngineError(
            "Attempt-history summary is incorrect."
        )

    print(
        "Retry engine self-test completed successfully."
    )

    print(
        "Payload recovery retry:",
        payload_assessment.retry_allowed,
    )

    print(
        "GP gray-zone retry:",
        gray_zone_assessment.retry_allowed,
    )

    print(
        "Strong attack retry:",
        strong_attack_assessment.retry_allowed,
    )

    print(
        "Replay retry:",
        replay_assessment.retry_allowed,
    )

    print(
        "Maximum attempts:",
        engine.policy.max_authentication_attempts,
    )


__all__ = [
    "DEFAULT_MAX_AUTHENTICATION_ATTEMPTS",
    "DEFAULT_RETRY_PROBABILITY_LIMIT",
    "DEFAULT_RETRY_QBER_LIMIT",
    "DEFAULT_MAX_ACCEPTABLE_LOSS_RATE",
    "DEFAULT_MIN_OBSERVED_CHECK_BLOCKS",
    "DEFAULT_GP_GRAY_ZONE_RETRY_UPPER",
    "PAYLOAD_BLOCK_UNRECOVERABLE",
    "AUTHENTICATION_TAG_MISMATCH",
    "GP_POLICY_REJECTION_REASON",
    "RETRY_KIND_NONE",
    "RETRY_KIND_PAYLOAD_RECOVERY",
    "RETRY_KIND_GP_GRAY_ZONE",
    "RETRY_REASON_PAYLOAD_RECOVERY",
    "RETRY_REASON_GP_GRAY_ZONE",
    "RETRY_REASON_ALREADY_ACCEPTED",
    "RETRY_REASON_ATTEMPT_LIMIT",
    "RETRY_REASON_NONRETRYABLE_EVIDENCE",
    "RETRYABLE_PAYLOAD_FAILURE_REASONS",
    "RetryEngineError",
    "InvalidRetryPolicyError",
    "InvalidRetrySessionError",
    "InvalidRetryAttemptError",
    "RetryPolicy",
    "RetryAssessment",
    "FTQuPAPRetryEngine",
    "DEFAULT_RETRY_POLICY",
    "normalize_retry_policy",
    "session_decision",
    "session_value",
    "session_accepted",
    "session_deterministic_pass",
    "session_deterministic_reasons",
    "session_attack_probability",
    "session_qber",
    "session_loss_rate",
    "session_observed_check_blocks",
    "tag_matches",
    "low_risk_observables",
    "retryable_payload_recovery_failure",
    "retryable_low_risk_gp_rejection",
    "retryable_reauthentication_condition",
    "evaluate_retry",
    "annotate_retry_recommendation",
    "summarize_attempt_history",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        RetryEngineError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[RETRY ENGINE ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error