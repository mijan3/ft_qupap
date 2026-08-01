"""
FT-QuPAP Decision Engine

This module produces the final authentication decision after:

    1. Mandatory deterministic verification
    2. Calibrated Gaussian Process attack prediction
    3. Optional uncertainty policy
    4. Calibrated operational-threshold comparison

The GP detector never replaces deterministic cryptographic and
quantum-verification checks.

Notebook-compatible GP acceptance order:

    if deterministic checks fail:
        reject

    if a calibrated uncertainty limit exists and uncertainty exceeds it:
        reject

    selected_threshold = calibrated threshold

    if the calibrated threshold is unavailable:
        selected_threshold = (
            false_reject_cost
            / (false_accept_cost + false_reject_cost)
        )

    operational_threshold = max(
        selected_threshold,
        minimum_operational_threshold,
    )

    if P(attack) < operational_threshold:
        accept
    else:
        reject

The threshold comparison is intentionally strict:

    attack_probability < selected_threshold

Therefore, an attack probability exactly equal to the threshold is
rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .result_models import (
    AttackDetectionResult,
    DecisionResult,
    VerificationResult,
)


DEFAULT_FALSE_ACCEPT_COST = 10.0
DEFAULT_FALSE_REJECT_COST = 1.0

DEFAULT_MIN_OPERATIONAL_THRESHOLD = 0.15
DEFAULT_FIXED_QBER_THRESHOLD = 0.11

DEFAULT_GP_MAX_UNCERTAINTY: float | None = None
DEFAULT_GP_GRAY_ZONE_RETRY_UPPER = 0.20


REASON_DETERMINISTIC_FAILURE = (
    "deterministic_protocol_check_failed"
)

REASON_GP_UNCERTAINTY_TOO_HIGH = (
    "gp_uncertainty_too_high"
)

REASON_ACCEPTED_BY_GP_POLICY = (
    "accepted_by_calibrated_bayesian_policy"
)

REASON_REJECTED_BY_GP_POLICY = (
    "rejected_by_calibrated_bayesian_policy"
)

REASON_FIXED_QBER_ACCEPTED = "accepted"

REASON_FIXED_QBER_EXCEEDED = (
    "fixed_qber_threshold_exceeded"
)


class DecisionEngineError(Exception):
    """Base exception for FT-QuPAP decision failures."""


class InvalidDecisionPolicyError(DecisionEngineError):
    """Raised when decision-policy parameters are invalid."""


class InvalidDecisionEvidenceError(DecisionEngineError):
    """Raised when GP or QBER evidence is malformed."""


class InconsistentDecisionError(DecisionEngineError):
    """Raised when a decision contradicts its evidence."""


def validate_boolean(
    value: Any,
    field_name: str,
) -> bool:
    """Validate a required boolean value."""

    if not isinstance(value, bool):
        raise TypeError(
            f"{field_name} must be boolean."
        )

    return value


def validate_nonempty_string(
    value: Any,
    field_name: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


def validate_probability(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> float | None:
    """
    Validate a finite probability in the interval [0, 1].
    """

    if value is None:
        if allow_none:
            return None

        raise InvalidDecisionEvidenceError(
            f"{field_name} cannot be None."
        )

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


def validate_positive_cost(
    value: Any,
    field_name: str,
) -> float:
    """Validate a positive Bayesian decision cost."""

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

    if normalized <= 0.0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    return normalized


def normalize_reasons(
    reasons: Sequence[str] | None,
) -> tuple[str, ...]:
    """
    Validate deterministic reasons while preserving their order.
    """

    if reasons is None:
        return ()

    if isinstance(
        reasons,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise TypeError(
            "deterministic_reasons must be "
            "a sequence of strings."
        )

    if not isinstance(reasons, Sequence):
        raise TypeError(
            "deterministic_reasons must be a sequence."
        )

    normalized: list[str] = []

    for index, reason in enumerate(reasons):
        reason = validate_nonempty_string(
            reason,
            f"deterministic_reasons[{index}]",
        )

        if reason not in normalized:
            normalized.append(reason)

    return tuple(normalized)


@dataclass(frozen=True)
class DecisionPolicy:
    """
    Calibrated FT-QuPAP authentication-decision policy.

    Attributes:
        false_accept_cost:
            Bayesian cost of accepting an attack.

        false_reject_cost:
            Bayesian cost of rejecting a legitimate session.

        gp_attack_threshold:
            Calibrated operational threshold loaded from the model
            artifacts. It may be None before calibration.

        raw_calibration_threshold:
            Raw threshold selected from calibration data before the
            operational lower bound is applied.

        min_operational_threshold:
            Availability-constrained lower bound for the deployed
            threshold.

        gp_max_uncertainty:
            Optional separately calibrated uncertainty limit.
            The final notebook leaves this as None.

        fixed_qber_threshold:
            Threshold used only by fixed-QBER baseline protocols.

        gp_gray_zone_retry_upper:
            Upper attack-probability boundary that may later be used by
            retry_engine.py. This module does not approve retry using
            attack probability alone.
    """

    false_accept_cost: float = (
        DEFAULT_FALSE_ACCEPT_COST
    )

    false_reject_cost: float = (
        DEFAULT_FALSE_REJECT_COST
    )

    gp_attack_threshold: float | None = None

    raw_calibration_threshold: float | None = None

    min_operational_threshold: float = (
        DEFAULT_MIN_OPERATIONAL_THRESHOLD
    )

    gp_max_uncertainty: float | None = (
        DEFAULT_GP_MAX_UNCERTAINTY
    )

    fixed_qber_threshold: float = (
        DEFAULT_FIXED_QBER_THRESHOLD
    )

    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_RETRY_UPPER
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "false_accept_cost",
            validate_positive_cost(
                self.false_accept_cost,
                "false_accept_cost",
            ),
        )

        object.__setattr__(
            self,
            "false_reject_cost",
            validate_positive_cost(
                self.false_reject_cost,
                "false_reject_cost",
            ),
        )

        object.__setattr__(
            self,
            "gp_attack_threshold",
            validate_probability(
                self.gp_attack_threshold,
                "gp_attack_threshold",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "raw_calibration_threshold",
            validate_probability(
                self.raw_calibration_threshold,
                "raw_calibration_threshold",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "min_operational_threshold",
            validate_probability(
                self.min_operational_threshold,
                "min_operational_threshold",
            ),
        )

        object.__setattr__(
            self,
            "gp_max_uncertainty",
            validate_probability(
                self.gp_max_uncertainty,
                "gp_max_uncertainty",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "fixed_qber_threshold",
            validate_probability(
                self.fixed_qber_threshold,
                "fixed_qber_threshold",
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

        if (
            self.gp_gray_zone_retry_upper
            < self.min_operational_threshold
        ):
            raise InvalidDecisionPolicyError(
                "gp_gray_zone_retry_upper cannot be lower "
                "than min_operational_threshold."
            )

    @property
    def bayes_risk_threshold(self) -> float:
        """
        Calculate the pre-calibration Bayesian fallback threshold.

        Formula from the notebook:

            C_false_reject
            ---------------------------------
            C_false_accept + C_false_reject
        """

        return float(
            self.false_reject_cost
            / (
                self.false_accept_cost
                + self.false_reject_cost
            )
        )

    @property
    def threshold_before_operational_bound(self) -> float:
        """
        Return the best currently available threshold.

        Priority:

            1. Calibrated deployed threshold
            2. Raw calibration threshold
            3. Bayesian cost fallback
        """

        if self.gp_attack_threshold is not None:
            return self.gp_attack_threshold

        if self.raw_calibration_threshold is not None:
            return self.raw_calibration_threshold

        return self.bayes_risk_threshold

    @property
    def selected_attack_threshold(self) -> float:
        """
        Return the final availability-constrained threshold.

        operational_threshold = max(
            calibration_or_bayes_threshold,
            min_operational_threshold,
        )
        """

        return float(
            max(
                self.threshold_before_operational_bound,
                self.min_operational_threshold,
            )
        )

    def with_calibrated_threshold(
        self,
        threshold: float,
        *,
        raw_threshold: float | None = None,
    ) -> "DecisionPolicy":
        """
        Return a new policy containing loaded calibration evidence.
        """

        normalized_threshold = validate_probability(
            threshold,
            "threshold",
        )

        normalized_raw_threshold = validate_probability(
            raw_threshold,
            "raw_threshold",
            allow_none=True,
        )

        return replace(
            self,
            gp_attack_threshold=normalized_threshold,
            raw_calibration_threshold=(
                normalized_raw_threshold
            ),
        )

    def as_security_policy(self) -> dict[str, Any]:
        """
        Return notebook-compatible security-policy keys.
        """

        return {
            "bayes_cost_false_accept":
                self.false_accept_cost,

            "bayes_cost_false_reject":
                self.false_reject_cost,

            "gp_attack_threshold":
                self.selected_attack_threshold,

            "raw_calibration_gp_attack_threshold":
                self.raw_calibration_threshold,

            "min_operational_gp_threshold":
                self.min_operational_threshold,

            "gp_max_uncertainty":
                self.gp_max_uncertainty,

            "fixed_qber_threshold":
                self.fixed_qber_threshold,

            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,
        }


DEFAULT_DECISION_POLICY = DecisionPolicy()


def policy_from_mapping(
    policy: Mapping[str, Any],
) -> DecisionPolicy:
    """
    Create a DecisionPolicy from notebook-style SECURITY_POLICY data.
    """

    if not isinstance(policy, Mapping):
        raise TypeError(
            "policy must be a mapping."
        )

    return DecisionPolicy(
        false_accept_cost=policy.get(
            "bayes_cost_false_accept",
            DEFAULT_FALSE_ACCEPT_COST,
        ),

        false_reject_cost=policy.get(
            "bayes_cost_false_reject",
            DEFAULT_FALSE_REJECT_COST,
        ),

        gp_attack_threshold=policy.get(
            "gp_attack_threshold"
        ),

        raw_calibration_threshold=policy.get(
            "raw_calibration_gp_attack_threshold"
        ),

        min_operational_threshold=policy.get(
            "min_operational_gp_threshold",
            DEFAULT_MIN_OPERATIONAL_THRESHOLD,
        ),

        gp_max_uncertainty=policy.get(
            "gp_max_uncertainty"
        ),

        fixed_qber_threshold=policy.get(
            "fixed_qber_threshold",
            DEFAULT_FIXED_QBER_THRESHOLD,
        ),

        gp_gray_zone_retry_upper=policy.get(
            "gp_gray_zone_retry_upper",
            DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
        ),
    )


def normalize_policy(
    policy: DecisionPolicy | Mapping[str, Any] | None,
) -> DecisionPolicy:
    """Normalize supported policy inputs."""

    if policy is None:
        return DEFAULT_DECISION_POLICY

    if isinstance(policy, DecisionPolicy):
        return policy

    if isinstance(policy, Mapping):
        return policy_from_mapping(policy)

    raise TypeError(
        "policy must be DecisionPolicy, Mapping, or None."
    )


@dataclass(frozen=True)
class DecisionEvaluation:
    """
    Detailed output from one decision-engine evaluation.
    """

    decision: DecisionResult
    attack_detection: AttackDetectionResult | None

    decision_mode: str
    selected_threshold: float | None

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            DecisionResult,
        ):
            raise TypeError(
                "decision must be DecisionResult."
            )

        if (
            self.attack_detection is not None
            and not isinstance(
                self.attack_detection,
                AttackDetectionResult,
            )
        ):
            raise TypeError(
                "attack_detection must be "
                "AttackDetectionResult or None."
            )

        if self.decision_mode not in {
            "gp",
            "fixed_qber",
        }:
            raise ValueError(
                "decision_mode must be 'gp' or 'fixed_qber'."
            )

        validate_probability(
            self.selected_threshold,
            "selected_threshold",
            allow_none=True,
        )

    @property
    def accepted(self) -> bool:
        """Return the final authentication status."""

        return self.decision.accepted

    @property
    def reason(self) -> str:
        """Return the final decision reason."""

        return self.decision.reason

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable decision evaluation."""

        return {
            "decision":
                self.decision.as_dict(),

            "attack_detection": (
                None
                if self.attack_detection is None
                else self.attack_detection.as_dict()
            ),

            "decision_mode":
                self.decision_mode,

            "selected_threshold":
                self.selected_threshold,
        }


def policy_acceptance_rule(
    deterministic_pass: bool,
    attack_probability: float | None,
    uncertainty: float | None,
    policy: DecisionPolicy | Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Apply the notebook-compatible calibrated GP acceptance rule.

    This function keeps the original three required parameters so the
    protocol engine can call it in the same way as the notebook.
    """

    deterministic_pass = validate_boolean(
        deterministic_pass,
        "deterministic_pass",
    )

    active_policy = normalize_policy(
        policy
    )

    # Mandatory deterministic checks always take precedence.
    if not deterministic_pass:
        return (
            False,
            REASON_DETERMINISTIC_FAILURE,
        )

    normalized_probability = validate_probability(
        attack_probability,
        "attack_probability",
    )

    normalized_uncertainty = validate_probability(
        uncertainty,
        "uncertainty",
        allow_none=True,
    )

    if active_policy.gp_max_uncertainty is not None:
        if normalized_uncertainty is None:
            raise InvalidDecisionEvidenceError(
                "uncertainty is required when "
                "gp_max_uncertainty is configured."
            )

        if (
            normalized_uncertainty
            > active_policy.gp_max_uncertainty
        ):
            return (
                False,
                REASON_GP_UNCERTAINTY_TOO_HIGH,
            )

    selected_threshold = (
        active_policy.selected_attack_threshold
    )

    if normalized_probability < selected_threshold:
        return (
            True,
            REASON_ACCEPTED_BY_GP_POLICY,
        )

    return (
        False,
        REASON_REJECTED_BY_GP_POLICY,
    )


def fixed_qber_acceptance_rule(
    deterministic_pass: bool,
    qber_raw: float | None,
    fixed_qber_threshold: float = (
        DEFAULT_FIXED_QBER_THRESHOLD
    ),
) -> tuple[bool, str]:
    """
    Apply the notebook-compatible fixed-QBER baseline decision.

    Fixed-QBER mode does not execute the adaptive GP detector.
    """

    deterministic_pass = validate_boolean(
        deterministic_pass,
        "deterministic_pass",
    )

    threshold = validate_probability(
        fixed_qber_threshold,
        "fixed_qber_threshold",
    )

    if not deterministic_pass:
        return (
            False,
            REASON_DETERMINISTIC_FAILURE,
        )

    normalized_qber = validate_probability(
        qber_raw,
        "qber_raw",
    )

    if normalized_qber <= threshold:
        return (
            True,
            REASON_FIXED_QBER_ACCEPTED,
        )

    return (
        False,
        REASON_FIXED_QBER_EXCEEDED,
    )


def build_decision_record(
    accepted: bool,
    reason: str,
    deterministic_reasons: Sequence[str] | None,
    attack_probability: float | None,
    uncertainty: float | None,
    gp_attack_threshold: float | None = None,
) -> dict[str, Any]:
    """
    Create the notebook-compatible auditable decision dictionary.

    Returned keys:

        accepted
        reason
        deterministic_pass
        deterministic_reasons
        p_attack
        uncertainty
        gp_attack_threshold
    """

    accepted = validate_boolean(
        accepted,
        "accepted",
    )

    reason = validate_nonempty_string(
        reason,
        "reason",
    )

    normalized_reasons = normalize_reasons(
        deterministic_reasons
    )

    normalized_probability = validate_probability(
        attack_probability,
        "attack_probability",
        allow_none=True,
    )

    normalized_uncertainty = validate_probability(
        uncertainty,
        "uncertainty",
        allow_none=True,
    )

    if gp_attack_threshold is None:
        normalized_threshold = (
            DEFAULT_DECISION_POLICY
            .selected_attack_threshold
        )
    else:
        normalized_threshold = validate_probability(
            gp_attack_threshold,
            "gp_attack_threshold",
        )

    deterministic_pass = (
        len(normalized_reasons) == 0
    )

    if accepted and not deterministic_pass:
        raise InconsistentDecisionError(
            "A session cannot be accepted when "
            "deterministic checks failed."
        )

    if (
        accepted
        and normalized_probability is not None
        and normalized_probability
        >= normalized_threshold
    ):
        raise InconsistentDecisionError(
            "Accepted GP decision has an attack probability "
            "at or above the selected threshold."
        )

    return {
        "accepted":
            accepted,

        "reason":
            reason,

        "deterministic_pass":
            deterministic_pass,

        "deterministic_reasons":
            list(normalized_reasons),

        "p_attack":
            normalized_probability,

        "uncertainty":
            normalized_uncertainty,

        "gp_attack_threshold":
            normalized_threshold,
    }


def evaluate_gp_decision(
    verification: VerificationResult,
    attack_probability: float | None,
    uncertainty: float | None,
    *,
    policy: DecisionPolicy | Mapping[str, Any] | None = None,
    retry_recommended: bool = False,
) -> DecisionEvaluation:
    """
    Combine deterministic verification and calibrated GP evidence.

    retry_recommended should normally be set by retry_engine.py after
    checking QBER, loss, tag recovery, deterministic reasons, and the
    maximum-attempt limit.
    """

    if not isinstance(
        verification,
        VerificationResult,
    ):
        raise TypeError(
            "verification must be VerificationResult."
        )

    retry_recommended = validate_boolean(
        retry_recommended,
        "retry_recommended",
    )

    active_policy = normalize_policy(
        policy
    )

    selected_threshold = (
        active_policy.selected_attack_threshold
    )

    attack_detection: AttackDetectionResult | None

    if attack_probability is None:
        attack_detection = None

    else:
        normalized_probability = validate_probability(
            attack_probability,
            "attack_probability",
        )

        normalized_uncertainty = validate_probability(
            uncertainty,
            "uncertainty",
            allow_none=True,
        )

        attack_detection = AttackDetectionResult(
            p_attack=normalized_probability,
            uncertainty=normalized_uncertainty,
            threshold=selected_threshold,
            model_available=True,
            model_name="GaussianProcessClassifier",
            calibrated=True,
        )

    accepted, reason = policy_acceptance_rule(
        deterministic_pass=(
            verification.deterministic_pass
        ),
        attack_probability=attack_probability,
        uncertainty=uncertainty,
        policy=active_policy,
    )

    decision = DecisionResult.build(
        accepted=accepted,
        reason=reason,
        verification=verification,
        attack_detection=attack_detection,
        retry_recommended=retry_recommended,
    )

    return DecisionEvaluation(
        decision=decision,
        attack_detection=attack_detection,
        decision_mode="gp",
        selected_threshold=selected_threshold,
    )


def evaluate_fixed_qber_decision(
    verification: VerificationResult,
    qber_raw: float | None,
    *,
    policy: DecisionPolicy | Mapping[str, Any] | None = None,
) -> DecisionEvaluation:
    """
    Produce a fixed-QBER baseline decision.

    GP probability and uncertainty remain None in baseline mode.
    """

    if not isinstance(
        verification,
        VerificationResult,
    ):
        raise TypeError(
            "verification must be VerificationResult."
        )

    active_policy = normalize_policy(
        policy
    )

    accepted, reason = fixed_qber_acceptance_rule(
        deterministic_pass=(
            verification.deterministic_pass
        ),
        qber_raw=qber_raw,
        fixed_qber_threshold=(
            active_policy.fixed_qber_threshold
        ),
    )

    decision = DecisionResult(
        accepted=accepted,
        reason=reason,
        deterministic_pass=(
            verification.deterministic_pass
        ),
        deterministic_reasons=(
            verification.reasons
        ),
        p_attack=None,
        uncertainty=None,
        gp_attack_threshold=None,
        retry_recommended=False,
    )

    return DecisionEvaluation(
        decision=decision,
        attack_detection=None,
        decision_mode="fixed_qber",
        selected_threshold=(
            active_policy.fixed_qber_threshold
        ),
    )


class FTQuPAPDecisionEngine:
    """
    Reusable FT-QuPAP final-decision service.
    """

    def __init__(
        self,
        policy: DecisionPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.policy = normalize_policy(
            policy
        )

    @property
    def selected_attack_threshold(self) -> float:
        """Return the active operational GP threshold."""

        return self.policy.selected_attack_threshold

    def update_calibrated_threshold(
        self,
        threshold: float,
        *,
        raw_threshold: float | None = None,
    ) -> None:
        """
        Install a calibrated threshold loaded from model artifacts.
        """

        self.policy = (
            self.policy.with_calibrated_threshold(
                threshold,
                raw_threshold=raw_threshold,
            )
        )

    def decide_gp(
        self,
        verification: VerificationResult,
        attack_probability: float | None,
        uncertainty: float | None,
        *,
        retry_recommended: bool = False,
    ) -> DecisionEvaluation:
        """Execute the calibrated GP decision path."""

        return evaluate_gp_decision(
            verification=verification,
            attack_probability=attack_probability,
            uncertainty=uncertainty,
            policy=self.policy,
            retry_recommended=retry_recommended,
        )

    def decide_fixed_qber(
        self,
        verification: VerificationResult,
        qber_raw: float | None,
    ) -> DecisionEvaluation:
        """Execute the fixed-QBER baseline path."""

        return evaluate_fixed_qber_decision(
            verification=verification,
            qber_raw=qber_raw,
            policy=self.policy,
        )

    def decide(
        self,
        *,
        decision_mode: str,
        verification: VerificationResult,
        attack_probability: float | None = None,
        uncertainty: float | None = None,
        qber_raw: float | None = None,
        retry_recommended: bool = False,
    ) -> DecisionEvaluation:
        """
        Execute either the GP or fixed-QBER decision path.
        """

        if decision_mode == "gp":
            return self.decide_gp(
                verification=verification,
                attack_probability=attack_probability,
                uncertainty=uncertainty,
                retry_recommended=retry_recommended,
            )

        if decision_mode == "fixed_qber":
            return self.decide_fixed_qber(
                verification=verification,
                qber_raw=qber_raw,
            )

        raise ValueError(
            "decision_mode must be 'gp' or 'fixed_qber'."
        )


def run_self_test() -> None:
    """
    Verify deterministic, GP, uncertainty, and fixed-QBER decisions.
    """

    passing_verification = (
        VerificationResult.from_checks(
            credential_valid=True,
            request_fresh=True,
            replay_safe=True,
            schedule_valid=True,
            check_evidence_sufficient=True,
            required_blocks_correctable=True,
            tag_valid=True,
            loss_policy_valid=True,
        )
    )

    failing_verification = (
        VerificationResult.from_checks(
            credential_valid=True,
            request_fresh=True,
            replay_safe=True,
            schedule_valid=True,
            check_evidence_sufficient=True,
            required_blocks_correctable=True,
            tag_valid=False,
            loss_policy_valid=True,
        )
    )

    engine = FTQuPAPDecisionEngine()

    expected_bayes_threshold = (
        DEFAULT_FALSE_REJECT_COST
        / (
            DEFAULT_FALSE_ACCEPT_COST
            + DEFAULT_FALSE_REJECT_COST
        )
    )

    if not math.isclose(
        engine.policy.bayes_risk_threshold,
        expected_bayes_threshold,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DecisionEngineError(
            "Bayesian fallback threshold is incorrect."
        )

    if engine.selected_attack_threshold != 0.15:
        raise DecisionEngineError(
            "Operational minimum threshold was not applied."
        )

    low_risk = engine.decide_gp(
        verification=passing_verification,
        attack_probability=0.05,
        uncertainty=0.25,
    )

    if not low_risk.accepted:
        raise DecisionEngineError(
            "Low-risk valid session was rejected."
        )

    if low_risk.reason != REASON_ACCEPTED_BY_GP_POLICY:
        raise DecisionEngineError(
            "Low-risk acceptance reason is incorrect."
        )

    threshold_boundary = engine.decide_gp(
        verification=passing_verification,
        attack_probability=0.15,
        uncertainty=0.61,
    )

    if threshold_boundary.accepted:
        raise DecisionEngineError(
            "Probability equal to the threshold was accepted."
        )

    if (
        threshold_boundary.reason
        != REASON_REJECTED_BY_GP_POLICY
    ):
        raise DecisionEngineError(
            "Threshold rejection reason is incorrect."
        )

    deterministic_failure = engine.decide_gp(
        verification=failing_verification,
        attack_probability=None,
        uncertainty=None,
    )

    if deterministic_failure.accepted:
        raise DecisionEngineError(
            "Deterministic failure was accepted."
        )

    if (
        deterministic_failure.reason
        != REASON_DETERMINISTIC_FAILURE
    ):
        raise DecisionEngineError(
            "Deterministic rejection reason is incorrect."
        )

    uncertainty_engine = FTQuPAPDecisionEngine(
        DecisionPolicy(
            gp_max_uncertainty=0.50,
        )
    )

    uncertain_result = (
        uncertainty_engine.decide_gp(
            verification=passing_verification,
            attack_probability=0.05,
            uncertainty=0.80,
        )
    )

    if uncertain_result.accepted:
        raise DecisionEngineError(
            "High-uncertainty session was accepted."
        )

    if (
        uncertain_result.reason
        != REASON_GP_UNCERTAINTY_TOO_HIGH
    ):
        raise DecisionEngineError(
            "Uncertainty rejection reason is incorrect."
        )

    fixed_qber_accept = (
        engine.decide_fixed_qber(
            verification=passing_verification,
            qber_raw=0.05,
        )
    )

    if not fixed_qber_accept.accepted:
        raise DecisionEngineError(
            "Valid fixed-QBER session was rejected."
        )

    fixed_qber_reject = (
        engine.decide_fixed_qber(
            verification=passing_verification,
            qber_raw=0.12,
        )
    )

    if fixed_qber_reject.accepted:
        raise DecisionEngineError(
            "Excessive fixed-QBER session was accepted."
        )

    decision_record = build_decision_record(
        accepted=True,
        reason=REASON_ACCEPTED_BY_GP_POLICY,
        deterministic_reasons=[],
        attack_probability=0.05,
        uncertainty=0.25,
        gp_attack_threshold=(
            engine.selected_attack_threshold
        ),
    )

    expected_keys = {
        "accepted",
        "reason",
        "deterministic_pass",
        "deterministic_reasons",
        "p_attack",
        "uncertainty",
        "gp_attack_threshold",
    }

    if set(decision_record) != expected_keys:
        raise DecisionEngineError(
            "Decision record keys are incorrect."
        )

    print(
        "Decision engine self-test completed successfully."
    )

    print(
        "Bayesian fallback threshold:",
        f"{engine.policy.bayes_risk_threshold:.6f}",
    )

    print(
        "Operational GP threshold:",
        f"{engine.selected_attack_threshold:.6f}",
    )


__all__ = [
    "DEFAULT_FALSE_ACCEPT_COST",
    "DEFAULT_FALSE_REJECT_COST",
    "DEFAULT_MIN_OPERATIONAL_THRESHOLD",
    "DEFAULT_FIXED_QBER_THRESHOLD",
    "DEFAULT_GP_MAX_UNCERTAINTY",
    "DEFAULT_GP_GRAY_ZONE_RETRY_UPPER",
    "REASON_DETERMINISTIC_FAILURE",
    "REASON_GP_UNCERTAINTY_TOO_HIGH",
    "REASON_ACCEPTED_BY_GP_POLICY",
    "REASON_REJECTED_BY_GP_POLICY",
    "REASON_FIXED_QBER_ACCEPTED",
    "REASON_FIXED_QBER_EXCEEDED",
    "DecisionEngineError",
    "InvalidDecisionPolicyError",
    "InvalidDecisionEvidenceError",
    "InconsistentDecisionError",
    "DecisionPolicy",
    "DecisionEvaluation",
    "FTQuPAPDecisionEngine",
    "DEFAULT_DECISION_POLICY",
    "validate_boolean",
    "validate_nonempty_string",
    "validate_probability",
    "validate_positive_cost",
    "normalize_reasons",
    "policy_from_mapping",
    "normalize_policy",
    "policy_acceptance_rule",
    "fixed_qber_acceptance_rule",
    "build_decision_record",
    "evaluate_gp_decision",
    "evaluate_fixed_qber_decision",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        DecisionEngineError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[DECISION ENGINE ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error