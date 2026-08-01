"""
FT-QuPAP Protocol Result Models

This module defines safe, structured result objects for:

- deterministic protocol verification
- Gaussian Process attack detection
- final authentication decisions
- retry-attempt history
- complete authentication-session results

Security rule:

These result models must never contain:

- ML-KEM secret keys
- ML-DSA secret keys
- ML-KEM shared secrets
- K_auth
- K_ctrl
- raw KMAC authentication tags
- raw subscriber identity
- hidden Eve attack positions
- attacked-mask values

Only non-secret protocol evidence should be stored, displayed,
exported, or written to the session database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_PROTOCOL_NAME = "FT-QuPAP"
DEFAULT_PROTOCOL_VERSION = "FT-QuPAP-1.0"


SENSITIVE_RESULT_KEYS = {
    "secret_key",
    "private_key",
    "mlkem_secret_key",
    "mldsa_secret_key",
    "shared_secret",
    "session_key",
    "k_ss",
    "k_auth",
    "k_ctrl",
    "raw_tag",
    "authentication_tag",
    "subscriber_identity",
    "raw_subscriber_identity",
    "attacked_mask",
    "eve_positions",
    "eve_basis",
}


class ResultModelError(Exception):
    """Base exception for FT-QuPAP result-model failures."""


class InvalidVerificationResultError(ResultModelError):
    """Raised when deterministic verification data is invalid."""


class InvalidAttackDetectionResultError(ResultModelError):
    """Raised when GP detection data is invalid."""


class InvalidDecisionResultError(ResultModelError):
    """Raised when a final decision result is inconsistent."""


class InvalidRetryResultError(ResultModelError):
    """Raised when retry information is invalid."""


class InvalidAuthenticationResultError(ResultModelError):
    """Raised when a complete session result is invalid."""


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
    allow_none: bool = False,
) -> float | None:
    """Validate a probability in the closed interval [0, 1]."""

    if value is None:
        if allow_none:
            return None

        raise TypeError(
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

    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def validate_nonnegative_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a nonnegative integer."""

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field_name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return value


def validate_positive_integer(
    value: Any,
    field_name: str,
) -> int:
    """Validate a positive integer."""

    normalized = validate_nonnegative_integer(
        value,
        field_name,
    )

    if normalized == 0:
        raise ValueError(
            f"{field_name} must be greater than zero."
        )

    return normalized


def normalize_reason_collection(
    reasons: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and normalize deterministic reason strings."""

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
            "reasons must be a sequence of strings."
        )

    if not isinstance(reasons, Sequence):
        raise TypeError(
            "reasons must be a sequence."
        )

    normalized: list[str] = []

    for reason in reasons:
        normalized_reason = validate_nonempty_string(
            reason,
            "reason",
        )

        if normalized_reason not in normalized:
            normalized.append(
                normalized_reason
            )

    return tuple(normalized)


def to_json_safe(
    value: Any,
) -> Any:
    """
    Convert protocol-result values into JSON-safe structures.

    Sensitive mapping fields are removed automatically.
    """

    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "redacted": True,
        }

    if isinstance(value, Mapping):
        safe_mapping: dict[str, Any] = {}

        for raw_key, raw_value in value.items():
            key = str(raw_key)

            if key.lower() in SENSITIVE_RESULT_KEYS:
                continue

            safe_mapping[key] = to_json_safe(
                raw_value
            )

        return safe_mapping

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            to_json_safe(item)
            for item in value
        ]

    if hasattr(value, "as_dict"):
        return to_json_safe(
            value.as_dict()
        )

    if hasattr(value, "tolist"):
        try:
            return to_json_safe(
                value.tolist()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return to_json_safe(
            vars(value)
        )

    return str(value)


@dataclass(frozen=True)
class VerificationResult:
    """
    Deterministic FT-QuPAP verification outcome.

    Authentication can proceed to the calibrated GP policy only when
    every mandatory deterministic condition passes.
    """

    credential_valid: bool
    request_fresh: bool
    replay_safe: bool
    schedule_valid: bool
    check_evidence_sufficient: bool
    required_blocks_correctable: bool
    tag_valid: bool
    loss_policy_valid: bool

    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        boolean_fields = {
            "credential_valid":
                self.credential_valid,
            "request_fresh":
                self.request_fresh,
            "replay_safe":
                self.replay_safe,
            "schedule_valid":
                self.schedule_valid,
            "check_evidence_sufficient":
                self.check_evidence_sufficient,
            "required_blocks_correctable":
                self.required_blocks_correctable,
            "tag_valid":
                self.tag_valid,
            "loss_policy_valid":
                self.loss_policy_valid,
        }

        for field_name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{field_name} must be boolean."
                )

        normalized_reasons = (
            normalize_reason_collection(
                self.reasons
            )
        )

        object.__setattr__(
            self,
            "reasons",
            normalized_reasons,
        )

        if self.deterministic_pass and self.reasons:
            raise InvalidVerificationResultError(
                "A passing verification result cannot "
                "contain deterministic failure reasons."
            )

    @property
    def deterministic_pass(self) -> bool:
        """Return whether all mandatory checks passed."""

        return all(
            (
                self.credential_valid,
                self.request_fresh,
                self.replay_safe,
                self.schedule_valid,
                self.check_evidence_sufficient,
                self.required_blocks_correctable,
                self.tag_valid,
                self.loss_policy_valid,
            )
        )

    @classmethod
    def from_checks(
        cls,
        *,
        credential_valid: bool,
        request_fresh: bool,
        replay_safe: bool,
        schedule_valid: bool,
        check_evidence_sufficient: bool,
        required_blocks_correctable: bool,
        tag_valid: bool,
        loss_policy_valid: bool,
    ) -> "VerificationResult":
        """Create a result and generate standard failure reasons."""

        reasons: list[str] = []

        checks = (
            (
                credential_valid,
                "invalid_server_credential",
            ),
            (
                request_fresh,
                "request_not_fresh",
            ),
            (
                replay_safe,
                "nonce_replay_detected",
            ),
            (
                schedule_valid,
                "invalid_control_schedule",
            ),
            (
                check_evidence_sufficient,
                "insufficient_check_evidence",
            ),
            (
                required_blocks_correctable,
                "required_block_recovery_failed",
            ),
            (
                tag_valid,
                "authentication_tag_mismatch",
            ),
            (
                loss_policy_valid,
                "loss_policy_failed",
            ),
        )

        for passed, reason in checks:
            if not passed:
                reasons.append(reason)

        return cls(
            credential_valid=credential_valid,
            request_fresh=request_fresh,
            replay_safe=replay_safe,
            schedule_valid=schedule_valid,
            check_evidence_sufficient=(
                check_evidence_sufficient
            ),
            required_blocks_correctable=(
                required_blocks_correctable
            ),
            tag_valid=tag_valid,
            loss_policy_valid=loss_policy_valid,
            reasons=tuple(reasons),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic verification evidence."""

        return {
            "credential_valid":
                self.credential_valid,
            "request_fresh":
                self.request_fresh,
            "replay_safe":
                self.replay_safe,
            "schedule_valid":
                self.schedule_valid,
            "check_evidence_sufficient":
                self.check_evidence_sufficient,
            "required_blocks_correctable":
                self.required_blocks_correctable,
            "tag_valid":
                self.tag_valid,
            "loss_policy_valid":
                self.loss_policy_valid,
            "deterministic_pass":
                self.deterministic_pass,
            "deterministic_reasons":
                list(self.reasons),
        }


@dataclass(frozen=True)
class AttackDetectionResult:
    """
    Calibrated GP attack-detection result.

    Hidden Eve simulator values are not included.
    """

    p_attack: float | None
    uncertainty: float | None
    threshold: float | None

    model_available: bool = True
    model_name: str = "GaussianProcessClassifier"
    calibrated: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "p_attack",
            validate_probability(
                self.p_attack,
                "p_attack",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "uncertainty",
            validate_probability(
                self.uncertainty,
                "uncertainty",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "threshold",
            validate_probability(
                self.threshold,
                "threshold",
                allow_none=True,
            ),
        )

        if not isinstance(
            self.model_available,
            bool,
        ):
            raise TypeError(
                "model_available must be boolean."
            )

        if not isinstance(
            self.calibrated,
            bool,
        ):
            raise TypeError(
                "calibrated must be boolean."
            )

        validate_nonempty_string(
            self.model_name,
            "model_name",
        )

        if self.model_available:
            if self.p_attack is None:
                raise InvalidAttackDetectionResultError(
                    "An available model must provide p_attack."
                )

            if self.threshold is None:
                raise InvalidAttackDetectionResultError(
                    "An available model must provide a threshold."
                )

    @property
    def attack_detected(self) -> bool | None:
        """Return the calibrated GP threshold decision."""

        if (
            self.p_attack is None
            or self.threshold is None
        ):
            return None

        return (
            self.p_attack
            >= self.threshold
        )

    @property
    def below_threshold(self) -> bool | None:
        """Return whether attack probability is below threshold."""

        detected = self.attack_detected

        if detected is None:
            return None

        return not detected

    def as_dict(self) -> dict[str, Any]:
        """Return receiver-observable GP evidence."""

        return {
            "p_attack":
                self.p_attack,
            "uncertainty":
                self.uncertainty,
            "gp_attack_threshold":
                self.threshold,
            "attack_detected":
                self.attack_detected,
            "model_available":
                self.model_available,
            "model_name":
                self.model_name,
            "calibrated":
                self.calibrated,
        }


@dataclass(frozen=True)
class DecisionResult:
    """
    Final FT-QuPAP authentication decision.

    Acceptance requires deterministic verification to pass and the
    calibrated policy to consider the session acceptable.
    """

    accepted: bool
    reason: str

    deterministic_pass: bool
    deterministic_reasons: tuple[str, ...]

    p_attack: float | None
    uncertainty: float | None
    gp_attack_threshold: float | None

    retry_recommended: bool = False

    def __post_init__(self) -> None:
        if not isinstance(
            self.accepted,
            bool,
        ):
            raise TypeError(
                "accepted must be boolean."
            )

        validate_nonempty_string(
            self.reason,
            "reason",
        )

        if not isinstance(
            self.deterministic_pass,
            bool,
        ):
            raise TypeError(
                "deterministic_pass must be boolean."
            )

        normalized_reasons = (
            normalize_reason_collection(
                self.deterministic_reasons
            )
        )

        object.__setattr__(
            self,
            "deterministic_reasons",
            normalized_reasons,
        )

        object.__setattr__(
            self,
            "p_attack",
            validate_probability(
                self.p_attack,
                "p_attack",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "uncertainty",
            validate_probability(
                self.uncertainty,
                "uncertainty",
                allow_none=True,
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

        if not isinstance(
            self.retry_recommended,
            bool,
        ):
            raise TypeError(
                "retry_recommended must be boolean."
            )

        if (
            self.deterministic_pass
            and self.deterministic_reasons
        ):
            raise InvalidDecisionResultError(
                "deterministic_pass=True is inconsistent "
                "with deterministic failure reasons."
            )

        if (
            not self.deterministic_pass
            and not self.deterministic_reasons
        ):
            raise InvalidDecisionResultError(
                "A deterministic failure must include "
                "at least one reason."
            )

        if self.accepted and not self.deterministic_pass:
            raise InvalidDecisionResultError(
                "A session cannot be accepted after "
                "deterministic verification failure."
            )

        if self.accepted and self.retry_recommended:
            raise InvalidDecisionResultError(
                "An accepted result cannot recommend retry."
            )

    @classmethod
    def build(
        cls,
        *,
        accepted: bool,
        reason: str,
        verification: VerificationResult,
        attack_detection: AttackDetectionResult | None,
        retry_recommended: bool = False,
    ) -> "DecisionResult":
        """Build a decision from verification and GP evidence."""

        if not isinstance(
            verification,
            VerificationResult,
        ):
            raise TypeError(
                "verification must be VerificationResult."
            )

        if (
            attack_detection is not None
            and not isinstance(
                attack_detection,
                AttackDetectionResult,
            )
        ):
            raise TypeError(
                "attack_detection must be "
                "AttackDetectionResult or None."
            )

        return cls(
            accepted=accepted,
            reason=reason,
            deterministic_pass=(
                verification.deterministic_pass
            ),
            deterministic_reasons=(
                verification.reasons
            ),
            p_attack=(
                None
                if attack_detection is None
                else attack_detection.p_attack
            ),
            uncertainty=(
                None
                if attack_detection is None
                else attack_detection.uncertainty
            ),
            gp_attack_threshold=(
                None
                if attack_detection is None
                else attack_detection.threshold
            ),
            retry_recommended=(
                retry_recommended
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """
        Return the notebook-compatible decision record.
        """

        return {
            "accepted":
                self.accepted,
            "reason":
                self.reason,
            "deterministic_pass":
                self.deterministic_pass,
            "deterministic_reasons":
                list(
                    self.deterministic_reasons
                ),
            "p_attack":
                self.p_attack,
            "uncertainty":
                self.uncertainty,
            "gp_attack_threshold":
                self.gp_attack_threshold,
            "retry_recommended":
                self.retry_recommended,
        }


@dataclass(frozen=True)
class RetryAttemptResult:
    """
    Non-secret summary of one authentication attempt.
    """

    attempt_number: int
    session_id: str

    accepted: bool
    reason: str
    deterministic_pass: bool

    qber_raw: float | None
    p_attack: float | None
    loss_rate: float | None

    tag_recovered: bool
    retryable: bool
    retry_reason: str | None = None

    def __post_init__(self) -> None:
        validate_positive_integer(
            self.attempt_number,
            "attempt_number",
        )

        validate_nonempty_string(
            self.session_id,
            "session_id",
        )

        if not isinstance(
            self.accepted,
            bool,
        ):
            raise TypeError(
                "accepted must be boolean."
            )

        validate_nonempty_string(
            self.reason,
            "reason",
        )

        if not isinstance(
            self.deterministic_pass,
            bool,
        ):
            raise TypeError(
                "deterministic_pass must be boolean."
            )

        object.__setattr__(
            self,
            "qber_raw",
            validate_probability(
                self.qber_raw,
                "qber_raw",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "p_attack",
            validate_probability(
                self.p_attack,
                "p_attack",
                allow_none=True,
            ),
        )

        object.__setattr__(
            self,
            "loss_rate",
            validate_probability(
                self.loss_rate,
                "loss_rate",
                allow_none=True,
            ),
        )

        if not isinstance(
            self.tag_recovered,
            bool,
        ):
            raise TypeError(
                "tag_recovered must be boolean."
            )

        if not isinstance(
            self.retryable,
            bool,
        ):
            raise TypeError(
                "retryable must be boolean."
            )

        if self.retry_reason is not None:
            validate_nonempty_string(
                self.retry_reason,
                "retry_reason",
            )

        if self.accepted and self.retryable:
            raise InvalidRetryResultError(
                "An accepted attempt cannot be retryable."
            )

        if self.retryable and self.retry_reason is None:
            raise InvalidRetryResultError(
                "A retryable attempt must include retry_reason."
            )

    def as_dict(self) -> dict[str, Any]:
        """Return an attempt-history dictionary."""

        return {
            "attempt_number":
                self.attempt_number,
            "session_id":
                self.session_id,
            "accepted":
                self.accepted,
            "reason":
                self.reason,
            "deterministic_pass":
                self.deterministic_pass,
            "qber_raw":
                self.qber_raw,
            "p_attack":
                self.p_attack,
            "loss_rate":
                self.loss_rate,
            "tag_recovered":
                self.tag_recovered,
            "retryable":
                self.retryable,
            "retry_reason":
                self.retry_reason,
        }


@dataclass
class AuthenticationResult:
    """
    Complete non-secret FT-QuPAP authentication result.

    This is the main result object returned by protocol_engine.py.
    """

    session_id: str
    pseudonym_id: str

    decision: DecisionResult
    verification: VerificationResult
    attack_detection: AttackDetectionResult | None

    qber_raw: float | None = None
    qber_mismatches: int = 0
    qber_observed: int = 0
    observed_check_blocks: int = 0

    mean_syndrome_weight: float | None = None
    max_syndrome_weight: float | None = None
    correction_failure_rate: float | None = None
    loss_rate: float | None = None

    tag_recovered: bool = False
    physical_qubits: int = 0

    retry_attempts: int = 1
    attempt_history: list[RetryAttemptResult] = field(
        default_factory=list
    )

    channel_name: str = "unknown"
    channel_context: str = "unknown"

    timings: dict[str, float] = field(
        default_factory=dict
    )

    features: dict[str, float] = field(
        default_factory=dict
    )

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )

    protocol_name: str = DEFAULT_PROTOCOL_NAME
    protocol_version: str = DEFAULT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        self.session_id = validate_nonempty_string(
            self.session_id,
            "session_id",
        )

        self.pseudonym_id = validate_nonempty_string(
            self.pseudonym_id,
            "pseudonym_id",
        )

        if not isinstance(
            self.decision,
            DecisionResult,
        ):
            raise TypeError(
                "decision must be DecisionResult."
            )

        if not isinstance(
            self.verification,
            VerificationResult,
        ):
            raise TypeError(
                "verification must be VerificationResult."
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

        self.qber_raw = validate_probability(
            self.qber_raw,
            "qber_raw",
            allow_none=True,
        )

        self.loss_rate = validate_probability(
            self.loss_rate,
            "loss_rate",
            allow_none=True,
        )

        self.correction_failure_rate = (
            validate_probability(
                self.correction_failure_rate,
                "correction_failure_rate",
                allow_none=True,
            )
        )

        self.qber_mismatches = (
            validate_nonnegative_integer(
                self.qber_mismatches,
                "qber_mismatches",
            )
        )

        self.qber_observed = (
            validate_nonnegative_integer(
                self.qber_observed,
                "qber_observed",
            )
        )

        self.observed_check_blocks = (
            validate_nonnegative_integer(
                self.observed_check_blocks,
                "observed_check_blocks",
            )
        )

        self.physical_qubits = (
            validate_nonnegative_integer(
                self.physical_qubits,
                "physical_qubits",
            )
        )

        self.retry_attempts = (
            validate_positive_integer(
                self.retry_attempts,
                "retry_attempts",
            )
        )

        if self.qber_mismatches > self.qber_observed:
            raise InvalidAuthenticationResultError(
                "qber_mismatches cannot exceed qber_observed."
            )

        if not isinstance(
            self.tag_recovered,
            bool,
        ):
            raise TypeError(
                "tag_recovered must be boolean."
            )

        if self.decision.accepted != self.accepted:
            raise InvalidAuthenticationResultError(
                "Authentication acceptance does not match "
                "the final decision."
            )

        if (
            self.verification.deterministic_pass
            != self.decision.deterministic_pass
        ):
            raise InvalidAuthenticationResultError(
                "Verification and decision deterministic "
                "status values do not match."
            )

        if len(self.attempt_history) > self.retry_attempts:
            raise InvalidRetryResultError(
                "attempt_history contains more entries "
                "than retry_attempts."
            )

        for attempt in self.attempt_history:
            if not isinstance(
                attempt,
                RetryAttemptResult,
            ):
                raise TypeError(
                    "attempt_history items must be "
                    "RetryAttemptResult objects."
                )

        self.channel_name = (
            validate_nonempty_string(
                self.channel_name,
                "channel_name",
            )
        )

        self.channel_context = (
            validate_nonempty_string(
                self.channel_context,
                "channel_context",
            )
        )

        self.protocol_name = (
            validate_nonempty_string(
                self.protocol_name,
                "protocol_name",
            )
        )

        self.protocol_version = (
            validate_nonempty_string(
                self.protocol_version,
                "protocol_version",
            )
        )

        self.timings = self._validate_numeric_mapping(
            self.timings,
            "timings",
            require_nonnegative=True,
        )

        self.features = self._validate_numeric_mapping(
            self.features,
            "features",
            require_nonnegative=False,
        )

        self.diagnostics = dict(
            to_json_safe(
                self.diagnostics
            )
        )

    @staticmethod
    def _validate_numeric_mapping(
        value: Mapping[str, Any],
        field_name: str,
        require_nonnegative: bool,
    ) -> dict[str, float]:
        """Validate a string-to-number mapping."""

        if not isinstance(
            value,
            Mapping,
        ):
            raise TypeError(
                f"{field_name} must be a mapping."
            )

        normalized: dict[str, float] = {}

        for raw_key, raw_value in value.items():
            key = validate_nonempty_string(
                str(raw_key),
                f"{field_name} key",
            )

            if isinstance(
                raw_value,
                bool,
            ) or not isinstance(
                raw_value,
                (int, float),
            ):
                raise TypeError(
                    f"{field_name}[{key!r}] must be numeric."
                )

            number = float(
                raw_value
            )

            if require_nonnegative and number < 0:
                raise ValueError(
                    f"{field_name}[{key!r}] cannot be negative."
                )

            normalized[key] = number

        return normalized

    @property
    def accepted(self) -> bool:
        """Return the final authentication status."""

        return self.decision.accepted

    @property
    def reason(self) -> str:
        """Return the final decision reason."""

        return self.decision.reason

    @property
    def retry_used(self) -> bool:
        """Return whether more than one attempt was executed."""

        return self.retry_attempts > 1

    @property
    def deterministic_pass(self) -> bool:
        """Return deterministic verification status."""

        return self.verification.deterministic_pass

    @property
    def p_attack(self) -> float | None:
        """Return calibrated attack probability."""

        if self.attack_detection is None:
            return None

        return self.attack_detection.p_attack

    def compact_summary(self) -> dict[str, Any]:
        """Return dashboard and session-history fields."""

        return {
            "session_id":
                self.session_id,
            "pseudonym_id":
                self.pseudonym_id,
            "accepted":
                self.accepted,
            "reason":
                self.reason,
            "deterministic_pass":
                self.deterministic_pass,
            "deterministic_reasons":
                list(
                    self.verification.reasons
                ),
            "qber_raw":
                self.qber_raw,
            "p_attack":
                self.p_attack,
            "loss_rate":
                self.loss_rate,
            "tag_recovered":
                self.tag_recovered,
            "physical_qubits":
                self.physical_qubits,
            "retry_attempts":
                self.retry_attempts,
            "retry_used":
                self.retry_used,
            "channel_name":
                self.channel_name,
            "channel_context":
                self.channel_context,
            "protocol_name":
                self.protocol_name,
            "protocol_version":
                self.protocol_version,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe authentication result."""

        result = {
            **self.compact_summary(),
            "qber_mismatches":
                self.qber_mismatches,
            "qber_observed":
                self.qber_observed,
            "observed_check_blocks":
                self.observed_check_blocks,
            "mean_syndrome_weight":
                self.mean_syndrome_weight,
            "max_syndrome_weight":
                self.max_syndrome_weight,
            "correction_failure_rate":
                self.correction_failure_rate,
            "decision":
                self.decision.as_dict(),
            "verification":
                self.verification.as_dict(),
            "attack_detection": (
                None
                if self.attack_detection is None
                else self.attack_detection.as_dict()
            ),
            "attempt_history": [
                attempt.as_dict()
                for attempt in self.attempt_history
            ],
            "timings":
                dict(self.timings),
            "features":
                dict(self.features),
            "diagnostics":
                dict(self.diagnostics),
        }

        return to_json_safe(
            result
        )


def build_rejected_result(
    *,
    session_id: str,
    pseudonym_id: str,
    reason: str,
    deterministic_reasons: Sequence[str],
    channel_name: str = "unknown",
    channel_context: str = "unknown",
) -> AuthenticationResult:
    """
    Build an early deterministic rejection result.

    This helper is useful for replay, freshness, credential, malformed
    ciphertext, and invalid-schedule rejection paths.
    """

    normalized_reasons = (
        normalize_reason_collection(
            deterministic_reasons
        )
    )

    if not normalized_reasons:
        normalized_reasons = (
            reason,
        )

    verification = VerificationResult(
        credential_valid=False,
        request_fresh=False,
        replay_safe=False,
        schedule_valid=False,
        check_evidence_sufficient=False,
        required_blocks_correctable=False,
        tag_valid=False,
        loss_policy_valid=False,
        reasons=normalized_reasons,
    )

    decision = DecisionResult(
        accepted=False,
        reason=reason,
        deterministic_pass=False,
        deterministic_reasons=(
            normalized_reasons
        ),
        p_attack=None,
        uncertainty=None,
        gp_attack_threshold=None,
        retry_recommended=False,
    )

    return AuthenticationResult(
        session_id=session_id,
        pseudonym_id=pseudonym_id,
        decision=decision,
        verification=verification,
        attack_detection=None,
        channel_name=channel_name,
        channel_context=channel_context,
    )


def run_self_test() -> None:
    """Verify result consistency and JSON-safe export."""

    verification = VerificationResult.from_checks(
        credential_valid=True,
        request_fresh=True,
        replay_safe=True,
        schedule_valid=True,
        check_evidence_sufficient=True,
        required_blocks_correctable=True,
        tag_valid=True,
        loss_policy_valid=True,
    )

    attack_detection = AttackDetectionResult(
        p_attack=0.05,
        uncertainty=0.10,
        threshold=0.25,
        model_available=True,
        calibrated=True,
    )

    decision = DecisionResult.build(
        accepted=True,
        reason=(
            "accepted_by_calibrated_bayesian_policy"
        ),
        verification=verification,
        attack_detection=attack_detection,
    )

    attempt = RetryAttemptResult(
        attempt_number=1,
        session_id="SESSION-TEST-001",
        accepted=True,
        reason=decision.reason,
        deterministic_pass=True,
        qber_raw=0.01,
        p_attack=0.05,
        loss_rate=0.0,
        tag_recovered=True,
        retryable=False,
    )

    result = AuthenticationResult(
        session_id="SESSION-TEST-001",
        pseudonym_id="PID-TEST-001",
        decision=decision,
        verification=verification,
        attack_detection=attack_detection,
        qber_raw=0.01,
        qber_mismatches=2,
        qber_observed=224,
        observed_check_blocks=32,
        mean_syndrome_weight=0.05,
        max_syndrome_weight=1.0,
        correction_failure_rate=0.0,
        loss_rate=0.0,
        tag_recovered=True,
        physical_qubits=1120,
        retry_attempts=1,
        attempt_history=[attempt],
        channel_name="ideal",
        channel_context="urban",
        timings={
            "end_to_end_s": 0.25,
        },
        features={
            "qber_raw": 0.01,
            "loss_rate": 0.0,
        },
        diagnostics={
            "safe_note": "self-test",
            "k_auth": b"must-not-appear",
        },
    )

    exported = result.as_dict()

    if not result.accepted:
        raise ResultModelError(
            "Self-test result should be accepted."
        )

    if result.retry_used:
        raise ResultModelError(
            "Single-attempt result incorrectly reports retry."
        )

    if exported["physical_qubits"] != 1120:
        raise ResultModelError(
            "Physical-qubit count was not preserved."
        )

    if "k_auth" in exported["diagnostics"]:
        raise ResultModelError(
            "Sensitive diagnostic field was not removed."
        )

    print(
        "Result models self-test completed successfully."
    )


__all__ = [
    "DEFAULT_PROTOCOL_NAME",
    "DEFAULT_PROTOCOL_VERSION",
    "SENSITIVE_RESULT_KEYS",
    "ResultModelError",
    "InvalidVerificationResultError",
    "InvalidAttackDetectionResultError",
    "InvalidDecisionResultError",
    "InvalidRetryResultError",
    "InvalidAuthenticationResultError",
    "VerificationResult",
    "AttackDetectionResult",
    "DecisionResult",
    "RetryAttemptResult",
    "AuthenticationResult",
    "validate_nonempty_string",
    "validate_probability",
    "validate_nonnegative_integer",
    "validate_positive_integer",
    "normalize_reason_collection",
    "to_json_safe",
    "build_rejected_result",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ResultModelError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[RESULT MODEL ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error