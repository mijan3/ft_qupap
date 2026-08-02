"""
Status-card components for the FT-QuPAP Streamlit dashboard.

This module converts non-secret protocol-session data into concise visual
cards for:

- Mobile Station readiness
- Authentication Server readiness
- post-quantum bootstrap state
- quantum channel and Steane CSS state
- deterministic security checks
- calibrated GP detector state
- retry-policy state
- final authentication decision
- optional ESP32/Arduino indicator state

The card derivation is deliberately evidence-driven. Missing values are shown
as unavailable or inactive; they are never silently treated as successful.

Expected session shape
----------------------
The module accepts a flat or nested mapping. A typical completed session is:

    {
        "decision": {
            "accepted": True,
            "reason": "authentication_successful",
            "deterministic_pass": True,
            "deterministic_reasons": [],
            "p_attack": 0.04,
            "uncertainty": 0.08,
            "gp_attack_threshold": 0.15,
        },
        "credential_valid": True,
        "freshness_valid": True,
        "replay_detected": False,
        "mlkem_decapsulation_success": True,
        "schedule_valid": True,
        "decoder_success": True,
        "tag_recovered": True,
        "loss_policy_pass": True,
        "qber_raw": 0.02,
        "loss_rate": 0.01,
        "observed_check_blocks": 32,
        "required_check_blocks": 24,
        "retry_attempts": 1,
        "retry_used": False,
        "physical_qubits": 1120,
        "timings": {"end_to_end_s": 0.25},
        "hardware_status": "GREEN",
    }

Security boundary
-----------------
Only non-secret Boolean states, probabilities, rates, counts, reasons, and
timings are displayed. Private keys, shared secrets, K_auth, K_ctrl, raw tags,
raw identities, signatures, and ciphertexts are never rendered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Any, Final, Mapping, Sequence

from .components import (
    MetricItem,
    format_duration,
    format_percentage,
    format_probability,
    format_value,
    render_metric_grid,
    sanitize_mapping,
)
from .theme import (
    THEME,
    normalize_status,
    status_badge_html,
    status_style,
)


DEFAULT_PLACEHOLDER: Final[str] = "—"

DEFAULT_FIXED_QBER_THRESHOLD: Final[float] = 0.11
DEFAULT_MAXIMUM_LOSS_RATE: Final[float] = 0.15
DEFAULT_OPERATIONAL_GP_THRESHOLD: Final[float] = 0.15
DEFAULT_RETRY_UPPER_PROBABILITY: Final[float] = 0.20
DEFAULT_REQUIRED_CHECK_BLOCKS: Final[int] = 24
DEFAULT_TOTAL_PHYSICAL_QUBITS: Final[int] = 1120


class StatusCardError(ValueError):
    """Raised when card configuration is invalid."""


@dataclass(frozen=True)
class StatusCard:
    """One high-level protocol status card."""

    key: str
    title: str
    status: str
    value: str
    subtitle: str = ""
    icon: str = ""
    detail: str = ""
    progress: float | None = None


@dataclass(frozen=True)
class SecurityCheckCard:
    """One deterministic or probabilistic security check."""

    key: str
    label: str
    status: str
    summary: str
    icon: str = ""
    value: str = ""


@dataclass(frozen=True)
class ProtocolThresholds:
    """Runtime thresholds used for card derivation."""

    fixed_qber_threshold: float = DEFAULT_FIXED_QBER_THRESHOLD
    maximum_loss_rate: float = DEFAULT_MAXIMUM_LOSS_RATE
    operational_gp_threshold: float = DEFAULT_OPERATIONAL_GP_THRESHOLD
    retry_upper_probability: float = DEFAULT_RETRY_UPPER_PROBABILITY
    required_check_blocks: int = DEFAULT_REQUIRED_CHECK_BLOCKS

    def validate(self) -> None:
        for name, value in (
            ("fixed_qber_threshold", self.fixed_qber_threshold),
            ("maximum_loss_rate", self.maximum_loss_rate),
            (
                "operational_gp_threshold",
                self.operational_gp_threshold,
            ),
            (
                "retry_upper_probability",
                self.retry_upper_probability,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise StatusCardError(
                    f"{name} must be a finite value in [0, 1]."
                )

        if self.required_check_blocks < 1:
            raise StatusCardError(
                "required_check_blocks must be positive."
            )


@dataclass(frozen=True)
class ProtocolStatusSummary:
    """Card-ready summary of one protocol session."""

    status_cards: tuple[StatusCard, ...]
    security_checks: tuple[SecurityCheckCard, ...]
    final_status: str
    final_reason: str
    accepted: bool | None
    retry_used: bool
    completed_check_count: int
    total_check_count: int


def _streamlit() -> Any:
    """Import Streamlit only when rendering is requested."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required for status-card rendering."
        ) from exc

    return st


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping."""

    return value if isinstance(value, Mapping) else {}


def _first_present(
    mapping: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first present non-None field."""

    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]

    return default


def _nested_first(
    payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = None,
) -> Any:
    """Resolve the first available nested field path."""

    for path in paths:
        current: Any = payload
        found = True

        for segment in path:
            if not isinstance(current, Mapping) or segment not in current:
                found = False
                break
            current = current[segment]

        if found and current is not None:
            return current

    return default


def _optional_bool(value: Any) -> bool | None:
    """Convert common Boolean values to a tri-state Boolean."""

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "y",
        "pass",
        "passed",
        "valid",
        "success",
        "successful",
        "verified",
        "accepted",
        "green",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "n",
        "fail",
        "failed",
        "invalid",
        "rejected",
        "red",
    }:
        return False

    return None


def _optional_float(value: Any) -> float | None:
    """Return a finite floating-point value or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    """Return an integer or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_for_boolean(
    value: bool | None,
    *,
    true_status: str = "verified",
    false_status: str = "failed",
) -> str:
    """Map a tri-state Boolean to a semantic status."""

    if value is True:
        return true_status

    if value is False:
        return false_status

    return "unavailable"


def _yes_no_unknown(value: bool | None) -> str:
    """Format a tri-state Boolean."""

    if value is True:
        return "Yes"

    if value is False:
        return "No"

    return DEFAULT_PLACEHOLDER


def _clamp_progress(value: float | None) -> float | None:
    """Clamp a progress value to [0, 1]."""

    if value is None or not math.isfinite(value):
        return None

    return min(max(value, 0.0), 1.0)


def _decision_mapping(session: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the final decision mapping."""

    return _mapping(session.get("decision"))


def resolve_thresholds(
    session: Mapping[str, Any],
    overrides: ProtocolThresholds | None = None,
) -> ProtocolThresholds:
    """Resolve deployed thresholds from session data with safe defaults."""

    if overrides is not None:
        overrides.validate()
        return overrides

    decision = _decision_mapping(session)
    policy = _mapping(
        _first_present(
            session,
            "threshold_policy",
            "security_policy",
            default={},
        )
    )

    thresholds = ProtocolThresholds(
        fixed_qber_threshold=(
            _optional_float(
                _first_present(
                    policy,
                    "fixed_qber_threshold",
                    default=session.get("fixed_qber_threshold"),
                )
            )
            or DEFAULT_FIXED_QBER_THRESHOLD
        ),
        maximum_loss_rate=(
            _optional_float(
                _first_present(
                    policy,
                    "maximum_loss_rate",
                    "max_acceptable_loss_rate",
                    default=session.get("maximum_loss_rate"),
                )
            )
            or DEFAULT_MAXIMUM_LOSS_RATE
        ),
        operational_gp_threshold=(
            _optional_float(
                _first_present(
                    decision,
                    "gp_attack_threshold",
                    "operational_threshold",
                    default=_first_present(
                        policy,
                        "operational_threshold",
                        "gp_attack_threshold",
                        default=session.get(
                            "gp_attack_threshold"
                        ),
                    ),
                )
            )
            or DEFAULT_OPERATIONAL_GP_THRESHOLD
        ),
        retry_upper_probability=(
            _optional_float(
                _first_present(
                    policy,
                    "retry_upper_probability",
                    "gp_gray_zone_retry_upper",
                    default=session.get(
                        "retry_upper_probability"
                    ),
                )
            )
            or DEFAULT_RETRY_UPPER_PROBABILITY
        ),
        required_check_blocks=(
            _optional_int(
                _first_present(
                    session,
                    "required_check_blocks",
                    "minimum_observed_check_blocks",
                    default=_first_present(
                        policy,
                        "minimum_observed_check_blocks",
                        default=DEFAULT_REQUIRED_CHECK_BLOCKS,
                    ),
                )
            )
            or DEFAULT_REQUIRED_CHECK_BLOCKS
        ),
    )
    thresholds.validate()
    return thresholds


def resolve_final_decision(
    session: Mapping[str, Any],
) -> tuple[bool | None, bool, str, str]:
    """
    Resolve accepted/retry/rejected state.

    Returns
    -------
    accepted, retry_used, status, reason
    """

    decision = _decision_mapping(session)

    accepted = _optional_bool(
        _first_present(
            decision,
            "accepted",
            default=session.get("accepted"),
        )
    )
    retry_used = bool(
        _optional_bool(
            _first_present(
                session,
                "retry_used",
                default=decision.get("retry_used"),
            )
        )
        or False
    )
    reason = str(
        _first_present(
            decision,
            "reason",
            default=_first_present(
                session,
                "reason",
                default="No decision available",
            ),
        )
    )

    execution_status = str(
        session.get("execution_status", "")
    ).strip().lower()

    if execution_status == "error":
        return False, retry_used, "failed", reason

    if accepted is True and retry_used:
        status = "retry"
    elif accepted is True:
        status = "accepted"
    elif accepted is False:
        status = "rejected"
    else:
        status = "inactive"

    return accepted, retry_used, status, reason


def _resolve_mobile_station_state(
    session: Mapping[str, Any],
) -> StatusCard:
    """Build the Mobile Station summary card."""

    explicit_status = _first_present(
        session,
        "mobile_station_status",
        "ms_status",
    )

    request_ready = _optional_bool(
        _nested_first(
            session,
            (
                ("mobile_station", "request_prepared"),
                ("request_prepared",),
                ("authentication_request_created",),
            ),
        )
    )
    credential_valid = _optional_bool(
        _nested_first(
            session,
            (
                ("mobile_station", "credential_valid"),
                ("credential_valid",),
                ("server_signature_valid",),
            ),
        )
    )
    token_prepared = _optional_bool(
        _nested_first(
            session,
            (
                ("mobile_station", "quantum_token_prepared"),
                ("quantum_token_prepared",),
                ("steane_encoding_success",),
            ),
        )
    )

    known = [
        value
        for value in (
            request_ready,
            credential_valid,
            token_prepared,
        )
        if value is not None
    ]

    if explicit_status is not None:
        status = normalize_status(explicit_status)
    elif any(value is False for value in known):
        status = "failed"
    elif known and all(value is True for value in known):
        status = "verified"
    elif known:
        status = "running"
    else:
        status = "inactive"

    completed = sum(value is True for value in known)
    total = len(known)
    progress = completed / total if total else None

    return StatusCard(
        key="mobile_station",
        title="Mobile Station",
        status=status,
        value=(
            f"{completed}/{total} checks"
            if total
            else "Awaiting session"
        ),
        subtitle=(
            "Request, server credential, and quantum-token preparation"
        ),
        icon="📱",
        detail=(
            "No raw identity, secret key, or authentication tag is shown."
        ),
        progress=progress,
    )


def _resolve_authentication_server_state(
    session: Mapping[str, Any],
) -> StatusCard:
    """Build the Authentication Server summary card."""

    accepted, retry_used, final_status, reason = (
        resolve_final_decision(session)
    )

    explicit_status = _first_present(
        session,
        "authentication_server_status",
        "server_status",
        "as_status",
    )

    if explicit_status is not None:
        status = normalize_status(explicit_status)
    elif accepted is not None:
        status = final_status
    else:
        deterministic_pass = _optional_bool(
            _nested_first(
                session,
                (
                    ("decision", "deterministic_pass"),
                    ("deterministic_pass",),
                ),
            )
        )
        status = _status_for_boolean(
            deterministic_pass,
            true_status="running",
        )

    return StatusCard(
        key="authentication_server",
        title="Authentication Server",
        status=status,
        value=(
            "Accepted after retry"
            if accepted is True and retry_used
            else (
                "Accepted"
                if accepted is True
                else (
                    "Rejected"
                    if accepted is False
                    else "Awaiting decision"
                )
            )
        ),
        subtitle="Freshness, replay, decoder, KMAC, GP, and policy",
        icon="🛡️",
        detail=reason,
        progress=1.0 if accepted is not None else None,
    )


def _resolve_pqc_state(
    session: Mapping[str, Any],
) -> StatusCard:
    """Build the post-quantum bootstrap summary card."""

    credential = _optional_bool(
        _nested_first(
            session,
            (
                ("credential_valid",),
                ("server_signature_valid",),
                ("bootstrap", "credential_valid"),
            ),
        )
    )
    encapsulation = _optional_bool(
        _nested_first(
            session,
            (
                ("mlkem_encapsulation_success",),
                ("bootstrap", "encapsulation_success"),
                ("mobile_station", "mlkem_encapsulation_success"),
            ),
        )
    )
    decapsulation = _optional_bool(
        _nested_first(
            session,
            (
                ("mlkem_decapsulation_success",),
                ("bootstrap", "decapsulation_success"),
                (
                    "authentication_server",
                    "mlkem_decapsulation_success",
                ),
            ),
        )
    )
    key_match = _optional_bool(
        _nested_first(
            session,
            (
                ("shared_secret_match",),
                ("session_secret_match",),
                ("bootstrap", "shared_secret_match"),
            ),
        )
    )

    checks = [
        value
        for value in (
            credential,
            encapsulation,
            decapsulation,
            key_match,
        )
        if value is not None
    ]

    if any(value is False for value in checks):
        status = "failed"
    elif checks and all(value is True for value in checks):
        status = "verified"
    elif checks:
        status = "running"
    else:
        status = "inactive"

    completed = sum(value is True for value in checks)

    return StatusCard(
        key="pqc_bootstrap",
        title="PQC Bootstrap",
        status=status,
        value=(
            f"{completed}/{len(checks)} verified"
            if checks
            else "Not started"
        ),
        subtitle="ML-DSA-65 credential and ML-KEM-768 fresh session",
        icon="🔐",
        detail=(
            "Secret keys and shared-session material remain hidden."
        ),
        progress=(
            completed / len(checks)
            if checks
            else None
        ),
    )


def _resolve_quantum_state(
    session: Mapping[str, Any],
    thresholds: ProtocolThresholds,
) -> StatusCard:
    """Build the quantum-channel and CSS summary card."""

    qber = _optional_float(
        _first_present(
            session,
            "qber_raw",
            "raw_qber",
            "qber",
        )
    )
    loss_rate = _optional_float(
        _first_present(
            session,
            "loss_rate",
            "observed_loss_rate",
        )
    )
    observed = _optional_int(
        _first_present(
            session,
            "observed_check_blocks",
            "qber_observed",
        )
    )
    physical_qubits = _optional_int(
        _first_present(
            session,
            "physical_qubits",
            default=DEFAULT_TOTAL_PHYSICAL_QUBITS,
        )
    )
    decoder_success = _optional_bool(
        _first_present(
            session,
            "decoder_success",
            "css_decoder_success",
        )
    )

    qber_pass = (
        None
        if qber is None
        else qber <= thresholds.fixed_qber_threshold
    )
    loss_pass = (
        None
        if loss_rate is None
        else loss_rate <= thresholds.maximum_loss_rate
    )
    observed_pass = (
        None
        if observed is None
        else observed >= thresholds.required_check_blocks
    )

    checks = [
        value
        for value in (
            qber_pass,
            loss_pass,
            observed_pass,
            decoder_success,
        )
        if value is not None
    ]

    if any(value is False for value in checks):
        status = "warning" if decoder_success is not False else "failed"
    elif checks and all(value is True for value in checks):
        status = "quantum"
    elif checks:
        status = "running"
    else:
        status = "inactive"

    value_parts = []

    if qber is not None:
        value_parts.append(f"QBER {format_percentage(qber)}")

    if loss_rate is not None:
        value_parts.append(f"Loss {format_percentage(loss_rate)}")

    value_text = " · ".join(value_parts) or "Awaiting measurement"

    detail_parts = []

    if observed is not None:
        detail_parts.append(
            f"{observed}/{thresholds.required_check_blocks} "
            "required check blocks observed"
        )

    if physical_qubits is not None:
        detail_parts.append(
            f"{physical_qubits:,} Steane data qubits"
        )

    progress = (
        observed / thresholds.required_check_blocks
        if observed is not None
        else None
    )

    return StatusCard(
        key="quantum_channel",
        title="Quantum Channel",
        status=status,
        value=value_text,
        subtitle="Steane [[7,1,3]], loss, raw QBER, and decoding",
        icon="⚛️",
        detail=" · ".join(detail_parts),
        progress=_clamp_progress(progress),
    )


def _resolve_gp_state(
    session: Mapping[str, Any],
    thresholds: ProtocolThresholds,
) -> StatusCard:
    """Build the calibrated GP detector summary card."""

    decision = _decision_mapping(session)

    p_attack = _optional_float(
        _first_present(
            decision,
            "p_attack",
            default=_first_present(
                session,
                "p_attack",
                "attack_probability",
            ),
        )
    )
    uncertainty = _optional_float(
        _first_present(
            decision,
            "uncertainty",
            default=session.get("uncertainty"),
        )
    )
    model_ready = _optional_bool(
        _nested_first(
            session,
            (
                ("model_ready",),
                ("gp_model_ready",),
                ("model_validation", "valid"),
            ),
        )
    )

    if p_attack is None:
        status = (
            _status_for_boolean(
                model_ready,
                true_status="ready",
                false_status="failed",
            )
            if model_ready is not None
            else "inactive"
        )
    elif p_attack < thresholds.operational_gp_threshold:
        status = "verified"
    elif p_attack < thresholds.retry_upper_probability:
        status = "gray_zone"
    else:
        status = "attack"

    detail_parts = [
        (
            "Operational threshold "
            f"{format_probability(thresholds.operational_gp_threshold)}"
        )
    ]

    if uncertainty is not None:
        detail_parts.append(
            f"uncertainty {format_probability(uncertainty)}"
        )

    return StatusCard(
        key="gp_detector",
        title="GP Detector",
        status=status,
        value=(
            f"P(attack) {format_probability(p_attack)}"
            if p_attack is not None
            else "Awaiting features"
        ),
        subtitle="Nine observable features with isotonic calibration",
        icon="🧠",
        detail=" · ".join(detail_parts),
        progress=_clamp_progress(p_attack),
    )


def _resolve_retry_state(
    session: Mapping[str, Any],
) -> StatusCard:
    """Build the bounded fresh-retry summary card."""

    retry_used = bool(
        _optional_bool(session.get("retry_used"))
        or False
    )
    attempts = _optional_int(
        _first_present(
            session,
            "retry_attempts",
            "attempt_count",
            "attempts",
            default=1,
        )
    ) or 1
    max_attempts = _optional_int(
        _nested_first(
            session,
            (
                ("maximum_authentication_attempts",),
                (
                    "threshold_policy",
                    "maximum_authentication_attempts",
                ),
            ),
            default=3,
        )
    ) or 3

    accepted, _, _, reason = resolve_final_decision(session)

    if retry_used and accepted is True:
        status = "retry"
        value = f"Accepted on attempt {attempts}"
    elif retry_used and accepted is False:
        status = "rejected"
        value = f"Rejected after {attempts} attempts"
    elif retry_used:
        status = "running"
        value = f"Attempt {attempts}/{max_attempts}"
    else:
        status = "inactive"
        value = "Retry not used"

    progress = attempts / max_attempts if max_attempts > 0 else None

    return StatusCard(
        key="retry_policy",
        title="Retry Policy",
        status=status,
        value=value,
        subtitle="Fresh nonce, request, ML-KEM session, tag, and payload",
        icon="🔁",
        detail=reason if retry_used else "Maximum two retries.",
        progress=_clamp_progress(progress),
    )


def _resolve_hardware_state(
    session: Mapping[str, Any],
) -> StatusCard:
    """Build the optional external indicator card."""

    hardware = _first_present(
        session,
        "hardware_status",
        "led_status",
        default="OFF",
    )
    normalized = str(hardware).strip().upper()

    if "GREEN" in normalized:
        status = "accepted"
    elif "YELLOW" in normalized:
        status = "retry"
    elif "RED" in normalized:
        status = "rejected"
    elif normalized in {"OFF", "DISABLED", "NONE", ""}:
        status = "inactive"
    elif "FALLBACK" in normalized:
        status = "warning"
    else:
        status = "ready"

    return StatusCard(
        key="hardware",
        title="Hardware Indicator",
        status=status,
        value=normalized or "OFF",
        subtitle="Optional ESP32/Arduino or software fallback",
        icon="💡",
        detail=(
            "GREEN = accepted, YELLOW = retry, RED = rejected"
        ),
        progress=None,
    )


def build_status_cards(
    session: Mapping[str, Any] | None,
    *,
    thresholds: ProtocolThresholds | None = None,
    include_hardware: bool = True,
) -> tuple[StatusCard, ...]:
    """Build the high-level status-card set."""

    safe_session = _mapping(sanitize_mapping(session or {}))
    resolved_thresholds = resolve_thresholds(
        safe_session,
        thresholds,
    )

    cards = [
        _resolve_mobile_station_state(safe_session),
        _resolve_authentication_server_state(safe_session),
        _resolve_pqc_state(safe_session),
        _resolve_quantum_state(
            safe_session,
            resolved_thresholds,
        ),
        _resolve_gp_state(
            safe_session,
            resolved_thresholds,
        ),
        _resolve_retry_state(safe_session),
    ]

    if include_hardware:
        cards.append(
            _resolve_hardware_state(safe_session)
        )

    return tuple(cards)


def _check_card(
    key: str,
    label: str,
    value: bool | None,
    *,
    true_summary: str,
    false_summary: str,
    unknown_summary: str,
    invert: bool = False,
    icon: str = "",
    display_value: str = "",
) -> SecurityCheckCard:
    """Construct one tri-state deterministic check card."""

    effective = (
        None
        if value is None
        else (not value if invert else value)
    )

    if effective is True:
        status = "verified"
        summary = true_summary
    elif effective is False:
        status = "failed"
        summary = false_summary
    else:
        status = "unavailable"
        summary = unknown_summary

    return SecurityCheckCard(
        key=key,
        label=label,
        status=status,
        summary=summary,
        icon=icon,
        value=display_value,
    )


def build_security_check_cards(
    session: Mapping[str, Any] | None,
    *,
    thresholds: ProtocolThresholds | None = None,
) -> tuple[SecurityCheckCard, ...]:
    """Build deterministic and GP security-check cards."""

    safe_session = _mapping(sanitize_mapping(session or {}))
    decision = _decision_mapping(safe_session)
    resolved_thresholds = resolve_thresholds(
        safe_session,
        thresholds,
    )

    credential_valid = _optional_bool(
        _first_present(
            safe_session,
            "credential_valid",
            "server_signature_valid",
        )
    )
    freshness_valid = _optional_bool(
        _first_present(
            safe_session,
            "freshness_valid",
            "timestamp_valid",
        )
    )
    replay_detected = _optional_bool(
        _first_present(
            safe_session,
            "replay_detected",
            "nonce_reused",
        )
    )
    schedule_valid = _optional_bool(
        _first_present(
            safe_session,
            "schedule_valid",
            "schedule_binding_valid",
        )
    )
    decoder_success = _optional_bool(
        _first_present(
            safe_session,
            "decoder_success",
            "css_decoder_success",
        )
    )
    tag_valid = _optional_bool(
        _first_present(
            safe_session,
            "tag_recovered",
            "tag_match",
            "tag_verified",
        )
    )

    qber = _optional_float(
        _first_present(
            safe_session,
            "qber_raw",
            "raw_qber",
            "qber",
        )
    )
    qber_pass = (
        None
        if qber is None
        else qber <= resolved_thresholds.fixed_qber_threshold
    )

    loss_rate = _optional_float(
        _first_present(
            safe_session,
            "loss_rate",
            "observed_loss_rate",
        )
    )
    loss_pass = (
        None
        if loss_rate is None
        else loss_rate <= resolved_thresholds.maximum_loss_rate
    )

    observed = _optional_int(
        _first_present(
            safe_session,
            "observed_check_blocks",
            "qber_observed",
        )
    )
    checks_sufficient = (
        None
        if observed is None
        else observed >= resolved_thresholds.required_check_blocks
    )

    p_attack = _optional_float(
        _first_present(
            decision,
            "p_attack",
            default=safe_session.get("p_attack"),
        )
    )
    gp_pass = (
        None
        if p_attack is None
        else p_attack
        < resolved_thresholds.operational_gp_threshold
    )

    cards = (
        _check_card(
            "credential",
            "Server credential",
            credential_valid,
            true_summary="ML-DSA credential verified.",
            false_summary="Server credential verification failed.",
            unknown_summary="Credential check not available.",
            icon="🪪",
        ),
        _check_card(
            "freshness",
            "Request freshness",
            freshness_valid,
            true_summary="Timestamp is inside the freshness window.",
            false_summary="Request is stale or invalid.",
            unknown_summary="Freshness result not available.",
            icon="⏱️",
        ),
        _check_card(
            "replay",
            "Replay protection",
            replay_detected,
            true_summary="Nonce is fresh and unused.",
            false_summary="Nonce reuse or replay detected.",
            unknown_summary="Replay-cache result not available.",
            invert=True,
            icon="🔁",
        ),
        _check_card(
            "schedule",
            "Schedule binding",
            schedule_valid,
            true_summary="Control schedule matches the transcript.",
            false_summary="Schedule or transcript binding failed.",
            unknown_summary="Schedule validation not available.",
            icon="🗓️",
        ),
        _check_card(
            "decoder",
            "Steane decoder",
            decoder_success,
            true_summary="Required logical payload was recoverable.",
            false_summary="Required payload block was unrecoverable.",
            unknown_summary="Decoder result not available.",
            icon="🧩",
        ),
        _check_card(
            "tag",
            "KMAC tag",
            tag_valid,
            true_summary="Recovered tag matches in constant time.",
            false_summary="Authentication tag mismatch.",
            unknown_summary="Tag-verification result not available.",
            icon="🏷️",
        ),
        _check_card(
            "qber",
            "Raw QBER",
            qber_pass,
            true_summary="QBER is within the fixed baseline policy.",
            false_summary="QBER exceeds the fixed baseline policy.",
            unknown_summary="Raw QBER not available.",
            icon="📉",
            display_value=(
                f"{format_percentage(qber)} / "
                f"{format_percentage(resolved_thresholds.fixed_qber_threshold)}"
                if qber is not None
                else DEFAULT_PLACEHOLDER
            ),
        ),
        _check_card(
            "loss",
            "Loss policy",
            loss_pass,
            true_summary="Observed loss is within policy.",
            false_summary="Observed loss exceeds policy.",
            unknown_summary="Loss rate not available.",
            icon="📶",
            display_value=(
                f"{format_percentage(loss_rate)} / "
                f"{format_percentage(resolved_thresholds.maximum_loss_rate)}"
                if loss_rate is not None
                else DEFAULT_PLACEHOLDER
            ),
        ),
        _check_card(
            "check_blocks",
            "Check-block evidence",
            checks_sufficient,
            true_summary="Enough independent check blocks were observed.",
            false_summary="Too few independent check blocks were observed.",
            unknown_summary="Observed check-block count not available.",
            icon="🔬",
            display_value=(
                f"{observed}/{resolved_thresholds.required_check_blocks}"
                if observed is not None
                else DEFAULT_PLACEHOLDER
            ),
        ),
        _check_card(
            "gp",
            "Calibrated GP risk",
            gp_pass,
            true_summary="P(attack) is below the operational threshold.",
            false_summary="P(attack) reaches or exceeds the threshold.",
            unknown_summary="GP probability not available.",
            icon="🧠",
            display_value=(
                f"{format_probability(p_attack)} / "
                f"{format_probability(resolved_thresholds.operational_gp_threshold)}"
                if p_attack is not None
                else DEFAULT_PLACEHOLDER
            ),
        ),
    )

    return cards


def summarize_protocol_status(
    session: Mapping[str, Any] | None,
    *,
    thresholds: ProtocolThresholds | None = None,
    include_hardware: bool = True,
) -> ProtocolStatusSummary:
    """Build all cards and aggregate completion information."""

    safe_session = _mapping(sanitize_mapping(session or {}))
    status_cards = build_status_cards(
        safe_session,
        thresholds=thresholds,
        include_hardware=include_hardware,
    )
    checks = build_security_check_cards(
        safe_session,
        thresholds=thresholds,
    )
    accepted, retry_used, final_status, reason = (
        resolve_final_decision(safe_session)
    )

    completed = sum(
        normalize_status(card.status)
        not in {"inactive", "unavailable", "unknown"}
        for card in checks
    )

    return ProtocolStatusSummary(
        status_cards=status_cards,
        security_checks=checks,
        final_status=final_status,
        final_reason=reason,
        accepted=accepted,
        retry_used=retry_used,
        completed_check_count=completed,
        total_check_count=len(checks),
    )


def status_card_html(card: StatusCard) -> str:
    """Return escaped HTML for one high-level status card."""

    semantic = status_style(card.status)
    progress = _clamp_progress(card.progress)
    progress_html = ""

    if progress is not None:
        progress_html = f"""
<div style="
    height:0.36rem;
    border-radius:999px;
    overflow:hidden;
    background:{THEME.colors.border_soft};
    margin-top:0.75rem;
">
    <div style="
        width:{progress * 100.0:.1f}%;
        height:100%;
        background:{semantic.foreground};
    "></div>
</div>
"""

    return f"""
<div class="ft-card" style="
    border-color:{semantic.border};
    min-height:14rem;
">
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:0.65rem;
    ">
        <div style="
            font-size:1.35rem;
            line-height:1;
        ">{escape(card.icon)}</div>
        {status_badge_html(card.status)}
    </div>
    <div class="ft-card__title" style="margin-top:0.8rem;">
        {escape(card.title)}
    </div>
    <div class="ft-metric-value" style="
        font-size:1.35rem;
        margin-top:0.25rem;
    ">{escape(card.value)}</div>
    <div class="ft-muted" style="margin-top:0.4rem;">
        {escape(card.subtitle)}
    </div>
    {
        f'<div style="color:{THEME.colors.text_secondary};'
        f'margin-top:0.65rem;font-size:0.86rem;">'
        f'{escape(card.detail)}</div>'
        if card.detail
        else ''
    }
    {progress_html}
</div>
"""


def security_check_card_html(
    card: SecurityCheckCard,
) -> str:
    """Return escaped HTML for one security check."""

    semantic = status_style(card.status)

    return f"""
<div class="ft-card" style="
    border-color:{semantic.border};
    min-height:10.5rem;
">
    <div style="
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:0.6rem;
    ">
        <div style="
            color:{THEME.colors.text_primary};
            font-weight:760;
        ">{escape(card.icon)} {escape(card.label)}</div>
        {status_badge_html(card.status, include_icon=False)}
    </div>
    {
        f'<div class="ft-metric-value" style="font-size:1.15rem;'
        f'margin-top:0.55rem;">{escape(card.value)}</div>'
        if card.value
        else ''
    }
    <div style="
        color:{THEME.colors.text_secondary};
        margin-top:0.55rem;
        font-size:0.88rem;
    ">{escape(card.summary)}</div>
</div>
"""


def render_status_card_grid(
    cards: Sequence[StatusCard],
    *,
    columns: int = 3,
) -> None:
    """Render high-level status cards in Streamlit columns."""

    if columns < 1:
        raise StatusCardError("columns must be at least 1.")

    if not cards:
        return

    st = _streamlit()

    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        containers = st.columns(len(row))

        for container, card in zip(containers, row):
            with container:
                st.markdown(
                    status_card_html(card),
                    unsafe_allow_html=True,
                )


def render_security_check_grid(
    cards: Sequence[SecurityCheckCard],
    *,
    columns: int = 5,
) -> None:
    """Render deterministic and GP security checks."""

    if columns < 1:
        raise StatusCardError("columns must be at least 1.")

    if not cards:
        return

    st = _streamlit()

    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        containers = st.columns(len(row))

        for container, card in zip(containers, row):
            with container:
                st.markdown(
                    security_check_card_html(card),
                    unsafe_allow_html=True,
                )


def render_session_metric_strip(
    session: Mapping[str, Any] | None,
    *,
    thresholds: ProtocolThresholds | None = None,
    columns: int = 6,
) -> None:
    """Render the key session-level quantitative indicators."""

    safe_session = _mapping(sanitize_mapping(session or {}))
    decision = _decision_mapping(safe_session)
    resolved_thresholds = resolve_thresholds(
        safe_session,
        thresholds,
    )

    qber = _optional_float(
        _first_present(
            safe_session,
            "qber_raw",
            "raw_qber",
            "qber",
        )
    )
    loss = _optional_float(
        _first_present(
            safe_session,
            "loss_rate",
            "observed_loss_rate",
        )
    )
    p_attack = _optional_float(
        _first_present(
            decision,
            "p_attack",
            default=safe_session.get("p_attack"),
        )
    )
    observed = _optional_int(
        _first_present(
            safe_session,
            "observed_check_blocks",
            "qber_observed",
        )
    )
    attempts = _optional_int(
        _first_present(
            safe_session,
            "retry_attempts",
            "attempt_count",
            default=1,
        )
    )
    timings = _mapping(safe_session.get("timings"))
    end_to_end = _optional_float(
        _first_present(
            timings,
            "end_to_end_s",
            "end_to_end_seconds",
            default=safe_session.get("end_to_end_seconds"),
        )
    )

    qber_status = (
        None
        if qber is None
        else (
            "verified"
            if qber <= resolved_thresholds.fixed_qber_threshold
            else "warning"
        )
    )
    loss_status = (
        None
        if loss is None
        else (
            "verified"
            if loss <= resolved_thresholds.maximum_loss_rate
            else "failed"
        )
    )
    gp_status = (
        None
        if p_attack is None
        else (
            "verified"
            if p_attack
            < resolved_thresholds.operational_gp_threshold
            else (
                "gray_zone"
                if p_attack
                < resolved_thresholds.retry_upper_probability
                else "attack"
            )
        )
    )

    metrics = (
        MetricItem(
            "Raw QBER",
            format_percentage(qber),
            help_text=(
                "Fixed policy "
                f"{format_percentage(resolved_thresholds.fixed_qber_threshold)}"
            ),
            status=qber_status,
            icon="📉",
        ),
        MetricItem(
            "Loss rate",
            format_percentage(loss),
            help_text=(
                "Maximum "
                f"{format_percentage(resolved_thresholds.maximum_loss_rate)}"
            ),
            status=loss_status,
            icon="📶",
        ),
        MetricItem(
            "P(attack)",
            format_probability(p_attack),
            help_text=(
                "Threshold "
                f"{format_probability(resolved_thresholds.operational_gp_threshold)}"
            ),
            status=gp_status,
            icon="🧠",
        ),
        MetricItem(
            "Check blocks",
            (
                f"{observed}/{resolved_thresholds.required_check_blocks}"
                if observed is not None
                else DEFAULT_PLACEHOLDER
            ),
            status=(
                "verified"
                if observed is not None
                and observed
                >= resolved_thresholds.required_check_blocks
                else (
                    "failed"
                    if observed is not None
                    else None
                )
            ),
            icon="🔬",
        ),
        MetricItem(
            "Attempts",
            attempts or DEFAULT_PLACEHOLDER,
            help_text="Maximum 3",
            status=(
                "retry"
                if attempts is not None and attempts > 1
                else "inactive"
            ),
            icon="🔁",
        ),
        MetricItem(
            "End-to-end",
            format_duration(end_to_end),
            status=(
                "ready"
                if end_to_end is not None
                else "inactive"
            ),
            icon="⏱️",
        ),
    )

    render_metric_grid(
        metrics,
        columns=columns,
    )


def render_protocol_status(
    session: Mapping[str, Any] | None,
    *,
    thresholds: ProtocolThresholds | None = None,
    include_hardware: bool = True,
    status_columns: int = 3,
    check_columns: int = 5,
    show_metric_strip: bool = True,
) -> ProtocolStatusSummary:
    """
    Render the complete status overview and return its pure summary.

    View modules may reuse the returned summary for further conditional
    rendering.
    """

    summary = summarize_protocol_status(
        session,
        thresholds=thresholds,
        include_hardware=include_hardware,
    )

    render_status_card_grid(
        summary.status_cards,
        columns=status_columns,
    )

    if show_metric_strip:
        st = _streamlit()
        st.markdown(
            '<div class="ft-section-title">'
            'Session indicators</div>',
            unsafe_allow_html=True,
        )
        render_session_metric_strip(
            session,
            thresholds=thresholds,
        )

    st = _streamlit()
    st.markdown(
        '<div class="ft-section-title">'
        'Security checks</div>',
        unsafe_allow_html=True,
    )
    render_security_check_grid(
        summary.security_checks,
        columns=check_columns,
    )

    return summary


def status_card_diagnostics(
    session: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return non-secret card derivation diagnostics."""

    summary = summarize_protocol_status(session)

    return {
        "status_card_count": len(summary.status_cards),
        "security_check_count": len(summary.security_checks),
        "final_status": summary.final_status,
        "final_reason": summary.final_reason,
        "completed_check_count": summary.completed_check_count,
        "total_check_count": summary.total_check_count,
        "status_card_keys": [
            card.key for card in summary.status_cards
        ],
        "security_check_keys": [
            card.key for card in summary.security_checks
        ],
    }


__all__ = [
    "DEFAULT_FIXED_QBER_THRESHOLD",
    "DEFAULT_MAXIMUM_LOSS_RATE",
    "DEFAULT_OPERATIONAL_GP_THRESHOLD",
    "DEFAULT_REQUIRED_CHECK_BLOCKS",
    "DEFAULT_RETRY_UPPER_PROBABILITY",
    "ProtocolStatusSummary",
    "ProtocolThresholds",
    "SecurityCheckCard",
    "StatusCard",
    "StatusCardError",
    "build_security_check_cards",
    "build_status_cards",
    "render_protocol_status",
    "render_security_check_grid",
    "render_session_metric_strip",
    "render_status_card_grid",
    "resolve_final_decision",
    "resolve_thresholds",
    "security_check_card_html",
    "status_card_diagnostics",
    "status_card_html",
    "summarize_protocol_status",
]