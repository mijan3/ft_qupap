"""
End-to-end protocol monitor for the FT-QuPAP Streamlit dashboard.

This page visualizes the complete stored authentication flow:

1. Authentication request preparation
2. Freshness and replay validation
3. ML-DSA server package generation
4. Server credential verification
5. ML-KEM encapsulation
6. ML-KEM decapsulation
7. Transcript-bound K_auth / K_ctrl derivation
8. KMAC authentication-tag generation
9. Protected control-schedule generation
10. Payload and independent check-block preparation
11. Steane [[7,1,3]] encoding
12. Quantum-channel transmission
13. Controlled measurement
14. Raw-QBER calculation
15. Syndrome extraction and bounded correction
16. Payload recovery and constant-time tag verification
17. Nine observable GP-feature construction
18. Calibrated P(attack) inference
19. Accept, reject, or bounded fresh retry

The monitor is evidence-driven. It never marks a stage successful merely
because an earlier or later stage exists. Missing values remain inactive or
unavailable. Hard deterministic failures are displayed separately from the
Gaussian Process risk estimate.

The page does not execute cryptography or quantum simulation itself. It
visualizes stored session records and optional event traces produced by the
local FT-QuPAP protocol engine.

Security boundary
-----------------
Private keys, shared secrets, K_ss, K_auth, K_ctrl, raw subscriber identities,
raw nonces, signatures, ciphertexts, raw authentication tags, and encoded
quantum-frame contents are never displayed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, Mapping, Sequence

from .charts import (
    ChartDataError,
    build_attempt_timeline,
    build_decision_gauge,
    build_feature_profile,
    build_timing_breakdown,
    render_plotly_figure,
)
from .components import (
    DEFAULT_PROTOCOL_STEPS,
    GP_FEATURE_LABELS,
    KeyValueItem,
    MetricItem,
    ProtocolStep,
    format_duration,
    format_percentage,
    format_probability,
    render_attempt_history,
    render_card,
    render_banner,
    render_decision_banner,
    render_empty_state,
    render_feature_table,
    render_json_viewer,
    render_key_value_grid,
    render_metric_grid,
    render_protocol_stepper,
    render_records_table,
    render_sensitive_data_notice,
    sanitize_mapping,
)
from .home import load_latest_session
from .status_cards import (
    render_protocol_status,
    resolve_final_decision,
    resolve_thresholds,
)
from .theme import (
    apply_dashboard_theme,
    normalize_status,
    protocol_stage_color,
    render_divider,
    render_page_header,
    render_research_notice,
    render_section_title,
    status_badge_html,
)


PROTOCOL_STAGE_COUNT: Final[int] = 19

STAGE_GROUPS: Final[dict[str, tuple[int, int]]] = {
    "Request and admission": (1, 4),
    "PQC bootstrap": (5, 7),
    "Authentication material": (8, 10),
    "Quantum transmission": (11, 14),
    "Fault-tolerant recovery": (15, 16),
    "Adaptive detection": (17, 18),
    "Final policy": (19, 19),
}

EVENT_COLLECTION_KEYS: Final[tuple[str, ...]] = (
    "protocol_events",
    "event_log",
    "events",
    "trace",
    "protocol_trace",
    "monitor_trace",
)

ATTEMPT_HISTORY_KEYS: Final[tuple[str, ...]] = (
    "attempt_history",
    "attempts",
    "retry_history",
)

STAGE_FIELD_HINTS: Final[dict[int, tuple[str, ...]]] = {
    1: (
        "request_prepared",
        "authentication_request_created",
        "request_received",
    ),
    2: (
        "freshness_valid",
        "timestamp_valid",
        "replay_detected",
    ),
    3: (
        "server_package_signed",
        "ml_dsa_signature_created",
        "signed_server_package_created",
    ),
    4: (
        "credential_valid",
        "server_signature_valid",
        "ml_dsa_verification_success",
    ),
    5: (
        "mlkem_encapsulation_success",
        "encapsulation_success",
    ),
    6: (
        "mlkem_decapsulation_success",
        "decapsulation_success",
    ),
    7: (
        "key_derivation_success",
        "session_keys_derived",
        "shared_secret_match",
    ),
    8: (
        "kmac_tag_generated",
        "tag_generated",
        "authentication_tag_generated",
    ),
    9: (
        "control_schedule_generated",
        "schedule_generated",
        "schedule_valid",
    ),
    10: (
        "logical_blocks_prepared",
        "payload_and_checks_prepared",
    ),
    11: (
        "steane_encoding_success",
        "css_encoding_success",
        "quantum_token_prepared",
    ),
    12: (
        "transmission_ready",
        "quantum_channel_completed",
        "frame_transmitted",
    ),
    13: (
        "measurement_completed",
        "observed_check_blocks",
        "qber_observed",
    ),
    14: (
        "qber_raw",
        "raw_qber",
        "qber",
    ),
    15: (
        "syndrome_extraction_success",
        "decoder_success",
        "css_decoder_success",
    ),
    16: (
        "tag_recovered",
        "tag_match",
        "tag_verified",
    ),
    17: (
        "features",
        "gp_features",
        "observable_features",
    ),
    18: (
        "p_attack",
        "attack_probability",
        "gp_inference_completed",
    ),
    19: (
        "accepted",
        "decision",
        "reason",
    ),
}


class ProtocolMonitorError(ValueError):
    """Raised when protocol-monitor evidence is invalid."""


@dataclass(frozen=True)
class ProtocolEvent:
    """One safe event from the stored protocol trace."""

    sequence: int
    timestamp: str
    stage_number: int | None
    stage_key: str
    actor: str
    event: str
    status: str
    message: str
    duration_seconds: float | None = None

    def to_dictionary(self) -> dict[str, Any]:
        return sanitize_mapping(asdict(self))


@dataclass(frozen=True)
class StageEvidence:
    """Derived state for one protocol stage."""

    number: int
    key: str
    label: str
    stage: str
    owner: str
    description: str
    status: str
    evidence: str
    duration_seconds: float | None = None


@dataclass(frozen=True)
class ProtocolMonitorSnapshot:
    """Safe state used by the protocol monitor."""

    source: str | None
    session_available: bool
    session_id: str
    scenario_id: str
    context: str
    execution_status: str
    overall_status: str

    stages: tuple[StageEvidence, ...]
    events: tuple[ProtocolEvent, ...]
    current_stage_number: int | None
    first_failed_stage_number: int | None
    completed_stage_count: int
    failed_stage_count: int
    inactive_stage_count: int

    qber_raw: float | None
    loss_rate: float | None
    observed_check_blocks: int | None
    required_check_blocks: int
    p_attack: float | None
    uncertainty: float | None
    operational_threshold: float
    retry_upper_probability: float

    deterministic_pass: bool | None
    accepted: bool | None
    retry_used: bool
    attempt_count: int
    decision_reason: str

    features: Mapping[str, float]
    timings: Mapping[str, float]
    attempt_history: tuple[Mapping[str, Any], ...]
    safe_session: Mapping[str, Any] = field(default_factory=dict)

    def to_dictionary(self) -> dict[str, Any]:
        return sanitize_mapping(asdict(self))


def _streamlit() -> Any:
    """Import Streamlit only when rendering is requested."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to render protocol_monitor_view.py."
        ) from exc

    return st


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(
    mapping: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _nested_first(
    payload: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    *,
    default: Any = None,
) -> Any:
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
        "pass",
        "passed",
        "valid",
        "verified",
        "success",
        "successful",
        "completed",
        "ready",
        "accepted",
        "matched",
        "fresh",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "fail",
        "failed",
        "invalid",
        "error",
        "rejected",
        "mismatch",
        "stale",
    }:
        return False

    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _safe_timestamp(value: Any) -> str:
    if value is None:
        return "—"

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                float(value),
                tz=timezone.utc,
            ).isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            return str(value)

    return str(value)


def _normalize_event_status(value: Any) -> str:
    normalized = normalize_status(value)

    if normalized == "unknown":
        text = str(value).strip().lower()

        if text in {"complete", "completed", "done"}:
            return "verified"

        if text in {"start", "started", "begin", "running"}:
            return "running"

    return normalized


def _status_from_tristate(
    value: bool | None,
    *,
    success_status: str = "verified",
    failure_status: str = "failed",
    unknown_status: str = "inactive",
) -> str:
    if value is True:
        return success_status

    if value is False:
        return failure_status

    return unknown_status


def _extract_features(
    session: Mapping[str, Any],
) -> dict[str, float]:
    source = _mapping(
        _nested_first(
            session,
            (
                ("features",),
                ("gp_features",),
                ("observable_features",),
                ("decision", "features"),
            ),
            default={},
        )
    )
    features: dict[str, float] = {}

    for key in GP_FEATURE_LABELS:
        value = _optional_float(
            source.get(
                key,
                session.get(key),
            )
        )

        if value is not None:
            features[key] = value

    return features


def _extract_timings(
    session: Mapping[str, Any],
) -> dict[str, float]:
    source = dict(_mapping(session.get("timings")))

    for key, value in session.items():
        if key.startswith("timing_") or key.endswith("_seconds"):
            source.setdefault(str(key), value)

    timings: dict[str, float] = {}

    for key, value in source.items():
        number = _optional_float(value)

        if number is not None and number >= 0:
            timings[str(key)] = number

    return timings


def _extract_attempt_history(
    session: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw: Any = ()

    for key in ATTEMPT_HISTORY_KEYS:
        if key in session:
            raw = session[key]
            break

    if not raw:
        decision = _mapping(session.get("decision"))

        for key in ATTEMPT_HISTORY_KEYS:
            if key in decision:
                raw = decision[key]
                break

    if not isinstance(raw, Sequence) or isinstance(
        raw,
        (str, bytes, bytearray),
    ):
        return ()

    rows = []

    for item in raw:
        if isinstance(item, Mapping):
            safe = sanitize_mapping(item)

            if isinstance(safe, Mapping):
                rows.append(dict(safe))

    return tuple(rows)


def _extract_events(
    session: Mapping[str, Any],
) -> tuple[ProtocolEvent, ...]:
    raw_events: Any = ()

    for key in EVENT_COLLECTION_KEYS:
        value = session.get(key)

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raw_events = value
            break

    events = []

    for index, raw in enumerate(raw_events, start=1):
        if not isinstance(raw, Mapping):
            continue

        safe = sanitize_mapping(raw)

        if not isinstance(safe, Mapping):
            continue

        stage_number = _optional_int(
            _first_present(
                safe,
                "stage_number",
                "step_number",
                "step",
            )
        )

        if stage_number is not None and not (
            1 <= stage_number <= PROTOCOL_STAGE_COUNT
        ):
            stage_number = None

        stage_key = str(
            _first_present(
                safe,
                "stage_key",
                "step_key",
                "stage",
                default="unspecified",
            )
        )
        event_name = str(
            _first_present(
                safe,
                "event",
                "action",
                "name",
                default=stage_key,
            )
        )
        actor = str(
            _first_present(
                safe,
                "actor",
                "owner",
                "component",
                default="FT-QuPAP",
            )
        )
        message = str(
            _first_present(
                safe,
                "message",
                "detail",
                "description",
                "reason",
                default="",
            )
        )
        timestamp = _safe_timestamp(
            _first_present(
                safe,
                "timestamp_utc",
                "timestamp",
                "created_at_utc",
                default="—",
            )
        )
        duration = _optional_float(
            _first_present(
                safe,
                "duration_seconds",
                "duration_s",
                "elapsed_seconds",
            )
        )

        events.append(
            ProtocolEvent(
                sequence=_optional_int(
                    safe.get("sequence")
                )
                or index,
                timestamp=timestamp,
                stage_number=stage_number,
                stage_key=stage_key,
                actor=actor,
                event=event_name,
                status=_normalize_event_status(
                    _first_present(
                        safe,
                        "status",
                        "result",
                        default="ready",
                    )
                ),
                message=message,
                duration_seconds=duration,
            )
        )

    events.sort(key=lambda event: event.sequence)
    return tuple(events)


def _explicit_stage_statuses(
    session: Mapping[str, Any],
    events: Sequence[ProtocolEvent],
) -> dict[int, str]:
    """Resolve explicit stage statuses from stored trace/state mappings."""

    statuses: dict[int, str] = {}

    stage_mapping = _nested_first(
        session,
        (
            ("stage_statuses",),
            ("protocol_stage_statuses",),
            ("monitor", "stage_statuses"),
        ),
        default={},
    )

    if isinstance(stage_mapping, Mapping):
        for key, value in stage_mapping.items():
            stage_number = _optional_int(key)

            if stage_number is None:
                for step in DEFAULT_PROTOCOL_STEPS:
                    if str(key) == step.key:
                        stage_number = step.number
                        break

            if stage_number is not None and (
                1 <= stage_number <= PROTOCOL_STAGE_COUNT
            ):
                statuses[stage_number] = _normalize_event_status(
                    value
                )

    for event in events:
        if event.stage_number is None:
            continue

        current = statuses.get(event.stage_number)
        incoming = event.status

        # Failure dominates, then verified, then running.
        priority = {
            "failed": 4,
            "rejected": 4,
            "attack": 4,
            "verified": 3,
            "accepted": 3,
            "retry": 2,
            "running": 2,
            "ready": 1,
            "inactive": 0,
            "unavailable": 0,
            "unknown": 0,
        }

        if current is None or priority.get(
            incoming,
            0,
        ) >= priority.get(current, 0):
            statuses[event.stage_number] = incoming

    return statuses


def _field_value(
    session: Mapping[str, Any],
    *names: str,
) -> Any:
    """Find a field at root or common component/decision locations."""

    components = (
        session,
        _mapping(session.get("mobile_station")),
        _mapping(session.get("authentication_server")),
        _mapping(session.get("bootstrap")),
        _mapping(session.get("decision")),
        _mapping(session.get("monitor")),
    )

    for component in components:
        for name in names:
            if name in component and component[name] is not None:
                return component[name]

    return None


def _derive_stage_status(
    stage_number: int,
    session: Mapping[str, Any],
    *,
    features: Mapping[str, float],
    accepted: bool | None,
    deterministic_pass: bool | None,
) -> tuple[str, str]:
    """Derive one stage conservatively from stored protocol evidence."""

    if stage_number == 1:
        value = _optional_bool(
            _field_value(
                session,
                "request_prepared",
                "authentication_request_created",
                "request_received",
            )
        )
        return (
            _status_from_tristate(value),
            "Authentication request evidence",
        )

    if stage_number == 2:
        freshness = _optional_bool(
            _field_value(
                session,
                "freshness_valid",
                "timestamp_valid",
            )
        )
        replay_detected = _optional_bool(
            _field_value(
                session,
                "replay_detected",
                "nonce_reused",
            )
        )

        if freshness is False or replay_detected is True:
            return "failed", "Freshness or replay protection failed"

        if freshness is True and replay_detected is False:
            return "verified", "Fresh timestamp and unused nonce"

        return "inactive", "Freshness/replay evidence unavailable"

    if stage_number == 3:
        value = _optional_bool(
            _field_value(
                session,
                "server_package_signed",
                "ml_dsa_signature_created",
                "signed_server_package_created",
            )
        )
        return (
            _status_from_tristate(value),
            "ML-DSA server-package generation",
        )

    if stage_number == 4:
        value = _optional_bool(
            _field_value(
                session,
                "credential_valid",
                "server_signature_valid",
                "ml_dsa_verification_success",
            )
        )
        return (
            _status_from_tristate(value),
            "Pinned-trust credential verification",
        )

    if stage_number == 5:
        value = _optional_bool(
            _field_value(
                session,
                "mlkem_encapsulation_success",
                "encapsulation_success",
            )
        )
        return (
            _status_from_tristate(value),
            "ML-KEM-768 encapsulation",
        )

    if stage_number == 6:
        value = _optional_bool(
            _field_value(
                session,
                "mlkem_decapsulation_success",
                "decapsulation_success",
            )
        )
        return (
            _status_from_tristate(value),
            "ML-KEM-768 decapsulation",
        )

    if stage_number == 7:
        derived = _optional_bool(
            _field_value(
                session,
                "key_derivation_success",
                "session_keys_derived",
            )
        )
        shared_match = _optional_bool(
            _field_value(
                session,
                "shared_secret_match",
                "session_secret_match",
            )
        )

        if derived is False or shared_match is False:
            return "failed", "Session-key derivation or secret match failed"

        if derived is True or shared_match is True:
            return "verified", "Transcript-bound keys derived"

        return "inactive", "Key-derivation evidence unavailable"

    if stage_number == 8:
        value = _optional_bool(
            _field_value(
                session,
                "kmac_tag_generated",
                "tag_generated",
                "authentication_tag_generated",
            )
        )
        return (
            _status_from_tristate(value),
            "128-bit KMAC256 tag generation",
        )

    if stage_number == 9:
        generated = _optional_bool(
            _field_value(
                session,
                "control_schedule_generated",
                "schedule_generated",
            )
        )
        valid = _optional_bool(
            _field_value(
                session,
                "schedule_valid",
                "schedule_binding_valid",
            )
        )

        if generated is False or valid is False:
            return "failed", "Control schedule generation or binding failed"

        if generated is True or valid is True:
            return "verified", "Protected schedule available"

        return "inactive", "Schedule evidence unavailable"

    if stage_number == 10:
        value = _optional_bool(
            _field_value(
                session,
                "logical_blocks_prepared",
                "payload_and_checks_prepared",
            )
        )
        return (
            _status_from_tristate(value),
            "128 payload and 32 check blocks",
        )

    if stage_number == 11:
        value = _optional_bool(
            _field_value(
                session,
                "steane_encoding_success",
                "css_encoding_success",
                "quantum_token_prepared",
            )
        )
        return (
            _status_from_tristate(
                value,
                success_status="quantum",
            ),
            "Steane [[7,1,3]] encoding",
        )

    if stage_number == 12:
        value = _optional_bool(
            _field_value(
                session,
                "transmission_ready",
                "quantum_channel_completed",
                "frame_transmitted",
            )
        )

        if value is None:
            qber = _optional_float(
                _field_value(
                    session,
                    "qber_raw",
                    "raw_qber",
                )
            )

            if qber is not None:
                value = True

        return (
            _status_from_tristate(
                value,
                success_status="quantum",
            ),
            "Quantum-channel frame transmission",
        )

    if stage_number == 13:
        measurement = _optional_bool(
            _field_value(
                session,
                "measurement_completed",
            )
        )
        observed = _optional_int(
            _field_value(
                session,
                "observed_check_blocks",
                "qber_observed",
            )
        )

        if measurement is False:
            return "failed", "Controlled measurement failed"

        if measurement is True or observed is not None:
            return "quantum", "Independent checks measured"

        return "inactive", "Measurement evidence unavailable"

    if stage_number == 14:
        qber = _optional_float(
            _field_value(
                session,
                "qber_raw",
                "raw_qber",
                "qber",
            )
        )

        if qber is not None:
            return "verified", f"Raw QBER = {qber:.6f}"

        return "inactive", "Raw-QBER value unavailable"

    if stage_number == 15:
        syndrome = _optional_bool(
            _field_value(
                session,
                "syndrome_extraction_success",
                "syndrome_success",
            )
        )
        decoder = _optional_bool(
            _field_value(
                session,
                "decoder_success",
                "css_decoder_success",
            )
        )

        if syndrome is False or decoder is False:
            return "failed", "Syndrome recovery or decoder failed"

        if syndrome is True or decoder is True:
            return "verified", "Syndrome recovery and bounded correction"

        return "inactive", "Decoder evidence unavailable"

    if stage_number == 16:
        tag_verified = _optional_bool(
            _field_value(
                session,
                "tag_recovered",
                "tag_match",
                "tag_verified",
            )
        )
        return (
            _status_from_tristate(tag_verified),
            "Recovered KMAC tag comparison",
        )

    if stage_number == 17:
        if len(features) == len(GP_FEATURE_LABELS):
            return "verified", "All nine observable features available"

        if features:
            return "warning", (
                f"{len(features)}/{len(GP_FEATURE_LABELS)} "
                "observable features available"
            )

        return "inactive", "Observable GP features unavailable"

    if stage_number == 18:
        p_attack = _optional_float(
            _field_value(
                session,
                "p_attack",
                "attack_probability",
            )
        )
        inferred = _optional_bool(
            _field_value(
                session,
                "gp_inference_completed",
                "gp_probability_computed",
            )
        )

        if inferred is False:
            return "failed", "GP inference failed"

        if p_attack is not None or inferred is True:
            return "verified", (
                f"Calibrated P(attack) = {p_attack:.6f}"
                if p_attack is not None
                else "Calibrated GP inference completed"
            )

        return "inactive", "GP-inference evidence unavailable"

    if stage_number == 19:
        if accepted is True:
            retry_used = bool(
                _optional_bool(session.get("retry_used")) or False
            )
            return (
                "retry" if retry_used else "accepted",
                "Accepted after fresh retry"
                if retry_used
                else "Authentication accepted",
            )

        if accepted is False:
            return "rejected", "Authentication rejected"

        if deterministic_pass is False:
            return "rejected", "Hard deterministic rejection"

        return "inactive", "Final policy decision unavailable"

    raise ProtocolMonitorError(
        f"Unsupported stage number: {stage_number}."
    )


def _stage_duration(
    stage: ProtocolStep,
    timings: Mapping[str, float],
) -> float | None:
    """Find a likely timing value for a protocol stage."""

    tokens = {
        1: ("request", "payload_preparation"),
        2: ("freshness", "replay", "request_validation"),
        3: ("ml_dsa_sign", "server_package", "signature_generation"),
        4: ("ml_dsa_verify", "credential", "signature_verification"),
        5: ("encapsulation", "mlkem_encapsulation"),
        6: ("decapsulation", "mlkem_decapsulation"),
        7: ("key_derivation", "kdf"),
        8: ("kmac", "tag_generation"),
        9: ("schedule",),
        10: ("logical_block", "payload_check"),
        11: ("steane_encoding", "css_encoding"),
        12: ("channel", "transmission"),
        13: ("measurement",),
        14: ("qber",),
        15: ("syndrome", "decoder", "correction"),
        16: ("tag_verification", "kmac_verification"),
        17: ("feature",),
        18: ("gp", "calibration", "inference"),
        19: ("decision", "policy"),
    }[stage.number]

    matching = []

    for key, value in timings.items():
        normalized = str(key).lower()

        if any(token in normalized for token in tokens):
            matching.append(value)

    if not matching:
        return None

    return float(sum(matching))


def _build_stages(
    session: Mapping[str, Any],
    *,
    events: Sequence[ProtocolEvent],
    features: Mapping[str, float],
    timings: Mapping[str, float],
    accepted: bool | None,
    deterministic_pass: bool | None,
) -> tuple[StageEvidence, ...]:
    explicit = _explicit_stage_statuses(session, events)
    stages = []

    for step in DEFAULT_PROTOCOL_STEPS:
        derived_status, evidence = _derive_stage_status(
            step.number,
            session,
            features=features,
            accepted=accepted,
            deterministic_pass=deterministic_pass,
        )
        # The current final-decision fields are authoritative for stage 19.
        # A copied or stale event trace must not override an accepted-after-
        # retry or rejected result from the current session record.
        if step.number == 19 and derived_status in {
            "accepted",
            "retry",
            "rejected",
        }:
            status = derived_status
        else:
            status = explicit.get(step.number, derived_status)

        stages.append(
            StageEvidence(
                number=step.number,
                key=step.key,
                label=step.label,
                stage=step.stage,
                owner=step.owner,
                description=step.description,
                status=normalize_status(status),
                evidence=evidence,
                duration_seconds=_stage_duration(
                    step,
                    timings,
                ),
            )
        )

    return tuple(stages)


def _current_stage(
    stages: Sequence[StageEvidence],
) -> int | None:
    for stage in stages:
        if stage.status in {"running", "ready"}:
            return stage.number

    for stage in stages:
        if stage.status in {
            "inactive",
            "unavailable",
            "unknown",
        }:
            return stage.number

    return stages[-1].number if stages else None


def build_protocol_monitor_snapshot(
    session: Mapping[str, Any] | None = None,
    *,
    source: str | None = None,
) -> ProtocolMonitorSnapshot:
    """Build the pure, redacted state for the protocol monitor."""

    if session is None:
        latest, latest_source, _ = load_latest_session()
        session = latest
        source = source or latest_source

    safe = sanitize_mapping(session or {})

    if not isinstance(safe, Mapping):
        safe = {}

    safe_session = dict(safe)
    decision = _mapping(safe_session.get("decision"))
    accepted, retry_used, final_status, reason = (
        resolve_final_decision(safe_session)
    )
    deterministic_pass = _optional_bool(
        _first_present(
            decision,
            "deterministic_pass",
            default=safe_session.get("deterministic_pass"),
        )
    )
    thresholds = resolve_thresholds(safe_session)

    features = _extract_features(safe_session)
    timings = _extract_timings(safe_session)
    events = _extract_events(safe_session)
    stages = _build_stages(
        safe_session,
        events=events,
        features=features,
        timings=timings,
        accepted=accepted,
        deterministic_pass=deterministic_pass,
    )

    failed = [
        stage
        for stage in stages
        if stage.status in {
            "failed",
            "rejected",
            "attack",
        }
    ]
    completed = [
        stage
        for stage in stages
        if stage.status in {
            "verified",
            "accepted",
            "retry",
            "quantum",
        }
    ]
    inactive = [
        stage
        for stage in stages
        if stage.status in {
            "inactive",
            "unavailable",
            "unknown",
        }
    ]

    execution_status = str(
        _first_present(
            safe_session,
            "execution_status",
            "run_status",
            default=(
                "completed"
                if accepted is not None
                else "not_available"
            ),
        )
    )

    if accepted is not None:
        overall_status = final_status
    elif failed:
        overall_status = "failed"
    elif completed:
        overall_status = "running"
    else:
        overall_status = "inactive"

    p_attack = _optional_float(
        _first_present(
            decision,
            "p_attack",
            "attack_probability",
            default=_first_present(
                safe_session,
                "p_attack",
                "attack_probability",
            ),
        )
    )
    uncertainty = _optional_float(
        _first_present(
            decision,
            "uncertainty",
            default=safe_session.get("uncertainty"),
        )
    )

    attempt_count = (
        _optional_int(
            _first_present(
                safe_session,
                "retry_attempts",
                "attempt_count",
                "attempts",
                default=1,
            )
        )
        or 1
    )

    return ProtocolMonitorSnapshot(
        source=source,
        session_available=bool(safe_session),
        session_id=str(
            _first_present(
                safe_session,
                "session_id",
                "record_id",
                "run_id",
                default="—",
            )
        ),
        scenario_id=str(
            _first_present(
                safe_session,
                "scenario_id",
                "scenario",
                default="—",
            )
        ),
        context=str(
            _first_present(
                safe_session,
                "context",
                "service_context",
                default="—",
            )
        ),
        execution_status=execution_status,
        overall_status=overall_status,
        stages=stages,
        events=events,
        current_stage_number=_current_stage(stages),
        first_failed_stage_number=(
            failed[0].number if failed else None
        ),
        completed_stage_count=len(completed),
        failed_stage_count=len(failed),
        inactive_stage_count=len(inactive),
        qber_raw=_optional_float(
            _first_present(
                safe_session,
                "qber_raw",
                "raw_qber",
                "qber",
            )
        ),
        loss_rate=_optional_float(
            _first_present(
                safe_session,
                "loss_rate",
                "observed_loss_rate",
            )
        ),
        observed_check_blocks=_optional_int(
            _first_present(
                safe_session,
                "observed_check_blocks",
                "qber_observed",
            )
        ),
        required_check_blocks=thresholds.required_check_blocks,
        p_attack=p_attack,
        uncertainty=uncertainty,
        operational_threshold=(
            thresholds.operational_gp_threshold
        ),
        retry_upper_probability=(
            thresholds.retry_upper_probability
        ),
        deterministic_pass=deterministic_pass,
        accepted=accepted,
        retry_used=retry_used,
        attempt_count=attempt_count,
        decision_reason=reason,
        features=features,
        timings=timings,
        attempt_history=_extract_attempt_history(
            safe_session
        ),
        safe_session=safe_session,
    )


def _shared_steps(
    snapshot: ProtocolMonitorSnapshot,
) -> tuple[ProtocolStep, ...]:
    return tuple(
        ProtocolStep(
            number=stage.number,
            key=stage.key,
            label=stage.label,
            stage=stage.stage,
            description=(
                f"{stage.description} Evidence: {stage.evidence}"
            ),
            owner=stage.owner,
            status=stage.status,
        )
        for stage in snapshot.stages
    )


def _render_overview(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_page_header(
        title="Protocol Monitor",
        subtitle=(
            "Evidence-driven view of the complete FT-QuPAP request, PQC, "
            "quantum, fault-tolerant recovery, GP, and decision pipeline"
        ),
        icon="📡",
        status=snapshot.overall_status,
    )

    render_research_notice()

    if snapshot.session_available:
        render_banner(
            "Stored session loaded",
            (
                "Protocol-monitor source: "
                f"{snapshot.source or 'runtime session'}"
            ),
            status="ready",
            icon="📄",
        )
    else:
        render_banner(
            "No completed session loaded",
            (
                "The monitor is displaying the protocol structure only. "
                "Run a controlled scenario to populate stage evidence."
            ),
            status="inactive",
            icon="📭",
        )

    metrics = (
        MetricItem(
            "Completed stages",
            (
                f"{snapshot.completed_stage_count}/"
                f"{PROTOCOL_STAGE_COUNT}"
            ),
            status=(
                "verified"
                if snapshot.completed_stage_count
                == PROTOCOL_STAGE_COUNT
                else "running"
            ),
            icon="✅",
        ),
        MetricItem(
            "Failed stages",
            snapshot.failed_stage_count,
            status=(
                "failed"
                if snapshot.failed_stage_count > 0
                else "verified"
            ),
            icon="⛔",
        ),
        MetricItem(
            "Current stage",
            snapshot.current_stage_number or "—",
            status=(
                "running"
                if snapshot.current_stage_number is not None
                else "inactive"
            ),
            icon="▶️",
        ),
        MetricItem(
            "Raw QBER",
            format_percentage(snapshot.qber_raw),
            help_text=(
                "Computed from independent observed check blocks"
            ),
            status=(
                "ready"
                if snapshot.qber_raw is not None
                else "inactive"
            ),
            icon="📉",
        ),
        MetricItem(
            "P(attack)",
            format_probability(snapshot.p_attack),
            help_text=(
                "Threshold "
                f"{format_probability(snapshot.operational_threshold)}"
            ),
            status=(
                "verified"
                if snapshot.p_attack is not None
                and snapshot.p_attack
                < snapshot.operational_threshold
                else (
                    "attack"
                    if snapshot.p_attack is not None
                    and snapshot.p_attack
                    >= snapshot.retry_upper_probability
                    else (
                        "gray_zone"
                        if snapshot.p_attack is not None
                        else "inactive"
                    )
                )
            ),
            icon="🧠",
        ),
        MetricItem(
            "Attempts",
            snapshot.attempt_count,
            status=(
                "retry"
                if snapshot.attempt_count > 1
                else "inactive"
            ),
            icon="🔁",
        ),
    )
    render_metric_grid(metrics, columns=6)


def _render_session_identity(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Session identity and policy state",
        icon="🪪",
    )

    items = (
        KeyValueItem(
            "Session ID",
            snapshot.session_id,
            code=True,
        ),
        KeyValueItem(
            "Scenario",
            snapshot.scenario_id,
        ),
        KeyValueItem(
            "Context",
            snapshot.context,
        ),
        KeyValueItem(
            "Execution status",
            snapshot.execution_status,
            status=(
                "verified"
                if snapshot.execution_status.lower()
                == "completed"
                else "inactive"
            ),
        ),
        KeyValueItem(
            "Deterministic gates",
            (
                "Passed"
                if snapshot.deterministic_pass is True
                else (
                    "Failed"
                    if snapshot.deterministic_pass is False
                    else "Not available"
                )
            ),
            status=_status_from_tristate(
                snapshot.deterministic_pass
            ),
        ),
        KeyValueItem(
            "Final decision",
            (
                "Accepted after retry"
                if snapshot.accepted is True
                and snapshot.retry_used
                else (
                    "Accepted"
                    if snapshot.accepted is True
                    else (
                        "Rejected"
                        if snapshot.accepted is False
                        else "Not available"
                    )
                )
            ),
            status=snapshot.overall_status,
        ),
        KeyValueItem(
            "Decision reason",
            snapshot.decision_reason,
        ),
        KeyValueItem(
            "Evidence source",
            snapshot.source or "runtime session",
            code=True,
        ),
    )
    render_key_value_grid(items, columns=4)


def _render_full_pipeline(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Complete 19-stage protocol pipeline",
        icon="🧬",
    )

    render_protocol_stepper(
        steps=_shared_steps(snapshot),
        compact=False,
    )


def _render_stage_group_summary(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Protocol-phase summary",
        icon="🧩",
    )

    st = _streamlit()
    groups = list(STAGE_GROUPS.items())

    for start in range(0, len(groups), 4):
        row = groups[start : start + 4]
        columns = st.columns(len(row))

        for container, (
            group_name,
            (start_stage, end_stage),
        ) in zip(columns, row):
            selected = [
                stage
                for stage in snapshot.stages
                if start_stage <= stage.number <= end_stage
            ]

            if any(
                stage.status
                in {"failed", "rejected", "attack"}
                for stage in selected
            ):
                status = "failed"
            elif all(
                stage.status
                in {
                    "verified",
                    "accepted",
                    "retry",
                    "quantum",
                }
                for stage in selected
            ):
                status = "verified"
            elif any(
                stage.status in {"running", "ready"}
                for stage in selected
            ):
                status = "running"
            else:
                status = "inactive"

            completed = sum(
                stage.status
                in {
                    "verified",
                    "accepted",
                    "retry",
                    "quantum",
                }
                for stage in selected
            )

            with container:
                render_card(
                    title=group_name,
                    body=(
                        f"{completed}/{len(selected)} stages completed"
                    ),
                    footer=(
                        f"Stages {start_stage}–{end_stage}"
                        if start_stage != end_stage
                        else f"Stage {start_stage}"
                    ),
                    icon="🧩",
                    status=status,
                )


def _render_playback(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    """Render a local visual inspection control for stored stage evidence."""

    render_section_title(
        "Stored-session stage inspector",
        icon="🔎",
    )

    st = _streamlit()

    selected_number = st.slider(
        "Inspect stage",
        min_value=1,
        max_value=PROTOCOL_STAGE_COUNT,
        value=(
            snapshot.current_stage_number
            if snapshot.current_stage_number is not None
            else 1
        ),
        step=1,
        key="protocol_monitor_stage_inspector",
    )

    selected = snapshot.stages[selected_number - 1]

    render_key_value_grid(
        (
            KeyValueItem(
                "Stage",
                f"{selected.number}. {selected.label}",
            ),
            KeyValueItem(
                "Owner",
                selected.owner,
            ),
            KeyValueItem(
                "Category",
                selected.stage,
                code=True,
            ),
            KeyValueItem(
                "Status",
                selected.status,
                status=selected.status,
            ),
            KeyValueItem(
                "Evidence",
                selected.evidence,
            ),
            KeyValueItem(
                "Duration",
                format_duration(
                    selected.duration_seconds
                ),
            ),
        ),
        columns=3,
    )

    st.markdown(
        (
            '<div class="ft-card" style="'
            f'border-left:4px solid '
            f'{protocol_stage_color(selected.stage)};">'
            f'<div class="ft-card__title">'
            f'{selected.number}. {selected.label}</div>'
            f'<div class="ft-card__body">'
            f'{selected.description}</div>'
            f'<div style="margin-top:0.6rem;">'
            f'{status_badge_html(selected.status)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    if snapshot.events:
        matching = [
            event.to_dictionary()
            for event in snapshot.events
            if event.stage_number == selected_number
        ]

        if matching:
            render_records_table(
                matching,
                columns=(
                    "sequence",
                    "timestamp",
                    "actor",
                    "event",
                    "status",
                    "message",
                    "duration_seconds",
                ),
                height=240,
            )
        else:
            st.caption(
                "No explicit stored event is linked to this stage."
            )


def _render_quantum_and_gp_evidence(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Quantum-channel and adaptive-detection evidence",
        icon="⚛️",
    )

    items = (
        KeyValueItem(
            "Observed check blocks",
            (
                f"{snapshot.observed_check_blocks}/"
                f"{snapshot.required_check_blocks}"
                if snapshot.observed_check_blocks is not None
                else "Not available"
            ),
            status=(
                "verified"
                if snapshot.observed_check_blocks is not None
                and snapshot.observed_check_blocks
                >= snapshot.required_check_blocks
                else (
                    "failed"
                    if snapshot.observed_check_blocks is not None
                    else "inactive"
                )
            ),
        ),
        KeyValueItem(
            "Raw QBER",
            format_percentage(snapshot.qber_raw),
        ),
        KeyValueItem(
            "Loss rate",
            format_percentage(snapshot.loss_rate),
        ),
        KeyValueItem(
            "Observable GP features",
            f"{len(snapshot.features)}/9",
            status=(
                "verified"
                if len(snapshot.features) == 9
                else (
                    "warning"
                    if snapshot.features
                    else "inactive"
                )
            ),
        ),
        KeyValueItem(
            "Calibrated P(attack)",
            format_probability(snapshot.p_attack),
        ),
        KeyValueItem(
            "Uncertainty",
            format_probability(snapshot.uncertainty),
        ),
        KeyValueItem(
            "Operational threshold",
            format_probability(
                snapshot.operational_threshold
            ),
            code=True,
        ),
        KeyValueItem(
            "Retry upper boundary",
            format_probability(
                snapshot.retry_upper_probability
            ),
            code=True,
        ),
    )
    render_key_value_grid(items, columns=4)

    st = _streamlit()
    chart_builders = []

    if snapshot.features:
        chart_builders.append(
            (
                "feature_profile",
                lambda: build_feature_profile(
                    snapshot.features,
                    title="Protocol Monitor GP Feature Profile",
                ),
            )
        )

    if snapshot.p_attack is not None:
        chart_builders.append(
            (
                "risk_gauge",
                lambda: build_decision_gauge(
                    snapshot.p_attack,
                    operational_threshold=(
                        snapshot.operational_threshold
                    ),
                    retry_upper_probability=(
                        snapshot.retry_upper_probability
                    ),
                    title="Protocol Monitor Calibrated Attack Risk",
                ),
            )
        )

    if not chart_builders:
        render_empty_state(
            "No GP charts available",
            (
                "The selected session does not contain complete feature or "
                "calibrated-probability evidence."
            ),
            icon="🧠",
        )
        return

    columns = st.columns(len(chart_builders))

    for container, (key, builder) in zip(
        columns,
        chart_builders,
    ):
        with container:
            try:
                render_plotly_figure(
                    builder(),
                    key=f"protocol_monitor_{key}",
                )
            except ChartDataError as exc:
                render_empty_state(
                    "Chart unavailable",
                    str(exc),
                    icon="📉",
                )


def _render_event_log(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Protocol event log",
        icon="🧾",
    )

    if not snapshot.events:
        render_empty_state(
            "No explicit event trace",
            (
                "Stage states were derived conservatively from stored "
                "session fields. Add protocol_events or protocol_trace to "
                "the session record for an exact event timeline."
            ),
            icon="🧾",
        )
        return

    render_records_table(
        [event.to_dictionary() for event in snapshot.events],
        columns=(
            "sequence",
            "timestamp",
            "stage_number",
            "stage_key",
            "actor",
            "event",
            "status",
            "message",
            "duration_seconds",
        ),
        height=420,
    )


def _render_timing_and_attempts(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Timing and retry evidence",
        icon="⏱️",
    )

    st = _streamlit()
    available = []

    if snapshot.timings:
        available.append("timing")

    if snapshot.attempt_history:
        available.append("attempts")

    if not available:
        render_empty_state(
            "No timing or retry trace",
            (
                "The selected session does not contain stage timings or "
                "per-attempt history."
            ),
            icon="⏱️",
        )
        return

    columns = st.columns(len(available))

    for container, item in zip(columns, available):
        with container:
            if item == "timing":
                try:
                    figure = build_timing_breakdown(
                        snapshot.timings,
                        title="End-to-End Protocol Timing Breakdown",
                    )
                    render_plotly_figure(
                        figure,
                        key="protocol_monitor_timing",
                    )
                except ChartDataError as exc:
                    render_empty_state(
                        "Timing chart unavailable",
                        str(exc),
                        icon="📉",
                    )
            else:
                try:
                    figure = build_attempt_timeline(
                        snapshot.attempt_history,
                        title="Bounded Fresh-Retry Timeline",
                    )
                    render_plotly_figure(
                        figure,
                        key="protocol_monitor_attempts",
                    )
                except ChartDataError as exc:
                    render_empty_state(
                        "Attempt chart unavailable",
                        str(exc),
                        icon="📉",
                    )

    if snapshot.attempt_history:
        render_attempt_history(snapshot.attempt_history)


def _render_final_decision(
    snapshot: ProtocolMonitorSnapshot,
) -> None:
    render_section_title(
        "Final authentication policy",
        icon="⚖️",
    )

    if snapshot.accepted is not None:
        render_decision_banner(
            accepted=bool(snapshot.accepted),
            reason=snapshot.decision_reason,
            retry_used=snapshot.retry_used,
            p_attack=snapshot.p_attack,
            threshold=snapshot.operational_threshold,
        )
    else:
        render_banner(
            "Final decision unavailable",
            snapshot.decision_reason,
            status="inactive",
            icon="—",
        )

    if snapshot.deterministic_pass is False:
        render_banner(
            "Deterministic rejection is authoritative",
            (
                "A low GP probability cannot override credential, freshness, "
                "replay, ML-KEM decapsulation, schedule, decoder, KMAC, QBER, "
                "loss, or minimum-evidence failure."
            ),
            status="rejected",
            icon="⛔",
        )

    if snapshot.retry_used:
        render_banner(
            "Fresh retry used",
            (
                "The retry must use a new nonce, request, ephemeral ML-KEM "
                "exchange, transcript, KMAC tag, control schedule, and "
                "quantum payload."
            ),
            status="retry",
            icon="🔁",
        )


def render(
    session: Mapping[str, Any] | None = None,
    *,
    source: str | None = None,
) -> None:
    """Render the complete FT-QuPAP protocol monitor."""

    apply_dashboard_theme()

    snapshot = build_protocol_monitor_snapshot(
        session,
        source=source,
    )

    _render_overview(snapshot)
    render_divider()

    _render_session_identity(snapshot)
    _render_stage_group_summary(snapshot)
    _render_full_pipeline(snapshot)
    _render_playback(snapshot)
    _render_quantum_and_gp_evidence(snapshot)
    _render_event_log(snapshot)
    _render_timing_and_attempts(snapshot)
    _render_final_decision(snapshot)

    render_section_title(
        "Complete session status",
        icon="🛡️",
    )
    render_protocol_status(
        snapshot.safe_session,
        include_hardware=True,
        status_columns=3,
        check_columns=5,
        show_metric_strip=False,
    )

    render_section_title(
        "Safe monitor evidence",
        icon="🔒",
    )
    render_sensitive_data_notice()

    if snapshot.session_available:
        render_json_viewer(
            snapshot.safe_session,
            title="Redacted protocol-monitor session record",
            expanded=False,
        )

    st = _streamlit()
    st.caption(
        (
            "The monitor derives stage state only from stored evidence. "
            "Missing fields remain inactive; no successful stage or metric "
            "is fabricated."
        )
    )


def protocol_monitor_view_status(
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return pure diagnostics for tests and application startup."""

    snapshot = build_protocol_monitor_snapshot(session)

    return {
        "session_available": snapshot.session_available,
        "source": snapshot.source,
        "overall_status": snapshot.overall_status,
        "stage_count": len(snapshot.stages),
        "completed_stage_count": (
            snapshot.completed_stage_count
        ),
        "failed_stage_count": snapshot.failed_stage_count,
        "inactive_stage_count": snapshot.inactive_stage_count,
        "event_count": len(snapshot.events),
        "current_stage_number": snapshot.current_stage_number,
        "first_failed_stage_number": (
            snapshot.first_failed_stage_number
        ),
        "gp_feature_count": len(snapshot.features),
        "accepted": snapshot.accepted,
        "retry_used": snapshot.retry_used,
        "attempt_count": snapshot.attempt_count,
        "sensitive_material_displayed": False,
    }


__all__ = [
    "PROTOCOL_STAGE_COUNT",
    "ProtocolEvent",
    "ProtocolMonitorError",
    "ProtocolMonitorSnapshot",
    "STAGE_GROUPS",
    "StageEvidence",
    "build_protocol_monitor_snapshot",
    "protocol_monitor_view_status",
    "render",
]