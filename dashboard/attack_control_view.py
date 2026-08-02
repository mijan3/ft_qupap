"""
Controlled attack-scenario page for the FT-QuPAP Streamlit dashboard.

This page configures safe, reproducible capstone demonstrations. It does not
perform real network intrusion or target external systems. The controls only
create structured scenario specifications for the local FT-QuPAP simulator.

Supported protocol test classes
-------------------------------
- normal legitimate session
- intercept/resend eavesdropping
- partial quantum interception
- replayed nonce/request
- stale authentication request
- forged or tampered KMAC tag
- invalid server credential
- malformed ML-KEM ciphertext / decapsulation failure
- control-schedule tampering
- excessive quantum-channel loss
- high channel noise
- combined controlled attack

The generated specification can be consumed by the local scenario runner,
for example ``scripts/run_demo_scenarios.py`` or an application session-state
adapter. Secret keys, raw tags, ciphertext bytes, and subscriber identities
are never collected by this page.

Research boundary
-----------------
Dashboard previews and development fixtures are not automatically final
research evidence. Final evidence must come from the preserved notebook or
the approved independent-session evaluation pipeline, with its evidence scope
recorded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .charts import (
    ChartDataError,
    build_attack_probability_comparison,
    build_qber_comparison,
    render_plotly_figure,
)
from .components import (
    KeyValueItem,
    MetricItem,
    format_percentage,
    format_probability,
    render_banner,
    render_card,
    render_empty_state,
    render_json_download,
    render_json_viewer,
    render_key_value_grid,
    render_metric_grid,
    render_records_table,
    sanitize_mapping,
)
from .home import (
    DASHBOARD_RESULTS_FILE,
    DEMO_SESSION_LOGS_FILE,
    load_latest_session,
)
from .theme import (
    apply_dashboard_theme,
    render_divider,
    render_page_header,
    render_research_notice,
    render_section_title,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SCENARIOS_DIR: Final[Path] = PROJECT_ROOT / "scenarios"
DEMO_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "demo"
SCENARIO_EXPORT_FILE: Final[Path] = (
    DEMO_DATA_DIR / "selected_attack_scenario.json"
)

DEFAULT_MASTER_SEED: Final[int] = 20260701
DEFAULT_CONTEXT: Final[str] = "urban"
DEFAULT_BASE_NOISE: Final[float] = 0.01
DEFAULT_BASE_LOSS: Final[float] = 0.01
DEFAULT_EVE_FRACTION: Final[float] = 0.0
DEFAULT_REQUEST_AGE_SECONDS: Final[int] = 0
DEFAULT_REPETITIONS: Final[int] = 1

ALLOWED_CONTEXTS: Final[tuple[str, ...]] = (
    "urban",
    "suburban",
    "rural",
)
ALLOWED_EVE_MODES: Final[tuple[str, ...]] = (
    "none",
    "intercept_resend",
    "random_basis",
    "partial_measurement",
)
ALLOWED_TAMPER_TARGETS: Final[tuple[str, ...]] = (
    "none",
    "kmac_tag",
    "control_schedule",
    "server_credential",
    "mlkem_ciphertext",
    "request_timestamp",
    "request_nonce",
)

MAX_REPETITIONS: Final[int] = 1000
MAX_REQUEST_AGE_SECONDS: Final[int] = 24 * 60 * 60


class AttackControlError(ValueError):
    """Raised when a controlled scenario specification is invalid."""


@dataclass(frozen=True)
class ScenarioDefinition:
    """Immutable metadata for one controlled demonstration scenario."""

    scenario_id: str
    display_name: str
    category: str
    description: str
    expected_policy_response: str
    default_context: str = DEFAULT_CONTEXT
    default_noise: float = DEFAULT_BASE_NOISE
    default_loss: float = DEFAULT_BASE_LOSS
    default_eve_fraction: float = DEFAULT_EVE_FRACTION
    default_eve_mode: str = "none"
    default_tamper_target: str = "none"
    default_request_age_seconds: int = DEFAULT_REQUEST_AGE_SECONDS
    replay_request: bool = False
    research_note: str = (
        "Controlled simulator scenario; evidence eligibility depends on "
        "the run metadata and approved evaluation pipeline."
    )


SCENARIOS: Final[tuple[ScenarioDefinition, ...]] = (
    ScenarioDefinition(
        scenario_id="normal_session",
        display_name="Normal legitimate session",
        category="baseline",
        description=(
            "A registered Mobile Station completes the authenticated PQC "
            "bootstrap and quantum-token exchange under low noise and loss."
        ),
        expected_policy_response="Accept when all mandatory checks pass.",
    ),
    ScenarioDefinition(
        scenario_id="intercept_resend_attack",
        display_name="Intercept/resend eavesdropping",
        category="quantum_attack",
        description=(
            "Eve measures and retransmits a controlled fraction of quantum "
            "blocks, increasing QBER and syndrome disturbance."
        ),
        expected_policy_response=(
            "Reject or classify as high attack risk when disturbance is "
            "sufficiently observable."
        ),
        default_eve_fraction=0.50,
        default_eve_mode="intercept_resend",
    ),
    ScenarioDefinition(
        scenario_id="partial_eavesdropping",
        display_name="Partial quantum interception",
        category="quantum_attack",
        description=(
            "Eve interacts with only part of the transmitted frame to test "
            "adaptive detection under weaker disturbance."
        ),
        expected_policy_response=(
            "Use deterministic evidence plus calibrated P(attack); a bounded "
            "fresh retry may occur only in the approved low-risk gray zone."
        ),
        default_eve_fraction=0.20,
        default_eve_mode="partial_measurement",
    ),
    ScenarioDefinition(
        scenario_id="replay_attack",
        display_name="Replay attack",
        category="classical_attack",
        description=(
            "A previously observed request nonce is reused in the local "
            "simulator replay cache."
        ),
        expected_policy_response=(
            "Hard reject before quantum payload acceptance; GP cannot "
            "override replay protection."
        ),
        default_tamper_target="request_nonce",
        replay_request=True,
    ),
    ScenarioDefinition(
        scenario_id="stale_request",
        display_name="Stale authentication request",
        category="classical_attack",
        description=(
            "The request timestamp falls outside the configured freshness "
            "window."
        ),
        expected_policy_response=(
            "Hard reject during freshness validation."
        ),
        default_tamper_target="request_timestamp",
        default_request_age_seconds=300,
    ),
    ScenarioDefinition(
        scenario_id="forged_kmac_tag",
        display_name="Forged KMAC tag",
        category="cryptographic_attack",
        description=(
            "The recovered authentication payload is altered so the KMAC "
            "comparison fails."
        ),
        expected_policy_response=(
            "Hard reject on constant-time tag mismatch."
        ),
        default_tamper_target="kmac_tag",
    ),
    ScenarioDefinition(
        scenario_id="invalid_server_credential",
        display_name="Invalid server credential",
        category="cryptographic_attack",
        description=(
            "The ML-DSA-authenticated server package is marked invalid in "
            "the local test harness."
        ),
        expected_policy_response=(
            "Mobile Station rejects the server package before ML-KEM "
            "encapsulation."
        ),
        default_tamper_target="server_credential",
    ),
    ScenarioDefinition(
        scenario_id="malformed_mlkem_ciphertext",
        display_name="Malformed ML-KEM ciphertext",
        category="cryptographic_attack",
        description=(
            "The server-side test harness receives malformed or inconsistent "
            "ML-KEM ciphertext metadata."
        ),
        expected_policy_response=(
            "Hard reject on ML-KEM decapsulation failure."
        ),
        default_tamper_target="mlkem_ciphertext",
    ),
    ScenarioDefinition(
        scenario_id="schedule_tampering",
        display_name="Control-schedule tampering",
        category="protocol_attack",
        description=(
            "The protected payload/check control schedule is altered in the "
            "local simulator."
        ),
        expected_policy_response=(
            "Hard reject when transcript or schedule binding fails."
        ),
        default_tamper_target="control_schedule",
    ),
    ScenarioDefinition(
        scenario_id="excessive_loss",
        display_name="Excessive quantum-channel loss",
        category="channel_attack",
        description=(
            "Observed loss exceeds the maximum policy or leaves fewer than "
            "the minimum required check blocks."
        ),
        expected_policy_response=(
            "Reject on loss policy or insufficient independent evidence."
        ),
        default_loss=0.25,
    ),
    ScenarioDefinition(
        scenario_id="high_noise",
        display_name="High channel noise",
        category="channel_attack",
        description=(
            "The quantum channel has elevated bit, phase, and depolarizing "
            "noise without an explicit attacker."
        ),
        expected_policy_response=(
            "Reject, or use a fresh retry only when every deterministic "
            "condition remains retry-eligible."
        ),
        default_noise=0.05,
    ),
    ScenarioDefinition(
        scenario_id="combined_attack",
        display_name="Combined controlled attack",
        category="combined_attack",
        description=(
            "A controlled intercept/resend attack is combined with elevated "
            "noise, loss, or protocol tampering."
        ),
        expected_policy_response=(
            "Reject through deterministic gates and/or high calibrated "
            "attack probability."
        ),
        default_noise=0.04,
        default_loss=0.10,
        default_eve_fraction=0.60,
        default_eve_mode="intercept_resend",
        default_tamper_target="control_schedule",
    ),
)

SCENARIO_BY_ID: Final[dict[str, ScenarioDefinition]] = {
    scenario.scenario_id: scenario
    for scenario in SCENARIOS
}


@dataclass(frozen=True)
class AttackScenarioConfig:
    """Validated local simulator configuration."""

    scenario_id: str
    display_name: str
    category: str
    context: str
    base_noise: float
    base_loss: float
    eve_fraction: float
    eve_mode: str
    tamper_target: str
    replay_request: bool
    request_age_seconds: int
    repetitions: int
    master_seed: int
    evidence_scope: str
    notes: str
    expected_policy_response: str
    created_at_utc: str
    configuration_id: str

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible scenario specification."""

        return sanitize_mapping(asdict(self))


@dataclass(frozen=True)
class AttackControlSnapshot:
    """Safe state for the attack-control page."""

    selected_scenario: ScenarioDefinition
    latest_session: Mapping[str, Any] | None
    latest_session_source: str | None
    available_result_rows: tuple[Mapping[str, Any], ...]
    scenario_count: int
    categories: tuple[str, ...]
    selected_configuration: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dictionary(self) -> dict[str, Any]:
        return sanitize_mapping(asdict(self))


def _streamlit() -> Any:
    """Import Streamlit only when rendering is required."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to render attack_control_view.py."
        ) from exc

    return st


def _pandas() -> Any:
    """Import pandas lazily for stored result charts."""

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for stored attack-result charts."
        ) from exc

    return pd


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate a bounded numeric value."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AttackControlError(
            f"{name} must be numeric."
        ) from exc

    if not math.isfinite(number):
        raise AttackControlError(
            f"{name} must be finite."
        )

    if number < minimum or number > maximum:
        raise AttackControlError(
            f"{name} must be in [{minimum}, {maximum}]."
        )

    return number


def _bounded_int(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Validate a bounded integer."""

    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AttackControlError(
            f"{name} must be an integer."
        ) from exc

    if number < minimum or number > maximum:
        raise AttackControlError(
            f"{name} must be in [{minimum}, {maximum}]."
        )

    return number


def _safe_read_result_rows() -> tuple[Mapping[str, Any], ...]:
    """Read stored dashboard/demo rows when pandas is available."""

    rows: list[Mapping[str, Any]] = []

    try:
        pd = _pandas()
    except RuntimeError:
        return ()

    for path in (
        DASHBOARD_RESULTS_FILE,
        DEMO_SESSION_LOGS_FILE,
    ):
        if not path.is_file():
            continue

        try:
            table = pd.read_csv(path)
        except Exception:
            continue

        if table.empty:
            continue

        for record in table.to_dict(orient="records"):
            safe = sanitize_mapping(record)

            if isinstance(safe, Mapping):
                rows.append(dict(safe))

    return tuple(rows)


def get_scenario(scenario_id: str) -> ScenarioDefinition:
    """Return one registered scenario definition."""

    normalized = str(scenario_id).strip().lower()

    try:
        return SCENARIO_BY_ID[normalized]
    except KeyError as exc:
        raise AttackControlError(
            f"Unknown scenario_id: {scenario_id!r}."
        ) from exc


def _configuration_digest(payload: Mapping[str, Any]) -> str:
    """Create a stable ID for a non-secret scenario specification."""

    serialized = json.dumps(
        sanitize_mapping(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return (
        "ATKCFG-"
        + hashlib.sha256(serialized).hexdigest()[:16].upper()
    )


def create_attack_scenario_config(
    *,
    scenario_id: str,
    context: str,
    base_noise: float,
    base_loss: float,
    eve_fraction: float,
    eve_mode: str,
    tamper_target: str,
    replay_request: bool,
    request_age_seconds: int,
    repetitions: int,
    master_seed: int,
    evidence_scope: str = "synthetic_demo_development_only",
    notes: str = "",
) -> AttackScenarioConfig:
    """Validate and create a local simulator scenario specification."""

    scenario = get_scenario(scenario_id)

    normalized_context = str(context).strip().lower()

    if normalized_context not in ALLOWED_CONTEXTS:
        raise AttackControlError(
            f"context must be one of {ALLOWED_CONTEXTS}."
        )

    normalized_eve_mode = str(eve_mode).strip().lower()

    if normalized_eve_mode not in ALLOWED_EVE_MODES:
        raise AttackControlError(
            f"eve_mode must be one of {ALLOWED_EVE_MODES}."
        )

    normalized_tamper = str(tamper_target).strip().lower()

    if normalized_tamper not in ALLOWED_TAMPER_TARGETS:
        raise AttackControlError(
            f"tamper_target must be one of {ALLOWED_TAMPER_TARGETS}."
        )

    resolved_noise = _finite_float(
        base_noise,
        name="base_noise",
        minimum=0.0,
        maximum=0.20,
    )
    resolved_loss = _finite_float(
        base_loss,
        name="base_loss",
        minimum=0.0,
        maximum=1.0,
    )
    resolved_eve = _finite_float(
        eve_fraction,
        name="eve_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    resolved_age = _bounded_int(
        request_age_seconds,
        name="request_age_seconds",
        minimum=0,
        maximum=MAX_REQUEST_AGE_SECONDS,
    )
    resolved_repetitions = _bounded_int(
        repetitions,
        name="repetitions",
        minimum=1,
        maximum=MAX_REPETITIONS,
    )
    resolved_seed = _bounded_int(
        master_seed,
        name="master_seed",
        minimum=0,
        maximum=2**63 - 1,
    )

    if resolved_eve > 0.0 and normalized_eve_mode == "none":
        raise AttackControlError(
            "eve_mode cannot be 'none' when eve_fraction is positive."
        )

    if resolved_eve == 0.0 and normalized_eve_mode != "none":
        raise AttackControlError(
            "eve_fraction must be positive when an Eve mode is selected."
        )

    if normalized_tamper == "request_nonce" and not replay_request:
        raise AttackControlError(
            "request_nonce tampering requires replay_request=True."
        )

    scope = str(evidence_scope).strip()

    if not scope:
        raise AttackControlError(
            "evidence_scope is required."
        )

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    base_payload = {
        "scenario_id": scenario.scenario_id,
        "context": normalized_context,
        "base_noise": resolved_noise,
        "base_loss": resolved_loss,
        "eve_fraction": resolved_eve,
        "eve_mode": normalized_eve_mode,
        "tamper_target": normalized_tamper,
        "replay_request": bool(replay_request),
        "request_age_seconds": resolved_age,
        "repetitions": resolved_repetitions,
        "master_seed": resolved_seed,
        "evidence_scope": scope,
        "notes": notes.strip(),
    }

    return AttackScenarioConfig(
        scenario_id=scenario.scenario_id,
        display_name=scenario.display_name,
        category=scenario.category,
        context=normalized_context,
        base_noise=resolved_noise,
        base_loss=resolved_loss,
        eve_fraction=resolved_eve,
        eve_mode=normalized_eve_mode,
        tamper_target=normalized_tamper,
        replay_request=bool(replay_request),
        request_age_seconds=resolved_age,
        repetitions=resolved_repetitions,
        master_seed=resolved_seed,
        evidence_scope=scope,
        notes=notes.strip(),
        expected_policy_response=scenario.expected_policy_response,
        created_at_utc=created_at,
        configuration_id=_configuration_digest(base_payload),
    )


def default_config_for_scenario(
    scenario_id: str,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> AttackScenarioConfig:
    """Create the validated default configuration for one scenario."""

    scenario = get_scenario(scenario_id)

    return create_attack_scenario_config(
        scenario_id=scenario.scenario_id,
        context=scenario.default_context,
        base_noise=scenario.default_noise,
        base_loss=scenario.default_loss,
        eve_fraction=scenario.default_eve_fraction,
        eve_mode=scenario.default_eve_mode,
        tamper_target=scenario.default_tamper_target,
        replay_request=scenario.replay_request,
        request_age_seconds=(
            scenario.default_request_age_seconds
        ),
        repetitions=DEFAULT_REPETITIONS,
        master_seed=master_seed,
    )


def build_attack_control_snapshot(
    *,
    selected_scenario_id: str = "normal_session",
    selected_configuration: Mapping[str, Any] | None = None,
) -> AttackControlSnapshot:
    """Build pure state for rendering and startup diagnostics."""

    scenario = get_scenario(selected_scenario_id)
    latest, latest_source, _ = load_latest_session()
    warnings = []

    if not _safe_read_result_rows():
        warnings.append(
            "No stored controlled-scenario result rows are available."
        )

    if selected_configuration is not None:
        safe_configuration = sanitize_mapping(
            selected_configuration
        )

        if not isinstance(safe_configuration, Mapping):
            safe_configuration = None
    else:
        safe_configuration = None

    return AttackControlSnapshot(
        selected_scenario=scenario,
        latest_session=latest,
        latest_session_source=latest_source,
        available_result_rows=_safe_read_result_rows(),
        scenario_count=len(SCENARIOS),
        categories=tuple(
            sorted({scenario.category for scenario in SCENARIOS})
        ),
        selected_configuration=safe_configuration,
        warnings=tuple(warnings),
    )


def _render_overview() -> None:
    """Render the page header and safety boundary."""

    render_page_header(
        title="Attack Control",
        subtitle=(
            "Configure reproducible, local FT-QuPAP security scenarios "
            "without targeting external systems"
        ),
        icon="⚔️",
        status="ready",
    )

    render_research_notice(
        (
            "This page creates controlled simulator configurations. "
            "Development previews are not final research evidence unless "
            "their run metadata explicitly marks them eligible."
        )
    )

    render_banner(
        "Controlled environment only",
        (
            "All controls apply to the local FT-QuPAP simulator. The page "
            "does not scan networks, intercept real traffic, recover keys, "
            "or attack third-party systems."
        ),
        status="verified",
        icon="🧪",
    )


def _render_scenario_catalog() -> None:
    """Render compact cards for all registered scenarios."""

    render_section_title(
        "Scenario catalog",
        icon="🗂️",
    )

    st = _streamlit()

    for start in range(0, len(SCENARIOS), 3):
        row = SCENARIOS[start : start + 3]
        columns = st.columns(len(row))

        for container, scenario in zip(columns, row):
            with container:
                render_card(
                    title=scenario.display_name,
                    body=scenario.description,
                    footer=(
                        "Expected response: "
                        + scenario.expected_policy_response
                    ),
                    icon=(
                        "✅"
                        if scenario.category == "baseline"
                        else "⚔️"
                    ),
                    status=(
                        "verified"
                        if scenario.category == "baseline"
                        else "attack"
                    ),
                )


def _render_configuration_form() -> Mapping[str, Any] | None:
    """Render scenario controls and return the active safe config."""

    render_section_title(
        "Controlled scenario configuration",
        icon="🎛️",
    )

    st = _streamlit()

    display_to_id = {
        scenario.display_name: scenario.scenario_id
        for scenario in SCENARIOS
    }

    selected_display = st.selectbox(
        "Scenario",
        tuple(display_to_id),
        key="attack_control_scenario",
    )
    scenario = get_scenario(
        display_to_id[selected_display]
    )

    st.caption(scenario.description)

    left, middle, right = st.columns(3)

    with left:
        context = st.selectbox(
            "Channel context",
            ALLOWED_CONTEXTS,
            index=ALLOWED_CONTEXTS.index(
                scenario.default_context
            ),
            key=f"attack_context_{scenario.scenario_id}",
        )
        base_noise = st.slider(
            "Base channel noise",
            min_value=0.0,
            max_value=0.20,
            value=float(scenario.default_noise),
            step=0.005,
            key=f"attack_noise_{scenario.scenario_id}",
        )
        base_loss = st.slider(
            "Base channel loss",
            min_value=0.0,
            max_value=1.0,
            value=float(scenario.default_loss),
            step=0.01,
            key=f"attack_loss_{scenario.scenario_id}",
        )

    with middle:
        eve_fraction = st.slider(
            "Eve interception fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(scenario.default_eve_fraction),
            step=0.05,
            key=f"attack_eve_fraction_{scenario.scenario_id}",
        )
        eve_mode = st.selectbox(
            "Eve mode",
            ALLOWED_EVE_MODES,
            index=ALLOWED_EVE_MODES.index(
                scenario.default_eve_mode
            ),
            key=f"attack_eve_mode_{scenario.scenario_id}",
        )
        tamper_target = st.selectbox(
            "Tamper target",
            ALLOWED_TAMPER_TARGETS,
            index=ALLOWED_TAMPER_TARGETS.index(
                scenario.default_tamper_target
            ),
            key=f"attack_tamper_{scenario.scenario_id}",
        )

    with right:
        replay_request = st.checkbox(
            "Replay an existing request",
            value=scenario.replay_request,
            key=f"attack_replay_{scenario.scenario_id}",
        )
        request_age_seconds = st.number_input(
            "Request age (seconds)",
            min_value=0,
            max_value=MAX_REQUEST_AGE_SECONDS,
            value=int(
                scenario.default_request_age_seconds
            ),
            step=10,
            key=f"attack_age_{scenario.scenario_id}",
        )
        repetitions = st.number_input(
            "Repetitions",
            min_value=1,
            max_value=MAX_REPETITIONS,
            value=DEFAULT_REPETITIONS,
            step=1,
            key=f"attack_repetitions_{scenario.scenario_id}",
        )
        master_seed = st.number_input(
            "Master seed",
            min_value=0,
            max_value=2**63 - 1,
            value=DEFAULT_MASTER_SEED,
            step=1,
            key=f"attack_seed_{scenario.scenario_id}",
        )

    notes = st.text_area(
        "Run notes",
        value="",
        max_chars=500,
        key=f"attack_notes_{scenario.scenario_id}",
    )

    prepared = st.button(
        "Prepare controlled scenario",
        type="primary",
        use_container_width=True,
        key="prepare_attack_scenario",
    )

    if prepared:
        try:
            configuration = create_attack_scenario_config(
                scenario_id=scenario.scenario_id,
                context=context,
                base_noise=base_noise,
                base_loss=base_loss,
                eve_fraction=eve_fraction,
                eve_mode=eve_mode,
                tamper_target=tamper_target,
                replay_request=replay_request,
                request_age_seconds=int(request_age_seconds),
                repetitions=int(repetitions),
                master_seed=int(master_seed),
                notes=notes,
            )
            st.session_state[
                "ft_qupap_attack_configuration"
            ] = configuration.to_dictionary()
        except AttackControlError as exc:
            st.error(str(exc))

    active = st.session_state.get(
        "ft_qupap_attack_configuration"
    )

    if not isinstance(active, Mapping):
        try:
            active = default_config_for_scenario(
                scenario.scenario_id
            ).to_dictionary()
        except AttackControlError:
            return None

    return sanitize_mapping(active)


def _render_active_configuration(
    configuration: Mapping[str, Any] | None,
) -> None:
    """Render the selected scenario specification."""

    render_section_title(
        "Active scenario specification",
        icon="📋",
    )

    if not isinstance(configuration, Mapping):
        render_empty_state(
            "No valid scenario specification",
            (
                "Correct the scenario controls and prepare the "
                "configuration again."
            ),
            icon="⚠️",
        )
        return

    eve_fraction = configuration.get("eve_fraction")
    noise = configuration.get("base_noise")
    loss = configuration.get("base_loss")

    metrics = (
        MetricItem(
            "Scenario",
            configuration.get("display_name", "—"),
            status=(
                "verified"
                if configuration.get("category") == "baseline"
                else "attack"
            ),
            icon="🧪",
        ),
        MetricItem(
            "Noise",
            format_percentage(noise),
            status=(
                "warning"
                if isinstance(noise, (int, float))
                and noise >= 0.04
                else "ready"
            ),
            icon="〰️",
        ),
        MetricItem(
            "Loss",
            format_percentage(loss),
            status=(
                "failed"
                if isinstance(loss, (int, float))
                and loss > 0.15
                else "ready"
            ),
            icon="📶",
        ),
        MetricItem(
            "Eve fraction",
            format_percentage(eve_fraction),
            status=(
                "attack"
                if isinstance(eve_fraction, (int, float))
                and eve_fraction > 0
                else "inactive"
            ),
            icon="👁️",
        ),
        MetricItem(
            "Repetitions",
            configuration.get("repetitions", "—"),
            status="ready",
            icon="🔁",
        ),
        MetricItem(
            "Configuration ID",
            configuration.get("configuration_id", "—"),
            status="ready",
            icon="🆔",
        ),
    )
    render_metric_grid(metrics, columns=6)

    items = (
        KeyValueItem(
            "Scenario ID",
            configuration.get("scenario_id"),
            code=True,
        ),
        KeyValueItem(
            "Category",
            configuration.get("category"),
        ),
        KeyValueItem(
            "Context",
            configuration.get("context"),
        ),
        KeyValueItem(
            "Eve mode",
            configuration.get("eve_mode"),
            code=True,
        ),
        KeyValueItem(
            "Tamper target",
            configuration.get("tamper_target"),
            code=True,
        ),
        KeyValueItem(
            "Replay request",
            configuration.get("replay_request"),
        ),
        KeyValueItem(
            "Request age",
            (
                f"{configuration.get('request_age_seconds', 0)} seconds"
            ),
        ),
        KeyValueItem(
            "Master seed",
            configuration.get("master_seed"),
            code=True,
        ),
        KeyValueItem(
            "Evidence scope",
            configuration.get("evidence_scope"),
            status="warning",
        ),
        KeyValueItem(
            "Expected policy response",
            configuration.get("expected_policy_response"),
        ),
    )
    render_key_value_grid(items, columns=5)

    render_json_viewer(
        configuration,
        title="Safe scenario JSON",
        expanded=False,
    )
    render_json_download(
        configuration,
        filename="selected_attack_scenario.json",
        label="Download scenario JSON",
        key="download_attack_scenario",
    )

    render_banner(
        "Runner integration",
        (
            "The downloaded JSON is a local simulator specification. "
            "Pass its fields to the controlled scenario runner; do not "
            "treat the preview itself as an executed authentication result."
        ),
        status="ready",
        icon="▶️",
    )


def _render_policy_expectation(
    configuration: Mapping[str, Any] | None,
) -> None:
    """Explain how FT-QuPAP should react to the active scenario."""

    render_section_title(
        "Expected protocol reaction",
        icon="🛡️",
    )

    if not isinstance(configuration, Mapping):
        render_empty_state(
            "No policy expectation",
            "Prepare a valid scenario first.",
            icon="🛡️",
        )
        return

    scenario = get_scenario(
        str(configuration.get("scenario_id"))
    )

    render_banner(
        scenario.display_name,
        scenario.expected_policy_response,
        status=(
            "verified"
            if scenario.category == "baseline"
            else "warning"
        ),
        icon="⚖️",
    )

    st = _streamlit()
    columns = st.columns(3)

    with columns[0]:
        render_card(
            title="Deterministic gates",
            body=(
                "Credential, freshness, replay, ML-KEM decapsulation, "
                "schedule binding, minimum check evidence, decoder, KMAC "
                "tag, QBER, and loss rules remain mandatory."
            ),
            icon="🚧",
            status="verified",
        )

    with columns[1]:
        render_card(
            title="Adaptive GP",
            body=(
                "The GP uses nine observable session features and returns "
                "calibrated P(attack). It supplements hard checks and cannot "
                "override a deterministic failure."
            ),
            icon="🧠",
            status="ready",
        )

    with columns[2]:
        render_card(
            title="Fresh retry",
            body=(
                "A retry is allowed only for approved low-risk conditions "
                "and creates a new nonce, ML-KEM exchange, transcript, tag, "
                "schedule, and quantum payload."
            ),
            icon="🔁",
            status="retry",
        )


def _result_dataframe(
    rows: Sequence[Mapping[str, Any]],
) -> Any | None:
    """Return a pandas dataframe for available stored results."""

    if not rows:
        return None

    try:
        pd = _pandas()
    except RuntimeError:
        return None

    table = pd.DataFrame(
        [sanitize_mapping(row) for row in rows]
    )

    return table if not table.empty else None


def _render_stored_results(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Render scenario history and chart-ready stored evidence."""

    render_section_title(
        "Stored controlled-scenario outcomes",
        icon="📊",
    )

    if not rows:
        render_empty_state(
            "No scenario outcomes available",
            (
                "Run scripts/run_demo_scenarios.py to create stored "
                "controlled-session evidence."
            ),
            icon="📭",
        )
        return

    render_records_table(
        rows[-100:],
        columns=(
            "executed_at_utc",
            "scenario_id",
            "accepted",
            "reason",
            "qber_raw",
            "loss_rate",
            "p_attack",
            "retry_used",
            "retry_attempts",
            "evidence_scope",
        ),
        height=380,
    )

    table = _result_dataframe(rows)

    if table is None:
        return

    st = _streamlit()
    left, right = st.columns(2)

    with left:
        try:
            qber_figure = build_qber_comparison(
                table,
                fixed_qber_threshold=0.11,
            )
            render_plotly_figure(
                qber_figure,
                key="attack_control_qber_comparison",
            )
        except (ChartDataError, KeyError, TypeError) as exc:
            render_empty_state(
                "QBER comparison unavailable",
                str(exc),
                icon="📉",
            )

    with right:
        try:
            attack_figure = (
                build_attack_probability_comparison(
                    table,
                    operational_threshold=0.15,
                )
            )
            render_plotly_figure(
                attack_figure,
                key="attack_control_probability_comparison",
            )
        except (ChartDataError, KeyError, TypeError) as exc:
            render_empty_state(
                "Attack-probability comparison unavailable",
                str(exc),
                icon="📉",
            )


def _render_latest_session_context(
    snapshot: AttackControlSnapshot,
) -> None:
    """Show a compact latest-session reference."""

    render_section_title(
        "Latest executed session reference",
        icon="🕘",
    )

    if snapshot.latest_session is None:
        render_empty_state(
            "No executed session available",
            (
                "Preparing a scenario does not execute it. Run the controlled "
                "scenario pipeline to create a session record."
            ),
            icon="📭",
        )
        return

    session = snapshot.latest_session
    decision = (
        session.get("decision")
        if isinstance(session.get("decision"), Mapping)
        else {}
    )

    accepted = decision.get(
        "accepted",
        session.get("accepted"),
    )
    p_attack = decision.get(
        "p_attack",
        session.get("p_attack"),
    )

    items = (
        KeyValueItem(
            "Source",
            snapshot.latest_session_source or "runtime session",
            code=True,
        ),
        KeyValueItem(
            "Scenario",
            session.get(
                "scenario_id",
                session.get("scenario", "—"),
            ),
        ),
        KeyValueItem(
            "Accepted",
            accepted,
            status=(
                "accepted"
                if accepted is True
                else (
                    "rejected"
                    if accepted is False
                    else "inactive"
                )
            ),
        ),
        KeyValueItem(
            "Reason",
            decision.get(
                "reason",
                session.get("reason", "—"),
            ),
        ),
        KeyValueItem(
            "Raw QBER",
            format_percentage(
                session.get(
                    "qber_raw",
                    session.get("raw_qber"),
                )
            ),
        ),
        KeyValueItem(
            "P(attack)",
            format_probability(p_attack),
        ),
    )
    render_key_value_grid(items, columns=3)


def render() -> None:
    """Render the FT-QuPAP controlled attack-scenario page."""

    apply_dashboard_theme()
    _render_overview()
    render_divider()

    _render_scenario_catalog()
    configuration = _render_configuration_form()
    snapshot = build_attack_control_snapshot(
        selected_scenario_id=(
            str(configuration.get("scenario_id"))
            if isinstance(configuration, Mapping)
            else "normal_session"
        ),
        selected_configuration=configuration,
    )

    _render_active_configuration(configuration)
    _render_policy_expectation(configuration)
    _render_latest_session_context(snapshot)
    _render_stored_results(snapshot.available_result_rows)

    if snapshot.warnings:
        render_section_title(
            "Readiness notes",
            icon="⚠️",
        )

        for warning in snapshot.warnings:
            render_banner(
                "Attention",
                warning,
                status="warning",
            )

    st = _streamlit()
    st.caption(
        (
            f"{snapshot.scenario_count} controlled scenarios are registered "
            "for the local FT-QuPAP demonstrator. Configuration previews do "
            "not execute attacks or contact external systems."
        )
    )


def attack_control_view_status() -> dict[str, Any]:
    """Return pure startup diagnostics."""

    default = default_config_for_scenario(
        "normal_session"
    )
    combined = default_config_for_scenario(
        "combined_attack"
    )

    return {
        "scenario_count": len(SCENARIOS),
        "scenario_ids": [
            scenario.scenario_id for scenario in SCENARIOS
        ],
        "categories": sorted(
            {scenario.category for scenario in SCENARIOS}
        ),
        "default_configuration_valid": bool(
            default.configuration_id
        ),
        "combined_attack_eve_fraction": (
            combined.eve_fraction
        ),
        "combined_attack_tamper_target": (
            combined.tamper_target
        ),
        "external_targeting_supported": False,
        "secret_material_required": False,
    }


__all__ = [
    "ALLOWED_CONTEXTS",
    "ALLOWED_EVE_MODES",
    "ALLOWED_TAMPER_TARGETS",
    "AttackControlError",
    "AttackControlSnapshot",
    "AttackScenarioConfig",
    "SCENARIOS",
    "SCENARIO_BY_ID",
    "SCENARIO_EXPORT_FILE",
    "ScenarioDefinition",
    "attack_control_view_status",
    "build_attack_control_snapshot",
    "create_attack_scenario_config",
    "default_config_for_scenario",
    "get_scenario",
    "render",
]