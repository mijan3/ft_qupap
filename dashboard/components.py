"""
Reusable UI components for the FT-QuPAP Streamlit dashboard.

The module contains generic presentation building blocks shared by all
dashboard pages:

- cards and metric tiles
- protocol-stage stepper
- decision and information banners
- key/value grids
- GP feature tables
- retry-attempt history
- timing tables
- safe JSON viewers and downloads
- empty states and asset images

Security boundary
-----------------
Dashboard components must never expose ML-DSA or ML-KEM secret keys, K_ss,
K_auth, K_ctrl, raw subscriber identities, reusable authentication tags, or
raw ciphertext material. Mapping and JSON helpers therefore redact common
secret-field names before rendering or downloading.

The module imports Streamlit lazily. Pure formatting and HTML helpers can be
unit-tested without starting a Streamlit application.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from html import escape
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from .theme import (
    THEME,
    normalize_status,
    protocol_stage_color,
    status_badge_html,
    status_style,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"
IMAGES_DIR: Final[Path] = ASSETS_DIR / "images"
ICONS_DIR: Final[Path] = ASSETS_DIR / "icons"

DEFAULT_VALUE_PLACEHOLDER: Final[str] = "—"
DEFAULT_TABLE_HEIGHT: Final[int] = 360

SENSITIVE_FIELD_TOKENS: Final[tuple[str, ...]] = (
    "private_key",
    "secret_key",
    "shared_secret",
    "session_secret",
    "key_material",
    "k_auth",
    "k_ctrl",
    "k_ss",
    "ciphertext",
    "raw_tag",
    "authentication_tag",
    "received_tag",
    "expected_tag",
    "subscriber_identity",
    "raw_identity",
    "imsi",
    "password",
    "api_key",
    "access_token",
)

SAFE_FIELD_EXCEPTIONS: Final[set[str]] = {
    "private_key_present",
    "secret_key_present",
    "shared_secret_match",
    "ciphertext_valid",
    "tag_recovered",
    "tag_match",
    "tag_verified",
    "authentication_tag_valid",
    "server_signature_valid",
    "credential_valid",
}

GP_FEATURE_LABELS: Final[dict[str, str]] = {
    "qber_raw": "Raw QBER",
    "mean_syndrome_weight": "Mean syndrome weight",
    "max_syndrome_weight": "Maximum syndrome weight",
    "correction_failure_rate": "Correction failure rate",
    "loss_rate": "Loss rate",
    "noise_estimate": "Noise estimate",
    "ctx_urban": "Urban context",
    "ctx_suburban": "Suburban context",
    "ctx_rural": "Rural context",
}

DEFAULT_PROTOCOL_STEPS: Final[tuple["ProtocolStep", ...]]


class ComponentError(ValueError):
    """Raised when a dashboard component receives invalid data."""


@dataclass(frozen=True)
class MetricItem:
    """One generic dashboard metric."""

    label: str
    value: Any
    delta: str | None = None
    help_text: str | None = None
    status: str | None = None
    icon: str = ""


@dataclass(frozen=True)
class KeyValueItem:
    """One label/value pair shown in a detail grid."""

    label: str
    value: Any
    help_text: str | None = None
    code: bool = False
    status: str | None = None


@dataclass(frozen=True)
class ProtocolStep:
    """One visual step in the FT-QuPAP online protocol."""

    number: int
    key: str
    label: str
    stage: str
    description: str = ""
    owner: str = ""
    status: str = "inactive"


@dataclass(frozen=True)
class Banner:
    """Semantic dashboard message."""

    title: str
    message: str
    status: str = "ready"
    icon: str | None = None


DEFAULT_PROTOCOL_STEPS = (
    ProtocolStep(
        1,
        "authentication_request",
        "Authentication request",
        "request",
        "Mobile station sends pseudonym, timestamp, nonce, and context.",
        "Mobile Station",
    ),
    ProtocolStep(
        2,
        "freshness_replay",
        "Freshness and replay checks",
        "freshness",
        "Authentication server validates time window and nonce reuse.",
        "Authentication Server",
    ),
    ProtocolStep(
        3,
        "server_credential",
        "ML-DSA server package",
        "ml_dsa",
        "Server signs the bootstrapping package with ML-DSA-65.",
        "Authentication Server",
    ),
    ProtocolStep(
        4,
        "credential_verification",
        "Verify server credential",
        "ml_dsa",
        "Mobile station verifies the trusted ML-DSA-65 signature.",
        "Mobile Station",
    ),
    ProtocolStep(
        5,
        "mlkem_encapsulation",
        "ML-KEM encapsulation",
        "ml_kem",
        "Mobile station encapsulates a fresh ML-KEM-768 shared secret.",
        "Mobile Station",
    ),
    ProtocolStep(
        6,
        "mlkem_decapsulation",
        "ML-KEM decapsulation",
        "ml_kem",
        "Authentication server decapsulates the received ciphertext.",
        "Authentication Server",
    ),
    ProtocolStep(
        7,
        "session_key_derivation",
        "Transcript-bound key derivation",
        "key_derivation",
        "Both parties derive K_auth and K_ctrl from the fresh session.",
        "Both",
    ),
    ProtocolStep(
        8,
        "kmac_tag",
        "KMAC authentication tag",
        "kmac",
        "Mobile station creates the 128-bit transcript-bound KMAC tag.",
        "Mobile Station",
    ),
    ProtocolStep(
        9,
        "control_schedule",
        "Quantum control schedule",
        "schedule",
        "K_ctrl defines payload/check positions and measurement controls.",
        "Both",
    ),
    ProtocolStep(
        10,
        "logical_blocks",
        "Logical payload and checks",
        "schedule",
        "Prepare 128 payload blocks and 32 independent check blocks.",
        "Mobile Station",
    ),
    ProtocolStep(
        11,
        "steane_encoding",
        "Steane [[7,1,3]] encoding",
        "steane_encoding",
        "Encode each logical block into seven physical data qubits.",
        "Mobile Station",
    ),
    ProtocolStep(
        12,
        "quantum_channel",
        "Noisy or untrusted channel",
        "quantum_channel",
        "Transmit encoded blocks through noise, loss, and optional Eve.",
        "Quantum Channel",
    ),
    ProtocolStep(
        13,
        "measurement",
        "Controlled measurement",
        "measurement",
        "Authentication server measures blocks using the derived schedule.",
        "Authentication Server",
    ),
    ProtocolStep(
        14,
        "raw_qber",
        "Raw QBER calculation",
        "qber",
        "Calculate mismatch rate from independent observed check blocks.",
        "Authentication Server",
    ),
    ProtocolStep(
        15,
        "syndrome_processing",
        "Syndrome processing",
        "syndrome",
        "Extract Steane syndromes and attempt bounded correction.",
        "Authentication Server",
    ),
    ProtocolStep(
        16,
        "payload_and_tag",
        "Payload recovery and tag verification",
        "tag_verification",
        "Recover the tag payload and compare it in constant time.",
        "Authentication Server",
    ),
    ProtocolStep(
        17,
        "gp_features",
        "Observable GP features",
        "gp_detection",
        "Build the nine session-level features without simulator leakage.",
        "Authentication Server",
    ),
    ProtocolStep(
        18,
        "gp_probability",
        "Calibrated P(attack)",
        "gp_detection",
        "Scale features, infer GP probability, and apply calibration.",
        "Authentication Server",
    ),
    ProtocolStep(
        19,
        "decision_retry",
        "Decision or fresh retry",
        "decision",
        "Accept, reject, or retry with a fresh nonce and ML-KEM session.",
        "Authentication Server",
    ),
)


def _streamlit() -> Any:
    """Import Streamlit lazily with an actionable error."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required for dashboard rendering. "
            "Install the project requirements first."
        ) from exc

    return st


def _pandas() -> Any:
    """Import pandas lazily."""

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for dashboard tables."
        ) from exc

    return pd


def _finite_number(value: Any) -> float | None:
    """Return a finite float or None."""

    if isinstance(value, bool) or value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def format_probability(
    value: Any,
    *,
    decimals: int = 3,
    placeholder: str = DEFAULT_VALUE_PLACEHOLDER,
) -> str:
    """Format a probability in [0, 1]."""

    number = _finite_number(value)

    if number is None:
        return placeholder

    return f"{min(max(number, 0.0), 1.0):.{decimals}f}"


def format_percentage(
    value: Any,
    *,
    decimals: int = 1,
    placeholder: str = DEFAULT_VALUE_PLACEHOLDER,
) -> str:
    """Format a ratio as a percentage."""

    number = _finite_number(value)

    if number is None:
        return placeholder

    return f"{number * 100.0:.{decimals}f}%"


def format_duration(
    seconds: Any,
    *,
    placeholder: str = DEFAULT_VALUE_PLACEHOLDER,
) -> str:
    """Format seconds using an appropriate unit."""

    number = _finite_number(seconds)

    if number is None or number < 0:
        return placeholder

    if number < 0.001:
        return f"{number * 1_000_000:.1f} µs"

    if number < 1.0:
        return f"{number * 1_000:.2f} ms"

    if number < 60.0:
        return f"{number:.3f} s"

    minutes, remaining = divmod(number, 60.0)
    return f"{int(minutes)} min {remaining:.1f} s"


def format_bytes(
    value: Any,
    *,
    placeholder: str = DEFAULT_VALUE_PLACEHOLDER,
) -> str:
    """Format a byte count."""

    number = _finite_number(value)

    if number is None or number < 0:
        return placeholder

    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = number

    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return (
                f"{amount:.0f} {unit}"
                if unit == "B"
                else f"{amount:.2f} {unit}"
            )
        amount /= 1024.0

    return placeholder


def format_value(
    value: Any,
    *,
    placeholder: str = DEFAULT_VALUE_PLACEHOLDER,
    max_length: int = 120,
) -> str:
    """Convert a dashboard value to readable non-secret text."""

    if value is None:
        return placeholder

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, float):
        if not math.isfinite(value):
            return placeholder
        return f"{value:.6g}"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple, set)):
        text = ", ".join(format_value(item) for item in value)
    elif isinstance(value, Mapping):
        text = json.dumps(
            sanitize_mapping(value),
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        text = str(value)

    if len(text) > max_length:
        return text[: max_length - 1] + "…"

    return text


def is_sensitive_field(name: str) -> bool:
    """Return whether a field name represents secret/session material."""

    normalized = (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in SAFE_FIELD_EXCEPTIONS:
        return False

    return any(token in normalized for token in SENSITIVE_FIELD_TOKENS)


def _convert_mapping_value(value: Any, *, key_name: str = "") -> Any:
    """Recursively produce JSON-safe redacted values."""

    if key_name and is_sensitive_field(key_name):
        if isinstance(value, (bytes, bytearray, memoryview)):
            return f"<redacted:{len(value)} bytes>"
        return "<redacted>"

    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<redacted:{len(value)} bytes>"

    if is_dataclass(value):
        return _convert_mapping_value(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _convert_mapping_value(
                item,
                key_name=str(key),
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_convert_mapping_value(item) for item in value]

    item_method = getattr(value, "item", None)

    if callable(item_method):
        try:
            return _convert_mapping_value(item_method())
        except Exception:
            pass

    list_method = getattr(value, "tolist", None)

    if callable(list_method):
        try:
            return _convert_mapping_value(list_method())
        except Exception:
            pass

    return str(value)


def sanitize_mapping(
    payload: Mapping[str, Any] | Any,
) -> Any:
    """Return a JSON-safe representation with sensitive fields redacted."""

    return _convert_mapping_value(payload)


def safe_json_text(
    payload: Any,
    *,
    indent: int = 2,
) -> str:
    """Serialize redacted dashboard data as JSON."""

    return json.dumps(
        sanitize_mapping(payload),
        indent=indent,
        sort_keys=True,
        ensure_ascii=False,
    )


def card_html(
    *,
    title: str,
    body: str = "",
    icon: str = "",
    footer: str = "",
    status: str | None = None,
    elevated: bool = False,
) -> str:
    """Return escaped generic card HTML."""

    title_html = escape(title)
    body_html = escape(body).replace("\n", "<br>")
    footer_html = escape(footer)
    icon_html = f"{escape(icon)} " if icon else ""
    badge = status_badge_html(status) if status else ""
    card_class = "ft-card ft-card--elevated" if elevated else "ft-card"

    return f"""
<div class="{card_class}">
    <div style="
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:0.75rem;
    ">
        <div class="ft-card__title">{icon_html}{title_html}</div>
        {badge}
    </div>
    {
        f'<div class="ft-card__body">{body_html}</div>'
        if body
        else ''
    }
    {
        f'<div class="ft-muted" style="margin-top:0.7rem;">'
        f'{footer_html}</div>'
        if footer
        else ''
    }
</div>
"""


def render_card(
    *,
    title: str,
    body: str = "",
    icon: str = "",
    footer: str = "",
    status: str | None = None,
    elevated: bool = False,
) -> None:
    """Render a generic dashboard card."""

    st = _streamlit()
    st.markdown(
        card_html(
            title=title,
            body=body,
            icon=icon,
            footer=footer,
            status=status,
            elevated=elevated,
        ),
        unsafe_allow_html=True,
    )


def metric_card_html(item: MetricItem) -> str:
    """Return HTML for one metric card."""

    style = (
        status_style(item.status)
        if item.status
        else None
    )
    icon = f"{escape(item.icon)} " if item.icon else ""
    value = escape(format_value(item.value))
    delta = (
        f'<div class="ft-muted">{escape(item.delta)}</div>'
        if item.delta
        else ""
    )
    help_text = (
        f'<div class="ft-muted" style="margin-top:0.45rem;">'
        f'{escape(item.help_text)}</div>'
        if item.help_text
        else ""
    )
    border_style = (
        f"border-color:{style.border};"
        if style is not None
        else ""
    )

    return f"""
<div class="ft-card" style="{border_style}">
    <div class="ft-muted">{icon}{escape(item.label)}</div>
    <div class="ft-metric-value">{value}</div>
    {delta}
    {help_text}
</div>
"""


def render_metric_grid(
    metrics: Sequence[MetricItem],
    *,
    columns: int = 4,
) -> None:
    """Render metrics in responsive Streamlit columns."""

    if columns < 1:
        raise ComponentError("columns must be at least 1.")

    st = _streamlit()

    for start in range(0, len(metrics), columns):
        row = metrics[start : start + columns]
        streamlit_columns = st.columns(len(row))

        for container, metric in zip(streamlit_columns, row):
            with container:
                st.markdown(
                    metric_card_html(metric),
                    unsafe_allow_html=True,
                )


def key_value_grid_html(
    items: Sequence[KeyValueItem],
    *,
    columns: int = 2,
) -> str:
    """Return HTML for a responsive key/value detail grid."""

    if columns < 1:
        raise ComponentError("columns must be at least 1.")

    cells = []

    for item in items:
        value = format_value(item.value, max_length=220)
        value_html = (
            f"<code>{escape(value)}</code>"
            if item.code
            else escape(value)
        )
        badge = (
            status_badge_html(item.status)
            if item.status
            else ""
        )
        help_html = (
            f'<div class="ft-muted">{escape(item.help_text)}</div>'
            if item.help_text
            else ""
        )

        cells.append(
            f"""
<div class="ft-card" style="padding:0.8rem;">
    <div class="ft-muted">{escape(item.label)}</div>
    <div style="
        margin-top:0.22rem;
        color:{THEME.colors.text_primary};
        font-weight:700;
        word-break:break-word;
    ">{value_html}</div>
    {badge}
    {help_html}
</div>
"""
        )

    return f"""
<div style="
    display:grid;
    grid-template-columns:repeat({columns}, minmax(0, 1fr));
    gap:0.75rem;
">
    {''.join(cells)}
</div>
"""


def render_key_value_grid(
    items: Sequence[KeyValueItem],
    *,
    columns: int = 2,
) -> None:
    """Render a key/value information grid."""

    st = _streamlit()
    st.markdown(
        key_value_grid_html(
            items,
            columns=columns,
        ),
        unsafe_allow_html=True,
    )


def banner_html(banner: Banner) -> str:
    """Return semantic information-banner HTML."""

    style = status_style(banner.status)
    icon = banner.icon if banner.icon is not None else style.icon

    return f"""
<div style="
    padding:0.9rem 1rem;
    border:1px solid {style.border};
    border-radius:{THEME.layout.small_radius};
    background:{style.background};
    margin:0.4rem 0 0.8rem;
">
    <div style="
        color:{style.foreground};
        font-weight:780;
        margin-bottom:0.18rem;
    ">{escape(icon)} {escape(banner.title)}</div>
    <div style="color:{THEME.colors.text_secondary};">
        {escape(banner.message)}
    </div>
</div>
"""


def render_banner(
    title: str,
    message: str,
    *,
    status: str = "ready",
    icon: str | None = None,
) -> None:
    """Render a semantic banner."""

    st = _streamlit()
    st.markdown(
        banner_html(
            Banner(
                title=title,
                message=message,
                status=status,
                icon=icon,
            )
        ),
        unsafe_allow_html=True,
    )


def render_decision_banner(
    *,
    accepted: bool,
    reason: str,
    retry_used: bool = False,
    p_attack: float | None = None,
    threshold: float | None = None,
) -> None:
    """Render the final authentication decision."""

    if retry_used and accepted:
        status = "retry"
        title = "Accepted after fresh retry"
    elif accepted:
        status = "accepted"
        title = "Authentication accepted"
    else:
        status = "rejected"
        title = "Authentication rejected"

    details = [reason or "No decision reason was provided."]

    if p_attack is not None:
        details.append(
            f"Calibrated P(attack): {format_probability(p_attack)}"
        )

    if threshold is not None:
        details.append(
            f"Operational threshold: {format_probability(threshold)}"
        )

    render_banner(
        title,
        " · ".join(details),
        status=status,
    )


def protocol_stepper_html(
    steps: Sequence[ProtocolStep],
    *,
    compact: bool = False,
) -> str:
    """Return HTML for the FT-QuPAP protocol-stage stepper."""

    step_html = []

    for step in steps:
        semantic = status_style(step.status)
        stage_color = protocol_stage_color(step.stage)
        description = (
            ""
            if compact or not step.description
            else (
                f'<div class="ft-muted" style="margin-top:0.25rem;">'
                f'{escape(step.description)}</div>'
            )
        )
        owner = (
            f'<div class="ft-muted" style="margin-top:0.22rem;">'
            f'Owner: {escape(step.owner)}</div>'
            if step.owner and not compact
            else ""
        )

        step_html.append(
            f"""
<div style="
    position:relative;
    display:grid;
    grid-template-columns:2.3rem 1fr auto;
    gap:0.7rem;
    align-items:start;
    padding:0.75rem;
    border:1px solid {semantic.border};
    border-left:4px solid {stage_color};
    border-radius:{THEME.layout.small_radius};
    background:{semantic.background};
">
    <div style="
        width:2.1rem;
        height:2.1rem;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:999px;
        background:{stage_color};
        color:{THEME.colors.white};
        font-weight:800;
    ">{step.number}</div>
    <div>
        <div style="
            color:{THEME.colors.text_primary};
            font-weight:760;
        ">{escape(step.label)}</div>
        {description}
        {owner}
    </div>
    {status_badge_html(step.status)}
</div>
"""
        )

    return f"""
<div style="
    display:flex;
    flex-direction:column;
    gap:0.55rem;
">
    {''.join(step_html)}
</div>
"""


def update_protocol_step_statuses(
    *,
    statuses: Mapping[str, str] | None = None,
    current_step: int | None = None,
    failed_step: int | None = None,
) -> tuple[ProtocolStep, ...]:
    """
    Create a display-ready protocol step sequence.

    Explicit ``statuses`` take priority. Otherwise:
    - steps before current_step become verified
    - current_step becomes running
    - later steps remain inactive
    - failed_step becomes failed
    """

    explicit = statuses or {}
    updated = []

    for step in DEFAULT_PROTOCOL_STEPS:
        status = explicit.get(step.key)

        if status is None:
            if failed_step is not None and step.number == failed_step:
                status = "failed"
            elif current_step is None:
                status = step.status
            elif step.number < current_step:
                status = "verified"
            elif step.number == current_step:
                status = "running"
            else:
                status = "inactive"

        updated.append(
            ProtocolStep(
                number=step.number,
                key=step.key,
                label=step.label,
                stage=step.stage,
                description=step.description,
                owner=step.owner,
                status=normalize_status(status),
            )
        )

    return tuple(updated)


def render_protocol_stepper(
    steps: Sequence[ProtocolStep] | None = None,
    *,
    statuses: Mapping[str, str] | None = None,
    current_step: int | None = None,
    failed_step: int | None = None,
    compact: bool = False,
) -> None:
    """Render the online protocol progression."""

    st = _streamlit()

    resolved_steps = (
        tuple(steps)
        if steps is not None
        else update_protocol_step_statuses(
            statuses=statuses,
            current_step=current_step,
            failed_step=failed_step,
        )
    )

    st.markdown(
        protocol_stepper_html(
            resolved_steps,
            compact=compact,
        ),
        unsafe_allow_html=True,
    )


def _mapping_to_rows(
    mapping: Mapping[str, Any],
    *,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert a mapping into table rows."""

    safe = sanitize_mapping(mapping)
    selected_keys = (
        list(order)
        if order is not None
        else list(safe)
    )
    rows = []

    for key in selected_keys:
        if key not in safe:
            continue

        rows.append(
            {
                "Field": (
                    labels.get(key, key)
                    if labels is not None
                    else key
                ),
                "Value": safe[key],
            }
        )

    return rows


def render_mapping_table(
    mapping: Mapping[str, Any],
    *,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    height: int | None = None,
) -> None:
    """Render a redacted mapping as a two-column table."""

    st = _streamlit()
    pd = _pandas()

    rows = _mapping_to_rows(
        mapping,
        labels=labels,
        order=order,
    )

    if not rows:
        render_empty_state(
            "No values available",
            "The selected record does not contain displayable fields.",
        )
        return

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=height,
    )


def render_feature_table(
    features: Mapping[str, Any],
    *,
    feature_order: Sequence[str] | None = None,
) -> None:
    """Render the nine observable GP features."""

    order = (
        tuple(feature_order)
        if feature_order is not None
        else tuple(GP_FEATURE_LABELS)
    )

    render_mapping_table(
        features,
        labels=GP_FEATURE_LABELS,
        order=order,
    )


def render_timing_table(
    timings: Mapping[str, Any],
) -> None:
    """Render timing data with human-readable units."""

    formatted = {
        key: format_duration(value)
        for key, value in sanitize_mapping(timings).items()
    }

    render_mapping_table(formatted)


def _flatten_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten one level of nested session data for tables."""

    safe = sanitize_mapping(record)
    flattened: dict[str, Any] = {}

    for key, value in safe.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                flattened[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, (list, tuple)):
            flattened[key] = " | ".join(
                format_value(item)
                for item in value
            )
        else:
            flattened[key] = value

    return flattened


def render_records_table(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
    height: int = DEFAULT_TABLE_HEIGHT,
    empty_title: str = "No records available",
    empty_message: str = (
        "Run a protocol scenario to populate this table."
    ),
) -> None:
    """Render redacted records in a dataframe."""

    if not records:
        render_empty_state(empty_title, empty_message)
        return

    st = _streamlit()
    pd = _pandas()

    rows = [_flatten_record(record) for record in records]
    table = pd.DataFrame(rows)

    if columns is not None:
        existing = [
            column
            for column in columns
            if column in table.columns
        ]
        table = table[existing]

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=height,
    )


def render_attempt_history(
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    """Render non-secret retry-attempt history."""

    preferred_columns = (
        "attempt_index",
        "accepted",
        "reason",
        "qber_raw",
        "loss_rate",
        "p_attack",
        "deterministic_pass",
        "tag_recovered",
        "end_to_end_seconds",
    )

    render_records_table(
        attempts,
        columns=preferred_columns,
        empty_title="No retry attempts",
        empty_message=(
            "This session did not create an attempt-history record."
        ),
    )


def render_json_viewer(
    payload: Any,
    *,
    title: str = "Safe JSON details",
    expanded: bool = False,
) -> None:
    """Render redacted JSON inside an expander."""

    st = _streamlit()

    with st.expander(title, expanded=expanded):
        st.code(
            safe_json_text(payload),
            language="json",
        )


def render_json_download(
    payload: Any,
    *,
    filename: str,
    label: str = "Download safe JSON",
    key: str | None = None,
) -> None:
    """Provide a download button for redacted JSON."""

    if not filename.lower().endswith(".json"):
        filename = f"{filename}.json"

    st = _streamlit()

    st.download_button(
        label=label,
        data=safe_json_text(payload) + "\n",
        file_name=Path(filename).name,
        mime="application/json",
        key=key,
    )


def render_empty_state(
    title: str,
    message: str,
    *,
    icon: str = "📭",
) -> None:
    """Render an empty-state card."""

    render_card(
        title=title,
        body=message,
        icon=icon,
        status="inactive",
    )


def safe_asset_path(
    directory: Path,
    filename: str,
) -> Path:
    """Resolve an asset filename without allowing path traversal."""

    resolved_directory = directory.resolve()
    path = (resolved_directory / filename).resolve()

    try:
        path.relative_to(resolved_directory)
    except ValueError as exc:
        raise ComponentError(
            "Asset path must remain inside its configured directory."
        ) from exc

    return path


def render_image(
    filename: str,
    *,
    caption: str | None = None,
    width: int | None = None,
    use_container_width: bool = True,
    directory: Path = IMAGES_DIR,
) -> bool:
    """
    Render an image asset.

    Returns False and displays a safe empty state when the file is absent.
    """

    path = safe_asset_path(directory, filename)

    if not path.is_file():
        render_empty_state(
            "Image unavailable",
            f"The dashboard asset {filename!r} was not found.",
            icon="🖼️",
        )
        return False

    st = _streamlit()
    image_kwargs: dict[str, Any] = {
        "image": str(path),
        "caption": caption,
    }

    if width is not None:
        image_kwargs["width"] = width
    else:
        image_kwargs["use_container_width"] = use_container_width

    st.image(**image_kwargs)
    return True


def render_progress(
    value: Any,
    *,
    label: str = "",
) -> None:
    """Render a normalized progress bar."""

    number = _finite_number(value)
    normalized = 0.0 if number is None else min(max(number, 0.0), 1.0)

    st = _streamlit()

    if label:
        st.caption(
            f"{label}: {format_percentage(normalized)}"
        )

    st.progress(normalized)


def render_sensitive_data_notice() -> None:
    """Display the dashboard's secret-redaction boundary."""

    render_banner(
        "Sensitive values are hidden",
        (
            "Private keys, shared secrets, K_auth, K_ctrl, raw identities, "
            "ciphertexts, and authentication tags are never displayed."
        ),
        status="verified",
        icon="🔒",
    )


def component_status() -> dict[str, Any]:
    """Return non-secret component diagnostics."""

    return {
        "default_protocol_step_count": len(
            DEFAULT_PROTOCOL_STEPS
        ),
        "gp_feature_count": len(GP_FEATURE_LABELS),
        "sensitive_field_token_count": len(
            SENSITIVE_FIELD_TOKENS
        ),
        "images_directory": str(IMAGES_DIR),
        "images_directory_exists": IMAGES_DIR.is_dir(),
    }


__all__ = [
    "ASSETS_DIR",
    "Banner",
    "ComponentError",
    "DEFAULT_PROTOCOL_STEPS",
    "GP_FEATURE_LABELS",
    "ICONS_DIR",
    "IMAGES_DIR",
    "KeyValueItem",
    "MetricItem",
    "PROJECT_ROOT",
    "ProtocolStep",
    "SENSITIVE_FIELD_TOKENS",
    "banner_html",
    "card_html",
    "component_status",
    "format_bytes",
    "format_duration",
    "format_percentage",
    "format_probability",
    "format_value",
    "is_sensitive_field",
    "key_value_grid_html",
    "metric_card_html",
    "protocol_stepper_html",
    "render_attempt_history",
    "render_banner",
    "render_card",
    "render_decision_banner",
    "render_empty_state",
    "render_feature_table",
    "render_image",
    "render_json_download",
    "render_json_viewer",
    "render_key_value_grid",
    "render_mapping_table",
    "render_metric_grid",
    "render_progress",
    "render_protocol_stepper",
    "render_records_table",
    "render_sensitive_data_notice",
    "render_timing_table",
    "safe_asset_path",
    "safe_json_text",
    "sanitize_mapping",
    "update_protocol_step_statuses",
]
