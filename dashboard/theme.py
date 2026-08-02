"""
Visual theme utilities for the FT-QuPAP Streamlit dashboard.

This module centralizes page configuration, semantic status styling,
custom CSS loading, protocol-stage colors, and reusable dashboard headers.

The theme is designed for the FT-QuPAP capstone demonstration:

- BLUE   : classical/PQC preparation and neutral protocol information
- PURPLE : quantum encoding and transmission
- GREEN  : accepted / verified / healthy
- YELLOW : retry / warning / gray-zone decision
- RED    : rejected / attack / failed deterministic check
- GRAY   : unavailable / inactive / not executed

Typical use
-----------
    import streamlit as st
    from dashboard.theme import apply_dashboard_theme
    from dashboard.theme import render_page_header

    apply_dashboard_theme()
    render_page_header(
        title="Protocol Monitor",
        subtitle="Live FT-QuPAP authentication pipeline",
        icon="📡",
    )

The module does not import Streamlit at package-import time. This keeps
unit tests, command-line utilities, and static validation lightweight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any, Final, Mapping


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"
STYLES_DIR: Final[Path] = ASSETS_DIR / "styles"
DEFAULT_CSS_FILE: Final[Path] = STYLES_DIR / "dashboard.css"

PROTOCOL_NAME: Final[str] = "FT-QuPAP"
PROTOCOL_VERSION: Final[str] = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)


@dataclass(frozen=True)
class ThemeColors:
    """FT-QuPAP dashboard color palette."""

    background: str = "#07111f"
    background_soft: str = "#0d1b2a"
    surface: str = "#102438"
    surface_alt: str = "#153047"
    surface_elevated: str = "#193a55"

    text_primary: str = "#f4f8fc"
    text_secondary: str = "#b8c8d9"
    text_muted: str = "#7f95aa"

    border: str = "#294762"
    border_soft: str = "#1d374d"

    primary: str = "#26a7ff"
    primary_soft: str = "#123c5d"
    secondary: str = "#9b7cff"
    quantum: str = "#b26cff"

    success: str = "#21c98b"
    success_soft: str = "#103b31"
    warning: str = "#f5b942"
    warning_soft: str = "#493914"
    danger: str = "#ff5c70"
    danger_soft: str = "#4b1d28"
    info: str = "#4fc3f7"
    info_soft: str = "#123b4d"
    inactive: str = "#7d8b99"
    inactive_soft: str = "#26313b"

    white: str = "#ffffff"
    black: str = "#000000"


@dataclass(frozen=True)
class ThemeTypography:
    """Font sizes used by CSS and inline components."""

    title_size: str = "2.15rem"
    section_title_size: str = "1.35rem"
    body_size: str = "0.98rem"
    small_size: str = "0.82rem"
    metric_size: str = "1.85rem"
    line_height: str = "1.55"


@dataclass(frozen=True)
class ThemeLayout:
    """Reusable spacing and shape values."""

    content_max_width: str = "1500px"
    card_radius: str = "16px"
    small_radius: str = "10px"
    card_padding: str = "1rem"
    section_gap: str = "1rem"
    border_width: str = "1px"
    shadow: str = "0 12px 34px rgba(0, 0, 0, 0.22)"


@dataclass(frozen=True)
class DashboardTheme:
    """Complete immutable dashboard theme."""

    colors: ThemeColors = ThemeColors()
    typography: ThemeTypography = ThemeTypography()
    layout: ThemeLayout = ThemeLayout()
    name: str = "FT-QuPAP Dark"
    version: str = "1.0.0"

    def to_dictionary(self) -> dict[str, Any]:
        """Return JSON-compatible theme metadata."""

        return asdict(self)


THEME: Final[DashboardTheme] = DashboardTheme()


@dataclass(frozen=True)
class StatusStyle:
    """Semantic presentation for one protocol state."""

    key: str
    label: str
    icon: str
    foreground: str
    background: str
    border: str


def _status(
    key: str,
    label: str,
    icon: str,
    foreground: str,
    background: str,
) -> StatusStyle:
    """Construct a status style using the same foreground as border."""

    return StatusStyle(
        key=key,
        label=label,
        icon=icon,
        foreground=foreground,
        background=background,
        border=foreground,
    )


STATUS_STYLES: Final[dict[str, StatusStyle]] = {
    "accepted": _status(
        "accepted",
        "Accepted",
        "✅",
        THEME.colors.success,
        THEME.colors.success_soft,
    ),
    "verified": _status(
        "verified",
        "Verified",
        "🛡️",
        THEME.colors.success,
        THEME.colors.success_soft,
    ),
    "healthy": _status(
        "healthy",
        "Healthy",
        "●",
        THEME.colors.success,
        THEME.colors.success_soft,
    ),
    "retry": _status(
        "retry",
        "Retry",
        "🔁",
        THEME.colors.warning,
        THEME.colors.warning_soft,
    ),
    "warning": _status(
        "warning",
        "Warning",
        "⚠️",
        THEME.colors.warning,
        THEME.colors.warning_soft,
    ),
    "gray_zone": _status(
        "gray_zone",
        "Gray Zone",
        "🟡",
        THEME.colors.warning,
        THEME.colors.warning_soft,
    ),
    "rejected": _status(
        "rejected",
        "Rejected",
        "⛔",
        THEME.colors.danger,
        THEME.colors.danger_soft,
    ),
    "attack": _status(
        "attack",
        "Attack",
        "⚔️",
        THEME.colors.danger,
        THEME.colors.danger_soft,
    ),
    "failed": _status(
        "failed",
        "Failed",
        "❌",
        THEME.colors.danger,
        THEME.colors.danger_soft,
    ),
    "running": _status(
        "running",
        "Running",
        "⏳",
        THEME.colors.primary,
        THEME.colors.primary_soft,
    ),
    "ready": _status(
        "ready",
        "Ready",
        "●",
        THEME.colors.info,
        THEME.colors.info_soft,
    ),
    "quantum": _status(
        "quantum",
        "Quantum",
        "⚛️",
        THEME.colors.quantum,
        THEME.colors.surface_alt,
    ),
    "inactive": _status(
        "inactive",
        "Inactive",
        "○",
        THEME.colors.inactive,
        THEME.colors.inactive_soft,
    ),
    "unavailable": _status(
        "unavailable",
        "Unavailable",
        "—",
        THEME.colors.inactive,
        THEME.colors.inactive_soft,
    ),
    "unknown": _status(
        "unknown",
        "Unknown",
        "❔",
        THEME.colors.inactive,
        THEME.colors.inactive_soft,
    ),
}


STATUS_ALIASES: Final[dict[str, str]] = {
    "accept": "accepted",
    "success": "accepted",
    "pass": "verified",
    "passed": "verified",
    "valid": "verified",
    "ok": "healthy",
    "green": "accepted",
    "accept_after_retry": "retry",
    "accepted_after_retry": "retry",
    "yellow": "retry",
    "retrying": "retry",
    "caution": "warning",
    "low_risk_retry": "gray_zone",
    "reject": "rejected",
    "denied": "rejected",
    "blocked": "rejected",
    "red": "rejected",
    "malicious": "attack",
    "error": "failed",
    "failure": "failed",
    "processing": "running",
    "pending": "running",
    "active": "ready",
    "idle": "inactive",
    "not_run": "inactive",
    "not_executed": "inactive",
    "n/a": "unavailable",
    "none": "unavailable",
}


PROTOCOL_STAGE_COLORS: Final[dict[str, str]] = {
    "registration": THEME.colors.info,
    "request": THEME.colors.primary,
    "freshness": THEME.colors.primary,
    "replay": THEME.colors.warning,
    "ml_dsa": THEME.colors.primary,
    "ml_kem": THEME.colors.primary,
    "key_derivation": THEME.colors.primary,
    "kmac": THEME.colors.secondary,
    "schedule": THEME.colors.secondary,
    "steane_encoding": THEME.colors.quantum,
    "quantum_channel": THEME.colors.quantum,
    "measurement": THEME.colors.quantum,
    "qber": THEME.colors.warning,
    "syndrome": THEME.colors.secondary,
    "decoding": THEME.colors.secondary,
    "tag_verification": THEME.colors.success,
    "deterministic_checks": THEME.colors.success,
    "gp_detection": THEME.colors.info,
    "retry": THEME.colors.warning,
    "decision": THEME.colors.success,
}


def _streamlit() -> Any:
    """
    Import Streamlit lazily.

    Raises an actionable error when the dashboard dependency is absent.
    """

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required for dashboard rendering. "
            "Install the project requirements first."
        ) from exc

    return st


def normalize_status(status: Any) -> str:
    """Normalize arbitrary result text to a registered semantic status."""

    if status is None:
        return "unknown"

    normalized = (
        str(status)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    normalized = STATUS_ALIASES.get(normalized, normalized)

    return normalized if normalized in STATUS_STYLES else "unknown"


def status_style(status: Any) -> StatusStyle:
    """Return semantic colors, label, and icon for a status value."""

    return STATUS_STYLES[normalize_status(status)]


def status_color(status: Any) -> str:
    """Return the semantic foreground color for a status."""

    return status_style(status).foreground


def protocol_stage_color(stage: str) -> str:
    """Return a stable color for one FT-QuPAP protocol stage."""

    normalized = (
        stage.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    return PROTOCOL_STAGE_COLORS.get(
        normalized,
        THEME.colors.info,
    )


def load_css_file(
    path: Path | None = None,
) -> str:
    """
    Read an external CSS file.

    Missing files return an empty string because the module includes a
    complete built-in fallback stylesheet.
    """

    css_path = DEFAULT_CSS_FILE if path is None else path

    if not css_path.is_file():
        return ""

    try:
        return css_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def css_variables() -> str:
    """Create CSS custom properties from the immutable theme."""

    colors = THEME.colors
    typography = THEME.typography
    layout = THEME.layout

    return f"""
:root {{
    --ft-bg: {colors.background};
    --ft-bg-soft: {colors.background_soft};
    --ft-surface: {colors.surface};
    --ft-surface-alt: {colors.surface_alt};
    --ft-surface-elevated: {colors.surface_elevated};
    --ft-text: {colors.text_primary};
    --ft-text-secondary: {colors.text_secondary};
    --ft-text-muted: {colors.text_muted};
    --ft-border: {colors.border};
    --ft-border-soft: {colors.border_soft};
    --ft-primary: {colors.primary};
    --ft-secondary: {colors.secondary};
    --ft-quantum: {colors.quantum};
    --ft-success: {colors.success};
    --ft-warning: {colors.warning};
    --ft-danger: {colors.danger};
    --ft-info: {colors.info};
    --ft-card-radius: {layout.card_radius};
    --ft-small-radius: {layout.small_radius};
    --ft-card-padding: {layout.card_padding};
    --ft-shadow: {layout.shadow};
    --ft-title-size: {typography.title_size};
    --ft-section-title-size: {typography.section_title_size};
    --ft-body-size: {typography.body_size};
    --ft-small-size: {typography.small_size};
    --ft-metric-size: {typography.metric_size};
    --ft-line-height: {typography.line_height};
}}
"""


def fallback_css() -> str:
    """Return built-in dashboard CSS used when no external file exists."""

    return """
html, body, [class*="css"] {
    color: var(--ft-text);
}

.stApp {
    background:
        radial-gradient(
            circle at 12% 8%,
            rgba(38, 167, 255, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 88% 14%,
            rgba(178, 108, 255, 0.09),
            transparent 30%
        ),
        var(--ft-bg);
}

[data-testid="stHeader"] {
    background: rgba(7, 17, 31, 0.84);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            var(--ft-bg-soft),
            var(--ft-bg)
        );
    border-right: 1px solid var(--ft-border-soft);
}

[data-testid="stSidebar"] * {
    color: var(--ft-text);
}

.main .block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: var(--ft-text);
    letter-spacing: -0.02em;
}

p, li, label, .stMarkdown {
    color: var(--ft-text-secondary);
    line-height: var(--ft-line-height);
}

.ft-page-header {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--ft-border-soft);
    border-radius: var(--ft-card-radius);
    background:
        linear-gradient(
            135deg,
            rgba(38, 167, 255, 0.10),
            rgba(155, 124, 255, 0.06)
        ),
        var(--ft-surface);
    box-shadow: var(--ft-shadow);
}

.ft-page-header__icon {
    font-size: 2rem;
    line-height: 1;
}

.ft-page-header__title {
    margin: 0;
    color: var(--ft-text);
    font-size: var(--ft-title-size);
    font-weight: 750;
}

.ft-page-header__subtitle {
    margin: 0.32rem 0 0;
    color: var(--ft-text-secondary);
    font-size: var(--ft-body-size);
}

.ft-card {
    height: 100%;
    padding: var(--ft-card-padding);
    border: 1px solid var(--ft-border-soft);
    border-radius: var(--ft-card-radius);
    background: var(--ft-surface);
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.13);
}

.ft-card--elevated {
    background: var(--ft-surface-elevated);
    box-shadow: var(--ft-shadow);
}

.ft-card__title {
    color: var(--ft-text);
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.ft-card__body {
    color: var(--ft-text-secondary);
}

.ft-metric-value {
    color: var(--ft-text);
    font-size: var(--ft-metric-size);
    font-weight: 780;
    line-height: 1.1;
}

.ft-muted {
    color: var(--ft-text-muted);
    font-size: var(--ft-small-size);
}

.ft-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    padding: 0.28rem 0.58rem;
    border-radius: 999px;
    border: 1px solid;
    font-size: 0.78rem;
    font-weight: 750;
    letter-spacing: 0.01em;
}

.ft-section-title {
    color: var(--ft-text);
    font-size: var(--ft-section-title-size);
    font-weight: 740;
    margin: 1rem 0 0.6rem;
}

.ft-research-notice {
    padding: 0.82rem 1rem;
    border: 1px solid var(--ft-warning);
    border-radius: var(--ft-small-radius);
    background: rgba(245, 185, 66, 0.10);
    color: var(--ft-text-secondary);
    font-size: 0.88rem;
}

.ft-research-notice strong {
    color: var(--ft-warning);
}

.ft-divider {
    height: 1px;
    margin: 1rem 0;
    background: var(--ft-border-soft);
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.55rem;
    border: 1px solid var(--ft-primary);
    border-radius: 10px;
    color: var(--ft-text);
    background:
        linear-gradient(
            135deg,
            rgba(38, 167, 255, 0.22),
            rgba(155, 124, 255, 0.15)
        ),
        var(--ft-surface-alt);
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--ft-info);
    color: white;
    transform: translateY(-1px);
}

[data-testid="stMetric"] {
    padding: 0.85rem;
    border: 1px solid var(--ft-border-soft);
    border-radius: var(--ft-card-radius);
    background: var(--ft-surface);
}

[data-testid="stMetricValue"] {
    color: var(--ft-text);
}

[data-testid="stMetricLabel"] {
    color: var(--ft-text-secondary);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--ft-border-soft);
    border-radius: var(--ft-small-radius);
    overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea,
input {
    border-color: var(--ft-border) !important;
    background: var(--ft-surface) !important;
    color: var(--ft-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    color: var(--ft-text-secondary);
    background: var(--ft-surface);
}

.stTabs [aria-selected="true"] {
    color: var(--ft-text);
    border-bottom-color: var(--ft-primary);
    background: var(--ft-surface-alt);
}

details {
    border: 1px solid var(--ft-border-soft) !important;
    border-radius: var(--ft-small-radius) !important;
    background: var(--ft-surface) !important;
}

code {
    color: #d7ebff;
}

@media (max-width: 700px) {
    .main .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
    }

    .ft-page-header {
        padding: 0.85rem;
    }

    .ft-page-header__title {
        font-size: 1.65rem;
    }
}
"""


def combined_css(
    external_css_path: Path | None = None,
) -> str:
    """
    Combine theme variables, built-in CSS, and the project stylesheet.

    External CSS is appended last so project-specific rules can override the
    safe defaults.
    """

    external = load_css_file(external_css_path)

    return "\n".join(
        section
        for section in (
            css_variables(),
            fallback_css(),
            external,
        )
        if section.strip()
    )


def inject_css(
    css: str,
) -> None:
    """Inject CSS into the current Streamlit page."""

    st = _streamlit()
    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def configure_page(
    *,
    page_title: str | None = None,
    page_icon: str = "🛡️",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
) -> None:
    """
    Configure the Streamlit page.

    Call this once, before any other Streamlit rendering operation.
    """

    st = _streamlit()

    st.set_page_config(
        page_title=(
            page_title
            or f"{PROTOCOL_NAME} Capstone Demonstrator"
        ),
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )


def apply_dashboard_theme(
    *,
    configure_streamlit_page: bool = False,
    page_title: str | None = None,
    page_icon: str = "🛡️",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    external_css_path: Path | None = None,
) -> None:
    """
    Apply FT-QuPAP styling to the current Streamlit page.

    ``app.py`` should normally call this once with
    ``configure_streamlit_page=True``. Individual view modules should call
    it with the default ``False`` only when rendered independently.
    """

    if configure_streamlit_page:
        configure_page(
            page_title=page_title,
            page_icon=page_icon,
            layout=layout,
            initial_sidebar_state=initial_sidebar_state,
        )

    inject_css(combined_css(external_css_path))


def status_badge_html(
    status: Any,
    *,
    label: str | None = None,
    include_icon: bool = True,
) -> str:
    """Return escaped semantic status-badge HTML."""

    style = status_style(status)
    resolved_label = escape(label or style.label)
    icon = f"{escape(style.icon)} " if include_icon else ""

    return (
        '<span class="ft-status-badge" '
        f'style="color:{style.foreground};'
        f'background:{style.background};'
        f'border-color:{style.border};">'
        f"{icon}{resolved_label}</span>"
    )


def render_status_badge(
    status: Any,
    *,
    label: str | None = None,
    include_icon: bool = True,
) -> None:
    """Render one semantic status badge."""

    st = _streamlit()
    st.markdown(
        status_badge_html(
            status,
            label=label,
            include_icon=include_icon,
        ),
        unsafe_allow_html=True,
    )


def page_header_html(
    *,
    title: str,
    subtitle: str = "",
    icon: str = "🛡️",
    status: Any | None = None,
) -> str:
    """Return HTML for a standard dashboard page header."""

    status_html = (
        status_badge_html(status)
        if status is not None
        else ""
    )

    escaped_subtitle = escape(subtitle)

    return f"""
<div class="ft-page-header">
    <div class="ft-page-header__icon">{escape(icon)}</div>
    <div style="flex:1;">
        <div style="
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:0.75rem;
            flex-wrap:wrap;
        ">
            <h1 class="ft-page-header__title">{escape(title)}</h1>
            {status_html}
        </div>
        {
            f'<p class="ft-page-header__subtitle">{escaped_subtitle}</p>'
            if subtitle
            else ''
        }
    </div>
</div>
"""


def render_page_header(
    *,
    title: str,
    subtitle: str = "",
    icon: str = "🛡️",
    status: Any | None = None,
) -> None:
    """Render a standard FT-QuPAP page header."""

    st = _streamlit()
    st.markdown(
        page_header_html(
            title=title,
            subtitle=subtitle,
            icon=icon,
            status=status,
        ),
        unsafe_allow_html=True,
    )


def render_section_title(
    title: str,
    *,
    icon: str = "",
) -> None:
    """Render a consistent section heading."""

    st = _streamlit()
    prefix = f"{escape(icon)} " if icon else ""

    st.markdown(
        f'<div class="ft-section-title">'
        f"{prefix}{escape(title)}</div>",
        unsafe_allow_html=True,
    )


def render_research_notice(
    message: str | None = None,
) -> None:
    """
    Display the simulator/evidence-boundary notice.

    This prevents development fixtures or capstone demo results from being
    silently presented as final paper evidence.
    """

    st = _streamlit()

    notice = message or (
        "This dashboard presents simulator and controlled-demonstration "
        "results. Only artifacts explicitly marked research-eligible and "
        "generated from disjoint training, calibration, and independent "
        "test runs may be used as final research evidence."
    )

    st.markdown(
        '<div class="ft-research-notice">'
        '<strong>Research boundary:</strong> '
        f"{escape(notice)}"
        "</div>",
        unsafe_allow_html=True,
    )


def render_divider() -> None:
    """Render a light dashboard divider."""

    st = _streamlit()
    st.markdown(
        '<div class="ft-divider"></div>',
        unsafe_allow_html=True,
    )


def theme_status() -> dict[str, Any]:
    """Return non-secret theme diagnostics."""

    return {
        "theme": THEME.to_dictionary(),
        "default_css_file": str(DEFAULT_CSS_FILE),
        "default_css_exists": DEFAULT_CSS_FILE.is_file(),
        "status_keys": sorted(STATUS_STYLES),
        "protocol_stage_keys": sorted(PROTOCOL_STAGE_COLORS),
    }


__all__ = [
    "ASSETS_DIR",
    "DEFAULT_CSS_FILE",
    "DashboardTheme",
    "PROTOCOL_STAGE_COLORS",
    "PROJECT_ROOT",
    "STATUS_STYLES",
    "STYLES_DIR",
    "StatusStyle",
    "THEME",
    "ThemeColors",
    "ThemeLayout",
    "ThemeTypography",
    "apply_dashboard_theme",
    "combined_css",
    "configure_page",
    "css_variables",
    "fallback_css",
    "inject_css",
    "load_css_file",
    "normalize_status",
    "page_header_html",
    "protocol_stage_color",
    "render_divider",
    "render_page_header",
    "render_research_notice",
    "render_section_title",
    "render_status_badge",
    "status_badge_html",
    "status_color",
    "status_style",
    "theme_status",
]