"""
Root Streamlit application for the FT-QuPAP capstone demonstrator.

Run from the project root:

    streamlit run app.py

The application provides lazy navigation to:

- Home
- Mobile Station
- Authentication Server
- Attack Control
- Protocol Monitor
- Research Results
- Session History

The root application is intentionally small. Protocol logic remains in the
project's service/simulation modules, while page-specific presentation remains
inside ``dashboard/``.

Security boundary
-----------------
The root router does not load, print, or export cryptographic secret material.
Page-rendering exceptions are shown in a restricted form so that keys,
ciphertexts, raw tags, identities, and other sensitive values are not exposed
through the user interface.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Mapping

from dashboard import (
    DASHBOARD_VIEWS,
    PROJECT_ROOT,
    DashboardView,
    available_view_keys,
    get_view,
    get_view_module,
    iter_views,
    package_status,
)
from dashboard.theme import (
    THEME,
    apply_dashboard_theme,
    status_badge_html,
)


APP_TITLE: Final[str] = "FT-QuPAP Capstone Demonstrator"
APP_ICON: Final[str] = "🛡️"
APP_VERSION: Final[str] = "1.0.0"
DEFAULT_VIEW_KEY: Final[str] = "home"

NAVIGATION_STATE_KEY: Final[str] = "ft_qupap_selected_view"
QUERY_PARAMETER_NAME: Final[str] = "page"

MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATABASE_DIR: Final[Path] = PROJECT_ROOT / "database"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"

MODEL_BUNDLE_FILES: Final[tuple[Path, ...]] = (
    MODEL_DIR / "gp_model.pkl",
    MODEL_DIR / "feature_scaler.pkl",
    MODEL_DIR / "calibration_model.pkl",
    MODEL_DIR / "threshold.json",
    MODEL_DIR / "feature_order.json",
    MODEL_DIR / "model_metadata.json",
)

RUNTIME_DIRECTORIES: Final[tuple[Path, ...]] = (
    DATA_DIR,
    DATABASE_DIR,
    OUTPUTS_DIR,
)

SENSITIVE_ERROR_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\b("
        r"private[_ -]?key|secret[_ -]?key|shared[_ -]?secret|"
        r"k[_ -]?auth|k[_ -]?ctrl|k[_ -]?ss|ciphertext|"
        r"authentication[_ -]?tag|raw[_ -]?tag|subscriber[_ -]?id|"
        r"mobile[_ -]?id|imsi|password|token"
        r")\b\s*[:=]\s*[^,\s;]+"
    ),
    re.compile(r"(?i)\b[0-9a-f]{64,}\b"),
)


class DashboardApplicationError(RuntimeError):
    """Raised when a registered dashboard page cannot be loaded safely."""


@dataclass(frozen=True)
class AppStatus:
    """Non-secret startup diagnostics for the root application."""

    app_title: str
    app_version: str
    project_root: str
    registered_view_count: int
    registered_view_keys: tuple[str, ...]
    available_view_file_count: int
    missing_view_files: tuple[str, ...]
    model_bundle_available_count: int
    model_bundle_required_count: int
    model_bundle_complete: bool
    runtime_directory_count: int
    available_runtime_directory_count: int
    dashboard_css_exists: bool

    def to_dictionary(self) -> dict[str, Any]:
        """Return JSON-compatible application diagnostics."""

        return asdict(self)


def _streamlit() -> Any:
    """Import Streamlit only when application rendering is requested."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to run app.py. Install the project "
            "dependencies and execute: streamlit run app.py"
        ) from exc

    return st


def resolve_view_key(
    candidate: Any,
    *,
    default: str = DEFAULT_VIEW_KEY,
) -> str:
    """Resolve an arbitrary navigation value to a registered page key."""

    default_key = str(default).strip().lower()

    if default_key not in available_view_keys():
        raise DashboardApplicationError(
            f"Invalid default dashboard view: {default!r}."
        )

    if candidate is None:
        return default_key

    if isinstance(candidate, (list, tuple)):
        candidate = candidate[0] if candidate else None

    normalized = str(candidate or "").strip().lower()

    return (
        normalized
        if normalized in available_view_keys()
        else default_key
    )


def _view_file(view: DashboardView) -> Path:
    """Return the expected Python file for one registered page."""

    return PROJECT_ROOT / "dashboard" / f"{view.module}.py"


def build_app_status() -> AppStatus:
    """Inspect non-secret application readiness without importing pages."""

    views = tuple(iter_views())
    missing = tuple(
        str(_view_file(view).relative_to(PROJECT_ROOT))
        for view in views
        if not _view_file(view).is_file()
    )
    available_model_files = sum(
        path.is_file() and path.stat().st_size > 0
        for path in MODEL_BUNDLE_FILES
    )
    available_runtime_directories = sum(
        path.is_dir()
        for path in RUNTIME_DIRECTORIES
    )
    dashboard_package = package_status()

    return AppStatus(
        app_title=APP_TITLE,
        app_version=APP_VERSION,
        project_root=str(PROJECT_ROOT),
        registered_view_count=len(views),
        registered_view_keys=tuple(
            view.key for view in views
        ),
        available_view_file_count=len(views) - len(missing),
        missing_view_files=missing,
        model_bundle_available_count=available_model_files,
        model_bundle_required_count=len(MODEL_BUNDLE_FILES),
        model_bundle_complete=(
            available_model_files == len(MODEL_BUNDLE_FILES)
        ),
        runtime_directory_count=len(RUNTIME_DIRECTORIES),
        available_runtime_directory_count=(
            available_runtime_directories
        ),
        dashboard_css_exists=bool(
            dashboard_package.get("dashboard_css_exists")
        ),
    )


def safe_error_message(
    error: BaseException,
    *,
    maximum_length: int = 320,
) -> str:
    """Return a restricted, redacted exception message for the dashboard."""

    message = str(error).strip()

    if not message:
        message = "No additional error message was provided."

    for pattern in SENSITIVE_ERROR_PATTERNS:
        message = pattern.sub("<redacted>", message)

    message = message.replace("\n", " ").replace("\r", " ")
    message = " ".join(message.split())

    if len(message) > maximum_length:
        message = message[: maximum_length - 1] + "…"

    return message


def _query_parameter_view(st: Any) -> str | None:
    """Read the optional page query parameter across Streamlit versions."""

    try:
        value = st.query_params.get(QUERY_PARAMETER_NAME)
    except Exception:
        try:
            values = st.experimental_get_query_params()
        except Exception:
            return None

        value = values.get(QUERY_PARAMETER_NAME)

    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None

    return str(value) if value is not None else None


def _set_query_parameter_view(
    st: Any,
    view_key: str,
) -> None:
    """Persist the selected page in the URL when supported."""

    try:
        st.query_params[QUERY_PARAMETER_NAME] = view_key
        return
    except Exception:
        pass

    try:
        st.experimental_set_query_params(
            **{QUERY_PARAMETER_NAME: view_key}
        )
    except Exception:
        # Query parameters are optional. Session state remains authoritative.
        return


def _initial_view_key(st: Any) -> str:
    """Resolve the first page from session state or URL query parameters."""

    session_value = st.session_state.get(
        NAVIGATION_STATE_KEY
    )
    query_value = _query_parameter_view(st)

    candidate = (
        session_value
        if session_value is not None
        else query_value
    )
    resolved = resolve_view_key(candidate)

    st.session_state[NAVIGATION_STATE_KEY] = resolved
    return resolved


def _navigation_label(view: DashboardView) -> str:
    """Return the user-facing sidebar navigation label."""

    return f"{view.icon}  {view.label}"


def _render_sidebar_brand(st: Any) -> None:
    """Render the sidebar project identity."""

    st.markdown(
        f"""
<div style="
    padding:0.15rem 0 0.85rem 0;
">
    <div style="
        font-size:1.4rem;
        font-weight:800;
        color:{THEME.colors.text_primary};
        letter-spacing:0.01em;
    ">{escape(APP_ICON)} FT-QuPAP</div>
    <div style="
        margin-top:0.25rem;
        color:{THEME.colors.text_secondary};
        font-size:0.84rem;
        line-height:1.45;
    ">
        Fault-Tolerant Quantum Authentication with authenticated PQC
        bootstrapping and adaptive attack detection
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_sidebar_readiness(
    st: Any,
    status: AppStatus,
) -> None:
    """Render compact non-secret application readiness details."""

    st.markdown("### System readiness")

    model_status = (
        "verified"
        if status.model_bundle_complete
        else "warning"
    )
    view_status = (
        "verified"
        if not status.missing_view_files
        else "failed"
    )

    st.markdown(
        (
            f"{status_badge_html(view_status, label='Dashboard pages')} "
            f"`{status.available_view_file_count}/"
            f"{status.registered_view_count}`"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f"{status_badge_html(model_status, label='GP model bundle')} "
            f"`{status.model_bundle_available_count}/"
            f"{status.model_bundle_required_count}`"
        ),
        unsafe_allow_html=True,
    )

    directory_status = (
        "verified"
        if status.available_runtime_directory_count
        == status.runtime_directory_count
        else "warning"
    )
    st.markdown(
        (
            f"{status_badge_html(directory_status, label='Runtime folders')} "
            f"`{status.available_runtime_directory_count}/"
            f"{status.runtime_directory_count}`"
        ),
        unsafe_allow_html=True,
    )

    if status.missing_view_files:
        with st.expander(
            "Missing dashboard files",
            expanded=False,
        ):
            for path in status.missing_view_files:
                st.code(path)

    if not status.dashboard_css_exists:
        st.caption(
            (
                "External dashboard.css is not available. "
                "The built-in fallback theme remains active."
            )
        )


def _render_sidebar_boundary(st: Any) -> None:
    """Render the research and security boundary."""

    st.markdown("---")
    st.markdown("### Demonstrator boundary")
    st.caption(
        (
            "Research simulator only. Full sessions use syndrome-level "
            "Steane modeling; Qiskit/Aer is representative validation."
        )
    )
    st.caption(
        (
            "The GP supplies calibrated supporting evidence. Mandatory "
            "cryptographic, freshness, replay, decoder, tag, QBER, loss, "
            "and check-count gates remain authoritative."
        )
    )
    st.caption(
        "Secret keys, session keys, raw identities, tags, and ciphertexts "
        "must never be displayed."
    )


def render_sidebar(
    st: Any,
    *,
    status: AppStatus,
) -> DashboardView:
    """Render navigation and return the selected page metadata."""

    views = tuple(iter_views())
    initial_key = _initial_view_key(st)
    initial_view = get_view(initial_key)

    label_to_view = {
        _navigation_label(view): view
        for view in views
    }
    labels = tuple(label_to_view)
    initial_index = views.index(initial_view)

    with st.sidebar:
        _render_sidebar_brand(st)

        selected_label = st.radio(
            "Navigation",
            labels,
            index=initial_index,
            key="ft_qupap_navigation_radio",
            label_visibility="collapsed",
        )
        selected = label_to_view[selected_label]

        st.session_state[
            NAVIGATION_STATE_KEY
        ] = selected.key
        _set_query_parameter_view(st, selected.key)

        st.markdown("### Current page")
        st.markdown(
            (
                f"**{selected.icon} {selected.label}**\n\n"
                f"{selected.description}"
            )
        )

        _render_sidebar_readiness(st, status)
        _render_sidebar_boundary(st)

        st.markdown("---")
        st.caption(
            f"Dashboard v{APP_VERSION}"
        )

    return selected


def load_view_module(view_key: str) -> ModuleType:
    """Load a registered view and validate its render entry point."""

    try:
        module = get_view_module(view_key)
    except Exception as exc:
        raise DashboardApplicationError(
            f"Could not load dashboard page {view_key!r}: "
            f"{safe_error_message(exc)}"
        ) from exc

    render = getattr(module, "render", None)

    if not callable(render):
        raise DashboardApplicationError(
            f"Dashboard page {view_key!r} has no callable render() function."
        )

    return module


def render_selected_view(
    st: Any,
    view: DashboardView,
) -> bool:
    """Render one page inside a restricted application error boundary."""

    try:
        module = load_view_module(view.key)
        module.render()
        return True
    except Exception as exc:
        message = safe_error_message(exc)

        st.error(
            f"{view.label} could not be rendered."
        )
        st.markdown(
            f"""
<div class="ft-card" style="
    border-color:{THEME.colors.danger};
">
    <div class="ft-card__title">
        {escape(type(exc).__name__)}
    </div>
    <div style="
        color:{THEME.colors.text_secondary};
        margin-top:0.45rem;
    ">
        {escape(message)}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.info(
            (
                "Check that the dashboard files are in the dashboard/ "
                "package and that project dependencies and generated "
                "artifacts are available."
            )
        )

        debug_enabled = (
            os.getenv(
                "FT_QUPAP_DASHBOARD_DEBUG",
                "",
            ).strip()
            == "1"
        )

        if debug_enabled:
            st.caption(
                (
                    "Debug mode is enabled, but raw local variables and "
                    "cryptographic material are intentionally not displayed."
                )
            )

        return False


def _render_footer(st: Any, view: DashboardView) -> None:
    """Render the common application footer."""

    st.markdown("---")
    st.caption(
        (
            f"{APP_TITLE} · {view.label} · "
            "Stored evidence only · No cryptographic secret material shown"
        )
    )


def main() -> None:
    """Configure and run the FT-QuPAP Streamlit application."""

    st = _streamlit()

    # This must be the first Streamlit rendering operation.
    apply_dashboard_theme(
        configure_streamlit_page=True,
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    status = build_app_status()
    selected_view = render_sidebar(
        st,
        status=status,
    )

    render_selected_view(
        st,
        selected_view,
    )
    _render_footer(
        st,
        selected_view,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "APP_ICON",
    "APP_TITLE",
    "APP_VERSION",
    "AppStatus",
    "DashboardApplicationError",
    "build_app_status",
    "load_view_module",
    "main",
    "render_selected_view",
    "render_sidebar",
    "resolve_view_key",
    "safe_error_message",
]