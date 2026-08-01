"""
FT-QuPAP Capstone Dashboard
===========================

Main Streamlit application entry point for the FT-QuPAP v5.1
capstone demonstration.

Run the application with:

    streamlit run app.py

The dashboard provides interfaces for:

- System overview
- Mobile Station operations
- Authentication Server monitoring
- Attack scenario control
- Live protocol monitoring
- Authentication results
- Session history

The dashboard is a visualization and demonstration layer. All security
decisions must be produced by the protocol implementation under ``src/``.
"""

from __future__ import annotations

import importlib
import logging
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from config import (
    ApplicationConfig,
    initialize_application_environment,
)


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------

APP_NAME = "FT-QuPAP Capstone Demo"
APP_VERSION = "5.1.0"

PAGE_HOME = "Home"
PAGE_MOBILE_STATION = "Mobile Station"
PAGE_AUTHENTICATION_SERVER = "Authentication Server"
PAGE_ATTACK_CONTROL = "Attack Control"
PAGE_PROTOCOL_MONITOR = "Protocol Monitor"
PAGE_RESULTS = "Results"
PAGE_SESSION_HISTORY = "Session History"


@dataclass(frozen=True)
class DashboardPage:
    """Configuration for one dashboard page."""

    name: str
    icon: str
    module_name: str
    renderer_name: str
    description: str


DASHBOARD_PAGES: tuple[DashboardPage, ...] = (
    DashboardPage(
        name=PAGE_HOME,
        icon="🏠",
        module_name="dashboard.home",
        renderer_name="render_home_page",
        description=(
            "FT-QuPAP system overview and protocol summary."
        ),
    ),
    DashboardPage(
        name=PAGE_MOBILE_STATION,
        icon="📱",
        module_name="dashboard.mobile_station_view",
        renderer_name="render_mobile_station_view",
        description=(
            "Prepare authentication requests and observe "
            "Mobile Station operations."
        ),
    ),
    DashboardPage(
        name=PAGE_AUTHENTICATION_SERVER,
        icon="🖥️",
        module_name=(
            "dashboard.authentication_server_view"
        ),
        renderer_name=(
            "render_authentication_server_view"
        ),
        description=(
            "Observe server-side verification and "
            "authentication decisions."
        ),
    ),
    DashboardPage(
        name=PAGE_ATTACK_CONTROL,
        icon="⚔️",
        module_name="dashboard.attack_control_view",
        renderer_name="render_attack_control_view",
        description=(
            "Select controlled attack and channel scenarios."
        ),
    ),
    DashboardPage(
        name=PAGE_PROTOCOL_MONITOR,
        icon="📡",
        module_name="dashboard.protocol_monitor_view",
        renderer_name="render_protocol_monitor_view",
        description=(
            "Monitor the FT-QuPAP authentication flow."
        ),
    ),
    DashboardPage(
        name=PAGE_RESULTS,
        icon="📊",
        module_name="dashboard.results_view",
        renderer_name="render_results_view",
        description=(
            "View QBER, GP probability, retry, and "
            "authentication results."
        ),
    ),
    DashboardPage(
        name=PAGE_SESSION_HISTORY,
        icon="🗂️",
        module_name="dashboard.session_history_view",
        renderer_name="render_session_history_view",
        description=(
            "Inspect previously completed demonstration sessions."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("ft_qupap.app")


def configure_logging(
    config: ApplicationConfig,
) -> None:
    """Configure basic application logging."""

    if LOGGER.handlers:
        return

    log_level = getattr(
        logging,
        config.logging.level,
        logging.INFO,
    )

    LOGGER.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    )

    if config.logging.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        LOGGER.addHandler(console_handler)

    if config.logging.file_logging_enabled:
        log_file = config.logging.protocol_log_file
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            from logging.handlers import (
                RotatingFileHandler,
            )

            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=(
                    config.logging
                    .maximum_file_size_bytes
                ),
                backupCount=(
                    config.logging.backup_count
                ),
                encoding="utf-8",
            )

            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            LOGGER.addHandler(file_handler)

        except OSError as error:
            LOGGER.warning(
                "Unable to initialize file logging: %s",
                error,
            )


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------

def configure_streamlit_page(
    config: ApplicationConfig,
) -> None:
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title=config.dashboard.page_title,
        page_icon=config.dashboard.page_icon,
        layout=config.dashboard.layout,
        initial_sidebar_state=(
            "expanded"
            if config.dashboard.show_sidebar
            else "collapsed"
        ),
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": (
                "FT-QuPAP v5.1 Capstone Demonstration\n\n"
                "Fault-Tolerant Quantum Authentication "
                "Protocol with Post-Quantum Cryptographic "
                "Bootstrapping and Adaptive "
                "Eavesdropping Detection."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

def initialize_session_state(
    config: ApplicationConfig,
) -> None:
    """Create dashboard session-state variables."""

    default_state: dict[str, Any] = {
        "selected_page": PAGE_HOME,
        "selected_scenario": (
            config.dashboard.default_scenario
        ),
        "active_session_id": None,
        "active_session": None,
        "latest_result": None,
        "latest_protocol_events": [],
        "authentication_running": False,
        "authentication_completed": False,
        "retry_requested": False,
        "retry_attempt": 0,
        "attack_enabled": False,
        "hardware_connected": False,
        "dashboard_initialized": True,
    }

    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Theme loading
# ---------------------------------------------------------------------------

def apply_theme() -> None:
    """Apply the custom FT-QuPAP dashboard theme."""

    try:
        theme_module = importlib.import_module(
            "dashboard.theme"
        )

        theme_function = getattr(
            theme_module,
            "apply_dashboard_theme",
            None,
        )

        if callable(theme_function):
            theme_function()
            return

    except Exception as error:
        LOGGER.warning(
            "Custom dashboard theme could not be loaded: %s",
            error,
        )

    apply_default_theme()


def apply_default_theme() -> None:
    """Apply a built-in theme when dashboard.theme is unavailable."""

    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            [data-testid="stSidebar"] {
                min-width: 280px;
                max-width: 280px;
            }

            .ft-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.15rem;
            }

            .ft-subtitle {
                font-size: 1rem;
                opacity: 0.8;
                margin-bottom: 1.2rem;
            }

            .ft-protocol-badge {
                display: inline-block;
                padding: 0.25rem 0.7rem;
                border-radius: 999px;
                border: 1px solid rgba(128, 128, 128, 0.35);
                font-size: 0.82rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
            }

            .ft-footer {
                opacity: 0.65;
                font-size: 0.8rem;
                text-align: center;
                padding-top: 2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------------------------

def render_application_header(
    config: ApplicationConfig,
) -> None:
    """Render the dashboard header."""

    st.markdown(
        '<div class="ft-protocol-badge">'
        "FT-QuPAP v5.1"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ft-title">'
        "Fault-Tolerant Quantum Authentication Protocol"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ft-subtitle">'
        "Post-quantum cryptographic bootstrapping, "
        "Steane [[7,1,3]] protection, deterministic "
        "verification, calibrated GP attack detection, "
        "and bounded retry processing."
        "</div>",
        unsafe_allow_html=True,
    )

    if config.debug:
        st.info(
            "Debug mode is enabled.",
            icon="🛠️",
        )


def render_sidebar(
    config: ApplicationConfig,
) -> DashboardPage:
    """Render navigation and return the selected page."""

    with st.sidebar:
        st.title("FT-QuPAP")
        st.caption(
            f"Capstone demonstration v{APP_VERSION}"
        )

        st.divider()

        page_names = [
            f"{page.icon} {page.name}"
            for page in DASHBOARD_PAGES
        ]

        current_page_name = st.session_state.get(
            "selected_page",
            PAGE_HOME,
        )

        current_index = 0

        for index, page in enumerate(
            DASHBOARD_PAGES
        ):
            if page.name == current_page_name:
                current_index = index
                break

        selected_label = st.radio(
            "Navigation",
            options=page_names,
            index=current_index,
            label_visibility="collapsed",
        )

        selected_index = page_names.index(
            selected_label
        )

        selected_page = DASHBOARD_PAGES[
            selected_index
        ]

        st.session_state.selected_page = (
            selected_page.name
        )

        st.divider()

        render_sidebar_protocol_status(config)

        st.divider()

        st.caption(
            "Deterministic verification remains mandatory. "
            "The Gaussian Process model supplements rather "
            "than replaces protocol checks."
        )

        return selected_page


def render_sidebar_protocol_status(
    config: ApplicationConfig,
) -> None:
    """Render important protocol settings in the sidebar."""

    st.subheader("Protocol status")

    st.markdown(
        f"**ML-DSA:** "
        f"{config.cryptography.ml_dsa_parameter_set}"
    )

    st.markdown(
        f"**ML-KEM:** "
        f"{config.cryptography.ml_kem_parameter_set}"
    )

    st.markdown(
        f"**KMAC tag:** "
        f"{config.cryptography.kmac_tag_bits} bits"
    )

    st.markdown(
        f"**Logical blocks:** "
        f"{config.quantum.total_logical_blocks}"
    )

    st.markdown(
        f"**Physical qubits:** "
        f"{config.quantum.total_physical_qubits}"
    )

    st.markdown(
        f"**QBER limit:** "
        f"{config.quantum.fixed_qber_threshold:.2f}"
    )

    st.markdown(
        f"**Maximum retries:** "
        f"{config.protocol.maximum_retries}"
    )

    hardware_status = (
        "Enabled"
        if config.hardware.enabled
        else "Simulation"
    )

    st.markdown(
        f"**Hardware:** {hardware_status}"
    )


# ---------------------------------------------------------------------------
# Dashboard page loading
# ---------------------------------------------------------------------------

def load_page_renderer(
    page: DashboardPage,
) -> Callable[..., Any] | None:
    """Load a dashboard page-rendering function."""

    try:
        module = importlib.import_module(
            page.module_name
        )

    except ModuleNotFoundError as error:
        LOGGER.error(
            "Dashboard module could not be loaded: %s",
            error,
        )

        return None

    except Exception as error:
        LOGGER.exception(
            "Unexpected error loading dashboard module "
            "%s: %s",
            page.module_name,
            error,
        )

        return None

    renderer = getattr(
        module,
        page.renderer_name,
        None,
    )

    if not callable(renderer):
        LOGGER.error(
            "Renderer %s was not found in %s.",
            page.renderer_name,
            page.module_name,
        )

        return None

    return renderer


def render_selected_page(
    page: DashboardPage,
    config: ApplicationConfig,
) -> None:
    """Load and display the selected dashboard page."""

    renderer = load_page_renderer(page)

    if renderer is None:
        render_missing_page(page)
        return

    try:
        renderer(config=config)

    except TypeError as error:
        # Allows an early page implementation that does not yet
        # accept the config keyword argument.
        if "unexpected keyword argument" in str(error):
            renderer()
            return

        render_page_error(
            page=page,
            error=error,
        )

    except Exception as error:
        render_page_error(
            page=page,
            error=error,
        )


def render_missing_page(
    page: DashboardPage,
) -> None:
    """Render a message for a page not yet implemented."""

    st.warning(
        f"{page.icon} **{page.name}** is not available.",
        icon="⚠️",
    )

    st.write(page.description)

    st.code(
        f"{page.module_name}.{page.renderer_name}()",
        language="python",
    )


def render_page_error(
    *,
    page: DashboardPage,
    error: Exception,
) -> None:
    """Display a dashboard-page execution error."""

    LOGGER.exception(
        "Error rendering %s: %s",
        page.name,
        error,
    )

    st.error(
        f"Unable to render the {page.name} page.",
        icon="🚨",
    )

    st.write(str(error))

    with st.expander(
        "Technical details",
        expanded=False,
    ):
        st.code(
            traceback.format_exc(),
            language="text",
        )


# ---------------------------------------------------------------------------
# Application footer
# ---------------------------------------------------------------------------

def render_footer() -> None:
    """Render the dashboard footer."""

    st.markdown(
        """
        <div class="ft-footer">
            FT-QuPAP v5.1 Capstone Demonstration ·
            Post-Quantum and Quantum Authentication Research
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application bootstrap
# ---------------------------------------------------------------------------

def bootstrap_application() -> ApplicationConfig:
    """Initialize configuration, storage, and logging."""

    config = initialize_application_environment()

    configure_logging(config)

    LOGGER.info(
        "Starting %s version %s.",
        APP_NAME,
        APP_VERSION,
    )

    return config


def main() -> None:
    """Run the FT-QuPAP Streamlit dashboard."""

    try:
        config = bootstrap_application()

    except Exception as error:
        st.set_page_config(
            page_title="FT-QuPAP Configuration Error",
            page_icon="🚨",
            layout="wide",
        )

        st.error(
            "FT-QuPAP application initialization failed.",
            icon="🚨",
        )

        st.code(
            str(error),
            language="text",
        )

        return

    configure_streamlit_page(config)
    initialize_session_state(config)
    apply_theme()

    render_application_header(config)

    selected_page = render_sidebar(config)

    render_selected_page(
        page=selected_page,
        config=config,
    )

    render_footer()


if __name__ == "__main__":
    main()