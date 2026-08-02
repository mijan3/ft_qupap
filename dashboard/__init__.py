"""
FT-QuPAP Streamlit dashboard package.

This package contains the visual interface for the mobile station,
authentication server, attack controls, protocol monitor, result analysis,
and session history.

The module intentionally avoids importing Streamlit view files at package
import time. This prevents page rendering side effects and circular imports.

Typical use
-----------
    from dashboard import DASHBOARD_VIEWS
    from dashboard import get_view_module

    module = get_view_module("home")
    module.render()

Project structure
-----------------
    dashboard/
    ├── __init__.py
    ├── theme.py
    ├── components.py
    ├── status_cards.py
    ├── charts.py
    ├── home.py
    ├── mobile_station_view.py
    ├── authentication_server_view.py
    ├── attack_control_view.py
    ├── protocol_monitor_view.py
    ├── results_view.py
    └── session_history_view.py
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Final, Iterable


PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = PACKAGE_ROOT.parent
ASSETS_ROOT: Final[Path] = PROJECT_ROOT / "assets"
IMAGES_ROOT: Final[Path] = ASSETS_ROOT / "images"
ICONS_ROOT: Final[Path] = ASSETS_ROOT / "icons"
STYLES_ROOT: Final[Path] = ASSETS_ROOT / "styles"
DASHBOARD_CSS_FILE: Final[Path] = STYLES_ROOT / "dashboard.css"

PACKAGE_NAME: Final[str] = "dashboard"
PACKAGE_VERSION: Final[str] = "1.0.0"


@dataclass(frozen=True)
class DashboardView:
    """Metadata for one Streamlit dashboard page."""

    key: str
    label: str
    module: str
    icon: str
    description: str
    order: int

    @property
    def import_path(self) -> str:
        """Return the full Python module path."""

        return f"{PACKAGE_NAME}.{self.module}"


DASHBOARD_VIEWS: Final[tuple[DashboardView, ...]] = (
    DashboardView(
        key="home",
        label="Home",
        module="home",
        icon="🏠",
        description="Project overview and live protocol summary.",
        order=1,
    ),
    DashboardView(
        key="mobile_station",
        label="Mobile Station",
        module="mobile_station_view",
        icon="📱",
        description=(
            "Authentication request, ML-KEM encapsulation, KMAC tag "
            "generation, and quantum token preparation."
        ),
        order=2,
    ),
    DashboardView(
        key="authentication_server",
        label="Authentication Server",
        module="authentication_server_view",
        icon="🛡️",
        description=(
            "Freshness, replay, credential, decoding, KMAC, GP, and "
            "final decision processing."
        ),
        order=3,
    ),
    DashboardView(
        key="attack_control",
        label="Attack Control",
        module="attack_control_view",
        icon="⚔️",
        description=(
            "Controlled Eve, replay, tampering, loss, and forged-message "
            "scenario selection."
        ),
        order=4,
    ),
    DashboardView(
        key="protocol_monitor",
        label="Protocol Monitor",
        module="protocol_monitor_view",
        icon="📡",
        description=(
            "Step-by-step FT-QuPAP execution, QBER, syndrome, GP, and "
            "retry monitoring."
        ),
        order=5,
    ),
    DashboardView(
        key="results",
        label="Results",
        module="results_view",
        icon="📊",
        description=(
            "ROC, PR, calibration, confusion matrix, QBER, attack "
            "probability, and retry analysis."
        ),
        order=6,
    ),
    DashboardView(
        key="session_history",
        label="Session History",
        module="session_history_view",
        icon="🗂️",
        description=(
            "Non-secret historical authentication decisions and "
            "scenario execution records."
        ),
        order=7,
    ),
)


_VIEW_BY_KEY: Final[dict[str, DashboardView]] = {
    view.key: view for view in DASHBOARD_VIEWS
}


def get_view(key: str) -> DashboardView:
    """
    Return page metadata for a dashboard view.

    Raises
    ------
    KeyError
        If the requested page key is not registered.
    """

    normalized = key.strip().lower()

    try:
        return _VIEW_BY_KEY[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_VIEW_BY_KEY))
        raise KeyError(
            f"Unknown dashboard view {key!r}. Available views: {available}"
        ) from exc


def get_view_module(key: str) -> ModuleType:
    """
    Lazily import and return a registered dashboard view module.

    View modules should expose a ``render()`` function.
    """

    view = get_view(key)
    module = import_module(view.import_path)

    render = getattr(module, "render", None)
    if not callable(render):
        raise AttributeError(
            f"{view.import_path} must expose a callable render() function."
        )

    return module


def iter_views() -> Iterable[DashboardView]:
    """Yield dashboard views in navigation order."""

    return iter(sorted(DASHBOARD_VIEWS, key=lambda view: view.order))


def available_view_keys() -> tuple[str, ...]:
    """Return all registered page keys."""

    return tuple(view.key for view in iter_views())


def asset_path(*parts: str) -> Path:
    """
    Build a path below the assets directory.

    The function rejects path traversal outside ``assets/``.
    """

    path = ASSETS_ROOT.joinpath(*parts).resolve()

    try:
        path.relative_to(ASSETS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            "Asset path must remain inside the project's assets directory."
        ) from exc

    return path


def image_path(filename: str) -> Path:
    """Return a path below ``assets/images``."""

    return asset_path("images", filename)


def icon_path(filename: str) -> Path:
    """Return a path below ``assets/icons``."""

    return asset_path("icons", filename)


def dashboard_css_path() -> Path:
    """Return the configured dashboard stylesheet path."""

    return DASHBOARD_CSS_FILE


def package_status() -> dict[str, object]:
    """Return non-secret dashboard package diagnostics."""

    return {
        "package": PACKAGE_NAME,
        "version": PACKAGE_VERSION,
        "package_root": str(PACKAGE_ROOT),
        "project_root": str(PROJECT_ROOT),
        "view_count": len(DASHBOARD_VIEWS),
        "views": list(available_view_keys()),
        "dashboard_css_exists": DASHBOARD_CSS_FILE.is_file(),
        "images_directory_exists": IMAGES_ROOT.is_dir(),
        "icons_directory_exists": ICONS_ROOT.is_dir(),
    }


__all__ = [
    "ASSETS_ROOT",
    "DASHBOARD_CSS_FILE",
    "DASHBOARD_VIEWS",
    "DashboardView",
    "ICONS_ROOT",
    "IMAGES_ROOT",
    "PACKAGE_NAME",
    "PACKAGE_ROOT",
    "PACKAGE_VERSION",
    "PROJECT_ROOT",
    "STYLES_ROOT",
    "asset_path",
    "available_view_keys",
    "dashboard_css_path",
    "get_view",
    "get_view_module",
    "icon_path",
    "image_path",
    "iter_views",
    "package_status",
]
