"""
Home page for the FT-QuPAP Streamlit dashboard.

The home page gives a safe, evidence-driven summary of the complete protocol:

- project identity and protocol design
- model-bundle readiness
- latest controlled demonstration session
- FT-QuPAP design parameters
- available research metrics
- high-level protocol status
- compact protocol flow
- generated result-figure preview
- reproducibility and security notices

No result is invented. Model metrics are shown only when the corresponding
result files exist. Missing data is rendered as unavailable.

Security boundary
-----------------
The page does not display private keys, shared secrets, K_auth, K_ctrl, K_ss,
raw subscriber identities, raw authentication tags, signatures, or
ciphertexts. Any session loaded from JSON/CSV is passed through the shared
dashboard redaction layer before rendering.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .charts import (
    chart_source_status,
    render_saved_figure_gallery,
    render_session_charts,
)
from .components import (
    KeyValueItem,
    MetricItem,
    format_percentage,
    format_probability,
    format_value,
    render_banner,
    render_card,
    render_empty_state,
    render_json_viewer,
    render_key_value_grid,
    render_metric_grid,
    render_protocol_stepper,
    render_sensitive_data_notice,
    sanitize_mapping,
)
from .status_cards import render_protocol_status
from .theme import (
    apply_dashboard_theme,
    render_divider,
    render_page_header,
    render_research_notice,
    render_section_title,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DATABASE_DIR: Final[Path] = PROJECT_ROOT / "database"
OUTPUTS_DIR: Final[Path] = PROJECT_ROOT / "outputs"
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"

MODEL_METADATA_FILE: Final[Path] = MODEL_DIR / "model_metadata.json"
THRESHOLD_FILE: Final[Path] = MODEL_DIR / "threshold.json"
FEATURE_ORDER_FILE: Final[Path] = MODEL_DIR / "feature_order.json"

MODEL_BUNDLE_FILES: Final[tuple[Path, ...]] = (
    MODEL_DIR / "gp_model.pkl",
    MODEL_DIR / "feature_scaler.pkl",
    MODEL_DIR / "calibration_model.pkl",
    THRESHOLD_FILE,
    FEATURE_ORDER_FILE,
    MODEL_METADATA_FILE,
)

PERFORMANCE_METRICS_FILE: Final[Path] = (
    DATA_DIR / "results" / "performance_metrics.csv"
)
BASELINE_COMPARISON_FILE: Final[Path] = (
    DATA_DIR / "results" / "baseline_comparison.csv"
)
DASHBOARD_RESULTS_FILE: Final[Path] = (
    DATA_DIR / "demo" / "dashboard_results.csv"
)
DEMO_SESSION_LOGS_FILE: Final[Path] = (
    DATA_DIR / "demo" / "demo_session_logs.csv"
)
DEMO_SESSIONS_FILE: Final[Path] = (
    DATABASE_DIR / "demo_sessions.json"
)

FLOWCHART_IMAGE_FILE: Final[Path] = (
    ASSETS_DIR / "images" / "protocol_flowchart.png"
)
ARCHITECTURE_IMAGE_FILE: Final[Path] = (
    ASSETS_DIR / "images" / "system_architecture.png"
)

PROTOCOL_NAME: Final[str] = "FT-QuPAP"
PROTOCOL_SUBTITLE: Final[str] = (
    "Fault-Tolerant Quantum Authentication Protocol with PQC Bootstrapping "
    "and Adaptive Eavesdropping Detection"
)

DEFAULT_PROTOCOL_VERSION: Final[str] = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
DEFAULT_MASTER_SEED: Final[int] = 20260701

DEFAULT_PROTOCOL_PARAMETERS: Final[dict[str, Any]] = {
    "ML-KEM": "ML-KEM-768",
    "ML-DSA": "ML-DSA-65",
    "KMAC tag": "128 bits",
    "Logical payload blocks": 128,
    "Logical check blocks": 32,
    "Minimum observed checks": 24,
    "Steane code": "[[7,1,3]]",
    "Total logical blocks": 160,
    "Physical data qubits": 1120,
    "Fixed QBER threshold": 0.11,
    "Maximum loss rate": 0.15,
    "Minimum GP threshold": 0.15,
    "GP retry upper boundary": 0.20,
    "Maximum attempts": 3,
}

METRIC_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "roc_auc": (
        "roc_auc",
        "roc-auc",
        "receiver_operating_characteristic_auc",
    ),
    "pr_auc": (
        "pr_auc",
        "pr-auc",
        "average_precision",
        "precision_recall_auc",
    ),
    "brier_score": (
        "brier_score",
        "brier",
        "brier_loss",
    ),
    "operational_threshold": (
        "operational_threshold",
        "gp_attack_threshold",
        "threshold",
    ),
    "attack_detection_rate": (
        "attack_detection_rate",
        "attack_detection",
        "true_positive_rate",
    ),
    "attack_acceptance_rate": (
        "attack_acceptance_rate",
        "attack_acceptance",
        "false_accept_rate",
    ),
    "valid_user_acceptance_rate": (
        "valid_user_acceptance_rate",
        "valid_user_acceptance",
        "benign_acceptance_rate",
    ),
}


class HomePageError(ValueError):
    """Raised when home-page evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class ModelBundleStatus:
    """Readiness state for the six-file GP model bundle."""

    complete: bool
    available_count: int
    required_count: int
    files: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class HomeSnapshot:
    """Pure, non-secret state used to render the home page."""

    generated_at_utc: str
    protocol_name: str
    protocol_version: str
    master_seed: int
    evidence_scope: str
    model_id: str | None
    model_bundle: ModelBundleStatus
    protocol_parameters: Mapping[str, Any]
    performance_metrics: Mapping[str, Any]
    latest_session: Mapping[str, Any] | None
    latest_session_source: str | None
    session_count: int
    chart_status: Mapping[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return sanitize_mapping(asdict(self))


def _streamlit() -> Any:
    """Import Streamlit lazily."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to render dashboard/home.py."
        ) from exc

    return st


def _utc_now_iso() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _coerce_scalar(value: Any) -> Any:
    """Convert a CSV/JSON scalar into a useful safe Python value."""

    if value is None:
        return None

    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()

    if not text:
        return None

    lower = text.lower()

    if lower in {"true", "yes", "y"}:
        return True

    if lower in {"false", "no", "n"}:
        return False

    if lower in {"none", "null", "nan", "n/a"}:
        return None

    try:
        integer = int(text)

        if str(integer) == text or text in {f"+{integer}", f"-{abs(integer)}"}:
            return integer
    except ValueError:
        pass

    try:
        number = float(text)

        if math.isfinite(number):
            return number
    except ValueError:
        pass

    if (
        (text.startswith("{") and text.endswith("}"))
        or (text.startswith("[") and text.endswith("]"))
    ):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    return text


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and redact one session or result record."""

    normalized = {
        str(key): _coerce_scalar(value)
        for key, value in record.items()
    }

    safe = sanitize_mapping(normalized)

    return dict(safe) if isinstance(safe, Mapping) else {}


def _safe_read_json(path: Path) -> Any:
    """Read a JSON file or return None when unavailable/invalid."""

    if not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _safe_read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows without requiring pandas."""

    if not path.is_file():
        return []

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            reader = csv.DictReader(stream)

            if reader.fieldnames is None:
                return []

            return [
                _normalize_record(row)
                for row in reader
                if any(
                    value is not None and str(value).strip()
                    for value in row.values()
                )
            ]
    except (OSError, UnicodeError, csv.Error):
        return []


def _extract_sessions_from_json(payload: Any) -> list[dict[str, Any]]:
    """Extract non-secret session mappings from supported JSON shapes."""

    if isinstance(payload, list):
        return [
            _normalize_record(item)
            for item in payload
            if isinstance(item, Mapping)
        ]

    if isinstance(payload, Mapping):
        sessions = payload.get("sessions")

        if isinstance(sessions, list):
            return [
                _normalize_record(item)
                for item in sessions
                if isinstance(item, Mapping)
            ]

        # Treat a standalone session object as one record only when it appears
        # session-like. Database metadata-only objects are ignored.
        session_keys = {
            "accepted",
            "decision",
            "scenario_id",
            "record_id",
            "run_id",
            "qber_raw",
            "p_attack",
        }

        if session_keys.intersection(payload):
            return [_normalize_record(payload)]

    return []


def _session_timestamp(record: Mapping[str, Any]) -> str:
    """Return the best sortable timestamp string for one session."""

    for name in (
        "executed_at_utc",
        "timestamp_utc",
        "created_at_utc",
        "updated_at_utc",
        "timestamp",
    ):
        value = record.get(name)

        if value is not None:
            return str(value)

    return ""


def load_latest_session() -> tuple[
    Mapping[str, Any] | None,
    str | None,
    int,
]:
    """
    Load the latest non-secret session from managed runtime evidence.

    Source priority:
    1. database/demo_sessions.json
    2. data/demo/dashboard_results.csv
    3. data/demo/demo_session_logs.csv
    """

    candidates: list[tuple[dict[str, Any], str]] = []

    json_payload = _safe_read_json(DEMO_SESSIONS_FILE)

    for session in _extract_sessions_from_json(json_payload):
        candidates.append(
            (session, str(DEMO_SESSIONS_FILE.relative_to(PROJECT_ROOT)))
        )

    for path in (
        DASHBOARD_RESULTS_FILE,
        DEMO_SESSION_LOGS_FILE,
    ):
        for session in _safe_read_csv_rows(path):
            candidates.append(
                (session, str(path.relative_to(PROJECT_ROOT)))
            )

    if not candidates:
        return None, None, 0

    candidates.sort(
        key=lambda item: _session_timestamp(item[0])
    )
    latest, source = candidates[-1]

    return latest, source, len(candidates)


def load_model_bundle_status() -> ModelBundleStatus:
    """Inspect the exported model bundle without loading pickle objects."""

    files: dict[str, dict[str, Any]] = {}

    for path in MODEL_BUNDLE_FILES:
        exists = path.is_file()
        files[str(path.relative_to(PROJECT_ROOT))] = {
            "exists": exists,
            "bytes": path.stat().st_size if exists else 0,
        }

    available = sum(
        bool(state["exists"])
        for state in files.values()
    )

    return ModelBundleStatus(
        complete=available == len(MODEL_BUNDLE_FILES),
        available_count=available,
        required_count=len(MODEL_BUNDLE_FILES),
        files=files,
    )


def _normalize_metric_name(name: Any) -> str:
    """Normalize a result-table metric name."""

    return (
        str(name)
        .strip()
        .lower()
        .replace("%", "percent")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _metric_mapping_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Convert common performance-table shapes into a metric mapping."""

    if not rows:
        return {}

    first = rows[0]
    columns = set(first)

    metric_column = next(
        (
            name
            for name in (
                "metric",
                "metric_name",
                "name",
                "measure",
            )
            if name in columns
        ),
        None,
    )
    value_column = next(
        (
            name
            for name in (
                "value",
                "metric_value",
                "score",
                "result",
            )
            if name in columns
        ),
        None,
    )

    if metric_column and value_column:
        return {
            _normalize_metric_name(row.get(metric_column)): row.get(
                value_column
            )
            for row in rows
            if row.get(metric_column) is not None
        }

    # Wide one-row format.
    return {
        _normalize_metric_name(key): value
        for key, value in first.items()
        if value is not None
    }


def _lookup_metric(
    metrics: Mapping[str, Any],
    canonical_name: str,
) -> Any:
    """Resolve a canonical metric using supported aliases."""

    aliases = METRIC_ALIASES.get(
        canonical_name,
        (canonical_name,),
    )

    for alias in aliases:
        normalized = _normalize_metric_name(alias)

        if normalized in metrics:
            return metrics[normalized]

    return None


def load_performance_metrics() -> dict[str, Any]:
    """Load supported performance metrics from generated evidence."""

    raw = _metric_mapping_from_rows(
        _safe_read_csv_rows(PERFORMANCE_METRICS_FILE)
    )

    # Add deployed threshold when the performance file does not carry it.
    threshold_payload = _safe_read_json(THRESHOLD_FILE)

    if isinstance(threshold_payload, Mapping):
        for key in (
            "operational_threshold",
            "minimum_operational_threshold",
            "retry_upper_probability",
            "fixed_qber_threshold",
            "maximum_loss_rate",
        ):
            if key in threshold_payload and key not in raw:
                raw[key] = threshold_payload[key]

    canonical = {}

    for name in METRIC_ALIASES:
        value = _lookup_metric(raw, name)

        if value is not None:
            canonical[name] = value

    return canonical


def _load_configuration_snapshot() -> dict[str, Any]:
    """
    Load central config.py when available.

    Import failure is non-fatal because the dashboard may be inspected before
    setup. Defaults remain aligned with the notebook configuration.
    """

    snapshot = {
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "master_seed": DEFAULT_MASTER_SEED,
        "protocol_parameters": dict(DEFAULT_PROTOCOL_PARAMETERS),
    }

    try:
        from config import SETTINGS  # type: ignore
    except Exception:
        return snapshot

    try:
        protocol = SETTINGS.protocol
        security = SETTINGS.security
        runtime = SETTINGS.runtime

        snapshot["protocol_version"] = str(protocol.version)
        snapshot["master_seed"] = int(runtime.master_seed)
        snapshot["protocol_parameters"] = {
            "ML-KEM": str(protocol.ml_kem_name),
            "ML-DSA": str(protocol.ml_dsa_name),
            "KMAC tag": f"{protocol.tag_length_bits} bits",
            "Logical payload blocks": int(
                protocol.payload_block_count
            ),
            "Logical check blocks": int(
                protocol.check_block_count
            ),
            "Minimum observed checks": int(
                protocol.min_observed_check_blocks
            ),
            "Steane code": (
                f"[[{protocol.steane_block_size},"
                f"{protocol.steane_logical_qubits},"
                f"{protocol.steane_code_distance}]]"
            ),
            "Total logical blocks": int(
                protocol.total_logical_blocks
            ),
            "Physical data qubits": int(
                protocol.total_physical_qubits
            ),
            "Fixed QBER threshold": float(
                security.fixed_qber_threshold
            ),
            "Maximum loss rate": float(
                security.max_acceptable_loss_rate
            ),
            "Minimum GP threshold": float(
                security.min_operational_gp_threshold
            ),
            "GP retry upper boundary": float(
                security.gp_gray_zone_retry_upper
            ),
            "Maximum attempts": int(
                security.max_authentication_attempts
            ),
        }
    except Exception:
        return snapshot

    return snapshot


def _load_model_identity() -> tuple[str, str | None, list[str]]:
    """Load evidence scope and model ID from model metadata."""

    payload = _safe_read_json(MODEL_METADATA_FILE)
    warnings = []

    if not isinstance(payload, Mapping):
        return "unspecified", None, [
            "Model metadata is unavailable."
        ]

    scope = str(
        payload.get("evidence_scope", "unspecified")
    )
    model_id_value = payload.get("model_id")
    model_id = (
        str(model_id_value)
        if model_id_value is not None
        else None
    )

    if scope in {
        "development_only",
        "synthetic_demo_development_only",
    }:
        warnings.append(
            "The current model bundle is marked development-only."
        )
    elif scope in {
        "unspecified",
        "source_scope_unspecified",
        "mixed_or_partially_specified",
    }:
        warnings.append(
            "The current model evidence scope is not fully established."
        )

    return scope, model_id, warnings


def build_home_snapshot() -> HomeSnapshot:
    """Build the complete non-secret home-page state."""

    configuration = _load_configuration_snapshot()
    evidence_scope, model_id, warnings = (
        _load_model_identity()
    )
    model_bundle = load_model_bundle_status()
    latest_session, source, session_count = (
        load_latest_session()
    )
    metrics = load_performance_metrics()

    if not model_bundle.complete:
        warnings.append(
            "The six-file GP deployment bundle is incomplete."
        )

    if latest_session is None:
        warnings.append(
            "No completed controlled demonstration session is available."
        )

    return HomeSnapshot(
        generated_at_utc=_utc_now_iso(),
        protocol_name=PROTOCOL_NAME,
        protocol_version=str(
            configuration["protocol_version"]
        ),
        master_seed=int(configuration["master_seed"]),
        evidence_scope=evidence_scope,
        model_id=model_id,
        model_bundle=model_bundle,
        protocol_parameters=dict(
            configuration["protocol_parameters"]
        ),
        performance_metrics=metrics,
        latest_session=latest_session,
        latest_session_source=source,
        session_count=session_count,
        chart_status=chart_source_status(),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _metric_status(
    canonical_name: str,
    value: Any,
) -> str | None:
    """Apply conservative semantic styling to available metrics."""

    number = _finite_float(value)

    if number is None:
        return None

    if canonical_name in {"roc_auc", "pr_auc"}:
        if number >= 0.90:
            return "verified"
        if number >= 0.75:
            return "warning"
        return "failed"

    if canonical_name == "brier_score":
        if number <= 0.10:
            return "verified"
        if number <= 0.20:
            return "warning"
        return "failed"

    if canonical_name == "attack_detection_rate":
        if number >= 0.90:
            return "verified"
        if number >= 0.75:
            return "warning"
        return "failed"

    if canonical_name == "attack_acceptance_rate":
        if number <= 0.05:
            return "verified"
        if number <= 0.15:
            return "warning"
        return "failed"

    return "ready"


def _performance_metric_items(
    metrics: Mapping[str, Any],
) -> tuple[MetricItem, ...]:
    """Convert available evidence metrics into dashboard items."""

    definitions = (
        ("roc_auc", "ROC-AUC", "Receiver operating characteristic"),
        ("pr_auc", "PR-AUC", "Precision–recall performance"),
        ("brier_score", "Brier score", "Probability calibration error"),
        (
            "operational_threshold",
            "GP threshold",
            "Deployed operational threshold",
        ),
        (
            "attack_detection_rate",
            "Attack detection",
            "Detected attack-session fraction",
        ),
        (
            "attack_acceptance_rate",
            "Attack acceptance",
            "Accepted attack-session fraction",
        ),
        (
            "valid_user_acceptance_rate",
            "Valid-user acceptance",
            "Accepted legitimate-session fraction",
        ),
    )

    items = []

    for key, label, help_text in definitions:
        value = metrics.get(key)

        if value is None:
            continue

        number = _finite_float(value)

        if key in {
            "attack_detection_rate",
            "attack_acceptance_rate",
            "valid_user_acceptance_rate",
        }:
            display = format_percentage(number)
        elif key in {
            "roc_auc",
            "pr_auc",
            "brier_score",
            "operational_threshold",
        }:
            display = format_probability(number)
        else:
            display = format_value(value)

        items.append(
            MetricItem(
                label=label,
                value=display,
                help_text=help_text,
                status=_metric_status(key, number),
            )
        )

    return tuple(items)


def _render_project_identity(snapshot: HomeSnapshot) -> None:
    """Render project identity and readiness information."""

    render_page_header(
        title=PROTOCOL_NAME,
        subtitle=PROTOCOL_SUBTITLE,
        icon="🛡️",
        status=(
            "ready"
            if snapshot.model_bundle.complete
            else "warning"
        ),
    )

    render_research_notice()

    identity_items = (
        KeyValueItem(
            "Protocol version",
            snapshot.protocol_version,
            code=True,
        ),
        KeyValueItem(
            "Master seed",
            snapshot.master_seed,
            code=True,
        ),
        KeyValueItem(
            "Model ID",
            snapshot.model_id or "Not available",
            code=True,
        ),
        KeyValueItem(
            "Evidence scope",
            snapshot.evidence_scope,
            status=(
                "verified"
                if snapshot.evidence_scope
                in {
                    "research_eligible",
                    "research_eligible_session_traces",
                }
                else "warning"
            ),
        ),
        KeyValueItem(
            "Model bundle",
            (
                f"{snapshot.model_bundle.available_count}/"
                f"{snapshot.model_bundle.required_count} files"
            ),
            status=(
                "verified"
                if snapshot.model_bundle.complete
                else "warning"
            ),
        ),
        KeyValueItem(
            "Recorded sessions",
            snapshot.session_count,
            status=(
                "ready"
                if snapshot.session_count > 0
                else "inactive"
            ),
        ),
    )

    render_key_value_grid(
        identity_items,
        columns=3,
    )


def _render_design_overview() -> None:
    """Render the four principal FT-QuPAP design pillars."""

    render_section_title(
        "Protocol design",
        icon="🧬",
    )

    st = _streamlit()
    columns = st.columns(4)

    cards = (
        (
            "PQC bootstrapping",
            (
                "ML-DSA-65 authenticates the server package and "
                "ML-KEM-768 creates a fresh session secret."
            ),
            "🔐",
            "ready",
        ),
        (
            "Fault-tolerant token",
            (
                "A 128-bit KMAC tag and 32 independent check blocks are "
                "encoded with the Steane [[7,1,3]] CSS code."
            ),
            "⚛️",
            "quantum",
        ),
        (
            "Hybrid detection",
            (
                "Deterministic cryptographic checks remain mandatory while "
                "a calibrated GP estimates session-level attack risk."
            ),
            "🧠",
            "ready",
        ),
        (
            "Bounded fresh retry",
            (
                "Only low-risk deterministic conditions may retry; every "
                "retry uses a new nonce, ML-KEM exchange, transcript, tag, "
                "and payload."
            ),
            "🔁",
            "retry",
        ),
    )

    for container, (title, body, icon, status) in zip(
        columns,
        cards,
    ):
        with container:
            render_card(
                title=title,
                body=body,
                icon=icon,
                status=status,
                elevated=True,
            )


def _render_protocol_parameters(
    snapshot: HomeSnapshot,
) -> None:
    """Render the notebook-aligned protocol parameters."""

    render_section_title(
        "Notebook-aligned protocol parameters",
        icon="⚙️",
    )

    items = []

    probability_names = {
        "Fixed QBER threshold",
        "Maximum loss rate",
        "Minimum GP threshold",
        "GP retry upper boundary",
    }

    for label, value in snapshot.protocol_parameters.items():
        display = (
            format_probability(value)
            if label in probability_names
            else format_value(value)
        )

        items.append(
            KeyValueItem(
                label=label,
                value=display,
                code=(
                    label
                    in {
                        "ML-KEM",
                        "ML-DSA",
                        "Steane code",
                    }
                ),
            )
        )

    render_key_value_grid(
        tuple(items),
        columns=4,
    )


def _render_performance_metrics(
    snapshot: HomeSnapshot,
) -> None:
    """Render only metrics supported by generated evidence files."""

    render_section_title(
        "Available performance evidence",
        icon="📊",
    )

    items = _performance_metric_items(
        snapshot.performance_metrics
    )

    if not items:
        render_empty_state(
            "Performance metrics unavailable",
            (
                "Run scripts/export_gp_model.py and the protocol evaluation "
                "workflow to generate data/results/performance_metrics.csv."
            ),
            icon="📉",
        )
        return

    render_metric_grid(
        items,
        columns=min(4, len(items)),
    )


def _render_latest_session(
    snapshot: HomeSnapshot,
) -> None:
    """Render the latest controlled session when available."""

    render_section_title(
        "Latest controlled session",
        icon="🧪",
    )

    if snapshot.latest_session is None:
        render_empty_state(
            "No session available",
            (
                "Run scripts/run_demo_scenarios.py to populate the "
                "controlled demonstration evidence."
            ),
            icon="🧪",
        )
        return

    render_banner(
        "Session evidence loaded",
        (
            "Source: "
            f"{snapshot.latest_session_source or 'unknown'}"
        ),
        status="ready",
        icon="📄",
    )

    render_protocol_status(
        snapshot.latest_session,
        include_hardware=True,
        status_columns=3,
        check_columns=5,
        show_metric_strip=True,
    )

    render_section_title(
        "Latest-session charts",
        icon="📈",
    )
    render_session_charts(snapshot.latest_session)

    render_json_viewer(
        snapshot.latest_session,
        title="Safe latest-session details",
        expanded=False,
    )


def _render_protocol_flow() -> None:
    """Render a compact 19-step protocol sequence."""

    render_section_title(
        "End-to-end protocol flow",
        icon="📡",
    )

    st = _streamlit()

    with st.expander(
        "Show the complete 19-step FT-QuPAP flow",
        expanded=False,
    ):
        render_protocol_stepper(
            current_step=None,
            compact=True,
        )


def _render_figure_preview() -> None:
    """Render generated result figures when available."""

    render_section_title(
        "Generated result figures",
        icon="🖼️",
    )

    shown = render_saved_figure_gallery(columns=2)

    if shown > 0:
        st = _streamlit()
        st.caption(
            (
                f"{shown} generated figure(s) found under "
                "outputs/figures/. Values are taken from stored evidence."
            )
        )


def _render_warnings(snapshot: HomeSnapshot) -> None:
    """Render snapshot readiness warnings."""

    if not snapshot.warnings:
        return

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


def _render_security_and_reproducibility() -> None:
    """Render the final operational boundary."""

    render_section_title(
        "Security and reproducibility",
        icon="🔒",
    )

    st = _streamlit()
    columns = st.columns(2)

    with columns[0]:
        render_sensitive_data_notice()

    with columns[1]:
        render_card(
            title="Reproducibility package",
            body=(
                "Use scripts/create_backup.py to archive the executed "
                "notebook, source code, model bundle, result tables, "
                "figures, metadata, and SHA-256 checksums."
            ),
            icon="📦",
            status="verified",
        )


def render() -> None:
    """Render the FT-QuPAP dashboard home page."""

    apply_dashboard_theme()

    snapshot = build_home_snapshot()

    _render_project_identity(snapshot)
    render_divider()

    _render_design_overview()
    _render_protocol_parameters(snapshot)
    _render_performance_metrics(snapshot)
    _render_latest_session(snapshot)
    _render_protocol_flow()
    _render_figure_preview()
    _render_warnings(snapshot)
    _render_security_and_reproducibility()

    st = _streamlit()
    st.caption(
        (
            f"Snapshot generated at {snapshot.generated_at_utc}. "
            "The page shows stored evidence only and does not expose "
            "cryptographic secret material."
        )
    )


def home_page_status() -> dict[str, Any]:
    """Return pure non-secret diagnostics for tests and app startup."""

    snapshot = build_home_snapshot()

    return {
        "protocol_name": snapshot.protocol_name,
        "protocol_version": snapshot.protocol_version,
        "master_seed": snapshot.master_seed,
        "evidence_scope": snapshot.evidence_scope,
        "model_id": snapshot.model_id,
        "model_bundle_complete": snapshot.model_bundle.complete,
        "model_bundle_available": (
            snapshot.model_bundle.available_count
        ),
        "model_bundle_required": (
            snapshot.model_bundle.required_count
        ),
        "performance_metric_count": len(
            snapshot.performance_metrics
        ),
        "latest_session_available": (
            snapshot.latest_session is not None
        ),
        "session_count": snapshot.session_count,
        "warning_count": len(snapshot.warnings),
    }


__all__ = [
    "ARCHITECTURE_IMAGE_FILE",
    "BASELINE_COMPARISON_FILE",
    "DASHBOARD_RESULTS_FILE",
    "DEMO_SESSIONS_FILE",
    "DEMO_SESSION_LOGS_FILE",
    "FLOWCHART_IMAGE_FILE",
    "HomePageError",
    "HomeSnapshot",
    "MODEL_BUNDLE_FILES",
    "MODEL_METADATA_FILE",
    "ModelBundleStatus",
    "PERFORMANCE_METRICS_FILE",
    "PROTOCOL_NAME",
    "PROTOCOL_SUBTITLE",
    "build_home_snapshot",
    "home_page_status",
    "load_latest_session",
    "load_model_bundle_status",
    "load_performance_metrics",
    "render",
]