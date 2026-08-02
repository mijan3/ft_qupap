"""
Mobile Station page for the FT-QuPAP Streamlit dashboard.

This page presents the prover-side operations supported by the project
flowchart and notebook:

1. Prepare a pseudonymous authentication request.
2. Send IDp, timestamp, nonce, network/service context, and protocol version.
3. Receive the ML-DSA-authenticated ephemeral ML-KEM server package.
4. Verify the server credential using the pinned ML-DSA-65 trust anchor.
5. Encapsulate with ML-KEM-768 and produce a fresh ciphertext.
6. Derive transcript-bound K_auth and K_ctrl.
7. Compute a 128-bit KMAC256 authentication tag.
8. Derive the protected control schedule.
9. Prepare 128 logical payload blocks and 32 independent check blocks.
10. Encode 160 logical blocks with Steane [[7,1,3]] into 1,120 physical
    data qubits and hand the frame to the untrusted quantum channel.

The dashboard does not implement cryptography itself. It visualizes stored
protocol-engine evidence and creates only a local request preview for the
capstone demonstration.

Security boundary
-----------------
The page never renders ML-KEM or ML-DSA secret keys, K_ss, K_auth, K_ctrl,
raw ciphertext, raw KMAC tags, raw subscriber identities, or reusable
authentication material. Nonce and transcript values are shown only as short
fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .charts import (
    ChartDataError,
    build_timing_breakdown,
    render_plotly_figure,
)
from .components import (
    KeyValueItem,
    MetricItem,
    ProtocolStep,
    format_bytes,
    format_duration,
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
from .home import load_latest_session
from .theme import (
    apply_dashboard_theme,
    render_divider,
    render_page_header,
    render_research_notice,
    render_section_title,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATABASE_DIR: Final[Path] = PROJECT_ROOT / "database"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

TRUSTED_SERVER_KEYS_FILE: Final[Path] = (
    DATABASE_DIR / "trusted_server_keys.json"
)
REGISTRATION_RECORDS_FILE: Final[Path] = (
    DATABASE_DIR / "registration_records.json"
)

PROTOCOL_NAME: Final[str] = "FT-QuPAP"
DEFAULT_PROTOCOL_VERSION: Final[str] = (
    "research-simulator-v5-1-large-ml-operational-threshold"
)
DEFAULT_SERVER_ID: Final[str] = "AS-6G-001"
DEFAULT_NETWORK_ID: Final[str] = "FT-QUPAP-6G-DEMO"
DEFAULT_CONTEXT: Final[str] = "urban"

DEFAULT_TAG_BITS: Final[int] = 128
DEFAULT_PAYLOAD_BLOCKS: Final[int] = 128
DEFAULT_CHECK_BLOCKS: Final[int] = 32
DEFAULT_TOTAL_LOGICAL_BLOCKS: Final[int] = 160
DEFAULT_STEANE_BLOCK_SIZE: Final[int] = 7
DEFAULT_PHYSICAL_QUBITS: Final[int] = 1120

MOBILE_STATION_STEP_KEYS: Final[tuple[str, ...]] = (
    "request_prepared",
    "server_package_received",
    "credential_verified",
    "mlkem_encapsulated",
    "session_keys_derived",
    "kmac_tag_generated",
    "control_schedule_generated",
    "logical_blocks_prepared",
    "steane_encoding_completed",
    "transmission_ready",
)


class MobileStationViewError(ValueError):
    """Raised when mobile-station evidence is invalid."""


@dataclass(frozen=True)
class MobileStationResources:
    """Quantum-token resource allocation."""

    tag_bits: int = DEFAULT_TAG_BITS
    payload_blocks: int = DEFAULT_PAYLOAD_BLOCKS
    check_blocks: int = DEFAULT_CHECK_BLOCKS
    total_logical_blocks: int = DEFAULT_TOTAL_LOGICAL_BLOCKS
    steane_block_size: int = DEFAULT_STEANE_BLOCK_SIZE
    physical_qubits: int = DEFAULT_PHYSICAL_QUBITS

    def validate(self) -> None:
        if self.tag_bits < 1:
            raise MobileStationViewError("tag_bits must be positive.")

        if self.payload_blocks != self.tag_bits:
            raise MobileStationViewError(
                "One logical payload block is required for each KMAC tag bit."
            )

        if self.check_blocks < 1:
            raise MobileStationViewError(
                "check_blocks must be positive."
            )

        if self.total_logical_blocks != (
            self.payload_blocks + self.check_blocks
        ):
            raise MobileStationViewError(
                "total_logical_blocks must equal payload plus check blocks."
            )

        if self.steane_block_size != 7:
            raise MobileStationViewError(
                "The project uses Steane [[7,1,3]] block size 7."
            )

        if self.physical_qubits != (
            self.total_logical_blocks * self.steane_block_size
        ):
            raise MobileStationViewError(
                "physical_qubits must equal logical blocks × 7."
            )


@dataclass(frozen=True)
class MobileStationStage:
    """One prover-side processing stage."""

    key: str
    label: str
    status: str
    owner: str
    description: str


@dataclass(frozen=True)
class MobileStationSnapshot:
    """Safe dashboard representation of one mobile-station session."""

    source: str | None
    session_available: bool
    overall_status: str

    pseudonym_display: str
    session_id: str
    request_timestamp: str
    nonce_fingerprint: str
    context: str
    network_id: str
    protocol_version: str
    server_id: str

    credential_valid: bool | None
    mlkem_encapsulation_success: bool | None
    shared_secret_match: bool | None
    key_derivation_success: bool | None
    kmac_tag_generated: bool | None
    schedule_generated: bool | None
    logical_blocks_prepared: bool | None
    steane_encoding_success: bool | None
    transmission_ready: bool | None

    ciphertext_bytes: int | None
    transcript_fingerprint: str
    schedule_entry_count: int | None
    frame_digest_fingerprint: str

    resources: MobileStationResources
    stages: tuple[MobileStationStage, ...]
    timings: Mapping[str, float]
    safe_session: Mapping[str, Any] = field(default_factory=dict)

    def to_dictionary(self) -> dict[str, Any]:
        """Return JSON-safe dashboard data."""

        return sanitize_mapping(asdict(self))


@dataclass(frozen=True)
class AuthenticationRequestPreview:
    """Local-only request preview shown by the dashboard."""

    pseudonym: str
    session_id: str
    timestamp: int
    timestamp_utc: str
    nonce_fingerprint: str
    network_id: str
    context: str
    protocol_version: str
    serialized_length_bytes: int
    request_digest: str

    def to_dictionary(self) -> dict[str, Any]:
        return asdict(self)


def _streamlit() -> Any:
    """Import Streamlit only when rendering is requested."""

    try:
        import streamlit as st  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is required to render mobile_station_view.py."
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
        "success",
        "successful",
        "verified",
        "ready",
        "generated",
        "completed",
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


def _fingerprint(
    value: Any,
    *,
    length: int = 16,
    prefix: str = "sha256:",
) -> str:
    """Create a short one-way fingerprint without exposing the value."""

    if value is None:
        return "—"

    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        payload = json.dumps(
            sanitize_mapping(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    return prefix + hashlib.sha256(payload).hexdigest()[:length]


def _safe_pseudonym(
    session: Mapping[str, Any],
) -> str:
    """Prefer a pseudonym; fingerprint any raw mobile/subscriber identifier."""

    pseudonym = _nested_first(
        session,
        (
            ("pseudonym",),
            ("pseudonymous_id",),
            ("subscriber_pseudonym",),
            ("authentication_request", "pseudonym"),
            ("authentication_request", "pseudonymous_id"),
            ("payload", "pseudonym"),
        ),
    )

    if pseudonym is not None:
        text = str(pseudonym).strip()
        return text[:64] if text else "—"

    raw_identifier = _nested_first(
        session,
        (
            ("mobile_id",),
            ("subscriber_id",),
            ("imsi",),
            ("authentication_request", "mobile_id"),
        ),
    )

    if raw_identifier is not None:
        return _fingerprint(
            raw_identifier,
            length=12,
            prefix="ID fingerprint ",
        )

    return "—"


def _safe_timestamp(value: Any) -> str:
    """Render Unix or ISO timestamps without failure."""

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


def _status_from_bool(
    value: bool | None,
    *,
    waiting: str = "inactive",
) -> str:
    if value is True:
        return "verified"

    if value is False:
        return "failed"

    return waiting


def _load_protocol_defaults() -> dict[str, Any]:
    """Read safe protocol defaults from config.py when available."""

    values = {
        "protocol_version": DEFAULT_PROTOCOL_VERSION,
        "server_id": DEFAULT_SERVER_ID,
        "network_id": DEFAULT_NETWORK_ID,
        "tag_bits": DEFAULT_TAG_BITS,
        "payload_blocks": DEFAULT_PAYLOAD_BLOCKS,
        "check_blocks": DEFAULT_CHECK_BLOCKS,
        "total_logical_blocks": DEFAULT_TOTAL_LOGICAL_BLOCKS,
        "steane_block_size": DEFAULT_STEANE_BLOCK_SIZE,
        "physical_qubits": DEFAULT_PHYSICAL_QUBITS,
    }

    try:
        from config import SETTINGS  # type: ignore
    except Exception:
        return values

    try:
        protocol = SETTINGS.protocol
        values.update(
            {
                "protocol_version": str(protocol.version),
                "server_id": str(protocol.server_id),
                "network_id": str(protocol.network_id),
                "tag_bits": int(protocol.tag_length_bits),
                "payload_blocks": int(protocol.payload_block_count),
                "check_blocks": int(protocol.check_block_count),
                "total_logical_blocks": int(
                    protocol.total_logical_blocks
                ),
                "steane_block_size": int(
                    protocol.steane_block_size
                ),
                "physical_qubits": int(
                    protocol.total_physical_qubits
                ),
            }
        )
    except Exception:
        return values

    return values


def _resolve_resources(
    session: Mapping[str, Any],
) -> MobileStationResources:
    defaults = _load_protocol_defaults()

    tag_bits = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("tag_length_bits",),
                    ("kmac_tag_bits",),
                    ("resources", "tag_bits"),
                ),
            )
        )
        or defaults["tag_bits"]
    )
    payload_blocks = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("logical_payload_blocks",),
                    ("payload_block_count",),
                    ("resources", "payload_blocks"),
                ),
            )
        )
        or defaults["payload_blocks"]
    )
    check_blocks = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("logical_check_blocks",),
                    ("check_block_count",),
                    ("resources", "check_blocks"),
                ),
            )
        )
        or defaults["check_blocks"]
    )
    total_logical = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("total_logical_blocks",),
                    ("resources", "total_logical_blocks"),
                ),
            )
        )
        or (payload_blocks + check_blocks)
    )
    steane_size = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("steane_block_size",),
                    ("resources", "steane_block_size"),
                ),
            )
        )
        or defaults["steane_block_size"]
    )
    physical = (
        _optional_int(
            _nested_first(
                session,
                (
                    ("physical_qubits",),
                    ("resources", "physical_qubits"),
                ),
            )
        )
        or (total_logical * steane_size)
    )

    resources = MobileStationResources(
        tag_bits=tag_bits,
        payload_blocks=payload_blocks,
        check_blocks=check_blocks,
        total_logical_blocks=total_logical,
        steane_block_size=steane_size,
        physical_qubits=physical,
    )

    try:
        resources.validate()
    except MobileStationViewError:
        # Stored data may be a partial representative-circuit run. The main
        # dashboard still shows notebook-aligned full-session resources.
        resources = MobileStationResources()

    return resources


def _resolve_stage_values(
    session: Mapping[str, Any],
) -> dict[str, bool | None]:
    """Resolve prover-side status fields from flat or nested evidence."""

    request_prepared = _optional_bool(
        _nested_first(
            session,
            (
                ("request_prepared",),
                ("authentication_request_created",),
                ("mobile_station", "request_prepared"),
                ("authentication_request", "created"),
            ),
        )
    )

    if request_prepared is None:
        request_has_fields = all(
            _nested_first(
                session,
                paths,
            )
            is not None
            for paths in (
                (
                    ("pseudonym",),
                    ("pseudonymous_id",),
                    ("authentication_request", "pseudonym"),
                ),
                (
                    ("timestamp",),
                    ("authentication_request", "timestamp"),
                ),
                (
                    ("nonce",),
                    ("nonce_fingerprint",),
                    ("authentication_request", "nonce"),
                ),
            )
        )
        request_prepared = True if request_has_fields else None

    server_package_received = _optional_bool(
        _nested_first(
            session,
            (
                ("server_package_received",),
                ("signed_server_package_received",),
                ("mobile_station", "server_package_received"),
                ("bootstrap", "server_package_received"),
            ),
        )
    )

    credential_verified = _optional_bool(
        _nested_first(
            session,
            (
                ("credential_valid",),
                ("server_signature_valid",),
                ("ml_dsa_verification_success",),
                ("mobile_station", "credential_valid"),
                ("bootstrap", "credential_valid"),
            ),
        )
    )

    mlkem_encapsulated = _optional_bool(
        _nested_first(
            session,
            (
                ("mlkem_encapsulation_success",),
                ("encapsulation_success",),
                ("mobile_station", "mlkem_encapsulation_success"),
                ("bootstrap", "encapsulation_success"),
            ),
        )
    )

    shared_secret_match = _optional_bool(
        _nested_first(
            session,
            (
                ("shared_secret_match",),
                ("session_secret_match",),
                ("bootstrap", "shared_secret_match"),
            ),
        )
    )

    key_derivation_success = _optional_bool(
        _nested_first(
            session,
            (
                ("key_derivation_success",),
                ("session_keys_derived",),
                ("mobile_station", "key_derivation_success"),
                ("bootstrap", "key_derivation_success"),
            ),
        )
    )

    if key_derivation_success is None and shared_secret_match is True:
        key_derivation_success = True

    kmac_tag_generated = _optional_bool(
        _nested_first(
            session,
            (
                ("kmac_tag_generated",),
                ("tag_generated",),
                ("authentication_tag_generated",),
                ("mobile_station", "kmac_tag_generated"),
            ),
        )
    )

    schedule_generated = _optional_bool(
        _nested_first(
            session,
            (
                ("control_schedule_generated",),
                ("schedule_generated",),
                ("mobile_station", "schedule_generated"),
            ),
        )
    )

    logical_blocks_prepared = _optional_bool(
        _nested_first(
            session,
            (
                ("logical_blocks_prepared",),
                ("payload_and_checks_prepared",),
                ("mobile_station", "logical_blocks_prepared"),
            ),
        )
    )

    steane_encoding_completed = _optional_bool(
        _nested_first(
            session,
            (
                ("steane_encoding_success",),
                ("css_encoding_success",),
                ("quantum_token_prepared",),
                ("mobile_station", "steane_encoding_success"),
            ),
        )
    )

    transmission_ready = _optional_bool(
        _nested_first(
            session,
            (
                ("transmission_ready",),
                ("quantum_frame_ready",),
                ("mobile_station", "transmission_ready"),
            ),
        )
    )

    mandatory = (
        request_prepared,
        credential_verified,
        mlkem_encapsulated,
        key_derivation_success,
        kmac_tag_generated,
        schedule_generated,
        logical_blocks_prepared,
        steane_encoding_completed,
    )

    if transmission_ready is None:
        if any(value is False for value in mandatory):
            transmission_ready = False
        elif all(value is True for value in mandatory):
            transmission_ready = True

    return {
        "request_prepared": request_prepared,
        "server_package_received": server_package_received,
        "credential_verified": credential_verified,
        "mlkem_encapsulated": mlkem_encapsulated,
        "session_keys_derived": key_derivation_success,
        "kmac_tag_generated": kmac_tag_generated,
        "control_schedule_generated": schedule_generated,
        "logical_blocks_prepared": logical_blocks_prepared,
        "steane_encoding_completed": steane_encoding_completed,
        "transmission_ready": transmission_ready,
        "_shared_secret_match": shared_secret_match,
    }


def _build_mobile_steps(
    values: Mapping[str, bool | None],
) -> tuple[MobileStationStage, ...]:
    definitions = (
        (
            "request_prepared",
            "Prepare authentication request",
            "Create IDp, timestamp, nonce, network and context fields.",
        ),
        (
            "server_package_received",
            "Receive signed server package",
            "Receive the ephemeral ML-KEM public key and ML-DSA credential.",
        ),
        (
            "credential_verified",
            "Verify ML-DSA-65 credential",
            "Authenticate the server package using the pinned trust anchor.",
        ),
        (
            "mlkem_encapsulated",
            "Encapsulate with ML-KEM-768",
            "Produce a fresh ciphertext and local shared-secret material.",
        ),
        (
            "session_keys_derived",
            "Derive K_auth and K_ctrl",
            "Use the transcript-bound KDF with explicit domain separation.",
        ),
        (
            "kmac_tag_generated",
            "Generate 128-bit KMAC tag",
            "Authenticate the pseudonym, timestamp, nonce, and transcript.",
        ),
        (
            "control_schedule_generated",
            "Generate control schedule",
            "Derive protected positions and measurement controls from K_ctrl.",
        ),
        (
            "logical_blocks_prepared",
            "Prepare payload and check blocks",
            "Create 128 tag-bit payload blocks and 32 independent checks.",
        ),
        (
            "steane_encoding_completed",
            "Apply Steane [[7,1,3]] encoding",
            "Encode every logical block into seven physical data qubits.",
        ),
        (
            "transmission_ready",
            "Hand frame to quantum channel",
            "Release the interleaved 1,120-qubit data frame for transmission.",
        ),
    )

    return tuple(
        MobileStationStage(
            key=key,
            label=label,
            status=_status_from_bool(
                values.get(key),
                waiting="inactive",
            ),
            owner="Mobile Station",
            description=description,
        )
        for key, label, description in definitions
    )


def _extract_timings(
    session: Mapping[str, Any],
) -> dict[str, float]:
    """Extract only mobile-station timing values."""

    timing_mapping = _mapping(session.get("timings"))
    source: dict[str, Any] = dict(timing_mapping)

    for key, value in session.items():
        if key.startswith("timing_") or key.endswith("_seconds"):
            source.setdefault(key, value)

    allowed_tokens = (
        "request",
        "credential",
        "signature",
        "encapsulation",
        "mlkem",
        "kdf",
        "key_derivation",
        "kmac",
        "tag",
        "schedule",
        "logical",
        "steane",
        "encoding",
        "mobile_station",
    )

    timings = {}

    for key, value in source.items():
        normalized = str(key).lower()

        if not any(token in normalized for token in allowed_tokens):
            continue

        number = _optional_float(value)

        if number is not None and number >= 0:
            timings[str(key)] = number

    return timings


def _ciphertext_size(
    session: Mapping[str, Any],
) -> int | None:
    explicit = _optional_int(
        _nested_first(
            session,
            (
                ("ciphertext_bytes",),
                ("mlkem_ciphertext_bytes",),
                ("bootstrap", "ciphertext_bytes"),
            ),
        )
    )

    if explicit is not None:
        return explicit

    # Sanitization redacts bytes. Size should be measured by the protocol
    # engine and stored explicitly; the view does not inspect raw ciphertext.
    return None


def build_mobile_station_snapshot(
    session: Mapping[str, Any] | None = None,
    *,
    source: str | None = None,
) -> MobileStationSnapshot:
    """Create the pure safe state used by the Mobile Station page."""

    if session is None:
        latest, latest_source, _ = load_latest_session()
        session = latest
        source = source or latest_source

    safe = sanitize_mapping(session or {})

    if not isinstance(safe, Mapping):
        safe = {}

    safe_session = dict(safe)
    defaults = _load_protocol_defaults()
    resources = _resolve_resources(safe_session)
    values = _resolve_stage_values(safe_session)
    stages = _build_mobile_steps(values)

    stage_values = [
        values[key]
        for key in MOBILE_STATION_STEP_KEYS
    ]

    if any(value is False for value in stage_values):
        overall_status = "failed"
    elif stage_values and all(value is True for value in stage_values):
        overall_status = "verified"
    elif any(value is True for value in stage_values):
        overall_status = "running"
    else:
        overall_status = "inactive"

    request = _mapping(
        _nested_first(
            safe_session,
            (
                ("authentication_request",),
                ("request",),
                ("payload",),
            ),
            default={},
        )
    )

    session_id = str(
        _first_present(
            safe_session,
            "session_id",
            "record_id",
            "run_id",
            default=_first_present(
                request,
                "session_id",
                default="—",
            ),
        )
    )
    timestamp = _first_present(
        safe_session,
        "timestamp",
        "request_timestamp",
        "executed_at_utc",
        default=_first_present(
            request,
            "timestamp",
            default=None,
        ),
    )
    nonce_value = _first_present(
        safe_session,
        "nonce",
        "request_nonce",
        default=_first_present(
            request,
            "nonce",
            "nonce_fingerprint",
            default=None,
        ),
    )
    context = str(
        _first_present(
            safe_session,
            "context",
            "service_context",
            default=_first_present(
                request,
                "context",
                "service_context",
                default=DEFAULT_CONTEXT,
            ),
        )
    )
    network_id = str(
        _first_present(
            safe_session,
            "network_id",
            default=_first_present(
                request,
                "network_id",
                default=defaults["network_id"],
            ),
        )
    )
    protocol_version = str(
        _first_present(
            safe_session,
            "protocol_version",
            default=_first_present(
                request,
                "protocol_version",
                default=defaults["protocol_version"],
            ),
        )
    )
    server_id = str(
        _first_present(
            safe_session,
            "server_id",
            "authentication_server_id",
            default=defaults["server_id"],
        )
    )

    transcript_value = _nested_first(
        safe_session,
        (
            ("transcript_hash",),
            ("transcript_digest",),
            ("bootstrap", "transcript_hash"),
        ),
    )
    frame_value = _nested_first(
        safe_session,
        (
            ("frame_digest",),
            ("quantum_frame_digest",),
            ("encoded_frame_digest",),
        ),
    )
    schedule_count = _optional_int(
        _nested_first(
            safe_session,
            (
                ("schedule_entry_count",),
                ("control_schedule_size",),
                ("resources", "schedule_entries"),
            ),
        )
    )

    return MobileStationSnapshot(
        source=source,
        session_available=bool(safe_session),
        overall_status=overall_status,
        pseudonym_display=_safe_pseudonym(safe_session),
        session_id=session_id,
        request_timestamp=_safe_timestamp(timestamp),
        nonce_fingerprint=_fingerprint(
            nonce_value,
            length=12,
            prefix="nonce:",
        ),
        context=context,
        network_id=network_id,
        protocol_version=protocol_version,
        server_id=server_id,
        credential_valid=values["credential_verified"],
        mlkem_encapsulation_success=values[
            "mlkem_encapsulated"
        ],
        shared_secret_match=values["_shared_secret_match"],
        key_derivation_success=values[
            "session_keys_derived"
        ],
        kmac_tag_generated=values["kmac_tag_generated"],
        schedule_generated=values[
            "control_schedule_generated"
        ],
        logical_blocks_prepared=values[
            "logical_blocks_prepared"
        ],
        steane_encoding_success=values[
            "steane_encoding_completed"
        ],
        transmission_ready=values["transmission_ready"],
        ciphertext_bytes=_ciphertext_size(safe_session),
        transcript_fingerprint=_fingerprint(
            transcript_value,
            length=16,
            prefix="transcript:",
        ),
        schedule_entry_count=schedule_count,
        frame_digest_fingerprint=_fingerprint(
            frame_value,
            length=16,
            prefix="frame:",
        ),
        resources=resources,
        stages=stages,
        timings=_extract_timings(safe_session),
        safe_session=safe_session,
    )


def create_request_preview(
    *,
    pseudonym: str,
    network_id: str = DEFAULT_NETWORK_ID,
    context: str = DEFAULT_CONTEXT,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
    session_id: str | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> AuthenticationRequestPreview:
    """
    Create a local request preview.

    The preview is not sent, persisted, signed, or used for authentication.
    Only a nonce fingerprint is returned.
    """

    resolved_pseudonym = pseudonym.strip()

    if not resolved_pseudonym:
        raise MobileStationViewError(
            "A pseudonymous identifier is required."
        )

    if len(resolved_pseudonym) > 128:
        raise MobileStationViewError(
            "The pseudonym must not exceed 128 characters."
        )

    resolved_context = context.strip().lower()

    if resolved_context not in {
        "urban",
        "suburban",
        "rural",
    }:
        raise MobileStationViewError(
            "Context must be urban, suburban, or rural."
        )

    resolved_network = network_id.strip()

    if not resolved_network:
        raise MobileStationViewError(
            "network_id is required."
        )

    resolved_timestamp = (
        int(time.time())
        if timestamp is None
        else int(timestamp)
    )
    resolved_nonce = nonce or secrets.token_hex(16)
    resolved_session_id = (
        session_id.strip()
        if session_id is not None and session_id.strip()
        else f"MS-PREVIEW-{resolved_timestamp}"
    )

    canonical = {
        "pseudonym": resolved_pseudonym,
        "session_id": resolved_session_id,
        "timestamp": resolved_timestamp,
        "nonce": resolved_nonce,
        "network_id": resolved_network,
        "context": resolved_context,
        "protocol_version": protocol_version,
    }
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return AuthenticationRequestPreview(
        pseudonym=resolved_pseudonym,
        session_id=resolved_session_id,
        timestamp=resolved_timestamp,
        timestamp_utc=datetime.fromtimestamp(
            resolved_timestamp,
            tz=timezone.utc,
        ).isoformat(timespec="seconds"),
        nonce_fingerprint=_fingerprint(
            resolved_nonce,
            length=12,
            prefix="nonce:",
        ),
        network_id=resolved_network,
        context=resolved_context,
        protocol_version=protocol_version,
        serialized_length_bytes=len(serialized),
        request_digest=_fingerprint(
            serialized,
            length=20,
            prefix="request:",
        ),
    )


def _stage_protocol_steps(
    snapshot: MobileStationSnapshot,
) -> tuple[ProtocolStep, ...]:
    """Convert mobile-station stages to shared protocol-step components."""

    return tuple(
        ProtocolStep(
            number=index,
            key=stage.key,
            label=stage.label,
            stage=(
                "request"
                if index == 1
                else (
                    "ml_dsa"
                    if index in {2, 3}
                    else (
                        "ml_kem"
                        if index == 4
                        else (
                            "key_derivation"
                            if index == 5
                            else (
                                "kmac"
                                if index == 6
                                else (
                                    "schedule"
                                    if index in {7, 8}
                                    else (
                                        "steane_encoding"
                                        if index == 9
                                        else "quantum_channel"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            description=stage.description,
            owner=stage.owner,
            status=stage.status,
        )
        for index, stage in enumerate(snapshot.stages, start=1)
    )


def _render_overview(
    snapshot: MobileStationSnapshot,
) -> None:
    render_page_header(
        title="Mobile Station",
        subtitle=(
            "Pseudonymous request, authenticated PQC bootstrap, KMAC tag, "
            "Steane CSS token preparation, and quantum-frame handoff"
        ),
        icon="📱",
        status=snapshot.overall_status,
    )

    render_research_notice()

    if snapshot.session_available:
        render_banner(
            "Stored session loaded",
            (
                "Mobile-station evidence source: "
                f"{snapshot.source or 'runtime session'}"
            ),
            status="ready",
            icon="📄",
        )
    else:
        render_banner(
            "No completed session loaded",
            (
                "The page is showing notebook-aligned defaults. Run a "
                "controlled scenario to populate live mobile-station states."
            ),
            status="inactive",
            icon="📭",
        )

    metrics = (
        MetricItem(
            "Tag length",
            f"{snapshot.resources.tag_bits} bits",
            status="verified",
            icon="🏷️",
        ),
        MetricItem(
            "Logical payload",
            snapshot.resources.payload_blocks,
            help_text="One logical block per KMAC tag bit",
            status="quantum",
            icon="◻️",
        ),
        MetricItem(
            "Check blocks",
            snapshot.resources.check_blocks,
            help_text="Independent raw-QBER evidence",
            status="quantum",
            icon="🔬",
        ),
        MetricItem(
            "Total logical blocks",
            snapshot.resources.total_logical_blocks,
            status="quantum",
            icon="🧩",
        ),
        MetricItem(
            "Physical data qubits",
            f"{snapshot.resources.physical_qubits:,}",
            help_text="Syndrome-level full-session model",
            status="quantum",
            icon="⚛️",
        ),
        MetricItem(
            "Transmission state",
            (
                "Ready"
                if snapshot.transmission_ready is True
                else (
                    "Blocked"
                    if snapshot.transmission_ready is False
                    else "Not available"
                )
            ),
            status=_status_from_bool(
                snapshot.transmission_ready
            ),
            icon="📡",
        ),
    )

    render_metric_grid(metrics, columns=6)


def _render_request_metadata(
    snapshot: MobileStationSnapshot,
) -> None:
    render_section_title(
        "Authentication request metadata",
        icon="🪪",
    )

    items = (
        KeyValueItem(
            "Pseudonymous identity",
            snapshot.pseudonym_display,
            code=True,
        ),
        KeyValueItem(
            "Session ID",
            snapshot.session_id,
            code=True,
        ),
        KeyValueItem(
            "Timestamp",
            snapshot.request_timestamp,
            code=True,
        ),
        KeyValueItem(
            "Nonce fingerprint",
            snapshot.nonce_fingerprint,
            code=True,
            help_text="The raw nonce is not displayed.",
        ),
        KeyValueItem(
            "Channel context",
            snapshot.context,
        ),
        KeyValueItem(
            "Network ID",
            snapshot.network_id,
            code=True,
        ),
        KeyValueItem(
            "Authentication server",
            snapshot.server_id,
            code=True,
        ),
        KeyValueItem(
            "Protocol version",
            snapshot.protocol_version,
            code=True,
        ),
    )

    render_key_value_grid(items, columns=4)


def _render_request_builder() -> None:
    """Render a local-only request-preview form."""

    st = _streamlit()

    with st.expander(
        "Create a local authentication-request preview",
        expanded=False,
    ):
        st.caption(
            (
                "This tool demonstrates payload preparation only. It does "
                "not contact the server or execute ML-DSA, ML-KEM, KMAC, "
                "or quantum operations."
            )
        )

        left, right = st.columns(2)

        with left:
            pseudonym = st.text_input(
                "Pseudonymous ID",
                value="PID-6G-UE-0001",
                max_chars=128,
                key="ms_preview_pseudonym",
            )
            network_id = st.text_input(
                "Network ID",
                value=DEFAULT_NETWORK_ID,
                key="ms_preview_network_id",
            )

        with right:
            context = st.selectbox(
                "Channel context",
                ("urban", "suburban", "rural"),
                key="ms_preview_context",
            )
            protocol_version = st.text_input(
                "Protocol version",
                value=DEFAULT_PROTOCOL_VERSION,
                key="ms_preview_protocol_version",
            )

        create_preview = st.button(
            "Prepare request preview",
            type="primary",
            use_container_width=True,
            key="ms_prepare_request_preview",
        )

        if create_preview:
            try:
                preview = create_request_preview(
                    pseudonym=pseudonym,
                    network_id=network_id,
                    context=context,
                    protocol_version=protocol_version,
                )
                st.session_state[
                    "ft_qupap_mobile_request_preview"
                ] = preview.to_dictionary()
            except MobileStationViewError as exc:
                st.error(str(exc))

        preview_data = st.session_state.get(
            "ft_qupap_mobile_request_preview"
        )

        if isinstance(preview_data, Mapping):
            render_banner(
                "Request preview prepared",
                (
                    "The nonce is represented only by a one-way "
                    "fingerprint and the preview is not persisted."
                ),
                status="verified",
                icon="✅",
            )
            render_key_value_grid(
                (
                    KeyValueItem(
                        "Pseudonym",
                        preview_data.get("pseudonym"),
                        code=True,
                    ),
                    KeyValueItem(
                        "Session ID",
                        preview_data.get("session_id"),
                        code=True,
                    ),
                    KeyValueItem(
                        "Timestamp",
                        preview_data.get("timestamp_utc"),
                        code=True,
                    ),
                    KeyValueItem(
                        "Nonce fingerprint",
                        preview_data.get("nonce_fingerprint"),
                        code=True,
                    ),
                    KeyValueItem(
                        "Context",
                        preview_data.get("context"),
                    ),
                    KeyValueItem(
                        "Serialized request",
                        format_bytes(
                            preview_data.get(
                                "serialized_length_bytes"
                            )
                        ),
                    ),
                    KeyValueItem(
                        "Request digest",
                        preview_data.get("request_digest"),
                        code=True,
                    ),
                    KeyValueItem(
                        "Transmission",
                        "Not sent",
                        status="inactive",
                    ),
                ),
                columns=4,
            )


def _render_mobile_pipeline(
    snapshot: MobileStationSnapshot,
) -> None:
    render_section_title(
        "Mobile Station processing pipeline",
        icon="🧬",
    )

    render_protocol_stepper(
        steps=_stage_protocol_steps(snapshot),
        compact=False,
    )


def _render_pqc_summary(
    snapshot: MobileStationSnapshot,
) -> None:
    render_section_title(
        "Authenticated PQC bootstrap",
        icon="🔐",
    )

    st = _streamlit()
    columns = st.columns(4)

    cards = (
        (
            "ML-DSA-65 credential",
            (
                "Verified"
                if snapshot.credential_valid is True
                else (
                    "Failed"
                    if snapshot.credential_valid is False
                    else "Not available"
                )
            ),
            (
                "The Mobile Station validates the signed ephemeral "
                "ML-KEM server package against the pinned trust anchor."
            ),
            "🪪",
            _status_from_bool(snapshot.credential_valid),
        ),
        (
            "ML-KEM-768 encapsulation",
            (
                "Completed"
                if snapshot.mlkem_encapsulation_success is True
                else (
                    "Failed"
                    if snapshot.mlkem_encapsulation_success is False
                    else "Not available"
                )
            ),
            (
                "The page records only success state and ciphertext size; "
                "ciphertext and shared-secret bytes remain hidden."
            ),
            "🔑",
            _status_from_bool(
                snapshot.mlkem_encapsulation_success
            ),
        ),
        (
            "Transcript-bound KDF",
            (
                "Derived"
                if snapshot.key_derivation_success is True
                else (
                    "Failed"
                    if snapshot.key_derivation_success is False
                    else "Not available"
                )
            ),
            (
                "K_auth and K_ctrl are domain-separated from the fresh "
                "session secret and transcript hash."
            ),
            "🧬",
            _status_from_bool(
                snapshot.key_derivation_success
            ),
        ),
        (
            "KMAC256 tag",
            (
                "Generated"
                if snapshot.kmac_tag_generated is True
                else (
                    "Failed"
                    if snapshot.kmac_tag_generated is False
                    else "Not available"
                )
            ),
            (
                "The 128-bit tag authenticates the pseudonym, timestamp, "
                "nonce, and transcript without exposing a raw IMSI."
            ),
            "🏷️",
            _status_from_bool(snapshot.kmac_tag_generated),
        ),
    )

    for container, (
        title,
        value,
        body,
        icon,
        status,
    ) in zip(columns, cards):
        with container:
            render_card(
                title=title,
                body=f"{value}\n\n{body}",
                icon=icon,
                status=status,
                elevated=True,
            )

    details = (
        KeyValueItem(
            "Shared-secret consistency",
            (
                "Matched"
                if snapshot.shared_secret_match is True
                else (
                    "Mismatch"
                    if snapshot.shared_secret_match is False
                    else "Not available"
                )
            ),
            status=_status_from_bool(
                snapshot.shared_secret_match
            ),
        ),
        KeyValueItem(
            "Ciphertext size",
            format_bytes(snapshot.ciphertext_bytes),
            help_text="Raw ML-KEM ciphertext is never rendered.",
        ),
        KeyValueItem(
            "Transcript fingerprint",
            snapshot.transcript_fingerprint,
            code=True,
        ),
        KeyValueItem(
            "Control schedule",
            (
                f"{snapshot.schedule_entry_count} entries"
                if snapshot.schedule_entry_count is not None
                else (
                    "Generated"
                    if snapshot.schedule_generated is True
                    else "Not available"
                )
            ),
            status=_status_from_bool(
                snapshot.schedule_generated
            ),
        ),
    )

    render_key_value_grid(details, columns=4)


def _render_quantum_token(
    snapshot: MobileStationSnapshot,
) -> None:
    render_section_title(
        "Steane CSS quantum-token preparation",
        icon="⚛️",
    )

    resource_items = (
        KeyValueItem(
            "Tag-to-payload mapping",
            (
                f"{snapshot.resources.tag_bits} tag bits → "
                f"{snapshot.resources.payload_blocks} logical blocks"
            ),
        ),
        KeyValueItem(
            "Independent checks",
            f"{snapshot.resources.check_blocks} logical blocks",
        ),
        KeyValueItem(
            "Interleaved logical frame",
            f"{snapshot.resources.total_logical_blocks} blocks",
        ),
        KeyValueItem(
            "Steane expansion",
            (
                f"1 logical qubit → "
                f"{snapshot.resources.steane_block_size} physical qubits"
            ),
        ),
        KeyValueItem(
            "Physical data frame",
            f"{snapshot.resources.physical_qubits:,} qubits",
            status="quantum",
        ),
        KeyValueItem(
            "Logical blocks prepared",
            (
                "Yes"
                if snapshot.logical_blocks_prepared is True
                else (
                    "No"
                    if snapshot.logical_blocks_prepared is False
                    else "Not available"
                )
            ),
            status=_status_from_bool(
                snapshot.logical_blocks_prepared
            ),
        ),
        KeyValueItem(
            "Steane encoding",
            (
                "Completed"
                if snapshot.steane_encoding_success is True
                else (
                    "Failed"
                    if snapshot.steane_encoding_success is False
                    else "Not available"
                )
            ),
            status=_status_from_bool(
                snapshot.steane_encoding_success
            ),
        ),
        KeyValueItem(
            "Frame fingerprint",
            snapshot.frame_digest_fingerprint,
            code=True,
            help_text="The encoded frame itself is not displayed.",
        ),
    )

    render_key_value_grid(resource_items, columns=4)

    render_banner(
        "Simulation interpretation",
        (
            "Complete sessions use the scalable syndrome-level Steane "
            "model. Representative Qiskit/Aer circuits validate selected "
            "encoded blocks; this is not a physical 1,120-qubit device run."
        ),
        status="warning",
        icon="🧪",
    )


def _render_timings(
    snapshot: MobileStationSnapshot,
) -> None:
    render_section_title(
        "Mobile Station timing evidence",
        icon="⏱️",
    )

    if not snapshot.timings:
        render_empty_state(
            "Timing data unavailable",
            (
                "The selected session does not contain prover-side request, "
                "ML-KEM, KDF, KMAC, schedule, or encoding timings."
            ),
            icon="⏱️",
        )
        return

    timing_metrics = tuple(
        MetricItem(
            label=(
                key.replace("_seconds", "")
                .replace("_s", "")
                .replace("timing_", "")
                .replace("_", " ")
                .title()
            ),
            value=format_duration(value),
            status="ready",
            icon="⏱️",
        )
        for key, value in list(snapshot.timings.items())[:6]
    )

    render_metric_grid(
        timing_metrics,
        columns=min(3, len(timing_metrics)),
    )

    try:
        figure = build_timing_breakdown(
            snapshot.timings,
            title="Mobile Station Processing-Time Breakdown",
        )
        render_plotly_figure(
            figure,
            key="mobile_station_timing_chart",
        )
    except ChartDataError as exc:
        render_empty_state(
            "Timing chart unavailable",
            str(exc),
            icon="📉",
        )


def render(
    session: Mapping[str, Any] | None = None,
    *,
    source: str | None = None,
) -> None:
    """Render the FT-QuPAP Mobile Station dashboard page."""

    apply_dashboard_theme()

    snapshot = build_mobile_station_snapshot(
        session,
        source=source,
    )

    _render_overview(snapshot)
    render_divider()

    _render_request_metadata(snapshot)
    _render_request_builder()
    _render_mobile_pipeline(snapshot)
    _render_pqc_summary(snapshot)
    _render_quantum_token(snapshot)
    _render_timings(snapshot)

    render_section_title(
        "Safe session evidence",
        icon="🔒",
    )
    render_sensitive_data_notice()

    if snapshot.session_available:
        render_json_viewer(
            snapshot.safe_session,
            title="Redacted Mobile Station session record",
            expanded=False,
        )

    st = _streamlit()
    st.caption(
        (
            "The Mobile Station page displays prover-side evidence only. "
            "Final acceptance is made by the Authentication Server after "
            "freshness, replay, decoder, KMAC, loss, QBER, and GP checks."
        )
    )


def mobile_station_view_status(
    session: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return pure diagnostics for tests and application startup."""

    snapshot = build_mobile_station_snapshot(session)

    return {
        "session_available": snapshot.session_available,
        "source": snapshot.source,
        "overall_status": snapshot.overall_status,
        "stage_count": len(snapshot.stages),
        "completed_stage_count": sum(
            stage.status == "verified"
            for stage in snapshot.stages
        ),
        "failed_stage_count": sum(
            stage.status == "failed"
            for stage in snapshot.stages
        ),
        "tag_bits": snapshot.resources.tag_bits,
        "payload_blocks": snapshot.resources.payload_blocks,
        "check_blocks": snapshot.resources.check_blocks,
        "total_logical_blocks": (
            snapshot.resources.total_logical_blocks
        ),
        "physical_qubits": snapshot.resources.physical_qubits,
        "transmission_ready": snapshot.transmission_ready,
        "sensitive_material_displayed": False,
    }


__all__ = [
    "AuthenticationRequestPreview",
    "DEFAULT_CHECK_BLOCKS",
    "DEFAULT_PHYSICAL_QUBITS",
    "DEFAULT_TAG_BITS",
    "MobileStationResources",
    "MobileStationSnapshot",
    "MobileStationStage",
    "MobileStationViewError",
    "REGISTRATION_RECORDS_FILE",
    "TRUSTED_SERVER_KEYS_FILE",
    "build_mobile_station_snapshot",
    "create_request_preview",
    "mobile_station_view_status",
    "render",
]