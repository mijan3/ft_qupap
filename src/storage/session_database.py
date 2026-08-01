"""
FT-QuPAP Session Database
=========================

JSON-backed storage for compact, non-secret FT-QuPAP v5.1 session evidence.

The notebook returns a complete session dictionary containing request,
credential, ML-KEM, KMAC, quantum-channel, decoder, GP, retry, and timing
objects. This module stores only the fields required by the dashboard,
session history, and evaluation pipeline.

Never store ML-DSA/ML-KEM private keys, shared secrets, K_auth, K_ctrl, raw
nonces, ciphertexts, raw tags, encrypted schedules, transcript material, or
quantum state vectors.
"""

from __future__ import annotations

import copy
import hmac
import json
import math
import os
import tempfile
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_DATABASE_PATH = Path("database/demo_sessions.json")

DATABASE_VERSION = 1
PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DEFAULT_PROTOCOL_VARIANT = "P1_FT_QuPAP_GP"

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_REQUIRED_CHECK_BLOCKS = 24
DEFAULT_GP_GRAY_ZONE_RETRY_UPPER = 0.20

SUPPORTED_CONTEXTS = {
    "urban",
    "suburban",
    "rural",
    "unknown",
}

SUPPORTED_STATUSES = {
    "accepted",
    "rejected",
    "failed",
    "aborted",
}

SUPPORTED_OUTCOMES = {
    "accepted",
    "accepted_after_retry",
    "rejected_replay",
    "rejected_credential",
    "rejected_ciphertext",
    "rejected_deterministic",
    "rejected_gp",
    "rejected_retry_exhausted",
    "rejected",
    "failed",
    "aborted",
}

SAFE_FEATURE_NAMES = (
    "qber_raw",
    "mean_syndrome_weight",
    "max_syndrome_weight",
    "correction_failure_rate",
    "loss_rate",
    "noise_estimate",
    "ctx_urban",
    "ctx_suburban",
    "ctx_rural",
)

SAFE_TIMING_NAMES = (
    "credential_and_kem_keygen_s",
    "mlkem_encapsulation_s",
    "mlkem_decapsulation_s",
    "tag_and_schedule_s",
    "css_encoding_s",
    "quantum_channel_simulation_s",
    "measurement_and_css_decoding_s",
    "gp_inference_s",
    "end_to_end_s",
    "total_retry_end_to_end_s",
)

SAFE_ATTEMPT_FIELDS = (
    "attempt",
    "accepted",
    "reason",
    "deterministic_pass",
    "deterministic_reasons",
    "qber_raw",
    "qber_mismatches",
    "qber_observed",
    "observed_check_blocks",
    "loss_rate",
    "p_attack",
    "uncertainty",
    "gp_attack_threshold",
    "tag_recovered",
    "retryable",
    "timing_end_to_end_s",
)

PROHIBITED_METADATA_FIELDS = {
    "imsi",
    "raw_imsi",
    "nonce",
    "raw_nonce",
    "private_key",
    "secret_key",
    "ml_dsa_private_key",
    "mldsa_private_key",
    "ml_kem_private_key",
    "mlkem_private_key",
    "shared_secret",
    "session_secret",
    "k_auth",
    "k_ctrl",
    "authentication_key",
    "control_key",
    "ciphertext",
    "mlkem_ciphertext",
    "expected_tag",
    "received_tag",
    "tag_ms",
    "kmac_tag",
    "authentication_tag",
    "encrypted_schedule",
    "schedule_binding",
    "transcript_hash",
    "quantum_state",
    "statevector",
    "density_matrix",
    "eve_fraction",
    "eve_mode",
}


def current_utc_timestamp() -> str:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionRecord:
    """Compact persistent record for one completed FT-QuPAP session."""

    session_id: str

    pseudonym_id: str | None = None
    request_timestamp: int | None = None
    request_type: str = "FT-QuPAP-Authentication"

    scenario_name: str = "normal_session"
    channel_name: str = "unknown"
    context: str = "unknown"

    protocol_version: str = PROTOCOL_VERSION
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT

    created_at: str = field(default_factory=current_utc_timestamp)
    completed_at: str = field(default_factory=current_utc_timestamp)

    status: str = "rejected"
    accepted: bool = False
    outcome: str = "rejected"
    reason: str = "unspecified"

    deterministic_pass: bool = False
    deterministic_reasons: list[str] = field(default_factory=list)

    schedule_valid: bool | None = None
    tag_recovered: bool | None = None

    qber_raw: float | None = None
    qber_mismatches: int | None = None
    qber_observed: int | None = None
    observed_check_blocks: int | None = None
    required_check_blocks: int = DEFAULT_REQUIRED_CHECK_BLOCKS
    loss_rate: float | None = None

    payload_failure_count: int = 0
    corrected_block_count: int = 0
    uncorrectable_block_count: int = 0

    features: dict[str, float] = field(default_factory=dict)
    p_attack: float | None = None
    uncertainty: float | None = None
    raw_calibration_gp_attack_threshold: float | None = None
    gp_attack_threshold: float | None = None
    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_RETRY_UPPER
    )

    retry_attempts: int = 1
    retry_used: bool = False
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    attempt_history: list[dict[str, Any]] = field(default_factory=list)

    physical_qubits: int | None = None
    logical_payload_blocks: int | None = None
    logical_check_blocks: int | None = None

    use_css: bool = True
    bootstrap_mode: str = "mlkem"
    decision_mode: str = "gp"

    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate all stored session fields."""

        self.session_id = normalize_required_string(
            "session_id",
            self.session_id,
        )
        self.pseudonym_id = normalize_optional_string(self.pseudonym_id)

        if self.request_timestamp is not None:
            validate_nonnegative_integer(
                "request_timestamp",
                self.request_timestamp,
            )

        self.request_type = normalize_required_string(
            "request_type",
            self.request_type,
        )
        self.scenario_name = normalize_required_string(
            "scenario_name",
            self.scenario_name,
        )
        self.channel_name = normalize_required_string(
            "channel_name",
            self.channel_name,
        )
        self.context = normalize_context(self.context)

        self.protocol_version = normalize_required_string(
            "protocol_version",
            self.protocol_version,
        )
        self.protocol_variant = normalize_required_string(
            "protocol_variant",
            self.protocol_variant,
        )

        self.created_at = normalize_timestamp(self.created_at)
        self.completed_at = normalize_timestamp(self.completed_at)

        self.status = validate_choice(
            "status",
            self.status,
            SUPPORTED_STATUSES,
        )
        validate_boolean("accepted", self.accepted)
        self.outcome = validate_choice(
            "outcome",
            self.outcome,
            SUPPORTED_OUTCOMES,
        )
        self.reason = normalize_required_string("reason", self.reason)

        validate_boolean(
            "deterministic_pass",
            self.deterministic_pass,
        )
        self.deterministic_reasons = normalize_reason_list(
            self.deterministic_reasons
        )

        self.schedule_valid = normalize_optional_boolean(
            self.schedule_valid
        )
        self.tag_recovered = normalize_optional_boolean(
            self.tag_recovered
        )

        count_fields = {
            "payload_failure_count": self.payload_failure_count,
            "corrected_block_count": self.corrected_block_count,
            "uncorrectable_block_count": self.uncorrectable_block_count,
            "required_check_blocks": self.required_check_blocks,
            "retry_attempts": self.retry_attempts,
            "max_attempts": self.max_attempts,
        }

        for name, value in count_fields.items():
            validate_nonnegative_integer(name, value)

        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be at least one.")

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")

        if self.retry_attempts > self.max_attempts:
            raise ValueError(
                "retry_attempts cannot exceed max_attempts."
            )

        validate_boolean("retry_used", self.retry_used)

        if self.retry_used != (self.retry_attempts > 1):
            raise ValueError(
                "retry_used must be True exactly when "
                "retry_attempts is greater than one."
            )

        optional_count_fields = {
            "qber_mismatches": self.qber_mismatches,
            "qber_observed": self.qber_observed,
            "observed_check_blocks": self.observed_check_blocks,
            "physical_qubits": self.physical_qubits,
            "logical_payload_blocks": self.logical_payload_blocks,
            "logical_check_blocks": self.logical_check_blocks,
        }

        for name, value in optional_count_fields.items():
            if value is not None:
                validate_nonnegative_integer(name, value)

        probability_fields = {
            "qber_raw": self.qber_raw,
            "loss_rate": self.loss_rate,
            "p_attack": self.p_attack,
            "uncertainty": self.uncertainty,
            "raw_calibration_gp_attack_threshold": (
                self.raw_calibration_gp_attack_threshold
            ),
            "gp_attack_threshold": self.gp_attack_threshold,
            "gp_gray_zone_retry_upper": (
                self.gp_gray_zone_retry_upper
            ),
        }

        for name, value in probability_fields.items():
            validate_optional_probability(name, value)

        self.features = normalize_features(self.features)
        self.timings = normalize_timings(self.timings)
        self.attempt_history = normalize_attempt_history(
            self.attempt_history
        )

        validate_boolean("use_css", self.use_css)
        self.bootstrap_mode = normalize_required_string(
            "bootstrap_mode",
            self.bootstrap_mode,
        )
        self.decision_mode = normalize_required_string(
            "decision_mode",
            self.decision_mode,
        )

        self.metadata = validate_metadata(self.metadata)

        if self.accepted:
            if self.status != "accepted":
                raise ValueError(
                    "An accepted session must use status 'accepted'."
                )
            if self.outcome not in {
                "accepted",
                "accepted_after_retry",
            }:
                raise ValueError(
                    "An accepted session must use an accepted outcome."
                )
        elif self.outcome in {
            "accepted",
            "accepted_after_retry",
        }:
            raise ValueError(
                "A rejected session cannot use an accepted outcome."
            )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible session dictionary."""

        return asdict(self)

    @classmethod
    def from_dictionary(
        cls,
        data: Mapping[str, Any],
    ) -> "SessionRecord":
        """Create a record from persistent JSON data."""

        if not isinstance(data, Mapping):
            raise TypeError("Session data must be a mapping.")

        return cls(
            session_id=str(data["session_id"]),
            pseudonym_id=normalize_optional_string(
                data.get("pseudonym_id")
            ),
            request_timestamp=optional_integer(
                data.get("request_timestamp")
            ),
            request_type=str(
                data.get(
                    "request_type",
                    "FT-QuPAP-Authentication",
                )
            ),
            scenario_name=str(
                data.get("scenario_name", "normal_session")
            ),
            channel_name=str(data.get("channel_name", "unknown")),
            context=str(data.get("context", "unknown")),
            protocol_version=str(
                data.get("protocol_version", PROTOCOL_VERSION)
            ),
            protocol_variant=str(
                data.get(
                    "protocol_variant",
                    DEFAULT_PROTOCOL_VARIANT,
                )
            ),
            created_at=str(
                data.get("created_at", current_utc_timestamp())
            ),
            completed_at=str(
                data.get("completed_at", current_utc_timestamp())
            ),
            status=str(data.get("status", "rejected")),
            accepted=bool(data.get("accepted", False)),
            outcome=str(data.get("outcome", "rejected")),
            reason=str(data.get("reason", "unspecified")),
            deterministic_pass=bool(
                data.get("deterministic_pass", False)
            ),
            deterministic_reasons=list(
                data.get("deterministic_reasons", [])
            ),
            schedule_valid=normalize_optional_boolean(
                data.get("schedule_valid")
            ),
            tag_recovered=normalize_optional_boolean(
                data.get("tag_recovered")
            ),
            qber_raw=optional_float(data.get("qber_raw")),
            qber_mismatches=optional_integer(
                data.get("qber_mismatches")
            ),
            qber_observed=optional_integer(
                data.get("qber_observed")
            ),
            observed_check_blocks=optional_integer(
                data.get("observed_check_blocks")
            ),
            required_check_blocks=int(
                data.get(
                    "required_check_blocks",
                    DEFAULT_REQUIRED_CHECK_BLOCKS,
                )
            ),
            loss_rate=optional_float(data.get("loss_rate")),
            payload_failure_count=int(
                data.get("payload_failure_count", 0)
            ),
            corrected_block_count=int(
                data.get("corrected_block_count", 0)
            ),
            uncorrectable_block_count=int(
                data.get("uncorrectable_block_count", 0)
            ),
            features=dict(data.get("features", {})),
            p_attack=optional_float(data.get("p_attack")),
            uncertainty=optional_float(data.get("uncertainty")),
            raw_calibration_gp_attack_threshold=optional_float(
                data.get("raw_calibration_gp_attack_threshold")
            ),
            gp_attack_threshold=optional_float(
                data.get("gp_attack_threshold")
            ),
            gp_gray_zone_retry_upper=float(
                data.get(
                    "gp_gray_zone_retry_upper",
                    DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
                )
            ),
            retry_attempts=int(data.get("retry_attempts", 1)),
            retry_used=bool(data.get("retry_used", False)),
            max_attempts=int(
                data.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
            ),
            attempt_history=list(data.get("attempt_history", [])),
            physical_qubits=optional_integer(
                data.get("physical_qubits")
            ),
            logical_payload_blocks=optional_integer(
                data.get("logical_payload_blocks")
            ),
            logical_check_blocks=optional_integer(
                data.get("logical_check_blocks")
            ),
            use_css=bool(data.get("use_css", True)),
            bootstrap_mode=str(data.get("bootstrap_mode", "mlkem")),
            decision_mode=str(data.get("decision_mode", "gp")),
            timings=dict(data.get("timings", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_protocol_result(
        cls,
        protocol_result: Mapping[str, Any],
        *,
        session_id: str | None = None,
        scenario_name: str | None = None,
        protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SessionRecord":
        """Extract a compact record from a notebook-style result."""

        if not isinstance(protocol_result, Mapping):
            raise TypeError("protocol_result must be a mapping.")

        validate_positive_integer("max_attempts", max_attempts)

        request = safe_mapping(protocol_result.get("request"))
        decision = extract_decision(protocol_result)
        features = safe_mapping(protocol_result.get("features"))
        timings = safe_mapping(protocol_result.get("timings"))
        channel = safe_mapping(protocol_result.get("channel"))

        accepted = bool(decision.get("accepted", False))
        reason = str(decision.get("reason", "unspecified"))
        deterministic_reasons = normalize_reason_list(
            list(decision.get("deterministic_reasons", []))
        )

        raw_attempt_history = protocol_result.get(
            "attempt_history",
            [],
        )
        attempt_history = normalize_attempt_history(raw_attempt_history)

        retry_attempts = int(
            protocol_result.get(
                "retry_attempts",
                len(attempt_history) if attempt_history else 1,
            )
        )
        retry_attempts = max(retry_attempts, 1)

        if not attempt_history:
            attempt_history = [
                build_attempt_summary(
                    protocol_result,
                    attempt=retry_attempts,
                )
            ]
        else:
            final_summary = build_attempt_summary(
                protocol_result,
                attempt=retry_attempts,
            )
            replaced = False

            for index, attempt in enumerate(attempt_history):
                if attempt.get("attempt") == retry_attempts:
                    attempt_history[index] = {
                        **attempt,
                        **final_summary,
                    }
                    replaced = True
                    break

            if not replaced:
                attempt_history.append(final_summary)

            attempt_history.sort(
                key=lambda item: int(item.get("attempt", 0))
            )

        retry_attempts = len(attempt_history)
        retry_used = retry_attempts > 1

        if accepted and retry_used:
            reason = "accepted_after_retry"

        outcome = derive_outcome(
            accepted=accepted,
            reason=reason,
            deterministic_reasons=deterministic_reasons,
            retry_attempts=retry_attempts,
            max_attempts=max_attempts,
        )
        status = derive_status(accepted, outcome, reason)

        decoder_records = protocol_result.get(
            "decoder_records",
            [],
        )
        if not isinstance(decoder_records, list):
            decoder_records = []

        payload_failures = protocol_result.get(
            "payload_failures",
            [],
        )
        if not isinstance(payload_failures, list):
            payload_failures = []

        corrected_block_count = sum(
            bool(
                record.get("corrected", False)
                or record.get("correction_applied", False)
            )
            for record in decoder_records
            if isinstance(record, Mapping)
        )
        uncorrectable_block_count = sum(
            not bool(record.get("correctable", False))
            for record in decoder_records
            if isinstance(record, Mapping)
        )

        schedule_reason = protocol_result.get("schedule_reason")
        schedule_valid = (
            schedule_reason == "schedule_valid"
            if schedule_reason is not None
            else normalize_optional_boolean(
                protocol_result.get("schedule_valid")
            )
        )

        event_time = current_utc_timestamp()
        resolved_context = str(
            channel.get(
                "context",
                infer_context_from_features(features),
            )
        )

        return cls(
            session_id=(
                generate_session_id()
                if session_id is None
                else normalize_required_string(
                    "session_id",
                    session_id,
                )
            ),
            pseudonym_id=normalize_optional_string(
                request.get(
                    "pseudonym_id",
                    protocol_result.get("pseudonym_id"),
                )
            ),
            request_timestamp=optional_integer(
                request.get("timestamp")
            ),
            request_type=str(
                request.get(
                    "request_type",
                    "FT-QuPAP-Authentication",
                )
            ),
            scenario_name=(
                scenario_name
                or str(channel.get("name", "normal_session"))
            ),
            channel_name=str(channel.get("name", "unknown")),
            context=resolved_context,
            protocol_variant=protocol_variant,
            created_at=event_time,
            completed_at=event_time,
            status=status,
            accepted=accepted,
            outcome=outcome,
            reason=reason,
            deterministic_pass=bool(
                decision.get("deterministic_pass", False)
            ),
            deterministic_reasons=deterministic_reasons,
            schedule_valid=schedule_valid,
            tag_recovered=infer_tag_recovered(protocol_result),
            qber_raw=optional_float(protocol_result.get("qber_raw")),
            qber_mismatches=optional_integer(
                protocol_result.get("qber_mismatches")
            ),
            qber_observed=optional_integer(
                protocol_result.get("qber_observed")
            ),
            observed_check_blocks=optional_integer(
                protocol_result.get("observed_check_blocks")
            ),
            required_check_blocks=int(
                protocol_result.get(
                    "required_check_blocks",
                    DEFAULT_REQUIRED_CHECK_BLOCKS,
                )
            ),
            loss_rate=optional_float(
                protocol_result.get("loss_rate")
            ),
            payload_failure_count=len(payload_failures),
            corrected_block_count=corrected_block_count,
            uncorrectable_block_count=(
                uncorrectable_block_count
            ),
            features=features,
            p_attack=optional_float(decision.get("p_attack")),
            uncertainty=optional_float(
                decision.get("uncertainty")
            ),
            raw_calibration_gp_attack_threshold=optional_float(
                decision.get(
                    "raw_calibration_gp_attack_threshold"
                )
            ),
            gp_attack_threshold=optional_float(
                decision.get("gp_attack_threshold")
            ),
            gp_gray_zone_retry_upper=float(
                decision.get(
                    "gp_gray_zone_retry_upper",
                    DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
                )
            ),
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            max_attempts=max_attempts,
            attempt_history=attempt_history,
            physical_qubits=optional_integer(
                protocol_result.get("physical_qubits")
            ),
            logical_payload_blocks=optional_integer(
                protocol_result.get("logical_payload_blocks")
            ),
            logical_check_blocks=optional_integer(
                protocol_result.get("logical_check_blocks")
            ),
            use_css=bool(protocol_result.get("use_css", True)),
            bootstrap_mode=str(
                protocol_result.get("bootstrap_mode", "mlkem")
            ),
            decision_mode=str(
                protocol_result.get("decision_mode", "gp")
            ),
            timings=timings,
            metadata=dict(metadata or {}),
        )


class SessionDatabase:
    """Thread-safe JSON repository for completed FT-QuPAP sessions."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_SESSION_DATABASE_PATH,
    ) -> None:
        """Initialize or load the session database."""

        self._database_path = Path(database_path)
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionRecord] = {}
        self._initialize_database()

    @property
    def database_path(self) -> Path:
        """Return the persistent JSON file path."""

        return self._database_path

    def _initialize_database(self) -> None:
        """Create the database file or load existing records."""

        with self._lock:
            self._database_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if self._database_path.exists():
                self._load()
            else:
                self._save()

    def _load(self) -> None:
        """Load records and support the older top-level list format."""

        with self._lock:
            try:
                stored_data = json.loads(
                    self._database_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Session database contains invalid JSON."
                ) from error
            except OSError as error:
                raise OSError(
                    "Unable to read the session database."
                ) from error

            loaded: dict[str, SessionRecord] = {}

            if isinstance(stored_data, list):
                for raw_result in stored_data:
                    if not isinstance(raw_result, Mapping):
                        continue
                    record = SessionRecord.from_protocol_result(
                        raw_result
                    )
                    loaded[record.session_id] = record

                self._sessions = loaded
                return

            if not isinstance(stored_data, Mapping):
                raise ValueError(
                    "Session database root must be an object "
                    "or a legacy list."
                )

            version = int(
                stored_data.get("version", DATABASE_VERSION)
            )
            if version != DATABASE_VERSION:
                raise ValueError(
                    "Unsupported session database version: "
                    f"{version}"
                )

            raw_sessions = stored_data.get("sessions", {})
            if not isinstance(raw_sessions, Mapping):
                raise ValueError(
                    "'sessions' must be an object indexed by session_id."
                )

            for session_id, raw_record in raw_sessions.items():
                if not isinstance(raw_record, Mapping):
                    raise ValueError(
                        "Each stored session must be a JSON object."
                    )

                record_data = dict(raw_record)
                record_data.setdefault("session_id", session_id)
                record = SessionRecord.from_dictionary(record_data)

                if record.session_id != session_id:
                    raise ValueError(
                        "Session key does not match record session_id."
                    )

                loaded[session_id] = record

            self._sessions = loaded

    def reload(self) -> None:
        """Reload session records from disk."""

        self._load()

    def _save(self) -> None:
        """Persist records through atomic file replacement."""

        with self._lock:
            content = {
                "version": DATABASE_VERSION,
                "protocol": PROTOCOL_VERSION,
                "sessions": {
                    session_id: record.to_dictionary()
                    for session_id, record in sorted(
                        self._sessions.items(),
                        key=lambda item: (
                            item[1].completed_at,
                            item[0],
                        ),
                    )
                },
            }

            temporary_path: Path | None = None

            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f"{self._database_path.name}.",
                    suffix=".tmp",
                    dir=str(self._database_path.parent),
                )
                temporary_path = Path(temporary_name)

                with os.fdopen(
                    descriptor,
                    "w",
                    encoding="utf-8",
                ) as temporary_file:
                    json.dump(
                        content,
                        temporary_file,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    temporary_file.write("\n")
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())

                os.replace(
                    temporary_path,
                    self._database_path,
                )

            except (OSError, ValueError) as error:
                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    temporary_path.unlink(missing_ok=True)

                raise OSError(
                    "Unable to save the session database."
                ) from error

    def add_session(
        self,
        record: SessionRecord,
        *,
        replace: bool = False,
    ) -> SessionRecord:
        """Store a validated SessionRecord."""

        if not isinstance(record, SessionRecord):
            raise TypeError(
                "record must be a SessionRecord instance."
            )

        with self._lock:
            previous = self._sessions.get(record.session_id)

            if previous is not None and not replace:
                raise ValueError(
                    f"Session already exists: {record.session_id}"
                )

            self._sessions[record.session_id] = copy.deepcopy(
                record
            )

            try:
                self._save()
            except Exception:
                if previous is None:
                    self._sessions.pop(record.session_id, None)
                else:
                    self._sessions[record.session_id] = previous
                raise

            return copy.deepcopy(record)

    def save_protocol_result(
        self,
        protocol_result: Mapping[str, Any],
        *,
        session_id: str | None = None,
        scenario_name: str | None = None,
        protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> SessionRecord:
        """Convert and save a notebook-style protocol result."""

        record = SessionRecord.from_protocol_result(
            protocol_result,
            session_id=session_id,
            scenario_name=scenario_name,
            protocol_variant=protocol_variant,
            max_attempts=max_attempts,
            metadata=metadata,
        )

        return self.add_session(record, replace=replace)

    def get_session(
        self,
        session_id: str,
    ) -> SessionRecord | None:
        """Return one session by identifier."""

        normalized_id = normalize_required_string(
            "session_id",
            session_id,
        )

        with self._lock:
            record = self._sessions.get(normalized_id)
            return copy.deepcopy(record) if record else None

    def require_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """Return a session or raise LookupError."""

        record = self.get_session(session_id)
        if record is None:
            raise LookupError(
                f"Session '{session_id}' was not found."
            )
        return record

    def list_sessions(
        self,
        *,
        status: str | None = None,
        accepted: bool | None = None,
        outcome: str | None = None,
        scenario_name: str | None = None,
        context: str | None = None,
    ) -> list[SessionRecord]:
        """List stored sessions using optional filters."""

        normalized_status = (
            validate_choice(
                "status",
                status,
                SUPPORTED_STATUSES,
            )
            if status is not None
            else None
        )
        normalized_outcome = (
            validate_choice(
                "outcome",
                outcome,
                SUPPORTED_OUTCOMES,
            )
            if outcome is not None
            else None
        )
        normalized_scenario = (
            normalize_required_string(
                "scenario_name",
                scenario_name,
            )
            if scenario_name is not None
            else None
        )
        normalized_context = (
            normalize_context(context)
            if context is not None
            else None
        )

        if accepted is not None:
            validate_boolean("accepted", accepted)

        with self._lock:
            records: Iterable[SessionRecord] = (
                self._sessions.values()
            )

            if normalized_status is not None:
                records = (
                    record
                    for record in records
                    if record.status == normalized_status
                )

            if accepted is not None:
                records = (
                    record
                    for record in records
                    if record.accepted == accepted
                )

            if normalized_outcome is not None:
                records = (
                    record
                    for record in records
                    if record.outcome == normalized_outcome
                )

            if normalized_scenario is not None:
                records = (
                    record
                    for record in records
                    if record.scenario_name == normalized_scenario
                )

            if normalized_context is not None:
                records = (
                    record
                    for record in records
                    if record.context == normalized_context
                )

            result = sorted(
                records,
                key=lambda record: (
                    record.completed_at,
                    record.session_id,
                ),
                reverse=True,
            )

        return [copy.deepcopy(record) for record in result]

    def get_recent_sessions(
        self,
        limit: int = 20,
    ) -> list[SessionRecord]:
        """Return the most recently completed sessions."""

        validate_positive_integer("limit", limit)
        return self.list_sessions()[:limit]

    def count_sessions(
        self,
        *,
        status: str | None = None,
        outcome: str | None = None,
    ) -> int:
        """Return the number of matching records."""

        return len(
            self.list_sessions(
                status=status,
                outcome=outcome,
            )
        )

    def get_summary(self) -> dict[str, Any]:
        """Return dashboard-friendly aggregate statistics."""

        records = self.list_sessions()
        total = len(records)

        accepted = sum(record.accepted for record in records)
        rejected = sum(record.status == "rejected" for record in records)
        failed = sum(record.status == "failed" for record in records)
        retry_used = sum(record.retry_used for record in records)
        accepted_after_retry = sum(
            record.outcome == "accepted_after_retry"
            for record in records
        )

        qber_values = [
            record.qber_raw
            for record in records
            if record.qber_raw is not None
        ]
        attack_probabilities = [
            record.p_attack
            for record in records
            if record.p_attack is not None
        ]
        runtime_values = [
            record.timings.get(
                "total_retry_end_to_end_s",
                record.timings.get("end_to_end_s"),
            )
            for record in records
        ]
        runtime_values = [
            value for value in runtime_values if value is not None
        ]

        return {
            "total_sessions": total,
            "accepted_sessions": accepted,
            "rejected_sessions": rejected,
            "failed_sessions": failed,
            "retry_used_sessions": retry_used,
            "accepted_after_retry": accepted_after_retry,
            "acceptance_rate": accepted / total if total else 0.0,
            "average_qber_raw": safe_mean(qber_values),
            "average_attack_probability": safe_mean(
                attack_probabilities
            ),
            "average_end_to_end_s": safe_mean(runtime_values),
        }

    def delete_session(
        self,
        session_id: str,
    ) -> bool:
        """Delete one stored session."""

        normalized_id = normalize_required_string(
            "session_id",
            session_id,
        )

        with self._lock:
            record = self._sessions.pop(normalized_id, None)
            if record is None:
                return False

            try:
                self._save()
            except Exception:
                self._sessions[normalized_id] = record
                raise

            return True

    def clear(self) -> None:
        """Remove all sessions for a controlled demonstration reset."""

        with self._lock:
            previous = copy.deepcopy(self._sessions)
            self._sessions.clear()

            try:
                self._save()
            except Exception:
                self._sessions = previous
                raise

    def __len__(self) -> int:
        """Return the number of stored sessions."""

        with self._lock:
            return len(self._sessions)


def append_session(
    result: Mapping[str, Any],
    *,
    database_path: str | Path = DEFAULT_SESSION_DATABASE_PATH,
    session_id: str | None = None,
    scenario_name: str | None = None,
) -> SessionRecord:
    """
    Append one session using the simple API expected by the dashboard.
    """

    database = SessionDatabase(database_path)
    return database.save_protocol_result(
        result,
        session_id=session_id,
        scenario_name=scenario_name,
    )


def load_sessions(
    *,
    database_path: str | Path = DEFAULT_SESSION_DATABASE_PATH,
) -> list[dict[str, Any]]:
    """Load all stored sessions as JSON-compatible dictionaries."""

    database = SessionDatabase(database_path)
    return [
        record.to_dictionary()
        for record in database.list_sessions()
    ]


def extract_decision(
    protocol_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return nested or compact-summary decision data."""

    decision = safe_mapping(protocol_result.get("decision"))
    if decision:
        return decision

    return {
        "accepted": protocol_result.get("accepted", False),
        "reason": protocol_result.get("reason", "unspecified"),
        "deterministic_pass": protocol_result.get(
            "deterministic_pass",
            False,
        ),
        "deterministic_reasons": protocol_result.get(
            "deterministic_reasons",
            [],
        ),
        "p_attack": protocol_result.get("p_attack"),
        "uncertainty": protocol_result.get("uncertainty"),
        "raw_calibration_gp_attack_threshold": protocol_result.get(
            "raw_calibration_gp_attack_threshold"
        ),
        "gp_attack_threshold": protocol_result.get(
            "gp_attack_threshold"
        ),
        "gp_gray_zone_retry_upper": protocol_result.get(
            "gp_gray_zone_retry_upper",
            DEFAULT_GP_GRAY_ZONE_RETRY_UPPER,
        ),
    }


def build_attempt_summary(
    protocol_result: Mapping[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    """Build the notebook-compatible compact final-attempt summary."""

    decision = extract_decision(protocol_result)

    return {
        "attempt": int(attempt),
        "accepted": bool(decision.get("accepted", False)),
        "reason": str(decision.get("reason", "unspecified")),
        "deterministic_pass": bool(
            decision.get("deterministic_pass", False)
        ),
        "deterministic_reasons": normalize_reason_list(
            list(decision.get("deterministic_reasons", []))
        ),
        "qber_raw": optional_float(
            protocol_result.get("qber_raw")
        ),
        "qber_mismatches": optional_integer(
            protocol_result.get("qber_mismatches")
        ),
        "qber_observed": optional_integer(
            protocol_result.get("qber_observed")
        ),
        "observed_check_blocks": optional_integer(
            protocol_result.get("observed_check_blocks")
        ),
        "loss_rate": optional_float(
            protocol_result.get("loss_rate")
        ),
        "p_attack": optional_float(decision.get("p_attack")),
        "uncertainty": optional_float(
            decision.get("uncertainty")
        ),
        "gp_attack_threshold": optional_float(
            decision.get("gp_attack_threshold")
        ),
        "tag_recovered": infer_tag_recovered(protocol_result),
        "retryable": normalize_optional_boolean(
            protocol_result.get("retryable")
        ),
        "timing_end_to_end_s": optional_float(
            safe_mapping(protocol_result.get("timings")).get(
                "end_to_end_s"
            )
        ),
    }


def normalize_attempt_history(
    history: Any,
) -> list[dict[str, Any]]:
    """Retain only compact, non-secret per-attempt evidence."""

    if history is None:
        return []

    if not isinstance(history, list):
        raise TypeError("attempt_history must be a list.")

    normalized_history: list[dict[str, Any]] = []

    for index, raw_attempt in enumerate(history, start=1):
        if not isinstance(raw_attempt, Mapping):
            continue

        attempt_number = optional_integer(
            raw_attempt.get("attempt")
        )
        if attempt_number is None or attempt_number < 1:
            attempt_number = index

        normalized_attempt = {
            "attempt": attempt_number,
            "accepted": bool(
                raw_attempt.get("accepted", False)
            ),
            "reason": str(
                raw_attempt.get("reason", "unspecified")
            ),
            "deterministic_pass": bool(
                raw_attempt.get(
                    "deterministic_pass",
                    not bool(
                        raw_attempt.get(
                            "deterministic_reasons",
                            [],
                        )
                    ),
                )
            ),
            "deterministic_reasons": normalize_reason_list(
                list(
                    raw_attempt.get(
                        "deterministic_reasons",
                        [],
                    )
                )
            ),
            "qber_raw": optional_float(
                raw_attempt.get("qber_raw")
            ),
            "qber_mismatches": optional_integer(
                raw_attempt.get("qber_mismatches")
            ),
            "qber_observed": optional_integer(
                raw_attempt.get("qber_observed")
            ),
            "observed_check_blocks": optional_integer(
                raw_attempt.get("observed_check_blocks")
            ),
            "loss_rate": optional_float(
                raw_attempt.get("loss_rate")
            ),
            "p_attack": optional_float(
                raw_attempt.get("p_attack")
            ),
            "uncertainty": optional_float(
                raw_attempt.get("uncertainty")
            ),
            "gp_attack_threshold": optional_float(
                raw_attempt.get("gp_attack_threshold")
            ),
            "tag_recovered": normalize_optional_boolean(
                raw_attempt.get("tag_recovered")
            ),
            "retryable": normalize_optional_boolean(
                raw_attempt.get("retryable")
            ),
            "timing_end_to_end_s": optional_float(
                raw_attempt.get("timing_end_to_end_s")
            ),
        }

        normalized_history.append(
            {
                key: value
                for key, value in normalized_attempt.items()
                if key in SAFE_ATTEMPT_FIELDS
            }
        )

    normalized_history.sort(
        key=lambda item: int(item["attempt"])
    )

    attempt_numbers = [
        int(item["attempt"])
        for item in normalized_history
    ]
    if len(attempt_numbers) != len(set(attempt_numbers)):
        raise ValueError(
            "attempt_history contains duplicate attempt numbers."
        )

    return normalized_history


def infer_tag_recovered(
    protocol_result: Mapping[str, Any],
) -> bool | None:
    """Compare raw tags in memory and store only the boolean result."""

    direct_value = protocol_result.get("tag_recovered")
    if isinstance(direct_value, bool):
        return direct_value

    received_tag = protocol_result.get("received_tag")
    expected_tag = protocol_result.get("expected_tag")

    if not isinstance(received_tag, (bytes, bytearray)):
        return None

    if not isinstance(expected_tag, (bytes, bytearray)):
        return None

    return hmac.compare_digest(
        bytes(received_tag),
        bytes(expected_tag),
    )


def derive_outcome(
    *,
    accepted: bool,
    reason: str,
    deterministic_reasons: list[str],
    retry_attempts: int,
    max_attempts: int,
) -> str:
    """Map notebook reasons to a stable storage outcome."""

    normalized_reason = str(reason).strip().lower()
    reasons = {
        normalized_reason,
        *(
            str(value).strip().lower()
            for value in deterministic_reasons
        ),
    }

    if accepted:
        return (
            "accepted_after_retry"
            if retry_attempts > 1
            or normalized_reason == "accepted_after_retry"
            else "accepted"
        )

    if any(
        "replay" in value or "nonce" in value
        for value in reasons
    ):
        return "rejected_replay"

    if any(
        "credential" in value
        or "signature" in value
        or "trust_anchor" in value
        for value in reasons
    ):
        return "rejected_credential"

    if any(
        "ciphertext" in value
        or "decapsulation" in value
        or "session_secret" in value
        for value in reasons
    ):
        return "rejected_ciphertext"

    if (
        normalized_reason
        == "rejected_by_calibrated_bayesian_policy"
        or "gp" in normalized_reason
    ):
        return "rejected_gp"

    if (
        retry_attempts >= max_attempts
        and any(
            value in {
                "payload_block_unrecoverable",
                "authentication_tag_mismatch",
            }
            for value in reasons
        )
    ):
        return "rejected_retry_exhausted"

    if deterministic_reasons:
        return "rejected_deterministic"

    if "abort" in normalized_reason:
        return "aborted"

    if "fail" in normalized_reason or "error" in normalized_reason:
        return "failed"

    return "rejected"


def derive_status(
    accepted: bool,
    outcome: str,
    reason: str,
) -> str:
    """Derive accepted, rejected, failed, or aborted status."""

    if accepted:
        return "accepted"
    if outcome == "aborted":
        return "aborted"
    if outcome == "failed" or "exception" in reason.lower():
        return "failed"
    return "rejected"


def infer_context_from_features(
    features: Mapping[str, Any],
) -> str:
    """Infer context from the three one-hot GP features."""

    scores = {
        "urban": optional_float(
            features.get("ctx_urban")
        ) or 0.0,
        "suburban": optional_float(
            features.get("ctx_suburban")
        ) or 0.0,
        "rural": optional_float(
            features.get("ctx_rural")
        ) or 0.0,
    }

    selected = max(scores, key=scores.get)
    return selected if scores[selected] > 0.0 else "unknown"


def normalize_features(
    features: Mapping[str, Any],
) -> dict[str, float]:
    """Retain only notebook-defined observable GP features."""

    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping.")

    normalized: dict[str, float] = {}

    for feature_name in SAFE_FEATURE_NAMES:
        value = optional_float(features.get(feature_name))
        if value is not None:
            normalized[feature_name] = value

    return normalized


def normalize_timings(
    timings: Mapping[str, Any],
) -> dict[str, float]:
    """Retain only finite, nonnegative notebook timing fields."""

    if not isinstance(timings, Mapping):
        raise TypeError("timings must be a mapping.")

    normalized: dict[str, float] = {}

    for timing_name in SAFE_TIMING_NAMES:
        value = optional_float(timings.get(timing_name))
        if value is None:
            continue
        if value < 0.0:
            raise ValueError(
                f"{timing_name} cannot be negative."
            )
        normalized[timing_name] = value

    return normalized


def validate_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate optional non-secret session metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")

    copied = copy.deepcopy(dict(metadata))
    inspect_metadata(copied)

    try:
        json.dumps(copied, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "metadata must contain finite JSON-compatible values."
        ) from error

    return copied


def inspect_metadata(
    value: Any,
    path: str = "metadata",
) -> None:
    """Reject sensitive names, binary values, and non-finite numbers."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = normalize_field_name(str(key))

            if normalized_key in PROHIBITED_METADATA_FIELDS:
                raise ValueError(
                    "Sensitive field cannot be stored in "
                    f"session metadata: {path}.{key}"
                )

            inspect_metadata(
                nested_value,
                f"{path}.{key}",
            )

    elif isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            inspect_metadata(
                nested_value,
                f"{path}[{index}]",
            )

    elif isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"Binary values cannot be stored in {path}."
        )

    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"Non-finite numeric value found in {path}."
        )


def safe_mapping(value: Any) -> dict[str, Any]:
    """Return a plain dictionary when value is mapping-like."""

    return dict(value) if isinstance(value, Mapping) else {}


def generate_session_id() -> str:
    """Generate a unique session identifier."""

    return f"SESSION-{uuid.uuid4().hex.upper()}"


def normalize_field_name(value: str) -> str:
    """Normalize a name for sensitive-field matching."""

    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    )

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def normalize_required_string(
    name: str,
    value: str,
) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty.")

    return normalized


def normalize_optional_string(value: Any) -> str | None:
    """Normalize an optional string."""

    if value is None:
        return None

    normalized = str(value).strip()
    return normalized or None


def normalize_reason_list(values: list[Any]) -> list[str]:
    """Normalize and deduplicate deterministic reason values."""

    if not isinstance(values, list):
        raise TypeError("deterministic_reasons must be a list.")

    normalized = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    return list(dict.fromkeys(normalized))


def normalize_context(value: str) -> str:
    """Normalize urban, suburban, rural, or unknown context."""

    normalized = normalize_required_string(
        "context",
        value,
    ).lower()

    return (
        normalized
        if normalized in SUPPORTED_CONTEXTS
        else "unknown"
    )


def validate_choice(
    name: str,
    value: str,
    supported_values: set[str],
) -> str:
    """Validate a normalized string against a supported set."""

    normalized = normalize_required_string(name, value).lower()

    if normalized not in supported_values:
        raise ValueError(
            f"Unsupported {name}: {normalized}"
        )

    return normalized


def validate_boolean(name: str, value: bool) -> None:
    """Validate a required boolean."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def normalize_optional_boolean(value: Any) -> bool | None:
    """Normalize an optional boolean value."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False

    raise TypeError("Optional boolean value is invalid.")


def validate_nonnegative_integer(
    name: str,
    value: int,
) -> None:
    """Validate a nonnegative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def validate_positive_integer(
    name: str,
    value: int,
) -> None:
    """Validate an integer greater than zero."""

    validate_nonnegative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be greater than zero.")


def optional_integer(value: Any) -> int | None:
    """Convert an integer-like finite value to int or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric) or not numeric.is_integer():
        return None

    return int(numeric)


def optional_float(value: Any) -> float | None:
    """Convert a finite numeric value to float or None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    return numeric if math.isfinite(numeric) else None


def validate_optional_probability(
    name: str,
    value: float | None,
) -> None:
    """Validate an optional probability in [0, 1]."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0."
        )


def normalize_timestamp(
    value: datetime | str,
) -> str:
    """Convert a datetime or ISO string into UTC ISO format."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("Timestamp cannot be empty.")

        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"

        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError(
                "Timestamp must use ISO 8601 format."
            ) from error
    else:
        raise TypeError(
            "Timestamp must be a datetime or ISO string."
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat()


def safe_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean or zero for an empty sequence."""

    finite_values = [
        float(value)
        for value in values
        if optional_float(value) is not None
    ]

    return (
        sum(finite_values) / len(finite_values)
        if finite_values
        else 0.0
    )


def run_self_test() -> None:
    """Run persistence, retry, and secret-exclusion tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory) / "demo_sessions.json"
        )
        database = SessionDatabase(database_path)

        expected_tag = bytes.fromhex(
            "00112233445566778899aabbccddeeff"
        )

        result = {
            "request": {
                "pseudonym_id": "PID-6G-UE-0001",
                "timestamp": 1_800_000_000,
                "nonce": "MUST-NOT-BE-STORED",
                "request_type": "FT-QuPAP-Authentication",
            },
            "decision": {
                "accepted": True,
                "reason": "accepted_after_retry",
                "deterministic_pass": True,
                "deterministic_reasons": [],
                "p_attack": 0.04,
                "uncertainty": 0.24,
                "gp_attack_threshold": 0.15,
            },
            "schedule_reason": "schedule_valid",
            "received_tag": expected_tag,
            "expected_tag": expected_tag,
            "shared_secret": b"MUST-NOT-BE-STORED",
            "k_auth": b"MUST-NOT-BE-STORED",
            "k_ctrl": b"MUST-NOT-BE-STORED",
            "qber_raw": 0.015625,
            "qber_mismatches": 7,
            "qber_observed": 224,
            "observed_check_blocks": 32,
            "required_check_blocks": 24,
            "loss_rate": 0.01,
            "payload_failures": [],
            "features": {
                "qber_raw": 0.015625,
                "mean_syndrome_weight": 0.08,
                "max_syndrome_weight": 1.0,
                "correction_failure_rate": 0.0,
                "loss_rate": 0.01,
                "noise_estimate": 0.012,
                "ctx_urban": 1.0,
                "ctx_suburban": 0.0,
                "ctx_rural": 0.0,
                "eve_fraction": 1.0,
            },
            "retry_attempts": 2,
            "retry_used": True,
            "attempt_history": [
                {
                    "attempt": 1,
                    "accepted": False,
                    "reason": "authentication_tag_mismatch",
                    "deterministic_reasons": [
                        "authentication_tag_mismatch"
                    ],
                    "qber_raw": 0.02,
                    "p_attack": 0.08,
                    "uncertainty": 0.40,
                    "tag_recovered": False,
                },
                {
                    "attempt": 2,
                    "accepted": True,
                    "reason": "accepted_after_retry",
                    "deterministic_reasons": [],
                    "qber_raw": 0.015625,
                    "p_attack": 0.04,
                    "uncertainty": 0.24,
                    "tag_recovered": True,
                },
            ],
            "physical_qubits": 1120,
            "logical_payload_blocks": 128,
            "logical_check_blocks": 32,
            "use_css": True,
            "bootstrap_mode": "mlkem",
            "decision_mode": "gp",
            "channel": {
                "name": "benign_noisy_session",
                "context": "urban",
                "eve_fraction": 0.0,
            },
            "timings": {
                "end_to_end_s": 0.06,
                "total_retry_end_to_end_s": 0.12,
            },
        }

        stored = database.save_protocol_result(
            result,
            session_id="SESSION-0001",
            scenario_name="accept_after_retry",
        )

        assert stored.status == "accepted"
        assert stored.outcome == "accepted_after_retry"
        assert stored.retry_attempts == 2
        assert stored.retry_used is True
        assert stored.tag_recovered is True
        assert stored.physical_qubits == 1120
        assert "eve_fraction" not in stored.features

        reloaded = SessionDatabase(database_path)
        loaded = reloaded.require_session("SESSION-0001")

        assert loaded.outcome == "accepted_after_retry"
        assert len(loaded.attempt_history) == 2
        assert reloaded.get_summary()["accepted_sessions"] == 1

        stored_text = database_path.read_text(encoding="utf-8")

        assert "MUST-NOT-BE-STORED" not in stored_text
        assert expected_tag.hex() not in stored_text
        assert "shared_secret" not in stored_text
        assert "k_auth" not in stored_text
        assert "k_ctrl" not in stored_text
        assert "eve_fraction" not in stored_text

        print("Session database self-test passed.")
        print(f"Database path: {database_path}")
        print(f"Stored outcome: {loaded.outcome}")
        print(f"Retry attempts: {loaded.retry_attempts}")


if __name__ == "__main__":
    run_self_test()
