"""
FT-QuPAP Protocol Logger

This module stores non-secret authentication evidence for:

- dashboard monitoring
- session-history display
- protocol evaluation
- rejection-reason analysis
- GP decision analysis
- optional JSON/JSONL export

Security requirements:

Never log:

- ML-KEM secret keys
- ML-DSA secret keys
- ML-KEM shared secrets
- K_auth
- K_ctrl
- raw KMAC tags
- raw subscriber identity
- encrypted control-schedule keys
- hidden Eve simulator settings
- attacked physical-qubit masks

Notebook-compatible audit fields:

    time
    pseudonym_id
    accepted
    reason
    deterministic_pass
    deterministic_reasons
    p_attack
    uncertainty
    gp_attack_threshold
    features
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from .protocol_state import ProtocolState
from .result_models import (
    AuthenticationResult,
    DecisionResult,
    to_json_safe,
)


AUDIT_LOG: list[dict[str, Any]] = []

_AUDIT_LOCK = RLock()


SENSITIVE_LOG_KEYS = frozenset(
    {
        "secret",
        "secret_key",
        "private_key",
        "shared_secret",
        "session_key",
        "session_keys",
        "mlkem_secret_key",
        "ml_kem_secret_key",
        "mldsa_secret_key",
        "ml_dsa_secret_key",
        "kem_secret_key",
        "signing_secret_key",
        "k_ss",
        "k_auth",
        "k_ctrl",
        "raw_tag",
        "kmac_tag",
        "authentication_tag",
        "expected_tag",
        "received_tag",
        "recovered_tag",
        "control_key",
        "encryption_key",
        "decryption_key",
        "raw_subscriber_identity",
        "subscriber_identity",
        "imsi",
        "attacked_mask",
        "eve_positions",
        "eve_basis",
    }
)


HIDDEN_SIMULATOR_KEYS = frozenset(
    {
        "eve_fraction",
        "eve_mode",
        "actual_attack",
        "label_attack",
        "scenario_severity",
        "attack_positions",
        "intercepted_positions",
        "hidden_attack_setting",
    }
)


class ProtocolLoggerError(Exception):
    """Base exception for protocol-logging failures."""


class InvalidAuditRequestError(ProtocolLoggerError):
    """Raised when an authentication request cannot be interpreted."""


class InvalidAuditDecisionError(ProtocolLoggerError):
    """Raised when a decision record is malformed."""


class InvalidAuditFeatureError(ProtocolLoggerError):
    """Raised when observable feature evidence is malformed."""


class AuditExportError(ProtocolLoggerError):
    """Raised when an audit-log export fails."""


def current_timestamp() -> int:
    """
    Return the current Unix timestamp.

    This reproduces the notebook's current_timestamp() behavior.
    """

    return int(time.time())


def normalize_field_name(
    value: Any,
) -> str:
    """Normalize a mapping key for security checks."""

    return str(value).strip().lower()


def is_sensitive_field(
    field_name: Any,
) -> bool:
    """
    Return whether a field name represents secret material.
    """

    normalized = normalize_field_name(
        field_name
    )

    if normalized in SENSITIVE_LOG_KEYS:
        return True

    sensitive_fragments = (
        "secret_key",
        "private_key",
        "shared_secret",
        "session_key",
        "raw_tag",
        "attacked_mask",
    )

    return any(
        fragment in normalized
        for fragment in sensitive_fragments
    )


def is_hidden_simulator_field(
    field_name: Any,
) -> bool:
    """
    Return whether a field contains simulator-only attack knowledge.

    Hidden Eve values must not be treated as Authentication Server
    observations or GP features.
    """

    normalized = normalize_field_name(
        field_name
    )

    return normalized in HIDDEN_SIMULATOR_KEYS


def normalize_mapping(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> dict[str, Any] | None:
    """
    Convert a mapping-like protocol object to a detached dictionary.

    Supported inputs:

    - Mapping
    - object implementing as_dict()
    - object implementing to_dictionary()
    """

    if value is None:
        if allow_none:
            return None

        raise TypeError(
            f"{field_name} cannot be None."
        )

    if isinstance(value, Mapping):
        result = dict(value)

    elif hasattr(value, "as_dict"):
        result = value.as_dict()

    elif hasattr(value, "to_dictionary"):
        result = value.to_dictionary()

    else:
        raise TypeError(
            f"{field_name} must be a mapping or provide "
            "as_dict()/to_dictionary()."
        )

    if not isinstance(result, Mapping):
        raise TypeError(
            f"{field_name} conversion must return a mapping."
        )

    detached: dict[str, Any] = {}

    for key, item in result.items():
        if not isinstance(key, str):
            raise TypeError(
                f"Every {field_name} key must be a string."
            )

        detached[key] = copy.deepcopy(
            item
        )

    return detached


def extract_pseudonym_id(
    request: Any | None,
) -> str | None:
    """
    Extract only the pseudonymous subscriber reference.

    The request nonce, timestamp, raw identity, and other request fields
    are intentionally not copied into the audit record.
    """

    if request is None:
        return None

    normalized_request = normalize_mapping(
        request,
        "request",
    )

    if normalized_request is None:
        return None

    pseudonym_id = normalized_request.get(
        "pseudonym_id"
    )

    if pseudonym_id is None:
        return None

    if not isinstance(
        pseudonym_id,
        str,
    ):
        raise InvalidAuditRequestError(
            "request pseudonym_id must be a string."
        )

    pseudonym_id = pseudonym_id.strip()

    return pseudonym_id or None


def normalize_reason_list(
    reasons: Any,
) -> list[str]:
    """
    Validate and normalize deterministic rejection reasons.
    """

    if reasons is None:
        return []

    if isinstance(reasons, str):
        reasons = [
            reasons
        ]

    if not isinstance(
        reasons,
        (
            list,
            tuple,
            set,
        ),
    ):
        raise InvalidAuditDecisionError(
            "deterministic_reasons must be a sequence."
        )

    normalized: list[str] = []

    for reason in reasons:
        if not isinstance(reason, str):
            raise InvalidAuditDecisionError(
                "Every deterministic reason must be a string."
            )

        reason = reason.strip()

        if reason and reason not in normalized:
            normalized.append(
                reason
            )

    return normalized


def normalize_optional_probability(
    value: Any,
    field_name: str,
) -> float | None:
    """
    Validate an optional finite value in the interval [0, 1].
    """

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise InvalidAuditDecisionError(
            f"{field_name} must be numeric or None."
        )

    normalized = float(
        value
    )

    if not math.isfinite(
        normalized
    ):
        raise InvalidAuditDecisionError(
            f"{field_name} must be finite."
        )

    if not 0.0 <= normalized <= 1.0:
        raise InvalidAuditDecisionError(
            f"{field_name} must be between 0 and 1."
        )

    return normalized


def normalize_decision(
    decision: (
        DecisionResult
        | AuthenticationResult
        | Mapping[str, Any]
    ),
) -> dict[str, Any]:
    """
    Convert a supported decision object into the audit format.
    """

    if isinstance(
        decision,
        AuthenticationResult,
    ):
        raw_decision = (
            decision.decision.as_dict()
        )

    elif isinstance(
        decision,
        DecisionResult,
    ):
        raw_decision = (
            decision.as_dict()
        )

    elif isinstance(
        decision,
        Mapping,
    ):
        raw_decision = dict(
            decision
        )

    else:
        raise TypeError(
            "decision must be DecisionResult, "
            "AuthenticationResult, or Mapping."
        )

    accepted = raw_decision.get(
        "accepted"
    )

    if accepted is not None and not isinstance(
        accepted,
        bool,
    ):
        raise InvalidAuditDecisionError(
            "accepted must be boolean or None."
        )

    reason = raw_decision.get(
        "reason"
    )

    if reason is not None:
        if not isinstance(reason, str):
            raise InvalidAuditDecisionError(
                "reason must be a string or None."
            )

        reason = reason.strip() or None

    deterministic_reasons = (
        normalize_reason_list(
            raw_decision.get(
                "deterministic_reasons",
                [],
            )
        )
    )

    deterministic_pass = raw_decision.get(
        "deterministic_pass"
    )

    if deterministic_pass is None:
        deterministic_pass = (
            len(deterministic_reasons) == 0
        )

    if not isinstance(
        deterministic_pass,
        bool,
    ):
        raise InvalidAuditDecisionError(
            "deterministic_pass must be boolean."
        )

    if (
        deterministic_pass
        and deterministic_reasons
    ):
        raise InvalidAuditDecisionError(
            "deterministic_pass=True cannot contain "
            "deterministic failure reasons."
        )

    if (
        accepted is True
        and not deterministic_pass
    ):
        raise InvalidAuditDecisionError(
            "An accepted session cannot have failed "
            "deterministic verification."
        )

    return {
        "accepted":
            accepted,

        "reason":
            reason,

        "deterministic_pass":
            deterministic_pass,

        "deterministic_reasons":
            deterministic_reasons,

        "p_attack":
            normalize_optional_probability(
                raw_decision.get(
                    "p_attack"
                ),
                "p_attack",
            ),

        "uncertainty":
            normalize_optional_probability(
                raw_decision.get(
                    "uncertainty"
                ),
                "uncertainty",
            ),

        "gp_attack_threshold":
            normalize_optional_probability(
                raw_decision.get(
                    "gp_attack_threshold"
                ),
                "gp_attack_threshold",
            ),
    }


def sanitize_feature_value(
    value: Any,
    field_name: str,
) -> float:
    """
    Validate one numeric receiver-observable GP feature.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise InvalidAuditFeatureError(
            f"Feature {field_name!r} must be numeric."
        )

    normalized = float(
        value
    )

    if not math.isfinite(
        normalized
    ):
        raise InvalidAuditFeatureError(
            f"Feature {field_name!r} must be finite."
        )

    return normalized


def sanitize_features(
    features: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    """
    Validate non-secret receiver-observable feature evidence.

    Sensitive and hidden simulator-only fields are removed. Remaining
    values must be finite numbers.
    """

    if features is None:
        return None

    if not isinstance(
        features,
        Mapping,
    ):
        raise TypeError(
            "features must be a mapping or None."
        )

    safe_features: dict[str, float] = {}

    for raw_key, raw_value in features.items():
        if not isinstance(raw_key, str):
            raise InvalidAuditFeatureError(
                "Every feature name must be a string."
            )

        key = raw_key.strip()

        if not key:
            raise InvalidAuditFeatureError(
                "Feature names cannot be empty."
            )

        if (
            is_sensitive_field(key)
            or is_hidden_simulator_field(key)
        ):
            continue

        safe_features[key] = (
            sanitize_feature_value(
                raw_value,
                key,
            )
        )

    return safe_features


def build_audit_record(
    request: Any | None,
    decision: (
        DecisionResult
        | AuthenticationResult
        | Mapping[str, Any]
    ),
    features: Mapping[str, Any] | None = None,
    *,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """
    Build one notebook-compatible non-secret audit record.
    """

    normalized_decision = normalize_decision(
        decision
    )

    if timestamp is None:
        record_time = current_timestamp()

    else:
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
        ):
            raise TypeError(
                "timestamp must be an integer or None."
            )

        if timestamp < 0:
            raise ValueError(
                "timestamp cannot be negative."
            )

        record_time = timestamp

    return {
        "time":
            record_time,

        "pseudonym_id":
            extract_pseudonym_id(
                request
            ),

        "accepted":
            normalized_decision[
                "accepted"
            ],

        "reason":
            normalized_decision[
                "reason"
            ],

        "deterministic_pass":
            normalized_decision[
                "deterministic_pass"
            ],

        "deterministic_reasons":
            list(
                normalized_decision[
                    "deterministic_reasons"
                ]
            ),

        "p_attack":
            normalized_decision[
                "p_attack"
            ],

        "uncertainty":
            normalized_decision[
                "uncertainty"
            ],

        "gp_attack_threshold":
            normalized_decision[
                "gp_attack_threshold"
            ],

        "features":
            sanitize_features(
                features
            ),
    }


def append_audit_record(
    request: Any | None,
    decision: (
        DecisionResult
        | AuthenticationResult
        | Mapping[str, Any]
    ),
    features: Mapping[str, Any] | None = None,
) -> None:
    """
    Append non-secret diagnostic evidence to the audit log.

    This preserves the notebook-compatible function signature and
    returns None.
    """

    record = build_audit_record(
        request=request,
        decision=decision,
        features=features,
    )

    with _AUDIT_LOCK:
        AUDIT_LOG.append(
            record
        )


def append_authentication_result(
    result: AuthenticationResult,
) -> None:
    """
    Append an AuthenticationResult using only safe result fields.
    """

    if not isinstance(
        result,
        AuthenticationResult,
    ):
        raise TypeError(
            "result must be AuthenticationResult."
        )

    append_audit_record(
        request={
            "pseudonym_id":
                result.pseudonym_id,
        },
        decision=result.decision,
        features=result.features,
    )


def get_audit_log(
    *,
    newest_first: bool = False,
) -> list[dict[str, Any]]:
    """
    Return a detached snapshot of the in-memory audit log.
    """

    if not isinstance(
        newest_first,
        bool,
    ):
        raise TypeError(
            "newest_first must be boolean."
        )

    with _AUDIT_LOCK:
        snapshot = copy.deepcopy(
            AUDIT_LOG
        )

    if newest_first:
        snapshot.reverse()

    return snapshot


def get_latest_audit_record() -> dict[str, Any] | None:
    """
    Return the newest detached audit record.
    """

    with _AUDIT_LOCK:
        if not AUDIT_LOG:
            return None

        return copy.deepcopy(
            AUDIT_LOG[-1]
        )


def clear_audit_log() -> None:
    """
    Remove every in-memory audit record.
    """

    with _AUDIT_LOCK:
        AUDIT_LOG.clear()


def audit_record_count() -> int:
    """
    Return the number of stored audit records.
    """

    with _AUDIT_LOCK:
        return len(
            AUDIT_LOG
        )


def export_audit_log(
    destination: str | Path,
) -> Path:
    """
    Export the audit log as a JSON document.

    The storage package may later move records into its own repository,
    but this helper supports direct protocol-log export.
    """

    path = Path(
        destination
    )

    if path.suffix.lower() != ".json":
        raise AuditExportError(
            "Audit-log export path must end with .json."
        )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                to_json_safe(
                    get_audit_log()
                ),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise AuditExportError(
            f"Could not export audit log to {path}."
        ) from error

    return path


def append_jsonl_record(
    destination: str | Path,
    record: Mapping[str, Any],
) -> Path:
    """
    Append one safe audit record to a JSON Lines log file.
    """

    if not isinstance(
        record,
        Mapping,
    ):
        raise TypeError(
            "record must be a mapping."
        )

    path = Path(
        destination
    )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = json.dumps(
            to_json_safe(
                dict(record)
            ),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        with path.open(
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                serialized + "\n"
            )

    except (
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise AuditExportError(
            f"Could not append protocol log to {path}."
        ) from error

    return path


class FTQuPAPProtocolLogger:
    """
    ProtocolEngine-compatible non-secret result recorder.

    The class can be passed as protocol_engine.result_recorder because
    it implements:

        logger(result, state)
    """

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
    ) -> None:
        self.jsonl_path = (
            None
            if jsonl_path is None
            else Path(jsonl_path)
        )

    def record(
        self,
        result: AuthenticationResult,
        state: ProtocolState | None = None,
    ) -> dict[str, Any]:
        """
        Store one complete authentication result safely.
        """

        if not isinstance(
            result,
            AuthenticationResult,
        ):
            raise TypeError(
                "result must be AuthenticationResult."
            )

        if (
            state is not None
            and not isinstance(
                state,
                ProtocolState,
            )
        ):
            raise TypeError(
                "state must be ProtocolState or None."
            )

        record = build_audit_record(
            request={
                "pseudonym_id":
                    result.pseudonym_id,
            },
            decision=result.decision,
            features=result.features,
        )

        with _AUDIT_LOCK:
            AUDIT_LOG.append(
                record
            )

        if self.jsonl_path is not None:
            append_jsonl_record(
                self.jsonl_path,
                record,
            )

        return copy.deepcopy(
            record
        )

    def __call__(
        self,
        result: AuthenticationResult,
        state: ProtocolState,
    ) -> None:
        """
        ProtocolEngine result-recorder interface.
        """

        self.record(
            result=result,
            state=state,
        )


def run_self_test() -> None:
    """
    Verify notebook compatibility and secret-data filtering.
    """

    clear_audit_log()

    request = {
        "pseudonym_id":
            "PID-6G-UE-0001",

        "timestamp":
            1785578400,

        "nonce":
            "THIS-MUST-NOT-BE-LOGGED",

        "raw_subscriber_identity":
            "RAW-IMSI-MUST-NOT-BE-LOGGED",
    }

    decision = {
        "accepted":
            True,

        "reason":
            "accepted_by_calibrated_bayesian_policy",

        "deterministic_pass":
            True,

        "deterministic_reasons":
            [],

        "p_attack":
            0.05,

        "uncertainty":
            0.10,

        "gp_attack_threshold":
            0.15,

        "k_auth":
            b"SECRET-MUST-NOT-BE-LOGGED",
    }

    features = {
        "qber_raw":
            0.01,

        "mean_syndrome_weight":
            0.05,

        "max_syndrome_weight":
            1.0,

        "correction_failure_rate":
            0.0,

        "loss_rate":
            0.0,

        "noise_estimate":
            0.01,

        "ctx_urban":
            1.0,

        "ctx_suburban":
            0.0,

        "ctx_rural":
            0.0,

        # Hidden simulator evidence must be discarded.
        "eve_fraction":
            0.50,
    }

    append_audit_record(
        request=request,
        decision=decision,
        features=features,
    )

    if audit_record_count() != 1:
        raise ProtocolLoggerError(
            "Audit record was not appended."
        )

    record = get_latest_audit_record()

    if record is None:
        raise ProtocolLoggerError(
            "Latest audit record is missing."
        )

    expected_keys = {
        "time",
        "pseudonym_id",
        "accepted",
        "reason",
        "deterministic_pass",
        "deterministic_reasons",
        "p_attack",
        "uncertainty",
        "gp_attack_threshold",
        "features",
    }

    if set(record) != expected_keys:
        raise ProtocolLoggerError(
            "Audit record fields are incorrect."
        )

    if (
        record["pseudonym_id"]
        != "PID-6G-UE-0001"
    ):
        raise ProtocolLoggerError(
            "Pseudonym was not preserved."
        )

    serialized_record = json.dumps(
        record,
        sort_keys=True,
    )

    forbidden_values = (
        "THIS-MUST-NOT-BE-LOGGED",
        "RAW-IMSI-MUST-NOT-BE-LOGGED",
        "SECRET-MUST-NOT-BE-LOGGED",
    )

    for forbidden_value in forbidden_values:
        if forbidden_value in serialized_record:
            raise ProtocolLoggerError(
                "Sensitive information entered the audit log."
            )

    if (
        record["features"] is not None
        and "eve_fraction"
        in record["features"]
    ):
        raise ProtocolLoggerError(
            "Hidden Eve evidence entered the feature log."
        )

    if (
        record["features"] is None
        or record["features"]["qber_raw"]
        != 0.01
    ):
        raise ProtocolLoggerError(
            "Observable features were not preserved."
        )

    snapshot = get_audit_log()

    snapshot[0]["accepted"] = False

    if AUDIT_LOG[0]["accepted"] is not True:
        raise ProtocolLoggerError(
            "Audit snapshot modified the source log."
        )

    print(
        "Protocol logger self-test completed successfully."
    )

    print(
        "Audit records:",
        audit_record_count(),
    )

    print(
        "Logged pseudonym:",
        record["pseudonym_id"],
    )

    print(
        "Accepted:",
        record["accepted"],
    )

    print(
        "Sensitive data excluded: True"
    )


__all__ = [
    "AUDIT_LOG",
    "SENSITIVE_LOG_KEYS",
    "HIDDEN_SIMULATOR_KEYS",
    "ProtocolLoggerError",
    "InvalidAuditRequestError",
    "InvalidAuditDecisionError",
    "InvalidAuditFeatureError",
    "AuditExportError",
    "FTQuPAPProtocolLogger",
    "current_timestamp",
    "normalize_field_name",
    "is_sensitive_field",
    "is_hidden_simulator_field",
    "normalize_mapping",
    "extract_pseudonym_id",
    "normalize_reason_list",
    "normalize_optional_probability",
    "normalize_decision",
    "sanitize_feature_value",
    "sanitize_features",
    "build_audit_record",
    "append_audit_record",
    "append_authentication_result",
    "get_audit_log",
    "get_latest_audit_record",
    "clear_audit_log",
    "audit_record_count",
    "export_audit_log",
    "append_jsonl_record",
]


if __name__ == "__main__":
    try:
        run_self_test()

    except (
        ProtocolLoggerError,
        TypeError,
        ValueError,
    ) as error:
        print(
            "\n[PROTOCOL LOGGER ERROR] "
            f"{error}"
        )

        raise SystemExit(1) from error