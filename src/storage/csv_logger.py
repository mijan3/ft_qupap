"""
CSV Logging Utilities
FT-QuPAP v5.1

This module provides thread-safe CSV logging and table-export utilities
for FT-QuPAP demonstrations, experiments, dashboards, and evaluation.

Typical output files include:

    data/demo/demo_session_logs.csv
    data/demo/dashboard_results.csv
    data/results/performance_metrics.csv
    data/results/baseline_comparison.csv
    data/results/retry_results.csv
    data/results/confusion_matrix.csv
    data/results/calibration_results.csv
    data/results/threshold_analysis.csv

The logger supports:

1. Appending protocol events
2. Appending compact authentication-session results
3. Exporting complete experiment tables
4. Automatically extending CSV headers
5. Serializing nested dictionaries and lists as JSON
6. Atomic file replacement
7. Removing secret protocol material before storage

Security restrictions:
    The logger must not persist:

    - ML-DSA secret/private keys
    - ML-KEM secret/private keys
    - ML-KEM shared secrets
    - K_auth or K_ctrl
    - Raw KMAC authentication tags
    - Raw nonces
    - Raw ML-KEM ciphertexts
    - AES-GCM control-schedule keys
    - Quantum state vectors
    - Complete confidential payloads

Only non-secret protocol evidence, measurements, decisions, digests,
public metadata, and reproducibility information should be exported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import threading
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SESSION_LOG_PATH = Path(
    "data/demo/demo_session_logs.csv"
)

DEFAULT_PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DEFAULT_PROTOCOL_VARIANT = "P1_FT_QuPAP_GP"

DEFAULT_PAYLOAD_LOGICAL_BLOCKS = 128
DEFAULT_CHECK_LOGICAL_BLOCKS = 32
DEFAULT_PHYSICAL_QUBITS = 1120
DEFAULT_MAX_ATTEMPTS = 3

DEFAULT_OPERATIONAL_GP_THRESHOLD = 0.15
DEFAULT_GP_GRAY_ZONE_UPPER = 0.20
DEFAULT_REQUIRED_CHECK_BLOCKS = 24


# Field names are normalized before checking this set.
SENSITIVE_FIELD_NAMES = {
    "private_key",
    "secret_key",
    "signing_private_key",
    "signing_secret_key",
    "ml_dsa_private_key",
    "ml_dsa_secret_key",
    "mldsa_private_key",
    "mldsa_secret_key",
    "ml_kem_private_key",
    "ml_kem_secret_key",
    "mlkem_private_key",
    "mlkem_secret_key",
    "shared_secret",
    "session_secret",
    "mlkem_shared_secret",
    "k_ss",
    "k_auth",
    "k_ctrl",
    "authentication_key",
    "control_key",
    "kmac_key",
    "raw_nonce",
    "nonce",
    "authentication_nonce",
    "ciphertext",
    "mlkem_ciphertext",
    "raw_ciphertext",
    "expected_tag",
    "received_tag",
    "kmac_tag",
    "raw_tag",
    "authentication_tag",
    "payload_bits",
    "raw_payload",
    "confidential_payload",
    "encrypted_schedule",
    "control_schedule_plaintext",
    "aes_gcm_key",
    "quantum_state",
    "quantum_statevector",
    "statevector",
    "density_matrix",
}


# These values are safe when they are cryptographic digests rather than
# the original secret material.
PERMITTED_DIGEST_FIELDS = {
    "nonce_digest",
    "request_digest",
    "transcript_digest",
    "public_key_fingerprint",
    "trust_anchor_fingerprint",
    "file_sha256",
    "model_sha256",
}


SUPPORTED_CONTEXTS = {
    "urban",
    "suburban",
    "rural",
    "unknown",
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


class CSVLogger:
    """
    Thread-safe CSV logger for FT-QuPAP protocol evidence.

    The logger supports changing schemas. When a record contains new
    columns, the existing file is atomically rewritten using the union
    of the old and new field names.

    Args:
        file_path:
            Destination CSV file.

        protected_mode:
            When True, sensitive fields are removed before writing.

        flatten_nested:
            When True, nested dictionaries are flattened using dot
            notation. Lists remain JSON-encoded values.

        include_log_metadata:
            When True, every row receives log_id and logged_at fields.
    """

    def __init__(
        self,
        file_path: str | Path = DEFAULT_SESSION_LOG_PATH,
        *,
        protected_mode: bool = True,
        flatten_nested: bool = True,
        include_log_metadata: bool = True,
    ) -> None:
        """Initialize the CSV logger."""

        self._file_path = Path(file_path)
        self._protected_mode = validate_boolean(
            "protected_mode",
            protected_mode,
        )
        self._flatten_nested = validate_boolean(
            "flatten_nested",
            flatten_nested,
        )
        self._include_log_metadata = validate_boolean(
            "include_log_metadata",
            include_log_metadata,
        )

        self._lock = threading.RLock()
        self._fieldnames: list[str] = []

        self._initialize_file()

    @property
    def file_path(self) -> Path:
        """Return the CSV destination path."""

        return self._file_path

    @property
    def protected_mode(self) -> bool:
        """Return whether sensitive-field filtering is enabled."""

        return self._protected_mode

    @property
    def fieldnames(self) -> list[str]:
        """Return a copy of the current CSV field names."""

        with self._lock:
            return list(self._fieldnames)

    def _initialize_file(self) -> None:
        """Create the parent directory and inspect an existing CSV."""

        with self._lock:
            self._file_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not self._file_path.exists():
                self._file_path.touch()
                self._fieldnames = []
                return

            self._fieldnames = self._read_header()

    def _read_header(self) -> list[str]:
        """Read the CSV header without loading all rows."""

        if self._file_path.stat().st_size == 0:
            return []

        try:
            with self._file_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader, [])

        except csv.Error as error:
            raise ValueError(
                f"CSV file contains an invalid header: "
                f"{self._file_path}"
            ) from error

        except OSError as error:
            raise OSError(
                f"Unable to read CSV file: {self._file_path}"
            ) from error

        return [
            normalize_required_string(
                "CSV field name",
                field_name,
            )
            for field_name in header
        ]

    def log(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, str]:
        """
        Append one record to the CSV file.

        Returns:
            The normalized row written to the file.
        """

        normalized_record = self.prepare_record(record)

        with self._lock:
            self._append_prepared_records(
                [normalized_record]
            )

        return normalized_record

    def log_many(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> int:
        """
        Append multiple records in one operation.

        Returns:
            Number of records written.
        """

        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError(
                "records must be an iterable of mappings."
            )

        prepared_records = [
            self.prepare_record(record)
            for record in records
        ]

        if not prepared_records:
            return 0

        with self._lock:
            self._append_prepared_records(
                prepared_records
            )

        return len(prepared_records)

    def prepare_record(
        self,
        record: Mapping[str, Any],
    ) -> dict[str, str]:
        """
        Sanitize, flatten, and serialize a record for CSV storage.
        """

        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping.")

        working_record = dict(record)

        if self._include_log_metadata:
            working_record.setdefault(
                "log_id",
                generate_log_id(),
            )

            working_record.setdefault(
                "logged_at",
                current_utc_timestamp(),
            )

        if self._protected_mode:
            working_record = sanitize_record(
                working_record
            )

        if self._flatten_nested:
            working_record = flatten_dictionary(
                working_record
            )

        normalized_record: dict[str, str] = {}

        for key, value in working_record.items():
            normalized_key = normalize_required_string(
                "CSV field name",
                str(key),
            )

            normalized_record[
                normalized_key
            ] = serialize_csv_value(value)

        return normalized_record

    def _append_prepared_records(
        self,
        records: list[dict[str, str]],
    ) -> None:
        """Append prepared rows, extending the header when required."""

        if not records:
            return

        incoming_fields = ordered_union(
            record.keys()
            for record in records
        )

        if not self._fieldnames:
            self._fieldnames = incoming_fields
            self._rewrite_file(records)
            return

        combined_fields = ordered_union(
            [
                self._fieldnames,
                incoming_fields,
            ]
        )

        if combined_fields != self._fieldnames:
            existing_records = self.read_records()
            self._fieldnames = combined_fields

            self._rewrite_file(
                existing_records + records
            )
            return

        try:
            with self._file_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=self._fieldnames,
                    extrasaction="ignore",
                )

                for record in records:
                    writer.writerow(record)

                csv_file.flush()
                os.fsync(csv_file.fileno())

        except (OSError, csv.Error) as error:
            raise OSError(
                f"Unable to append to CSV file: "
                f"{self._file_path}"
            ) from error

    def _rewrite_file(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """
        Atomically rewrite the complete CSV file.
        """

        temporary_path: Path | None = None

        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f"{self._file_path.name}.",
                suffix=".tmp",
                dir=str(self._file_path.parent),
            )

            temporary_path = Path(temporary_name)

            with os.fdopen(
                file_descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as temporary_file:
                if self._fieldnames:
                    writer = csv.DictWriter(
                        temporary_file,
                        fieldnames=self._fieldnames,
                        extrasaction="ignore",
                    )

                    writer.writeheader()

                    for record in records:
                        writer.writerow(
                            {
                                field_name:
                                    serialize_csv_value(
                                        record.get(
                                            field_name,
                                            "",
                                        )
                                    )
                                for field_name
                                in self._fieldnames
                            }
                        )

                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(
                temporary_path,
                self._file_path,
            )

        except (OSError, csv.Error) as error:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink(missing_ok=True)

            raise OSError(
                f"Unable to rewrite CSV file: "
                f"{self._file_path}"
            ) from error

    def read_records(
        self,
        *,
        deserialize_json: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Read all stored CSV records.

        Args:
            deserialize_json:
                Parse values that appear to contain JSON objects or
                arrays.

        Returns:
            List of stored rows.
        """

        with self._lock:
            if (
                not self._file_path.exists()
                or self._file_path.stat().st_size == 0
            ):
                return []

            try:
                with self._file_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as csv_file:
                    reader = csv.DictReader(csv_file)

                    records: list[dict[str, Any]] = []

                    for row in reader:
                        if deserialize_json:
                            records.append(
                                {
                                    key:
                                        deserialize_csv_value(
                                            value
                                        )
                                    for key, value
                                    in row.items()
                                }
                            )
                        else:
                            records.append(dict(row))

                    return records

            except csv.Error as error:
                raise ValueError(
                    f"CSV file contains invalid data: "
                    f"{self._file_path}"
                ) from error

            except OSError as error:
                raise OSError(
                    f"Unable to read CSV file: "
                    f"{self._file_path}"
                ) from error

    def count_records(self) -> int:
        """Return the number of stored CSV rows."""

        return len(self.read_records())

    def clear(self) -> None:
        """Remove all rows and the existing CSV header."""

        with self._lock:
            previous_records = self.read_records()
            previous_fieldnames = list(
                self._fieldnames
            )

            self._fieldnames = []

            try:
                self._rewrite_file([])
            except Exception:
                self._fieldnames = previous_fieldnames
                self._rewrite_file(previous_records)
                raise

    def replace_all(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str] | None = None,
    ) -> int:
        """
        Replace the complete CSV table with new records.

        This operation is suitable for exported result tables such as
        baseline_comparison.csv or calibration_results.csv.
        """

        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError(
                "records must be an iterable of mappings."
            )

        prepared_records = [
            self.prepare_record(record)
            for record in records
        ]

        if fieldnames is None:
            resolved_fields = ordered_union(
                record.keys()
                for record in prepared_records
            )
        else:
            resolved_fields = [
                normalize_required_string(
                    "fieldname",
                    str(field_name),
                )
                for field_name in fieldnames
            ]

            additional_fields = ordered_union(
                record.keys()
                for record in prepared_records
            )

            resolved_fields = ordered_union(
                [
                    resolved_fields,
                    additional_fields,
                ]
            )

        with self._lock:
            previous_records = self.read_records()
            previous_fieldnames = list(
                self._fieldnames
            )

            self._fieldnames = resolved_fields

            try:
                self._rewrite_file(
                    prepared_records
                )
            except Exception:
                self._fieldnames = previous_fieldnames
                self._rewrite_file(previous_records)
                raise

        return len(prepared_records)

    def delete_where(
        self,
        predicate: Any,
    ) -> int:
        """
        Delete records for which predicate(record) returns True.

        The predicate receives CSV string values.

        Returns:
            Number of deleted records.
        """

        if not callable(predicate):
            raise TypeError("predicate must be callable.")

        with self._lock:
            records = self.read_records()

            remaining_records = [
                record
                for record in records
                if not predicate(record)
            ]

            removed_count = (
                len(records) - len(remaining_records)
            )

            if removed_count == 0:
                return 0

            self._rewrite_file(
                remaining_records
            )

            return removed_count

    def calculate_sha256(self) -> str:
        """Return the SHA-256 digest of the CSV file."""

        with self._lock:
            digest = hashlib.sha256()

            with self._file_path.open("rb") as csv_file:
                for block in iter(
                    lambda: csv_file.read(
                        1024 * 1024
                    ),
                    b"",
                ):
                    digest.update(block)

            return digest.hexdigest()

    def log_event(
        self,
        *,
        event_type: str,
        session_id: str | None = None,
        component: str | None = None,
        status: str | None = None,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Store one generic protocol event.

        This method is useful for dashboard monitoring and demo logs.
        """

        event_record: dict[str, Any] = {
            "record_type": "protocol_event",
            "protocol_version":
                DEFAULT_PROTOCOL_VERSION,
            "event_type":
                normalize_required_string(
                    "event_type",
                    event_type,
                ),
            "session_id":
                normalize_optional_string(
                    session_id
                ),
            "component":
                normalize_optional_string(
                    component
                ),
            "status":
                normalize_optional_string(
                    status
                ),
            "message":
                normalize_optional_string(
                    message
                ),
            "details": dict(details or {}),
        }

        return self.log(event_record)

    def log_authentication_result(
        self,
        protocol_result: Mapping[str, Any],
        *,
        session_id: str,
        scenario_name: str | None = None,
        protocol_variant: str = (
            DEFAULT_PROTOCOL_VARIANT
        ),
        actual_attack: bool | None = None,
        seed: int | None = None,
    ) -> dict[str, str]:
        """
        Extract and store a compact FT-QuPAP session result.

        The method deliberately selects only non-secret fields from the
        full protocol result.
        """

        compact_record = build_session_log_record(
            protocol_result=protocol_result,
            session_id=session_id,
            scenario_name=scenario_name,
            protocol_variant=protocol_variant,
            actual_attack=actual_attack,
            seed=seed,
        )

        return self.log(compact_record)

    @classmethod
    def export_table(
        cls,
        destination: str | Path,
        records: Iterable[Mapping[str, Any]],
        *,
        fieldnames: Sequence[str] | None = None,
        protected_mode: bool = True,
    ) -> int:
        """
        Export a complete table to a destination CSV file.
        """

        logger = cls(
            destination,
            protected_mode=protected_mode,
            flatten_nested=False,
            include_log_metadata=False,
        )

        return logger.replace_all(
            records,
            fieldnames=fieldnames,
        )


def build_session_log_record(
    protocol_result: Mapping[str, Any],
    *,
    session_id: str,
    scenario_name: str | None = None,
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT,
    actual_attack: bool | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Convert a complete FT-QuPAP result into a safe session-log row.

    The output follows the observable result structure used by the
    notebook's evaluation and retry pipeline.
    """

    if not isinstance(protocol_result, Mapping):
        raise TypeError(
            "protocol_result must be a mapping."
        )

    normalized_session_id = normalize_required_string(
        "session_id",
        session_id,
    )

    request = safe_mapping(
        protocol_result.get("request")
    )

    decision = safe_mapping(
        protocol_result.get("decision")
    )

    features = safe_mapping(
        protocol_result.get("features")
    )

    timings = safe_mapping(
        protocol_result.get("timings")
    )

    channel = safe_mapping(
        protocol_result.get("channel")
    )

    deterministic_reasons = normalize_reason_list(
        decision.get("deterministic_reasons")
    )

    retry_attempts = safe_integer(
        protocol_result.get(
            "retry_attempts",
            1,
        ),
        default=1,
    )

    retry_used = bool(
        protocol_result.get(
            "retry_used",
            retry_attempts > 1,
        )
    )

    accepted = bool(
        decision.get("accepted", False)
    )

    reason = str(
        decision.get(
            "reason",
            "unspecified",
        )
    )

    qber_raw = safe_float_or_none(
        protocol_result.get("qber_raw")
    )

    p_attack = safe_float_or_none(
        decision.get("p_attack")
    )

    threshold = safe_float_or_none(
        decision.get(
            "gp_attack_threshold"
        )
    )

    predicted_attack = (
        p_attack >= threshold
        if (
            p_attack is not None
            and threshold is not None
        )
        else None
    )

    resolved_scenario = (
        scenario_name
        or channel.get("name")
        or "normal_session"
    )

    context = normalize_context(
        str(
            channel.get(
                "context",
                infer_context(features),
            )
        )
    )

    schedule_reason = protocol_result.get(
        "schedule_reason"
    )

    schedule_valid = (
        schedule_reason == "schedule_valid"
        if schedule_reason is not None
        else None
    )

    payload_failures = protocol_result.get(
        "payload_failures",
        [],
    )

    if not isinstance(payload_failures, list):
        payload_failures = []

    outcome = derive_session_outcome(
        accepted=accepted,
        reason=reason,
        deterministic_reasons=(
            deterministic_reasons
        ),
        retry_attempts=retry_attempts,
    )

    return {
        "record_type": "authentication_result",
        "protocol_version":
            DEFAULT_PROTOCOL_VERSION,
        "protocol_variant":
            normalize_required_string(
                "protocol_variant",
                protocol_variant,
            ),
        "session_id": normalized_session_id,
        "pseudonym_id":
            normalize_optional_string(
                request.get("pseudonym_id")
            ),
        "scenario_name": str(
            resolved_scenario
        ),
        "channel_name": str(
            channel.get("name", "unknown")
        ),
        "context": context,
        "seed": seed,

        "accepted": accepted,
        "outcome": outcome,
        "reason": reason,

        "deterministic_pass": bool(
            decision.get(
                "deterministic_pass",
                False,
            )
        ),
        "deterministic_reasons":
            deterministic_reasons,

        "schedule_valid": schedule_valid,
        "tag_recovered":
            infer_tag_recovery_status(
                protocol_result
            ),

        "qber_raw": qber_raw,
        "qber_mismatches":
            safe_integer_or_none(
                protocol_result.get(
                    "qber_mismatches"
                )
            ),
        "qber_observed":
            safe_integer_or_none(
                protocol_result.get(
                    "qber_observed"
                )
            ),
        "observed_check_blocks":
            safe_integer_or_none(
                protocol_result.get(
                    "observed_check_blocks"
                )
            ),
        "required_check_blocks":
            safe_integer(
                protocol_result.get(
                    "required_check_blocks",
                    DEFAULT_REQUIRED_CHECK_BLOCKS,
                ),
                default=(
                    DEFAULT_REQUIRED_CHECK_BLOCKS
                ),
            ),
        "loss_rate":
            safe_float_or_none(
                protocol_result.get(
                    "loss_rate"
                )
            ),

        "payload_failure_count":
            len(payload_failures),

        "mean_syndrome_weight":
            safe_float_or_none(
                features.get(
                    "mean_syndrome_weight"
                )
            ),
        "max_syndrome_weight":
            safe_float_or_none(
                features.get(
                    "max_syndrome_weight"
                )
            ),
        "correction_failure_rate":
            safe_float_or_none(
                features.get(
                    "correction_failure_rate"
                )
            ),
        "noise_estimate":
            safe_float_or_none(
                features.get(
                    "noise_estimate"
                )
            ),

        "ctx_urban":
            safe_float(
                features.get(
                    "ctx_urban",
                    0.0,
                )
            ),
        "ctx_suburban":
            safe_float(
                features.get(
                    "ctx_suburban",
                    0.0,
                )
            ),
        "ctx_rural":
            safe_float(
                features.get(
                    "ctx_rural",
                    0.0,
                )
            ),

        "p_attack": p_attack,
        "uncertainty":
            safe_float_or_none(
                decision.get(
                    "uncertainty"
                )
            ),
        "raw_calibration_gp_attack_threshold":
            safe_float_or_none(
                decision.get(
                    "raw_calibration_gp_attack_threshold"
                )
            ),
        "gp_attack_threshold": threshold,
        "gp_gray_zone_retry_upper":
            safe_float(
                decision.get(
                    "gp_gray_zone_retry_upper",
                    DEFAULT_GP_GRAY_ZONE_UPPER,
                )
            ),
        "predicted_attack":
            predicted_attack,
        "actual_attack":
            actual_attack,

        "retry_attempts":
            retry_attempts,
        "retry_used":
            retry_used,
        "max_attempts":
            safe_integer(
                protocol_result.get(
                    "max_attempts",
                    DEFAULT_MAX_ATTEMPTS,
                ),
                default=DEFAULT_MAX_ATTEMPTS,
            ),

        "logical_payload_blocks":
            safe_integer(
                protocol_result.get(
                    "logical_payload_blocks",
                    DEFAULT_PAYLOAD_LOGICAL_BLOCKS,
                ),
                default=(
                    DEFAULT_PAYLOAD_LOGICAL_BLOCKS
                ),
            ),
        "logical_check_blocks":
            safe_integer(
                protocol_result.get(
                    "logical_check_blocks",
                    DEFAULT_CHECK_LOGICAL_BLOCKS,
                ),
                default=(
                    DEFAULT_CHECK_LOGICAL_BLOCKS
                ),
            ),
        "physical_qubits":
            safe_integer(
                protocol_result.get(
                    "physical_qubits",
                    DEFAULT_PHYSICAL_QUBITS,
                ),
                default=DEFAULT_PHYSICAL_QUBITS,
            ),

        "use_css": bool(
            protocol_result.get(
                "use_css",
                True,
            )
        ),
        "bootstrap_mode": str(
            protocol_result.get(
                "bootstrap_mode",
                "mlkem",
            )
        ),
        "decision_mode": str(
            protocol_result.get(
                "decision_mode",
                "gp",
            )
        ),

        "timing_credential_and_kem_keygen_s":
            safe_float_or_none(
                timings.get(
                    "credential_and_kem_keygen_s"
                )
            ),
        "timing_mlkem_encapsulation_s":
            safe_float_or_none(
                timings.get(
                    "mlkem_encapsulation_s"
                )
            ),
        "timing_mlkem_decapsulation_s":
            safe_float_or_none(
                timings.get(
                    "mlkem_decapsulation_s"
                )
            ),
        "timing_tag_and_schedule_s":
            safe_float_or_none(
                timings.get(
                    "tag_and_schedule_s"
                )
            ),
        "timing_css_encoding_s":
            safe_float_or_none(
                timings.get(
                    "css_encoding_s"
                )
            ),
        "timing_quantum_channel_simulation_s":
            safe_float_or_none(
                timings.get(
                    "quantum_channel_simulation_s"
                )
            ),
        "timing_measurement_and_css_decoding_s":
            safe_float_or_none(
                timings.get(
                    "measurement_and_css_decoding_s"
                )
            ),
        "timing_gp_inference_s":
            safe_float_or_none(
                timings.get(
                    "gp_inference_s"
                )
            ),
        "timing_end_to_end_s":
            safe_float_or_none(
                timings.get(
                    "end_to_end_s"
                )
            ),
        "timing_total_retry_end_to_end_s":
            safe_float_or_none(
                timings.get(
                    "total_retry_end_to_end_s"
                )
            ),

        "attempt_history":
            sanitize_attempt_history(
                protocol_result.get(
                    "attempt_history",
                    [],
                )
            ),
    }


def sanitize_record(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively remove secret fields from a record.
    """

    sanitized: dict[str, Any] = {}

    for raw_key, value in record.items():
        key = str(raw_key)
        normalized_key = normalize_field_name(key)

        if (
            normalized_key in SENSITIVE_FIELD_NAMES
            and normalized_key
            not in PERMITTED_DIGEST_FIELDS
        ):
            continue

        if isinstance(value, Mapping):
            sanitized[key] = sanitize_record(value)

        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_record(item)
                if isinstance(item, Mapping)
                else sanitize_sequence_value(item)
                for item in value
            ]

        elif isinstance(value, tuple):
            sanitized[key] = [
                sanitize_record(item)
                if isinstance(item, Mapping)
                else sanitize_sequence_value(item)
                for item in value
            ]

        else:
            sanitized[key] = value

    return sanitized


def sanitize_sequence_value(value: Any) -> Any:
    """Sanitize a value contained in a list or tuple."""

    if isinstance(value, Mapping):
        return sanitize_record(value)

    if isinstance(value, (list, tuple)):
        return [
            sanitize_sequence_value(item)
            for item in value
        ]

    if isinstance(value, (bytes, bytearray)):
        return {
            "binary_value_removed": True,
            "byte_length": len(value),
        }

    return value


def flatten_dictionary(
    data: Mapping[str, Any],
    *,
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """
    Flatten nested dictionaries using dot-separated field names.

    Example:
        {"decision": {"accepted": True}}
        becomes:
        {"decision.accepted": True}
    """

    flattened: dict[str, Any] = {}

    for raw_key, value in data.items():
        key = str(raw_key)

        combined_key = (
            f"{parent_key}{separator}{key}"
            if parent_key
            else key
        )

        if isinstance(value, Mapping):
            nested = flatten_dictionary(
                value,
                parent_key=combined_key,
                separator=separator,
            )

            flattened.update(nested)
        else:
            flattened[combined_key] = value

    return flattened


def serialize_csv_value(value: Any) -> str:
    """Convert a Python value into a deterministic CSV string."""

    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, datetime):
        return normalize_timestamp(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return json.dumps(
            {
                "binary_value_removed": True,
                "byte_length": len(value),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    if isinstance(value, bytearray):
        return json.dumps(
            {
                "binary_value_removed": True,
                "byte_length": len(value),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    if isinstance(value, Mapping):
        return json.dumps(
            sanitize_record(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    if isinstance(value, (list, tuple, set)):
        return json.dumps(
            [
                sanitize_sequence_value(item)
                for item in value
            ],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    if isinstance(value, float):
        if not math.isfinite(value):
            return ""

        return repr(value)

    return str(value)


def deserialize_csv_value(value: str) -> Any:
    """
    Parse JSON-looking CSV values while leaving normal text unchanged.
    """

    if value is None:
        return None

    normalized_value = value.strip()

    if normalized_value == "":
        return ""

    if normalized_value == "true":
        return True

    if normalized_value == "false":
        return False

    if (
        normalized_value.startswith("{")
        and normalized_value.endswith("}")
    ) or (
        normalized_value.startswith("[")
        and normalized_value.endswith("]")
    ):
        try:
            return json.loads(normalized_value)
        except json.JSONDecodeError:
            return value

    return value


def ordered_union(
    collections: Iterable[Iterable[str]],
) -> list[str]:
    """Return unique values while preserving first-seen order."""

    output: list[str] = []
    seen: set[str] = set()

    for collection in collections:
        for value in collection:
            normalized_value = str(value)

            if normalized_value not in seen:
                seen.add(normalized_value)
                output.append(normalized_value)

    return output


def sanitize_attempt_history(
    attempt_history: Any,
) -> list[dict[str, Any]]:
    """
    Retain only safe retry evidence from attempt history.
    """

    if not isinstance(attempt_history, list):
        return []

    safe_attempts: list[dict[str, Any]] = []

    permitted_fields = {
        "attempt",
        "attempt_index",
        "accepted",
        "reason",
        "deterministic_pass",
        "deterministic_reasons",
        "qber_raw",
        "qber_mismatches",
        "qber_observed",
        "observed_check_blocks",
        "loss_rate",
        "tag_recovered",
        "p_attack",
        "uncertainty",
        "gp_attack_threshold",
        "retryable",
        "timing_end_to_end_s",
    }

    for raw_attempt in attempt_history:
        if not isinstance(raw_attempt, Mapping):
            continue

        safe_attempt = {
            str(key): sanitize_sequence_value(value)
            for key, value in raw_attempt.items()
            if str(key) in permitted_fields
        }

        safe_attempts.append(safe_attempt)

    return safe_attempts


def infer_tag_recovery_status(
    protocol_result: Mapping[str, Any],
) -> bool | None:
    """
    Determine tag recovery without storing either tag.

    A direct tag_recovered field is preferred. If the result contains
    expected_tag and received_tag byte values, only their equality is
    returned.
    """

    direct_value = protocol_result.get(
        "tag_recovered"
    )

    if isinstance(direct_value, bool):
        return direct_value

    expected_tag = protocol_result.get(
        "expected_tag"
    )

    received_tag = protocol_result.get(
        "received_tag"
    )

    if not isinstance(
        expected_tag,
        (bytes, bytearray),
    ):
        return None

    if not isinstance(
        received_tag,
        (bytes, bytearray),
    ):
        return None

    import hmac

    return hmac.compare_digest(
        bytes(expected_tag),
        bytes(received_tag),
    )


def derive_session_outcome(
    *,
    accepted: bool,
    reason: str,
    deterministic_reasons: list[str],
    retry_attempts: int,
) -> str:
    """Convert final protocol reasons into a normalized outcome."""

    normalized_reason = reason.strip().lower()

    normalized_reasons = {
        normalized_reason,
        *(
            reason_value.strip().lower()
            for reason_value
            in deterministic_reasons
        ),
    }

    if accepted:
        if (
            retry_attempts > 1
            or normalized_reason
            == "accepted_after_retry"
        ):
            return "accepted_after_retry"

        return "accepted"

    if any(
        "replay" in reason_value
        or "nonce" in reason_value
        for reason_value in normalized_reasons
    ):
        return "rejected_replay"

    if any(
        "credential" in reason_value
        or "signature" in reason_value
        or "trust_anchor" in reason_value
        for reason_value in normalized_reasons
    ):
        return "rejected_credential"

    if any(
        "ciphertext" in reason_value
        or "decapsulation" in reason_value
        or "session_secret" in reason_value
        for reason_value in normalized_reasons
    ):
        return "rejected_ciphertext"

    if (
        "gp" in normalized_reason
        or normalized_reason
        == "rejected_by_calibrated_bayesian_policy"
    ):
        return "rejected_gp"

    if (
        retry_attempts >= DEFAULT_MAX_ATTEMPTS
        and any(
            reason_value in {
                "payload_block_unrecoverable",
                "authentication_tag_mismatch",
            }
            for reason_value in normalized_reasons
        )
    ):
        return "rejected_retry_exhausted"

    if deterministic_reasons:
        return "rejected_deterministic"

    if "abort" in normalized_reason:
        return "aborted"

    if "fail" in normalized_reason:
        return "failed"

    return "rejected"


def infer_context(
    features: Mapping[str, Any],
) -> str:
    """Infer channel context from GP one-hot features."""

    scores = {
        "urban": safe_float(
            features.get("ctx_urban", 0.0)
        ),
        "suburban": safe_float(
            features.get(
                "ctx_suburban",
                0.0,
            )
        ),
        "rural": safe_float(
            features.get("ctx_rural", 0.0)
        ),
    }

    selected_context = max(
        scores,
        key=scores.get,
    )

    if scores[selected_context] <= 0.0:
        return "unknown"

    return selected_context


def normalize_context(context: str) -> str:
    """Validate a supported channel context."""

    normalized_context = (
        normalize_required_string(
            "context",
            context,
        ).lower()
    )

    if normalized_context not in SUPPORTED_CONTEXTS:
        return "unknown"

    return normalized_context


def normalize_reason_list(
    value: Any,
) -> list[str]:
    """Normalize deterministic-reason values."""

    if value is None:
        return []

    if isinstance(value, str):
        normalized_value = value.strip()

        if not normalized_value:
            return []

        return [normalized_value]

    if isinstance(value, Iterable):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [str(value)]


def safe_mapping(value: Any) -> dict[str, Any]:
    """Return a dictionary when value is mapping-like."""

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a finite numeric value to float."""

    converted = safe_float_or_none(value)

    return default if converted is None else converted


def safe_float_or_none(
    value: Any,
) -> float | None:
    """Convert a finite numeric value to float or return None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(converted):
        return None

    return converted


def safe_integer(
    value: Any,
    *,
    default: int = 0,
) -> int:
    """Convert an integer-like value to int."""

    converted = safe_integer_or_none(value)

    return default if converted is None else converted


def safe_integer_or_none(
    value: Any,
) -> int | None:
    """Convert an integer-like value to int or return None."""

    if value is None or isinstance(value, bool):
        return None

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(converted):
        return None

    if not converted.is_integer():
        return None

    return int(converted)


def normalize_field_name(value: str) -> str:
    """Normalize a field name for security-policy matching."""

    output: list[str] = []

    for character in value.strip():
        if character.isalnum():
            output.append(character.lower())
        else:
            output.append("_")

    normalized = "".join(output)

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

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(f"{name} cannot be empty.")

    return normalized_value


def normalize_optional_string(
    value: Any,
) -> str | None:
    """Normalize an optional string."""

    if value is None:
        return None

    normalized_value = str(value).strip()

    return normalized_value or None


def validate_boolean(
    name: str,
    value: bool,
) -> bool:
    """Validate and return a required boolean."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")

    return value


def current_utc_timestamp() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(
    value: datetime | str,
) -> str:
    """Convert a datetime or ISO value into UTC ISO format."""

    if isinstance(value, datetime):
        parsed_datetime = value

    elif isinstance(value, str):
        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError(
                "Timestamp cannot be empty."
            )

        if normalized_value.endswith("Z"):
            normalized_value = (
                normalized_value[:-1]
                + "+00:00"
            )

        try:
            parsed_datetime = datetime.fromisoformat(
                normalized_value
            )
        except ValueError as error:
            raise ValueError(
                "Timestamp must use ISO 8601 format."
            ) from error

    else:
        raise TypeError(
            "Timestamp must be a datetime or ISO string."
        )

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(
            tzinfo=timezone.utc
        )

    return parsed_datetime.astimezone(
        timezone.utc
    ).isoformat()


def generate_log_id() -> str:
    """Generate a unique CSV log identifier."""

    return f"LOG-{uuid.uuid4().hex.upper()}"


def run_self_test() -> None:
    """Run CSV security, schema-extension, and session-log tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        log_path = (
            Path(temporary_directory)
            / "demo_session_logs.csv"
        )

        logger = CSVLogger(log_path)

        logger.log_event(
            event_type="authentication_started",
            session_id="SESSION-0001",
            component="protocol_engine",
            status="initialized",
            details={
                "scenario": "normal_session",
                "context": "urban",
            },
        )

        expected_tag = bytes.fromhex(
            "00112233445566778899aabbccddeeff"
        )

        protocol_result = {
            "request": {
                "pseudonym_id":
                    "PID-6G-UE-0001",
                "nonce":
                    "THIS-MUST-NOT-BE-STORED",
            },
            "decision": {
                "accepted": True,
                "reason": "accepted_after_retry",
                "deterministic_pass": True,
                "deterministic_reasons": [],
                "p_attack": 0.04,
                "uncertainty": 0.24,
                "gp_attack_threshold": 0.15,
                "raw_calibration_gp_attack_threshold":
                    0.12,
            },
            "channel": {
                "name": "benign_noisy_session",
                "context": "urban",
            },
            "features": {
                "qber_raw": 0.015625,
                "mean_syndrome_weight": 0.08,
                "max_syndrome_weight": 1.0,
                "correction_failure_rate": 0.0,
                "noise_estimate": 0.012,
                "ctx_urban": 1.0,
                "ctx_suburban": 0.0,
                "ctx_rural": 0.0,
            },
            "qber_raw": 0.015625,
            "qber_mismatches": 7,
            "qber_observed": 224,
            "observed_check_blocks": 32,
            "required_check_blocks": 24,
            "loss_rate": 0.01,
            "schedule_reason": "schedule_valid",
            "expected_tag": expected_tag,
            "received_tag": expected_tag,
            "shared_secret":
                b"DO-NOT-STORE-SHARED-SECRET",
            "k_auth":
                b"DO-NOT-STORE-K-AUTH",
            "k_ctrl":
                b"DO-NOT-STORE-K-CTRL",
            "payload_failures": [],
            "retry_attempts": 2,
            "retry_used": True,
            "attempt_history": [
                {
                    "attempt": 1,
                    "accepted": False,
                    "reason":
                        "authentication_tag_mismatch",
                    "qber_raw": 0.02,
                    "p_attack": 0.08,
                    "tag_recovered": False,
                    "expected_tag": expected_tag,
                },
                {
                    "attempt": 2,
                    "accepted": True,
                    "reason":
                        "accepted_after_retry",
                    "qber_raw": 0.015625,
                    "p_attack": 0.04,
                    "tag_recovered": True,
                },
            ],
            "logical_payload_blocks": 128,
            "logical_check_blocks": 32,
            "physical_qubits": 1120,
            "timings": {
                "mlkem_encapsulation_s": 0.002,
                "mlkem_decapsulation_s": 0.002,
                "css_encoding_s": 0.01,
                "gp_inference_s": 0.001,
                "end_to_end_s": 0.06,
                "total_retry_end_to_end_s": 0.12,
            },
        }

        logger.log_authentication_result(
            protocol_result,
            session_id="SESSION-0001",
            scenario_name="accept_after_retry",
            actual_attack=False,
            seed=9102,
        )

        assert logger.count_records() == 2

        records = logger.read_records(
            deserialize_json=True
        )

        assert len(records) == 2

        result_row = records[1]

        assert result_row["accepted"] is True
        assert (
            result_row["outcome"]
            == "accepted_after_retry"
        )
        assert (
            result_row["retry_attempts"]
            == "2"
        )
        assert (
            result_row["physical_qubits"]
            == "1120"
        )

        stored_text = log_path.read_text(
            encoding="utf-8"
        )

        assert "THIS-MUST-NOT-BE-STORED" not in stored_text
        assert expected_tag.hex() not in stored_text
        assert "DO-NOT-STORE-SHARED-SECRET" not in stored_text
        assert "DO-NOT-STORE-K-AUTH" not in stored_text
        assert "DO-NOT-STORE-K-CTRL" not in stored_text

        assert logger.calculate_sha256()
        assert len(logger.calculate_sha256()) == 64

        export_path = (
            Path(temporary_directory)
            / "baseline_comparison.csv"
        )

        exported_count = CSVLogger.export_table(
            export_path,
            [
                {
                    "protocol":
                        "B1_QuPAP_inspired_PSK_fixed_QBER",
                    "overall_acceptance": 0.50,
                    "mean_physical_qubits": 160,
                },
                {
                    "protocol":
                        "P1_FT_QuPAP_GP",
                    "overall_acceptance": 0.90,
                    "mean_physical_qubits": 1120,
                },
            ],
        )

        assert exported_count == 2
        assert export_path.exists()

        print("CSV logger self-test passed.")
        print(f"Log path: {log_path}")
        print(
            "Stored records: "
            f"{logger.count_records()}"
        )
        print(
            "Log SHA-256: "
            f"{logger.calculate_sha256()}"
        )


if __name__ == "__main__":
    run_self_test()