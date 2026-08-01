"""
Authentication Result Repository
FT-QuPAP v5.1

This module stores compact, non-secret FT-QuPAP authentication results
for the dashboard, controlled demonstrations, evaluation scripts, and
result exports.

The stored result follows the final notebook output structure:

- Final authentication decision
- Deterministic verification outcome and reasons
- Independent check-block QBER evidence
- Steane CSS decoding and correction evidence
- Observable Gaussian Process features
- Calibrated attack probability and uncertainty
- Raw and operational GP thresholds
- Retry attempts and accepted-after-retry outcome
- Quantum-resource and runtime measurements

Security restrictions:
    Never persist:

    - ML-DSA private keys
    - ML-KEM private keys
    - ML-KEM shared secrets
    - K_auth or K_ctrl
    - Raw KMAC tags
    - Raw authentication nonces
    - Raw ciphertexts
    - Quantum state vectors
    - Hidden Eve interception fractions as GP features

The default repository is:

    data/demo/dashboard_results.csv
"""

from __future__ import annotations

import csv
import copy
import hmac
import json
import math
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULT_DATABASE_PATH = Path(
    "data/demo/dashboard_results.csv"
)

DEFAULT_PROTOCOL_VERSION = "FT-QuPAP-v5.1"
DEFAULT_PROTOCOL_VARIANT = "P1_FT_QuPAP_GP"
DEFAULT_MAX_ATTEMPTS = 3

DEFAULT_OPERATIONAL_GP_THRESHOLD = 0.15
DEFAULT_GP_GRAY_ZONE_UPPER = 0.20
DEFAULT_REQUIRED_CHECK_BLOCKS = 24

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
}

JSON_COLUMNS = {
    "deterministic_reasons",
    "attempt_history",
    "metadata",
}


@dataclass
class AuthenticationResultRecord:
    """
    Compact persistent result from one completed FT-QuPAP session.

    Identity and experiment fields:
        result_id:
            Unique repository record identifier.

        session_id:
            Authentication-session identifier.

        pseudonym_id:
            Pseudonymous subscriber reference. A raw IMSI must never
            be stored.

        scenario_name:
            Demonstration or experimental scenario.

        protocol_variant:
            P1 FT-QuPAP or a controlled baseline identifier.

        seed:
            Optional simulation seed used for reproducibility.

    Decision fields:
        accepted:
            Final authentication decision.

        outcome:
            Normalized result category such as accepted,
            accepted_after_retry, rejected_gp, or rejected_replay.

        reason:
            Exact final decision reason produced by the protocol.

        deterministic_pass:
            Whether all mandatory deterministic checks passed.

        deterministic_reasons:
            Explicit deterministic failure reasons.

    Detection fields:
        qber_raw:
            QBER calculated from independent declared check blocks.

        p_attack:
            Calibrated GP attack probability.

        uncertainty:
            Binary predictive entropy recorded by the notebook.

        gp_attack_threshold:
            Operational GP decision threshold.

        raw_calibration_gp_attack_threshold:
            Threshold selected from calibration data before applying
            the operational lower bound.

    Retry fields:
        retry_attempts:
            Total number of authentication attempts.

        retry_used:
            True when more than one attempt was executed.

        attempt_history:
            Compact per-attempt diagnostic history.
    """

    result_id: str
    session_id: str

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    protocol_version: str = DEFAULT_PROTOCOL_VERSION
    protocol_variant: str = DEFAULT_PROTOCOL_VARIANT

    pseudonym_id: str | None = None
    scenario_name: str = "normal_session"
    channel_name: str = "unknown"
    context: str = "unknown"
    seed: int | None = None

    accepted: bool = False
    outcome: str = "rejected"
    reason: str = "unspecified"

    deterministic_pass: bool = False
    deterministic_reasons: list[str] = field(
        default_factory=list
    )

    schedule_valid: bool | None = None
    tag_recovered: bool | None = None

    payload_failure_count: int = 0
    corrected_block_count: int = 0
    uncorrectable_block_count: int = 0

    qber_raw: float | None = None
    qber_mismatches: int | None = None
    qber_observed: int | None = None
    observed_check_blocks: int | None = None
    required_check_blocks: int = (
        DEFAULT_REQUIRED_CHECK_BLOCKS
    )
    loss_rate: float | None = None

    mean_syndrome_weight: float | None = None
    max_syndrome_weight: float | None = None
    correction_failure_rate: float | None = None
    noise_estimate: float | None = None

    ctx_urban: float = 0.0
    ctx_suburban: float = 0.0
    ctx_rural: float = 0.0

    p_attack: float | None = None
    uncertainty: float | None = None

    raw_calibration_gp_attack_threshold: float | None = None
    gp_attack_threshold: float | None = None
    gp_gray_zone_retry_upper: float = (
        DEFAULT_GP_GRAY_ZONE_UPPER
    )

    predicted_attack: bool | None = None
    actual_attack: bool | None = None

    retry_attempts: int = 1
    retry_used: bool = False
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    physical_qubits: int | None = None
    logical_payload_blocks: int | None = None
    logical_check_blocks: int | None = None

    use_css: bool = True
    bootstrap_mode: str = "mlkem"
    decision_mode: str = "gp"

    credential_and_kem_keygen_s: float | None = None
    mlkem_encapsulation_s: float | None = None
    mlkem_decapsulation_s: float | None = None
    tag_and_schedule_s: float | None = None
    css_encoding_s: float | None = None
    quantum_channel_simulation_s: float | None = None
    measurement_and_css_decoding_s: float | None = None
    gp_inference_s: float | None = None
    end_to_end_s: float | None = None
    total_retry_end_to_end_s: float | None = None

    attempt_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the result record."""

        self.result_id = normalize_required_string(
            "result_id",
            self.result_id,
        )

        self.session_id = normalize_required_string(
            "session_id",
            self.session_id,
        )

        self.created_at = normalize_timestamp(
            self.created_at
        )

        self.protocol_version = normalize_required_string(
            "protocol_version",
            self.protocol_version,
        )

        self.protocol_variant = normalize_required_string(
            "protocol_variant",
            self.protocol_variant,
        )

        self.pseudonym_id = normalize_optional_string(
            self.pseudonym_id
        )

        self.scenario_name = normalize_required_string(
            "scenario_name",
            self.scenario_name,
        )

        self.channel_name = normalize_required_string(
            "channel_name",
            self.channel_name,
        )

        self.context = validate_context(self.context)
        self.outcome = validate_outcome(self.outcome)

        self.reason = normalize_required_string(
            "reason",
            self.reason,
        )

        validate_boolean("accepted", self.accepted)

        validate_boolean(
            "deterministic_pass",
            self.deterministic_pass,
        )

        self.deterministic_reasons = normalize_string_list(
            "deterministic_reasons",
            self.deterministic_reasons,
        )

        validate_optional_boolean(
            "schedule_valid",
            self.schedule_valid,
        )

        validate_optional_boolean(
            "tag_recovered",
            self.tag_recovered,
        )

        validate_optional_boolean(
            "predicted_attack",
            self.predicted_attack,
        )

        validate_optional_boolean(
            "actual_attack",
            self.actual_attack,
        )

        validate_boolean("retry_used", self.retry_used)
        validate_boolean("use_css", self.use_css)

        integer_fields = {
            "payload_failure_count":
                self.payload_failure_count,
            "corrected_block_count":
                self.corrected_block_count,
            "uncorrectable_block_count":
                self.uncorrectable_block_count,
            "required_check_blocks":
                self.required_check_blocks,
            "retry_attempts":
                self.retry_attempts,
            "max_attempts":
                self.max_attempts,
        }

        for name, value in integer_fields.items():
            validate_nonnegative_integer(name, value)

        if self.retry_attempts < 1:
            raise ValueError(
                "retry_attempts must be at least one."
            )

        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least one."
            )

        if self.retry_attempts > self.max_attempts:
            raise ValueError(
                "retry_attempts cannot exceed max_attempts."
            )

        if self.retry_used != (self.retry_attempts > 1):
            raise ValueError(
                "retry_used must be True exactly when "
                "retry_attempts is greater than one."
            )

        optional_integer_fields = {
            "seed": self.seed,
            "qber_mismatches": self.qber_mismatches,
            "qber_observed": self.qber_observed,
            "observed_check_blocks":
                self.observed_check_blocks,
            "physical_qubits": self.physical_qubits,
            "logical_payload_blocks":
                self.logical_payload_blocks,
            "logical_check_blocks":
                self.logical_check_blocks,
        }

        for name, value in optional_integer_fields.items():
            if value is not None:
                validate_nonnegative_integer(name, value)

        probability_fields = {
            "qber_raw": self.qber_raw,
            "loss_rate": self.loss_rate,
            "correction_failure_rate":
                self.correction_failure_rate,
            "p_attack": self.p_attack,
            "uncertainty": self.uncertainty,
            "raw_calibration_gp_attack_threshold":
                self.raw_calibration_gp_attack_threshold,
            "gp_attack_threshold":
                self.gp_attack_threshold,
            "gp_gray_zone_retry_upper":
                self.gp_gray_zone_retry_upper,
        }

        for name, value in probability_fields.items():
            validate_optional_probability(name, value)

        numeric_fields = {
            "mean_syndrome_weight":
                self.mean_syndrome_weight,
            "max_syndrome_weight":
                self.max_syndrome_weight,
            "noise_estimate":
                self.noise_estimate,
            "ctx_urban": self.ctx_urban,
            "ctx_suburban": self.ctx_suburban,
            "ctx_rural": self.ctx_rural,
        }

        for name, value in numeric_fields.items():
            validate_optional_finite_number(name, value)

        timing_fields = {
            "credential_and_kem_keygen_s":
                self.credential_and_kem_keygen_s,
            "mlkem_encapsulation_s":
                self.mlkem_encapsulation_s,
            "mlkem_decapsulation_s":
                self.mlkem_decapsulation_s,
            "tag_and_schedule_s":
                self.tag_and_schedule_s,
            "css_encoding_s":
                self.css_encoding_s,
            "quantum_channel_simulation_s":
                self.quantum_channel_simulation_s,
            "measurement_and_css_decoding_s":
                self.measurement_and_css_decoding_s,
            "gp_inference_s":
                self.gp_inference_s,
            "end_to_end_s":
                self.end_to_end_s,
            "total_retry_end_to_end_s":
                self.total_retry_end_to_end_s,
        }

        for name, value in timing_fields.items():
            validate_optional_nonnegative_number(
                name,
                value,
            )

        if not isinstance(self.attempt_history, list):
            raise TypeError(
                "attempt_history must be a list."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        if self.accepted and self.outcome not in {
            "accepted",
            "accepted_after_retry",
        }:
            raise ValueError(
                "Accepted results must use an accepted outcome."
            )

        if (
            not self.accepted
            and self.outcome
            in {"accepted", "accepted_after_retry"}
        ):
            raise ValueError(
                "Rejected results cannot use an accepted outcome."
            )

    def to_dictionary(self) -> dict[str, Any]:
        """Return a JSON-compatible result dictionary."""

        return asdict(self)

    def to_csv_row(self) -> dict[str, str]:
        """Convert the record into a CSV-compatible row."""

        result = self.to_dictionary()
        csv_row: dict[str, str] = {}

        for key, value in result.items():
            if key in JSON_COLUMNS:
                csv_row[key] = json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            elif value is None:
                csv_row[key] = ""
            elif isinstance(value, bool):
                csv_row[key] = (
                    "true" if value else "false"
                )
            else:
                csv_row[key] = str(value)

        return csv_row

    @classmethod
    def from_csv_row(
        cls,
        row: dict[str, str],
    ) -> "AuthenticationResultRecord":
        """Create a result record from a stored CSV row."""

        if not isinstance(row, dict):
            raise TypeError("CSV row must be a dictionary.")

        return cls(
            result_id=row["result_id"],
            session_id=row["session_id"],
            created_at=row["created_at"],
            protocol_version=row["protocol_version"],
            protocol_variant=row["protocol_variant"],
            pseudonym_id=empty_to_none(
                row.get("pseudonym_id")
            ),
            scenario_name=row.get(
                "scenario_name",
                "normal_session",
            ),
            channel_name=row.get(
                "channel_name",
                "unknown",
            ),
            context=row.get("context", "unknown"),
            seed=parse_optional_int(row.get("seed")),
            accepted=parse_bool(row.get("accepted")),
            outcome=row.get("outcome", "rejected"),
            reason=row.get("reason", "unspecified"),
            deterministic_pass=parse_bool(
                row.get("deterministic_pass")
            ),
            deterministic_reasons=parse_json_list(
                row.get("deterministic_reasons")
            ),
            schedule_valid=parse_optional_bool(
                row.get("schedule_valid")
            ),
            tag_recovered=parse_optional_bool(
                row.get("tag_recovered")
            ),
            payload_failure_count=parse_int(
                row.get("payload_failure_count"),
                default=0,
            ),
            corrected_block_count=parse_int(
                row.get("corrected_block_count"),
                default=0,
            ),
            uncorrectable_block_count=parse_int(
                row.get("uncorrectable_block_count"),
                default=0,
            ),
            qber_raw=parse_optional_float(
                row.get("qber_raw")
            ),
            qber_mismatches=parse_optional_int(
                row.get("qber_mismatches")
            ),
            qber_observed=parse_optional_int(
                row.get("qber_observed")
            ),
            observed_check_blocks=parse_optional_int(
                row.get("observed_check_blocks")
            ),
            required_check_blocks=parse_int(
                row.get("required_check_blocks"),
                default=DEFAULT_REQUIRED_CHECK_BLOCKS,
            ),
            loss_rate=parse_optional_float(
                row.get("loss_rate")
            ),
            mean_syndrome_weight=parse_optional_float(
                row.get("mean_syndrome_weight")
            ),
            max_syndrome_weight=parse_optional_float(
                row.get("max_syndrome_weight")
            ),
            correction_failure_rate=(
                parse_optional_float(
                    row.get(
                        "correction_failure_rate"
                    )
                )
            ),
            noise_estimate=parse_optional_float(
                row.get("noise_estimate")
            ),
            ctx_urban=parse_float(
                row.get("ctx_urban"),
                default=0.0,
            ),
            ctx_suburban=parse_float(
                row.get("ctx_suburban"),
                default=0.0,
            ),
            ctx_rural=parse_float(
                row.get("ctx_rural"),
                default=0.0,
            ),
            p_attack=parse_optional_float(
                row.get("p_attack")
            ),
            uncertainty=parse_optional_float(
                row.get("uncertainty")
            ),
            raw_calibration_gp_attack_threshold=(
                parse_optional_float(
                    row.get(
                        "raw_calibration_gp_attack_threshold"
                    )
                )
            ),
            gp_attack_threshold=parse_optional_float(
                row.get("gp_attack_threshold")
            ),
            gp_gray_zone_retry_upper=parse_float(
                row.get("gp_gray_zone_retry_upper"),
                default=DEFAULT_GP_GRAY_ZONE_UPPER,
            ),
            predicted_attack=parse_optional_bool(
                row.get("predicted_attack")
            ),
            actual_attack=parse_optional_bool(
                row.get("actual_attack")
            ),
            retry_attempts=parse_int(
                row.get("retry_attempts"),
                default=1,
            ),
            retry_used=parse_bool(
                row.get("retry_used")
            ),
            max_attempts=parse_int(
                row.get("max_attempts"),
                default=DEFAULT_MAX_ATTEMPTS,
            ),
            physical_qubits=parse_optional_int(
                row.get("physical_qubits")
            ),
            logical_payload_blocks=parse_optional_int(
                row.get("logical_payload_blocks")
            ),
            logical_check_blocks=parse_optional_int(
                row.get("logical_check_blocks")
            ),
            use_css=parse_bool(
                row.get("use_css"),
                default=True,
            ),
            bootstrap_mode=row.get(
                "bootstrap_mode",
                "mlkem",
            ),
            decision_mode=row.get(
                "decision_mode",
                "gp",
            ),
            credential_and_kem_keygen_s=(
                parse_optional_float(
                    row.get(
                        "credential_and_kem_keygen_s"
                    )
                )
            ),
            mlkem_encapsulation_s=parse_optional_float(
                row.get("mlkem_encapsulation_s")
            ),
            mlkem_decapsulation_s=parse_optional_float(
                row.get("mlkem_decapsulation_s")
            ),
            tag_and_schedule_s=parse_optional_float(
                row.get("tag_and_schedule_s")
            ),
            css_encoding_s=parse_optional_float(
                row.get("css_encoding_s")
            ),
            quantum_channel_simulation_s=(
                parse_optional_float(
                    row.get(
                        "quantum_channel_simulation_s"
                    )
                )
            ),
            measurement_and_css_decoding_s=(
                parse_optional_float(
                    row.get(
                        "measurement_and_css_decoding_s"
                    )
                )
            ),
            gp_inference_s=parse_optional_float(
                row.get("gp_inference_s")
            ),
            end_to_end_s=parse_optional_float(
                row.get("end_to_end_s")
            ),
            total_retry_end_to_end_s=(
                parse_optional_float(
                    row.get(
                        "total_retry_end_to_end_s"
                    )
                )
            ),
            attempt_history=parse_json_list(
                row.get("attempt_history")
            ),
            metadata=parse_json_dictionary(
                row.get("metadata")
            ),
        )

    @classmethod
    def from_protocol_result(
        cls,
        protocol_result: dict[str, Any],
        session_id: str | None = None,
        result_id: str | None = None,
        scenario_name: str | None = None,
        protocol_variant: str = (
            DEFAULT_PROTOCOL_VARIANT
        ),
        seed: int | None = None,
        actual_attack: bool | None = None,
        raw_calibration_threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AuthenticationResultRecord":
        """
        Convert a notebook-style FT-QuPAP result into a compact record.

        Raw keys, tags, ciphertexts, nonces, transcript hashes, encrypted
        schedules, and decoder bit patterns are deliberately excluded.
        """

        if not isinstance(protocol_result, dict):
            raise TypeError(
                "protocol_result must be a dictionary."
            )

        request = protocol_result.get("request") or {}
        decision = protocol_result.get("decision") or {}
        features = protocol_result.get("features") or {}
        timings = protocol_result.get("timings") or {}
        channel = protocol_result.get("channel") or {}

        normalized_session_id = (
            normalize_required_string(
                "session_id",
                session_id,
            )
            if session_id is not None
            else generate_session_id()
        )

        normalized_result_id = (
            normalize_required_string(
                "result_id",
                result_id,
            )
            if result_id is not None
            else generate_result_id()
        )

        accepted = bool(
            decision.get("accepted", False)
        )

        reason = str(
            decision.get("reason", "unspecified")
        )

        deterministic_reasons = [
            str(value)
            for value in decision.get(
                "deterministic_reasons",
                [],
            )
        ]

        retry_attempts = int(
            protocol_result.get(
                "retry_attempts",
                1,
            )
        )

        retry_used = bool(
            protocol_result.get(
                "retry_used",
                retry_attempts > 1,
            )
        )

        operational_threshold = finite_or_none(
            decision.get("gp_attack_threshold")
        )

        p_attack = finite_or_none(
            decision.get("p_attack")
        )

        predicted_attack = (
            bool(p_attack >= operational_threshold)
            if (
                p_attack is not None
                and operational_threshold is not None
            )
            else None
        )

        schedule_reason = protocol_result.get(
            "schedule_reason"
        )

        schedule_valid = (
            schedule_reason == "schedule_valid"
            if schedule_reason is not None
            else None
        )

        tag_recovered = infer_tag_recovery(
            protocol_result
        )

        decoder_records = protocol_result.get(
            "decoder_records"
        ) or []

        corrected_block_count = sum(
            bool(
                record.get("corrected", False)
                or record.get(
                    "correction_applied",
                    False,
                )
            )
            for record in decoder_records
            if isinstance(record, dict)
        )

        uncorrectable_block_count = sum(
            not bool(
                record.get(
                    "recoverable",
                    record.get(
                        "correctable",
                        True,
                    ),
                )
            )
            for record in decoder_records
            if isinstance(record, dict)
        )

        payload_failures = protocol_result.get(
            "payload_failures"
        ) or []

        resolved_scenario_name = (
            scenario_name
            or channel.get("name")
            or "normal_session"
        )

        context = str(
            channel.get(
                "context",
                infer_context_from_features(features),
            )
        )

        outcome = derive_outcome(
            accepted=accepted,
            reason=reason,
            deterministic_reasons=(
                deterministic_reasons
            ),
            retry_attempts=retry_attempts,
        )

        return cls(
            result_id=normalized_result_id,
            session_id=normalized_session_id,
            protocol_variant=protocol_variant,
            pseudonym_id=normalize_optional_string(
                request.get("pseudonym_id")
            ),
            scenario_name=str(
                resolved_scenario_name
            ),
            channel_name=str(
                channel.get("name", "unknown")
            ),
            context=context,
            seed=seed,
            accepted=accepted,
            outcome=outcome,
            reason=reason,
            deterministic_pass=bool(
                decision.get(
                    "deterministic_pass",
                    False,
                )
            ),
            deterministic_reasons=(
                deterministic_reasons
            ),
            schedule_valid=schedule_valid,
            tag_recovered=tag_recovered,
            payload_failure_count=len(
                payload_failures
            ),
            corrected_block_count=(
                corrected_block_count
            ),
            uncorrectable_block_count=(
                uncorrectable_block_count
            ),
            qber_raw=finite_or_none(
                protocol_result.get("qber_raw")
            ),
            qber_mismatches=integer_or_none(
                protocol_result.get(
                    "qber_mismatches"
                )
            ),
            qber_observed=integer_or_none(
                protocol_result.get(
                    "qber_observed"
                )
            ),
            observed_check_blocks=integer_or_none(
                protocol_result.get(
                    "observed_check_blocks"
                )
            ),
            required_check_blocks=int(
                protocol_result.get(
                    "required_check_blocks",
                    DEFAULT_REQUIRED_CHECK_BLOCKS,
                )
            ),
            loss_rate=finite_or_none(
                protocol_result.get("loss_rate")
            ),
            mean_syndrome_weight=finite_or_none(
                features.get(
                    "mean_syndrome_weight"
                )
            ),
            max_syndrome_weight=finite_or_none(
                features.get(
                    "max_syndrome_weight"
                )
            ),
            correction_failure_rate=finite_or_none(
                features.get(
                    "correction_failure_rate"
                )
            ),
            noise_estimate=finite_or_none(
                features.get("noise_estimate")
            ),
            ctx_urban=float(
                features.get("ctx_urban", 0.0)
            ),
            ctx_suburban=float(
                features.get("ctx_suburban", 0.0)
            ),
            ctx_rural=float(
                features.get("ctx_rural", 0.0)
            ),
            p_attack=p_attack,
            uncertainty=finite_or_none(
                decision.get("uncertainty")
            ),
            raw_calibration_gp_attack_threshold=(
                finite_or_none(
                    raw_calibration_threshold
                )
            ),
            gp_attack_threshold=(
                operational_threshold
            ),
            predicted_attack=predicted_attack,
            actual_attack=actual_attack,
            retry_attempts=retry_attempts,
            retry_used=retry_used,
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            physical_qubits=integer_or_none(
                protocol_result.get(
                    "physical_qubits"
                )
            ),
            logical_payload_blocks=integer_or_none(
                protocol_result.get(
                    "logical_payload_blocks"
                )
            ),
            logical_check_blocks=integer_or_none(
                protocol_result.get(
                    "logical_check_blocks"
                )
            ),
            use_css=bool(
                protocol_result.get(
                    "use_css",
                    True,
                )
            ),
            bootstrap_mode=str(
                protocol_result.get(
                    "bootstrap_mode",
                    "mlkem",
                )
            ),
            decision_mode=str(
                protocol_result.get(
                    "decision_mode",
                    "gp",
                )
            ),
            credential_and_kem_keygen_s=(
                finite_or_none(
                    timings.get(
                        "credential_and_kem_keygen_s"
                    )
                )
            ),
            mlkem_encapsulation_s=finite_or_none(
                timings.get(
                    "mlkem_encapsulation_s"
                )
            ),
            mlkem_decapsulation_s=finite_or_none(
                timings.get(
                    "mlkem_decapsulation_s"
                )
            ),
            tag_and_schedule_s=finite_or_none(
                timings.get(
                    "tag_and_schedule_s"
                )
            ),
            css_encoding_s=finite_or_none(
                timings.get("css_encoding_s")
            ),
            quantum_channel_simulation_s=(
                finite_or_none(
                    timings.get(
                        "quantum_channel_simulation_s"
                    )
                )
            ),
            measurement_and_css_decoding_s=(
                finite_or_none(
                    timings.get(
                        "measurement_and_css_decoding_s"
                    )
                )
            ),
            gp_inference_s=finite_or_none(
                timings.get("gp_inference_s")
            ),
            end_to_end_s=finite_or_none(
                timings.get("end_to_end_s")
            ),
            total_retry_end_to_end_s=(
                finite_or_none(
                    timings.get(
                        "total_retry_end_to_end_s"
                    )
                )
            ),
            attempt_history=copy.deepcopy(
                protocol_result.get(
                    "attempt_history",
                    [],
                )
            ),
            metadata=copy.deepcopy(metadata or {}),
        )


class ResultRepository:
    """
    Thread-safe CSV repository for FT-QuPAP results.

    CSV storage is used because dashboard and evaluation scripts can
    directly load the result table using pandas without processing a
    separate database format.
    """

    def __init__(
        self,
        database_path: str | Path = (
            DEFAULT_RESULT_DATABASE_PATH
        ),
    ) -> None:
        """Initialize or load the result repository."""

        self._database_path = Path(database_path)
        self._lock = threading.RLock()

        self._records_by_id: dict[
            str,
            AuthenticationResultRecord,
        ] = {}

        self._result_ids_by_session: dict[
            str,
            list[str],
        ] = {}

        self._initialize_repository()

    @property
    def database_path(self) -> Path:
        """Return the CSV repository path."""

        return self._database_path

    def _initialize_repository(self) -> None:
        """Create the CSV file or load existing results."""

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
        """Load all authentication results from CSV."""

        with self._lock:
            try:
                with self._database_path.open(
                    "r",
                    encoding="utf-8",
                    newline="",
                ) as csv_file:
                    reader = csv.DictReader(csv_file)

                    records_by_id: dict[
                        str,
                        AuthenticationResultRecord,
                    ] = {}

                    result_ids_by_session: dict[
                        str,
                        list[str],
                    ] = {}

                    for raw_row in reader:
                        record = (
                            AuthenticationResultRecord
                            .from_csv_row(raw_row)
                        )

                        if (
                            record.result_id
                            in records_by_id
                        ):
                            raise ValueError(
                                "Duplicate result_id detected: "
                                f"{record.result_id}"
                            )

                        records_by_id[
                            record.result_id
                        ] = record

                        result_ids_by_session.setdefault(
                            record.session_id,
                            [],
                        ).append(record.result_id)

            except csv.Error as error:
                raise ValueError(
                    "Result repository contains "
                    "invalid CSV data."
                ) from error

            except OSError as error:
                raise OSError(
                    "Unable to read the result repository."
                ) from error

            self._records_by_id = records_by_id
            self._result_ids_by_session = (
                result_ids_by_session
            )

    def reload(self) -> None:
        """Reload all result records from disk."""

        self._load()

    def _save(self) -> None:
        """Persist results through atomic CSV replacement."""

        with self._lock:
            field_names = list(
                AuthenticationResultRecord(
                    result_id="TEMP",
                    session_id="TEMP",
                ).to_dictionary().keys()
            )

            temporary_file_path: Path | None = None

            try:
                file_descriptor, temporary_name = (
                    tempfile.mkstemp(
                        prefix=(
                            f"{self._database_path.name}."
                        ),
                        suffix=".tmp",
                        dir=str(
                            self._database_path.parent
                        ),
                    )
                )

                temporary_file_path = Path(
                    temporary_name
                )

                with os.fdopen(
                    file_descriptor,
                    "w",
                    encoding="utf-8",
                    newline="",
                ) as temporary_file:
                    writer = csv.DictWriter(
                        temporary_file,
                        fieldnames=field_names,
                        extrasaction="ignore",
                    )

                    writer.writeheader()

                    for record in sorted(
                        self._records_by_id.values(),
                        key=lambda item: (
                            item.created_at,
                            item.result_id,
                        ),
                    ):
                        writer.writerow(
                            record.to_csv_row()
                        )

                    temporary_file.flush()
                    os.fsync(
                        temporary_file.fileno()
                    )

                os.replace(
                    temporary_file_path,
                    self._database_path,
                )

            except OSError as error:
                if (
                    temporary_file_path is not None
                    and temporary_file_path.exists()
                ):
                    temporary_file_path.unlink(
                        missing_ok=True
                    )

                raise OSError(
                    "Unable to save the result repository."
                ) from error

    def add_result(
        self,
        record: AuthenticationResultRecord,
    ) -> AuthenticationResultRecord:
        """Store a validated authentication result."""

        if not isinstance(
            record,
            AuthenticationResultRecord,
        ):
            raise TypeError(
                "record must be an "
                "AuthenticationResultRecord."
            )

        with self._lock:
            if (
                record.result_id
                in self._records_by_id
            ):
                raise ValueError(
                    "Result already exists: "
                    f"{record.result_id}"
                )

            self._insert_into_memory(record)

            try:
                self._save()
            except Exception:
                self._remove_from_memory(record)
                raise

            return record

    def save_protocol_result(
        self,
        protocol_result: dict[str, Any],
        session_id: str | None = None,
        result_id: str | None = None,
        scenario_name: str | None = None,
        protocol_variant: str = (
            DEFAULT_PROTOCOL_VARIANT
        ),
        seed: int | None = None,
        actual_attack: bool | None = None,
        raw_calibration_threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuthenticationResultRecord:
        """
        Convert and save a full notebook-style protocol result.
        """

        record = (
            AuthenticationResultRecord
            .from_protocol_result(
                protocol_result=protocol_result,
                session_id=session_id,
                result_id=result_id,
                scenario_name=scenario_name,
                protocol_variant=protocol_variant,
                seed=seed,
                actual_attack=actual_attack,
                raw_calibration_threshold=(
                    raw_calibration_threshold
                ),
                metadata=metadata,
            )
        )

        return self.add_result(record)

    def _insert_into_memory(
        self,
        record: AuthenticationResultRecord,
    ) -> None:
        """Insert a result into repository indexes."""

        self._records_by_id[
            record.result_id
        ] = record

        self._result_ids_by_session.setdefault(
            record.session_id,
            [],
        ).append(record.result_id)

    def _remove_from_memory(
        self,
        record: AuthenticationResultRecord,
    ) -> None:
        """Remove a result from repository indexes."""

        self._records_by_id.pop(
            record.result_id,
            None,
        )

        result_ids = self._result_ids_by_session.get(
            record.session_id,
            [],
        )

        if record.result_id in result_ids:
            result_ids.remove(record.result_id)

        if not result_ids:
            self._result_ids_by_session.pop(
                record.session_id,
                None,
            )

    def get_result(
        self,
        result_id: str,
    ) -> AuthenticationResultRecord | None:
        """Return a result by its repository identifier."""

        normalized_id = normalize_required_string(
            "result_id",
            result_id,
        )

        with self._lock:
            return self._records_by_id.get(
                normalized_id
            )

    def require_result(
        self,
        result_id: str,
    ) -> AuthenticationResultRecord:
        """Return a result or raise LookupError."""

        record = self.get_result(result_id)

        if record is None:
            raise LookupError(
                f"Result '{result_id}' was not found."
            )

        return record

    def get_by_session_id(
        self,
        session_id: str,
    ) -> list[AuthenticationResultRecord]:
        """Return every result associated with a session."""

        normalized_session_id = (
            normalize_required_string(
                "session_id",
                session_id,
            )
        )

        with self._lock:
            result_ids = list(
                self._result_ids_by_session.get(
                    normalized_session_id,
                    [],
                )
            )

            return [
                self._records_by_id[result_id]
                for result_id in result_ids
            ]

    def list_results(
        self,
        accepted: bool | None = None,
        outcome: str | None = None,
        scenario_name: str | None = None,
        context: str | None = None,
        protocol_variant: str | None = None,
    ) -> list[AuthenticationResultRecord]:
        """List result records using optional filters."""

        if accepted is not None:
            validate_boolean("accepted", accepted)

        normalized_outcome = (
            validate_outcome(outcome)
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
            validate_context(context)
            if context is not None
            else None
        )

        normalized_variant = (
            normalize_required_string(
                "protocol_variant",
                protocol_variant,
            )
            if protocol_variant is not None
            else None
        )

        with self._lock:
            records: Iterable[
                AuthenticationResultRecord
            ] = self._records_by_id.values()

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
                    if record.outcome
                    == normalized_outcome
                )

            if normalized_scenario is not None:
                records = (
                    record
                    for record in records
                    if record.scenario_name
                    == normalized_scenario
                )

            if normalized_context is not None:
                records = (
                    record
                    for record in records
                    if record.context
                    == normalized_context
                )

            if normalized_variant is not None:
                records = (
                    record
                    for record in records
                    if record.protocol_variant
                    == normalized_variant
                )

            return sorted(
                records,
                key=lambda record: (
                    record.created_at,
                    record.result_id,
                ),
                reverse=True,
            )

    def count_results(
        self,
        accepted: bool | None = None,
        outcome: str | None = None,
    ) -> int:
        """Return the number of matching results."""

        return len(
            self.list_results(
                accepted=accepted,
                outcome=outcome,
            )
        )

    def get_recent_results(
        self,
        limit: int = 20,
    ) -> list[AuthenticationResultRecord]:
        """Return the most recent result records."""

        validate_nonnegative_integer(
            "limit",
            limit,
        )

        if limit < 1:
            raise ValueError(
                "limit must be greater than zero."
            )

        return self.list_results()[:limit]

    def get_summary(self) -> dict[str, Any]:
        """Return dashboard-friendly result statistics."""

        with self._lock:
            records = list(
                self._records_by_id.values()
            )

        total = len(records)

        accepted = sum(
            record.accepted
            for record in records
        )

        rejected = total - accepted

        accepted_after_retry = sum(
            record.outcome
            == "accepted_after_retry"
            for record in records
        )

        retry_used = sum(
            record.retry_used
            for record in records
        )

        deterministic_failures = sum(
            not record.deterministic_pass
            for record in records
        )

        gp_rejections = sum(
            record.outcome == "rejected_gp"
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
            (
                record.total_retry_end_to_end_s
                if (
                    record.total_retry_end_to_end_s
                    is not None
                )
                else record.end_to_end_s
            )
            for record in records
        ]

        runtime_values = [
            value
            for value in runtime_values
            if value is not None
        ]

        return {
            "total_results": total,
            "accepted_results": accepted,
            "rejected_results": rejected,
            "accepted_after_retry":
                accepted_after_retry,
            "retry_used_results": retry_used,
            "deterministic_failures":
                deterministic_failures,
            "gp_rejections": gp_rejections,
            "acceptance_rate": (
                accepted / total
                if total
                else 0.0
            ),
            "average_qber_raw": safe_mean(
                qber_values
            ),
            "average_attack_probability": safe_mean(
                attack_probabilities
            ),
            "average_end_to_end_s": safe_mean(
                runtime_values
            ),
        }

    def get_confusion_matrix(self) -> dict[str, int]:
        """
        Calculate GP confusion-matrix values.

        Only records containing both actual_attack and predicted_attack
        are included.
        """

        records = [
            record
            for record in self.list_results()
            if (
                record.actual_attack is not None
                and record.predicted_attack is not None
            )
        ]

        true_positive = sum(
            record.actual_attack is True
            and record.predicted_attack is True
            for record in records
        )

        true_negative = sum(
            record.actual_attack is False
            and record.predicted_attack is False
            for record in records
        )

        false_positive = sum(
            record.actual_attack is False
            and record.predicted_attack is True
            for record in records
        )

        false_negative = sum(
            record.actual_attack is True
            and record.predicted_attack is False
            for record in records
        )

        return {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "evaluated_records": len(records),
        }

    def delete_result(
        self,
        result_id: str,
    ) -> bool:
        """Delete one stored result record."""

        normalized_id = normalize_required_string(
            "result_id",
            result_id,
        )

        with self._lock:
            record = self._records_by_id.get(
                normalized_id
            )

            if record is None:
                return False

            self._remove_from_memory(record)

            try:
                self._save()
            except Exception:
                self._insert_into_memory(record)
                raise

            return True

    def clear(self) -> None:
        """
        Remove every result.

        Intended for controlled demo resets and automated tests.
        """

        with self._lock:
            previous_records = dict(
                self._records_by_id
            )

            previous_session_index = copy.deepcopy(
                self._result_ids_by_session
            )

            self._records_by_id.clear()
            self._result_ids_by_session.clear()

            try:
                self._save()
            except Exception:
                self._records_by_id = previous_records
                self._result_ids_by_session = (
                    previous_session_index
                )
                raise


def generate_result_id() -> str:
    """Generate a unique result identifier."""

    return f"RESULT-{uuid.uuid4().hex.upper()}"


def generate_session_id() -> str:
    """Generate a unique session identifier."""

    return f"SESSION-{uuid.uuid4().hex.upper()}"


def derive_outcome(
    accepted: bool,
    reason: str,
    deterministic_reasons: list[str],
    retry_attempts: int,
) -> str:
    """Convert protocol reasons into a dashboard result category."""

    normalized_reason = reason.strip().lower()

    normalized_deterministic_reasons = {
        value.strip().lower()
        for value in deterministic_reasons
    }

    combined_reasons = {
        normalized_reason,
        *normalized_deterministic_reasons,
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
        "replay" in value
        or "nonce" in value
        for value in combined_reasons
    ):
        return "rejected_replay"

    if any(
        "credential" in value
        or "signature" in value
        or "trust_anchor" in value
        for value in combined_reasons
    ):
        return "rejected_credential"

    if any(
        "ciphertext" in value
        or "decapsulation" in value
        or "session_secret" in value
        for value in combined_reasons
    ):
        return "rejected_ciphertext"

    if (
        normalized_reason
        == "rejected_by_calibrated_bayesian_policy"
        or "gp" in normalized_reason
    ):
        return "rejected_gp"

    if retry_attempts >= DEFAULT_MAX_ATTEMPTS:
        if any(
            value in {
                "payload_block_unrecoverable",
                "authentication_tag_mismatch",
            }
            for value in combined_reasons
        ):
            return "rejected_retry_exhausted"

    if deterministic_reasons:
        return "rejected_deterministic"

    if "fail" in normalized_reason:
        return "failed"

    return "rejected"


def infer_tag_recovery(
    protocol_result: dict[str, Any],
) -> bool | None:
    """
    Compare recovered and expected tags without storing either tag.
    """

    received_tag = protocol_result.get(
        "received_tag"
    )

    expected_tag = protocol_result.get(
        "expected_tag"
    )

    if received_tag is None or expected_tag is None:
        return None

    if not isinstance(
        received_tag,
        (bytes, bytearray),
    ):
        return None

    if not isinstance(
        expected_tag,
        (bytes, bytearray),
    ):
        return None

    return hmac.compare_digest(
        bytes(received_tag),
        bytes(expected_tag),
    )


def infer_context_from_features(
    features: dict[str, Any],
) -> str:
    """Infer the context from one-hot GP features."""

    context_scores = {
        "urban": finite_or_none(
            features.get("ctx_urban")
        )
        or 0.0,
        "suburban": finite_or_none(
            features.get("ctx_suburban")
        )
        or 0.0,
        "rural": finite_or_none(
            features.get("ctx_rural")
        )
        or 0.0,
    }

    selected_context = max(
        context_scores,
        key=context_scores.get,
    )

    if context_scores[selected_context] <= 0.0:
        return "unknown"

    return selected_context


def safe_mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean or zero for an empty sequence."""

    normalized_values = [
        float(value)
        for value in values
        if finite_or_none(value) is not None
    ]

    if not normalized_values:
        return 0.0

    return sum(normalized_values) / len(
        normalized_values
    )


def finite_or_none(value: Any) -> float | None:
    """Convert finite numeric values to float."""

    if value is None or isinstance(value, bool):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def integer_or_none(value: Any) -> int | None:
    """Convert a valid integer-like value to int."""

    if value is None or isinstance(value, bool):
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric_value):
        return None

    if not numeric_value.is_integer():
        return None

    return int(numeric_value)


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

    if not isinstance(value, str):
        raise TypeError(
            "Optional value must be a string or None."
        )

    normalized_value = value.strip()

    return normalized_value or None


def normalize_string_list(
    name: str,
    values: list[Any],
) -> list[str]:
    """Normalize a list of strings."""

    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list.")

    return [
        normalize_required_string(
            f"{name} item",
            str(value),
        )
        for value in values
    ]


def validate_context(context: str) -> str:
    """Validate a channel context."""

    normalized_context = normalize_required_string(
        "context",
        context,
    ).lower()

    if normalized_context not in SUPPORTED_CONTEXTS:
        raise ValueError(
            "context must be one of: "
            + ", ".join(sorted(SUPPORTED_CONTEXTS))
        )

    return normalized_context


def validate_outcome(outcome: str) -> str:
    """Validate a result outcome."""

    normalized_outcome = normalize_required_string(
        "outcome",
        outcome,
    ).lower()

    if normalized_outcome not in SUPPORTED_OUTCOMES:
        raise ValueError(
            "Unsupported authentication outcome: "
            f"{normalized_outcome}"
        )

    return normalized_outcome


def validate_boolean(
    name: str,
    value: bool,
) -> None:
    """Validate a required boolean."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")


def validate_optional_boolean(
    name: str,
    value: bool | None,
) -> None:
    """Validate an optional boolean."""

    if value is not None:
        validate_boolean(name, value)


def validate_nonnegative_integer(
    name: str,
    value: int,
) -> None:
    """Validate a nonnegative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )


def validate_optional_probability(
    name: str,
    value: float | None,
) -> None:
    """Validate an optional value in the range [0, 1]."""

    if value is None:
        return

    validate_optional_finite_number(name, value)

    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0."
        )


def validate_optional_finite_number(
    name: str,
    value: float | None,
) -> None:
    """Validate an optional finite numeric value."""

    if value is None:
        return

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric.")

    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def validate_optional_nonnegative_number(
    name: str,
    value: float | None,
) -> None:
    """Validate an optional nonnegative numeric value."""

    validate_optional_finite_number(name, value)

    if value is not None and float(value) < 0.0:
        raise ValueError(
            f"{name} cannot be negative."
        )


def normalize_timestamp(
    value: datetime | str,
) -> str:
    """Convert a datetime or ISO value into UTC ISO format."""

    if isinstance(value, str):
        normalized_value = value.strip()

        if normalized_value.endswith("Z"):
            normalized_value = (
                normalized_value[:-1] + "+00:00"
            )

        try:
            parsed_datetime = datetime.fromisoformat(
                normalized_value
            )
        except ValueError as error:
            raise ValueError(
                "Timestamp must use ISO 8601 format."
            ) from error

    elif isinstance(value, datetime):
        parsed_datetime = value

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


def empty_to_none(value: Any) -> str | None:
    """Convert an empty CSV value to None."""

    if value is None:
        return None

    normalized_value = str(value).strip()

    return normalized_value or None


def parse_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """Parse a required CSV boolean."""

    if value is None or str(value).strip() == "":
        return default

    normalized_value = str(value).strip().lower()

    if normalized_value in {"true", "1", "yes"}:
        return True

    if normalized_value in {"false", "0", "no"}:
        return False

    raise ValueError(
        f"Invalid boolean value: {value}"
    )


def parse_optional_bool(
    value: Any,
) -> bool | None:
    """Parse an optional CSV boolean."""

    if value is None or str(value).strip() == "":
        return None

    return parse_bool(value)


def parse_int(
    value: Any,
    default: int = 0,
) -> int:
    """Parse a required CSV integer."""

    if value is None or str(value).strip() == "":
        return default

    return int(value)


def parse_optional_int(
    value: Any,
) -> int | None:
    """Parse an optional CSV integer."""

    if value is None or str(value).strip() == "":
        return None

    return int(value)


def parse_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Parse a required CSV float."""

    if value is None or str(value).strip() == "":
        return default

    return float(value)


def parse_optional_float(
    value: Any,
) -> float | None:
    """Parse an optional CSV float."""

    if value is None or str(value).strip() == "":
        return None

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        return None

    return numeric_value


def parse_json_list(
    value: Any,
) -> list[Any]:
    """Parse a JSON list stored in CSV."""

    if value is None or str(value).strip() == "":
        return []

    parsed_value = json.loads(str(value))

    if not isinstance(parsed_value, list):
        raise ValueError(
            "Stored JSON value must be a list."
        )

    return parsed_value


def parse_json_dictionary(
    value: Any,
) -> dict[str, Any]:
    """Parse a JSON dictionary stored in CSV."""

    if value is None or str(value).strip() == "":
        return {}

    parsed_value = json.loads(str(value))

    if not isinstance(parsed_value, dict):
        raise ValueError(
            "Stored JSON value must be a dictionary."
        )

    return parsed_value


def run_self_test() -> None:
    """Run result conversion and persistence tests."""

    with tempfile.TemporaryDirectory() as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "dashboard_results.csv"
        )

        repository = ResultRepository(database_path)

        expected_tag = bytes.fromhex(
            "00112233445566778899aabbccddeeff"
        )

        protocol_result = {
            "request": {
                "pseudonym_id": "PID-6G-UE-0001",
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
            },
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
            "physical_qubits": 1120,
            "logical_payload_blocks": 128,
            "logical_check_blocks": 32,
            "use_css": True,
            "bootstrap_mode": "mlkem",
            "decision_mode": "gp",
            "timings": {
                "credential_and_kem_keygen_s": 0.01,
                "mlkem_encapsulation_s": 0.002,
                "mlkem_decapsulation_s": 0.002,
                "tag_and_schedule_s": 0.003,
                "css_encoding_s": 0.01,
                "quantum_channel_simulation_s": 0.02,
                "measurement_and_css_decoding_s": 0.01,
                "gp_inference_s": 0.001,
                "end_to_end_s": 0.06,
                "total_retry_end_to_end_s": 0.12,
            },
            "channel": {
                "name": "benign_noisy_session",
                "context": "urban",
            },
        }

        stored_record = repository.save_protocol_result(
            protocol_result=protocol_result,
            session_id="SESSION-0001",
            result_id="RESULT-0001",
            scenario_name="accept_after_retry",
            seed=9102,
            actual_attack=False,
            raw_calibration_threshold=0.12,
        )

        assert stored_record.accepted
        assert (
            stored_record.outcome
            == "accepted_after_retry"
        )
        assert stored_record.tag_recovered is True
        assert stored_record.retry_attempts == 2
        assert stored_record.retry_used is True
        assert stored_record.physical_qubits == 1120
        assert repository.count_results() == 1

        reloaded_repository = ResultRepository(
            database_path
        )

        reloaded_record = (
            reloaded_repository.require_result(
                "RESULT-0001"
            )
        )

        assert reloaded_record.accepted
        assert reloaded_record.p_attack == 0.04
        assert reloaded_record.qber_raw == 0.015625
        assert len(
            reloaded_record.attempt_history
        ) == 2

        summary = reloaded_repository.get_summary()

        assert summary["total_results"] == 1
        assert summary["accepted_results"] == 1
        assert summary["accepted_after_retry"] == 1

        confusion = (
            reloaded_repository.get_confusion_matrix()
        )

        assert confusion["true_negative"] == 1

        stored_text = database_path.read_text(
            encoding="utf-8"
        )

        assert expected_tag.hex() not in stored_text
        assert "received_tag" not in stored_text
        assert "expected_tag" not in stored_text

        print("Result repository self-test passed.")
        print(f"Database path: {database_path}")
        print(
            "Stored outcome: "
            f"{reloaded_record.outcome}"
        )
        print(
            "Retry attempts: "
            f"{reloaded_record.retry_attempts}"
        )


if __name__ == "__main__":
    run_self_test()