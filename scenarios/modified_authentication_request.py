"""
FT-QuPAP Modified Authentication Request Scenario
==================================================

Defines a controlled classical-message modification attack against the
initial FT-QuPAP v5.1 authentication request.

A legitimate Mobile Station normally prepares an authentication request
containing session-bound information such as:

- Pseudonymous mobile identity
- Session identifier
- Timestamp
- Authentication nonce
- Network identifier
- Protocol version

In this scenario, an attacker intercepts the request and changes one
selected field before it reaches the Authentication Server.

Supported demonstration targets:

- mobile_identifier
- session_identifier
- network_identifier
- protocol_version

Expected security behavior:

1. The Authentication Server receives the modified request.
2. Request structure and required fields are validated.
3. Subscriber, session, network, or protocol consistency is checked.
4. The modification is detected during deterministic processing.
5. Authentication stops before ML-KEM and quantum processing.

Expected result:

    REJECTED_MODIFIED_REQUEST

The failure must not continue to:

- Server-package generation
- ML-KEM encapsulation or decapsulation
- Session-key derivation
- KMAC tag generation
- Quantum payload preparation
- Steane encoding
- Quantum transmission
- QBER calculation
- Gaussian Process evaluation
- Retry processing

A modified classical request is a deterministic failure and must never
be treated as benign quantum-channel noise.
"""

from __future__ import annotations

from typing import Any, Final, Mapping

from config import ApplicationConfig, get_config

from .scenario_config import (
    ChannelProfile,
    EveAttackMode,
    EveProfile,
    ExpectedOutcome,
    NoiseModelName,
    RetryProfile,
    ScenarioCategory,
    ScenarioConfig,
    TamperingProfile,
    create_scenario_config,
)


SCENARIO_NAME: Final[str] = (
    "modified_authentication_request"
)

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Modified Authentication Request"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "A classical network attacker changes a session-bound field in "
    "the Mobile Station authentication request before it reaches the "
    "Authentication Server. Deterministic request validation must "
    "detect the inconsistency and reject the session."
)

DEFAULT_RANDOM_SEED: Final[int] = 9303

DEFAULT_TARGET_FIELD: Final[str] = (
    "network_identifier"
)

EXPECTED_DECISION: Final[str] = (
    "rejected_modified_request"
)

EXPECTED_REJECTION_STAGE: Final[str] = (
    "authentication_request_validation"
)


SUPPORTED_TARGET_FIELDS: Final[
    tuple[str, ...]
] = (
    "mobile_identifier",
    "session_identifier",
    "network_identifier",
    "protocol_version",
)


TARGET_FIELD_DESCRIPTIONS: Final[
    dict[str, str]
] = {
    "mobile_identifier": (
        "Replace the pseudonymous mobile identifier with an "
        "unrecognized or inconsistent value."
    ),
    "session_identifier": (
        "Replace the session identifier so it no longer matches "
        "the active authentication context."
    ),
    "network_identifier": (
        "Replace the expected FT-QuPAP network identifier with "
        "an unauthorized network value."
    ),
    "protocol_version": (
        "Replace the supported FT-QuPAP protocol version with "
        "an unsupported or inconsistent version."
    ),
}


TARGET_EXPECTED_REASONS: Final[
    dict[str, str]
] = {
    "mobile_identifier": (
        "subscriber_identifier_mismatch"
    ),
    "session_identifier": (
        "session_identifier_mismatch"
    ),
    "network_identifier": (
        "network_identifier_mismatch"
    ),
    "protocol_version": (
        "unsupported_protocol_version"
    ),
}


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    target_field: str = DEFAULT_TARGET_FIELD,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the modified-authentication-request scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible demonstration seed.

        target_field:
            Authentication-request field changed by the attacker.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated modified-request scenario configuration.
    """

    config = app_config or get_config()

    normalized_target = normalize_target_field(
        target_field
    )

    tampering = create_tampering_profile(
        normalized_target
    )

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.CLASSICAL_ATTACK,
        expected_outcome=ExpectedOutcome.REJECTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            # The quantum channel should never be reached because
            # request validation must reject the message first.
            noise_model=NoiseModelName.COMBINED.value,
            bit_flip_probability=0.002,
            phase_flip_probability=0.002,
            depolarizing_probability=0.003,
            measurement_error_probability=0.001,
            loss_rate=0.01,
            burst_error_probability=0.0,
            burst_length=0,
            forced_lost_check_blocks=0,
            forced_uncorrectable_blocks=0,
        ),
        eve=EveProfile(
            enabled=False,
            attack_mode=EveAttackMode.NONE.value,
            interaction_fraction=0.0,
            measurement_basis_error_probability=0.5,
            resend_error_probability=0.0,
            target_check_blocks=True,
            target_payload_blocks=True,
        ),
        tampering=tampering,
        retry=RetryProfile(
            enabled=False,
            force_retry_on_attempts=(),
            noise_multiplier_after_retry=1.0,
            loss_multiplier_after_retry=1.0,
            change_random_seed_per_attempt=False,
        ),
        deterministic_verification_expected=True,

        # No quantum evidence should exist after an early
        # deterministic request-validation failure.
        gp_evaluation_expected=False,
        notes=(
            "The original Mobile Station request is legitimate.",
            f"The attacker modifies the {normalized_target} field.",
            "The authentication nonce is not intentionally replayed.",
            "The server must validate all required request fields.",
            "Request modification must cause deterministic rejection.",
            "ML-KEM bootstrapping must not start after rejection.",
            "No KMAC authentication tag should be processed.",
            "No quantum payload should be transmitted.",
            "GP evaluation is unnecessary after early rejection.",
            "Retry is forbidden for a modified classical request.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "classical_request_modification"
            ),
            "attack_enabled": True,
            "attacker": (
                "classical_network_attacker"
            ),
            "attack_stage": (
                "authentication_request_transmission"
            ),
            "modified_field": normalized_target,
            "modification_description": (
                TARGET_FIELD_DESCRIPTIONS[
                    normalized_target
                ]
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_reason": (
                TARGET_EXPECTED_REASONS[
                    normalized_target
                ]
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "freshness_check_expected": True,
            "replay_check_expected": True,
            "subscriber_check_expected": True,
            "request_validation_expected": True,
            "deterministic_rejection_expected": True,
            "server_package_generation_expected": False,
            "mlkem_encapsulation_expected": False,
            "mlkem_decapsulation_expected": False,
            "session_key_derivation_expected": False,
            "kmac_generation_expected": False,
            "quantum_transmission_expected": False,
            "qber_evaluation_expected": False,
            "gp_evaluation_expected": False,
            "retry_expected": False,
        },
    )

    scenario.validate(config)

    return scenario


def create_tampering_profile(
    target_field: str,
) -> TamperingProfile:
    """
    Create the field-specific tampering switches.

    The generic ``modify_authentication_request`` switch remains active
    for every target. A second field-specific switch identifies the
    exact modified request element.
    """

    normalized_target = normalize_target_field(
        target_field
    )

    return TamperingProfile(
        replay_authentication_request=False,
        reuse_nonce=False,
        stale_timestamp=False,
        forge_server_signature=False,

        modify_authentication_request=True,

        tamper_mlkem_ciphertext=False,
        forge_kmac_tag=False,

        modify_mobile_identifier=(
            normalized_target
            == "mobile_identifier"
        ),
        modify_session_identifier=(
            normalized_target
            == "session_identifier"
        ),
        modify_network_identifier=(
            normalized_target
            == "network_identifier"
        ),
        modify_protocol_version=(
            normalized_target
            == "protocol_version"
        ),
        modify_control_schedule=False,
    )


def create_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    target_field: str = DEFAULT_TARGET_FIELD,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the modified-request scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        target_field=target_field,
        app_config=app_config,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    target_field: str = DEFAULT_TARGET_FIELD,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly attack information."""

    normalized_target = normalize_target_field(
        target_field
    )

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        target_field=normalized_target,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": (
                "classical_network_attacker"
            ),
            "attack_stage": (
                "authentication_request_transmission"
            ),
            "attack_operation": (
                "modify_authentication_request_field"
            ),
            "modified_field": normalized_target,
            "modification_description": (
                TARGET_FIELD_DESCRIPTIONS[
                    normalized_target
                ]
            ),
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_reason": (
                TARGET_EXPECTED_REASONS[
                    normalized_target
                ]
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "server_package_expected": False,
            "mlkem_processing_expected": False,
            "quantum_transmission_expected": False,
            "gp_evaluation_expected": False,
            "retry_allowed": False,
            "expected_detection_sources": (
                detection_sources_for_target(
                    normalized_target
                )
            ),
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    target_field: str = DEFAULT_TARGET_FIELD,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """
    Return effective modified-request conditions for one attempt.
    """

    config = app_config or get_config()

    normalized_target = normalize_target_field(
        target_field
    )

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        target_field=normalized_target,
        app_config=config,
    )

    attempt = scenario.for_attempt(
        attempt_number=attempt_number,
        app_config=config,
    )

    result = attempt.to_dictionary()

    result.update(
        {
            "request_received": True,
            "request_modified": True,
            "modified_field": normalized_target,
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_rejection_reason": (
                TARGET_EXPECTED_REASONS[
                    normalized_target
                ]
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "continue_to_server_package": False,
            "continue_to_mlkem": False,
            "continue_to_quantum_channel": False,
            "continue_to_gp_evaluation": False,
            "retry_allowed": False,
        }
    )

    return result


def create_modification_plan(
    *,
    target_field: str = DEFAULT_TARGET_FIELD,
    original_value_label: str = "original-value",
    modified_value_label: str = "modified-value",
) -> list[dict[str, Any]]:
    """
    Return the controlled request-modification demonstration plan.

    Only descriptive labels are stored. Real identifiers, nonces, and
    other potentially sensitive values are not included.
    """

    normalized_target = normalize_target_field(
        target_field
    )

    normalized_original_value = (
        validate_nonempty_string(
            "original_value_label",
            original_value_label,
        )
    )

    normalized_modified_value = (
        validate_nonempty_string(
            "modified_value_label",
            modified_value_label,
        )
    )

    if (
        normalized_original_value
        == normalized_modified_value
    ):
        raise ValueError(
            "The original and modified value labels "
            "must differ."
        )

    return [
        {
            "step": 1,
            "actor": "mobile_station",
            "operation": (
                "prepare_authentication_request"
            ),
            "target_field": normalized_target,
            "field_value": (
                normalized_original_value
            ),
            "expected_result": (
                "valid_original_request"
            ),
        },
        {
            "step": 2,
            "actor": (
                "classical_network_attacker"
            ),
            "operation": (
                "modify_request_field_in_transit"
            ),
            "target_field": normalized_target,
            "original_value": (
                normalized_original_value
            ),
            "modified_value": (
                normalized_modified_value
            ),
            "expected_result": (
                "request_content_changed"
            ),
        },
        {
            "step": 3,
            "actor": "authentication_server",
            "operation": (
                "validate_authentication_request"
            ),
            "expected_validation_result": "fail",
            "expected_reason": (
                TARGET_EXPECTED_REASONS[
                    normalized_target
                ]
            ),
            "expected_decision": EXPECTED_DECISION,
        },
        {
            "step": 4,
            "actor": "authentication_server",
            "operation": "stop_authentication",
            "server_package_generated": False,
            "mlkem_processing_started": False,
            "quantum_transmission_started": False,
            "retry_requested": False,
        },
    ]


def detection_sources_for_target(
    target_field: str,
) -> list[str]:
    """Return expected validation sources for a target field."""

    normalized_target = normalize_target_field(
        target_field
    )

    common_sources = [
        "canonical_request_serialization",
        "required_field_validation",
        "session_context_validation",
    ]

    target_sources = {
        "mobile_identifier": [
            "subscriber_database",
            "pseudonymous_identity_validation",
        ],
        "session_identifier": [
            "active_session_record",
            "session_identifier_binding",
        ],
        "network_identifier": [
            "configured_network_identifier",
            "network_context_validation",
        ],
        "protocol_version": [
            "supported_protocol_versions",
            "protocol_version_validation",
        ],
    }

    return (
        common_sources
        + target_sources[normalized_target]
    )


def validate_modified_request_result(
    result: Mapping[str, Any],
    *,
    target_field: str = DEFAULT_TARGET_FIELD,
) -> bool:
    """
    Validate whether the modified request was correctly rejected.

    A valid result must show:

    - Authentication rejection
    - Request modification or field mismatch detection
    - No ML-KEM processing
    - No quantum transmission
    - No GP evaluation
    - No retry
    """

    if not isinstance(result, Mapping):
        raise TypeError(
            "result must be a mapping."
        )

    normalized_target = normalize_target_field(
        target_field
    )

    decision = normalize_text(
        result.get(
            "decision",
            result.get("status"),
        )
    )

    reason = normalize_text(
        result.get(
            "reason",
            result.get("rejection_reason"),
        )
    )

    detected_field = normalize_text(
        result.get(
            "modified_field",
            result.get("mismatched_field"),
        )
    )

    modification_detected = read_boolean(
        result,
        (
            "modification_detected",
            "request_modified",
            "request_validation_failed",
        ),
        default=False,
    )

    request_validation_performed = read_boolean(
        result,
        (
            "request_validation_performed",
            "authentication_request_checked",
        ),
        default=False,
    )

    server_package_generated = read_boolean(
        result,
        (
            "server_package_generated",
            "server_response_generated",
        ),
        default=False,
    )

    mlkem_started = read_boolean(
        result,
        (
            "mlkem_processing_started",
            "mlkem_encapsulation_started",
            "mlkem_decapsulation_started",
        ),
        default=False,
    )

    quantum_started = read_boolean(
        result,
        (
            "quantum_transmission_started",
            "quantum_processing_started",
        ),
        default=False,
    )

    gp_started = read_boolean(
        result,
        (
            "gp_evaluation_started",
            "gp_prediction_performed",
        ),
        default=False,
    )

    retry_used = read_boolean(
        result,
        (
            "retry_used",
            "retry_requested",
        ),
        default=False,
    )

    accepted = read_boolean(
        result,
        ("accepted",),
        default=False,
    )

    valid_decisions = {
        "rejected",
        "rejected_modified_request",
        "rejected_request",
        "rejected_request_validation",
        "rejected_subscriber",
        "rejected_network",
        "rejected_protocol_version",
        "failed",
    }

    valid_reasons = {
        TARGET_EXPECTED_REASONS[
            normalized_target
        ],
        "modified_authentication_request",
        "request_integrity_failure",
        "authentication_request_modified",
        "request_field_mismatch",
    }

    rejected = decision in valid_decisions

    reason_matches = (
        reason in valid_reasons
        or (
            modification_detected
            and detected_field
            in {
                "",
                normalized_target,
            }
        )
    )

    validation_confirmed = (
        request_validation_performed
        or modification_detected
    )

    return (
        rejected
        and reason_matches
        and validation_confirmed
        and not accepted
        and not server_package_generated
        and not mlkem_started
        and not quantum_started
        and not gp_started
        and not retry_used
    )


def normalize_target_field(
    target_field: str,
) -> str:
    """Validate and normalize a modification target."""

    normalized_target = validate_nonempty_string(
        "target_field",
        target_field,
    ).lower().replace("-", "_").replace(
        " ",
        "_",
    )

    aliases = {
        "mobile_id": "mobile_identifier",
        "pseudonymous_identity": (
            "mobile_identifier"
        ),
        "pseudonymous_mobile_id": (
            "mobile_identifier"
        ),
        "session_id": "session_identifier",
        "network_id": "network_identifier",
        "version": "protocol_version",
    }

    normalized_target = aliases.get(
        normalized_target,
        normalized_target,
    )

    if normalized_target not in (
        SUPPORTED_TARGET_FIELDS
    ):
        raise ValueError(
            "Unsupported authentication-request target: "
            f"{normalized_target}. Supported targets: "
            + ", ".join(SUPPORTED_TARGET_FIELDS)
        )

    return normalized_target


def validate_nonempty_string(
    name: str,
    value: str,
) -> str:
    """Validate and return a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return normalized_value


def normalize_text(value: Any) -> str:
    """Normalize an optional value as lowercase text."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def read_boolean(
    mapping: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: bool,
) -> bool:
    """Read the first boolean-compatible mapping field."""

    for field_name in field_names:
        if field_name not in mapping:
            continue

        value = mapping[field_name]

        if isinstance(value, bool):
            return value

        if isinstance(value, int) and value in {
            0,
            1,
        }:
            return bool(value)

        if isinstance(value, str):
            normalized_value = (
                value.strip().lower()
            )

            if normalized_value in {
                "true",
                "yes",
                "1",
                "detected",
                "failed",
                "performed",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "not_detected",
                "passed",
                "not_performed",
            }:
                return False

    return default


MODIFIED_AUTHENTICATION_REQUEST_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run modified-request scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert (
        scenario.name
        == "modified_authentication_request"
    )

    assert (
        scenario.category
        == "classical_attack"
    )

    assert scenario.expected_outcome == "rejected"
    assert scenario.attack_enabled is True
    assert scenario.retry_expected is False

    assert scenario.eve.enabled is False

    assert (
        scenario.tampering
        .modify_authentication_request
        is True
    )

    assert (
        scenario.tampering
        .modify_network_identifier
        is True
    )

    assert (
        scenario.tampering.reuse_nonce
        is False
    )

    assert scenario.tampering.enabled is True
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    mobile_identifier_scenario = build_scenario(
        target_field="mobile_id",
        app_config=config,
    )

    assert (
        mobile_identifier_scenario
        .tampering
        .modify_mobile_identifier
        is True
    )

    session_scenario = build_scenario(
        target_field="session_id",
        app_config=config,
    )

    assert (
        session_scenario
        .tampering
        .modify_session_identifier
        is True
    )

    version_scenario = build_scenario(
        target_field="version",
        app_config=config,
    )

    assert (
        version_scenario
        .tampering
        .modify_protocol_version
        is True
    )

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.tampering
        .modify_authentication_request
        is True
    )

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    attack_plan = create_modification_plan()

    assert len(attack_plan) == 4

    assert (
        attack_plan[1]["target_field"]
        == "network_identifier"
    )

    assert (
        attack_plan[2]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_result = (
        validate_modified_request_result(
            {
                "decision": (
                    "rejected_modified_request"
                ),
                "reason": (
                    "network_identifier_mismatch"
                ),
                "modified_field": (
                    "network_identifier"
                ),
                "modification_detected": True,
                "request_validation_performed": True,
                "server_package_generated": False,
                "mlkem_processing_started": False,
                "quantum_transmission_started": False,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": False,
            }
        )
    )

    assert valid_result is True

    invalid_retry_result = (
        validate_modified_request_result(
            {
                "decision": (
                    "rejected_modified_request"
                ),
                "reason": (
                    "network_identifier_mismatch"
                ),
                "modification_detected": True,
                "request_validation_performed": True,
                "server_package_generated": False,
                "mlkem_processing_started": False,
                "quantum_transmission_started": False,
                "gp_evaluation_started": False,
                "retry_requested": True,
                "accepted": False,
            }
        )
    )

    assert invalid_retry_result is False

    invalid_continuation_result = (
        validate_modified_request_result(
            {
                "decision": (
                    "rejected_modified_request"
                ),
                "reason": (
                    "network_identifier_mismatch"
                ),
                "modification_detected": True,
                "request_validation_performed": True,
                "server_package_generated": True,
                "mlkem_processing_started": True,
                "quantum_transmission_started": False,
                "gp_evaluation_started": False,
                "retry_requested": False,
                "accepted": False,
            }
        )
    )

    assert invalid_continuation_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True

    assert (
        summary["modified_field"]
        == "network_identifier"
    )

    assert (
        summary["expected_decision"]
        == "rejected_modified_request"
    )

    assert (
        summary["quantum_transmission_expected"]
        is False
    )

    assert summary["retry_allowed"] is False

    print(
        "FT-QuPAP modified-authentication-request "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_TARGET_FIELD",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "SUPPORTED_TARGET_FIELDS",
    "TARGET_FIELD_DESCRIPTIONS",
    "TARGET_EXPECTED_REASONS",
    "MODIFIED_AUTHENTICATION_REQUEST_CONFIG",
    "build_scenario",
    "create_tampering_profile",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_modification_plan",
    "detection_sources_for_target",
    "validate_modified_request_result",
    "normalize_target_field",
]


if __name__ == "__main__":
    run_self_test()