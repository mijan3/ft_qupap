"""
FT-QuPAP Replay Attack Scenario
===============================

Defines a controlled replay attack against the classical authentication
request of the FT-QuPAP v5.1 protocol.

Attack process:

1. A legitimate Mobile Station previously sends a valid authentication
   request containing:

   - Pseudonymous mobile identity
   - Session identifier
   - Timestamp
   - Authentication nonce
   - Network identifier
   - Protocol version

2. The Authentication Server processes the original request and stores
   a secure digest of the used nonce.

3. An attacker captures and resubmits the previous authentication
   request.

4. The replayed request therefore contains a nonce that has already
   been recorded.

5. The Authentication Server detects the duplicated nonce during the
   deterministic replay-protection stage.

Expected result:

    REJECTED_REPLAY

The replay must be rejected before:

- ML-KEM decapsulation
- Session-key derivation
- Control-schedule processing
- Quantum payload decoding
- QBER calculation
- Gaussian Process evaluation
- Retry processing

A replay failure must never be converted into a retry.
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


SCENARIO_NAME: Final[str] = "replay_attack"

SCENARIO_DISPLAY_NAME: Final[str] = "Replay Attack"

SCENARIO_DESCRIPTION: Final[str] = (
    "An attacker resubmits a previously valid FT-QuPAP "
    "authentication request. The reused authentication nonce must be "
    "detected by the server before cryptographic bootstrapping or "
    "quantum authentication processing continues."
)

DEFAULT_RANDOM_SEED: Final[int] = 9301

EXPECTED_DECISION: Final[str] = "rejected_replay"

EXPECTED_REJECTION_STAGE: Final[str] = "replay_detection"

REPLAYED_REQUEST_NUMBER: Final[int] = 2


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the FT-QuPAP replay-attack scenario.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Reproducible simulation seed.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated replay-attack scenario configuration.
    """

    config = app_config or get_config()

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.CLASSICAL_ATTACK,
        expected_outcome=ExpectedOutcome.REJECTED,
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            # The quantum channel is not expected to be used because
            # replay detection should reject the request first.
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
        tampering=TamperingProfile(
            replay_authentication_request=True,
            reuse_nonce=True,

            # The replay may occur within the freshness window.
            # Therefore, nonce replay detection remains essential.
            stale_timestamp=False,

            forge_server_signature=False,
            modify_authentication_request=False,
            tamper_mlkem_ciphertext=False,
            forge_kmac_tag=False,
            modify_mobile_identifier=False,
            modify_session_identifier=False,
            modify_network_identifier=False,
            modify_protocol_version=False,
            modify_control_schedule=False,
        ),
        retry=RetryProfile(
            enabled=False,
            force_retry_on_attempts=(),
            noise_multiplier_after_retry=1.0,
            loss_multiplier_after_retry=1.0,
            change_random_seed_per_attempt=False,
        ),
        deterministic_verification_expected=True,

        # GP evaluation should not be required because the request
        # must be rejected during the early replay-protection stage.
        gp_evaluation_expected=False,
        notes=(
            "The original authentication request is legitimate.",
            "The replayed request reuses the original nonce.",
            "The request may still contain a fresh timestamp.",
            "The nonce database must detect the duplicate.",
            "Only a digest of the nonce should be persisted.",
            "Replay detection must occur before ML-KEM decapsulation.",
            "Quantum transmission must not start after replay failure.",
            "GP evaluation is unnecessary after deterministic rejection.",
            "Replay rejection must never receive a retry.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "classical_authentication_replay"
            ),
            "attack_enabled": True,
            "attack_stage": (
                "authentication_request_validation"
            ),
            "replayed_request_number": (
                REPLAYED_REQUEST_NUMBER
            ),
            "reuse_nonce": True,
            "stale_timestamp_required": False,
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "freshness_window_seconds": (
                config.protocol
                .freshness_window_seconds
            ),
            "replay_check_required": (
                config.protocol
                .replay_check_required
            ),
            "deterministic_rejection_expected": True,
            "mlkem_decapsulation_expected": False,
            "session_key_derivation_expected": False,
            "quantum_transmission_expected": False,
            "qber_evaluation_expected": False,
            "gp_evaluation_expected": False,
            "retry_expected": False,
        },
    )

    scenario.validate(config)

    return scenario


def create_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """Alias for building the replay-attack scenario."""

    return build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )


def get_scenario_summary(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """Return dashboard-friendly replay-attack information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "attacker": "classical_network_attacker",
            "attack_stage": (
                "authentication_request_validation"
            ),
            "attack_operation": (
                "resubmit_previous_authentication_request"
            ),
            "reused_value": "authentication_nonce",
            "expected_decision": EXPECTED_DECISION,
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "freshness_failure_required": False,
            "replay_detection_required": True,
            "mlkem_decapsulation_expected": False,
            "quantum_transmission_expected": False,
            "gp_evaluation_expected": False,
            "retry_allowed": False,
            "expected_detection_sources": [
                "stored_nonce_digest",
                "request_nonce_digest",
                "constant_time_digest_comparison",
                "nonce_usage_record",
            ],
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = 1,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """
    Return effective replay-attack conditions for one attempt.

    The scenario represents the replayed submission itself. The
    previously accepted original request is treated as prerequisite
    test data.
    """

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=config,
    )

    attempt = scenario.for_attempt(
        attempt_number=attempt_number,
        app_config=config,
    )

    result = attempt.to_dictionary()

    result.update(
        {
            "request_type": "replayed_request",
            "reuse_nonce": True,
            "expected_attempt_decision": (
                EXPECTED_DECISION
            ),
            "expected_rejection_stage": (
                EXPECTED_REJECTION_STAGE
            ),
            "continue_to_mlkem": False,
            "continue_to_quantum_channel": False,
            "continue_to_gp_evaluation": False,
            "retry_allowed": False,
        }
    )

    return result


def create_replay_plan(
    *,
    original_session_id: str = "original-session",
    replay_session_id: str = "replay-session",
) -> list[dict[str, Any]]:
    """
    Return the controlled two-step replay demonstration plan.

    This helper contains only labels and expected behavior. It does not
    contain a raw authentication nonce or other secret material.
    """

    normalized_original_session_id = validate_identifier(
        "original_session_id",
        original_session_id,
    )

    normalized_replay_session_id = validate_identifier(
        "replay_session_id",
        replay_session_id,
    )

    if (
        normalized_original_session_id
        == normalized_replay_session_id
    ):
        raise ValueError(
            "Original and replay session identifiers must differ."
        )

    return [
        {
            "step": 1,
            "session_id": normalized_original_session_id,
            "request_type": "original",
            "nonce_status_before_request": "unused",
            "expected_replay_check": "pass",
            "expected_nonce_action": (
                "store_nonce_digest"
            ),
            "expected_decision": (
                "continue_authentication"
            ),
        },
        {
            "step": 2,
            "session_id": normalized_replay_session_id,
            "request_type": "replayed",
            "nonce_status_before_request": "already_used",
            "expected_replay_check": "fail",
            "expected_nonce_action": (
                "do_not_store_duplicate"
            ),
            "expected_decision": EXPECTED_DECISION,
        },
    ]


def validate_replay_result(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate whether a protocol result correctly rejected the replay.

    A valid replay result must indicate:

    - Rejection
    - Replay or duplicate-nonce detection
    - No retry
    - No successful quantum-processing continuation
    """

    if not isinstance(result, Mapping):
        raise TypeError(
            "result must be a mapping."
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

    replay_detected = read_boolean(
        result,
        (
            "replay_detected",
            "duplicate_nonce",
            "nonce_already_used",
        ),
    )

    retry_used = read_boolean(
        result,
        (
            "retry_used",
            "retry_requested",
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

    accepted = read_boolean(
        result,
        ("accepted",),
        default=False,
    )

    rejection_decisions = {
        "rejected",
        "rejected_replay",
        "rejected_nonce",
        "rejected_duplicate_nonce",
        "failed",
    }

    replay_reasons = {
        "replay",
        "replay_detected",
        "duplicate_nonce",
        "nonce_already_used",
        "reused_nonce",
    }

    decision_rejected = (
        decision in rejection_decisions
    )

    reason_matches = (
        reason in replay_reasons
        or replay_detected
    )

    return (
        decision_rejected
        and reason_matches
        and not accepted
        and not retry_used
        and not quantum_started
    )


def validate_identifier(
    name: str,
    value: str,
) -> str:
    """Validate a non-empty demonstration identifier."""

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

    return str(value).strip().lower()


def read_boolean(
    mapping: Mapping[str, Any],
    field_names: tuple[str, ...],
    *,
    default: bool = False,
) -> bool:
    """
    Read the first boolean-compatible field from a mapping.
    """

    for field_name in field_names:
        if field_name not in mapping:
            continue

        value = mapping[field_name]

        if isinstance(value, bool):
            return value

        if isinstance(value, int) and value in {0, 1}:
            return bool(value)

        if isinstance(value, str):
            normalized_value = value.strip().lower()

            if normalized_value in {
                "true",
                "yes",
                "1",
                "detected",
            }:
                return True

            if normalized_value in {
                "false",
                "no",
                "0",
                "not_detected",
            }:
                return False

    return default


REPLAY_ATTACK_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run replay-attack scenario consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "replay_attack"

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
        .replay_authentication_request
        is True
    )

    assert (
        scenario.tampering.reuse_nonce
        is True
    )

    assert (
        scenario.tampering.stale_timestamp
        is False
    )

    assert scenario.tampering.enabled is True
    assert scenario.retry.enabled is False

    assert (
        scenario.gp_evaluation_expected
        is False
    )

    attempt = scenario.for_attempt(
        1,
        config,
    )

    assert (
        attempt.tampering
        .replay_authentication_request
        is True
    )

    assert attempt.tampering.reuse_nonce is True

    assert (
        attempt.force_retry_gray_zone
        is False
    )

    replay_plan = create_replay_plan()

    assert len(replay_plan) == 2

    assert (
        replay_plan[0]["expected_replay_check"]
        == "pass"
    )

    assert (
        replay_plan[1]["expected_replay_check"]
        == "fail"
    )

    assert (
        replay_plan[1]["expected_decision"]
        == EXPECTED_DECISION
    )

    valid_result = validate_replay_result(
        {
            "decision": "rejected_replay",
            "reason": "duplicate_nonce",
            "replay_detected": True,
            "retry_used": False,
            "quantum_transmission_started": False,
            "accepted": False,
        }
    )

    assert valid_result is True

    valid_generic_result = validate_replay_result(
        {
            "decision": "rejected",
            "rejection_reason": "reused_nonce",
            "nonce_already_used": True,
            "retry_requested": False,
            "quantum_processing_started": False,
            "accepted": False,
        }
    )

    assert valid_generic_result is True

    invalid_retry_result = validate_replay_result(
        {
            "decision": "rejected_replay",
            "replay_detected": True,
            "retry_used": True,
            "quantum_transmission_started": False,
            "accepted": False,
        }
    )

    assert invalid_retry_result is False

    invalid_accepted_result = validate_replay_result(
        {
            "decision": "accepted",
            "replay_detected": False,
            "retry_used": False,
            "quantum_transmission_started": True,
            "accepted": True,
        }
    )

    assert invalid_accepted_result is False

    summary = get_scenario_summary(
        app_config=config
    )

    assert summary["attack_enabled"] is True

    assert (
        summary["expected_decision"]
        == "rejected_replay"
    )

    assert summary["retry_allowed"] is False

    assert (
        summary["quantum_transmission_expected"]
        is False
    )

    print(
        "FT-QuPAP replay-attack scenario "
        "self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "EXPECTED_DECISION",
    "EXPECTED_REJECTION_STAGE",
    "REPLAYED_REQUEST_NUMBER",
    "REPLAY_ATTACK_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "create_replay_plan",
    "validate_replay_result",
]


if __name__ == "__main__":
    run_self_test()