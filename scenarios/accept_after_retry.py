"""
FT-QuPAP Accept-After-Retry Scenario
====================================

Defines a deterministic demonstration in which the first authentication
attempt enters the FT-QuPAP protected retry region and the second
attempt succeeds under improved benign channel conditions.

Scenario characteristics:

- Registered and legitimate Mobile Station
- Fresh timestamp and unique nonce
- Valid ML-DSA-65 server signature
- Valid ML-KEM-768 encapsulation and decapsulation
- Matching transcript-bound session keys
- Correct 128-bit KMAC256 authentication tag
- No eavesdropper
- No replay or classical-message modification
- No cryptographic forgery
- Deterministic verification passes
- First attempt contains elevated benign channel uncertainty
- First attempt is placed in the retry-compatible gray zone
- Second attempt uses fresh transmission evidence
- Second attempt has reduced noise and loss
- Retry count remains within the configured maximum

Expected result:

    ACCEPTED_AFTER_RETRY

The scenario does not allow retry to override deterministic failures.
A replay, stale timestamp, invalid signature, ML-KEM failure, invalid
KMAC tag, excessive loss, or uncorrectable quantum error must still
cause immediate rejection.
"""

from __future__ import annotations

from typing import Any, Final

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


SCENARIO_NAME: Final[str] = "accept_after_retry"

SCENARIO_DISPLAY_NAME: Final[str] = (
    "Accept After Retry"
)

SCENARIO_DESCRIPTION: Final[str] = (
    "A legitimate Mobile Station experiences elevated benign channel "
    "uncertainty during the first authentication attempt. The protocol "
    "requests one bounded retry, receives fresh quantum evidence under "
    "improved channel conditions, and accepts the second attempt."
)

DEFAULT_RANDOM_SEED: Final[int] = 9102

FIRST_ATTEMPT_NUMBER: Final[int] = 1
SUCCESSFUL_ATTEMPT_NUMBER: Final[int] = 2

FIRST_ATTEMPT_TARGET_GP_PROBABILITY: Final[float] = 0.17
SECOND_ATTEMPT_TARGET_GP_PROBABILITY: Final[float] = 0.08


def build_scenario(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> ScenarioConfig:
    """
    Build the FT-QuPAP accept-after-retry scenario.

    The first attempt is intentionally marked as a retry-gray-zone
    demonstration. Later attempts use reduced benign noise and loss.

    Args:
        context:
            Network environment such as urban, suburban, or rural.

        random_seed:
            Deterministic seed used for the first transmission.

        app_config:
            Optional FT-QuPAP application configuration.

    Returns:
        Validated accept-after-retry scenario configuration.
    """

    config = app_config or get_config()

    scenario = create_scenario_config(
        name=SCENARIO_NAME,
        display_name=SCENARIO_DISPLAY_NAME,
        description=SCENARIO_DESCRIPTION,
        category=ScenarioCategory.RETRY,
        expected_outcome=(
            ExpectedOutcome.ACCEPTED_AFTER_RETRY
        ),
        context=context,
        random_seed=random_seed,
        channel=ChannelProfile(
            noise_model=NoiseModelName.COMBINED.value,

            # Elevated benign uncertainty during attempt one.
            bit_flip_probability=0.018,
            phase_flip_probability=0.015,
            depolarizing_probability=0.028,
            measurement_error_probability=0.008,

            # Still below the protocol maximum loss rate.
            loss_rate=0.09,

            # Small correlated errors may appear, but no logical
            # block is intentionally forced to be uncorrectable.
            burst_error_probability=0.03,
            burst_length=2,

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
            replay_authentication_request=False,
            reuse_nonce=False,
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
            enabled=True,

            # Attempt one is deliberately kept inside the
            # retry-compatible low-risk gray zone.
            force_retry_on_attempts=(
                FIRST_ATTEMPT_NUMBER,
            ),

            # The retransmission models improved channel conditions.
            noise_multiplier_after_retry=0.35,
            loss_multiplier_after_retry=0.50,

            # Every retry must use fresh deterministic simulation
            # evidence rather than repeating the previous sequence.
            change_random_seed_per_attempt=True,
        ),
        deterministic_verification_expected=True,
        gp_evaluation_expected=True,
        notes=(
            "All deterministic verification stages must pass.",
            "The first attempt demonstrates the protected GP gray zone.",
            "The first attempt must not be treated as an attack.",
            "A retry uses a fresh nonce and fresh quantum transmission.",
            "The second attempt uses reduced benign noise and loss.",
            "Only one retry is expected for this controlled scenario.",
            "The final outcome should be accepted_after_retry.",
        ),
        metadata={
            "protocol_version": (
                config.protocol.protocol_version
            ),
            "scenario_type": (
                "controlled_benign_retry"
            ),
            "expected_retry_count": 1,
            "expected_total_attempts": 2,
            "successful_attempt_number": (
                SUCCESSFUL_ATTEMPT_NUMBER
            ),
            "first_attempt_target_gp_probability": (
                FIRST_ATTEMPT_TARGET_GP_PROBABILITY
            ),
            "second_attempt_target_gp_probability": (
                SECOND_ATTEMPT_TARGET_GP_PROBABILITY
            ),
            "deterministic_pass_expected": True,
            "retry_used_expected": True,
            "accepted_after_retry_expected": True,
            "ml_dsa_parameter_set": (
                config.cryptography
                .ml_dsa_parameter_set
            ),
            "ml_kem_parameter_set": (
                config.cryptography
                .ml_kem_parameter_set
            ),
            "kmac_algorithm": (
                config.cryptography.kmac_algorithm
            ),
            "kmac_tag_bits": (
                config.cryptography.kmac_tag_bits
            ),
            "steane_code": "[[7,1,3]]",
            "logical_payload_blocks": (
                config.quantum.logical_payload_blocks
            ),
            "independent_check_blocks": (
                config.quantum
                .independent_check_blocks
            ),
            "total_logical_blocks": (
                config.quantum.total_logical_blocks
            ),
            "total_physical_qubits": (
                config.quantum.total_physical_qubits
            ),
            "fixed_qber_threshold": (
                config.quantum.fixed_qber_threshold
            ),
            "maximum_loss_rate": (
                config.quantum.maximum_loss_rate
            ),
            "minimum_observed_check_blocks": (
                config.quantum
                .minimum_observed_check_blocks
            ),
            "minimum_gp_threshold": (
                config.machine_learning
                .minimum_operational_threshold
            ),
            "retry_upper_probability": (
                config.machine_learning
                .retry_upper_probability
            ),
            "maximum_authentication_attempts": (
                config.protocol
                .maximum_authentication_attempts
            ),
            "maximum_retries": (
                config.protocol.maximum_retries
            ),
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
    """Alias for building the accept-after-retry scenario."""

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
    """Return dashboard-friendly scenario information."""

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=app_config,
    )

    summary = scenario.dashboard_summary()

    summary.update(
        {
            "expected_behavior": (
                "Attempt one requests a bounded retry; "
                "attempt two is accepted."
            ),
            "first_attempt_decision": "retry",
            "second_attempt_decision": "accepted",
            "final_decision": "accepted_after_retry",
            "expected_retry_count": 1,
            "expected_total_attempts": 2,
            "deterministic_checks_required": True,
            "fresh_nonce_required_on_retry": True,
            "fresh_quantum_transmission_required": True,
            "active_attacker": False,
        }
    )

    return summary


def get_attempt_configuration(
    attempt_number: int = FIRST_ATTEMPT_NUMBER,
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> dict[str, Any]:
    """
    Return effective channel values for one authentication attempt.
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
        get_expected_attempt_outcome(
            attempt_number
        )
    )

    return result


def get_expected_attempt_outcome(
    attempt_number: int,
) -> dict[str, Any]:
    """
    Return the controlled expected behavior for an attempt.

    This information is used only for demonstration and test
    assertions. The real decision must still be generated by the
    protocol engines.
    """

    if not isinstance(attempt_number, int) or isinstance(
        attempt_number,
        bool,
    ):
        raise TypeError(
            "attempt_number must be an integer."
        )

    if attempt_number < 1:
        raise ValueError(
            "attempt_number must be greater than zero."
        )

    if attempt_number == FIRST_ATTEMPT_NUMBER:
        return {
            "expected_attempt_decision": "retry",
            "expected_final_decision": None,
            "expected_gp_probability": (
                FIRST_ATTEMPT_TARGET_GP_PROBABILITY
            ),
            "expected_deterministic_pass": True,
            "expected_retry_requested": True,
            "expected_retry_used": True,
        }

    if attempt_number == SUCCESSFUL_ATTEMPT_NUMBER:
        return {
            "expected_attempt_decision": "accepted",
            "expected_final_decision": (
                "accepted_after_retry"
            ),
            "expected_gp_probability": (
                SECOND_ATTEMPT_TARGET_GP_PROBABILITY
            ),
            "expected_deterministic_pass": True,
            "expected_retry_requested": False,
            "expected_retry_used": True,
        }

    return {
        "expected_attempt_decision": (
            "not_expected"
        ),
        "expected_final_decision": (
            "accepted_after_retry"
        ),
        "expected_gp_probability": None,
        "expected_deterministic_pass": True,
        "expected_retry_requested": False,
        "expected_retry_used": True,
    }


def get_retry_demonstration_plan(
    *,
    context: str = "urban",
    random_seed: int = DEFAULT_RANDOM_SEED,
    app_config: ApplicationConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Return the planned first and second attempt conditions.

    This function is useful for dashboard tables and protocol-monitor
    visualizations.
    """

    config = app_config or get_config()

    scenario = build_scenario(
        context=context,
        random_seed=random_seed,
        app_config=config,
    )

    plan: list[dict[str, Any]] = []

    for attempt_number in (
        FIRST_ATTEMPT_NUMBER,
        SUCCESSFUL_ATTEMPT_NUMBER,
    ):
        attempt = scenario.for_attempt(
            attempt_number=attempt_number,
            app_config=config,
        )

        expected = get_expected_attempt_outcome(
            attempt_number
        )

        plan.append(
            {
                "attempt_number": (
                    attempt.attempt_number
                ),
                "random_seed": attempt.random_seed,
                "context": attempt.context,
                "bit_flip_probability": (
                    attempt.channel
                    .bit_flip_probability
                ),
                "phase_flip_probability": (
                    attempt.channel
                    .phase_flip_probability
                ),
                "depolarizing_probability": (
                    attempt.channel
                    .depolarizing_probability
                ),
                "measurement_error_probability": (
                    attempt.channel
                    .measurement_error_probability
                ),
                "loss_rate": (
                    attempt.channel.loss_rate
                ),
                "force_retry_gray_zone": (
                    attempt.force_retry_gray_zone
                ),
                **expected,
            }
        )

    return plan


def validate_retry_transition(
    first_attempt_result: dict[str, Any],
    second_attempt_result: dict[str, Any],
) -> bool:
    """
    Validate the expected accept-after-retry result transition.

    Expected transition:

        attempt 1 -> retry
        attempt 2 -> accepted_after_retry
    """

    if not isinstance(first_attempt_result, dict):
        raise TypeError(
            "first_attempt_result must be a dictionary."
        )

    if not isinstance(second_attempt_result, dict):
        raise TypeError(
            "second_attempt_result must be a dictionary."
        )

    first_decision = str(
        first_attempt_result.get(
            "decision",
            first_attempt_result.get(
                "status",
                "",
            ),
        )
    ).strip().lower()

    second_decision = str(
        second_attempt_result.get(
            "decision",
            second_attempt_result.get(
                "status",
                "",
            ),
        )
    ).strip().lower()

    first_deterministic_pass = bool(
        first_attempt_result.get(
            "deterministic_pass",
            False,
        )
    )

    second_deterministic_pass = bool(
        second_attempt_result.get(
            "deterministic_pass",
            False,
        )
    )

    retry_used = bool(
        second_attempt_result.get(
            "retry_used",
            False,
        )
    )

    return (
        first_decision
        in {
            "retry",
            "retry_pending",
            "retry_requested",
        }
        and second_decision
        in {
            "accepted",
            "accepted_after_retry",
        }
        and first_deterministic_pass
        and second_deterministic_pass
        and retry_used
    )


ACCEPT_AFTER_RETRY_CONFIG: Final[
    ScenarioConfig
] = build_scenario()


def run_self_test() -> None:
    """Run accept-after-retry consistency checks."""

    config = get_config()

    scenario = build_scenario(
        app_config=config
    )

    assert scenario.name == "accept_after_retry"
    assert scenario.category == "retry"

    assert (
        scenario.expected_outcome
        == "accepted_after_retry"
    )

    assert scenario.attack_enabled is False
    assert scenario.retry_expected is True

    assert scenario.eve.enabled is False
    assert scenario.tampering.enabled is False

    assert scenario.retry.enabled is True

    assert (
        scenario.retry.force_retry_on_attempts
        == (1,)
    )

    first_attempt = scenario.for_attempt(
        FIRST_ATTEMPT_NUMBER,
        config,
    )

    second_attempt = scenario.for_attempt(
        SUCCESSFUL_ATTEMPT_NUMBER,
        config,
    )

    assert first_attempt.attempt_number == 1
    assert second_attempt.attempt_number == 2

    assert first_attempt.random_seed == 9102
    assert second_attempt.random_seed == 9103

    assert (
        first_attempt.force_retry_gray_zone
        is True
    )

    assert (
        second_attempt.force_retry_gray_zone
        is False
    )

    assert (
        second_attempt.channel.loss_rate
        < first_attempt.channel.loss_rate
    )

    assert (
        second_attempt.channel
        .depolarizing_probability
        < first_attempt.channel
        .depolarizing_probability
    )

    assert (
        first_attempt.channel.loss_rate
        < config.quantum.maximum_loss_rate
    )

    demonstration_plan = (
        get_retry_demonstration_plan(
            app_config=config
        )
    )

    assert len(demonstration_plan) == 2

    assert (
        demonstration_plan[0]
        ["expected_attempt_decision"]
        == "retry"
    )

    assert (
        demonstration_plan[1]
        ["expected_final_decision"]
        == "accepted_after_retry"
    )

    valid_transition = validate_retry_transition(
        {
            "decision": "retry",
            "deterministic_pass": True,
        },
        {
            "decision": "accepted_after_retry",
            "deterministic_pass": True,
            "retry_used": True,
        },
    )

    assert valid_transition is True

    invalid_transition = validate_retry_transition(
        {
            "decision": "rejected_replay",
            "deterministic_pass": False,
        },
        {
            "decision": "accepted",
            "deterministic_pass": True,
            "retry_used": True,
        },
    )

    assert invalid_transition is False

    print(
        "FT-QuPAP accept-after-retry "
        "scenario self-test passed."
    )


__all__ = [
    "SCENARIO_NAME",
    "SCENARIO_DISPLAY_NAME",
    "SCENARIO_DESCRIPTION",
    "DEFAULT_RANDOM_SEED",
    "FIRST_ATTEMPT_NUMBER",
    "SUCCESSFUL_ATTEMPT_NUMBER",
    "FIRST_ATTEMPT_TARGET_GP_PROBABILITY",
    "SECOND_ATTEMPT_TARGET_GP_PROBABILITY",
    "ACCEPT_AFTER_RETRY_CONFIG",
    "build_scenario",
    "create_scenario",
    "get_scenario_summary",
    "get_attempt_configuration",
    "get_expected_attempt_outcome",
    "get_retry_demonstration_plan",
    "validate_retry_transition",
]


if __name__ == "__main__":
    run_self_test()